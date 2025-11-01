#ifndef MEM_PROFILER_HPP
#define MEM_PROFILER_HPP

#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

#ifdef __linux__
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __GLIBC__
#include <malloc.h> // mallinfo2 / mallinfo if available
#endif
#endif

namespace memprof {

// Configuration: choose which fields to sample.
struct SampleConfig {
    bool sample_ts_ns = true;        // monotonic timestamp (ns)
    bool sample_label = true;       // user-provided label
    bool sample_rss = true;          // VmRSS / RSS
    bool sample_vmsize = true;       // VmSize
    bool sample_pss = false;         // PSS via /proc/self/smaps (slower)
    bool sample_heap = true;         // malloc-heap uordblks via mallinfo2
    bool sample_vmhwm = true;        // VmHWM (peak RSS)
    bool sample_ru_maxrss = false;   // getrusage ru_maxrss (kB)
    bool sample_vmswap = false;      // VmSwap
    bool sample_private_clean = false; // Private_Clean (from smaps)
    bool sample_private_dirty = false; // Private_Dirty (from smaps)
    // Add more flags as needed
};

// Single sample record. Fields left zero if not collected.
struct MemSample {
    long long ts_ns = 0;
    std::string label;           // optional, used only if config.sample_label == true
    uint64_t rss_bytes = 0;
    uint64_t vmsize_bytes = 0;
    uint64_t pss_bytes = 0;
    uint64_t heap_bytes = 0;
    uint64_t vmhwm_bytes = 0;
    long long ru_maxrss_kb = 0;
    uint64_t vmswap_bytes = 0;
    uint64_t private_clean_bytes = 0;
    uint64_t private_dirty_bytes = 0;
};

// Helper: read small file into string. Return false if fail.
inline bool read_file_to_string(const std::string &path, std::string &out) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    out = ss.str();
    return true;
}

// Parse a "Key:   NNN kB" style line in status for kB values.
inline uint64_t parse_status_field_kb(const std::string &text, const std::string &key) {
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream L(line);
            std::string tag;
            uint64_t val = 0;
            std::string unit;
            L >> tag >> val >> unit;
            return val * 1024ULL;
        }
    }
    return 0;
}

// Fallback statm parser (pages).
inline bool sample_statm(uint64_t &rss_bytes, uint64_t &vmsize_bytes) {
    std::ifstream ifs("/proc/self/statm");
    if (!ifs) return false;
    long long size_pages = 0, resident_pages = 0;
    ifs >> size_pages >> resident_pages;
    long page = sysconf(_SC_PAGESIZE);
    if (size_pages > 0) vmsize_bytes = (uint64_t)size_pages * (uint64_t)page;
    if (resident_pages > 0) rss_bytes = (uint64_t)resident_pages * (uint64_t)page;
    return true;
}

// Parse /proc/self/status quickly for VmRSS, VmSize, VmHWM, VmSwap
inline bool sample_proc_status(uint64_t &rss_bytes, uint64_t &vmsize_bytes, uint64_t &vmhwm_bytes, uint64_t &vmswap_bytes) {
    std::string status;
    if (!read_file_to_string("/proc/self/status", status)) return false;
    rss_bytes = parse_status_field_kb(status, "VmRSS:");
    vmsize_bytes = parse_status_field_kb(status, "VmSize:");
    vmhwm_bytes = parse_status_field_kb(status, "VmHWM:");
    vmswap_bytes = parse_status_field_kb(status, "VmSwap:");
    return true;
}

// Parse /proc/self/smaps for Pss and optionally Private_Clean/Dirty. Note: expensive.
inline void parse_smaps_aggregate(uint64_t &out_pss_kb, uint64_t &out_priv_clean_kb, uint64_t &out_priv_dirty_kb) {
    out_pss_kb = 0;
    out_priv_clean_kb = 0;
    out_priv_dirty_kb = 0;
    std::ifstream ifs("/proc/self/smaps");
    if (!ifs) return;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("Pss:", 0) == 0) {
            std::istringstream L(line);
            std::string tag;
            uint64_t val;
            std::string unit;
            L >> tag >> val >> unit;
            out_pss_kb += val;
        } else if (line.rfind("Private_Clean:", 0) == 0) {
            std::istringstream L(line);
            std::string tag;
            uint64_t val;
            std::string unit;
            L >> tag >> val >> unit;
            out_priv_clean_kb += val;
        } else if (line.rfind("Private_Dirty:", 0) == 0) {
            std::istringstream L(line);
            std::string tag;
            uint64_t val;
            std::string unit;
            L >> tag >> val >> unit;
            out_priv_dirty_kb += val;
        }
    }
    // convert kB -> bytes
    out_pss_kb *= 1024ULL;
    out_priv_clean_kb *= 1024ULL;
    out_priv_dirty_kb *= 1024ULL;
}

// malloc-heap introspection (glibc mallinfo2 or mallinfo). Returns 0 if not available.
inline uint64_t get_heap_allocated_bytes() {
#ifdef __linux__
  #ifdef mallinfo2
    struct mallinfo2 mi = mallinfo2();
    return (uint64_t)mi.uordblks;
  #elif defined(mallinfo)
    struct mallinfo mi = mallinfo();
    return (uint64_t)mi.uordblks;
  #else
    return 0;
  #endif
#else
    return 0;
#endif
}

// getrusage ru_maxrss (units: kB on Linux)
inline long long get_ru_maxrss_kb() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return (long long)ru.ru_maxrss;
    }
    return 0;
}

// monotonic timestamp in nanoseconds using steady_clock
inline long long now_ns() {
    using namespace std::chrono;
    return (long long)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// check file existence
inline bool file_exists(const std::string &p) {
    std::ifstream f(p);
    return f.good();
}

// The main profiler class
class MemProfiler {
public:
    explicit MemProfiler(const SampleConfig &cfg = SampleConfig())
        : cfg_(cfg), running_(false) {}

    ~MemProfiler() {
        stop(); // ensure background thread stopped
    }

    // Take a synchronous single sample and append to internal buffer.
    MemSample sample(const std::string &label = "") {
        MemSample s;
        if (cfg_.sample_ts_ns) s.ts_ns = now_ns();
        if (cfg_.sample_label) s.label = label;

        // Try proc status first for rss/vmsize/vmhwm/vmswap
        uint64_t rss = 0, vms = 0, vmhwm = 0, vmswap = 0;
        if (cfg_.sample_rss || cfg_.sample_vmsize || cfg_.sample_vmhwm || cfg_.sample_vmswap) {
            bool ok = sample_proc_status(rss, vms, vmhwm, vmswap);
            if (!ok) {
                // fallback to statm for rss and vmsize
                sample_statm(rss, vms);
            }
            if (cfg_.sample_rss) s.rss_bytes = rss;
            if (cfg_.sample_vmsize) s.vmsize_bytes = vms;
            if (cfg_.sample_vmhwm) s.vmhwm_bytes = vmhwm;
            if (cfg_.sample_vmswap) s.vmswap_bytes = vmswap;
        }

        // PSS and private_* from smaps (expensive)
        if (cfg_.sample_pss || cfg_.sample_private_clean || cfg_.sample_private_dirty) {
            uint64_t pss = 0, priv_clean = 0, priv_dirty = 0;
            parse_smaps_aggregate(pss, priv_clean, priv_dirty);
            if (cfg_.sample_pss) s.pss_bytes = pss;
            if (cfg_.sample_private_clean) s.private_clean_bytes = priv_clean;
            if (cfg_.sample_private_dirty) s.private_dirty_bytes = priv_dirty;
        }

        // heap via mallinfo2
        if (cfg_.sample_heap) {
            s.heap_bytes = get_heap_allocated_bytes();
        }

        // ru_maxrss
        if (cfg_.sample_ru_maxrss) {
            s.ru_maxrss_kb = get_ru_maxrss_kb();
        }

        // If rss/vmsize not filled but still desired, ensure fallback
        if (cfg_.sample_rss && s.rss_bytes == 0) {
            uint64_t rss_fb=0, vms_fb=0;
            sample_statm(rss_fb, vms_fb);
            s.rss_bytes = rss_fb;
        }
        if (cfg_.sample_vmsize && s.vmsize_bytes == 0) {
            uint64_t rss_fb=0, vms_fb=0;
            sample_statm(rss_fb, vms_fb);
            s.vmsize_bytes = vms_fb;
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            samples_.push_back(s);
        }
        return s;
    }

    // Retrieve current recorded samples (thread-safe copy).
    std::vector<MemSample> get_samples() const {
        std::lock_guard<std::mutex> lk(mu_);
        return samples_;
    }

    // Clear in-memory samples.
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        samples_.clear();
    }

    // Start periodic sampling in background thread.
    // interval_ms: sampling interval in milliseconds (>=1)
    // label_prefix: optional prefix for generated labels (if cfg.sample_label enabled)
    void start_periodic(unsigned interval_ms, const std::string &label_prefix = "") {
        if (running_.load()) return; // already running
        if (interval_ms == 0) interval_ms = 1;
        running_.store(true);
        worker_thread_ = std::thread([this, interval_ms, label_prefix]() {
            unsigned idx = 0;
            while (running_.load()) {
                if (cfg_.sample_label) {
                    std::ostringstream ss;
                    ss << label_prefix << idx++;
                    this->sample(ss.str());
                } else {
                    this->sample();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        });
    }

    // Stop periodic sampling (join thread).
    void stop() {
        if (!running_.load()) return;
        running_.store(false);
        if (worker_thread_.joinable()) worker_thread_.join();
    }

    // Save recorded samples to CSV. The header is generated based on cfg_ and the chosen column order.
    // If append==true and file exists -> append rows (no header). If file missing -> write header first.
    // If append==false -> overwrite file and write header + all rows.
    bool save_csv(const std::string &path, bool append = true, std::string dataset = "") const {
        std::lock_guard<std::mutex> lk(mu_);
        bool exists = file_exists(path);
        std::ofstream ofs;
        if (append) ofs.open(path, std::ios::out | std::ios::app);
        else ofs.open(path, std::ios::out | std::ios::trunc);
        if (!ofs) return false;

        // decide columns and order
        std::vector<std::string> columns;
        if (cfg_.sample_ts_ns) columns.push_back("ts_ns");
        if (cfg_.sample_label) columns.push_back("label");
        if (dataset != "") columns.push_back("dataset");
        if (cfg_.sample_rss) columns.push_back("rss_bytes");
        if (cfg_.sample_vmsize) columns.push_back("vmsize_bytes");
        if (cfg_.sample_pss) columns.push_back("pss_bytes");
        if (cfg_.sample_heap) columns.push_back("heap_bytes");
        if (cfg_.sample_vmhwm) columns.push_back("vmhwm_bytes");
        if (cfg_.sample_ru_maxrss) columns.push_back("ru_maxrss_kb");
        if (cfg_.sample_vmswap) columns.push_back("vmswap_bytes");
        if (cfg_.sample_private_clean) columns.push_back("private_clean_bytes");
        if (cfg_.sample_private_dirty) columns.push_back("private_dirty_bytes");

        // write header if needed
        if (!exists || !append) {
            // write header
            for (size_t i = 0; i < columns.size(); ++i) {
                ofs << columns[i];
                if (i + 1 < columns.size()) ofs << ",";
            }
            ofs << "\n";
        }

        // write rows
        for (const auto &s : samples_) {
            bool first = true;
            auto write_val = [&](const std::string &val) {
                if (!first) ofs << ",";
                ofs << val;
                first = false;
            };
            for (const auto &col : columns) {
                if (col == "ts_ns") write_val(std::to_string(s.ts_ns));
                else if (col == "label") {
                    // escape quotes by replacing " with '
                    std::string lab = s.label;
                    for (auto &c : lab) if (c == '"') c = '\'';
                    write_val(std::string("\"") + lab + "\"");
                }
                else if (col == "dataset") write_val(dataset);
                else if (col == "rss_bytes") write_val(std::to_string(s.rss_bytes));
                else if (col == "vmsize_bytes") write_val(std::to_string(s.vmsize_bytes));
                else if (col == "pss_bytes") write_val(std::to_string(s.pss_bytes));
                else if (col == "heap_bytes") write_val(std::to_string(s.heap_bytes));
                else if (col == "vmhwm_bytes") write_val(std::to_string(s.vmhwm_bytes));
                else if (col == "ru_maxrss_kb") write_val(std::to_string(s.ru_maxrss_kb));
                else if (col == "vmswap_bytes") write_val(std::to_string(s.vmswap_bytes));
                else if (col == "private_clean_bytes") write_val(std::to_string(s.private_clean_bytes));
                else if (col == "private_dirty_bytes") write_val(std::to_string(s.private_dirty_bytes));
                else write_val(""); // unknown column, blank
            }
            ofs << "\n";
        }

        ofs.flush();
        return true;
    }

    // Convenience: sample N times with delay_ms between samples (blocking)
    void sample_n_times(size_t n, unsigned delay_ms = 0, const std::string &label_prefix = "") {
        for (size_t i = 0; i < n; ++i) {
            if (cfg_.sample_label) {
                std::ostringstream ss;
                ss << label_prefix << i;
                sample(ss.str());
            } else {
                sample();
            }
            if (i + 1 < n && delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    // Expose config for inspection or runtime modification.
    SampleConfig& config() { return cfg_; }
    const SampleConfig& config() const { return cfg_; }

private:
    SampleConfig cfg_;
    mutable std::mutex mu_;
    std::vector<MemSample> samples_;
    std::thread worker_thread_;
    std::atomic<bool> running_;
};

} // namespace memprof

#endif // MEM_PROFILER_HPP
