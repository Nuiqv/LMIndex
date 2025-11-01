#pragma once
#include <chrono>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>

#if defined(_MSC_VER)
  #define HPT_FORCE_INLINE __forceinline
  #define HPT_LIKELY(x)   (x)
  #define HPT_UNLIKELY(x) (x)
#elif defined(__GNUC__) || defined(__clang__)
  #define HPT_FORCE_INLINE inline __attribute__((always_inline))
  #define HPT_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define HPT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define HPT_FORCE_INLINE inline
  #define HPT_LIKELY(x)   (x)
  #define HPT_UNLIKELY(x) (x)
#endif

namespace hpt {

using clock_t = std::chrono::steady_clock; // equal to std::chrono::high_resolution_clock
using ns      = std::chrono::nanoseconds;

class Timer {
public:
    HPT_FORCE_INLINE Timer() noexcept { reset(); }

    HPT_FORCE_INLINE void start() noexcept {
        last_ = clock_t::now();
    }

    HPT_FORCE_INLINE void stop() noexcept {
        const auto now = clock_t::now();
        const auto d   = std::chrono::duration_cast<ns>(now - last_);
        elapsed_ += d;
        last_ = now;
    }

    // record a lap：base on [last_, now]，and set last_ to now time
    HPT_FORCE_INLINE ns lap() noexcept {
        const auto now = clock_t::now();
        const auto d   = std::chrono::duration_cast<ns>(now - last_);
        elapsed_ += d;
        laps_.push_back(d);
        // last_ = clock_t::now();
        return d;
    }

    //cut now time
    HPT_FORCE_INLINE ns cut() noexcept {
        const auto now = clock_t::now();
        return std::chrono::duration_cast<ns>(now - last_);
    }

    // no change last_（reset to now）
    HPT_FORCE_INLINE void reset() noexcept {
        elapsed_ = ns::zero();
        // last_ = clock_t::now();
        laps_.clear();
    }

    void laps_reserve(const int &size) {
        laps_.reserve(size);
    }

    HPT_FORCE_INLINE ns elapsed() const noexcept { return elapsed_; }
    HPT_FORCE_INLINE int64_t elapsed_ns() const noexcept { return elapsed_.count(); }
    HPT_FORCE_INLINE double elapsed_us() const noexcept { return elapsed_.count() / 1000.0; }
    HPT_FORCE_INLINE double elapsed_ms() const noexcept { return elapsed_.count() / 1e6; }
    HPT_FORCE_INLINE double elapsed_s () const noexcept { return elapsed_.count() / 1e9; }

    HPT_FORCE_INLINE std::vector<ns>& laps() noexcept { return laps_; }

private:
    clock_t::time_point last_{};
    ns                  elapsed_{};
    std::vector<ns>     laps_{};
};

// scope timer
class ScopeTimer {
public:
    HPT_FORCE_INLINE explicit ScopeTimer(std::string label,
                                         std::ostream& os = std::cerr,
                                         int precision = 3) noexcept
        : label_(std::move(label)), os_(os), precision_(precision) {
        last_ = clock_t::now();
    }

    HPT_FORCE_INLINE ~ScopeTimer() {
        const auto now = clock_t::now();
        const auto d   = std::chrono::duration_cast<ns>(now - last_);
        // const double ms = d.count() / 1e6;
        // os_ << "[TIMER] " << label_ << " : "
        //     << std::fixed << std::setprecision(precision_) << ms << " ms\n";
        os_ << "[TIMER] " << label_ << " : " << d.count() << " ns\n";
    }
private:
    std::string         label_;
    std::ostream&       os_;
    int                 precision_;
    clock_t::time_point last_{};
};
#define HPTIMER_SCOPE(label) ::hpt::ScopeTimer hpt_scope_timer__(label)


struct Stats {
    size_t n = 0;
    double min = 0;     
    double max = 0;     
    double mean = 0;    
    double median = 0;  
    double p95 = 0;     
    double stdev = 0;   
};

// summarize base on ns
inline Stats summarize_ns(const std::vector<ns>& samples) {
    Stats s{};
    if (samples.empty()) return s;

    s.n = samples.size();
    std::vector<double> v; 
    v.reserve(samples.size());
    for (auto d : samples) v.push_back(static_cast<double>(d.count()));

    std::sort(v.begin(), v.end());
    s.min   = v.front();
    s.max   = v.back();
    s.mean  = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    s.median = (v.size() & 1) ? v[v.size()/2]
                              : 0.5 * (v[v.size()/2 - 1] + v[v.size()/2]);

    size_t idx95 = static_cast<size_t>(std::ceil(0.95 * v.size()) - 1);
    if (idx95 >= v.size()) idx95 = v.size() - 1;
    s.p95 = v[idx95];

    double sq = 0.0;
    for (double x : v) {
        const double d = x - s.mean;
        sq += d * d;
    }
    s.stdev = std::sqrt(sq / v.size());
    return s;
}

enum class TimeUnit { NS, US, MS, S };

// print Stats, out NS in default
inline void print_stats(const Stats& s,
                        TimeUnit unit = TimeUnit::NS,
                        int precision = 5,
                        std::ostream& os = std::cerr)
{
    std::ios::fmtflags old_flags = os.flags();
    std::streamsize old_prec = os.precision();

    os.setf(std::ios::fixed);
    os << std::setprecision(precision);

    auto convert = [&](double ns_val) -> double {
        switch (unit) {
            case TimeUnit::NS: return ns_val;
            case TimeUnit::US: return ns_val / 1e3;
            case TimeUnit::MS: return ns_val / 1e6;
            case TimeUnit::S:  return ns_val / 1e9;
        }
        return ns_val;
    };

    const char* suffix = "";
    switch (unit) {
        case TimeUnit::NS: suffix = " ns"; break;
        case TimeUnit::US: suffix = " us"; break;
        case TimeUnit::MS: suffix = " ms"; break;
        case TimeUnit::S:  suffix = " s";  break;
    }

    if (s.n == 0) {
        os << "n=0 (no samples)\n";
    } else {
        os << "n=" << s.n
           << ", min="    << convert(s.min) 
           << suffix
           << ", mean="   << convert(s.mean) 
           << suffix
           << ", median=" << convert(s.median) 
           << suffix
           << ", p95="    << convert(s.p95) 
           << suffix
           << ", max="    << convert(s.max) 
           << suffix
           << ", stdev="  << convert(s.stdev) 
           << suffix
           << '\n';
    }

    os.precision(old_prec);
    os.flags(old_flags);
}

/// Save Stats to file in CSV format.
/// Each row: label,n,min,mean,median,p95,max,stdev,unit
/// - file_path: target CSV file
/// - label: a string to identify the experiment (optional)
/// - unit: output unit (default ms)
/// - append: if true, append to file; else overwrite
inline void save_stats(const Stats& s,
                       const std::string& file_path,
                       const std::string& label = "",
                       const std::string& dataset = "",
                       TimeUnit unit = TimeUnit::MS,
                       bool append = true)
{
    // Convert function: ns -> desired unit
    auto convert = [&](double ns_val) -> double {
        switch (unit) {
            case TimeUnit::NS: return ns_val;
            case TimeUnit::US: return ns_val / 1e3;
            case TimeUnit::MS: return ns_val / 1e6;
            case TimeUnit::S:  return ns_val / 1e9;
        }
        return ns_val;
    };

    const char* suffix = "";
    switch (unit) {
        case TimeUnit::NS: suffix = "ns"; break;
        case TimeUnit::US: suffix = "us"; break;
        case TimeUnit::MS: suffix = "ms"; break;
        case TimeUnit::S:  suffix = "s";  break;
    }

    std::ios::openmode mode = std::ios::out;
    if (append) mode |= std::ios::app;
    std::ofstream ofs(file_path, mode);

    if (!ofs) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    // If new file (not append or file just created), write header
    if (!append || ofs.tellp() == 0) {
        ofs << "label,dataset,n,min,mean,median,p95,max,stdev,unit\n";
    }

    ofs << label << ","
        << dataset << ","
        << s.n << ","
        << convert(s.min)    << ","
        << convert(s.mean)   << ","
        << convert(s.median) << ","
        << convert(s.p95)    << ","
        << convert(s.max)    << ","
        << convert(s.stdev)  << ","
        << suffix << "\n";
}

/// Build a filesystem-friendly token from an arbitrary label.
/// - Replace spaces with '_' and strip characters that are problematic in filenames.
inline std::string sanitize_label_for_filename(const std::string& label) {
    std::string out; out.reserve(label.size());
    for (char c : label) {
        if (c == ' ') { out.push_back('_'); continue; }
        // Allow alnum, '_', '-', '.', '@', '+'
        if (std::isalnum(static_cast<unsigned char>(c)) || c=='_' || c=='-' || c=='.' || c=='@' || c=='+') {
            out.push_back(c);
        }
        // else: drop the char
    }
    if (out.empty()) out = "run";
    return out;
}

/// Return unit suffix string used in filenames
inline const char* unit_suffix(TimeUnit unit) {
    switch (unit) {
        case TimeUnit::NS: return "ns";
        case TimeUnit::US: return "us";
        case TimeUnit::MS: return "ms";
        case TimeUnit::S:  return "s";
    }
    return "ns";
}

/// Convert ns -> desired unit as double
inline double convert_ns(double ns_val, TimeUnit unit) {
    switch (unit) {
        case TimeUnit::NS: return ns_val;
        case TimeUnit::US: return ns_val / 1e3;
        case TimeUnit::MS: return ns_val / 1e6;
        case TimeUnit::S:  return ns_val / 1e9;
    }
    return ns_val;
}

/// Save raw samples (vector<ns>) into a CSV that contains ONLY values (one per line).
/// The filename encodes metadata: label, sample size, and unit.
/// - base_stem: path without extension (e.g., "out/samples").
/// - label: arbitrary tag for this run; sanitized for filename.
/// - unit: output unit for values; appears in filename.
/// - append: if true, append values to the same file; no header is written.
/// Returns the final path used for writing.
inline std::string save_samples_raw(const std::vector<ns>& samples,
                                    const std::string& base_stem,
                                    const std::string& label = "",
                                    TimeUnit unit = TimeUnit::MS,
                                    bool append = true)
{
    const std::string lab  = sanitize_label_for_filename(label);
    const std::string unit_str = unit_suffix(unit);
    const std::string final_path =
        base_stem + "__label-" + lab +
        "__n-" + std::to_string(samples.size()) +
        "__unit-" + unit_str + ".csv";

    std::ios::openmode mode = std::ios::out;
    if (append) mode |= std::ios::app;

    std::ofstream ofs(final_path, mode);
    if (!ofs) {
        throw std::runtime_error("Cannot open file: " + final_path);
    }

    // Write only values, one per line
    for (const auto& d : samples) {
        const double v = convert_ns(static_cast<double>(d.count()), unit);
        ofs << v << '\n';
    }

    return final_path;
}


// ---------- Request model ----------

enum class StatKind {
    QUANTILE,  // param in [0,1], e.g., 0.5 for p50
    MEAN,      // param ignored
    STDEV,     // population stdev; param ignored
    MIN,       // param ignored
    MAX,       // param ignored
    COUNT      // number of samples; param ignored
};

struct StatRequest {
    StatKind kind;
    double   param;     // used by QUANTILE only; ignored otherwise
    // optional: you can add a name/label field for identification if needed
};

/// Convenience builder for quantile requests: q in [0,1]
inline StatRequest Q(double q) { return StatRequest{StatKind::QUANTILE, q}; }

/// Convenience builders for common metrics
inline StatRequest Mean()  { return StatRequest{StatKind::MEAN,   0.0}; }
inline StatRequest Stdev() { return StatRequest{StatKind::STDEV,  0.0}; }
inline StatRequest Min()   { return StatRequest{StatKind::MIN,    0.0}; }
inline StatRequest Max()   { return StatRequest{StatKind::MAX,    0.0}; }
inline StatRequest Count() { return StatRequest{StatKind::COUNT,  0.0}; }

namespace detail {

/// Quantile (type-7) on a sorted vector<long long> (unit: ns).
inline double quantile_type7_sorted_ll(const std::vector<long long>& sorted, double p) {
    if (sorted.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (p <= 0.0) return static_cast<double>(sorted.front());
    if (p >= 1.0) return static_cast<double>(sorted.back());
    const double n  = static_cast<double>(sorted.size());
    const double h  = 1.0 + (n - 1.0) * p;            // 1-based
    const size_t lo = static_cast<size_t>(std::floor(h));
    const size_t hi = static_cast<size_t>(std::ceil(h));
    const double frac = h - static_cast<double>(lo);
    const long long v_lo = sorted[lo - 1];
    const long long v_hi = sorted[hi - 1];
    return static_cast<double>(v_lo) + frac * (static_cast<double>(v_hi - v_lo));
}

/// Sum as 128-bit integer if available; fallback to long double with Kahan compensation.
inline long double precise_sum_ns_ll(const std::vector<ns>& samples) {
#if defined(__SIZEOF_INT128__)
    __int128 acc = 0;
    for (const auto& d : samples) acc += static_cast<__int128>(d.count());
    return static_cast<long double>(acc);
#else
    // Kahan compensated summation in long double
    long double sum = 0.0L, c = 0.0L;
    for (const auto& d : samples) {
        const long double x = static_cast<long double>(d.count());
        const long double y = x - c;
        const long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
#endif
}

/// Compute sum of squared deviations in long double: sum((x - mu)^2).
inline long double sumsq_from_mean_ns_ll(const std::vector<ns>& samples, long double mean_ns) {
    long double acc = 0.0L;
    for (const auto& d : samples) {
        const long double x = static_cast<long double>(d.count());
        const long double dx = x - mean_ns;
        acc += dx * dx;
    }
    return acc;
}

} // namespace detail

/// Precise, on-demand statistics:
/// - Only computes what is requested.
/// - mean: high-precision sum / n (int128 if available; else long double + Kahan).
/// - stdev: two-pass (mean first, then sumsq).
/// - quantiles: sort once (vector<long long>) and evaluate type-7.
/// - min/max: single pass if requested.
/// Returns values aligned with `requests` order (unit: ns).
inline std::vector<double> compute_stats_precise(
    const std::vector<ns>& samples,
    const std::vector<StatRequest>& requests)
{
    const size_t N = samples.size();
    // Fast return for empty input
    if (N == 0) {
        std::vector<double> out; out.reserve(requests.size());
        for (const auto& r : requests) {
            if (r.kind == StatKind::COUNT) out.push_back(0.0);
            else out.push_back(std::numeric_limits<double>::quiet_NaN());
        }
        return out;
    }

    // Plan: decide which quantities are needed
    bool need_mean = false;
    bool need_stdev = false;
    bool need_min = false;
    bool need_max = false;
    bool need_quant = false;

    for (const auto& r : requests) {
        switch (r.kind) {
            case StatKind::MEAN:     need_mean = true; break;
            case StatKind::STDEV:    need_stdev = true; need_mean = true; break; // stdev requires mean
            case StatKind::MIN:      need_min = true; break;
            case StatKind::MAX:      need_max = true; break;
            case StatKind::QUANTILE: need_quant = true; break;
            case StatKind::COUNT:    /* trivial */ break;
            default: break;
        }
    }

    // Compute min/max in a single pass only if requested
    long long vmin = std::numeric_limits<long long>::max();
    long long vmax = std::numeric_limits<long long>::min();
    if (need_min || need_max) {
        for (const auto& d : samples) {
            const long long v = d.count();
            if (need_min && v < vmin) vmin = v;
            if (need_max && v > vmax) vmax = v;
        }
    }

    // Compute mean (precise) only if requested
    long double mean_ld = 0.0L;
    if (need_mean) {
        const long double sum_ld = detail::precise_sum_ns_ll(samples);
        mean_ld = sum_ld / static_cast<long double>(N);
    }

    // Compute stdev only if requested (population stdev)
    double stdev = std::numeric_limits<double>::quiet_NaN();
    if (need_stdev) {
        if (N == 1) {
            stdev = 0.0; // population stdev with single sample is 0
        } else {
            const long double sumsq = detail::sumsq_from_mean_ns_ll(samples, mean_ld);
            stdev = std::sqrt(static_cast<long double>(sumsq / static_cast<long double>(N)));
        }
    }

    // Prepare quantiles only if requested: sort a copy of integer ns
    std::vector<long long> sorted_ll;
    if (need_quant) {
        sorted_ll.reserve(N);
        for (const auto& d : samples) sorted_ll.push_back(d.count());
        std::sort(sorted_ll.begin(), sorted_ll.end());
    }

    // Compose outputs aligned with requests
    std::vector<double> out; out.reserve(requests.size());
    for (const auto& r : requests) {
        switch (r.kind) {
            case StatKind::COUNT:
                out.push_back(static_cast<double>(N));
                break;
            case StatKind::MIN:
                out.push_back(static_cast<double>(vmin));
                break;
            case StatKind::MAX:
                out.push_back(static_cast<double>(vmax));
                break;
            case StatKind::MEAN:
                out.push_back(static_cast<double>(mean_ld)); // unit: ns
                break;
            case StatKind::STDEV:
                out.push_back(stdev); // unit: ns
                break;
            case StatKind::QUANTILE:
                out.push_back(detail::quantile_type7_sorted_ll(sorted_ll, r.param)); // unit: ns
                break;
            default:
                out.push_back(std::numeric_limits<double>::quiet_NaN());
                break;
        }
    }
    return out;
}

} // namespace hpt
