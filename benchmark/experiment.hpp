#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <string>
#include <tuple>
#include <bit>
#include "../include/LM_Index.hpp"
#include "../include/LM_Index_Gaps.hpp"
#include "lib/config.hpp"
#include "lib/pgm_index_dynamic.hpp"
#include "lib/Swix.hpp"
#include "lib/alex.h"
#include "lib/lipp.h"
#include "lib/btree_map.h"
//test
#include "./timer/hptimer.hpp"
#include "./memory/mem_profile.hpp"
//

#define TEST_SEED 233
const int SEED = 233;

#define PURE_QUERY_SIZE 10000000
#define MEMORY_TEST 0
#define DELETE_TEST 0
#define QUERY_ONLY_TEST 1
#define RANGE_QUERY_TEST 0
#define STABILITY_TEST 0
#define INSERT_STABILITY_TEST 0
#define INSERT_ONLY_TEST 0
#define UPDATE_TEST 0

const int range_test_count = 10000;
#define STABILITY_RANGE_SIZE 100

using namespace std;

class WorkLoad {
protected:
    size_t base_size;           //base init data for query
    size_t dynamic_data_size;   //=base_size at beginning, ++ after insert, not -- after delete
    size_t op_index = 0;
    vector<pair<int, uint64_t>> operations;
    double query_ratio = 1.0;
    double insert_ratio = 0;
public:
    enum OP_TYPE {QUERY, INSERT, DELETE};

    inline size_t get_total_insert_size() const {
        return size_t(operations.size() * insert_ratio);
    }

    inline size_t get_operations_size() const {
        return operations.size();
    }

    inline void set_init_size(const size_t &init_size) {
        dynamic_data_size = base_size = init_size;
    }

    inline void reset() {
        dynamic_data_size = base_size;
        op_index = 0;
    }

    inline int get_operation_type() const {
        return operations[op_index].first;
    }

    inline size_t get_operation_code() const {
        return operations[op_index].second % dynamic_data_size;
    }

    inline bool have_operation() const {
        return op_index < operations.size();
    }

    inline void next_operation() {
        ++op_index;
    }

    inline size_t get_op_index() {
        return op_index;
    }

    template<typename K>
    K & get_insert_pair(vector<K> &origin, vector<K> &insert) {
        K & i_key = insert[operations[op_index].second % insert.size()];
        origin.push_back(i_key);
        ++dynamic_data_size;
        return i_key;
    }

    void gen_operations(const size_t &data_init_size, const size_t &op_size, const double &q_ratio, const double &i_ratio) {
        if(q_ratio + i_ratio > 1.0) {
            cout << "Gen operations fail: ratio > 1" << endl;
            return;
        }
        operations.resize(op_size);

        base_size = data_init_size;
        reset();
        const size_t insert_size = size_t(round(op_size * i_ratio));
        // const size_t insert_size = round(op_size * i_ratio);
        const size_t max_size = data_init_size + insert_size;

        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, max_size - 1);

        auto op_it = operations.begin();
        auto limit = operations.begin() + insert_size;
        uint64_t i = 0;
        while(op_it != limit) {
            *(op_it++) = make_pair(INSERT, i++);
        }
        size_t query_size = size_t(round(op_size * q_ratio));
        limit += query_size;
        while(op_it != limit) {
            *(op_it++) = make_pair(QUERY, distr(gen));
        }
        limit = operations.end();
        while(op_it != limit) {
            *(op_it++) = make_pair(DELETE, distr(gen));
        }
        shuffle(operations.begin(), operations.end(), gen);
    }

    void gen_operations(size_t data_size, size_t op_size) {
        gen_operations(data_size, op_size, query_ratio, insert_ratio);
    }

    inline vector<pair<int, uint64_t>> &get_operations() {
        return operations;
    }

    template<typename T>
    static inline void split_insert_part(vector<T> &origin, const size_t& begin, const size_t& end, vector<T> &insert_part) {
        insert_part.resize(end - begin);
        auto begin_it = origin.begin() + begin, end_it = origin.begin() + end;
        copy(begin_it, end_it, insert_part.begin());
        origin.erase(begin_it, end_it);
    }

    template<typename T>
    static inline void random_split(vector<T> &origin, vector<T> &query_part, vector<T> &insert_part, const size_t& insert_size) {
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, origin.size() - 1);
        insert_part.resize(insert_size);
        query_part.resize(origin.size() - insert_size);
        set<size_t> unique;
        while(unique.size() < insert_size) {
            unique.insert(distr(gen));
        }
        auto u_it = unique.begin();
        auto q_it = query_part.begin(), i_it = insert_part.begin();
        size_t index = 0;
        while(u_it != unique.end()) {
            while(index != *u_it) {
                *(q_it++) = origin[index++];
            }
            *(i_it++) = origin[index++];
            ++u_it;
        }
        while(index != origin.size())
            *(q_it++) = origin[index++];
    }

    template<typename T>
    static inline void random_split(vector<pair<T, T>> &origin, vector<pair<T, T>> &query_part, vector<pair<T, T>> &insert_part) {
        query_part.resize(origin.size() - insert_part.size());
        auto q_it = query_part.begin(), i_it = insert_part.begin();
        auto index = origin.begin();
        while(i_it != insert_part.end()) {
            while(index->first != i_it->first) {
                //debug
                auto q_index = q_it - query_part.begin();
                //
                *(q_it++) = *(index++);
            }
            ++i_it;
            ++index;
        }
        while(index != origin.end())
            *(q_it++) = *(index++);
    }

    WorkLoad() = default;
    WorkLoad(double q_ratio, double i_ratio) : query_ratio(q_ratio), insert_ratio(i_ratio) {}

    void setRatio(double q_ratio, double i_ratio) {
        query_ratio = q_ratio;
        insert_ratio = i_ratio;
    }

    inline double get_query_ratio() const {
        return query_ratio;
    }

    inline double get_insert_ratio() const {
        return insert_ratio;
    }

};

WorkLoad public_workload;
string public_dataset = "";

#if MEMORY_TEST
memprof::SampleConfig global_cfg{true, true, true, true, true, true, true, true, true};
#endif

template<typename K>
void lm_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload, 
    size_t layer_size = MAX_LAYER_SIZE, uint32_t buffer_size = LM_BUFFER_SIZE, pair<K, K> *all_data = nullptr) {
    srand(SEED);
    
    string label = "lm";
    cout << label + " index test:" << endl;

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    LMD::LM_Index<K, K, ERROR_THRESHOLD> lmd_index(data, layer_size, buffer_size);
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;

    t.reset();

    #if DELETE_TEST
    cout << "\nstart delete test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        lmd_index.erase(i_it->first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    cout << "delete clock: " << stats.mean << endl;
    t.reset();
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        volatile auto x = lmd_index.search(data[*(index_it++)].first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            vector<pair<K, K>> scan_result;
            t.start();
            lmd_index.range_search(data[begin_pos].first, data[end_pos].first, scan_result);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                lmd_index.insert(i_it->first, i_it->second);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                volatile auto x = lmd_index.search(data[*(index_it) % base_size].first);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                lmd_index.range_search(data[begin_pos].first, data[end_pos].first, scan_result);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        lmd_index.insert(i_it->first, i_it->second);
        t.lap();
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    volatile K x;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            x = lmd_index.search(_pair.first).second;
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            lmd_index.insert(_pair.first, _pair.second);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();

    #endif
}

template<typename K>
void lm_gaps_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload, 
    float rate = EXPANSION_RATE, size_t layer_size = MAX_LAYER_SIZE, uint32_t buffer_size = LM_BUFFER_SIZE, pair<K, K> *all_data = nullptr) {
    srand(SEED);
    
    string label = "lmg";
    cout << label + " index test:" << endl;

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    LMD::LM_Index_Gaps<K, K, ERROR_THRESHOLD> lmd_index(data, rate, layer_size, buffer_size);
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;
    t.reset();

    #if DELETE_TEST
    cout << "\nstart delete test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        lmd_index.erase(i_it->first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    cout << "delete clock: " << stats.mean << endl;
    t.reset();
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        volatile auto x = lmd_index.search(data[*(index_it++)].first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            vector<pair<K, K>> scan_result;
            t.start();
            lmd_index.range_search(data[begin_pos].first, data[end_pos].first, scan_result);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                lmd_index.insert(i_it->first, i_it->second);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                volatile auto x = lmd_index.search(data[*(index_it) % base_size].first);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                lmd_index.range_search(data[begin_pos].first, data[end_pos].first, scan_result);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        lmd_index.insert(i_it->first, i_it->second);
        t.lap();
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    volatile K x;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            x = lmd_index.search(_pair.first).second;
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            lmd_index.insert(_pair.first, _pair.second);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();
    #endif
}

//pgm index
template<typename K>
void pgm_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload) {
    srand(SEED);
    
    string label = "pgm";
    cout << label + " index test:" << endl;

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    pgm::DynamicPGMIndex<K, K, pgm::PGMIndex<K, ERROR_THRESHOLD>> pgm(data.begin(), data.end());
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;
    t.reset();

    #if DELETE_TEST
    cout << "\nstart delete test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        pgm.erase(i_it->first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    cout << "delete clock: " << stats.mean << endl;
    t.reset();
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        volatile auto x = pgm.find(data[*(index_it++)].first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    
    
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            vector<pair<K, K>> scan_result;
            t.start();
            scan_result = pgm.range(data[begin_pos].first, data[end_pos].first);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                pgm.insert_or_assign(i_it->first, i_it->second);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                volatile auto x = *(pgm.find(data[*(index_it) % base_size].first));
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                scan_result = pgm.range(data[begin_pos].first, data[end_pos].first);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        pgm.insert_or_assign(i_it->first, i_it->second);
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    volatile K x;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            x = (*pgm.find(_pair.first)).second;
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            pgm.insert_or_assign(_pair.first, _pair.second);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();

    #endif
}

//swix index
template<typename K>
void swix_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload) {
    srand(SEED);
    
    #ifdef TUNE
    string label = "swix";
    #else
    string label = "swix(wo turn)";
    #endif
    cout << label + " index test:" << endl;

    for(auto &_pair: data) {
        _pair.second = 1;
    }

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    K resultCount = 0;

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    swix::SWmeta<K, K> swix(data);
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;
    t.reset();

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        swix.lookup(data[*(index_it++)], resultCount);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    hpt::save_stats(stats, "../benchmark/pure_query.csv", label, public_dataset, hpt::TimeUnit::NS);
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            auto search_tuple = make_tuple(data[begin_pos].first, K(0), data[end_pos].first);
            vector<pair<K, K>> scan_result;
            t.start();
            swix.range_search(search_tuple, scan_result);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                swix.insert(*i_it);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                swix.lookup(data[*(index_it) % base_size], resultCount);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                auto search_tuple = make_tuple(data[begin_pos].first, K(0), data[end_pos].first);
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                swix.range_search(search_tuple, scan_result);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        swix.insert(*i_it);
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            swix.lookup(_pair, resultCount);
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            swix.insert(_pair);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();

    #endif
}

//alex index
template<typename K>
void alex_range_search(alex::Alex<K, K> &alex_index, const K &key1, const K &key2, vector<pair<K, K>> &result) {
    auto begin = alex_index.lower_bound(key1);
    auto end = alex_index.lower_bound(key2);
    if(end != alex_index.end()) ++end;
    while(begin != end)
    {
        if(begin.key() <= key2)
        {
            result.push_back(make_pair(begin.key(), begin.payload()));
        }
        begin++;
    }
}

template<typename K>
void alex_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload) {
    srand(SEED);
    
    string label = "alex";
    cout << label + " index test:" << endl;

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    alex::Alex<K, K> alex_index;
    alex_index.set_max_node_size(MAX_LAYER_SIZE);
    alex_index.bulk_load(data.data(), data.size());
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;
    t.reset();

    #if DELETE_TEST
    cout << "\nstart delete test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        alex_index.erase(i_it->first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    cout << "delete clock: " << stats.mean << endl;
    t.reset();
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        volatile auto x = alex_index.get_payload(data[*(index_it++)].first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    
    
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            vector<pair<K, K>> scan_result;
            t.start();
            alex_range_search(alex_index, data[begin_pos].first, data[end_pos].first, scan_result);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                alex_index.insert(i_it->first, i_it->second);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                volatile auto x = alex_index.get_payload(data[*(index_it) % base_size].first);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                alex_range_search(alex_index, data[begin_pos].first, data[end_pos].first, scan_result);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        alex_index.insert(i_it->first, i_it->second);
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    volatile K x;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            x = *(alex_index.get_payload(_pair.first));
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            alex_index.insert(_pair.first, _pair.second);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();

    #endif
}

//btree index
template<typename Type>
class btree_traits_fanout{
public:
    static const bool selfverify = false;
    static const bool debug = false;
    static const int leafslots = 256;
    static const int innerslots = 256;
    static const size_t binsearch_threshold = 256;
};

template<typename K, class BTREE>
void btree_range_search(BTREE &btree, const K &key1, const K &key2, vector<pair<K, K>> &result) {
    auto it = btree.lower_bound(key1);
    while (it != btree.end() && it->first <= key2)
    { 
        result.push_back(*it);
        it++;
    }
}

template<typename K>
void btree_test(vector<pair<K, K>> &data, vector<pair<K, K>> &insert_data, WorkLoad &load = public_workload) {
    srand(SEED);
    
    string label = "btree";
    cout << label + " index test:" << endl;

    hpt::Timer t;
    t.laps_reserve(PURE_QUERY_SIZE);
    hpt::Stats stats;

    #if MEMORY_TEST
    memprof::MemProfiler prof(global_cfg);
    prof.sample(label + "_bfbulk_" + public_dataset);
    #endif

    //bulk load
    cout << "\nstart bulk loading..." << endl;
    t.start();
    stx::btree_map<K, K, less<K>> btree(data.begin(), data.end());
    t.lap();
    cout << "bulk clock: " << t.elapsed_ns() << endl;
    t.reset();

    #if DELETE_TEST
    cout << "\nstart delete test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        t.start();
        btree.erase(i_it->first);
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    cout << "delete clock: " << stats.mean << endl;
    t.reset();
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afbulk_" + public_dataset);
    #endif

    #if QUERY_ONLY_TEST
    
    cout << "\nstart pure query..." << endl;
    //pure_query
    vector<int> query_index;
    query_index.reserve(PURE_QUERY_SIZE);
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        query_index.push_back(rand() % data.size());
    }
    auto index_it = query_index.begin();
    for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
        t.start();
        volatile auto x = *(btree.find(data[*(index_it++)].first));
        t.lap();
    }
    stats = hpt::summarize_ns(t.laps());
    
    
    t.reset();

    cout << "avg query clock: " << stats.mean << endl;
    #endif

    #if RANGE_QUERY_TEST
    cout << "\nstart range query..." << endl;
    std::mt19937 gen(TEST_SEED);
    std::uniform_int_distribution<uint64_t> distr(0, data.size() - 1);
    
    for(int i = 0, range_size = 1; i < 6; ++i) {
        int max_size = data.size() - range_size + 1;
        for(int j = 0; j < range_test_count; ++j) {
            int begin_pos = distr(gen) % max_size;
            int end_pos = begin_pos + range_size - 1;
            vector<pair<K, K>> scan_result;
            t.start();
            btree_range_search(btree, data[begin_pos].first, data[end_pos].first, scan_result);
            t.lap();
        }
        range_size *= 10;
        stats = hpt::summarize_ns(t.laps());
        auto suffix = string("10^") + to_string(i);
        cout << suffix << ": " << stats.mean << endl;
        t.reset();
    }
    #endif

    #if STABILITY_TEST
    cout << "\nstart stability test..." << endl;
    int insert_batch_times = 10;
    int one_insert = insert_data.size() / insert_batch_times;
    {
        int base_size = data.size();
        int range_size = STABILITY_RANGE_SIZE;
        std::mt19937 gen(TEST_SEED);
        std::uniform_int_distribution<uint64_t> distr(0, data.size() - range_size);
        data.resize(data.size() + insert_data.size());
        std::copy(insert_data.begin(), insert_data.end(), data.begin() + base_size);
        vector<int> tmp_query_index;
        tmp_query_index.reserve(PURE_QUERY_SIZE);
        for(int i = 0; i < PURE_QUERY_SIZE; ++i) {
            tmp_query_index.push_back(rand());
        }
        hpt::Timer tmp_t;
        tmp_t.laps_reserve(PURE_QUERY_SIZE);
        auto i_it = insert_data.begin();
        auto end_it = i_it;
        for(int i = 0; i <= insert_batch_times * 10; i += 10, base_size += one_insert) {
            while(i_it != end_it) {
                t.start();
                btree.insert(*i_it);
                t.lap();
                ++i_it;
            }
            end_it = i_it + one_insert;
            #if INSERT_STABILITY_TEST
            if(t.laps().size()) {
                stats = hpt::summarize_ns(t.laps());
                string tmp_label = label + "_insert_after_insert_" + to_string(i - 10) + "M";
                cout << tmp_label << ": " << stats.mean << endl;
                t.reset();
            }
            continue;
            #endif
            for(auto index_it = tmp_query_index.begin(); index_it != tmp_query_index.end(); ++index_it) {
                tmp_t.start();
                volatile auto x = *(btree.find(data[*(index_it) % base_size].first));
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            string tmp_label = label + "_query_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();

            for(int j = 0; j < range_test_count; ++j) {
                int begin_pos = distr(gen);
                int end_pos = begin_pos + range_size - 1;
                vector<pair<K, K>> scan_result;
                tmp_t.start();
                btree_range_search(btree, data[begin_pos].first, data[end_pos].first, scan_result);
                tmp_t.lap();
            }
            stats = hpt::summarize_ns(tmp_t.laps());
            tmp_label = label + '_' + to_string(range_size) + "range_after_insert_" + to_string(i) + "M";
            cout << tmp_label << ": " << stats.mean << endl;
            tmp_t.reset();
        }
    }
    #endif

    #if INSERT_ONLY_TEST
    cout << "\nstart insert only test..." << endl;
    for(auto i_it = insert_data.begin(); i_it != insert_data.end(); ++i_it) {
        btree.insert(*i_it);
    }
    #endif

    #if MEMORY_TEST
    prof.sample(label + "_afinsert_" + public_dataset);
    auto samples = prof.get_samples();
    for (size_t i = 0; i < samples.size() && i < 10; ++i) {
        auto &s = samples[i];
        std::cout << s.label << " ts(ns)=" << s.ts_ns << " rss=" << s.rss_bytes << " pss=" << s.pss_bytes << "\n";
    }
    #endif
    

    #if UPDATE_TEST
    cout << "\nstart update test..." << endl;
    volatile K x;
    while(load.have_operation()) {
        int type = load.get_operation_type();
        if(type == WorkLoad::QUERY) {
            pair<K, K> &_pair = data[load.get_operation_code()];
            t.start();
            x = btree.find(_pair.first).key();
            t.lap();
        } else if(type == WorkLoad::INSERT) {
            pair<K, K> &_pair = load.get_insert_pair(data, insert_data);
            t.start();
            btree.insert(_pair);
            t.lap();
        }
        load.next_operation();
    }
    stats = hpt::summarize_ns(t.laps());
    string tmp_label = label + "_update_test-i" + to_string(public_workload.get_insert_ratio());
    cout << tmp_label << ": " << stats.mean << endl;
    t.reset();

    #endif
}

//