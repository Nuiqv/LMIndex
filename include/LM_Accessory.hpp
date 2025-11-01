#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <utility>
#include <initializer_list>
#include <limits>
#include <type_traits>
//debug
#include "../benchmark/timer/hptimer.hpp"
#include "../benchmark/memory/mem_profile.hpp"
//


namespace LMA {

const size_t THRESHOLD = 64ULL;
#define BLOCK_STACK_SIZE 1u << 10
#define PAGESIZE 0
#define SUB_EPS(x, threshold) ((x) <= (threshold) ? 0 : ((x) - (threshold)))
#define ADD_EPS(x, threshold, size) ((x) + (threshold) + 1 >= (size) ? (size) : (x) + (threshold) + 1)
#define USING_GMA 1
#define USING_OETA 1
#define MAX_MAPPING_SIZE 65536 //2^16
#define MAX_SEGMENT_FULL_RATE 0.9
#define INF_LAYER_CAPACITY 8

//debug
#define LMA_TEST 0
#define SHOW_MAP_SEARCH_DEPTH 0
#define LMA_SEARCH_LENGTH 0
#define LMA_LM_UPDATE 0
#define LMG_INSERT_COMPARE 0
#define LMA_LM_LOAD 0
//Binary search vs. expentional search
#define LMA_BS_TEST 0
#define LMA_ES_TEST 0
//
#if LMA_LM_UPDATE || LMA_TEST || LMA_LM_LOAD || LMA_BS_TEST || LMA_ES_TEST
hpt::Timer lma_t;
// hpt::ns lma_all_ns{0};
// int cut_count = 0;
#endif

template<typename T, std::enable_if_t<std::is_pointer_v<T>, int> = 0>
constexpr T get_tombstone() { return new std::remove_pointer_t<T>(); }

template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
constexpr T get_tombstone() { return std::numeric_limits<T>::max(); }

template <typename K, typename V>
struct KVPair {
    static const V TOMBSTONE = get_tombstone<V>();
    
    K key;
    V value;
    
    KVPair() = default;
    KVPair(const KVPair&) = default;
    explicit KVPair(const K &_key, const V& _value) : key(_key), value(_value) {}
    KVPair(const std::pair<K, V> &_pair) : key(_pair.first), value(_pair.second) {}

    inline bool is_deleted() const {
        return value == TOMBSTONE;
    }

    inline void deleted() {
        value = TOMBSTONE;
    }

    inline void set(const K &_key, const V& _value) {key = _key; value = _value;}

    inline void set(const K &_key) {
        key = _key; value = TOMBSTONE;
    }
    
    bool operator==(const KVPair &kv) const {
        return (kv.key == key && kv.value == value);
    }
    
    bool operator!=(const KVPair &kv) const {
        return (kv.key != key && kv.value != value);
    }
    
    bool operator<(const KVPair &kv) const{
        return key < kv.key;
    }
    
    bool operator>(const KVPair &kv) const{
        return key > kv.key;
    }

    operator K() const { return key; }
    operator std::pair<K, V>() const {return std::make_pair(key, value);}

};


template<typename T>
using BeSigned = typename std::conditional_t<std::is_floating_point<T>::value,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128_t>>;

template<typename K, typename P>
struct Slope {
    BeSigned<K> dx{};
    BeSigned<P> dy{};

    bool operator<(const Slope &p) const { return dy * p.dx < dx * p.dy; }
    bool operator>(const Slope &p) const { return dy * p.dx > dx * p.dy; }
    bool operator<=(const Slope &p) const { return dy * p.dx <= dx * p.dy; }
    bool operator>=(const Slope &p) const { return dy * p.dx >= dx * p.dy; }
    bool operator==(const Slope &p) const { return dy * p.dx == dx * p.dy; }
    bool operator!=(const Slope &p) const { return dy * p.dx != dx * p.dy; }
    explicit operator long double() const { return dy / (long double) dx; }
};

template<typename K, typename P>
struct Point {
    K x{};
    P y{};

    Slope<K, P> operator-(const Point &p) const { return {BeSigned<K>(x) - p.x, BeSigned<P>(y) - p.y}; }
};

template<class DT>
void data_read(const char* path, std::vector<DT>& data)
{
    std::ifstream ifs(path, std::ios::binary);
    if(ifs.is_open()){
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        data.resize(size);
        ifs.read(reinterpret_cast<char*>(data.data()), sizeof(DT) * size);
        ifs.close();
    }
}

template<class DT>
void data_save(const char* path, const std::vector<DT>& data)
{
    std::ofstream ofs(path, std::ios::binary);
    if(ofs.is_open()){
        size_t size = data.size();
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(data.data()), sizeof(DT) * size);
        ofs.close();
    }
}

template <typename K, typename V>
class UpdatableBuffer {
public:
    virtual inline void bulk_load(const KVPair<K, V> *first, const KVPair<K, V> *last) = 0;
    virtual inline std::pair<bool, V> search(const K& key) = 0;
    virtual inline void insert(const K &key, const V &value) = 0;
    virtual inline void erase(const K &key) = 0;
    virtual inline size_t get_size() const = 0;
    virtual inline const K& get_min() = 0;
    virtual inline const K& get_max() = 0;
    virtual inline std::pair<K, V> back() = 0;
    virtual inline void pop_back() = 0;
    virtual ~UpdatableBuffer() = 0;

    //cursor part
    virtual inline void clear() = 0;
    //set sequence cursor in begin
    virtual inline void seq_begin() = 0;
    //set cursor in last position
    virtual inline void seq_end() = 0;
    //if cursor in buffer end, return false, or get KV in cursor now, and move cursor to next position
    virtual inline bool seq_next(K &key, V& value) = 0;
    virtual inline bool seq_next() = 0;
    //set cursor in lower_bound of key
    virtual inline void random_begin(const K &key) = 0;
    //get K/V, and then cursor move to next position
    virtual inline bool random_next(K &key, V& value) = 0;
    virtual inline bool random_prev(K &key, V& value) = 0;
    virtual inline bool cursor_readable() = 0;
    virtual inline const K& cursor_key() = 0;
    virtual inline V& cursor_value() = 0;
    virtual inline void erase_cursor(){}
};

template <typename K, typename V> UpdatableBuffer<K, V>::~UpdatableBuffer(){}



}