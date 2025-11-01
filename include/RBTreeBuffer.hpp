#pragma once

#include "LM_Accessory.hpp"
#include <map>
#include <cassert>
#include <utility>

template <typename K, typename V>
class MapBuffer : public LMA::UpdatableBuffer<K, V> {
private:
    std::map<K, V> data;
    typename::std::map<K, V>::iterator cursor;
public:

    inline void bulk_load(const LMA::KVPair<K, V> *first, const LMA::KVPair<K, V> *last) override {
        while(first != last) {
            data.emplace(first->key, first->value);
            ++first;
        }
    }


    inline std::pair<bool, V> search(const K& key) override {
        auto it = data.find(key);
        return it != data.end() ? std::make_pair(true, it->second)
                                : std::make_pair(false, V(0));
    }

    inline void insert(const K& key, const V& value) override {
        data.insert_or_assign(key, value);
    }

    inline void erase(const K& key) override {
        data.erase(key);
    }

    inline size_t get_size() const override {
        return data.size();
    }

    inline const K& get_min() override {
        assert(!data.empty() && "Call get_min() on empty buffer");
        return data.begin()->first;
    }

    inline const K& get_max() override {
        assert(!data.empty() && "Call get_max() on empty buffer");
        return (--data.end())->first;
    }

    inline std::pair<K, V> back() override {
        return *data.rbegin();
    }

    inline void pop_back() override {
        data.erase(--data.end());
    }

    inline void clear() {
        data.clear();
    }
    
    inline void seq_begin() {
        cursor = data.begin();
    }

    inline void seq_end() {
        // cursor = --data.end();
        cursor = data.end();
    }

    inline bool seq_next(K &key, V& value) {
        if(cursor != data.end()) {
            key = cursor->first;
            value = cursor->second;
            ++cursor;
            return true;
        }
        return false;
    }

    inline bool seq_next() {
        if(cursor != data.end()) {
            ++cursor;
            return true;
        }
        return false;
    }

    inline void random_begin(const K &key) {
        cursor = data.lower_bound(key);
    }

    inline bool random_next(K &key, V& value) {
        return seq_next(key, value);
    }

    inline bool random_prev(K &key, V& value) {
        if(cursor == data.begin()) return false;
        --cursor;
        key = cursor->first;
        value = cursor->second;
        return true;
    }

    inline bool cursor_readable() {
        return cursor != data.end();
    }

    inline const K& cursor_key() {
        return cursor->first;
    }

    inline V& cursor_value() {
        return cursor->second;
    }

    template <typename Pair_It>
    inline void range_lookup(const K &key1, const K &key2, Pair_It &buffer) {
        auto first = data.lower_bound(key1);
        while(first != data.end() && first->first <= key2) {
            (*buffer++) = *(first++);
        }
    }

 };

template <typename K, typename V>
class MultimapBuffer : public LMA::UpdatableBuffer<K, V> {
private:
    std::multimap<K, V> data;
    typename::std::map<K, V>::iterator cursor;
public:
    inline void bulk_load(const LMA::KVPair<K, V> *first, const LMA::KVPair<K, V> *last) override {
        while(first != last) {
            data.emplace(first->key, first->value);
            ++first;
        }
    }

    inline std::pair<bool, V> search(const K& key) override {
        auto it = data.find(key);
        return it != data.end() ? std::make_pair(true, it->second)
                                : std::make_pair(false, V(0));
    }

    inline void insert(const K& key, const V& value) override {
        data.insert({key, value});
    }

    inline void erase(const K& key) override {
        data.erase(key);    //erase all same keys in multimap
    }

    inline size_t get_size() const override {
        return data.size();
    }

    inline const K& get_min() override {
        assert(!data.empty() && "Call get_min() on empty buffer");
        return data.begin()->first;
    }

    inline const K& get_max() override {
        assert(!data.empty() && "Call get_max() on empty buffer");
        return (--data.end())->first;
    }

    inline std::pair<K, V> back() override {
        return *data.rbegin();
    }

    inline void pop_back() override {
        data.erase(--data.end());
    }

    inline void clear() {
        data.clear();
    }
    
    inline void seq_begin() {
        cursor = data.begin();
    }

    inline void seq_end() {
        // cursor = --data.end();
        cursor = data.end();
    }

    inline bool seq_next(K &key, V& value) {
        if(cursor != data.end()) {
            key = cursor->first;
            value = cursor->second;
            ++cursor;
            return true;
        }
        return false;
    }

    inline bool seq_next() {
        if(cursor != data.end()) {
            ++cursor;
            return true;
        }
        return false;
    }

    inline void random_begin(const K &key) {
        cursor = data.lower_bound(key);
    }

    inline bool random_next(K &key, V& value) {
        return seq_next(key, value);
    }

    inline bool random_prev(K &key, V& value) {
        if(cursor == data.begin()) return false;
        --cursor;
        key = cursor->first;
        value = cursor->second;
        return true;
    }

    inline bool cursor_readable() {
        return cursor != data.end();
    }

    inline const K& cursor_key() {
        return cursor->first;
    }

    inline V& cursor_value() {
        return cursor->second;
    }

    inline void erase_cursor() {
        cursor = data.erase(cursor);
    }

    template <typename Pair_It>
    inline void range_lookup(const K &key1, const K &key2, Pair_It &buffer) {
        auto first = data.lower_bound(key1);
        while(first != data.end() && first->first <= key2) {
            (*buffer++) = *(first++);
        }
    }

 };