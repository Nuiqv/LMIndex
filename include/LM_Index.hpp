#pragma once

#include <iostream>
#include <utility>
#include <memory>
#include <cstring>
#include <algorithm>
#include <iterator>
#include "./LM_Accessory.hpp"
#include "./Multi_Linear_Map.hpp"
#include "./LM_Dynamic_Linear_Model.hpp"
#include "./skipList.hpp"
#include "./RBTreeBuffer.hpp"

namespace LMD
{
template <typename K, typename V> class SkiplistBuffer;

template <typename K, typename V, uint32_t Threshold = LMA::THRESHOLD, class Implement_Buffer = MapBuffer<K, V>, bool allow_duplicates = false>
class LM_Index {
    static_assert(std::is_base_of<LMA::UpdatableBuffer<K, V>, Implement_Buffer>::value,
                "Template class Implement_Buffer must derive from LMA::UpdatableBuffer");
protected:
    class Single_Model;
    using Node = LMA::KVPair<K, V>;
    using Maps = Updatable_Maps<K, Single_Model, std::allocator<Node>>;
    using Linear_Model = LMD::linear_model::Dynamic_Linear_Model<K, Node, Threshold>;
    using Buffer = LMA::UpdatableBuffer<K, V>;
    using Model_Alloc = typename std::allocator_traits<std::allocator<Node>>::template rebind_alloc<Single_Model>;
    using Buffer_Alloc = typename std::allocator_traits<std::allocator<Node>>::template rebind_alloc<Implement_Buffer>;
    static std::allocator<Node> alloc;

    Maps maps;
    Model_Alloc model_alloc;
    Buffer_Alloc buffer_alloc;
    Single_Model *model_head = nullptr, *model_tail = nullptr;
    Linear_Model *building_model = nullptr;
    //for update
    uint32_t _page_size;
    Implement_Buffer *front_buffer = nullptr;   //max size = _page_size
    //param
    const int max_map_expand_rate = 1;  //expand 2^rate
    const int min_map_shrink_rate = 2;  //shrink 1/2^rate

    inline void delete_model(Single_Model *model) {
        if(model->prev) model->prev->next = model->next;
        if(model->next) model->next->prev = model->prev;
        model->delete_data();
        if(model->buffer) {
            buffer_alloc.destroy((Implement_Buffer *)model->buffer);
            buffer_alloc.deallocate((Implement_Buffer *)model->buffer, 1);
        }
        model_alloc.deallocate(model, 1);
    }

    template<typename It>
    inline void bulk_load(It first, It last) {
        size_t n = last - first;
        It last_begin = first;
        It last_key;    //only valid when allow duplicates
        if(allow_duplicates) {
            for(last_key = last - 1; last_key != first && (last_key - 1)->first == last_key->first; --last_key);
        }

        Single_Model *last_model, *tmp_model;
        model_head = last_model = tmp_model = new (model_alloc.allocate(1)) Single_Model();

        auto new_segment = [&](){
            //data set
            if(allow_duplicates)
                tmp_model->new_data(first - last_begin);
            else
                tmp_model->new_data(building_model->get_num_of_points());
            tmp_model->bulk_load_from(last_begin, first);
            tmp_model->init_set_range();
            //model set
            #if USING_OETA
            building_model->get_single_model(*tmp_model);
            #else
            #if LMA_TEST
            LMA::lma_t.start();
            #endif
            least_squares(*tmp_model, tmp_model->data_set, tmp_model->data_set + tmp_model->data_size);
            #if LMA_TEST
            LMA::lma_t.lap();
            #endif
            tmp_model->data_num = building_model->get_num_of_points();
            building_model->clear_all();
            #endif
            //link set
            last_model->next = tmp_model;
            tmp_model->prev = last_model;
            last_model = tmp_model;
            tmp_model = new (model_alloc.allocate(1)) Single_Model();
        };

        auto _key = [](const std::pair<K, V> &it) constexpr {return it.first;};
        #if USING_GMA
        if(allow_duplicates) {
            while(!building_model->gnaw_duplicate_keys(first, last, last_key, n, _key)) {
                new_segment();
                last_begin = first;
            }
        } else {
            while(!building_model->gnaw(first, last, n, _key)) {
                new_segment();
                last_begin = first;
            }
        }
        #else
        while(!building_model->gnaw_linear(first, last, n, _key)) {
            new_segment();
        }
        #endif
        if(building_model->exist_current_model()) {
            new_segment();
            if(allow_duplicates) {
                if(last_begin->first == (first - 1)->first) {
                    last_model->data_num = first - last_begin;
                }
            }
        }
        
        //model head/tail link
        model_head->prev = nullptr;
        last_model->next = nullptr;
        model_tail = last_model;
        model_alloc.deallocate(tmp_model, 1);
        #if LMA_LM_LOAD
        LMA::lma_t.start();
        #endif
        //linear map build
        maps.build_exact(model_head->_min_key, model_tail->_max_key);
        #if LMA_LM_LOAD
        LMA::lma_t.lap();
        #endif
    }

    virtual inline Single_Model* train_seg(Single_Model *model, Node *first, Node *last) {
        building_model->clear_all();
        size_t _n = last - first;
        Node *last_first = first;
        Node* last_key;         //only valid when allow duplicates
        if(allow_duplicates) {
            for(last_key = last - 1; last_key != first && (last_key - 1)->key == last_key->key; --last_key);
        }

        Single_Model *tmp_model, *first_model;
        auto new_segment = [&]() {
            tmp_model = new (model_alloc.allocate(1)) Single_Model();
            building_model->get_single_model(*tmp_model);
            //link
            tmp_model->next = model;
            if(model->prev) {
                model->prev->next = tmp_model;
            }
            tmp_model->prev = model->prev;
            model->prev = tmp_model;
        };
        
        auto _key = [](const Node &it) constexpr {return it.key;};
        if(allow_duplicates) {
            building_model->gnaw_duplicate_keys(first, last, last_key, _n, _key);
        } else {
            building_model->gnaw(first, last, _n, _key);
        }
        new_segment();
        last_first = first;
        first_model = tmp_model;
        if(allow_duplicates) {
            while(!building_model->gnaw_duplicate_keys(first, last, last_key, _n, _key)) {
                new_segment();
                last_first = first;
            }
        } else {
            while(!building_model->gnaw(first, last, _n, _key)) {
                new_segment();
                last_first = first;
            }
        }
        if(building_model->exist_current_model()) {
            new_segment();
            if(allow_duplicates) {
                if(last_first->key == (first - 1)->key) {
                    model->prev->data_num = first - last_first;
                }
            }
        }
        return first_model;
    }

    inline void foremost_duplicate(int &pos, Single_Model &model) {
        K &key = model.key_at(pos);
        int limit = std::max(0, pos - 16);
        for(; pos != limit; --pos) {
            if(model.key_at(pos - 1) != key)
                return;
        }
        for(limit = 1; limit <= pos && model.key_at(pos - limit) >= key; limit *= 2); 
        pos = model.lower_bound_it(pos - std::min(limit, pos), pos - limit / 2, key);
    }

    //return backmost duplicate position
    inline void backmost_duplicate(int &pos, Single_Model &model) {
        K &key = model.key_at(pos);
        int limit = std::min(model.data_size - 1, pos + 16);
        for(; pos != limit; ++pos) {
            if(model.key_at(pos) != key)
                return;
        }
        int max_bound = model.data_size - pos;
        for(limit = 1; limit < max_bound && model.key_at(pos + limit) <= key; limit *= 2);
        pos = model.upper_bound_it(pos + limit / 2, pos + std::min(limit, max_bound - 1), key);
    }

    inline int search_iterator_upper_bound(const K& key, Single_Model &model) {
        int pos = model.cal_pos(key);
        auto exponential_right_search = [&]() {
            int index = 1;
            int max_bound = model.data_size - pos;
            while(index < max_bound && model.key_at(pos + index) <= key)
                index *= 2;
            return model.upper_bound_it(pos + index / 2, pos + std::min(index, max_bound - 1), key);
        };
        auto exponential_left_search = [&]() {
            int index = 1;
            while(index <= pos && model.key_at(pos - index) > key) 
                index *= 2;
            return model.upper_bound_it(pos - std::min(index, pos), pos - index / 2, key);
        };
        if(model.key_at(pos) < key) {
            int max_bound = pos + std::min<int>(model.min_error_threshold, model.data_size - pos - 1);
            pos = model.upper_bound_it(pos, max_bound, key);
            if(model.key_at(pos) == key && pos != max_bound)
                return pos;
            else {
                return exponential_right_search();
            }
        } else if(model.key_at(pos) > key) {
            Node *result = model.upper_bound(pos - std::min<int>(model.min_error_threshold, pos), pos, key);
            if(result->key == key)
                return result - model.data_set;
            else {
                pos = result- model.data_set;
                return exponential_left_search();
            }
        }
        if(allow_duplicates) {
            backmost_duplicate(pos, model);
        }
        return pos;
    }

    virtual inline int search_iterator(const K& key, Single_Model &model){
        int pos = model.cal_pos(key);
        if(model.key_at(pos) < key) {
            int right_offset = ADD_EPS(pos, model.min_error_threshold, model.data_size - 1);
            return model.lower_bound_it(pos, right_offset, key);
        } else if(model.key_at(pos) > key) {
            int left_offset = SUB_EPS(pos, model.min_error_threshold);
            return model.lower_bound_it(left_offset, pos, key);
        }
        if(allow_duplicates) {
            foremost_duplicate(pos, model);
        }
        return pos;
    }

    //return last model whose min <= key
    inline Single_Model* less_bound_map(const K &key) {
        Single_Model *pm = maps.lower_bound_map(key);
        if(!pm) pm = model_tail;
        if(pm->_min_key > key) pm = pm->prev;
        return pm;
    }

    //write kv node to model data, if exceed model's max key, try push back, or it will be pushed into buffer
    virtual inline bool write_node2segment(Single_Model &model, const K &key, const V &value) {
        int pos = search_iterator(key, model);
        if(model.key_at(pos) > key) {
            if(!model.key_exist(pos) || !model.key_exist(--pos)) {
                model.set_node(pos, key, value);
                ++model.data_num;
                return true;
            }
        } else if(model.key_at(pos) == key) {
            if(!model.key_exist(pos)) {
                ++model.data_num;
                model.value_at(pos) = value;
                return true;
            }
            if(allow_duplicates) {
                int last_pos = pos;
                for(++pos; pos < model.data_size && model.key_at(pos) == key; ++pos) {
                    if(model.key_exist(pos)) last_pos = pos;
                }
                if(++last_pos < model.data_size && !model.key_exist(last_pos)) {
                    ++model.data_num;
                    model.value_at(last_pos) = value;
                    return true;
                }
                return false;
            } else {
                model.value_at(pos) = value;
                return true;
            }
        } else if(key > model._max_key) {
            //expand maps
            if(key > model_tail->_max_key && 
                maps.forecast_layer_size(model_head->_min_key, key) > maps.get_max_layer_size() << max_map_expand_rate) {
                model.set_max_key(key);
                maps.build_exact(model_head->_min_key, key);
            } else {
                maps.expand_last(&model, key);
                model.set_max_key(key);
            }
            if(model.data_capacity > model.data_size) {
                uint32_t now_bound = std::abs(static_cast<long>(model.cal_pos_bound(key)) - static_cast<long>(model.data_size));
                if(now_bound <= Threshold) {
                    if(now_bound > model.min_error_threshold) model.min_error_threshold = now_bound;
                    model.resize(model.data_size + 1);
                    model.set_node(model.data_size - 1, key, value);
                    ++model.data_num;
                    return true;
                }
            }
        }
        return false;
    }

    inline Buffer* get_model_buffer(Single_Model &model) {
        if(!model.buffer)
            model.buffer = new (buffer_alloc.allocate(1)) Implement_Buffer();
        return model.buffer;
    }

    //merge buffer to model data
    inline void buffer_merge(Buffer *buffer, Single_Model *model) {
        Node *res;
        int _size1 = model->data_size, _size2 = buffer->get_size();
        if(_size1 == 0) {
            model->resize(_size2);
            res = model->data_set;
            for(buffer->seq_begin(); buffer->seq_next(res->key, res->value); ++res);
            buffer->clear();
            return;
        } else if(_size2 == 0) {
            return;
        }
        int now_size = _size1 + _size2;
        if(now_size > model->data_capacity) {
            Node *tmp = model->data_set;
            int tmp_size = model->data_capacity;
            res = new (alloc.allocate(now_size)) Node[now_size];
            model->set_data(res, now_size, now_size);
            Node *first = tmp, *last = tmp + _size1;
            buffer->seq_begin();
            while(first != last) {
                if(!buffer->cursor_readable()) {
                    std::memcpy(res, first, (last - first) * sizeof(Node));
                    buffer->clear();
                    alloc.deallocate(tmp, tmp_size);
                    return;
                }
                if(*first < buffer->cursor_key()) {
                    *(res++) = *(first++);
                } else {
                    buffer->seq_next(res->key, res->value);
                    ++res;
                }
            }
            for(; buffer->seq_next(res->key, res->value); ++res);
            buffer->clear();
            alloc.deallocate(tmp, tmp_size);
        } else {
            model->resize(now_size);
            model->data_num += _size2;
            res = model->data_set;
            Node *first = res - 1, *last = res + _size1 - 1;
            res += model->data_size - 1;
            buffer->seq_end();
            Node tmp_node;
            buffer->random_prev(tmp_node.key, tmp_node.value);
            while(last != first) {
                if(last->key < tmp_node.key) {
                    *(res--) = tmp_node;
                    if(!buffer->random_prev(tmp_node.key, tmp_node.value)) {
                        buffer->clear();
                        return;
                    }
                } else {
                    *(res--) = *(last--);
                }
            }
            for(*(res--) = tmp_node; buffer->random_prev(tmp_node.key, tmp_node.value); *(res--) = tmp_node);
            buffer->clear();
        }
    }

    virtual inline void load_data_set(size_t size, Node* first, Node* last, Single_Model *model) {
        model->new_data(size);
        model->bulk_load_from(first, last);
        model->init_set_range();
    }

    virtual inline void load_data_set(Node *data, const int &size, const int &capacity, Single_Model *model) {
        model->set_data(data, size, capacity);
        model->init_set_range();
    }

    virtual inline void merge_and_retrain(Buffer *buffer, Single_Model *model, bool alter_map = true) {
        model->clear_gaps();
        if(buffer) buffer_merge(buffer, model);
        Single_Model *fm = train_seg(model, model->data_set, model->data_set + model->data_size);
        Single_Model *pm = fm->next;
        Single_Model *next_model = model->next;
        Node *base_data = model->data_set;
        model->data_set = nullptr;
        //index status maintain
        if(!fm->prev) model_head = fm;
        if(!next_model) model_tail = model->prev;
        //result chain process
        auto alter_model_map = [&](Single_Model *first, Single_Model *last) {
            if(alter_map) {
                #if LMA_LM_UPDATE
                LMA::lma_t.start();
                #endif
                if(first->_max_key > model->_max_key)
                    maps.erase_model(model, true);
                else
                    maps.erase_model(model, false);
                #if LMA_LM_UPDATE
                LMA::lma_t.lap();
                #endif
                last = last->prev;
                delete_model(model);
                for(; last != first->prev; last = last->prev) {
                    #if LMA_LM_UPDATE
                    LMA::lma_t.start();
                    #endif
                    maps.draw_model(last);
                    #if LMA_LM_UPDATE
                    LMA::lma_t.lap();
                    #endif
                }
            } else {
                delete_model(model);
            }
        };
        load_data_set(base_data, fm->data_num, model->data_capacity, fm);
        int base_i = fm->data_num;
        if(pm != model) {
            while(pm->next != model) {
                load_data_set(pm->data_num, base_data + base_i, base_data + base_i + pm->data_num, pm);
                base_i += pm->data_num;
                pm = pm->next;
            }
            if(next_model && pm->data_num < Threshold << 1) {
                auto leave_num = pm->data_num;
                delete_model(pm);
                alter_model_map(fm, model);
                next_model->copy_into_front(base_data + base_i, leave_num);
                if(building_model->check_line(next_model->slope, next_model->intercept, next_model->_min_key, next_model->min_error_threshold, leave_num)) {
                    if(alter_map) {
                        maps.expand_first(next_model, base_data[base_i]);
                    }
                    next_model->set_min_key(base_data[base_i]);
                } else {
                    merge_and_retrain((Buffer *)next_model->buffer, next_model, alter_map);
                }
            } else {
                load_data_set(pm->data_num, base_data + base_i, base_data + base_i + pm->data_num, pm);
                alter_model_map(fm, model);
            }
        } else {
            alter_model_map(fm, pm);
        }
    }

    //if ideal size > 2 max layer size, return false (now need alter map, but rebuild)
    inline bool forecast_layer_map_alter(const K &min, const K &max) {
        auto forecast_size = maps.forecast_layer_size(min, max);
        if(forecast_size > maps.get_max_layer_size() << max_map_expand_rate)
            return false;
        return true;
    }

    //if ideal map size < 1/4 current size, return false
    inline bool forecast_small_map_alter(const K &min, const K &max) {
        auto forecast_size = maps.forecast_layer_size(min, max);
        if(forecast_size < maps.get_first_layer_size() >> min_map_shrink_rate)
            return false;
        return true;
    }

    template<typename It>
    inline void load_pair(Node *src, Node *src_last, It &res) {
        for(; src != src_last; ++src) {
            if(!src->is_deleted())
                *(res++) = *src;
        }
    }

    template<typename It>
    inline void load_pair(Node *src, Node *src_last, Buffer *buffer, It &res) {
        while(src != src_last) {
            if(!buffer->cursor_readable()) {
                load_pair(src, src_last, res);
                return;
            }
            if(src->is_deleted()) {
                ++src;
            } else if(src->key < buffer->cursor_key()) {
                *(res++) = *(src++);
            } else {
                buffer->random_next(res->first, res->second);
                ++res;
            }
        }
        for(; buffer->random_next(res->first, res->second); ++res);
    }

    template<typename It>
    inline void load_pair(Node *src, Node *src_last, Buffer *buffer, const K &buffer_max, It &res) {
        while(src != src_last) {
            if(!buffer->cursor_readable() || buffer->cursor_key() > buffer_max) {
                load_pair(src, src_last, res);
                return;
            }
            if(src->is_deleted()) {
                ++src;
            } else if(src->key < buffer->cursor_key()) {
                *(res++) = *(src++);
            } else {
                buffer->random_next(res->first, res->second);
                ++res;
            }
        }
        for(; buffer->cursor_readable() && buffer->cursor_key() <= buffer_max; ++res) {
            buffer->random_next(res->first, res->second);
        }
    }

    inline bool clear_empty_buffer(LMA::UpdatableBuffer<K, V> *buffer) {
        if(!buffer->get_size()) {
            buffer_alloc.destroy((Implement_Buffer *)buffer);
            buffer_alloc.deallocate((Implement_Buffer *)buffer, 1);
            return true;
        }
        return false;
    }

    virtual inline bool check_line_front(Single_Model *model, const K &new_front_key, const V &new_front_value) {
        int min_key_pos = model->first_min_key();
        if(min_key_pos == 0) {
            model->intercept += 1.0;
            double new_pos = -model->slope * static_cast<double>(model->_min_key - new_front_key) + model->intercept;
            if(new_pos >= -double(Threshold)) {
                Node tmp(new_front_key, new_front_value);
                model->copy_into_front(&tmp, 1);
                model->min_error_threshold = std::max<uint32_t>(model->min_error_threshold, std::ceil(std::abs(new_pos)));
                return true;
            }
            model->intercept -= 1.0;
        } else {
            double new_pos = -model->slope * static_cast<double>(model->_min_key - new_front_key) + model->intercept;
            if(new_pos > 0) {
                int true_pos = std::min<int>(min_key_pos - 1, new_pos);
                model->set_node(true_pos, new_front_key, new_front_value);
                for(int _pos = true_pos - 1; _pos >= 0 && model->key_at(_pos) >= new_front_key; --_pos) {
                    const K last_max_key = std::numeric_limits<K>::is_integer? new_front_key - 1 : new_front_key - std::numeric_limits<K>::epsilon();
                    model->set_node(_pos, last_max_key);
                }
                ++model->data_num;
                model->min_error_threshold = std::max<uint32_t>(model->min_error_threshold, std::ceil(std::abs(new_pos - true_pos)));
                return true;
            } else if(new_pos >= -double(Threshold)) {
                model->set_node(0, new_front_key, new_front_value);
                ++model->data_num;
                model->min_error_threshold = std::max<uint32_t>(model->min_error_threshold, -new_pos);
                return true;
            }
        }
        return false;
    }

    inline void alter_after_erase(const K &key, Single_Model *model) {
        if(key == model->_max_key) {
            int last_max_i = model->last_max_key();
            const K *p_max_key;
            if(model->buffer && model->buffer->get_max() > model->key_at(last_max_i)) {
                p_max_key = &model->buffer->get_max();
            } else {
                p_max_key = &model->key_at(last_max_i);
                model->resize(last_max_i + 1);
            }
            if(!model->next && !forecast_small_map_alter(model_head->_min_key, *p_max_key)) {
                model->set_max_key(*p_max_key);
                maps.build_exact(model_head->_min_key, *p_max_key);
            } else if(maps.shrink_model(model, *p_max_key, model->_max_key)) {
                model->set_max_key(*p_max_key);
            }   //if shrink fail, keep max_key unchange
        }
        const auto MIN_MODEL_SIZE = Threshold;
        if(--model->data_num <= MIN_MODEL_SIZE) {
            if(model->next) {
                if(model->prev) {
                    Single_Model *prev_model = model->prev;
                    if(prev_model->data_num > MIN_MODEL_SIZE + 1) {
                        int prev_max_key_pos = prev_model->last_max_key();
                        const K &prev_back_key = prev_model->key_at(prev_max_key_pos);
                        if(prev_model->buffer && prev_model->buffer->get_max() > prev_back_key) {
                            auto max_node_pair = prev_model->buffer->back();
                            if(check_line_front(model, max_node_pair.first, max_node_pair.second)) {
                                K tmp_prev_max = prev_model->_max_key;
                                prev_model->buffer->pop_back();
                                prev_model->_max_key = std::max(prev_back_key, prev_model->buffer->get_max());
                                if(!maps.shrink_model(prev_model, prev_model->_max_key, model->_min_key)) {
                                    maps.erase_model(prev_model, prev_model->_min_key, tmp_prev_max, false);
                                    maps.draw_model(prev_model);
                                }
                                maps.expand_first(model, max_node_pair.first);
                                model->set_min_key(max_node_pair.first);
                                return;
                            }
                        } else {
                            if(check_line_front(model, prev_back_key, prev_model->value_at(prev_max_key_pos))) {
                                K tmp_prev_max = prev_model->_max_key;
                                prev_model->delete_back(prev_max_key_pos);
                                if(!maps.shrink_model(prev_model, prev_model->_max_key, model->_min_key)) {
                                    maps.erase_model(prev_model, prev_model->_min_key, tmp_prev_max, false);
                                    maps.draw_model(prev_model);
                                }
                                maps.expand_first(model, prev_back_key);
                                model->set_min_key(prev_back_key);
                                return;
                            }
                        }
                    }

                    model->clear_gaps();
                    if(model->buffer) buffer_merge(model->buffer, model);
                    prev_model->clear_gaps();
                    if(prev_model->buffer) buffer_merge(prev_model->buffer, prev_model);
                    prev_model->copy_into_back(model->data_set, model->data_size);
                    maps.erase_model(model);
                    delete_model(model);
                    merge_and_retrain(prev_model->buffer, prev_model);
                } else {    //delete happen in first model
                    Single_Model *next_model = model->next;
                    model->clear_gaps();
                    if(model->buffer) buffer_merge(model->buffer, model);
                    next_model->copy_into_front(model->data_set, model->data_size);
                    maps.erase_model(model);
                    delete_model(model);
                    bool alter_map = forecast_small_map_alter(next_model->key_at(0), model_tail->_max_key);
                    merge_and_retrain(next_model->buffer, next_model, alter_map);
                    if(!alter_map) {
                        maps.build_exact(model_head->_min_key, model_tail->_max_key);
                    }
                }
            } else if(model->data_num == 0) {
                maps.erase_model(model);
                delete_model(model);
            }
        }
    }

    LM_Index(size_t max_map_layer = MAX_MAPPING_SIZE, uint32_t page_size = PAGESIZE) : maps(model_head, max_map_layer, alloc), model_alloc(Model_Alloc(alloc)), 
        buffer_alloc(Buffer_Alloc(alloc)), _page_size(page_size)
    {
        building_model = new Linear_Model();
    }

public:
    template<typename It>
    LM_Index(It first, It last, size_t max_map_layer = MAX_MAPPING_SIZE, uint32_t page_size = PAGESIZE) : LM_Index(max_map_layer, page_size) {
        bulk_load(first, last);
    }

    LM_Index(std::vector<std::pair<K, V>> &data, size_t max_map_layer = MAX_MAPPING_SIZE, uint32_t page_size = PAGESIZE) : LM_Index(data.begin(), data.end(), max_map_layer, page_size) 
    {}

    ~LM_Index() {
        while(model_head) {
            Single_Model *tmp_model = model_head;
            model_head = model_head->next;
            tmp_model->delete_data();
            if(tmp_model->buffer) {
                buffer_alloc.destroy((Implement_Buffer *)tmp_model->buffer);
                buffer_alloc.deallocate((Implement_Buffer *)tmp_model->buffer, 1);
            }
            model_alloc.deallocate(tmp_model, 1);
        }
        if(building_model) delete building_model;
        if(front_buffer) {
            buffer_alloc.destroy(front_buffer);
            buffer_alloc.deallocate(front_buffer, 1);
        }
    }

    std::pair<bool, V> search(const K& key) {
        Single_Model *model = maps.map_key_it(key);
        if(model) {
            int pos = search_iterator(key, *model);
            #if LMA_SEARCH_LENGTH
            int _gaps = std::abs(pos - model->cal_pos(key));
            total_search_length += _gaps;
            ++leaf_search_count;
            if(_gaps > max_search_gaps) {
                max_search_gaps = _gaps;
            }
            #endif
            if(model->key_at(pos) == key && model->key_exist(pos))
                return {true, model->value_at(pos)};
            if(model->buffer) {
                return std::move(model->buffer->search(key));
            }
        }
        if(key < model_head->_min_key) {
            if(front_buffer)
                return std::move(front_buffer->search(key));
        }
        return {false, 0};
    }

    std::vector<V> search_duplicates(const K& key) {
        if(!allow_duplicates) return {};
        std::vector<std::pair<K, V>> tmp;
        range_search(key, key, tmp);
        std::vector<V> results;
        results.reserve(tmp.size());
        for(auto it = tmp.begin(); it != tmp.end(); ++it) results.push_back(it->second);
        return results;
    }

    //search range in [key1, key2]
    void range_search(K key1, K key2, std::vector<std::pair<K, V>> &result) {
        key1 = std::max(key1, front_buffer? front_buffer->get_min(): model_head->_min_key);
        key2 = std::min(key2, model_tail->_max_key);
        if(key2 < key1) return;
        if(key1 == key2 && !allow_duplicates) {
            auto point_search_result = search(key1);
            if(point_search_result.first) {
                result.emplace_back(key1, point_search_result.second);
            }
            return;
        }
        Single_Model *first, *last;
        Node *first_node, *last_node;
        size_t reverse_size = 0;
        typename std::vector<std::pair<K, V>>::iterator res_it;
        if(key1 < model_head->_min_key) {
            reverse_size += front_buffer->get_size();
            first = model_head;
            first_node = first->data_set;
            for(last = first; last->_max_key < key2; last = last->next) {
                reverse_size += last->total_size();
            }
            if(allow_duplicates) 
                last_node = last->data_set + search_iterator_upper_bound(key2, *last);
            else
                last_node = last->data_set + search_iterator(key2, *last);
            if(last_node->key == key2) ++last_node;
            reverse_size += last_node - last->data_set + 1 + (last->buffer? last->buffer->get_size() : 0);
            result.resize(reverse_size);
            res_it = result.begin();
            front_buffer->range_lookup(key1, key2, res_it);
        } else {
            first = maps.lower_bound_map(key1);
            int tmp_position = search_iterator(key1, *first);
            first_node = first->data_set + tmp_position;
            if(first->_max_key < key2) {
                reverse_size += first->data_size + (first->buffer? first->buffer->get_size() : 0) - tmp_position;
                for(last = first->next; last->_max_key < key2; last = last->next) {
                    reverse_size += last->total_size();
                }
                tmp_position = 0;
            } else {
                last = first;
            }
            if(allow_duplicates) 
                last_node = last->data_set + search_iterator_upper_bound(key2, *last);
            else
                last_node = last->data_set + search_iterator(key2, *last);
            if(last_node->key == key2) ++last_node;
            reverse_size += last_node - last->data_set - tmp_position + 1 + (last->buffer? last->buffer->get_size() : 0);
            result.resize(reverse_size);
            res_it = result.begin();
        }
        if(first != last) {
            if(first->buffer && first->buffer->get_max() >= key1) {
                first->buffer->random_begin(key1);
                load_pair(first_node, first->data_set + first->data_size, first->buffer, res_it);
            } else {
                load_pair(first_node, first->data_set + first->data_size, res_it);
            }
            while(first->next != last) {
                first = first->next;
                if(first->buffer) {
                    first->buffer->seq_begin();
                    load_pair(first->data_set, first->data_set + first->data_size, first->buffer, res_it);
                } else {
                    load_pair(first->data_set, first->data_set + first->data_size, res_it);
                }
            }
            if(last->buffer && last->buffer->get_min() <= key2) {
                last->buffer->seq_begin();
                load_pair(last->data_set, last_node, last->buffer, key2, res_it);
            } else {
                load_pair(last->data_set, last_node, res_it);
            }
        } else {
            if(last->buffer) {
                last->buffer->random_begin(key1);
                load_pair(first_node, last_node, last->buffer, key2, res_it);
            } else {
                load_pair(first_node, last_node, res_it);
            }
        }
        result.resize(res_it - result.begin());
    }

    void insert(const K& key, const V &value) {
        //insert in front
        if(key < model_head->_min_key) {
            if(!front_buffer)
                front_buffer = new (buffer_alloc.allocate(1)) Implement_Buffer();
            front_buffer->insert(key, value);
            if(front_buffer->get_size() > _page_size) {
                //merge front
                size_t _n = front_buffer->get_size();
                front_buffer->seq_begin();
                model_head->copy_into_front(front_buffer, _n, model_head->data_size + _n + model_head->buffer_size());
                front_buffer->clear();

                bool alter_map = forecast_layer_map_alter(model_head->key_at(0), model_tail->_max_key);
                merge_and_retrain(model_head->buffer, model_head, alter_map);
                if(!alter_map) {
                    maps.build_exact(model_head->_min_key, model_tail->_max_key);
                }
                buffer_alloc.destroy(front_buffer);
                buffer_alloc.deallocate(front_buffer, 1);
                front_buffer = nullptr;
            }
        } else {
            //insert in model data or model buffer
            Single_Model *model = less_bound_map(key);

            if(write_node2segment(*model, key, value)) {
                return;
            }

            Buffer *now_buffer = get_model_buffer(*model);
            now_buffer->insert(key, value);
            if(now_buffer->get_size() > _page_size) {
                //merge
                if(key > model_tail->_max_key) {
                    bool alter_map = forecast_layer_map_alter(model_head->_min_key, key);
                    merge_and_retrain(now_buffer, model, alter_map);
                    if(!alter_map) {
                        maps.build_exact(model_head->_min_key, model_tail->_max_key);
                    }
                } else {
                    merge_and_retrain(now_buffer, model);
                }
            }
        }
    }

    void erase(const K &key) {
        if(key < model_head->_min_key) {
            if(front_buffer) {
                front_buffer->erase(key);
                if(clear_empty_buffer(front_buffer))
                    front_buffer = nullptr;
            }
            return;
        }
        Single_Model *model = maps.map_key_it(key);
        if(model) {
            int pos = search_iterator(key, *model);
            if(allow_duplicates) {
                int delete_num = 0;
                for(int tmp_pos = pos + 1; tmp_pos < model->data_size && model->key_at(tmp_pos) == key && model->key_exist(tmp_pos); ++tmp_pos, ++delete_num)
                    model->delete_at(tmp_pos);
                model->data_num -= delete_num;
            }
            if(model->key_at(pos) == key) {
                if(model->key_exist(pos))
                    model->delete_at(pos);
                else
                    return;
                alter_after_erase(key, model);
            } else if(model->buffer) {
                model->buffer->erase(key);
                if(clear_empty_buffer(model->buffer))
                    model->buffer = nullptr;
                if(model->_max_key == key) {
                    int last_max_i = model->last_max_key();
                    const K *p_max_key;
                    if(model->buffer && model->buffer->get_max() > model->key_at(last_max_i)) {
                        p_max_key = &model->buffer->get_max();
                    } else {
                        p_max_key = &model->key_at(last_max_i);
                        model->resize(last_max_i + 1);
                    }
                    if(!model->next && !forecast_small_map_alter(model_head->_min_key, *p_max_key)) {
                        model->set_max_key(*p_max_key);
                        maps.build_exact(model_head->_min_key, *p_max_key);
                    } else if(maps.shrink_model(model, *p_max_key, model->_max_key)) {
                        model->set_max_key(*p_max_key);
                    }   //if shrink fail, keep max_key unchange
                }
            }
        }
    }

    //debug
    void debug_multi_linear_model_cal() {
        std::vector<vector<Node>> model_first_set;
        model_first_set.push_back(std::vector<Node>());
        for(Single_Model *pm = model_head; pm != nullptr; pm = pm->next) {
            model_first_set[0].push_back(Node(pm->_min_key, 0));
        }
        building_model->clear_all();
        
        for(int i = 0; model_first_set[i].size() > 1; ++i) {
            size_t n = model_first_set[i].size();
            auto first = model_first_set[i].begin(), last = model_first_set[i].end();
            auto last_begin = first;
            model_first_set.push_back(std::vector<Node>());
            while(!building_model->gnaw(first, last, n)) {
                model_first_set[i+1].push_back(Node(last_begin->key, 0));
                last_begin = first;
                building_model->clear_all();
            }

            if(building_model->exist_current_model()) {
                model_first_set[i+1].push_back(Node(last_begin->key,0));
                last_begin = first;
                building_model->clear_all();
            }
        }

        std::cout << "Recurse build linear model:\n" 
            << "totol height:   " << model_first_set.size() << '\n'
            << "each layer size: ";
        for(auto &vec : model_first_set) {
            std::cout << vec.size() << " -> ";
        }
        std::cout << '\n';
    }

    void debug_model_info() {
        Single_Model *pm = model_head;
        int count = 0;
        size_t total_size = 0, max_num = 0;
        double model_intercept = 0;
        double model_slope = 0;
        for(; pm != nullptr; pm = pm->next) {
            ++count;
            total_size += pm->data_num;
            model_intercept += pm->intercept;
            model_slope += pm->slope;
            if(pm->data_num > max_num) {
                max_num = pm->data_num;
            }
        }
        std::cout << "the number of model: " << count << '\n'
            << "avg model size: " << (double)total_size / count << "\n"
            << "max model size: " << max_num << "\n"
            << "model avg intercept: " << model_intercept / count << "\n"
            << "model avg slope: " << model_slope / count << "\n";
        std::cout << "map info:\n";
        maps.debug_map_layers_info();
    }

    void debug_model_info(std::map<std::string, std::string> &info) {
        Single_Model *pm = model_head;
        int count = 0;
        size_t total_size = 0, max_num = 0;
        double model_intercept = 0;
        double model_slope = 0;
        for(; pm != nullptr; pm = pm->next) {
            ++count;
            total_size += pm->data_num;
            model_intercept += pm->intercept;
            model_slope += pm->slope;
            if(pm->data_num > max_num) {
                max_num = pm->data_num;
            }
        }
        info["model_number"] = std::to_string(count);
        info["avg_model_size"] = std::to_string((double)total_size / count);
        info["max_model_size"] = std::to_string(max_num);
        info["avg_intercept"] = std::to_string(model_intercept / count);
        info["avg_slope"] = std::to_string(model_slope / count);
        maps.debug_map_layers_info(info);
    }

    #if LMA_SEARCH_LENGTH
    size_t total_search_length = 0;
    size_t leaf_search_count = 0;
    size_t max_search_gaps = 0;
    
    void debug_leaf_search_length() {
        cout << "avg search length: " << double(total_search_length) / leaf_search_count << "\n";
        cout << "max search length: " << max_search_gaps << "\n";
        cout << "leaf search count: " << leaf_search_count << "\n";
    }

    void debug_leaf_search_length(std::map<std::string, std::string> &info) {
        info["avg_search_length"] = std::to_string(double(total_search_length) / leaf_search_count);
        info["max_search_length"] = std::to_string(max_search_gaps);
        info["leaf_search_count"] = std::to_string(leaf_search_count);
    }
    #endif

    #if SHOW_MAP_SEARCH_DEPTH
    void debug_show_debug_length() {
        std::cout << "map search count: " << maps.depth_count
            << "\navg search depth: " << (double)maps.all_search_length / maps.depth_count
            << "\nmax search depth: " << maps.max_search_depth
            << "\n";
    }

    void debug_show_debug_length(std::map<std::string, std::string> &info) {
        info["lm_total_search"] = std::to_string(maps.depth_count);
        info["avg_lm_search_depth"] = std::to_string((double)maps.all_search_length / maps.depth_count);
        info["max_lm_search_depth"] = std::to_string(maps.max_search_depth);
    }
    #endif

    bool debug_data_num(const K &key) {
        Single_Model *model = maps.map_key_it(key);
        if(model) {
            auto key = model->debug_get_keys_num();
            if(key != model->data_num) {
                std::cout << "some error happen!\n";
                return false;
            }
            return true;
        }
        return false;
    }

    size_t debug_lm_total_size() {
        return maps.debug_total_layer_size();
    }

    #if !USING_OETA || LMA_TEST
    void least_squares(Single_Model &_model, Node *first, Node *last) {
        double sumKey = 0,sumIndex = 0, sumKeyIndex = 0, sumKeySquared = 0;
        int cnt = 0;
        const K &first_key = *first;
        for(; first != last; ++first) {
            K offset_key = first->key - first_key;
            sumKey += (double)offset_key;
            sumIndex += cnt;
            sumKeyIndex += (double)offset_key * cnt;
            sumKeySquared += (double)pow(offset_key,2);
            cnt++;
        }
        double avg_key = sumKey/cnt, avg_index = sumIndex/cnt;
        _model.slope = (sumKeyIndex - sumKey * avg_index)/(sumKeySquared - sumKey*avg_key);
        _model.intercept = avg_index - avg_key * _model.slope;
    }

    void debug_seg_avg_max_error(std::map<std::string, std::string> &info, std::string prefix) {
        size_t seg_total_error = 0, seg_max_error = 0, total_avg_max = 0, model_count = 0;
        double total_avg_avg = 0;
        for(auto _model = model_head; _model != nullptr; _model = _model->next, ++model_count) {
            seg_total_error = seg_max_error = 0;
            for(int i = 0; i < _model->data_size; ++i) {
                if(_model->key_exist(i)) {
                    int error = std::abs(static_cast<int>(_model->slope * static_cast<double>(_model->key_at(i) - _model->_min_key) + _model->intercept) - i);
                    seg_total_error += error;
                    if(error > seg_max_error)
                        seg_max_error = error;
                }
            }
            total_avg_max += seg_max_error;
            total_avg_avg += (double)seg_total_error / _model->data_num;
        }
        std::string label = "avg_seg_avg_error";
        info[label] = std::to_string(total_avg_avg / model_count);
        std::cout << prefix + "_" + label << ": " << info[label] << "\n";
        label = "avg_seg_max_error";
        info[label] = std::to_string((double)total_avg_max / model_count);
        std::cout << prefix + "_" + label << ": " << info[label] << "\n";
    }
    #endif
    //
};
template <typename K, typename V, uint32_t Threshold, class Implement_Buffer, bool allow_duplicates>
std::allocator<typename LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Node> LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::alloc = std::allocator<typename LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Node>();

template <typename K, typename V>
class SkiplistBuffer : public LMA::UpdatableBuffer<K, V> {
public:
    SkipList<K, V> list;

    inline void bulk_load(const LMA::KVPair<K, V> *first, const LMA::KVPair<K, V> *last) override {
        while(first != last) {
            list.insert_key(first->key, first->value);
            ++first;
        }
    }

    inline std::pair<bool, V> search(const K &key) {
        bool found;
        V res = list.lookup(key, found);
        return {found, res};
    }       

    inline void insert(const K &key, const V &value) {
        list.insert_key(key, value);
    }

    inline void erase(const K &key) {
        list.delete_key(key);
    }

    inline size_t get_size() const {
        return list.num_elements();
    }

    inline const K& get_min() {
        return list.min_key();
    }

    inline const K& get_max() {
        return list.max_key();
    }

    inline std::pair<K, V> back() override {
        return std::make_pair(list.max_key(), list.max_key_value());
    }

    inline void pop_back() override {
        list.delete_key(list.max_key());
    }

    inline void clear() {
        if(get_size() == 0)
            list.seq_recover();
        else
            list.clear();
    }

    inline void seq_begin() {
        list.seq_reset();
    }

    inline void seq_end() {
        list.seq_bound_end();
        list.seq_set_in_bound();
        list.seq_node_next();
    }

    inline bool seq_next(K &key, V& value) {
        return list.seq_next_del(key, value);
    }

    inline bool seq_next() {
        if(list.sequence == list.p_listTail) return false;
        list.sequence = list.sequence->_forward[1];
        return true;
    }

    inline void random_begin(const K &key) {
        list.seq_lower_bound(key);
        list.seq_set_in_bound();
    }

    inline bool random_next(K &key, V& value) {
        return list.seq_next(key, value);
    }

    inline bool random_prev(K &key, V& value) {
        return list.seq_prev(key, value);
    }

    inline bool cursor_readable() {
        return list.seq_not_end();
    }

    inline const K& cursor_key() {
        return list.seq_key_now();
    }

    inline V& cursor_value() {
        return list.seq_value_now();
    }

    template<typename Pair_It>
    void range_lookup(const K &key1, const K &key2, Pair_It &buffer) {
        list.range_lookup(key1, key2, buffer);
    }
};

template <typename K, typename V, uint32_t Threshold, class Implement_Buffer, bool allow_duplicates>
class LM_Index_Gaps;

template <typename K, typename V, uint32_t Threshold, class Implement_Buffer, bool allow_duplicates>
class LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Single_Model {
public:
    friend class LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>;
    friend class LM_Index_Gaps<K, V, Threshold, Implement_Buffer, allow_duplicates>;
    using self_type = LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Single_Model;
    
    //states
    double slope;
    double intercept;
    int data_num = 0;           //actual existing number of node
    int data_size = 0;          //max data size
    int data_capacity = 0;      //max data capacity
    uint32_t min_error_threshold;
    K _max_key = K(0);
    K _min_key = K(0);
    self_type *prev = nullptr;
    self_type *next = nullptr;
    LMA::UpdatableBuffer<K, V> *buffer = nullptr;

    inline void set_max_key(const K &max_k) {
        _max_key = max_k;
    }

    inline void set_min_key(const K &min_k) {
        intercept += (min_k > _min_key? slope * static_cast<double>(min_k - _min_key) : -slope * static_cast<double>(_min_key - min_k));
        _min_key = min_k;
    }

    //assert: key >= _min_key, range in [0, data_size - 1]
    inline int cal_pos(const K &key) const {
        int pos = std::max(static_cast<int>(std::min(slope * static_cast<double>(key - _min_key) + intercept, static_cast<double>(data_size - 1))), 0);
        return pos;
    }

    //cal pos, range in [0, data_capacity - 1]
    inline int cal_pos_bound(const K &key) const {
        int pos = std::max(static_cast<int>(std::min(slope * static_cast<double>(key - _min_key) + intercept, static_cast<double>(data_capacity - 1))), 0);
        return pos;
    }

    inline K get_interval_size() const {return _max_key - _min_key;}

    inline int buffer_size() const {
        return buffer ? static_cast<int>(buffer->get_size()) : 0;
    }
    
    //data
protected:
    Node *data_set = nullptr;

public:
    inline K& key_at(const int& pos) {
        return data_set[pos].key;
    }

    inline V& value_at(const int& pos) {
        return data_set[pos].value;
    }

    inline void delete_at(const int &pos) {
        data_set[pos].deleted();
    }

    inline K& back_key() {
        return data_set[data_size - 1].key;
    }

    inline void set_node(const int &pos, const K &key, const V &value) {
        data_set[pos].set(key, value);
    }

    //value set tombstone without param 
    inline void set_node(const int &pos, const K &key) {
        data_set[pos].set(key);
    }

    inline void set_data(Node *data, const int &size, const int &capacity) {
        data_set = data;
        data_capacity = capacity;
        data_num = size;
        data_size = size;
    }

    inline void init_set_range() {      //set key range by data
        if(data_set) {
            _min_key = data_set[0].key;
            _max_key = data_set[data_size - 1].key;
        }
    }

    inline void set_range(const K &min, const K &max) {
        _min_key = min;
        _max_key = max;
    }

    inline void new_data(const int &capacity) {
        delete_data();
        data_set = new (alloc.allocate(capacity)) Node[capacity];
        data_size = data_capacity = capacity;
    }

    inline void delete_data() {
        if(data_set) {
            alloc.deallocate(data_set, data_capacity);
            data_set = nullptr;
        }
    }

    template<class It>
    void bulk_load_from(It first, It last) {
        if(data_set) {
            Node *dest = data_set;
            for(; first != last; (void)++first, (void)++dest) 
                *dest = *first;
        }
    }

    //resize data set and no guarantee null init for expand resize
    void resize(const int &size) {
        if(size > data_capacity) {
            Node *_data = new (alloc.allocate(size)) Node[size];
            std::memcpy(_data, data_set, data_size * sizeof(Node));
            delete_data();
            data_set = _data;
            data_size = data_capacity = size;
        } else {
            data_size = size;
        }
    }

    void resize(const int &size, const K &key) {
        if(size > data_size) {
            if(size > data_capacity) {
                Node *_data = new (alloc.allocate(size)) Node[size];
                std::memcpy(_data, data_set, data_size * sizeof(Node));
                delete_data();
                data_set = _data;
                data_capacity = size;
            }   
            for(Node *node = data_set + data_size; node != data_set + size; ++node) {
                node->set(key);
            }
        }
        data_size = size;
    }

    /**
     * @param size: specifically set data size, for some case need resize immediateky after copy front data
     */
    inline void copy_into_front(Node *front, const int &count, const int &size = 0) {
        int _size = std::max(data_size + count, size);
        if(_size > data_capacity) {
            Node *_data = new (alloc.allocate(_size)) Node[_size];
            std::memcpy(_data, front, count * sizeof(Node));
            std::memcpy(_data + count, data_set, data_size * sizeof(Node));
            delete_data();
            data_set = _data;
            data_capacity = _size;
        } else {
            for(Node *p1 = data_set + _size - 1, *p2 = data_set + data_size - 1, *last = data_set - 1; p2 != last;) {
                *(p1--) = *(p2--);
            }
            std::memcpy(data_set, front, count * sizeof(Node));
        }
        data_size = _size;
        data_num += count;
    }

    //assert: have set cursor begin in Buffer
    inline void copy_into_front(Buffer *front, const int &count, const int &size = 0) {
        int _size = std::max(data_size + count, size);
        if(_size > data_capacity) {
            Node *_data = new (alloc.allocate(_size)) Node[_size];
            for(Node *pn = _data; front->random_next(pn->key, pn->value); ++pn);
            std::memcpy(_data + count, data_set, data_size * sizeof(Node));
            delete_data();
            data_set = _data;
            data_capacity = _size;
            data_num += count;
        } else {
            for(Node *p1 = data_set + _size - 1, *p2 = data_set + data_size - 1, *last = data_set - 1; p2 != last;) {
                *(p1--) = *(p2--);
            }
            for(Node *pn = data_set; front->random_next(pn->key, pn->value); ++pn);
        }
        data_size = _size;
        data_num += count;
    }

    inline void copy_into_back(Node *back, const int &count) {
        int _size = data_size + count;
        if(_size > data_capacity) {
            Node *_data = new (alloc.allocate(_size)) Node[_size];
            std::memcpy(_data, data_set, data_size * sizeof(Node));
            std::memcpy(_data + data_size, back, count * sizeof(Node));
            delete_data();
            data_set = _data;
            data_capacity = _size;
        } else {
            for(Node *pt = data_set + data_size, *last = pt + count; pt != last;) {
                *(pt++) = *(back++);
            }
        }
        data_size = _size;
        data_num += count;
    }

    inline void clear_gaps() {
        if(data_size != data_num) {
            Node* p1 = data_set;
            for(Node *p2 = data_set; p2 != data_set + data_size; ++p2) {
                if(!p2->is_deleted()) {
                    if(p1 != p2)
                        *p1 = std::move(*p2);
                    ++p1;
                }
            }
            data_size = p1 - data_set;
        }
    }

    inline Node *lower_bound(const int &first, const int &last, const K &key) const {
        return std::lower_bound(data_set + first, data_set + last, key);
    }

    inline Node *upper_bound(const int &first, const int &last, const K &key) const {
        return std::upper_bound(data_set + first, data_set + last, key);
    }

    inline int lower_bound_it(const int &first, const int &last, const K &key) const {
        return lower_bound(first, last, key) - data_set;
    }

    inline int upper_bound_it(const int &first, const int &last, const K &key) const {
        return upper_bound(first, last, key) - data_set;
    }
    
    inline bool key_exist(const int &pos) const {
        return !data_set[pos].is_deleted();
    }

    inline size_t total_size() const {
        return data_num + (buffer? buffer->get_size() : 0);
    }

    //return last existing max key index, only vaild when data_num > 0
    inline int last_max_key() const {
        int pos = data_size - 1;
        for(; data_set[pos].is_deleted(); --pos);
        return pos;
    }

    //only vaild when data_num > 0
    inline int last_max_key(const int &begin_pos) const {
        int pos = begin_pos - 1;
        for(; data_set[pos].is_deleted(); --pos);
        return pos;
    }

    //return first exist key position, only vaild when data_num > 0
    inline int first_min_key(int pos = 0) const {
        for(; data_set[pos].is_deleted(); ++pos);
        return pos;
    }

    //return last filled key position, return bound - 1 if over bound
    inline int last_filled_key(int pos, int bound = 0) const {
        for(--pos; pos >= bound && data_set[pos].is_deleted(); --pos);
        return pos;
    }

    inline int next_filled_key(int pos, int bound = 0) const {
        for(++pos; pos < bound && data_set[pos].is_deleted(); ++pos);
        return pos;
    }

    //delete range [pos, last) keys using @param key
    inline void delete_range_key(int pos, const int &last, const K &key) {
        for(; pos < last; ++pos)
            data_set[pos].set(key);
    }

    inline void delete_back(const int &back_pos) {
        data_set[back_pos].deleted();
        --data_num;
        int _pos = last_max_key(back_pos);
        _max_key = data_set[_pos].key;
    }

    //debug
    inline int debug_get_keys_num() {
        int true_num = 0;
        for(int i = 0; i < data_size; ++i) {
            if(!data_set[i].is_deleted())
                ++true_num;
        }
        return true_num;
    }
    //
};



}   //namespace LMD