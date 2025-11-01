#pragma once

#include "LM_Index.hpp"
#include "LM_Dynamic_Linear_Model.hpp"

namespace LMD
{

template <typename K, typename V, uint32_t Threshold = LMA::THRESHOLD, class Implement_Buffer = MapBuffer<K, V>, bool allow_duplicates = false>
class LM_Index_Gaps : public LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates> {
protected:
    using Single_Model = typename LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Single_Model;
    using Node = typename LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Node;
    using Buffer = typename LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::Buffer;
    float expansion_rate;
    static const K MIN_KEY;                         //key must be numeric
    const int GAP_SEEK_RANGE = Threshold << 3;      //gaps search range

    constexpr static uint8_t ceil_log2(uint32_t n) { return n <= 1 ? 0 : sizeof(uint32_t) * 8 - __builtin_clz(n - 1); }

    #if LMG_INSERT_COMPARE
    template<typename It, typename KEY>
    inline void load_data_set(size_t size, It first, It last, Single_Model *model, KEY _key) {
        int num = size;
        size += std::ceil(size * expansion_rate);
        if(model->data_set) {
            model->resize(size);
        } else {
            model->new_data(size);
        }
        model->set_range(_key(*first), _key(*(last - 1)));
        model->min_error_threshold = 1 << ceil_log2(model->min_error_threshold);
        model->intercept += model->min_error_threshold >> 1;
        model->slope *= (1 + expansion_rate);
        Node *data = model->data_set;
        int last_pos = 0;   //last valid position
        K last_key = MIN_KEY;
        //test
        int max_error = 0;
        for(; first != last; ++last_pos, ++first, --num) {
            int pos = std::max(model->cal_pos(_key(*first)), last_pos);
            if(size - pos < num) {
                pos = size - num;
                for(; last_pos != pos; ++last_pos) {
                    data[last_pos].set(last_key);
                }
                data += pos;
                do{
                    //test
                    pos = std::abs(last_pos++ - model->cal_pos(_key(*first)));
                    if(pos > max_error)  max_error = pos;
                    //
                    *(data++) = *(first++);
                } while(first != last);
                //test
                if(max_error < model->min_error_threshold) {
                    model->min_error_threshold = 1 << ceil_log2(model->min_error_threshold);
                }
                return;
            }
            data[pos] = *first;
            for(; last_pos != pos; ++last_pos) {  //set gaps to last key
                data[last_pos].set(last_key);
            }
            last_key = _key(*first);
        }
        for(; last_pos != size; ++last_pos) {
            data[last_pos].set(last_key);
        }
        //test
        if(max_error < model->min_error_threshold) {
            model->min_error_threshold = 1 << ceil_log2(model->min_error_threshold);
        }
        //
    }
    
    #else

    template<typename It, typename KEY>
    inline void load_data_set(size_t size, It first, It last, Single_Model *model, KEY _key) {
        int num = size;
        size += std::ceil(size * expansion_rate);
        if(model->data_set) {
            model->resize(size);
        } else {
            model->new_data(size);
        }
        model->set_range(_key(*first), _key(*(last - 1)));
        model->min_error_threshold = 1 << ceil_log2(model->min_error_threshold);
        model->intercept += model->min_error_threshold >> 1;
        model->slope *= (1 + expansion_rate);
        Node *data = model->data_set;
        K last_key = MIN_KEY;
        int max_error = 0;

        int C, Cf, Cb, s;
        Cf = 0;
        Cb = model->cal_pos(_key(*first));
        while(first != last) {
            C = Cb;
            s = 0;
            auto tmp_it = first;
            if(allow_duplicates) {
                for(; C == Cb && ++tmp_it != last; ++s) {
                    if(_key(*tmp_it) != _key(*(tmp_it - 1)))
                        Cb = model->cal_pos(_key(*tmp_it));
                }
            } else {
                for(; C == Cb && ++tmp_it != last; ++s) {
                    Cb = model->cal_pos(_key(*tmp_it));
                }
            }

            int predict_pos = C;
            if(C - Cf <= s / 2)
                C = Cf;
            else if(Cb - C <= s - s / 2)
                C = std::max<int>({Cf, C - static_cast<int>(model->min_error_threshold), Cb - s});
            else
                C -= s / 2;
            if(max_error < model->min_error_threshold) {
                int error = std::max<int>(std::abs(C - predict_pos), C + s - 1 - predict_pos);
                if(error > max_error)
                    max_error = error;
            }

            if(size - C < num) {
                C = size - num;
                for(; Cf != C; ++Cf) {
                    data[Cf].set(last_key);
                }
                data += C;
                do{
                    if(max_error < model->min_error_threshold) {
                        C = std::abs(Cf++ - model->cal_pos(_key(*first)));
                        if(C > max_error)  max_error = C;
                    }
                    *(data++) = *(first++);
                } while(first != last);
                if(max_error < model->min_error_threshold) {
                    model->min_error_threshold = 1 << ceil_log2(max_error);
                }
                return;
            }

            for(; Cf != C; ++Cf) {
                data[Cf].set(last_key);
            }
            while(first != tmp_it) {
                data[Cf++] = *first++;
            }
            last_key = data[Cf - 1].key;
            num -= s;
        }
        for(; Cf != size; ++Cf) {
            data[Cf].set(last_key);
        }
        if(max_error < model->min_error_threshold) {
            model->min_error_threshold = 1 << ceil_log2(max_error);
        }
    }
    #endif

    inline void load_data_set(size_t size, Node* first, Node* last, Single_Model *model) override {
        auto _key = [](const Node &it) constexpr {return it.key;};
        load_data_set(size, first, last, model, _key);
    }


    template<typename It>
    inline void bulk_load_gaps(It first, It last) {
        size_t n = last - first;
        It last_begin = first;
        It last_key;    //only valid when allow duplicates
        if(allow_duplicates) {
            for(last_key = last - 1; last_key != first && (last_key - 1)->first == last_key->first; --last_key);
        }

        Single_Model *last_model, *tmp_model;
        this->model_head = last_model = tmp_model = new (this->model_alloc.allocate(1)) Single_Model();
        auto _key = [](const std::pair<K, V> &it) constexpr {return it.first;};

        auto new_segment = [&]() constexpr {
            tmp_model = new (this->model_alloc.allocate(1)) Single_Model();
            //model set
            this->building_model->get_single_model(*tmp_model);
            //data set
            if(allow_duplicates)
                load_data_set(first - last_begin, last_begin, first, tmp_model, _key);
            else
                load_data_set(tmp_model->data_num, last_begin, first, tmp_model, _key);
        };

        if(allow_duplicates) {
            this->building_model->gnaw_duplicate_keys(first, last, last_key, n, _key);
        } else {
            #if USING_GMA
            this->building_model->gnaw(first, last, n, _key);
            #else
            this->building_model->gnaw_linear(first, last, n, _key);
            #endif
        }
        new_segment();
        last_begin = first;
        this->model_head = last_model = tmp_model;

        if(allow_duplicates) {
            while(!this->building_model->gnaw_duplicate_keys(first, last, last_key, n, _key)) {
                new_segment();
                last_begin = first;
                //link set
                last_model->next = tmp_model;
                tmp_model->prev = last_model;
                last_model = tmp_model;
            }
        } else {
            #if USING_GMA
            while(!this->building_model->gnaw(first, last, n, _key)) {
            #else
            while(!this->building_model->gnaw_linear(first, last, n, _key)) {
            #endif
                new_segment();
                last_begin = first;
                //link set
                last_model->next = tmp_model;
                tmp_model->prev = last_model;
                last_model = tmp_model;
            }
        }

        if(this->building_model->exist_current_model()) {
            new_segment();
            //link set
            last_model->next = tmp_model;
            tmp_model->prev = last_model;
            last_model = tmp_model;
            if(allow_duplicates) {
                if(last_begin->first == (first - 1)->first) {
                    last_model->data_num = first - last_begin;
                }
            }
        }

        //model head/tail link
        this->model_head->prev = nullptr;
        last_model->next = nullptr;
        this->model_tail = last_model;
        #if LMA_LM_LOAD
        LMA::lma_t.start();
        #endif
        //linear map build
        this->maps.build_exact(this->model_head->_min_key, this->model_tail->_max_key);
        #if LMA_LM_LOAD
        LMA::lma_t.lap();
        #endif
    }
    
    //using exponential search to find key position
    inline int search_iterator(const K& key, Single_Model &model) override {
        int pos = model.cal_pos(key);
        auto exponential_right_search = [&]() {
            int index = 1;
            int max_bound = model.data_size - pos;
            while(index < max_bound && model.key_at(pos + index) < key)
                index *= 2;
            return model.lower_bound_it(pos + index / 2, pos + std::min(index, max_bound - 1), key);
        };
        auto exponential_left_search = [&]() {
            int index = 1;
            while(index <= pos && model.key_at(pos - index) >= key) 
                index *= 2;
            return model.lower_bound_it(pos - std::min(index, pos), pos - index / 2, key);
        };
        if(model.key_at(pos) < key) {
            int max_bound = model.data_size - pos;
            pos = model.lower_bound_it(pos, pos + std::min<int>(model.min_error_threshold, max_bound - 1), key);
            if(model.key_at(pos) == key)
                return pos;
            else {
                return exponential_right_search();
            }
        } else if(model.key_at(pos) > key) {
            pos = model.lower_bound_it(pos - std::min<int>(model.min_error_threshold, pos), pos, key);
            if(model.key_at(pos) == key) {
                if(allow_duplicates) {
                    this->foremost_duplicate(pos, model);
                }
                return pos;
            }
            else {
                return exponential_left_search();
            }

        }
        //
        if(allow_duplicates) {
            this->foremost_duplicate(pos, model);
        }

        return pos;
    }

    // @brief search last valid position of duplicate key list, if no gaps in tail, pointer to next key
    // @return last vaild position
    inline int search_valid_pos(const K& key, Single_Model &model) {
        Node *node = model.data_set + this->search_iterator_upper_bound(key, model);
        while(node > model.data_set && (node - 1)->key == key && (node - 1)->is_deleted()) {
            --node;
        }
        return node - model.data_set;
    }

    inline void move_right_gap(Node *now_pos, Node *right_gap) {
        for(; right_gap != now_pos; --right_gap) {
            *right_gap = *(right_gap - 1);
        }
    }

    inline void move_left_gap(Node *now_pos, Node *left_gap) {
        for(; left_gap != now_pos; ++left_gap) {
            *left_gap = *(left_gap + 1);
        }
    }

    inline void insert_key(const K &key, const V &value, Node *insert_pos, Node *last) {
        insert_pos->set(key, value);
        for(++insert_pos; insert_pos != last && insert_pos->is_deleted(); ++insert_pos) {
            insert_pos->set(key);
        }
    }

    //begin: first pointer, end: last vaild pointer + 1
    inline Node* get_closest_gap(Node *now_node, Node *begin, Node *end) {
        begin = std::max(begin, now_node - GAP_SEEK_RANGE);
        end = std::min(end, now_node + GAP_SEEK_RANGE + 1);
        Node *front = now_node;
        if(now_node - begin > end - now_node) {
            while(now_node != end) {
                if(now_node->is_deleted()) {
                    return now_node;
                }
                if((front - 1)->is_deleted()) {
                    return front - 1;
                }
                ++now_node;
                --front;
            }
            while(front != begin) {
                if((front - 1)->is_deleted()) {
                    return front - 1;
                }
                --front;
            }
        } else {
            while(front != begin) {
                if(now_node->is_deleted()) {
                    return now_node;
                }
                if((front - 1)->is_deleted()) {
                    return front - 1;
                }
                ++now_node;
                --front;
            }
            while(now_node != end) {
                if(now_node->is_deleted()) {
                    return now_node;
                }
                ++now_node;
            }
        }
        return nullptr;
    }

    inline bool write_node2segment(Single_Model &model, const K &key, const V &value) override {
        if(model.data_num >= model.data_size * MAX_SEGMENT_FULL_RATE) {
            if(key > model._max_key) {
                this->maps.expand_last(&model, key);
                model.set_max_key(key);
            }
            return false;
        }
        int pos, predict_pos;
        if(allow_duplicates) {
            pos = search_valid_pos(key, model);
            // pos = this->search_iterator_upper_bound(key, model);
            predict_pos = (pos > 0 && model.key_at(pos - 1) == key)? 
            pos : model.cal_pos_bound(key);
        } else {
            pos = search_iterator(key, model);
            predict_pos = model.cal_pos_bound(key);
        }
        if(model.key_at(pos) > key) {
            if(predict_pos < pos && !model.key_exist(pos - 1)) {
                pos = model.last_filled_key(pos, predict_pos) + 1;
            }
            Node *now_node = model.data_set + pos;
            Node *closest_gap = get_closest_gap(now_node, model.data_set, model.data_set + model.data_capacity);
            if(closest_gap) {
                Node *last = model.data_set + model.data_size;
                if(closest_gap < now_node) {
                    --now_node;
                    move_left_gap(now_node, closest_gap);
                    insert_key(key, value, now_node, last);
                    ++model.data_num;
                    return true;
                } else {
                    if(last == closest_gap) {
                        last = closest_gap + 1;
                        ++model.data_size;
                        if(model.data_size < model.data_capacity) {
                            last->set(0);
                        }
                    }
                    move_right_gap(now_node, closest_gap);
                    insert_key(key, value, now_node, last);
                    ++model.data_num;
                    return true;
                }
            }
        } else if(model.key_at(pos) == key) {
            if(!model.key_exist(pos))
                ++model.data_num;
            else if(allow_duplicates) //only happen in pos + 1 == data_size
                return false;
            if(predict_pos < pos) {
                pos = model.last_filled_key(pos, predict_pos) + 1;
                insert_key(key, value, model.data_set + pos, model.data_set + model.data_size);
            } else {
                if(predict_pos > pos) {
                    predict_pos = model.next_filled_key(pos, predict_pos + 1) - 1;
                    model.delete_range_key(pos, predict_pos, pos? model.key_at(pos - 1) : MIN_KEY);
                    pos = predict_pos;
                }
                model.set_node(pos, key, value);
            }
            return true;
        } else {
            if(key > model._max_key) {
                //expand maps
                if(key > this->model_tail->_max_key && 
                    this->maps.forecast_layer_size(this->model_head->_min_key, this->model_tail->_max_key) > this->maps.get_max_layer_size() << this->max_map_expand_rate) {
                    model.set_max_key(key);
                    this->maps.build_exact(this->model_head->_min_key, this->model_tail->_max_key);
                } else {
                    this->maps.expand_last(&model, key);
                    model.set_max_key(key);
                }
            }
            if(predict_pos > model.data_size && model.data_capacity > model.data_size) {
                pos = predict_pos;
                model.resize(pos + 1, model.back_key());
                model.set_node(pos, key, value);
                ++model.data_num;
                return true;
            }
            pos = model.last_filled_key(pos, predict_pos) + 1;
            if(model.key_exist(pos)) {
                Node *first = model.data_set + pos;
                Node *last = first - std::min(pos, GAP_SEEK_RANGE) - 1;
                for(--first; first != last && !first->is_deleted(); --first);
                if(first != last) {
                    move_left_gap(model.data_set + pos, first);
                } else 
                    return false;
            }
            insert_key(key, value, model.data_set + pos, model.data_set + model.data_size);
            ++model.data_num;
            return true;
        }
        return false;
    }

    inline void merge_and_retrain(Buffer *buffer, Single_Model *model, bool alter_map = true) override {
        model->clear_gaps();
        if(buffer) this->buffer_merge(buffer, model);
        Single_Model *fm = this->train_seg(model, model->data_set, model->data_set + model->data_size);
        Single_Model *pm = fm->next;
        Single_Model *next_model = model->next;
        Node *base_data = model->data_set;
        model->data_set = nullptr;
        //index status maintain
        if(!fm->prev) this->model_head = fm;
        if(!next_model) this->model_tail = model->prev;
        //result chain process
        auto alter_model_map = [&](Single_Model *first, Single_Model *last) {
            if(alter_map) {
                #if LMA_LM_UPDATE
                LMA::lma_t.start();
                #endif
                if(first->_max_key > model->_max_key)
                    this->maps.erase_model(model, true);
                else
                    this->maps.erase_model(model, false);
                #if LMA_LM_UPDATE
                LMA::lma_t.lap();
                #endif
                last = last->prev;
                this->delete_model(model);
                for(; last != first->prev; last = last->prev) {
                    #if LMA_LM_UPDATE
                    LMA::lma_t.start();
                    #endif
                    this->maps.draw_model(last);
                    #if LMA_LM_UPDATE
                    LMA::lma_t.lap();
                    #endif
                } 
            } else {
                this->delete_model(model);
            }
        };
        pm = model->prev;
        if(pm != fm && pm->data_num < Threshold << 1) {
            pm = pm->prev;
            pm->data_num += pm->next->data_num;
            pm->min_error_threshold = Threshold;
            this->delete_model(pm->next);
        }
        int base_i = model->data_num;
        for(; pm != fm; pm = pm->prev) {
            load_data_set(pm->data_num, base_data + base_i - pm->data_num, base_data + base_i, pm);
            base_i -= pm->data_num;
        }
        fm->data_set = base_data;
        Node* tmp_data = new (this->alloc.allocate(fm->data_num)) Node[fm->data_num];
        std::memcpy(tmp_data, base_data, sizeof(Node) * fm->data_num);
        load_data_set(fm->data_num, tmp_data, tmp_data + fm->data_num, fm);
        this->alloc.deallocate(tmp_data, fm->data_num);
        //set gaps after data_size, if exist
        if(fm->data_size < fm->data_capacity) {
            (fm->data_set)[fm->data_size].set(0);
        }
        alter_model_map(fm, model);
    }

    inline bool check_line_front(Single_Model *model, const K &new_front_key, const V &new_front_value)  {
        int min_key_pos = model->first_min_key();
        double new_pos = -model->slope * static_cast<double>(model->_min_key - new_front_key) + model->intercept;
        auto front_clear = [](Node *now_pos, Node *rend_pos, const K &key) {
            for(Node* _pos = now_pos - 1; _pos != rend_pos; --_pos) {
                if(_pos->key < key) {
                    const K &tmp = _pos->key;
                    for(++_pos; _pos != now_pos; ++_pos) {
                        _pos->set(tmp);
                    }
                    return;
                }
            }
        };
        if(new_pos >= -double(Threshold)) { //too big different when predict pos < -Threshold
            Node *now_node = model->data_set + std::max<int>(0, std::min<int>(min_key_pos - 1, new_pos));
            Node *closest_gap = get_closest_gap(now_node, model->data_set, model->data_set + model->data_capacity);
            if(closest_gap) {
                Node *last = model->data_set + model->data_size;
                if(closest_gap < now_node) {
                    --now_node;
                    move_left_gap(now_node, closest_gap);
                    insert_key(new_front_key, new_front_value, now_node, last);
                    front_clear(now_node, model->data_set - 1, new_front_key);
                    ++model->data_num;
                } else {
                    if(last == closest_gap) {
                        last = closest_gap + 1;
                        ++model->data_size;
                        if(model->data_size < model->data_capacity) {
                            last->set(0);
                        }
                    }
                    move_right_gap(now_node, closest_gap);
                    insert_key(new_front_key, new_front_value, now_node, last);
                    front_clear(now_node, model->data_set - 1, new_front_key);
                    ++model->data_num;
                }
                model->min_error_threshold = std::max<uint32_t>(model->min_error_threshold, std::ceil(std::abs((now_node - model->data_set) - new_pos)));
                return true;
            }
        }
        return false;
    }

public:
    template<typename It>
    LM_Index_Gaps(It first, It last, float rate = 0.5, size_t max_map_layer = MAX_MAPPING_SIZE, uint32_t page_size = PAGESIZE) : LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>(max_map_layer, page_size), expansion_rate(rate)
    {
        bulk_load_gaps(first, last);
    }

    LM_Index_Gaps(std::vector<std::pair<K, V>> &data, float rate = 0.5, size_t max_map_layer = MAX_MAPPING_SIZE, uint32_t page_size = PAGESIZE) : LM_Index_Gaps(data.begin(), data.end(), rate, max_map_layer, page_size) 
    {}

    class Iterator {
    private:
        Single_Model* cur_model = nullptr;
        Node* cur_node = nullptr;
        Node* begin_node = nullptr;
        Node* end_node = nullptr;
    protected:
        void set(Single_Model* model, Node *pos_ptr) {
            cur_model = model;
            cur_node = pos_ptr;
            begin_node = cur_model->data_set;
            end_node = begin_node + cur_model->data_size;
        }

        void set(Single_Model* model, int pos) {
            cur_model = model;
            begin_node = cur_model->data_set;
            end_node = begin_node + cur_model->data_size;
            cur_node = begin_node + pos;
        }
    public:
        Iterator() {}
        Iterator(Single_Model* model, Node *pos_ptr) {set(model, pos_ptr);}
        Iterator(Single_Model* model, int pos) {set(model, pos);}


        std::pair<K,V> operator*() const {
            return *cur_node;
        }

        K& key() const {
            return cur_node->key;
        }

        V& value() const {
            return cur_node->value;
        }

        Iterator& operator=(const Iterator& other) {
            if (this != &other) {
                cur_model = other.cur_model;
                cur_node = other.cur_node;
                begin_node = other.begin_node;
                end_node = other.end_node;
            }
            return *this;
        }

        Iterator& operator++() {
            do {
                ++cur_node;
                if(cur_node == end_node) {
                    if(cur_model = cur_model->next) {
                        cur_node = begin_node = cur_model->data_set;
                        end_node = begin_node + cur_model->data_size;
                    } else {
                        return *this;
                    }
                }
            } while(cur_node->is_deleted());
            return *this;
        }

        Iterator& operator--() {
            do {
                if(cur_node == begin_node && cur_model = cur_model->prev) {
                    if(cur_model = cur_model->prev) {
                        begin_node = cur_model->data_set;
                        end_node = begin_node + cur_model->data_size;
                        cur_node = end_node - 1;
                    } else {
                        return *this;
                    }
                } else {
                    --cur_node;
                }
            } while(cur_node->is_deleted());
            return *this;
        }

        bool operator==(const Iterator& rhs) const {
            return cur_node == rhs.cur_node;
        }

        bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };

        friend class LM_Index_Gaps;
    };

    //find the iterator in gapped array, ignore buffer
    Iterator find(const K& key) {
        Single_Model *model = this->maps.map_key_it(key);
        if(model) {
            int pos = search_iterator(key, *model);
            if(model->key_at(pos) == key && model->key_exist(pos))
                return {model, pos};
        }
        return end();
    }

    using LM_Index<K, V, Threshold, Implement_Buffer, allow_duplicates>::erase;
    Iterator erase(Iterator pos) {
        if(pos.cur_node) {
            Node *node = pos.cur_node;
            if(!node->is_deleted()) {
                node->deleted();
                pos.cur_model->data_num--;
                if((node == pos.begin_node || node->key != (node-1)->key) &&
                    (node != pos.end_node - 1 && node->key == (node+1)->key)) {
                    K &_key = node->key;
                    for(++node; node != pos.end_node && node->key == _key; ++node) {
                        if(!node->is_deleted()) {
                            *(pos.cur_node) = *node;
                            node->deleted();
                            return pos;
                        }
                    }
                    pos.cur_node = node;
                    return pos;
                }
                return ++pos;
            }
        }
        //error if happen in here
        return pos;
    }

    constexpr Iterator begin() noexcept {
        return Iterator(this->model_head, 0);
    }

    constexpr Iterator end() noexcept {
        return Iterator(this->model_tail, this->model_tail->data_size);
    }

};

template <typename K, typename V, uint32_t Threshold, class Implement_Buffer, bool allow_duplicates>
const K LM_Index_Gaps<K, V, Threshold, Implement_Buffer, allow_duplicates>::MIN_KEY = std::numeric_limits<K>::min();

}   //namespace LMD