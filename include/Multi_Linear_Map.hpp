/**
 * v2:  stable version, having root ptr and no inf map
 */
#pragma once

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <array>
#include <map>

namespace LMD {

template <typename K, class Single_Model, class Alloc>
class Updatable_Maps {
protected:
    enum MAP_UNIT_LABEL {UNKNOWN, MODEL, MAP};
    struct Unit
    {
        uint8_t type = UNKNOWN;
        void *ptr = nullptr;
        bool operator==(const Unit &ut) const {
            return (ut.type == type && ut.ptr == ptr);
        }

        bool operator!=(const Unit &ut) const {
            return (ut.type != type || ut.ptr != ptr);
        }

        inline void clear() {
            type = UNKNOWN;
            ptr = nullptr;
        }
    };
    using M_SIZE = unsigned int;
    using Unit_Alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<Unit>;
    class Model_Layer;
    using Layer_Alloc = typename std::allocator_traits<Alloc>::template rebind_alloc<Model_Layer>;

    M_SIZE max_layer_size;
    Single_Model *&_model = nullptr;
    Model_Layer _root;  //abstruct root map, the upperest map
    Model_Layer *root_ptr;  //dymatic search entry root
    std::vector<Model_Layer*> _maps_collect, _free_collect;
    Layer_Alloc _alloc;
    Model_Layer *reverse_map = nullptr;

    template<class Compare>
    inline K get_min_interval_size(Single_Model *& min_model, Compare comp) {
        if(_model == nullptr) return 0;
        K _min = _model->get_interval_size();
        min_model = _model;
        for(Single_Model *pm = _model->next; comp(pm); pm = pm->next) {
            if(pm->get_interval_size() < _min) {
                _min = pm->get_interval_size();
                min_model = pm;
            }
        }
        return _min;
    }

    //invaild when child layer was cleared in father layer
    inline M_SIZE get_upper_pos(const Model_Layer *it) {
        if(it->father == nullptr) return 0;
        Model_Layer *f_map = it->father;
        M_SIZE pos = f_map->cal_pos(it->min);
        if(f_map->type_at(pos) == MAP && f_map->ptr_at(pos) == (void*)it) {
            return pos;
        }
        return pos + 1;
    }

    //return sub-map max pos in father map, which is next cell after sub-map cell
    inline size_t get_upper_pos_right(const Model_Layer *it) {
        if(it->father == nullptr) return 0;
        Model_Layer *f_map = it->father;
        M_SIZE pos = f_map->cal_pos(it->max);
        if(f_map->type_at(pos) == MAP && f_map->ptr_at(pos) == (void*)it)
            return pos + 1;
        return pos;
    }

    inline void clear_map_in_father_map(const Model_Layer *it, const bool is_reverse = false) {
        if(it->father) {
            Model_Layer *f_map = it->father;
            M_SIZE pos;
            if(is_reverse)
                pos = f_map->rcal_pos_round(it->max);
            else
                pos = f_map->cal_pos_round(it->min);
            while(f_map->ptr_at(pos) == (void*)it) {
                f_map->at(pos++).clear();
            }
            --f_map->n_rel;
        }
    }

    //sub map must clear in father map
    inline void draw_supplement_in_father_map(Model_Layer *it, const bool is_reverse = false) {
        if(it->father) {
            Unit &u1 = it->layer_front();
            Unit &u2 = it->layer_back();
            Single_Model *tmp_model;
            Model_Layer *f_map = it->father;
            M_SIZE pos;
            if(u2.type == MODEL) {  //tail supplement
                tmp_model = (Single_Model *)u2.ptr;
                if(is_reverse? tmp_model->_min_key < it->min : tmp_model->_max_key > it->max) {
                    pos = is_reverse? f_map->rcal_pos(tmp_model->_max_key) : f_map->cal_pos(tmp_model->_min_key);
                    if(pos < f_map->real_layer_size() && f_map->ptr_at(pos) == nullptr) {
                        do {
                            f_map->at(pos++) = u2;
                        } while(pos < f_map->real_layer_size() && f_map->ptr_at(pos) == nullptr);
                        if(f_map->type_at(pos) == MAP) 
                            ++f_map->n_rel;
                    } 
                }
            }
            if(u1.type == MODEL) {  //head supplement
                tmp_model = (Single_Model *)u1.ptr;
                if(is_reverse? tmp_model->_max_key > it->max : tmp_model->_min_key < it->min) {
                    pos = is_reverse? f_map->rcal_pos(tmp_model->_min_key) : f_map->cal_pos(tmp_model->_max_key);
                    ++pos;
                    if(u2.type == MODEL && f_map->ptr_at(pos-1) == u2.ptr) --pos;
                    if(pos != 0 && f_map->ptr_at(pos-1) == nullptr) {
                        do {
                            f_map->at(--pos) = u1;
                        } while(pos != 0 && f_map->ptr_at(pos-1) == nullptr);
                        if(pos != 0 && f_map->type_at(pos-1) == MAP) 
                            ++f_map->n_rel;
                    }
                }
            }
        }
    }

    //sub map must be existed, link must be right, and cell in father mapping to sub is unit now
    inline void draw_supplement_in_sub_map(Unit &unit, Model_Layer *sub, const bool is_reverse = false) {
        if(unit.type == MODEL) {
            Single_Model *m_it = (Single_Model*) unit.ptr;
            if(is_reverse) {
                if(m_it->_max_key >= sub->max) {
                    reverse_fill_in_back(m_it->_min_key, sub, unit);
                } else if(m_it->_min_key <= sub->min) {
                    reverse_fill_in_front(m_it->_max_key, sub, unit);
                    for(m_it = m_it->next; m_it && m_it->_max_key < sub->max; m_it = m_it->next);
                    if(m_it && m_it->_min_key < sub->max) {
                        reverse_fill_in_back(m_it->_min_key, sub, {MODEL, (void*)m_it});
                    }
                }
            } else {
                if(m_it->_min_key <= sub->min) {
                    fill_in_back(m_it->_max_key, sub, unit);
                } else if(m_it->_max_key >= sub->max) {
                    fill_in_front(m_it->_min_key, sub, unit);
                    for(m_it = m_it->prev; m_it && m_it->_min_key > sub->min; m_it = m_it->prev);
                    if(m_it && m_it->_max_key > sub->min) {
                        fill_in_back(m_it->_max_key, sub, {MODEL, m_it});
                    }
                }
            }
        }
    }

    /**
     * only fill front map model, that is model->last > now_map->max
     */
    inline void fill_in_front(const K& left, Model_Layer *now_map, const Unit& unit, void *operation_ptr = nullptr) {
        M_SIZE pos;
        bool looping;
        if(left > now_map->max) return;
        do {
            looping = false;
            pos = now_map->cal_pos(left);
            Unit &_unit = now_map->at(pos);
            if(_unit.type == MAP) {
                looping = true;
                for(++pos; pos < now_map->_size && now_map->at(pos) == _unit; ++pos);
            } else if(operation_ptr) {
                Single_Model *m_it = prev_model(now_map, pos, left);
                if(m_it && now_map->cal_pos(m_it->_max_key) == pos) {
                    _unit = {MODEL, (void*)m_it};
                    ++pos;
                }
            }
            now_map->fill_n(pos, now_map->_size, unit);
            if(looping) {
                now_map = (Model_Layer *)_unit.ptr;
            }
        } while(looping);
    }

    /**
     * only fill back map model, that is model->first < now_map->min
     */
    inline void fill_in_back(const K& right, Model_Layer *now_map, const Unit& unit, void *operation_ptr = nullptr) {
        M_SIZE pos;
        bool looping;
        if(right < now_map->min) return;
        do {
            looping = false;
            pos = now_map->cal_pos(right);
            Unit &_unit = now_map->at(pos);
            if(_unit.type == MAP) {
                looping = true;
                for(; pos > 0 && now_map->at(pos-1) == _unit; --pos);
            } else if(_unit.ptr == operation_ptr) {  //weak tail fill
                _unit = unit;
            }
            now_map->fill_n(0, pos, unit);
            if(looping) {
                now_map = (Model_Layer *)_unit.ptr;
            }
        } while(looping);
    }

    //if operation ptr != nullptr, that is delete ptr
    inline void fill_in_map(const K& left, const K &right, Model_Layer *now_map, const Unit &unit, bool del_mark = false, void *operation_ptr = nullptr) {
        Model_Layer *tmp_map;
        M_SIZE _pos1, _pos2;
        _pos1 = now_map->cal_pos(left);
        _pos2 = right == now_map->max? now_map->real_layer_size() : now_map->cal_pos(right);
        while(now_map->type_at(_pos1) == MAP && now_map->ptr_at(_pos1) == now_map->ptr_at(_pos2)) {
            now_map = (Model_Layer *)now_map->ptr_at(_pos1);
            _pos1 = now_map->cal_pos(left);
            _pos2 = right == now_map->max? now_map->real_layer_size() : now_map->cal_pos(right);
        }

        if(del_mark) {
            if(now_map->n_rel == 1) {
                if(now_map->ptr_at(_pos2) == operation_ptr) ++_pos2;
                now_map->fill_n(_pos1, _pos2, unit);
                --now_map->n_rel;
                while(now_map->n_rel == 0) {
                    if(now_map != &_root) {
                        clear_map_in_father_map(now_map);
                        _free_collect.push_back(now_map);
                        now_map = now_map->father;
                    } else {
                        return;
                    }
                }
                tmp_map = _free_collect.back();

                draw_supplement_in_father_map(tmp_map);
                return;
            }
            if(_pos1 == _pos2) {
                ++_pos2;
                now_map->fill_n(_pos1, _pos2, unit);
                --now_map->n_rel;
                return;
            }
        } else if(_pos1 == _pos2) {
            return;
        }

        Unit &u1 = now_map->at(_pos1);
        Unit &u2 = now_map->at(_pos2);

        bool now_map_draw = false;

        if(u1.type == MAP) {
            tmp_map = (Model_Layer *)u1.ptr;
            fill_in_front(left, tmp_map, unit, operation_ptr);
            for(++_pos1; _pos1 < now_map->_size && now_map->ptr_at(_pos1) == u1.ptr; ++_pos1);
        } else if(operation_ptr) {  //if operation_ptr != nullptr, means del mode, and we need detect prev model tail supplement
            Single_Model *m_it = prev_model(now_map, _pos1, left);
            if(m_it && now_map->cal_pos(m_it->_max_key) == _pos1) {
                u1 = {MODEL, (void*)m_it};
                ++_pos1;
                now_map_draw = true;
            }
        }
        if(u2.type == MAP) {
            tmp_map = (Model_Layer *)u2.ptr;
            fill_in_back(right, tmp_map, unit, operation_ptr);
            for(; _pos2 > 0 && now_map->ptr_at(_pos2-1) == u2.ptr; --_pos2);
        } else if(u2.ptr == operation_ptr) {
            ++_pos2;
        }
        now_map_draw = now_map_draw || _pos1 < _pos2;

        now_map->fill_n(_pos1, _pos2, unit);
        if(now_map_draw) {
            if(unit.type == UNKNOWN)
                --now_map->n_rel;
            else
                ++now_map->n_rel;
        }
    }

    inline Single_Model* prev_model(Model_Layer *now_map, M_SIZE pos, const K &key) {
        while(pos == 0) {
            now_map = now_map->father;
            if(!now_map) return nullptr;
            pos = now_map->cal_pos(key);
        }
        --pos;
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer*)now_map->ptr_at(pos);
            pos = now_map->real_layer_size();
        }
        if(now_map->type_at(pos) == UNKNOWN) return nullptr;
        return (Single_Model*)now_map->ptr_at(pos);
    }

    inline Single_Model* prev_model(Model_Layer *now_map, M_SIZE pos) {
        while(pos == 0) {
            now_map = now_map->father;
            if(!now_map) return nullptr;
            pos = get_upper_pos(now_map);
        }
        --pos;
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer*)now_map->ptr_at(pos);
            pos = now_map->real_layer_size();
        }
        if(now_map->type_at(pos) == UNKNOWN) return nullptr;
        return (Single_Model*)now_map->ptr_at(pos);
    }

    /**
     * in reverse map, max is front
     */
    inline void reverse_fill_in_front(const K& right, Model_Layer *now_map, const Unit& unit, void *operation_ptr = nullptr) {
        M_SIZE pos;
        bool looping;
        if(right < now_map->min) return;
        do {
            looping = false;
            pos = now_map->rcal_pos(right);
            Unit &_unit = now_map->at(pos);
            if(_unit.type == MAP) {
                looping = true;
                for(++pos; pos < now_map->_size && now_map->at(pos) == _unit; ++pos);
            } else if(operation_ptr) {
                Single_Model *m_it = reverse_next_model(now_map, pos, right);
                if(m_it && now_map->cal_pos(m_it->_min_key) == pos) {
                    _unit = {MODEL, (void*)m_it};
                    ++pos;
                }
            }
            now_map->rfill_n(now_map->_size, pos, unit);
            if(looping) {
                now_map = (Model_Layer *)_unit.ptr;
            }
        } while(looping);
    }

    /**
     * in reverse map, min is back
     */
    inline void reverse_fill_in_back(const K& left, Model_Layer *now_map, const Unit& unit, void *operation_ptr = nullptr) {
        M_SIZE pos;
        bool looping;
        if(left > now_map->max) return;
        do {
            looping = false;
            pos = now_map->rcal_pos(left);
            Unit &_unit = now_map->at(pos);
            if(_unit.type == MAP) {
                looping = true;
                for(; pos > 0 && now_map->at(pos-1) == _unit; --pos);
            } else if(_unit.ptr == operation_ptr) {  //weak tail fill
                _unit = unit;
            }
            now_map->rfill_n(pos, 0, unit);
            if(looping) {
                now_map = (Model_Layer *)_unit.ptr;
            }
        } while(looping);
    }

    inline Single_Model* reverse_next_model(Model_Layer *now_map, M_SIZE pos, const K &key) {
        while(pos == 0) {
            now_map = now_map->father;
            if(!now_map) return nullptr;
            pos = now_map->rcal_pos(key);
        }
        --pos;
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer*)now_map->ptr_at(pos);
            pos = now_map->real_layer_size();
        }
        if(now_map->type_at(pos) == UNKNOWN) return nullptr;
        return (Single_Model*)now_map->ptr_at(pos);
    }

    inline void reverse_fill_in_map(const K& left, const K &right, Model_Layer *now_map, const Unit &unit, bool del_mark = false, void *operation_ptr = nullptr) {
        if(!reverse_map) return;
        Model_Layer *tmp_map;
        M_SIZE _pos1, _pos2;
        _pos1 = left == now_map->min? now_map->real_layer_size() : now_map->rcal_pos(left);
        _pos2 = now_map->rcal_pos(right);
        while(now_map->type_at(_pos1) == MAP && now_map->at(_pos1) == now_map->at(_pos2)) {
            now_map = (Model_Layer *)now_map->ptr_at(_pos1);
            _pos1 = left == now_map->min? now_map->real_layer_size() : now_map->rcal_pos(left);
            _pos2 = now_map->rcal_pos(right);
        }
        if(del_mark) {
            if(now_map->n_rel == 1) {
                if(now_map->ptr_at(_pos1) == operation_ptr) ++_pos1;
                now_map->rfill_n(_pos1, _pos2, unit);
                --now_map->n_rel;
                while(now_map->father && now_map->n_rel == 0) {
                    clear_map_in_father_map(now_map, true);
                    _free_collect.push_back(now_map);
                    now_map = now_map->father;
                }
                if(now_map == reverse_map && now_map->n_rel == 0) {
                    _free_collect.push_back(reverse_map);
                    reverse_map = nullptr;
                    return;
                }
                tmp_map = _free_collect.back();
                draw_supplement_in_father_map(tmp_map, true);
                return;
            }
            if(_pos1 == _pos2) {
                now_map->c_rfill_n(_pos1, _pos2, unit);
                --now_map->n_rel;
                return;
            }
        } else if(_pos1 == _pos2)
            return;

        bool now_map_draw = false;
        Unit &u1 = now_map->at(_pos1);
        Unit &u2 = now_map->at(_pos2);
        if(u1.type == MAP) {
            tmp_map = (Model_Layer *)u1.ptr;
            reverse_fill_in_back(left, tmp_map, unit, operation_ptr);
            for(; _pos1 > 0 && now_map->ptr_at(_pos1-1) == u1.ptr; --_pos1);
        } else if(u1.ptr == operation_ptr) {
            ++_pos1;
        }
        if(u2.type == MAP) {
            tmp_map = (Model_Layer *)u2.ptr;
            reverse_fill_in_front(right, tmp_map, unit, operation_ptr);
            for(++_pos2; _pos2 < now_map->_size && now_map->ptr_at(_pos2) == u2.ptr; ++_pos2);
        } else if(operation_ptr) {
            Single_Model *m_it = reverse_next_model(now_map, _pos2, right);
            if(m_it && now_map->cal_pos(m_it->_min_key) == _pos2) {
                u2 = {MODEL, (void*)m_it};
                ++_pos2;
                now_map_draw = true;
            }
        }
        now_map_draw = now_map_draw || _pos1 > _pos2;
        now_map->rfill_n(_pos1, _pos2, unit);
        if(now_map_draw) {
            if(unit.type == UNKNOWN) {
                --now_map->n_rel;
            } else {
                ++now_map->n_rel;
            }
        }
    }

    inline Single_Model* now_map_search(Model_Layer *now_map, M_SIZE pos) {
        while(pos < now_map->_size) {
            Unit &unit = now_map->at(pos);
            if(unit.type == MODEL) {
                return (Single_Model *)unit.ptr;
            }
            if(unit.type == MAP) {
                now_map = (Model_Layer *)unit.ptr;
                pos = 0;
                continue;
            }
            ++pos;
        }
        return nullptr;
    }

    inline Single_Model* reverse_now_map_search(Model_Layer *now_map, M_SIZE pos) {
        while(pos > 0) {
            Unit &unit = now_map->at(pos-1);
            if(unit.type == MODEL) {
                return (Single_Model *)unit.ptr;
            }
            if(unit.type == MAP) {
                now_map = (Model_Layer *)unit.ptr;
                pos = now_map->_size;
                continue;
            }
            --pos;
        }
        return nullptr;
    }

    inline Single_Model* right_neighbor_model(const K &key, Model_Layer *now_map, M_SIZE pos) {
        Single_Model *pm = now_map_search(now_map, pos);
        if(pm) return pm;
        while(now_map->father) {
            now_map = now_map->father;
            pos = now_map->cal_pos(key);
            Unit &map_unit = now_map->at(pos);
            for(++pos; pos < now_map->real_layer_size() && now_map->at(pos) == map_unit; ++pos);
            pm = now_map_search(now_map, pos);
            if(pm) return pm;
        }
        return nullptr;
    }

    inline Single_Model* reverse_right_neighbor_model(const K &key, Model_Layer *now_map, M_SIZE pos) {
        Single_Model *pm = reverse_now_map_search(now_map, pos);
        if(pm) return pm;
        while(now_map->father) {
            now_map = now_map->father;
            pos = now_map->rcal_pos(key);
            Unit &map_unit = now_map->at(pos);
            for(; pos > 0 && now_map->at(pos-1) == map_unit; --pos);
            pm = reverse_now_map_search(now_map, pos);
            if(pm) return pm;
        }
        return nullptr;
    }

    inline Model_Layer* get_free_layer() {
        Model_Layer *tmp;
        if(_free_collect.empty()) {
            tmp = new (_alloc.allocate(1)) Model_Layer();
            _maps_collect.push_back(tmp);
        } else {
            tmp = _free_collect.back();
            _free_collect.pop_back();
            tmp->_size = 0;
            tmp->n_rel = 0;
        }
        return tmp;
    }

    //get map from map collect, auto increase index
    inline Model_Layer* get_collect_layer(M_SIZE &index) {
        Model_Layer *tmp;
        if(index < _maps_collect.size()) {
            tmp = _maps_collect[index];
            tmp->_size = 0;
            tmp->n_rel = 0;
        } else {
            tmp = new (_alloc.allocate(1)) Model_Layer();
            _maps_collect.push_back(tmp);
        }
        ++index;
        return tmp;
    }
    
    inline void expand_map_right(const K &bound) {
        if(bound > _root.max) {
            _root.max = bound;
            _root.resize(_root.cal_pos(bound) + 1);
        }
    }

    inline void expand_map_left(const K &bound) {
        if(bound < _root.min) {
            if(reverse_map) {
                if(bound < reverse_map->min) {
                    reverse_map->min = bound;
                    reverse_map->resize(_root.rcal_pos(bound) + 1);
                }
            } else {
                reverse_map = get_free_layer();
                reverse_map->father = nullptr;
                reverse_map->set_range(bound, _root.max);
                reverse_map->mapping_rate = _root.mapping_rate;
                reverse_map->resize_clear(_root.rcal_pos(bound) + 1);
            }
        }
    }

    inline Model_Layer* erase_sub_map(Model_Layer *sub_map) {
        Model_Layer *father = sub_map->father;
        if(!father) return sub_map;
        M_SIZE pos = get_upper_pos(sub_map);
        for(; father->ptr_at(pos) == (void *)sub_map; ++pos) {
            father->at(pos).clear();
        }
        sub_map->clear();
        _free_collect.push_back(sub_map);
        return father;
    }

    inline bool magnify_layer(const K &left, const K &right, Model_Layer *now_map) {
        if(now_map->_size >= max_layer_size) return false;
        M_SIZE tmp = std::ceil(double(now_map->get_length()) / (right - left));   //expected_size
        if(tmp > now_map->real_layer_size() && tmp < max_layer_size) {
            M_SIZE last_pos = now_map->real_layer_size();
            M_SIZE rate = std::ceil((double)tmp / last_pos);
            now_map->resize_clear(last_pos * rate + 1);
            now_map->mapping_rate *= rate;
            Unit *first = now_map->layer, *last = now_map->layer + last_pos;
            // Unit *last2 = first + now_map->real_layer_size();
            last_pos = now_map->real_layer_size();
            Single_Model *m_it, *first_m_it = nullptr;
            M_SIZE pos1, pos2;
            Unit _unit;
            if(first->type == MODEL) {
                first_m_it = (Single_Model *) first->ptr;
                if(first_m_it->_min_key < now_map->min) {
                    for(++first; first->ptr == (void*)first_m_it; ++first);
                } else {
                    for(first_m_it = first_m_it->prev; first_m_it && first_m_it->_min_key >= now_map->min; first_m_it = first_m_it->prev);
                    if(first_m_it && first_m_it->_max_key <= now_map->min)
                        first_m_it = nullptr;
                }
            }
            --first;
            while(last != first) {
                if(last->type == MODEL) {
                    m_it = (Single_Model *) last->ptr;
                    pos1 = now_map->cal_pos(m_it->_min_key);
                    pos2 = std::min(now_map->cal_pos(m_it->_max_key), last_pos);
                    _unit = *last;
                    for((last--)->clear(); last != first && last->ptr == _unit.ptr; --last) last->clear();
                    now_map->w_fill_n(pos1, pos2, _unit);
                    last_pos = pos1;
                    //case: last cell is MAP, and have tail which be covered by m_it
                    if(last != first && last->type == MAP) {
                        if(pos1 > (pos2 = (last - now_map->layer + 1) * rate)) {
                            pos1 = pos2;
                            pos2 = now_map->cal_pos(m_it->prev->_max_key);
                            if(pos1 <= pos2) {
                                now_map->w_fill_n(pos1, pos2, {MODEL, (void*)m_it->prev});
                            }
                        }
                    }
                } else {
                    if(last->type == MAP) {
                        pos1 = (last - now_map->layer) * rate;
                        pos2 = pos1 + rate;
                        now_map->fill_n(pos1, pos2, *last);
                        last->clear();
                        last_pos = pos1;
                    }
                    --last;
                }
            }
            if(first_m_it) {
                now_map->w_fill_n(0, now_map->cal_pos(first_m_it->_max_key), {MODEL, (void*)first_m_it});
            }
            return true;
        }
        return false;
    }

    inline bool reverse_magnify_layer(const K &left, const K &right, Model_Layer *now_map) {
        if(now_map->_size >= max_layer_size) return false;
        M_SIZE expected_size = std::ceil(double(now_map->get_length()) / (right - left));
        if(expected_size > now_map->real_layer_size() && expected_size < max_layer_size) {
            M_SIZE last_pos = now_map->real_layer_size();
            M_SIZE rate = std::ceil((double)expected_size / last_pos);
            now_map->resize_clear(last_pos * rate + 1);
            now_map->mapping_rate *= rate;
            Unit *first = now_map->layer, *last = now_map->layer + last_pos;
            last_pos = now_map->real_layer_size();
            Single_Model *m_it, *first_m_it = nullptr;
            M_SIZE pos1, pos2;
            Unit _unit;
            if(first->type == MODEL) {
                first_m_it = (Single_Model *) first->ptr;
                if(first_m_it->_max_key > now_map->max) {
                    for(++first; first->ptr == (void*)first_m_it; ++first);
                } else {
                    for(first_m_it = first_m_it->next; first_m_it && first_m_it->_max_key <= now_map->max; first_m_it = first_m_it->next);
                    if(first_m_it && first_m_it->_min_key >= now_map->max)
                        first_m_it = nullptr;
                }
            }
            --first;
            while(last != first) {
                if(last->type == MODEL) {
                    m_it = (Single_Model *) last->ptr;
                    pos1 = std::min(now_map->rcal_pos(m_it->_min_key), last_pos);
                    pos2 = now_map->rcal_pos(m_it->_max_key);
                    _unit = *last;
                    for((last--)->clear(); last != first && last->ptr == _unit.ptr; --last) last->clear();
                    now_map->w_rfill_n(pos1, pos2, _unit);
                    last_pos = pos2;
                } else {
                    if(last->type == MAP) {
                        pos2 = (last - now_map->layer) * rate;
                        pos1 = pos2 + rate;
                        now_map->rfill_n(pos1, pos2, *last);
                        last->clear();
                        last_pos = pos2;
                    }
                    --last;
                }
            }
            if(first_m_it) {
                now_map->w_rfill_n(0, now_map->rcal_pos(first_m_it->_min_key), {MODEL, (void*)first_m_it});
            }
            return true;
        }
        return false;
    }

    //draw in _root map
    inline void draw(const K &left, const K &right, const Unit &unit) {
        Model_Layer *now_map = root_ptr;
        while(now_map->father && (right > now_map->max || left < now_map->min)) {
            now_map = now_map->father;
        }
        M_SIZE pos;
        while(now_map->out_resolution(left, right, pos)) {
            if(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer*) now_map->ptr_at(pos);
            } else {
                if(magnify_layer(left, right, now_map)) break;
                Model_Layer *tmp_map = get_free_layer();
                tmp_map->father = now_map;
                tmp_map->set_range(now_map->getKeybyPos(pos), now_map->getKeybyPos(pos + 1));
                tmp_map->resize_clear(std::min(max_layer_size, 
                    static_cast<M_SIZE>(std::ceil(double(tmp_map->get_length()) / (right - left)) + 1)));   //leave last cell for max key
                tmp_map->mapping_rate = tmp_map->real_layer_size() / double(tmp_map->get_length());
                draw_supplement_in_sub_map(now_map->at(pos), tmp_map);
                if(!is_short_model_in_now_map(now_map, pos))
                    ++now_map->n_rel;
                now_map->fill_n(pos, pos + 1, {MAP, (void*)tmp_map});
                now_map = tmp_map;
            }
        }
        fill_in_map(left, right, now_map, unit);
        if(now_map->n_rel > root_ptr->n_rel) {
            root_ptr = now_map;
        }
    }

    //one cell model in now map, not be supplement from father map
    inline bool is_short_model_in_now_map(Model_Layer *now_map, const M_SIZE &pos, const bool &is_reverse = false) {
        if(now_map->type_at(pos) == MODEL) {
            if(pos == 0) {
                if(now_map->ptr_at(pos+1) != now_map->ptr_at(pos)) {
                    Single_Model *m_it = (Single_Model*) now_map->ptr_at(pos);
                    if(is_reverse? m_it->_max_key <= now_map->max : m_it->_min_key >= now_map->min)
                        return true;
                }
            } else if(pos == now_map->real_layer_size()) {
                if(now_map->ptr_at(pos-1) != now_map->ptr_at(pos)) {
                    Single_Model *m_it = (Single_Model*) now_map->ptr_at(pos);
                    if(is_reverse? m_it->_min_key >= now_map->min : m_it->_max_key <= now_map->max)
                        return true;
                }
            } else if(now_map->ptr_at(pos-1) != now_map->ptr_at(pos) && now_map->ptr_at(pos+1) != now_map->ptr_at(pos)) {
                return true;
            }
        }
        return false;
    }

    inline void reverse_draw(const K &left, const K &right, const Unit &unit) {
        Model_Layer *now_map = reverse_map;
        M_SIZE pos;
        while(now_map->rout_resolution(left, right, pos)) {
            if(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer*) now_map->ptr_at(pos);
            } else {
                if(reverse_magnify_layer(left, right, now_map)) break;
                Model_Layer *tmp_map = get_free_layer();
                tmp_map->father = now_map;
                tmp_map->set_range(now_map->rgetKeybyPos(pos + 1), now_map->rgetKeybyPos(pos));
                tmp_map->resize_clear(std::min(max_layer_size, 
                    static_cast<M_SIZE>(std::ceil(double(tmp_map->get_length()) / (right - left)) + 1)));
                tmp_map->mapping_rate = tmp_map->real_layer_size() / double(tmp_map->get_length());
                draw_supplement_in_sub_map(now_map->at(pos), tmp_map, true);
                if(!is_short_model_in_now_map(now_map, pos, true))
                    ++now_map->n_rel;
                now_map->rfill_n(pos + 1, pos, {MAP, (void*)tmp_map});
                // ++now_map->n_rel;
                now_map = tmp_map;
            }
        }
        reverse_fill_in_map(left, right, now_map, unit);
    }

    Single_Model* reverse_map_key_it(const K &key) {
        Model_Layer *now_map = reverse_map;
        M_SIZE pos = std::min(now_map->rcal_pos(key), now_map->real_layer_size());
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer *)now_map->ptr_at(pos);
            pos = now_map->rcal_pos(key);
        }
        if(now_map->type_at(pos) == MODEL) { 
            Single_Model *pm = (Single_Model *)now_map->ptr_at(pos);
            if(pm->_max_key < key) {
                pm = pm->next;
            }
            if(pm->_min_key > key) {
                return nullptr;
            }
            return pm;
        }
        return nullptr;
    }

    Single_Model* reverse_lower_bound_map(const K &key) {
        Model_Layer *now_map = reverse_map;
        M_SIZE pos = std::min(now_map->rcal_pos(key), now_map->real_layer_size());
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer *)now_map->ptr_at(pos);
            pos = now_map->rcal_pos(key);
        }
        if(now_map->type_at(pos) == MODEL) { 
            Single_Model *pm = (Single_Model *)now_map->ptr_at(pos);
            if(pm->_max_key < key) {
                return pm->next;
            }
            return pm;
        } else {
            Single_Model *pm = reverse_right_neighbor_model(key, now_map, pos);
            return pm? pm : right_neighbor_model(_root.min, &_root, 0);
        }
    }
    
public:
    Updatable_Maps(Single_Model *&first_model, const M_SIZE& max_size, const Alloc &alloc) : 
        max_layer_size(max_size), _model(first_model), _alloc(alloc) {
        Model_Layer::_alloc = alloc;
        _maps_collect.reserve(1 << 16);
    }

    ~Updatable_Maps() {
        for(Model_Layer* pm : _maps_collect) {
            pm->delete_layer();
            _alloc.deallocate(pm, 1);
        }
        _root.delete_layer();
    }

    void build_exact(const K& min, const K& max) {
        if(_model == nullptr) return;
        if(_maps_collect.size()) {
            _free_collect.resize(_maps_collect.size());
            std::copy(_maps_collect.rbegin(), _maps_collect.rend(), _free_collect.begin());
        }
        K min_interval = std::ceil((double)(max - min) / (max_layer_size - 1));
        K model_length = max - min;
        M_SIZE expected_size = 0;
        for(Single_Model *pm = _model; pm->next != nullptr; pm = pm->next) {
            if(pm->get_interval_size() < model_length) {
                model_length = pm->get_interval_size();
                if(model_length < min_interval) {
                    expected_size = max_layer_size;
                    break;
                }
            }
        }
        if(!expected_size)
            expected_size = std::ceil((double)(max - min) / model_length) + 1;
        _root.clear();
        _root.resize(expected_size);
        _root.set_range(min, max);
        _root.mapping_rate = (_root.real_layer_size()) / (double)(_root.get_length());
        
        Unit current_unit = Unit{MODEL, nullptr};
        M_SIZE pos1, pos2;
        Single_Model *p_model = _model;
        Model_Layer *now_map = &_root;
        M_SIZE max_n_rel = 0;
        root_ptr = now_map;

        while(p_model->next != nullptr) {
            current_unit.ptr = (void*)p_model;
            model_length = p_model->get_interval_size();
            double mul_res = (double)max_layer_size * model_length;
            if(now_map->get_length() <= mul_res) {
                fill_in_map(p_model->_min_key, p_model->_max_key, now_map, current_unit);
                p_model = p_model->next;
                if(now_map->n_rel > max_n_rel) {
                    max_n_rel = now_map->n_rel;
                    root_ptr = now_map;
                }
                while(p_model->_max_key > now_map->max)
                    now_map = now_map->father;
                continue;
            }

            pos1 = now_map->cal_pos(p_model->_min_key);
            pos2 = now_map->cal_pos(p_model->_max_key);
            
            if(pos1 != pos2) {
                fill_in_map(p_model->_min_key, p_model->_max_key, now_map, current_unit);
                p_model = p_model->next;
                if(now_map->n_rel > max_n_rel) {
                    max_n_rel = now_map->n_rel;
                    root_ptr = now_map;
                }
                while(p_model->_max_key > now_map->max)
                    now_map = now_map->father;
                continue;
            }

            const int &label = now_map->type_at(pos1);
            K _min, _max;
            
            if(label == MAP) {
                now_map = (Model_Layer*)now_map->ptr_at(pos1);
                continue;
            } else {
                _min = now_map->getKeybyPos(pos1);
                _max = now_map->getKeybyPos(pos2 + 1);
            }

            Single_Model *head_model = nullptr;
            if(label == MODEL) {    //head supplement
                head_model = (Single_Model *)now_map->ptr_at(pos1);
            }

            Model_Layer *tmp_map = get_free_layer();
            now_map->fill_n(pos1, pos2 + 1, {MAP, (void*)tmp_map});
            ++now_map->n_rel;
            tmp_map->father = now_map;
            now_map = tmp_map;
            now_map->set_range(_min, _max);
            
            //find min model length in now_map range
            for(Single_Model *tmp_model = p_model->next; tmp_model != nullptr && tmp_model->_max_key < _max; tmp_model = tmp_model->next) {
                if(tmp_model->get_interval_size() < model_length)
                    model_length = tmp_model->get_interval_size();
            }
            expected_size = static_cast<M_SIZE>(std::min(std::ceil(double(now_map->get_length()) / model_length + 1.0), double(max_layer_size)));
            now_map->resize(expected_size);
            now_map->mapping_rate = now_map->real_layer_size() / double(now_map->get_length());

            if(head_model)
                now_map->c_fill_n(0, now_map->cal_pos(head_model->_max_key), {MODEL, (void*)head_model});
        }

        //fill the last model separately
        current_unit.ptr = (void*)p_model;
        fill_in_map(p_model->_min_key, p_model->_max_key, now_map, current_unit);
        now_map->layer_back() = current_unit;
    }

    inline Single_Model* map_key_it(const K &key) {
        #if SHOW_MAP_SEARCH_DEPTH
        ++depth_count;
        size_t _tmp_depth = 0;
        #endif
        Model_Layer *now_map = root_ptr;
        while(now_map->father && (key > now_map->max || key < now_map->min)) {
            #if SHOW_MAP_SEARCH_DEPTH
            ++all_search_length;
            ++_tmp_depth;
            #endif
            now_map = now_map->father;
        }
        M_SIZE pos = std::min(now_map->cal_pos(key), now_map->real_layer_size());
        while(now_map->type_at(pos) == MAP) {
            #if SHOW_MAP_SEARCH_DEPTH
            ++all_search_length;
            ++_tmp_depth;
            #endif
            now_map = (Model_Layer *)now_map->ptr_at(pos);
            pos = now_map->cal_pos(key);
        }
        #if SHOW_MAP_SEARCH_DEPTH
        if(_tmp_depth > max_search_depth)
            max_search_depth = _tmp_depth;
        #endif
        if(now_map->type_at(pos) == MODEL) {
            Single_Model *pm = (Single_Model *)now_map->ptr_at(pos);
            if(pm->_min_key > key) {
                pm = pm->prev;
            }
            if(pm->_max_key < key) {
                return nullptr;
            }
            return pm;
        }
        if(reverse_map && key < _root.min) return reverse_map_key_it(key);
        return nullptr;
    }

    //return first model whose max >= key
    inline Single_Model* lower_bound_map(const K &key) {
        if(key < _root.min) return reverse_map? reverse_lower_bound_map(key) : nullptr;
        Model_Layer *now_map = root_ptr;
        while(now_map->father && (key > now_map->max || key < now_map->min)) {
            now_map = now_map->father;
        }
        M_SIZE pos = std::min(now_map->cal_pos(key), now_map->real_layer_size());
        while(now_map->type_at(pos) == MAP) {
            now_map = (Model_Layer *)now_map->ptr_at(pos);
            pos = now_map->cal_pos(key);
        }
        Single_Model *pm;
        if(now_map->type_at(pos) == MODEL) {
            pm = (Single_Model *)now_map->ptr_at(pos);
            if(pm->_min_key > key) {
                pm = pm->prev;
            }
            if(pm->_max_key < key) {
                return pm->next;
            }
            return pm;
        } else {
            //when type is UNKNOWN, seek right first neighbor
            return right_neighbor_model(key, now_map, pos);
        }
    }

    //expand model right bound, which must be drawn before
    void expand_last(Single_Model *m_it, K ex_last) {
        expand_map_right(ex_last);
        K &origin_last = m_it->_max_key;
        Model_Layer *now_map, *tmp_map;
        M_SIZE pos;
        Unit _unit = {MODEL, (void*)m_it};
        bool all_model_clear = false;
        if(origin_last > _root.min) {
            now_map = root_ptr;
            while(now_map->father && (origin_last > now_map->max || origin_last < now_map->min)) {
                now_map = now_map->father;
            }
            pos = now_map->cal_pos(origin_last);
            while(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer *)now_map->ptr_at(pos);
                pos = now_map->cal_pos(origin_last);
            }
            tmp_map = now_map;
            while(ex_last > now_map->max && now_map->father) {
                if(now_map->n_rel <= 1) {
                    all_model_clear = true;
                    clear_map_in_father_map(now_map);
                    draw_supplement_in_father_map(now_map);
                    _free_collect.push_back(now_map);
                }
                now_map = now_map->father;
            }
            M_SIZE origin_n_rel = now_map->n_rel;
            if(all_model_clear)
                fill_in_map(m_it->_min_key, ex_last, now_map, _unit);
            else {
                fill_in_map(origin_last, ex_last, now_map, _unit);
            }
            pos = now_map->cal_pos(origin_last);
            if(now_map->type_at(pos) == MODEL) {
                now_map->n_rel = origin_n_rel;
            } else if(!all_model_clear) {
                --tmp_map->n_rel;
                if(now_map->n_rel > root_ptr->n_rel) {
                    root_ptr = now_map;
                }
            }
        } else {
            now_map = reverse_map;
            if(ex_last > now_map->max) {
                if(ex_last - _root.min > _root.min - m_it->_min_key)
                    fill_in_map(_root.min, ex_last, &_root, _unit);
                else
                    fill_in_back(ex_last, &_root, _unit);
                ex_last = now_map->max;
            }
            pos = now_map->rcal_pos(origin_last);
            while(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer *)now_map->ptr_at(pos);
                pos = now_map->rcal_pos(origin_last);
            }
            tmp_map = now_map;
            while(ex_last > now_map->max && now_map->father) {
                if(now_map->n_rel <= 1) {
                    all_model_clear = true;
                    clear_map_in_father_map(now_map, true);
                    draw_supplement_in_father_map(now_map, true);
                    _free_collect.push_back(now_map);
                }
                now_map = now_map->father;
            }
            M_SIZE origin_n_rel = now_map->n_rel;
            if(all_model_clear)
                reverse_fill_in_map(m_it->_min_key, ex_last, reverse_map, _unit);
            else {
                reverse_fill_in_map(origin_last, ex_last, now_map, _unit);
            } 
            pos = now_map->rcal_pos(origin_last);
            if(now_map->type_at(pos) == MODEL) {
                now_map->n_rel = origin_n_rel;
            } else if(!all_model_clear) {
                --tmp_map->n_rel;
            }
        }
    }

    void expand_first(Single_Model *m_it, K ex_first) {
        expand_map_left(ex_first);
        K &origin_first = m_it->_min_key;
        Model_Layer *now_map, *tmp_map;
        M_SIZE pos;
        Unit _unit = {MODEL, (void*)m_it};
        bool all_model_clear = false;
        if(origin_first < _root.min) {
            now_map = reverse_map;
            pos = now_map->rcal_pos(origin_first);
            while(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer *)now_map->ptr_at(pos);
                pos = now_map->rcal_pos(origin_first);
            }
            tmp_map = now_map;
            while(ex_first < now_map->min && now_map->father) {
                if(now_map->n_rel <= 1) {
                    all_model_clear = true;
                    clear_map_in_father_map(now_map, true);
                    draw_supplement_in_father_map(now_map, true);
                    _free_collect.push_back(now_map);
                }
                now_map = now_map->father;
            }
            M_SIZE origin_n_rel = now_map->n_rel;
            if(all_model_clear) {
                reverse_fill_in_map(ex_first, m_it->_max_key, now_map, _unit);
            }
            else
                reverse_fill_in_map(ex_first, origin_first, now_map, _unit);
            pos = now_map->rcal_pos(origin_first);
            if(now_map->type_at(pos) == MODEL) {
                now_map->n_rel = origin_n_rel;
            } else if(!all_model_clear) {
                --tmp_map->n_rel;
            }
        } else {
            if(ex_first < _root.min) {
                if(_root.min - ex_first > m_it->_max_key - _root.min)
                    reverse_fill_in_map(ex_first, _root.min, reverse_map, _unit);
                else
                    reverse_fill_in_back(ex_first, reverse_map, _unit);
                ex_first = _root.min;
            } 
            now_map = root_ptr;
            while(now_map->father && (origin_first > now_map->max || origin_first < now_map->min)) {
                now_map = now_map->father;
            }
            pos = now_map->cal_pos(origin_first);
            while(now_map->type_at(pos) == MAP) {
                now_map = (Model_Layer *)now_map->ptr_at(pos);
                pos = now_map->cal_pos(origin_first);
            }
            tmp_map = now_map;
            while(ex_first < now_map->min && now_map->father) {
                if(now_map->n_rel <= 1) {
                    all_model_clear = true;
                    clear_map_in_father_map(now_map);
                    draw_supplement_in_father_map(now_map);
                    _free_collect.push_back(now_map);
                }
                now_map = now_map->father;
            }
            M_SIZE origin_n_rel = now_map->n_rel;
            if(all_model_clear) {
                fill_in_map(ex_first, m_it->_max_key, now_map, _unit);
            } else
                fill_in_map(ex_first, origin_first, now_map, _unit);
            pos = now_map->cal_pos(origin_first);
            if(now_map->type_at(pos) == MODEL) {
                now_map->n_rel = origin_n_rel;
            } else if(!all_model_clear) {
                --tmp_map->n_rel;
                if(now_map->n_rel > root_ptr->n_rel) {
                    root_ptr = now_map;
                }
            }
        }
    }

    inline void erase_model(Single_Model *m_it, bool alter_map_structure = true) {
        erase_model(m_it, m_it->_min_key, m_it->_max_key, alter_map_structure);
    }

    inline void erase_model(Single_Model *m_it, const K &left, const K &right, bool alter_map_structure = true) {
        if(left >= _root.min) {
            fill_in_map(left, right, &_root, {UNKNOWN, nullptr}, alter_map_structure, m_it);
            if(!root_ptr->n_rel && root_ptr->father) {
                root_ptr = root_ptr->father;
            }
            if(!m_it->next) {
                _root.layer_back() = {UNKNOWN, nullptr};
            }
        } else if(right <= _root.min) {
            reverse_fill_in_map(left, right, reverse_map, {UNKNOWN, nullptr}, alter_map_structure, m_it);
            if(!m_it->prev) {
                reverse_map->layer_back() = {UNKNOWN, nullptr};
            }
        } else if(_root.min - left > right - _root.min) {
            reverse_fill_in_map(left, _root.min, reverse_map, {UNKNOWN, nullptr}, alter_map_structure, m_it);
            fill_in_back(right, &_root, {UNKNOWN, nullptr}, m_it);
        } else {
            reverse_fill_in_back(left, reverse_map, {UNKNOWN, nullptr}, m_it);
            fill_in_map(_root.min, right, &_root, {UNKNOWN, nullptr}, alter_map_structure, m_it);
            if(!root_ptr->n_rel && root_ptr->father) {
                root_ptr = root_ptr->father;
            }
        }
    }

    //only if m_it in reverse map, can shrink maybe create sub map, return false
    inline bool shrink_model(Single_Model *m_it, const K &sh_last, const K &origin_last) {
        Model_Layer *now_map;
        M_SIZE _pos;
        const Unit EMPTY = {UNKNOWN, nullptr};
        if(sh_last < _root.min) {
            now_map = reverse_map;
            _pos = now_map->rcal_pos(sh_last);
            while(now_map->type_at(_pos) == MAP) {
                now_map = (Model_Layer*) now_map->ptr_at(_pos);
                _pos = now_map->rcal_pos(sh_last);
            }
            M_SIZE tmp_rel = now_map->n_rel;
            if(origin_last < _root.min) {
                if(origin_last <= now_map->max) {
                    if(_pos == now_map->rcal_pos(origin_last))
                        return true;
                    else if(_pos == now_map->rcal_pos(m_it->_min_key))
                        return false;
                }
                reverse_fill_in_map(sh_last, origin_last, reverse_map, EMPTY, false, m_it);
            } else {
                reverse_fill_in_map(sh_last, _root.min, reverse_map, EMPTY, false, m_it);
                fill_in_map(_root.min, origin_last, &_root, EMPTY, false, m_it);
            }
            if(origin_last > now_map->max) {
                ++now_map->n_rel;
            } else {
                now_map->n_rel = tmp_rel;
            }
            if(now_map->type_at(_pos) == UNKNOWN) {
                now_map->at(_pos) = {MODEL, (void*)m_it};
            }
        } else {    //in forward map, model may be shrunk to one cell, don't need to build sub-map as model has priority in that cell
            now_map = root_ptr;
            while(now_map->father && (sh_last > now_map->max || sh_last < now_map->min)) {
                now_map = now_map->father;
            }
            _pos = now_map->cal_pos(sh_last);
            while(now_map->type_at(_pos) == MAP) {
                now_map = (Model_Layer*) now_map->ptr_at(_pos);
                _pos = now_map->cal_pos(sh_last);
            }
            Unit &tmp_u = now_map->at(_pos);
            if(tmp_u.ptr != m_it) return true;
            if(now_map->min <= m_it->_min_key) {    //no need to shrink only one cell
                if(now_map->cal_pos(m_it->_min_key) == _pos)
                    return false;
            }
            M_SIZE tmp_rel = now_map->n_rel;
            Model_Layer *tmp_map = now_map;
            while(tmp_map->max <= origin_last && tmp_map->father)
                tmp_map = tmp_map->father;
            fill_in_map(sh_last, origin_last, tmp_map, EMPTY, false, m_it);
            if(origin_last > now_map->max) {
                ++now_map->n_rel;
                if(now_map->n_rel > root_ptr->n_rel) {
                    root_ptr = now_map;
                }
            } else {
                now_map->n_rel = tmp_rel;
            }
            tmp_u = {MODEL, (void*)m_it};
        }
        return true;
    }

    inline void draw_model(Single_Model *m_it) {
        K &left = m_it->_min_key, &right = m_it->_max_key;
        expand_map_left(left);
        expand_map_right(right);
        Unit _unit = {MODEL, (void*)m_it};
        if(left >= _root.min) {
            if(!m_it->next) {
                fill_in_map(left, right, &_root, _unit);
                _root.layer_back() = _unit;
                return;
            }
            draw(left, right, _unit);
        } else if(right <= _root.min) {
            if(!m_it->prev) {
                reverse_fill_in_map(left, right, reverse_map, _unit);
                reverse_map->layer_back() = _unit;
                return;
            }
            reverse_draw(left, right, _unit);
        } else {
            if(_root.min - left > right - _root.min) {
                reverse_draw(left, _root.min, _unit);
                fill_in_back(right, &_root, _unit);
            } else {
                reverse_fill_in_back(left, reverse_map, _unit);
                draw(_root.min, right, _unit);
            }
        }
    }

    M_SIZE get_max_layer_size() const {return max_layer_size;}

    M_SIZE get_first_layer_size() const {return _root._size + (reverse_map? reverse_map->_size : 0);}

    //forecast futher first layer size if scale map to [min, max]
    M_SIZE forecast_layer_size(const K &min, const K &max) const {
        M_SIZE _size = 0;
        if(max == _root.max) {
            _size += _root._size;
        } else {
            _size += _root.cal_pos(max) + 1;
        }
        if(min < _root.min) {
            _size += static_cast<M_SIZE>((_root.min - min) * _root.mapping_rate);
        } else if(min > _root.min) {
            _size -= _root.cal_pos(min);
        }
        return _size;
    }

    //debug
    void debug_map_layers_info() {
        double sum_rel = 0, sum_size = 0;
        const M_SIZE max_bound = 8;
        int bound_count = 0;
        double sum_bound_rel = 0, sum_bound_size = 0;
        int max_size = 0;
        _maps_collect.push_back(&_root);
        for(auto &now_map : _maps_collect) {
            sum_rel += now_map->n_rel;
            sum_size += now_map->_size;
            if(now_map->_size > max_size)
                max_size = now_map->_size;
            if(now_map->n_rel <= max_bound) {
                ++bound_count;
                sum_bound_rel += now_map->n_rel;
                sum_bound_size += now_map->_size;
            }
        }
        int size = _maps_collect.size();
        std::cout << "layer count: " << size
            << "\navg rel: " << sum_rel / size
            << "\tavg size: " << sum_size / size
            << "\tmax size: " << max_size
            << "\nmax bound: " << max_bound
            << "\tbound count: " << bound_count
            << "\tavg bound rel: " << sum_bound_rel / bound_count
            << "\tavg bound size: " << sum_bound_size / bound_count
            << "\ninf layer save space: " << (sum_bound_size - sum_bound_rel) / sum_size
            << "\n";
        _maps_collect.pop_back();
    }

    void debug_map_layers_info(std::map<std::string, std::string> &info) {
        double sum_rel = 0, sum_size = 0;
        const M_SIZE max_bound = 8;
        int bound_count = 0;
        double sum_bound_rel = 0, sum_bound_size = 0;
        int max_size = 0;
        _maps_collect.push_back(&_root);
        for(auto &now_map : _maps_collect) {
            sum_rel += now_map->n_rel;
            sum_size += now_map->_size;
            if(now_map->_size > max_size)
                max_size = now_map->_size;
            if(now_map->n_rel <= max_bound) {
                ++bound_count;
                sum_bound_rel += now_map->n_rel;
                sum_bound_size += now_map->_size;
            }
        }
        int size = _maps_collect.size();
        info["layer_number"] = std::to_string(size);
        info["avg_nrel"] = std::to_string(sum_rel / size);
        info["avg_layer_size"] = std::to_string(sum_size / size);
        info["max_layer_size"] = std::to_string(max_size);
        _maps_collect.pop_back();
    }

    size_t debug_total_layer_size() {
        size_t sum_size = _root._size;
        for(auto &now_map : _maps_collect) {
            sum_size += now_map->_size;
        }
        return sum_size;
    }

#if SHOW_MAP_SEARCH_DEPTH
    size_t all_search_length = 0, depth_count = 0, max_search_depth = 0;
#endif
    //
};

template <typename K, class Single_Model, class Alloc>
class Updatable_Maps<K, Single_Model, Alloc>::Model_Layer {
public:
    static Unit_Alloc _alloc;
    K min, max;
    double mapping_rate;
    M_SIZE n_rel = 0;             //the number of maps or models in layer
    M_SIZE layer_capacity = 0;    //capacity size of layer
    M_SIZE _size = 0;
    Model_Layer *father = nullptr;
    Unit *layer = nullptr;

    inline void new_layer(const M_SIZE &capacity) {
        delete_layer();
        layer = new (_alloc.allocate(capacity)) Unit[capacity]();
        layer_capacity = capacity;
    }

    inline void delete_layer() {
        if(layer) {
            _alloc.deallocate(layer, layer_capacity);
            layer = nullptr;
        }
    }

    //fill cell [pos1, pos2)
    inline void fill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        std::fill_n(layer + pos1, pos2 - pos1, unit);
    }

    //fill cell [pos1, pos2], complete fill
    inline void c_fill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        std::fill_n(layer + pos1, pos2 - pos1 + 1, unit);
    }

    //weak fill_n, fill cell in [pos1, pos2), fill pos2 if layer[pos2].type == UNKNOWN
    inline void w_fill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        M_SIZE _n = layer[pos2].type == UNKNOWN? pos2 - pos1 + 1 : pos2 - pos1;
        std::fill_n(layer + pos1, _n, unit);
    }

    inline void rfill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        std::fill_n(layer + pos2, pos1 - pos2, unit);
    }

    inline void c_rfill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        std::fill_n(layer + pos2, pos1 - pos2 + 1, unit);
    }

    inline void w_rfill_n(const M_SIZE &pos1, const M_SIZE &pos2, const Unit &unit) {
        M_SIZE _n = layer[pos1].type == UNKNOWN? pos1 - pos2 + 1 : pos1 - pos2;
        std::fill_n(layer + pos2, _n, unit);
    }

    virtual inline M_SIZE cal_pos(const K &key) const {
        return static_cast<M_SIZE>((key - min) * mapping_rate);
    }

    inline M_SIZE cal_pos_round(const K &key) const {
        return static_cast<M_SIZE>(std::round((key - min) * mapping_rate));
    }

    virtual inline M_SIZE rcal_pos(const K &key) const {
        return static_cast<M_SIZE>((max - key) * mapping_rate);
    }

    inline M_SIZE rcal_pos_round(const K &key) const {
        return static_cast<M_SIZE>(std::round((max - key) * mapping_rate));
    }

    inline K getKeybyPos(const M_SIZE &pos) const {
        return static_cast<K>(pos / mapping_rate) + min;
    }

    inline K rgetKeybyPos(const M_SIZE &pos) const {
        return max - static_cast<K>(pos / mapping_rate);
    }

    inline void set_range(const K &min, const K &max) {
        this->min = min;
        this->max = max;
    }

    inline void resize(const M_SIZE &size) {
        if(layer) {
            if(layer_capacity < size) {
                // expand 2 * capacity at least
                M_SIZE new_capacity = std::max(size, layer_capacity << 1);
                Unit *pl = new (_alloc.allocate(new_capacity)) Unit[new_capacity]();
                std::memcpy(pl, layer, sizeof(Unit) * _size);
                _alloc.deallocate(layer, layer_capacity);
                layer = pl;
                layer_capacity = new_capacity;
            }
            _size = size;
        } else {
            new_layer(size);
            _size = size;
        }
    }

    inline void resize_clear(const M_SIZE &size) {
        if(layer) {
            if(layer_capacity < size) {
                M_SIZE new_capacity = std::max(size, layer_capacity << 1);
                Unit *pl = new (_alloc.allocate(new_capacity)) Unit[new_capacity]();
                std::memcpy(pl, layer, sizeof(Unit) * _size);
                _alloc.deallocate(layer, layer_capacity);
                layer = pl;
                layer_capacity = new_capacity;
            }
            if(_size < size) {
                for(M_SIZE i = _size; i < size; ++i)
                    layer[i].clear();
            }
            _size = size;
        } else {
            new_layer(size);
            _size = size;
        }
    }

    inline void clear() {
        _size = 0;
    }

    inline M_SIZE real_layer_size() const {
        return _size - 1;
    }

    inline K get_length() const {
        return max - min;
    }

    inline Unit& layer_back() {
        return layer[_size - 1];
    }

    inline Unit& layer_front() {
        return layer[0];
    }

    inline uint8_t& type_at(const M_SIZE &pos) {
        return layer[pos].type;
    }

    inline void*& ptr_at(const M_SIZE &pos) {
        return layer[pos].ptr;
    }

    inline Unit& at(const M_SIZE &pos) {
        return layer[pos];
    }

    virtual inline bool out_resolution(const K& left, const K& right, M_SIZE &pos) {
        pos = cal_pos(left);
        return pos == cal_pos(right);
    }

    virtual inline bool rout_resolution(const K& left, const K& right, M_SIZE &pos) {
        pos = rcal_pos(left);
        return pos == rcal_pos(right);
    }
};

template <typename K, class Single_Model, class Alloc>
typename Updatable_Maps<K, Single_Model, Alloc>::Unit_Alloc Updatable_Maps<K, Single_Model, Alloc>::Model_Layer::_alloc = Alloc();


}