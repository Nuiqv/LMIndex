#pragma once

#include <list>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <fstream>
#include <utility>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <array>
#include <cstdint>
#include "LM_Accessory.hpp"

#include <ctime>

namespace LMD::linear_model
{


template <typename K, class KVPair, size_t Threshold>
class Dynamic_Linear_Model {
public:
    using Point = LMA::Point<K, size_t>;
    using Slope = LMA::Slope<K, size_t>;
    using SK = LMA::BeSigned<K>;
    using SP = LMA::BeSigned<size_t>;

    struct Block {
        size_t first_key = 0;
        size_t num_of_points = 0;
        size_t upper_start = 0, lower_start = 0;
        Point limits[4];
        std::vector<Point> upper, lower;
        int64_t offset = 0;
        K first_element;
        Block() {
            upper.reserve(BLOCK_STACK_SIZE);    
            lower.reserve(BLOCK_STACK_SIZE);    
        }

        size_t last_key() const {return num_of_points == 0? 0 : first_key + num_of_points - 1;}

        Block(const Block &b) {
            first_key = b.first_key;
            num_of_points = b.num_of_points;
            upper_start = lower_start = 0;
            memcpy(limits, b.limits, sizeof(limits));
            upper.resize(b.upper.size() - b.upper_start);
            lower.resize(b.lower.size() - b.lower_start);
            for(size_t i = b.upper_start; i < b.upper.size(); ++i) {
                upper[i - b.upper_start] = b.upper[i];
            }
            for(size_t i = b.lower_start; i < b.lower.size(); ++i) {
                lower[i - b.lower_start] = b.lower[i];
            }
            offset = b.offset;
            first_element = b.first_element;
        }
    };

protected:

    Block base_block, current_block;

    auto cross(const Point &O, const Point &A, const Point &B) const {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

    const size_t MIN_P = std::numeric_limits<size_t>::lowest();
    virtual inline void getLowerPoint(Point &low, const K &key, const size_t &pos) {
        low.x = key;
        low.y = pos <= MIN_P + Threshold? MIN_P : (pos - Threshold);
    }
    inline void getLowerPoint(Point &low, const Point& origin) {
        getLowerPoint(low, origin.x, origin.y);
    }

    const size_t MAX_P = std::numeric_limits<size_t>::max();
    virtual inline void getUpperPoint(Point &up, const K &key, const size_t &pos) { 
        up.x = key;
        up.y = pos >= MAX_P - Threshold? MAX_P : (pos + Threshold);
    }
    inline void getUpperPoint(Point &up, const Point& origin) { 
        getUpperPoint(up, origin.x, origin.y);
    }

    inline size_t getPosbyLower(const Point &p) {
        return p.y + Threshold;
    }

    inline size_t getPosbyUpper(const Point &p) {
        return p.y - Threshold;
    }

    inline void block_init(Block &block, const K &key, const size_t &pos) {
        block.first_key = pos;
        block.num_of_points = 1;
        getUpperPoint(block.limits[0], key, pos);
        getLowerPoint(block.limits[1], key, pos);
        block.upper.clear();
        block.lower.clear();
        block.upper.push_back(block.limits[0]);
        block.lower.push_back(block.limits[1]);
        block.upper_start = block.lower_start = 0;
        block.first_element = key;
    }

    inline void block_second_init(Block &block, const K &key, const size_t &pos) {
        getUpperPoint(block.limits[3], key, pos);
        getLowerPoint(block.limits[2], key, pos);
        block.upper.push_back(block.limits[3]);
        block.lower.push_back(block.limits[2]);
        block.num_of_points++;
    }

    inline bool addLowerReverse(const Point &p, Block &block, Slope &s2) {
        Slope min = block.limits[3] - p;
        if(min < s2) {
            size_t min_i = block.upper.size() - 1;
            size_t begin;
            for(begin = block.upper_start; begin < block.upper.size() && p.x > block.upper[begin].x; ++begin);
            for(size_t i = block.upper.size() - 2; i >= begin; --i) {
                Slope tmp = block.upper[i] - p;
                if(tmp > min) break;
                min = tmp;
                min_i = i;
            }
            block.limits[1] = p;
            block.limits[3] = block.upper[min_i];
            s2 = block.limits[3] - block.limits[1];
            block.upper.resize(min_i + 1);
            for(; block.lower_start < block.lower.size() - 1 && cross(block.lower[block.lower_start + 1], block.lower[block.lower_start], p) <= 0; ++block.lower_start);
            if(block.lower_start == 0) {
                size_t i = block.lower.size();
                block.lower.resize(i << 1);
                for(size_t j = block.lower_start; j < i; ++j) {
                    block.lower[(j << 1) + 1] = block.lower[j];
                }
                block.lower_start = (block.lower_start << 1) + 1;
            }
            block.lower[--block.lower_start] = p;
            return true;
        }
        return false;
    }
    
    inline bool addUpperReverse(const Point &p, Block &block, Slope &s1) {
        Slope max = block.limits[2] - p;
        if(max > s1) {
            size_t max_i = block.lower.size() - 1;
            size_t begin;
            for(begin = block.lower_start; begin < block.lower.size() && p.x > block.lower[begin].x; ++begin);
            for(size_t i = block.lower.size() - 2; i >= begin; --i) {
                Slope tmp = block.lower[i] - p;
                if(tmp < max) break;
                max = tmp;
                max_i = i;
            }
            block.limits[0] = p;
            block.limits[2] = block.lower[max_i];
            s1 = block.limits[2] - block.limits[0];
            block.lower.resize(max_i + 1);
            for(; block.upper_start < block.upper.size() - 1 && cross(block.upper[block.upper_start + 1], block.upper[block.upper_start], p) >= 0; ++block.upper_start);
            if(block.upper_start == 0) {
                size_t i = block.upper.size();
                block.upper.resize(i << 1);
                for(size_t j = block.upper_start; j < i; ++j) {
                    block.upper[(j << 1) + 1] = block.upper[j];
                }
                block.upper_start = (block.upper_start << 1) + 1;
            }
            block.upper[--block.upper_start] = p;
            return true;
        } 
        return false;
    }

    inline void _addUpper(const Point &p, Block &block, Slope &s2, Slope &min) {
        size_t min_i = block.lower_start;
        size_t end;
        for(end = block.lower.size(); end > block.lower_start && p.x < block.lower[end-1].x; --end);
        for(size_t i = block.lower_start + 1; i < end; ++i) {
            Slope tmp = p - block.lower[i];
            if(tmp > min) break;
            min = tmp;
            min_i = i;
        }
        block.limits[1] = block.lower[min_i];
        block.limits[3] = p;
        s2 = block.limits[3] - block.limits[1];
        block.lower_start = min_i;
        end = block.upper.size();
        for(; end >= block.upper_start + 2 && cross(block.upper[end - 2], block.upper[end - 1], p) <= 0; --end);
        block.upper.resize(end);
        block.upper.push_back(p);
    }

    inline void _addLower(const Point &p, Block &block, Slope &s1, Slope &max) {
        size_t max_i = block.upper_start;
        size_t end;
        for(end = block.upper.size(); end > block.upper_start && p.x < block.upper[end-1].x; --end);
        for(size_t i = block.upper_start + 1; i < end; ++i) {
            Slope tmp = p - block.upper[i];
            if(tmp < max) break;
            max = tmp;
            max_i = i;
        }
        block.limits[0] = block.upper[max_i];
        block.limits[2] = p;
        s1 =  block.limits[2] - block.limits[0];
        block.upper_start = max_i;
        end = block.lower.size();
        for(; end >= block.lower_start + 2 && cross(block.lower[end - 2], block.lower[end - 1], p) >= 0; --end);
        block.lower.resize(end);
        block.lower.push_back(p);
    }

    inline bool addUpper(const Point &p, Block &block, Slope &s2) {
        Slope min = p - block.limits[1];
        if(min < s2) {
            _addUpper(p, block, s2, min);
            return true;
        }
        return false;
    }

    inline bool addLower(const Point &p, Block &block, Slope &s1) {
        Slope max = p - block.limits[0];
        if(max > s1) {
            _addLower(p, block, s1, max);
            return true;
        }
        return false;
    }

    inline bool is_upper_limit(Block &block, const K &key, const size_t &pos) {
        Point p;
        getUpperPoint(p, key, pos);
        return p - block.limits[1] < block.limits[3] - block.limits[1];
    }

    inline bool is_lower_limit(Block &block, const K &key, const size_t &pos) {
        Point p;
        getLowerPoint(p, key, pos);
        return p - block.limits[0] > block.limits[2] - block.limits[0];
    }

    inline bool model_compatible(const Block &m1, const Block &m2) const {
        Slope s1 = m2.limits[2] - m2.limits[0];
        Slope s2 = m2.limits[3] - m2.limits[1];
        Slope _s1 = m1.limits[2] - m1.limits[0];
        Slope _s2 = m1.limits[3] - m1.limits[1];
        Slope _min, _max, _tmp;

        _max = m1.lower[m1.lower_start] - m2.limits[3];
        for(size_t i = m1.lower_start + 1; i < m1.lower.size(); ++i) {
            _tmp = m1.lower[i] - m2.limits[3];
            if(_tmp < _max) 
                break;
            _max = _tmp;
        }
        if(_max > s2) return false;

        _min = m1.upper[m1.upper_start] - m2.limits[2];
        for(size_t i = m1.upper_start + 1; i < m1.upper.size(); ++i) {
            _tmp = m1.upper[i] - m2.limits[2];
            if(_tmp > _min) 
                break;
            _min = _tmp;
        }
        if(_min < s1) return false;

        _max = m1.limits[1] - m2.upper.back();
        for(size_t i = m2.upper.size() - 2; i >= m2.upper_start; --i) {
            _tmp = m1.limits[1] - m2.upper[i];
            if(_tmp < _max) 
                break;
            _max = _tmp;
        }
        if(_max > _s2) return false;

        _min = m1.limits[0] - m2.lower.back();
        for(size_t i = m2.lower.size() - 2; i >= m2.lower_start; --i) {
            _tmp = m1.limits[0] - m2.lower[i];
            if(_tmp > _min) 
                break;
            _min = _tmp;
        }
        if(_min < _s1) return false;

        return true;
    }

    //ensure current & base can merge into one block current
    void merge_blocks(Block &base, Block &current) {
        Slope s1 = base.limits[2] - base.limits[0];
        Slope s2 = base.limits[3] - base.limits[1];
        std::vector<size_t> upper_land, lower_land;
        upper_land.reserve(current.upper.size() - current.upper_start);
        lower_land.reserve(current.lower.size() - current.lower_start);

        addLower(current.limits[1], base, s1);
        addUpper(current.limits[0], base, s2);

        auto upper_load = [&]() {
            for(int i = upper_land.size() - 1; i >= 0; --i) {
                base.upper.push_back(current.upper[upper_land[i]]);
            }
        };
        auto lower_load = [&]() {
            for(int i = lower_land.size() - 1; i >= 0; --i) {
                base.lower.push_back(current.lower[lower_land[i]]);
            }
        };
        auto reverseAddUpper = [&](const Point &p, const size_t &pos) {
            Slope tmp_s = p - base.limits[1];
            if(tmp_s <= s2) {
                _addUpper(p, base, s2, tmp_s);
                base.upper.pop_back();
                upper_land.clear();
                upper_land.push_back(pos);
                return;
            }
            if(!upper_land.empty() && cross(base.upper.back(), p, current.upper[upper_land.back()]) > 0) {
                size_t end = base.upper.size();
                for(; end >= base.upper_start + 2 && cross(base.upper[end - 2], base.upper[end - 1], p) <= 0; --end);
                upper_land.push_back(pos);
                base.upper.resize(end);
                return;
            }
        };
        auto reverseAddLower = [&](const Point &p, const size_t &pos) {
            Slope tmp_s = p - base.limits[0];
            if(tmp_s >= s1) {
                _addLower(p, base, s1, tmp_s);
                base.lower.pop_back();
                lower_land.clear();
                lower_land.push_back(pos);
                return;
            }
            if(!lower_land.empty() && cross(base.lower.back(), p, current.lower[lower_land.back()]) < 0) {
                size_t end = base.lower.size();
                for(; end >= base.lower_start + 2 && cross(base.lower[end - 2], base.lower[end - 1], p) >= 0; --end);
                lower_land.push_back(pos);
                base.lower.resize(end);
                return;
            }
        };

        size_t upper_i = current.upper.size() - 1, lower_i = current.lower.size() - 1;
        while(upper_i > current.upper_start && lower_i > current.lower_start) {
            Point &up = current.upper[upper_i];
            Point &low = current.lower[lower_i];
            if(up.x > low.x){
                reverseAddUpper(up, upper_i);
                --upper_i;
            } else {
                reverseAddLower(low, lower_i);
                --lower_i;
            }
        }
        for(; upper_i > current.upper_start; --upper_i) {
            Point &up = current.upper[upper_i];
            reverseAddUpper(up, upper_i);
        }
        for(; lower_i > current.lower_start; --lower_i) {
            Point &low = current.lower[lower_i];
            reverseAddLower(low, lower_i);
        }
        upper_load();
        lower_load();
        base.num_of_points += current.num_of_points;
    }

    template<typename KVPairIt, typename KEY>
    void merge_outlimit(Block &current, Block &base, KVPairIt &start, KEY _key) {
        size_t upper_i = base.upper_start;
        size_t lower_i = base.lower_start;

        size_t data_pos = current.last_key() + 1;
        
        Point up = base.upper[upper_i];
        Point low = base.lower[lower_i];
        Point p1, p2;

        Slope s1 = current.limits[2] - current.limits[0];
        Slope s2 = current.limits[3] - current.limits[1];

        auto linear_search = [&](KVPairIt &start, const K &end, size_t& start_i) {
            while(_key(*start) <= end) {
                getLowerPoint(p1, _key(*start), start_i);
                getUpperPoint(p2, _key(*start), start_i);
                if(p1 - current.limits[1] > s2 || p2 - current.limits[0] < s1) {
                    current.num_of_points = start_i - current.first_key;
                    return false;
                }
                addLower(p1, current, s1);
                addUpper(p2, current, s2);
                ++start; ++start_i;
            }
            current.num_of_points = start_i - current.first_key;
            return true;
        };
        auto upper_search = [&](KVPairIt &start, const K &end, size_t& start_i) {
            while(_key(*start) <= end) {
                getUpperPoint(p2, _key(*start), start_i);
                if(p2 - current.limits[0] < s1) {
                    current.num_of_points = start_i - current.first_key;
                    return false;
                }
                addUpper(p2, current, s2);
                ++start; ++start_i;
            }
            current.num_of_points = start_i - current.first_key;
            return true;
        };
        auto lower_search = [&](KVPairIt &start, const K &end, size_t& start_i) {
            while(_key(*start) <= end) {
                getLowerPoint(p1, _key(*start), start_i);
                if(p1 - current.limits[1] > s2) {
                    current.num_of_points = start_i - current.first_key;
                    return false;
                }
                addLower(p1, current, s1);
                ++start; ++start_i;
            }
            current.num_of_points = start_i - current.first_key;
            return true;
        };

        if(up.x < low.x) {
            if(!linear_search(start, up.x, data_pos)) return;
            for(++upper_i; base.upper[upper_i].x < low.x && upper_i < base.upper.size(); ++upper_i) {
                up = base.upper[upper_i];

                if(up - current.limits[0] < s1) {
                    if(!linear_search(start, up.x, data_pos)) return;
                }
                if(low - current.limits[1] > s2) {
                    if(!linear_search(start, low.x, data_pos)) return;
                }
                if(!lower_search(start, up.x, data_pos)) return;
                addUpper(up, current, s2);

            }
            if(!linear_search(start, low.x, data_pos)) return;
            lower_i++;
        } else {
            if(!linear_search(start, low.x, data_pos)) return;
            for(++lower_i; base.lower[lower_i].x < up.x && lower_i < base.lower.size(); ++lower_i) {
                low = base.lower[lower_i];
                if(low - current.limits[1] > s2) {
                    if(!linear_search(start, low.x, data_pos)) return;
                }
                if(up - current.limits[0] < s1) {
                    if(!linear_search(start, up.x, data_pos)) return;
                }
                if(!upper_search(start, low.x, data_pos)) return;
                addLower(low, current, s1);
            }
            if(!linear_search(start, up.x, data_pos)) return;
            upper_i++;
        }

        while(upper_i < base.upper.size() && lower_i < base.lower.size()) {
            up = base.upper[upper_i];
            low = base.lower[lower_i];
            if(up.x < low.x){
                if(up - current.limits[0] < s1) {
                    if(!linear_search(start, up.x, data_pos)) return;
                }
                if(low - current.limits[1] > s2) {
                    if(!linear_search(start, low.x, data_pos)) return;
                }
                addUpper(up, current, s2);
                auto next_pos = getPosbyUpper(up);
                start = std::next(start, next_pos - data_pos);
                data_pos = next_pos;
                upper_i++;
            } else {
                if(low - current.limits[1] > s2) {
                    if(!linear_search(start, low.x, data_pos)) return;
                }
                if(up - current.limits[0] < s1) {
                    if(!linear_search(start, up.x, data_pos)) return;
                }
                addLower(low, current, s1);
                auto next_pos = getPosbyLower(low);
                start = std::next(start, next_pos - data_pos);
                data_pos = next_pos;
                lower_i++;
            }
        }
        for(; upper_i < base.upper.size(); ++upper_i) {
            up = base.upper[upper_i];
            if(up - current.limits[0] < s1) {
                if(!linear_search(start, up.x, data_pos)) return;
            } else {
                if(!lower_search(start, up.x, data_pos)) return;
                addUpper(up, current, s2);
            }
        }
        for(; lower_i < base.lower.size(); ++lower_i) {
            low = base.lower[lower_i];
            if(low - current.limits[1] > s2) {
                if(!linear_search(start, low.x, data_pos)) return;
            } else {
                if(!upper_search(start, low.x, data_pos)) return;
                addLower(low, current, s1);
            }
        }
        current.num_of_points = base.last_key() - current.first_key + 1;
    }

    #define INTERCEPT_EPSILON 1e-14 //avoid rational number, which may be produce .9999999999 result
    template<class Single_model>
    inline void oeta(Single_model &building_model, const Block &block) {
        if(block.num_of_points == 0) return;
        if(block.num_of_points == 1) {
            building_model.slope = 0;
            building_model.intercept = block.first_key;
            building_model.min_error_threshold = Threshold;
            return;
        }
        //select shrink point
        Slope s1 = block.limits[2] - block.limits[0], s2 = block.limits[3] - block.limits[1];
        if(s1 == s2) {
            building_model.slope = (long double)s1;
            building_model.intercept = -SK(block.limits[0].x - block.first_element) * building_model.slope + block.limits[0].y + INTERCEPT_EPSILON;
            building_model.min_error_threshold = Threshold;
            return;
        }
        int up_edge = block.upper_start;
        Slope up_slope = up_edge == block.upper.size() - 1? s2 : block.upper[up_edge + 1] - block.upper[up_edge];
        int low_edge = block.lower.size() - 1;
        Slope low_slope = low_edge == block.lower_start? s2 : block.lower[low_edge] - block.lower[low_edge - 1];
        bool max_in_up;
        while(up_edge < block.upper.size() && low_edge >= int(block.lower_start)) {
            if(up_slope < low_slope) {
                ++up_edge;
                if(block.upper[up_edge].x >= block.lower[low_edge].x) {
                    --up_edge;
                    max_in_up = true;
                    low_slope = low_edge == block.lower.size() - 1? s1 : block.lower[low_edge + 1] - block.lower[low_edge];
                    break;
                }
                up_slope = up_edge == block.upper.size() - 1? s2 : block.upper[up_edge + 1] - block.upper[up_edge];
            } else {
                --low_edge;
                if(block.upper[up_edge].x >= block.lower[low_edge].x) {
                    ++low_edge;
                    max_in_up = false;
                    up_slope = up_edge == block.upper_start? s1 : block.upper[up_edge] - block.upper[up_edge - 1];
                    break;
                }
                low_slope = low_edge == block.lower_start? s2 : block.lower[low_edge] - block.lower[low_edge - 1];
            }
        }
        //calculate max compress length & set model value
        auto setting_model = [&](const long double &slope) {
            const Point &up_line = block.upper[up_edge], &low_line = block.lower[low_edge];
            long double t_max = 0.5L * (slope * (SK(low_line.x) - up_line.x) - (SP(low_line.y) - up_line.y)) - INTERCEPT_EPSILON;   //max compress length
            building_model.slope = slope;
            if(max_in_up)
                building_model.intercept = -SK(up_line.x - block.first_element) * slope + up_line.y - t_max;
            else
                building_model.intercept = -SK(low_line.x - block.first_element) * slope + low_line.y + t_max;
            building_model.min_error_threshold = static_cast<size_t>(std::ceil(Threshold - t_max));
        };
        //choose & prepare
        if(max_in_up) {
            while(low_slope > up_slope) {
                ++low_edge;
                low_slope = low_edge >= block.lower.size() - 1? s1 : block.lower[low_edge + 1] - block.lower[low_edge];
            }
            setting_model((long double)(up_slope));
        } else {
            while(up_slope > low_slope) {
                --up_edge;
                up_slope = up_edge <= int(block.upper_start)? s1 : block.upper[up_edge] - block.upper[up_edge - 1];
            }
            setting_model((long double)(low_slope));
        }
    }

    inline bool addPoint(Block &current, const Point &p, Slope &s1, Slope &s2) {
        Point p1, p2;
        getLowerPoint(p1, p);
        getUpperPoint(p2, p);
        if(p1 - current.limits[1] > s2 || p2 - current.limits[0] < s1) return false;
        addLower(p1, current, s1);
        addUpper(p2, current, s2);
        return true;
    }

    /**
     * gnaw block from [@start, @over], have @_n elements, build include over
     */
    template <typename KVPairIt, typename KEY>
    void gnaw_block(Block &block, const KVPairIt &start, const KVPairIt& over, KEY _key, const size_t &begin_i, size_t _n) {
        size_t end_i;
        end_i = begin_i + _n - 1;
        if(start == over) {
            block_init(block, _key(*start), begin_i);
            return;
        } else if(std::prev(over, 1) == start) {
            block_init(block, _key(*start), begin_i);
            block_second_init(block, _key(*over), end_i);
            return;
        }

        getUpperPoint(block.limits[0], _key(*start), begin_i);
        getLowerPoint(block.limits[1], _key(*start), begin_i);
        getLowerPoint(block.limits[2], _key(*over), end_i);
        getUpperPoint(block.limits[3], _key(*over), end_i);
        Slope s1 = block.limits[2] - block.limits[0];
        Slope s2 = block.limits[3] - block.limits[1];

        Point p1, p2;

        size_t d1 = Threshold << 1;
        size_t d2 = block.limits[3].y - block.limits[2].y;
        long double delta_d = d1 == d2 ? (_key(*over) - _key(*start)) / 2 + _key(*start) : d1 * (_key(*over) - _key(*start)) / (long double)(d1 + d2) + _key(*start);
        KVPairIt center_it = std::lower_bound(start, over, delta_d, [&_key](const auto& it, const long double &val){return _key(it) < val;});
        size_t center = std::distance(start, center_it) + begin_i;

        size_t part_size = center - begin_i + 1;
        block.upper.resize(part_size);
        block.lower.resize(part_size);
        block.lower_start = block.upper_start = part_size == 0 ? 0 : part_size - 1;
        block.upper[block.upper_start] = block.limits[0];
        block.lower[block.lower_start] = block.limits[1];
        block.lower.push_back(block.limits[2]);
        block.upper.push_back(block.limits[3]);
        block.first_key = begin_i;
        block.num_of_points = _n;

        bool l_inner1, l_inner2, r_inner1, r_inner2;
        l_inner1 = l_inner2 = r_inner1 = r_inner2 = true;

        auto addCenter2Left = [&](const K &key, const size_t &pos) {
            getLowerPoint(p1, key, pos);
            getUpperPoint(p2, key, pos);

            bool added = addLowerReverse(p1, block, s2);
            if(l_inner1) {
                if(!added) {
                    if(cross(block.lower[block.lower_start + 1], p1, block.lower[block.lower_start]) > 0) {
                        Point front = block.lower[block.lower_start];
                        for(++block.lower_start; block.lower_start < block.lower.size() - 1 && cross(block.lower[block.lower_start + 1], block.lower[block.lower_start], p1) <= 0; ++block.lower_start);
                        block.lower[--block.lower_start] = p1;
                        block.lower[--block.lower_start] = front;
                    }
                } else {
                    l_inner1 = false;
                }
            }
            added = addUpperReverse(p2, block, s1);
            if(l_inner2) {
                if(!added) {
                    if(cross(block.upper[block.upper_start + 1], p2, block.upper[block.upper_start]) < 0) {
                        Point front = block.upper[block.upper_start];
                        for(++block.upper_start; block.upper_start < block.upper.size() - 1 && cross(block.upper[block.upper_start + 1], block.upper[block.upper_start], p2) >= 0; ++block.upper_start);
                        block.upper[--block.upper_start] = p2;
                        block.upper[--block.upper_start] = front;
                    }
                } else {
                    l_inner2 = false;
                }
            }
        };
        auto addCenter2Right = [&](const K &key, const size_t &pos) {
            getLowerPoint(p1, key, pos);
            getUpperPoint(p2, key, pos);

            bool added = addLower(p1, block, s1);
            if(r_inner1) {
                if(!added) {
                    size_t end = block.lower.size() - 1;
                    if(cross(block.lower[end - 1], p1, block.lower[end]) < 0) {
                        for(; end >= block.lower_start + 2 && cross(block.lower[end - 2], block.lower[end - 1], p1) >= 0; --end);
                        block.lower[end] = p1;
                        block.lower.resize(end + 1);
                        block.lower.push_back(block.limits[2]);
                    }
                } else {
                    r_inner1 = false;
                }
            }
            added = addUpper(p2, block, s2);
            if(r_inner2) {
                if(!added) {
                    size_t end = block.upper.size() - 1;
                    if(cross(block.upper[end - 1], p2, block.upper[end]) > 0) {
                        for(; end >= block.upper_start + 2 && cross(block.upper[end - 2], block.upper[end - 1], p2) <= 0; --end);
                        block.upper[end] = p2;
                        block.upper.resize(end + 1);
                        block.upper.push_back(block.limits[3]);
                    }
                } else {
                    r_inner2 = false;
                }
            }
        };

        auto right_it = center_it;
        auto left_it = --center_it;
        size_t ri = center, li = ri - 1;
        while(left_it != start && right_it != over) {
            addCenter2Left(_key(*left_it--), li--);
            addCenter2Right(_key(*right_it++), ri++);
        }
        while(left_it != start) {
            addCenter2Left(_key(*left_it--), li--);
        }
        while(right_it != over) {
            addCenter2Right(_key(*right_it++), ri++);
        }
    }

    virtual inline int64_t get_offset() const {
        return -static_cast<int64_t>(Threshold);
    }

public:

    Dynamic_Linear_Model() = default;

    /**
     * gnaw data from start to over (building model), you should contain in every @param pace range data, no exist out limit
     * @param n distance from [start, over)
     * building not include over
     * @return: true: building all range data; false: meeting ultra-limit point, current segment building over, start is next begin
     */
    template <typename KVPairIt, typename KEY>
    bool gnaw(KVPairIt &start, const KVPairIt& over, size_t &n, KEY _key, size_t pace = Threshold << 1) {
        if(start == over) return true;
        size_t s1_pace = pace - 1;
        auto begin = start, end = over;
        size_t counter = 0;
        if(base_block.num_of_points == 0) {
            base_block.first_key = Threshold;
            base_block.first_element = _key(*start);
            base_block.offset = get_offset();
            if(pace <= n) {
                end = std::next(begin, s1_pace);
                gnaw_block(base_block, begin, end, _key, base_block.first_key, pace);
                base_block.num_of_points = pace;
                counter += pace;
                begin = ++end;
            } else {
                --end;
                gnaw_block(base_block, begin, end, _key, base_block.first_key, n);
                base_block.num_of_points = n;
                counter += n;
            }
        }
        
        auto gnaw_process = [&](KVPairIt &begin, const KVPairIt &end, const size_t &_n) {
            auto next_end = std::next(end, 1);
            if(_n < 2 || !is_upper_limit(base_block, _key(*end), base_block.last_key() + _n) || !is_lower_limit(base_block, _key(*end), base_block.last_key() + _n)) {
                size_t begin_pos = base_block.last_key() + 1;
                Slope s1 = base_block.limits[2] - base_block.limits[0];
                Slope s2 = base_block.limits[3] - base_block.limits[1];
                while(begin != next_end) {
                    if(!addPoint(base_block, {_key(*begin), begin_pos++}, s1, s2)) {
                        base_block.num_of_points = (--begin_pos) - base_block.first_key;
                        return false;
                    } 
                    ++begin;
                }
                base_block.num_of_points += _n;
            } else {
                gnaw_block(current_block, begin, end, _key, base_block.last_key() + 1, _n);
                if(model_compatible(current_block, base_block)) {
                    merge_blocks(base_block, current_block);
                } else {
                    size_t begin_pos = base_block.last_key() + 1;
                    Slope s1 = base_block.limits[2] - base_block.limits[0];
                    Slope s2 = base_block.limits[3] - base_block.limits[1];
                    while(begin != next_end) {
                        if(!addPoint(base_block, {_key(*begin), begin_pos++}, s1, s2)) {
                            base_block.num_of_points = (--begin_pos) - base_block.first_key;
                            return false;
                        } 
                        ++begin;
                    }
                    return true;
                }
                begin = next_end;
            }
            return true;
        };
        while(counter + pace < n) {
            end = std::next(begin, s1_pace);
            if(!gnaw_process(begin, end, pace)) {
                start = begin;
                n -= base_block.num_of_points;
                return false;
            }
            counter += pace;
        }
        if(counter < n) {
            end = std::prev(over, 1);
            if(!gnaw_process(begin, end, n - counter)) {
                start = begin;
                n -= base_block.num_of_points;
                return false;
            }
        }
        start = over;
        n = 0;
        return true;
    }

    template <typename KVPairIt>
    inline bool gnaw(KVPairIt &start, const KVPairIt& over, size_t &n, size_t pace = Threshold << 1) {
        auto _key = [](const typename KVPairIt::value_type &it) constexpr {return it.key;};
        return gnaw(start, over, n, _key, pace);
    }

    template <typename KVPairIt, typename KEY>
    bool gnaw_linear(KVPairIt &start, const KVPairIt& over, size_t &n, KEY _key) {
        auto begin = start;
        size_t last_i = Threshold;
        if(begin != over) {
            block_init(base_block, _key(*begin++), last_i++);
            base_block.offset = get_offset();
        }
        if(begin != over) {
            block_second_init(base_block, _key(*begin++), last_i++);
        }
        Slope s1 = base_block.limits[2] - base_block.limits[0];
        Slope s2 = base_block.limits[3] - base_block.limits[1];
        while(begin != over) {
            if(!addPoint(base_block, {_key(*begin), last_i}, s1, s2)) {
                base_block.num_of_points = last_i - base_block.first_key;
                start = begin;
                n -= base_block.num_of_points;
                return false;
            }
            ++begin;
            ++last_i;
        }
        base_block.num_of_points = n;
        start = over;
        n = 0;
        return true;
    }

    template <typename KVPairIt>
    inline bool gnaw_linear(KVPairIt &start, const KVPairIt& over, size_t &n) {
        auto _key = [](const typename KVPairIt::value_type &it) constexpr {return it.key;};
        return gnaw_linear(start, over, n, _key);
    }

    //using linear gnaw, if first last key in pos k, @param last_key point to k
    template <typename KVPairIt, typename KEY>
    inline bool gnaw_duplicate_keys(KVPairIt &start, const KVPairIt& over, const KVPairIt &last_key, size_t &n, KEY _key) {
        if(start == over) return true;
        auto begin = start;
        size_t last_i = Threshold;
        // size_t unique_num = 2;
        auto true_prepare = [&]() constexpr {
            base_block.num_of_points = n;
            start = over;
            n = 0;
        };
        if(begin != last_key) {
            block_init(base_block, _key(*begin++), last_i++);
            for(; _key(*begin) == _key(*(begin-1)); ++begin, ++last_i);
            base_block.offset = get_offset();
        } else {
            block_init(base_block, _key(*begin), last_i);
            n = 1;  //don't count duplicates when only one point is gnaw
            true_prepare();
            return true;
        }
        if(begin != last_key) {
            block_second_init(base_block, _key(*begin++), last_i++);
            for(; _key(*begin) == _key(*(begin-1)); ++begin, ++last_i);
        } else {
            block_second_init(base_block, _key(*begin), last_i);
            true_prepare();
            return true;
        }
        Slope s1 = base_block.limits[2] - base_block.limits[0];
        Slope s2 = base_block.limits[3] - base_block.limits[1];
        while(begin != last_key) {
            if(!addPoint(base_block, {_key(*begin), last_i}, s1, s2)) {
                start = begin;
                base_block.num_of_points = last_i - base_block.first_key;
                n -= base_block.num_of_points;
                return false;
            }
            // ++unique_num;
            for(++begin, ++last_i; _key(*begin) == _key(*(begin-1)); ++begin, ++last_i);
        }
        if(!addPoint(base_block, {_key(*begin), last_i}, s1, s2)) {
            start = begin;
            base_block.num_of_points = last_i - base_block.first_key;
            n -= base_block.num_of_points;
            return false;
        }
        true_prepare();
        return true;
    }

    template <typename KVPairIt>
    inline bool gnaw_duplicate_keys(KVPairIt &start, const KVPairIt& over, size_t &n) {
        auto _key = [](const typename KVPairIt::value_type &it) constexpr {return it.key;};
        return gnaw_duplicate_keys(start, over, n, _key);
    }

    inline bool exist_current_model() const {
        return base_block.num_of_points > 0;
    }

    inline size_t get_num_of_points() const {
        return base_block.num_of_points;
    }

    inline void clear_all() {
        base_block.num_of_points = 0;
        current_block.num_of_points = 0;
    }

    template<class Single_model>
    bool get_single_model(Single_model& s_model) {
        if(base_block.num_of_points == 0) return false;
        s_model.data_num = base_block.num_of_points;
        #if LMA_TEST && USING_OETA
        LMA::lma_t.start();
        #endif
        oeta(s_model, base_block);
        #if LMA_TEST && USING_OETA
        LMA::lma_t.lap();
        #endif
        s_model.intercept += base_block.offset;
        clear_all();
        return true;
    }

    //check if that line can accept by base block, can be called after get_single_model()
    virtual inline bool check_line(const double &slope, double &intercept, const double &min_key, uint32_t &error_threshold, const size_t &front_size) {
        double _intercept = intercept + front_size, tmp_pos;
        uint32_t _error = error_threshold, tmp_error;
        if(front_size > 1) {
            auto s1 = base_block.limits[2] - base_block.limits[0],
            s2 = base_block.limits[3] - base_block.limits[1];
            if((long double)s1 > slope || (long double)s2 < slope)
                return false;
            _intercept += Threshold << 1;
            for(auto i = base_block.upper_start; i < base_block.upper.size(); ++i) {
                tmp_error = std::ceil(std::abs(slope * (base_block.upper[i].x - min_key) + _intercept - base_block.upper[i].y));
                if(tmp_error > Threshold)
                    return false;
                if(tmp_error > _error)
                    _error = tmp_error;
            }
            _intercept -= Threshold << 1;
            for(auto i = base_block.lower_start; i < base_block.lower.size(); ++i) {
                tmp_error = std::ceil(std::abs(slope * (base_block.lower[i].x - min_key) + _intercept - base_block.lower[i].y));
                if(tmp_error > Threshold)
                    return false;
                if(tmp_error > _error)
                    _error = tmp_error;
            }
        } else {
            tmp_pos = slope * (base_block.upper[base_block.upper_start].x - min_key) + _intercept;
            if(tmp_pos < double(-Threshold) || tmp_pos > Threshold)
                return false;
            _error = std::max<uint32_t>(_error, std::abs(tmp_pos));
        }
        intercept = _intercept;
        error_threshold = _error;
        return true;
    }

};

}