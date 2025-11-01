//
//  skiplist.hpp
//  lsm-tree
//
//    sLSM: Skiplist-Based LSM Tree
//    Copyright © 2017 Aron Szanto. All rights reserved.
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//        You should have received a copy of the GNU General Public License
//        along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//    This file is part of LMIndex.
//    Modifications made by xxx in 2024.
//      - added class of End_Node & Begin_Node
//      - added seq_* for lightweightly get element inner
//      - other delete & modify
//
#pragma once

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include "LM_Accessory.hpp"

#include <strings.h>

using namespace std;


template<typename K, typename V, unsigned MAXLEVEL>
class SkipList_Node {
   
    
    
public:
    const K key;
    V value;
    SkipList_Node<K,V,MAXLEVEL>* _forward[MAXLEVEL+1];
    
    
    SkipList_Node(const K searchKey):key(searchKey) {
        for (int i=1; i<=MAXLEVEL; i++) {
            _forward[i] = NULL;
        }
    }
    
    SkipList_Node(const K searchKey,const V val):key(searchKey),value(val) {
        for (int i=1; i<=MAXLEVEL; i++) {
            _forward[i] = NULL;
        }
    }
    
    virtual ~SkipList_Node(){}

    virtual bool operator<(const K& _k) const {return key < _k;}
    virtual bool operator<=(const K& _k) const {return key <= _k;}
    virtual bool operator>(const K& _k) const {return key > _k;}
    virtual bool operator>=(const K& _k) const {return key >= _k;}
    virtual bool operator==(const K& _k) const {return key == _k;}
};

template<typename K, typename V, unsigned MAXLEVEL>
class End_Node : public SkipList_Node<K, V, MAXLEVEL> {
public:
    End_Node() : SkipList_Node<K, V, MAXLEVEL>(K(0)){}
    bool operator<(const K& _k) const {return false;}
    bool operator<=(const K& _k) const {return false;}
    bool operator>(const K& _k) const {return true;}
    bool operator>=(const K& _k) const {return true;}
    bool operator==(const K& _k) const {return false;}
};

template<typename K, typename V, unsigned MAXLEVEL>
class Begin_Node : public SkipList_Node<K, V, MAXLEVEL> {
public:
    Begin_Node() : SkipList_Node<K, V, MAXLEVEL>(K(0)){}
    bool operator<(const K& _k) const {return true;}
    bool operator<=(const K& _k) const {return true;}
    bool operator>(const K& _k) const {return false;}
    bool operator>=(const K& _k) const {return false;}
    bool operator==(const K& _k) const {return false;}
};


template<typename K, typename V, int MAXLEVEL = 12>
class SkipList
{
public:
    
    typedef SkipList_Node<K, V, MAXLEVEL> Node;
    typedef LMA::KVPair<K, V> PureNode;

    const int max_level;
    
    SkipList():p_listHead(NULL),p_listTail(NULL),
    cur_max_level(1),max_level(MAXLEVEL), _n(0)
    {
        p_listHead = new Begin_Node<K, V, MAXLEVEL>();
        p_listTail = new End_Node<K, V, MAXLEVEL>();
        for (int i=1; i<=MAXLEVEL; i++) {
            p_listHead->_forward[i] = p_listTail;
        }
    }
    
    ~SkipList()
    {
        Node* currNode = p_listHead->_forward[1];
        while (currNode != p_listTail) {
            Node* tempNode = currNode;
            currNode = currNode->_forward[1];
            delete tempNode;
        }
        delete p_listHead;
        delete p_listTail;
    }
    
    void insert_key(const K &key, const V &value) {
        Node* update[MAXLEVEL];
        Node* currNode = p_listHead;
        for(int level = cur_max_level; level > 0; level--) {
            while (*(currNode->_forward[level]) < key) {
                currNode = currNode->_forward[level];
            }
            update[level] = currNode;
        }
        currNode = currNode->_forward[1];
        if (*currNode == key) {
            // update the value if the key already exists
            currNode->value = value;
        }
        else {
            int insertLevel = generateNodeLevel();
            
            if (insertLevel > cur_max_level && insertLevel < MAXLEVEL - 1) {
                for (int lv = cur_max_level + 1; lv <= insertLevel; lv++) {
                    update[lv] = p_listHead;
                }
                cur_max_level = insertLevel;
            }
            currNode = new Node(key,value);
            for (int level = 1; level <= cur_max_level; level++) {
                currNode->_forward[level] = update[level]->_forward[level];
                update[level]->_forward[level] = currNode;
            }
            if(currNode->_forward[1] == p_listTail) {
                p_listTail->_forward[1] = currNode;     //set max
            }
            ++_n;
        }
        
        
    }
    
    bool delete_key(const K &searchKey) {
        Node* update[MAXLEVEL];
        Node* currNode = p_listHead;
        for(int level=cur_max_level; level >=1; level--) {
            while (*(currNode->_forward[level]) < searchKey) {
                currNode = currNode->_forward[level];
            }
            update[level] = currNode;
        }
        currNode = currNode->_forward[1];
        if (*currNode == searchKey) {
            for (int level = 1; level <= cur_max_level; level++) {
                if (update[level]->_forward[level] != currNode) {
                    break;
                }
                update[level]->_forward[level] = currNode->_forward[level];
            }
            delete currNode;
            // update the max level
            while (cur_max_level > 1 && p_listHead->_forward[cur_max_level] == NULL) {
                cur_max_level--;
            }
            _n--;
            return true;
        }
        return false;
    }
    
    V lookup(const K &searchKey, bool &found) {
        Node* currNode = p_listHead;
        for(int level=cur_max_level; level >=1; level--) {
            while (*(currNode->_forward[level]) < searchKey) {
                currNode = currNode->_forward[level];
            }
        }
        currNode = currNode->_forward[1];
        if(*currNode == searchKey) {
            found = true;
            return currNode->value;
        }
        else {
            found = false;
            return (V) NULL;
        }
    }

    template<typename Pair_It>
    void range_lookup(const K &key1, const K &key2, Pair_It &buffer) {
        Node* currNode = p_listHead;
        for(int level=cur_max_level; level >=1; level--) {
            while (*(currNode->_forward[level]) < key1) {
                currNode = currNode->_forward[level];
            }
        }
        currNode = currNode->_forward[1];
        while(*currNode <= key2) {
            (*buffer++) = PureNode(currNode->key, currNode->value);
            currNode = currNode->_forward[1];
        }
    }

    inline const K& min_key() const {
        return p_listHead->_forward[1]->key;
    }

    inline const K& max_key() const {
        return p_listTail->_forward[1]->key;
    }

    inline const V& max_key_value() const {
        return p_listTail->_forward[1]->value;
    }
    
    Node* sequence = NULL;
    void seq_reset() {
        sequence = p_listHead->_forward[1];
    }

    inline const K& seq_key_now() {
        return sequence->key;
    }

    inline V& seq_value_now() {
        return sequence->value;
    }

    inline PureNode seq_node_next() {
        PureNode pn{sequence->key, sequence->value};
        sequence = sequence->_forward[1];
        return pn;
    }

    inline bool seq_not_end() const {
        return sequence != p_listTail;
    }

    bool seq_next(PureNode &pn) {
        return seq_next(pn.key, pn.value);
    }

    inline bool seq_next(K &key, V &value) {
        if(sequence == p_listTail) return false;
        key = sequence->key;
        value = sequence->value;
        sequence = sequence->_forward[1];
        return true;
    }

    inline bool seq_prev(K &key, V &value) {
        if(sequence == p_listTail->_forward[1]) return false;
        Node* tmp = sequence;
        sequence = p_listTail->_forward[1];
        while(sequence->_forward[1] != tmp) sequence = sequence->_forward[1];
        key = sequence->key;
        value = sequence->value;
        return true;
    }

    bool seq_next_del(PureNode &pn) {
        return seq_next_del(pn.key, pn.value);
    }

    inline bool seq_next_del(K &key, V& value) {
        if(sequence == p_listTail) {
            return false;
        }
        key = sequence->key;
        value = sequence->value;
        Node *tmp = sequence;
        sequence = sequence->_forward[1];
        delete tmp;
        --_n;
        return true;
    }

    void seq_recover() {
        if(_n != 0) return;
        cur_max_level = 1;
        for(int i=1; i<=MAXLEVEL; i++) {
            p_listHead->_forward[i] = p_listTail;
        }
    }

    Node *sequence_bound = NULL;

    void seq_lower_bound(const K &searchKey) {
        Node* currNode = p_listHead;
        for(int level=cur_max_level; level >=1; level--) {
            while (*(currNode->_forward[level]) < searchKey) {
                currNode = currNode->_forward[level];
            }
        }
        sequence_bound = currNode->_forward[1];
    }

    void seq_bound(const K &searchKey) {
        seq_lower_bound(searchKey);
        if(*sequence_bound == searchKey)
            sequence_bound = sequence_bound->_forward[1];
    }

    void seq_bound_end() {
        sequence_bound = p_listTail;
    }

    inline void seq_set_in_bound() {
        sequence = sequence_bound;
    }

    inline void seq_set_full_bound() {
        sequence = p_listHead->_forward[1];
        sequence_bound = p_listTail;
    }

    inline bool seq_not_in_bound() const {
        return sequence != sequence_bound;
    }

    void get_array_del(PureNode *array) {
        Node *it = p_listHead->_forward[1], *tmp;
        while(it != p_listTail) {
            (array++)->set(it->key, it->value);
            tmp = it;
            it = it->_forward[1];
            delete tmp;
        }
        _n = 0;
        seq_recover();
    }

    inline bool empty() {
        return _n == 0;
    }

    inline void clear() {
        Node *it = p_listHead->_forward[1], *tmp;
        while(it != p_listTail) {
            tmp = it;
            it = it->_forward[1];
            delete tmp;
        }
        _n = 0;
        seq_recover();
    }
    
    
    unsigned long long num_elements() const {
        return _n;
    }
    
    size_t get_size_bytes(){
        return _n * sizeof(K);
    }
    
    //    private:
    
    int generateNodeLevel() {
        
        return ffs(rand() & ((1 << MAXLEVEL) - 1)) - 1;
    }
    
    unsigned long long _n;
    int cur_max_level;
    Node* p_listHead;
    Node* p_listTail;
    
};