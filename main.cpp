#include <iostream>
#include <vector>
#include <utility>
#include "./include/LM_Index.hpp"
#include "./include/LM_Index_Gaps.hpp"

using namespace std;

int main()
{
    //random data
    int total_size = 1000000;
    int seed = 2333;
    using PAIR = pair<uint64_t, uint64_t>;
    vector<PAIR> data;
    data.reserve(total_size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> distr(0, total_size * 100);
    for (int i = 0; i < total_size; i++)
    {
        data.push_back(make_pair(distr(gen), i));
    }
    sort(data.begin(), data.end(), [](const PAIR & p1, const PAIR & p2){return p1.first < p2.first;});
    data.erase(unique(data.begin(), data.end()), data.end());

    //bulk load
    LMD::LM_Index_Gaps<uint64_t, uint64_t> lmg_index(data.begin(), data.end());

    //point query
    int pi = distr(gen) % (data.size() - 1);
    auto [existed, val] = lmg_index.search(data[pi].first);
    cout << "LMIndex successfully queried point (" << data[pi].first << ", " << val << ")\n";

    //range query
    vector<PAIR> range_result;
    int range_size = 10;
    pi = distr(gen) % (data.size() - range_size);
    lmg_index.range_search(data[pi].first, data[pi + range_size - 1].first, range_result);
    cout << "LMIndex successfully queried range points:\n";
    for(auto &p : range_result) {
        cout << "(" << p.first << ", " << p.second << ")\n";
    }

    //insert
    uint64_t key = distr(gen);
    lmg_index.insert(key, 2333);
    auto insert_result = lmg_index.search(key);
    if(insert_result.first) {
        cout << "LMIndex successfully inserted point (" << key << ", " << insert_result.second << ")\n";
    }

    //delete
    pi = distr(gen) % (data.size() - 1);
    lmg_index.erase(data[pi].first);
    auto delete_result = lmg_index.search(data[pi].first);
    if(!delete_result.first) {
        cout << "LMIndex successfully deleted point (" << data[pi].first << ", " << data[pi].second << ")\n";
    }

    return 0;
}
