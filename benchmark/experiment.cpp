#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include "experiment.hpp"

using namespace std;

#define USING_SOSD 0

template<class DT>
void load_binary_data(const string &path, std::vector<DT>& data)
{
    std::ifstream ifs(path.c_str(), std::ios::binary);
    if(ifs.is_open()){
        size_t size;
        ifs.read(reinterpret_cast<char*>(&size), sizeof(size_t));
        data.resize(size);
        ifs.read(reinterpret_cast<char*>(data.data()), sizeof(DT) * size);
        ifs.close();
    }
}

int main(int argc, char* argv[])
{
    vector<pair<uint64_t, uint64_t>> query_data, insert_data;

    #if USING_SOSD
    //load SOSD data
    string path = "sosd data path";    
    load_binary_data(path, query_data);
    #else
    int total_size = 100000000; //100M data
    int seed = 2333;
    query_data.reserve(total_size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint64_t> distr(0, total_size * 100);
    for (int i = 0; i < total_size; i++)
    {
        query_data.push_back(make_pair(distr(gen), i));
    }
    sort(query_data.begin(), query_data.end(), [](const pair<uint64_t, uint64_t> & p1, const pair<uint64_t, uint64_t> & p2){return p1.first < p2.first;});
    query_data.erase(unique(query_data.begin(), query_data.end()), query_data.end());
    #endif

    int test_code = 1;
    if(argc > 1) {
        test_code = stoi(string(argv[1]));
    }

    switch (test_code)
    {
    case 0: //lm index
        lm_test(query_data, insert_data);
        break;
    case 1: //lm index
        lm_gaps_test(query_data, insert_data);
        break;
    case 2: //pgm
        pgm_test(query_data, insert_data);
        break;
    case 3: //swix
        swix_test(query_data, insert_data);
        break;
    case 4: //alex
        alex_test(query_data, insert_data);
        break;
    case 5: //btree
        btree_test(query_data, insert_data);
        break;
    default:
        break;
    }
    
    cout << "test over\n";
    return 0;
}


