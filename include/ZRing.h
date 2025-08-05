#include <ctime>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thread>
#include <unordered_map>

#include "MurmurHash3.h"
#include "PackedVector.hpp"



class ZRing
{
private:
    
    
    
public:

    ZRing(int sketch_size, int register_size,int g_size);

    double hash(int data_id, uint32_t seed);
    int hash_G(int data_id, int row, uint32_t seed, int G);
    void Update(double* data, int data_num, std::vector<PackedVector> &g, int round, bool flag_insert=true);
    

    
    double EstimateColumnCardinality(std::vector<PackedVector> &g);
    double EstimateColumnCardinality_IVW(std::vector<PackedVector> &g);
    double EstimateCardinality_MLE(std::vector<PackedVector> &g);
    void EstimateCard(std::vector<PackedVector> &g);


    double Count();
    double Actual();
    double Estimated_card();

    double actual;
    double estimated_card;
    std::unordered_map<double, std::pair<int, double>> item_stats;
    void UpdateItemStats(int value, bool flag_insert, double weight);
    

    int r_max;
    int r_min;
    int offset;
    int sketch_size;
    int range;
    int G_size;


    uint32_t* seed;
    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
    
    double update_time;
    double estimation_time;
    
    std::vector<PackedVector> f;
    int* Zeros;

};


ZRing::ZRing(int sketch_size, int register_size,int g_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());                
    std::uniform_int_distribution<> dist(1, 100);  
    this->gen = gen;
    this->dist = dist;
    
    this->sketch_size = sketch_size;        // the number of rows
    this->G_size = pow(2, g_size);          // range per entry

    this->range = pow(2, register_size);    // the number of columns
    this->r_max = this->range - 1;
    this->r_min = 0;
    this->offset = pow(2, register_size - 1) - 1;

    this->seed = new uint32_t[sketch_size+1];
    for (int i = 0; i < sketch_size+1; i++) {
        this->seed[i] = rd(); // 
    }

    this->Zeros = new int[this->range];
    for (int i = 0; i < this->range; i++){
        this->Zeros[i] = this->sketch_size;
    }
    
    this->update_time = 0.0;
    this->estimation_time = 0.0;
    this->estimated_card = 0.0;
    this->actual = 0.0;

    this->f.reserve(sketch_size);
    for (int i = 0; i < sketch_size; ++i) {
        this->f.emplace_back(g_size, this->range);
    }
}


inline double ZRing::hash(int data_id, uint32_t seed)
{
    uint32_t hash_result;
    std::string key = std::to_string(data_id);
    // MurmurHash3_x86_32(&key, sizeof(key), seed, &hash_result);
    MurmurHash3_x86_32(key.data(),key.size(), seed, &hash_result);

    // double hash_value = (double)hash_result / (double)UINT32_MAX;
    return hash_result;
}

inline int ZRing::hash_G(int data_id, int row, uint32_t seed, int G)
{
    uint32_t hash_result;
    // std::string key = std::to_string(data_id);
    std::string key = std::to_string(row) + "|" + std::to_string(data_id);
    // MurmurHash3_x86_32(&key, sizeof(key), seed, &hash_result);
    MurmurHash3_x86_32(key.data(),key.size(), seed, &hash_result);

    int hash_value = hash_result & (G - 1);
    return hash_value;
}


void ZRing::UpdateItemStats(int value, bool flag_insert, double weight) {
    if (flag_insert == true) {
        auto &entry = this->item_stats[value];
        entry.first += 1;        
        entry.second = weight;   
        if(entry.first == 1){
            actual += weight;
        }
        
    } else {
        auto it = this->item_stats.find(value);
        if (it != this->item_stats.end()) {
            it->second.first -= 1;      
            it->second.second = weight; 
            if (it->second.first <= 0) {
                this->item_stats.erase(it);
                actual -= weight;
            }
        }
    }


    // actual = 0.0;
    // for (const auto &kv : this->item_stats) {
    //     actual += kv.second.second;
    // }
}


void ZRing::Update(double* data, int data_num, std::vector<PackedVector> &g, int round, bool flag_insert)
{
    std::chrono::high_resolution_clock::time_point qs_begin_update, qs_end_update;

    std::vector<int> indices(data_num);
    // for (int i = 0; i < data_num; ++i) indices[i] = i;
    if(flag_insert) for (int i = 0; i < data_num; ++i) indices[i] = i;
    else for (int i = 0; i < data_num; ++i) indices[i] = data_num - i - 1;


    // if(true or flag_insert == false){
    //     std::random_device rd;
    //     std::mt19937 g_rand(rd());
    //     std::shuffle(indices.begin(), indices.end(), g_rand);
    // }
    
    qs_begin_update = std::chrono::high_resolution_clock::now();
    
    for (int t_tmp = 0; t_tmp < data_num; t_tmp++) {
        int t = indices[t_tmp];
        if (data[t] == 0.0) continue;
        int index = t + round*data_num;

        double insert = (flag_insert == true)? 1.0: -1.0;
        uint32_t re_hash = ZRing::hash(index, this->seed[0]);
        uint32_t test_hash = ZRing::hash(index, this->seed[3]+insert);
        int re = re_hash % 3 + 1; // this->dist(this->gen);
        double re_test = (double)test_hash / (double)UINT32_MAX;
        // re = 1, re_test = 0.1;
        // this->actual += data[t];
        for (int r = 0; r < re; r++) { //this->dist(this->gen)
            // if(r == 2) {insert = -1.0; flag_insert=false;}
            // else {insert = 1.0; flag_insert=true;}
            this->UpdateItemStats(t, flag_insert, data[t]);

            uint32_t hv = ZRing::hash(index, this->seed[1]);
            int row_index = hv % this->sketch_size;
            double u = (double)hv / (double)UINT32_MAX;

            // int  row_index = ZRing::hash(t, this->seed[0]) * this->sketch_size;
            // double u = ZRing::hash(t, this->seed[1]);


            int y = floor(-log2(- log(u) / data[t]));
            int column_index = std::min(std::max(y + this->offset, this->r_min), this->r_max);

            int pre = g[row_index].get(column_index);
            int add_value = ZRing::hash_G(index, r, this->seed[2], this->G_size);
            int res = (pre + (int)insert * add_value) & (this->G_size - 1);
            
            // continue;
            int flag_V = int((res == 0) - (pre == 0));
            g[row_index].set(column_index, res);
            if(flag_V == 0) continue;

            // if(flag_insert == false && res == 0){
            //     this->f[row_index].set(column_index, 0);
            // } else {
            //     this->f[row_index].set(column_index, 1);
            // }

            // double f_add = 0.0, f_sub = 1.0, f_min = 0.0, G = (double)this->G_size;
            // for (int i = 0; i < this->sketch_size; i++){
            //     for (int k = 0; k < this->range; k++){
            //         int j = k - offset;

            //         double add = 0.0;
            //         if(k==0) {add = exp(-data[t] * pow(2, -j-1));}
            //         else if (k == this->range-1) {add = 1 - exp(-data[t] * pow(2, -j));}
            //         else{add = exp(-data[t] * pow(2, -j - 1)) - exp(-data[t] * pow(2, -j));}

            //         if (g[i].get(k) == 0 ){ //  && pre==0   || (i==row_index && k==column_index)
            //             f_add += (G - 1.0) / G * add;
            //         }
            //         else{
            //             f_min +=  1 / G * add;
            //         }
            //         // if (i==row_index && k==column_index && pre!=0){   
            //         //     f_sub = std::max(add, 1e-5);
            //         // }
            //     }
            // }

            // if((flag_V == 1 && flag_insert)) this->Zeros[column_index] += flag_V;
            if((flag_V == 1 && flag_insert) || (flag_V == 1 && !flag_insert)) this->Zeros[column_index] += flag_V;
            double f_add = 0.0, G = (double)this->G_size;
            for (int k = 0; k < this->range; k++){
                int j = k - offset;

                double add = 0.0;
                if(k==0) {add = exp(-data[t] * pow(2, -j-1));}
                else if (k == this->range-1) {add = 1 - exp(-data[t] * pow(2, -j));}
                else{add = exp(-data[t] * pow(2, -j - 1)) - exp(-data[t] * pow(2, -j));}
                
                f_add += (G - 1.0) / G * add * (double)this->Zeros[k];
            }

            // std::cout << re_test << std::endl;
            if(flag_V == -1 && flag_insert)
                this->estimated_card += (data[t] * 1) / (f_add / this->sketch_size); 
            else if(flag_V == 1 && flag_insert && re_test > 0.5)
                this->estimated_card -= (data[t] * 1) / (f_add / this->sketch_size); 
            else if(flag_V == -1 && !flag_insert && re_test > 0.5)
                this->estimated_card += (data[t] * 1) / (f_add / this->sketch_size); 
            else if(flag_V == 1 && !flag_insert)
                this->estimated_card -= (data[t] * 1) / (f_add / this->sketch_size); 
                // this->estimated_card -= (data[t] * 1) / (f_add / this->sketch_size + f_min / this->sketch_size); 

            // this->estimated_card = this->estimated_card - (data[t] * flag_V) / (f_add / (double)this->sketch_size);  //  + f_min / this->sketch_size
            // this->estimated_card = std::max(this->estimated_card, 0.0);
            // if(not (flag_V == 1 && flag_insert)) this->Zeros[column_index] += flag_V;
            if(not((flag_V == 1 && flag_insert) || (flag_V == 1 && !flag_insert))) this->Zeros[column_index] += flag_V;
            // std::cout << this->Zeros[column_index] << std::endl;
            // std::cout << flag_insert <<" data[t]: " << data[t];
            // std::cout <<" esti_card: " << this->estimated_card;
            // std::cout << " acc: " << this->actual;
            // std::cout << " AARE: " << fabs(1 - this->estimated_card / this->actual) << std::endl;
            // std::cout << " time: " << this->update_time << std::endl;
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        qs_end_update = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time1 = qs_end_update - qs_begin_update;
        this->update_time = time1.count();


        // std::cout <<"data[t]: " << data[t];
        // std::cout <<" esti_card: " << this->estimated_card;
        // std::cout << " acc: " << this->actual;
        // std::cout << " AARE: " << fabs(1 - this->estimated_card / this->actual) << std::endl;
        // std::cout << " time: " << this->update_time << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(50));

        if(!flag_insert && t_tmp > (int)(data_num*0.0))    break;
    }
}


















double ZRing::Count()
{
    double AARE = (double) fabs(fabs(this->estimated_card) - this->actual) / this->actual;
    std::cout <<"-------esti_card: " << this->estimated_card  << " acc: " << this->actual;
    std::cout << " AARE: " << AARE << "-------" <<std::endl;
    return AARE;
}

double ZRing::Actual()
{
    return this->actual;
}

double ZRing::Estimated_card()
{
    return this->estimated_card;
}





























