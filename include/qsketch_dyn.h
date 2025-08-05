#include <ctime>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include "MurmurHash3.h"
#include "PackedVector.hpp"

std::uniform_int_distribution<int> Uni(0, RAND_MAX);
std::chrono::high_resolution_clock::time_point qdyn_begin_update, qdyn_end_update;
std::random_device rd;

class QDyn{

    public:

        QDyn(int m, int b, int seed);
        void Update(double *data, int data_num, std::vector<PackedVector> &g, int i, bool flag_insert=true);

        uint32_t hash(int data_id, int counter_id);
        uint32_t hash2(int data_id, int counter_id);

        int r_max;
        int r_min;
        int offset;
        int sketch_size;
        int register_size;
        int* T;
        double estimated_card;
        double actual;
        int range;
        double update_time;
        double estimation_time;
        uint32_t *seed;
        int new_seed;

        PackedVector qdyn;

        double Count();
        double Actual();
        double Estimated_card();
};

QDyn::QDyn(int m, int b, int seed) : qdyn(b, m)
{
    this->range = pow(2, b);
    this->r_max = this->range - 1;
    this->r_min = 0;
    this->offset = pow(2, b-1) - 1;
    this->estimated_card = 0.0;
    this->actual = 0.0;
    this->register_size = b;
    this->sketch_size = m / 16;
    this->seed = new uint32_t[m];
    for (int i = 0; i < m; i++){
        this->seed[i] = rd();
    }
    this->new_seed = seed;

    int table_size = this->range;
    this->T = new int[table_size];
    for (int i = 0; i < table_size; i++){
        this->T[i] = 0;
    }
    this->T[0] = this->sketch_size;

    this->update_time = 0.0;
    this->estimation_time = 0.0;

}

inline uint32_t QDyn::hash(int data_id, int counter_id)
{
    uint32_t hash_result;
    std::string key = std::to_string(data_id) + "|" + std::to_string(counter_id);
    MurmurHash3_x86_32(&key, sizeof(key), this->new_seed, &hash_result);

    return hash_result;
}

inline uint32_t QDyn::hash2(int data_id, int counter_id)
{
    uint32_t hash_result;
    std::string key = std::to_string(data_id) + "|" + std::to_string(counter_id);
    MurmurHash3_x86_32(&key, sizeof(key), this->seed[0], &hash_result);

    return hash_result;
}

void QDyn::Update(double *data, int data_num, std::vector<PackedVector> &g, int i, bool flag_insert)
{
    qdyn_begin_update = std::chrono::high_resolution_clock::now();

    uint32_t counter_id = 0;
    double u = 0.0, r = 0.0;
    int32_t y = 0, origin_val = 0;
    double qR = 0.0, tmp = 1.0;

    for (int i = 0; i < data_num; i++) {

        if (data[i] == 0.0) {
            continue;
        }
        this->actual += data[i];

        uint32_t hv = QDyn::hash(i, counter_id);

        counter_id = hv % this->sketch_size;
        u = (double)hv / (double)UINT32_MAX;

        r = -log(u) / data[i];
        y = floor(-log2(r)) + this->offset;
        y = std::min(std::max(y, this->r_min), this->r_max);

        origin_val = qdyn.get(counter_id);
        if (y > origin_val){

            // update qR
            tmp = 0.0;
            for (int j = 0; j < this->range; j++){
                if(T[j]) tmp += this->T[j] * exp(-data[i] * (pow(2, this->offset-j-1))); 
            }
            qR = 1 - tmp / this->sketch_size;

            // update estimation value
            this->estimated_card += data[i] / qR;

            // update T
            if (this->T[origin_val] > 0){ // otherwise 
                this->T[origin_val] -= 1;
                this->T[y] += 1;
            }
            else{ // initialize
                this->T[y] += 1;
            }

            if (this->r_min < y && y < this->r_max) {
                qdyn.set(counter_id, y);
            } else if (y >= this->r_max) {
                qdyn.set(counter_id, this->r_max);
            } else {
                continue;
            }
        }
        // std::cout <<"data[t]: " << data[i];
        // std::cout <<" esti_card: " << this->estimated_card;
        // std::cout << " acc: " << this->actual;
        // std::cout << " AARE: " << fabs(1 - this->estimated_card / this->actual) << std::endl;
    }

    qdyn_end_update = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time1 = qdyn_end_update - qdyn_begin_update;
    this->update_time = time1.count();
    this->estimation_time = this->update_time;
}





double QDyn::Count()
{
    double AARE = (double) fabs(fabs(this->estimated_card) - this->actual) / this->actual;
    std::cout <<"-------esti_card: " << this->estimated_card  << " acc: " << this->actual;
    std::cout << " AARE: " << AARE << "-------" <<std::endl;
    return AARE;
}

double QDyn::Actual()
{
    return this->actual;
}

double QDyn::Estimated_card()
{
    return this->estimated_card;
}
