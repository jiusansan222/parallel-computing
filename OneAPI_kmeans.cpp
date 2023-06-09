#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>

#include <string.h>
#include <vector>

using namespace std;

void kMeans();
void init_centroids();
using DoubleArrary4 = double[4];


// 增加32字节的前缀，用来寻找32字节对齐的地址，判断方法为 addr % 32 == 0
double* Get32bytesMemory(int len) {
  char* p = (char*)malloc(32 + len);
  memset(p, 0, 32 + len);
  for (int i = 0; i < 32; i++) {
    char* ret = p + i;
    size_t addr = (size_t)ret;
    if (addr % 32 != 0) {
      //printf("get 32bytes align addr=%#x\n", addr);
      return (double*)ret;
    }
  }
  return 0;
}

// double a[N][4]
// 总长度为 N * 4 * 8
DoubleArrary4* Get32bytesVector(int N) {
  auto p = (DoubleArrary4*)Get32bytesMemory(N * 4 * 8);
  if (p == 0) {
    exit(-10001);
  }
  return p;
}


void fill_random(DoubleArrary4* p, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4; j++) {
            p[i][j] = double(rand()) / RAND_MAX;
        }
    }
}

void fill_zero(DoubleArrary4* p, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4; j++) {
            p[i][j] = 0;
        }
    }
}



int N = 100000;
int K = 20;
int maxIter = 2000;


DoubleArrary4* kmeans_data = nullptr;
DoubleArrary4* centroids = nullptr;
DoubleArrary4* backup_centroids = nullptr;
DoubleArrary4* newCentroids = nullptr;

vector<int> labels(N);


int main() {
    cout<<"parameter setting: "<<endl;
    cout<<"The number of data points is: "<<N<<endl;
    cout<<"The number of clusters is: "<<K<<endl;
    cout<<"The iteration round number is: "<<maxIter<<endl;
    cout<<endl;
    
    auto start_init = chrono::steady_clock::now();

    kmeans_data = Get32bytesVector(N);
    fill_random(kmeans_data, N);
    backup_centroids = Get32bytesVector(K);
    centroids = Get32bytesVector(K);
    newCentroids = Get32bytesVector(K);

    for (int i = 0; i < K; ++i) {
        int x = rand() % N;
        for (int j = 0; j < 4; j++){
            backup_centroids[i][j] = kmeans_data[x][j];
        }
    }
    
    auto ends_init = chrono::steady_clock::now();
    double init_time = chrono::duration_cast<chrono::microseconds>(ends_init - start_init).count() / 1000000.0;

    // Run k-means algorithm
    init_centroids();
    auto start = chrono::steady_clock::now();
    kMeans();
    auto ends = chrono::steady_clock::now();

    double time = chrono::duration_cast<chrono::microseconds>(ends - start).count() / 1000000.0;

    cout<< "---k-means algorithm---"<<endl;
    cout << "Init time: " << init_time << " seconds" << endl;
    cout << "Execution time: " << time << " seconds" << endl;
    cout<<endl;
 
    return 0;
}


void init_centroids() {
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < 4; j++){
            centroids[i][j] = backup_centroids[i][j];
        }
    }
}

void kMeans() {

    vector<int> labels(N);
    for (int iter = 0; iter < maxIter; ++iter) {
        for (int i = 0; i < N; ++i) {
            double minDist = numeric_limits<double>::max();
            int label = -1;
            for (int j = 0; j < K; ++j) {
                double dist = 0;
                for (int k = 0; k < 4; ++k) {
                    double diff = kmeans_data[i][k] - centroids[j][k];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    label = j;
                }
            }
            labels[i] = label;
        }

        fill_zero(newCentroids, K);
        vector<int> counts(K, 0);
        for (int i = 0; i < N; ++i) {
            int label = labels[i];
            for (int k = 0; k < 4; ++k) {
                newCentroids[label][k] += kmeans_data[i][k];
            }
            ++counts[label];
        }
        for (int j = 0; j < K; ++j) {
            if (counts[j] > 0) {
                for (int k = 0; k < 4; ++k) {
                    newCentroids[j][k] /= counts[j];
                }
            } else {
                for (int k = 0; k < 4; j++){
                    newCentroids[j][k] = centroids[j][k];
                }
            }
        }

        for (int j = 0; j < K; j++){
            for(int k = 0; k < 4; k++) {
                centroids[j][k] = newCentroids[j][k];
            }
        }
    }
}



