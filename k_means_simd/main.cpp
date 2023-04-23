#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>
#include <immintrin.h>

#include <string.h>
#include <vector>

using namespace std;

void kMeans();
void kMeans_avx();
void kMeans_avx_align();
void init_centroids();
double calculate_euclidean_distance(__m256d a, __m256d b);
double getAccuracy();

// 增加32字节的前缀，用来寻找32字节对齐的地址，判断方法为 addr % 32 == 0
double* Get32bytesAlignMemory(int len) {
  char* p = (char*)malloc(32 + len);
  memset(p, 0, 32 + len);
  for (int i = 0; i < 32; i++) {
    char* ret = p + i;
    size_t addr = (size_t)ret;
    if (addr % 32 == 0) {
      //printf("get 32bytes align addr=%#x\n", addr);
      return (double*)ret;
    }
  }
  return 0;
}

// double a[N][4]
// 总长度为 N * 4 * 8
using DoubleArrary4 = double[4];
DoubleArrary4* Get32bytesAlignVector(int N) {
  auto p = (DoubleArrary4*)Get32bytesAlignMemory(N * 4 * 8);
  if (p == 0) {
    exit(-10001);
  }
  return p;
}

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



int N = 10000;
int K = 20;
int maxIter = 200;

DoubleArrary4* kmeans_data = nullptr;
DoubleArrary4* centroids = nullptr;
DoubleArrary4* backup_centroids = nullptr;
DoubleArrary4* newCentroids = nullptr;


int main() {
    cout<<"parameter setting: "<<endl;
    cout<<"The number of data points is: "<<N<<endl;
    cout<<"The number of clusters is: "<<K<<endl;
    cout<<"The iteration round number is: "<<maxIter<<endl;
    cout<<endl;

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

    // Run k-means algorithm
    init_centroids();
    auto start = chrono::steady_clock::now();
    kMeans();
    auto ends = chrono::steady_clock::now();

    // Compute accuracy and execution time
    double accuracy = getAccuracy();
    double time = chrono::duration_cast<chrono::microseconds>(ends - start).count() / 1000000.0;

    //cout << "Accuracy: " << accuracy << endl;
    cout<< "---Non-parallelized k-means algorithm---"<<endl;
    cout << "Execution time: " << time << " seconds" << endl;
    cout<<endl;
    // Run k-means algorithm
    init_centroids();
    auto start_avx = chrono::steady_clock::now();
    kMeans_avx();
    auto end_avx = chrono::steady_clock::now();

    double accuracy_avx = getAccuracy();
    double time_avx = chrono::duration_cast<chrono::microseconds>(end_avx - start_avx).count() / 1000000.0;

    //cout << "Accuracy: " << accuracy_avx << endl;
    cout<< "---Parallelized k-means algorithm---"<<endl;
    cout << "Execution time: " << time_avx << " seconds" << endl;
    cout<<endl;
    kmeans_data = Get32bytesAlignVector(N);
    fill_random(kmeans_data, N);
    backup_centroids = Get32bytesAlignVector(K);
    centroids = Get32bytesAlignVector(K);
    newCentroids = Get32bytesAlignVector(K);

    for (int i = 0; i < K; ++i) {
        int x = rand() % N;
        for (int j = 0; j < 4; j++){
            backup_centroids[i][j] = kmeans_data[x][j];
        }
    }

    init_centroids();
    auto start_avx_align = chrono::steady_clock::now();
    kMeans_avx_align();
    auto end_avx_align = chrono::steady_clock::now();

    double accuracy_avx_align = getAccuracy();
    double time_avx_align = chrono::duration_cast<chrono::microseconds>(end_avx_align - start_avx_align).count() / 1000000.0;

    //cout << "Accuracy: " << accuracy_avx_align << endl;
    cout<< "---Advanced parallelized k-means algorithm---"<<endl;
    cout << "Execution time: " << time_avx_align << " seconds" << endl;
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

        // Check for convergence
        /*bool converged = true;
        for (int j = 0; j < K; ++j) {
            for (int k = 0; k < 4; ++k) {
                if (newCentroids[j][k] != centroids[j][k]) {
                    converged = false;
                    break;
                }
            }
            if (!converged) {
                break;
            }
        }

        if (converged) {
            break;
        }*/

        for (int j = 0; j < K; j++){
            for(int k = 0; k < 4; k++) {
                centroids[j][k] = newCentroids[j][k];
            }
        }
    }
}


void kMeans_avx() {
    vector<int> labels(N);
    for (int iter = 0; iter < maxIter; ++iter) {
        for (int i = 0; i < N; ++i) {
            double minDist = numeric_limits<double>::max();
            int label = -1;
            __m256d kmeans_data_i = _mm256_loadu_pd(&kmeans_data[i][0]);
            for (int j = 0; j < K; ++j) {
                __m256d centroid_j = _mm256_loadu_pd(&centroids[j][0]);
                double dist_sum = calculate_euclidean_distance(kmeans_data_i, centroid_j);
                if (dist_sum < minDist) {
                    minDist = dist_sum;
                    label = j;
                }
            }
            labels[i] = label;
        }

        fill_zero(newCentroids, K);
        vector<int> counts(K, 0);

        for (int i = 0; i < N; ++i) {
            int label = labels[i];
            __m256d kmeans_data_i = _mm256_loadu_pd(&kmeans_data[i][0]);
            __m256d newCentroid_label_0 = _mm256_loadu_pd(&newCentroids[label][0]);
            _mm256_storeu_pd(&newCentroids[label][0], _mm256_add_pd(newCentroid_label_0, kmeans_data_i));
            ++counts[label];
        }

        for (int j = 0; j < K; ++j) {
            if (counts[j] > 0) {
                __m256d newCentroid_j_0 = _mm256_loadu_pd(&newCentroids[j][0]);
                __m256d count_j = _mm256_set1_pd(counts[j]);
                //__m256d count_inv_j = _mm256_set1_pd(1.0 / counts[j]);
                __m256d centroid_j = _mm256_div_pd(newCentroid_j_0, count_j);
                _mm256_storeu_pd(&centroids[j][0], centroid_j);
            }
        }

        /*bool converged = true;
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < 4; ++k) {
                if (newCentroids[i][k] != centroids[i][k]) {
                    converged = false;
                    break;
                }
            }
        }
        if (converged) {
            break;
        }*/
        for (int j = 0; j < K; j++){
            for(int k = 0; k < 4; k++) {
                centroids[j][k] = newCentroids[j][k];
            }
        }
    }

}

void kMeans_avx_align() {
    vector<int> labels(N);
    for (int iter = 0; iter < maxIter; ++iter) {
        for (int i = 0; i < N; ++i) {
            double minDist = numeric_limits<double>::max();
            int label = -1;
            __m256d kmeans_data_i = _mm256_load_pd(&kmeans_data[i][0]);
            for (int j = 0; j < K; ++j) {
                __m256d centroid_j = _mm256_load_pd(&centroids[j][0]);
                double dist_sum = calculate_euclidean_distance(kmeans_data_i, centroid_j);
                if (dist_sum < minDist) {
                    minDist = dist_sum;
                    label = j;
                }
            }
            labels[i] = label;
        }

        fill_zero(newCentroids, K);
        vector<int> counts(K, 0);

        for (int i = 0; i < N; ++i) {
            int label = labels[i];
            __m256d kmeans_data_i = _mm256_load_pd(&kmeans_data[i][0]);
            __m256d newCentroid_label_0 = _mm256_load_pd(&newCentroids[label][0]);
            _mm256_store_pd(&newCentroids[label][0], _mm256_add_pd(newCentroid_label_0, kmeans_data_i));
            ++counts[label];
        }

        for (int j = 0; j < K; ++j) {
            if (counts[j] > 0) {
                __m256d newCentroid_j_0 = _mm256_load_pd(&newCentroids[j][0]);
                __m256d count_j = _mm256_set1_pd(counts[j]);
                //__m256d count_inv_j = _mm256_set1_pd(1.0 / counts[j]);
                __m256d centroid_j = _mm256_div_pd(newCentroid_j_0, count_j);
                _mm256_store_pd(&centroids[j][0], centroid_j);
            }
        }

        /*bool converged = true;
        for (int i = 0; i < K; ++i) {
            for (int k = 0; k < 4; ++k) {
                if (newCentroids[i][k] != centroids[i][k]) {
                    converged = false;
                    break;
                }
            }
        }
        if (converged) {
            break;
        }*/
        for (int j = 0; j < K; j++){
            for(int k = 0; k < 4; k++) {
                centroids[j][k] = newCentroids[j][k];
            }
        }
    }

}


double calculate_euclidean_distance(__m256d a, __m256d b) {
    __m256d diff = _mm256_sub_pd(a, b);
    __m256d squared_diff = _mm256_mul_pd(diff, diff);
    __m256d sum = _mm256_hadd_pd(squared_diff, squared_diff);
    sum = _mm256_hadd_pd(sum, sum);
    double result;
    _mm256_store_pd(&result, sum);
    return result;
}


double getAccuracy() {
    // Assign each kmeans_data point to the nearest centroid
    vector<int> labels(N);
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

    // Compute accuracy
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        if (labels[i] == i / (N / K)) {
            ++correct;
        }
    }
    double accuracy = static_cast<double>(correct) / N;

    return accuracy;
}



