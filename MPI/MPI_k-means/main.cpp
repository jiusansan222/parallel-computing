#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <time.h>
#include <limits>
#include <algorithm>
#include <immintrin.h>

#include <string.h>
#include <vector>

using namespace std;

void kMeans();
void init_centroids();
using DoubleArrary4 = double[4];


const int N = 1000000;
const int K = 20;
const int maxIter = 200;



// 增加32字节的前缀，用来寻找32字节对齐的地址，判断方法为 addr % 32 == 0
double* Get32bytesMemory(int len) {
  char* p = (char*)malloc(32 + len);
  memset(p, 0, 32 + len);
  for (int i = 0; i < 32; i++) {
    char* ret = p + i;
    size_t addr = (size_t)ret;
    if (addr % 32 != 0) {
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



vector<int> cluster(int start, int ends, DoubleArrary4* kmeans_data, DoubleArrary4* centroids) {
    vector<int> newlabels(N);
    for (int i = start; i < ends; ++i) {
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
        newlabels[i] = label;
    }
    return newlabels;
}


int main(int argc, char* argv[]) {
    DoubleArrary4* kmeans_data = nullptr;
    DoubleArrary4* centroids = nullptr;
    DoubleArrary4* backup_centroids = nullptr;
    DoubleArrary4* newCentroids = nullptr;
    vector<int> labels(N, 0);

    int ranks,sizes;
    int numsize;

    kmeans_data = Get32bytesVector(N);
    centroids = Get32bytesVector(K);
    clock_t startTime,endTime;
    startTime = clock();
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&ranks);
    MPI_Comm_size(MPI_COMM_WORLD,&sizes);

    if (ranks == 0) {
        cout<<"parameter setting: "<<endl;
        cout<<"The number of data points is: "<<N<<endl;
        cout<<"The number of clusters is: "<<K<<endl;
        cout<<"The iteration round number is: "<<maxIter<<endl;
        cout<<endl;

        fill_random(kmeans_data, N);
        backup_centroids = Get32bytesVector(K);

        newCentroids = Get32bytesVector(K);

        for (int i = 0; i < K; ++i) {
            int x = rand() % N;
            for (int j = 0; j < 4; j++){
                backup_centroids[i][j] = kmeans_data[x][j];
            }
        }
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < 4; j++){
                centroids[i][j] = backup_centroids[i][j];
            }
        }
        numsize = N / (sizes - 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);  //同步一下

    MPI_Bcast(kmeans_data,N*sizeof(double[4]),MPI_BYTE,0,MPI_COMM_WORLD);
    MPI_Bcast(&numsize,1,MPI_INT,0,MPI_COMM_WORLD);

    for (int iter = 0; iter < maxIter; ++iter) {
        MPI_Bcast(centroids,K*sizeof(double[4]),MPI_BYTE,0,MPI_COMM_WORLD);
        if(ranks){
            int starts = (ranks-1) * numsize;
            int ends = ranks * numsize;

            vector<int> newlabels = cluster(starts, ends, kmeans_data, centroids);
            MPI_Send(&newlabels[starts], numsize, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            for (int i = 1; i < sizes; ++i) {
                MPI_Recv(&labels[(i-1) * numsize], numsize, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);  //同步一下

        if (ranks == 0) {
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
        MPI_Barrier(MPI_COMM_WORLD);  //同步一下

    }

    MPI_Finalize();
    endTime = clock();
    cout <<ranks<< " : The run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}






