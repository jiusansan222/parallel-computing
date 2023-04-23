
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

// 数据点结构体
struct Point {
    double x, y;
};

// 计算两点之间的距离
double distance(Point p1, Point p2) {
    return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}

// K-means算法
vector<Point> kmeans(vector<Point> data, int k) {
    // 初始化簇中心
    vector<Point> centers;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, data.size() - 1);
    for (int i = 0; i < k; i++) {
        centers.push_back(data[dis(gen)]);
    }

    // 迭代聚类
    while (true) {
        // 初始化簇
        vector<vector<Point>> clusters(k);
        // 将每个数据点分配到最近的簇中心
        for (Point p : data) {
            int index = 0;
            double minDist = distance(p, centers[0]);
            for (int i = 1; i < k; i++) {
                double dist = distance(p, centers[i]);
                if (dist < minDist) {
                    index = i;
                    minDist = dist;
                }
            }
            clusters[index].push_back(p);
        }
        // 计算新的簇中心
        vector<Point> newCenters(k);
        bool converged = true;
        for (int i = 0; i < k; i++) {
            double sumX = 0, sumY = 0;
            for (Point p : clusters[i]) {
                sumX += p.x;
                sumY += p.y;
            }
            int size = clusters[i].size();
            if (size > 0) {
                Point newCenter = {sumX / size, sumY / size};
                if (newCenter.x != centers[i].x || newCenter.y != centers[i].y) {
                    converged = false;
                }
                newCenters[i] = newCenter;
            } else {
                newCenters[i] = centers[i];
            }
        }
        if (converged) {
            return newCenters;
        }
        centers = newCenters;
    }
}

int main() {
    // 测试数据
    vector<Point> data = {{1, 1}, {1, 2}, {2, 2}, {10, 10}, {10, 11}, {11, 11}, {71, 20}, {12, 82}};
    // K-means聚类2
    vector<Point> centers = kmeans(data, 2);
    // 输出聚类结果
    for (int i = 0; i < centers.size(); i++) {
        cout << "Cluster " << i + 1 << ":\n";
        for (Point p : data) {
            if (distance(p, centers[i]) < distance(p, centers[(i + 1) % 2])) {
                cout << "(" << p.x << ", " << p.y << ")\n";
            }
        }
    }
    return 0;
}
