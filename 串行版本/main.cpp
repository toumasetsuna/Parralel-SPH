#include <iostream>
#include<vector>
#include<cmath>

using namespace std;

struct dim3 {
    double x, y, z;

    dim3 operator+(dim3 b) {
        return dim3{x + b.x, y + b.y, z + b.z};
    }

    dim3 operator-(dim3 b) {
        return dim3{x - b.x, y - b.y, z - b.z};
    }

    dim3 operator+=(dim3 b) {
        this->x = this->x + b.x;
        this->y = this->y + b.y;
        this->z = this->z + b.z;
        return *this;
    }

    friend dim3 operator*(double a, dim3 b) {
        dim3 result;
        result.x = a * b.x;
        result.y = a * b.y;
        result.z = a * b.z;
        return result;
    }
    friend dim3 operator*(dim3 a,double b){
        dim3 result;
        result.x = a.x * b;
        result.y = a.y * b;
        result.z = a.z * b;
        return result;
    }
};

double K = 1.0;
double timetick=0.01;

dim3 const Gravity = {0, -9.8, 0};
double const Density = 1000;
double PI = 4 * atan(1);
double const H = 0.01;
double const MIU = 1.0;
//three dimension

struct Particle {
    dim3 position;//the position of the particle
    dim3 speed;//the speed of the particle
    dim3 acceleration;//the acceleration of the particle
    double density;
    double pressure;
    double weight;
};

Particle particleList[1000000];
int particleNum = 0;
vector<int> neighbour[1000000];

double poly6(double r2) {
    return 315 / (64 * PI * pow(H, 9)) * pow((H * H - r2), 3);
}

int init(dim3 lowbound, dim3 upbound) {
    int x0 = lowbound.x;
    int y0 = lowbound.y;
    int z0 = lowbound.z;
    for (int i = x0; i < upbound.x; i++) {
        for (int j = y0; j < upbound.y; j++) {
            for (int k = z0; k < upbound.z; k++) {
                particleNum++;
                Particle& particle = particleList[(int) ((i - x0) * upbound.y * upbound.z + (j - y0) * upbound.z +
                                                        (k - z0))];
                particle.position = dim3{i * 0.01, j * 0.01, k * 0.01};
                particle.speed = dim3{0, 0, 0};
                particle.acceleration = {0, 0, 0};
                particle.density = 1000;
                particle.pressure = 0;
                particle.weight = 0.001;
            }
        }
    }
    return 0;
}

double getDistance2(dim3 a, dim3 b) {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

int updateNeighbour() {
    for(int i=0;i<particleNum;i++){
        neighbour[i].clear();
    }
    for (int i = 0; i < particleNum; i++) {
        neighbour[i].push_back(i);
        for (int j = i + 1; j < particleNum; j++) {
            if (getDistance2(particleList[i].position, particleList[j].position) < H * H) {
                neighbour[i].push_back(j);
                neighbour[j].push_back(i);
            }
        }
    }
    return 0;
}

int updateDensity() {
    for (int i = 0; i < particleNum; i++) {
        double density = 0;
        vector<int>::iterator it = neighbour[i].begin();
        for (it; it != neighbour[i].end(); it++) {
            density += particleList[(*it)].weight *
                       poly6(getDistance2(particleList[i].position, particleList[(*it)].position));
        }
        particleList[i].density = density;
    }
    return 0;
}

int updatePressure() {
    for (int i = 0; i < particleNum; i++)
        particleList[i].pressure = K * (particleList[i].density - Density);
    return 0;
}

int updateAcceleration() {
    for (int i = 0; i < particleNum; i++) {
        vector<int>::iterator it = neighbour[i].begin();
        particleList[i].acceleration = Gravity;
        dim3 pressure_acc = dim3{0, 0, 0};
        dim3 viscosity_acc = dim3{0, 0, 0};
        Particle p1 = particleList[i];
        for (it; it != neighbour[i].end(); it++) {

            Particle p2 = particleList[*it];
            double r = sqrt(getDistance2(p1.position, p2.position));
            pressure_acc += (p1.pressure + p2.pressure) / (2 * p1.density * p2.density) * pow((H - r), 2) / r *
                            (p1.position - p2.position);
            viscosity_acc += (H - r) / (p1.density * p2.density) * (p2.speed - p1.speed);
        }
        pressure_acc = p1.weight * 45 / (PI * pow(H, 6)) * pressure_acc;
        viscosity_acc = p1.weight * 45 / (PI * pow(H, 6)) * MIU * viscosity_acc;
        particleList[i].acceleration += (pressure_acc + viscosity_acc);
    }
    return 0;
}

int updateSpeed() {
    for (int i = 0; i < particleNum; i++) {
        Particle& x=particleList[i];
        x.speed+=x.acceleration*timetick;
    }
    return 0;
}
int updatePosition(){
    for(int i=0;i<particleNum;i++){
        Particle& x=particleList[i];
        x.position+=x.speed*timetick;
    }
    return 0;
}
int main() {
    dim3 lowbound = dim3{0, 0, 0};
    dim3 upbound = dim3{10, 10, 10};
    init(lowbound, upbound);
    updateNeighbour();
    updateDensity();
    updatePressure();
    updateAcceleration();
    updateSpeed();
    updatePosition();
    return 0;
}

