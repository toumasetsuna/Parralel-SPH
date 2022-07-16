#include <iostream>
#include<vector>
#include<cmath>
#include<limits.h>
#include<unordered_map>
#include<map>
#include<queue>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)


using namespace std;

//three dimension
struct mydim3 {
    double x, y, z;

    
};

struct intdim3 {
    int x, y, z;
};
__constant__ double d_K = 1.0;
__constant__ double d_timetick = 0.02;
__constant__ mydim3 const d_Gravity = { 0, -9.8, 0 };
__constant__ double d_Density = 1000;
__device__ double d_PI = 3.141592653589793;
double const H = 0.01;
__constant__ double d_H = 0.03;
__constant__ double d_MIU = 0.001;
int const Max_particle_per_cell = 10;

mydim3 const areaupbound = mydim3{ 20, 10, 10 };
__constant__ mydim3 d_areaupbound = { 20, 10, 10 };
int t_volume = 2000;
struct Particle {
    mydim3 position;//the position of the particle
    mydim3 speed;//the speed of the particle
    mydim3 acceleration;//the acceleration of the particle
    double density;
    double pressure;
    double weight;
};

Particle particleList[1000000];
int particleNum = 0;
__device__ __managed__ int d_particleNum;
//__device__ Particle *d_ParticleList;
int const hashmap_size = 20000;
int cellmap[hashmap_size];

//__device__ int* d_hashmap;
//__device__ int **d_Neighbours;
__device__ double poly6(double r2) {
    return 315 / (64 * d_PI * pow(d_H, 9)) * pow((d_H * d_H - r2), 3);
}

__device__ double getDistance2(mydim3 a, mydim3 b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}

__device__ int d_transToIndex(int x, int y, int z) {
    int index = x * (int)d_areaupbound.y * (int)d_areaupbound.z * Max_particle_per_cell +
        y * (int)d_areaupbound.z * (int)Max_particle_per_cell + z * Max_particle_per_cell;
    return index;
}

__device__ int toNI(int x, int y) {
    int temp = 100;
    int index = x * temp + y;
    return index;
}

int transToIndex(int x, int y, int z) {
    int index = x * (int)areaupbound.y * (int)areaupbound.z * Max_particle_per_cell +
        y * (int)areaupbound.z * (int)Max_particle_per_cell + z * Max_particle_per_cell;
    return index;
}

int gethash() {
    for (int i = 0; i < particleNum; i++) {
        int x = (int)(particleList[i].position.x / H);
        int y = (int)(particleList[i].position.y / H);
        int z = (int)(particleList[i].position.z / H);
        int index = transToIndex(x, y, z);
        if (cellmap[index] < Max_particle_per_cell - 1) {
            cellmap[index]++;
            cellmap[index + cellmap[index]] = i;
        }
    }
    return 0;
}


int init(mydim3 lowbound, mydim3 upbound) {
    int x0 = lowbound.x;
    int y0 = lowbound.y;
    int z0 = lowbound.z;
    for (int i = x0; i < upbound.x; i++) {
        for (int j = y0; j < upbound.y; j++) {
            for (int k = z0; k < upbound.z; k++) {
                particleNum++;
                Particle& particle = particleList[(int)((i - x0) * upbound.y * upbound.z + (j - y0) * upbound.z +
                    (k - z0))];
                particle.position = mydim3{ i * 0.01, j * 0.01, k * 0.01 };
                particle.speed = mydim3{ 0, 0, 0 };
                particle.acceleration = mydim3{ 0, 0, 0 };
                particle.density = 1000;
                particle.pressure = 0;
                particle.weight = 0.01;
            }
        }
    }
    return 0;
}


__global__ void neighbour_kernel(int* d_hashmap, int* d_Neighbours, Particle* d_ParticleList) {
    //    int t0=   __CUDA_ARCH__;
    //    printf("%d\n", t0);
    int num = threadIdx.x + blockIdx.x * blockDim.x;
    if (num >= d_particleNum)return;
    //printf("%d ",num);
    //printf("%d ",d_particleNum);
    mydim3 position = d_ParticleList[num].position;
    int x = position.x / 0.01;
    int y = position.y / 0.01;
    int z = position.z / 0.01;
    intdim3 key[7];
    key[0] = intdim3{ x, y, z };
    key[1] = intdim3{ x + 1, y, z };
    key[2] = intdim3{ x - 1, y, z };
    key[3] = intdim3{ x, y + 1, z };
    key[4] = intdim3{ x, y - 1, z };
    key[5] = intdim3{ x, y, z + 1 };
    key[6] = intdim3{ x, y, z - 1 };
    d_Neighbours[toNI(num, 0)] = 0;
    for (int i = 0; i < 7; i++) {
        intdim3 the_key = key[i];
        int x0 = the_key.x;
        int y0 = the_key.y;
        int z0 = the_key.z;
        if (x0 < 0 || y0 < 0 || z0 < 0) continue;
        if (x0 >= d_areaupbound.x || y0 >= d_areaupbound.y || z0 >= d_areaupbound.z)continue;
        int index0 = d_transToIndex(x0, y0, z0);
        // printf("%d ", index0);
        for (int j = 1; j <= d_hashmap[index0]; j++) {
            int index = d_hashmap[index0 + j];
            if (index == num)continue;
            // printf("%d ", index);
            if (getDistance2(d_ParticleList[index].position, d_ParticleList[num].position) < d_H * d_H) {
                d_Neighbours[toNI(num, 0)]++;
                //printf("%d ",d_Neighbours[toNI(num,0)]);
                d_Neighbours[toNI(num, d_Neighbours[toNI(num, 0)])] = index;
            }
        }
    }
    //atomicAdd(m_test,1);
}

__global__ void updateDensityKernel(int* d_Neighbours, Particle* d_ParticleList) {
    int num = threadIdx.x + blockIdx.x * blockDim.x;
    if (num >= d_particleNum) {
        return;
    }
    double density = 0;
    for (int i = 1; i <= d_Neighbours[toNI(num, 0)]; i++) {
        density += d_ParticleList[d_Neighbours[toNI(num, i)]].weight *
            poly6(getDistance2(d_ParticleList[num].position,
                d_ParticleList[d_Neighbours[toNI(num, i)]].position));
    }
    d_ParticleList[num].density = density;
    return;
}

__global__ void updatePressureKernel(Particle* d_ParticleList) {
    int num = threadIdx.x + blockIdx.x * blockDim.x;
    if (num > d_particleNum) {
        return;
    }
    d_ParticleList[num].pressure = d_K * (d_ParticleList[num].density - d_Density);
    return;
}

__global__ void updateAccelerationKernel(int* d_Neighbours, Particle* d_ParticleList) {
    int num = threadIdx.x + blockIdx.x * blockDim.x;
   
    d_ParticleList[num].acceleration = d_Gravity;
 
    mydim3 pressure_acc = mydim3{ 0, 0, 0 };
    mydim3 viscosity_acc = mydim3{ 0, 0, 0 };
    Particle p1 = d_ParticleList[num];
    for (int i = 1; i <= d_Neighbours[toNI(num, 0)]; i++) {
        Particle p2 = d_ParticleList[d_Neighbours[toNI(num, i)]];
        double r = sqrt(getDistance2(p1.position, p2.position));
        double temp1 = (p1.pressure + p2.pressure) / (2 * p1.density * p2.density) * pow((d_H - r), 2) / r;

        pressure_acc.x += temp1 * (p1.position.x - p2.position.x);
        pressure_acc.y += temp1 * (p1.position.y - p2.position.y);
        pressure_acc.z += temp1 * (p1.position.z - p2.position.z);
        double temp2 = (d_H - r) / (p1.density * p2.density);
        printf("%f %f\n ", p2.speed.x, p1.speed.x);
         viscosity_acc.x += temp2 * (p2.speed.x - p1.speed.x);
         viscosity_acc.y += temp2 * (p2.speed.y - p1.speed.y);
         viscosity_acc.z += temp2 * (p2.speed.z - p1.speed.z);
    }
    double temp1 = p1.weight * 45 / (d_PI * pow(d_H, 6));
 
    pressure_acc.x = temp1 * pressure_acc.x;
    pressure_acc.y = temp1 * pressure_acc.y;
    pressure_acc.z = temp1 * pressure_acc.z;
    //printf("%f\n",pressure_acc.x);
   // printf("%f",pressure_acc.y);
    //printf("%f",pressure_acc.z);

    double temp2 = p1.weight * 45 / (d_PI * pow(d_H, 6)) * d_MIU;
    viscosity_acc.x = temp2 * viscosity_acc.x;
    viscosity_acc.y = temp2 * viscosity_acc.y;
    viscosity_acc.z = temp2 * viscosity_acc.z;
    // printf("%f\n",viscosity_acc.x);*/
    // printf("%f\n",viscosity_acc.y);
     //printf("%f",viscosity_acc.z);
     //printf("%d %f %f %f", num,viscosity_acc.x, viscosity_acc.y, viscosity_acc.z);

    d_ParticleList[num].acceleration.x += pressure_acc.x + viscosity_acc.x;
   


    d_ParticleList[num].acceleration.y += pressure_acc.y + viscosity_acc.y;
    //atomicAdd(&d_ParticleList[num].acceleration.y, pressure_acc.y + viscosity_acc.y);
    d_ParticleList[num].acceleration.z += pressure_acc.z + viscosity_acc.z;
   /*   printf("%f\n",d_ParticleList[num].acceleration.x);
      printf("%f\n",d_ParticleList[num].acceleration.y);
      printf("%f\n",d_ParticleList[num].acceleration.z);*/
      //atomicAdd(&d_ParticleList[num].acceleration.z, pressure_acc.z + viscosity_acc.z);
}


__global__ void updateSpeedKernel(Particle* d_ParticleList) {
    int num = threadIdx.x + blockIdx.x * blockDim.x;
    if (num > d_particleNum) {
        return;
    }
    d_ParticleList[num].speed.x += d_ParticleList[num].acceleration.x * d_timetick;
    d_ParticleList[num].speed.y += d_ParticleList[num].acceleration.y * d_timetick;
    d_ParticleList[num].speed.z += d_ParticleList[num].acceleration.z * d_timetick;
}


__global__ void updatePositionKernel(Particle* d_ParticleList) {
    int num = threadIdx.x + blockIdx.x * blockDim.x;
    if (num > d_particleNum) {
        return;
    }
    d_ParticleList[num].position.x += d_ParticleList[num].speed.x * d_timetick;
    d_ParticleList[num].position.y += d_ParticleList[num].speed.y * d_timetick;
    d_ParticleList[num].position.z += d_ParticleList[num].speed.z * d_timetick;
}

__device__ void print_arch() {
    const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
    printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}
__global__ void example()
{
    print_arch();
}

int main() {

    example << <1, 1 >> > ();
    cudaDeviceSynchronize();
    mydim3 lowbound = mydim3{ 0, 0, 0 };
    mydim3 upbound = mydim3{ 10, 10, 10 };

    init(lowbound, upbound);
    d_particleNum = particleNum;
    cudaError_t ret;


    int size_particle = particleNum * sizeof(Particle);
    int size_neighbour = 100 * particleNum * sizeof(int);
    Particle* d_ParticleList;
    ret = cudaMalloc(&d_ParticleList, size_particle);
    if (ret != cudaSuccess) {
        printf("malloc particle failed");
    }
    ret = cudaMemcpy(d_ParticleList, particleList, size_particle, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("copy particle failed");
    }
    int* d_cellmap;
    ret = cudaMalloc(&d_cellmap,
        (int)areaupbound.x * (int)areaupbound.y * (int)areaupbound.z * Max_particle_per_cell *
        sizeof(int));

    if (ret != cudaSuccess) {
        printf("malloc cellmap failed");
    }
    for (int i = 0; i < areaupbound.x; i++) {
        for (int j = 0; j < areaupbound.y; j++) {
            for (int k = 0; k < areaupbound.z; k++) {
                int index = transToIndex(i, j, k);
                cellmap[index] = 0;
            }
        }
    }
    gethash();
    //for(int i=0;i<20000;i++) printf("%d ",cellmap[i]);
    ret = cudaMemcpy(d_cellmap, cellmap,
        20000 * sizeof(int),
        cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        printf("copy cellmap failed");
    }
    int* d_Neighbours;
    ret = cudaMalloc(&d_Neighbours, particleNum * 100 * sizeof(int));
    if (ret != cudaSuccess) {
        printf("malloc neighbour failed");
    }

    int Block_Size = 1024;
    int temp = (int)floor(particleNum / Block_Size) + 1;
    dim3 grid1(temp);
    dim3 block1(Block_Size);
    neighbour_kernel << <grid1, block1 >> > (d_cellmap, d_Neighbours, d_ParticleList);
    int neighbour[100000];
    //cudaDeviceSynchronize();
    ret = cudaMemcpy(neighbour, d_Neighbours, 100000 * sizeof(int), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        printf("copy neighbour failed");
    }
    /* for (int i = 0; i < particleNum; i++) {
         printf("Particle:%d    ",i);
         for (int j = 0; j < 100; j++) {
             printf("%d ", neighbour[i * 100 + j]);
         }
         printf("\n");
     }*/


    updateDensityKernel << <grid1, block1 >> > (d_Neighbours, d_ParticleList);

    updatePressureKernel << <grid1, block1 >> > (d_ParticleList);

    cudaDeviceSynchronize();
    updateAccelerationKernel << <grid1, block1 >> > (d_Neighbours, d_ParticleList);
    cudaDeviceSynchronize();
    cudaGetErrorString(ret);
    if (ret != cudaSuccess) {
        printf("update acceleration failed");
    }
    updateSpeedKernel << <grid1, block1 >> > (d_ParticleList);
    updatePositionKernel << <grid1, block1 >> > (d_ParticleList);

    ret = cudaMemcpy(particleList, d_ParticleList, size_particle, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        printf("copy particle failed");
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < particleNum; i++) {
        printf("Particle:%d    ", i);
        printf("%f %f %f\n", particleList[i].position.x, particleList[i].position.y, particleList[i].position.z);
        printf("%f %f %f\n", particleList[i].speed.x, particleList[i].speed.y, particleList[i].speed.z);
        printf("%f %f %f\n", particleList[i].acceleration.x, particleList[i].acceleration.y,
            particleList[i].acceleration.z);
        printf("%f\n", particleList[i].density);
        printf("%f\n", particleList[i].pressure);
    }


    //int* particleNum=new int();

    //printf("%d\n",particleNum*);
    //printf("%f",ceil((double)particleNum / Block_Size));
    // updateNeighbour();
    //updateDensity();
    //updatePressure();
    //updateAcceleration();
    //updateSpeed();
    //updatePosition();
    return 0;
}

