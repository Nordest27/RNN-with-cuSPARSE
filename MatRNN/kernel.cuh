#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cusparse_v2.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <random>
#include <queue>
#include <fstream>
#include "../../MNIST_reader/include/mnist/mnist_reader_less.hpp"

double randomdouble();

int sign(double val);

int max_pos(int ini, int end, const std::vector<double> &values);

double sum(const std::vector<double> &v);

struct COO_matrix
{
    int rows;
    int cols;
    int nnz;

    std::vector<int> cooRowInd;
    std::vector<int> cooColInd;
    std::vector<double> cooValues;

    COO_matrix();

    void connect_layers(int ini1, int end1, int ini2, int end2, double sparsity, int nodes);
    bool insert_elm(int node_i, int node_j, double value);

    ~COO_matrix();
};

void mat_vect_mul(COO_matrix& mat, std::vector<double>& vect, std::vector<double>& result);

__global__ void addKernel(double* a, double c, double* b, int size);
__global__ void multKernel(double* a, double c, double* b, int size);
__global__ void resetVectKernel(double* a, int size);
__global__ void resetVectKernel(int* a, int size);
__global__ void copyVectorKernel(double* a, double* b, int size);
__global__ void TmultVectsKernel(double* values, int* cooRow, int* cooCol, double* a, double c, double* b, int size);
__global__ void useActKernel(double* values, double* dx, int* act_f, int size, int* mask);
__global__ void updateAdamKernel(double beta1, double beta2, double* m, double* v, double* m_corr, double* v_corr, double* g, int size, double powBeta1, double powBeta2);
__global__ void addWithAdamKernel(double* values, double alpha, double* m_corr, double* v_corr, double eps, int size);
void smpv(COO_matrix &mat, std::vector<double> &vect, std::vector<double> &result);

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);