#include "kernel.cuh"
#include <math.h>

__global__ void addKernel(double* a, double c,  double* b, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    a[id] = __fma_rn(c, b[id], a[id]);
}

__global__ void updateAdamKernel(double beta1, double beta2, double* m, double* v, 
                           double* m_corr, double* v_corr, double* g, 
                           int size, double powBeta1, double powBeta2)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    m[id] = beta1 * m[id] + (1.0 - beta1) * g[id];
    v[id] = beta2 * v[id] + (1.0 - beta2) * g[id] * g[id];

    m_corr[id] = m[id] / (1.0 - powBeta1);
    v_corr[id] = v[id] / (1.0 - powBeta2);
}

__global__ void addWithAdamKernel(double* values, double alpha, double* m_corr,
                                  double* v_corr, double eps, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size ) return;
    values[id] += alpha * (m_corr[id] / (sqrt(v_corr[id]) + eps) - values[id]*0.01);
}



__global__ void multKernel(double* a, double c, double* b, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    a[id] = __dmul_rn(a[id], __dmul_rn(c, b[id]) );
}

__global__ void resetVectKernel(double* a, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    a[id] = 0;
}

__global__ void resetVectKernel(int* a, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    a[id] = 0;
}

__global__ void copyVectorKernel(double* a, double* b, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    a[id] = b[id];
}

__global__ void useActKernel(double* values, double* dx, int* act_f, int size, int* mask)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    if (mask[id] == 1)
    {
        dx[id] = 0;
        values[id] = 0;
        return;
    }
    int aux_act_f = act_f[id];
    dx[id] = 0;
    //RELU 0, SIGMOID 1, LINEAL 2
    if (aux_act_f == 1) {
        values[id] = 1 / (1 + exp(-values[id]));
        dx[id] = values[id] * (1 - values[id]);
    }
    else if (aux_act_f == 0) {
        if (values[id] <= 0) {
            values[id] = 0;
            dx[id] = 0;
        }
        else dx[id] = 1;
    }
    else if (aux_act_f == 2) dx[id] = 1;

}

__global__ void TmultVectsKernel(double* values, int* cooRow, int* cooCol, double* a, double c, double* b, int size)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= size) return;

    //std::cout << a[cooRow[i]] << " " << b[cooCol[i]] << " " << a[cooRow[i]] * b[cooCol[i]] << std::endl;
    values[id] = __fma_rn(c,__dmul_rn(a[cooRow[id]], b[cooCol[id]]), values[id]);
}

//COO_MATRIX//

COO_matrix::COO_matrix()
{
    rows = 0;
    cols = 0;
    nnz = 0;
    cooRowInd = std::vector<int>();
    cooColInd = std::vector<int>();
    cooValues = std::vector<double>();
}

void COO_matrix::connect_layers(int ini1, int end1, int ini2, int end2, double sparsity, int nodes)
{
    /*int posi = 0;

    while (posi < nnz && cooRowInd[posi] < ini2 && cooColInd[posi] < ini1)
        posi++;
    
   for (int nj = ini2; nj < end2; nj++)
        for (int ni = ini1; ni < end1; ni++)
        {
            if (double(rand() % 1000) / 1000 < sparsity) {
                if (posi < nnz && cooRowInd[posi] == nj && cooColInd[posi] == ni)
                    posi++;
                continue;
            }
            if ( posi < nnz && cooRowInd[posi] == nj && cooColInd[posi] == ni)
            {
                posi++;
               // std::cout << posi << " "<< cooRowInd[posi] <<" "<<cooColInd[posi]<<std::endl;
                continue;
            }


            cooRowInd.insert(cooRowInd.begin() + posi, nj);
            cooColInd.insert(cooColInd.begin() + posi, ni);
            cooValues.insert(cooValues.begin() + posi, randomdouble()/10);
            
            posi++;
            nnz++;
        }*/
    int posi = 0;

    std::vector<int> row_aux;
    std::vector<int> col_aux;
    std::vector<double> values;
    std::vector<bool> seen(rows * cols, false);
    int ini_cols = cols;
    int ini_rows = rows;
    for (int i = 0; i < cooValues.size(); ++i)
        seen[cooRowInd[i] * ini_cols + cooColInd[i]] = true;

    std::cout << ini2 << " " << end2 << " " << ini1 << " " << end1<< std::endl;
   /* for (int i = 0; i < nnz; ++i)
    {
        std::cout << i << " " << cooRowInd[i] << " " << cooColInd[i] << std::endl;

    }*/


    for (int nj = ini2; nj < end2; nj++)
        for (int ni = ini1; ni < end1; ni++)
        {
            while (posi < cooValues.size() && (cooRowInd[posi] < nj 
                                              || (cooRowInd[posi] == nj && cooColInd[posi] < ni)) ) {
              //  std::cout << "YES! " << nj << " " << ni << " , " << cooRowInd[posi] << " " << cooColInd[posi] << std::endl;

                row_aux.push_back(cooRowInd[posi]);
                col_aux.push_back(cooColInd[posi]);
                values.push_back(cooValues[posi]);
                posi++;            }

            if (double(rand() % 1000) / 1000 < sparsity) {
                if (posi < cooValues.size() && cooRowInd[posi] == nj && cooColInd[posi] == ni) {
                    row_aux.push_back(nj);
                    col_aux.push_back(ni);
                    values.push_back(cooValues[posi]);
                    posi++;

                }
                continue;
            }

            if ( posi < cooValues.size() && cooRowInd[posi] == nj && cooColInd[posi] == ni)
            {
                //std::cout << "FOUND! " << nj << " " << ni << " , " << cooRowInd[posi] << " " << cooColInd[posi] << std::endl;

                row_aux.push_back(nj);
                col_aux.push_back(ni);
                values.push_back(cooValues[posi]);


                posi++;
                continue;
            }

            if (ini_cols>ni && ini_rows > nj && seen[nj*ini_cols+ni])
                std::cout << "BAD! " << nj << " " << ni <<" , " << cooRowInd[posi] << " " << cooColInd[posi] << std::endl;

            row_aux.push_back(nj);
            col_aux.push_back(ni);
            values.push_back(randomdouble()/sqrt(nodes));
            nnz++;
            if (nj >= rows)rows = nj + 1;
            if (ni >= cols)cols = ni + 1;


        }

    while (posi < cooValues.size())
    {
        row_aux.push_back(cooRowInd[posi]);
        col_aux.push_back(cooColInd[posi]);
        values.push_back(cooValues[posi]);


        posi++;
    }

    cooRowInd = row_aux;
    cooColInd = col_aux;
    cooValues = values;

   seen = std::vector<bool>(rows * cols, false);
 

    

}

bool COO_matrix::insert_elm(int node_i, int node_j, double value)
{
    /*int i = 0;
    for (i = 0; i < nnz; ++i)
        if (node_j == cooRowInd[i] && node_i == cooColInd[i])
            return false;
        else if (node_j < cooRowInd[i] || (node_j == cooRowInd[i] && node_i < cooColInd[i]) )
            break;
            */
    int left_i = 0;
    int right_i = nnz;
    int mid = 0;
    while (left_i < right_i)
    {
        mid = (left_i + right_i) / 2;
        if (node_j == cooRowInd[mid] && node_i == cooColInd[mid])
            return false;

        if (node_j < cooRowInd[mid]) right_i = mid;
        else if (node_j > cooRowInd[mid]) left_i = mid;
        else {
            if (node_i < cooColInd[mid]) right_i = mid;
            else if (node_i > cooColInd[mid]) left_i = mid;
        }

        if (abs(left_i - right_i) == 1)
        {
            if (node_j > cooRowInd[left_i] || (node_j == cooRowInd[left_i] && node_i > cooColInd[left_i]))
                mid = right_i;
            else mid = left_i;

            break;
        }
    }

    // if (i != mid) std::cout << i << " " << mid << std::endl;

    cooRowInd.insert(cooRowInd.begin() + mid, node_j);
    cooColInd.insert(cooColInd.begin() + mid, node_i);
    cooValues.insert(cooValues.begin() + mid, value);
    /*cooRowInd.push_back(node_j);
    cooColInd.push_back(node_i);
    cooValues.push_back(value);*/
    nnz++;

    if (node_j >= rows)rows = node_j + 1;
    if (node_i >= cols)cols = node_i + 1;


    return true;
}

COO_matrix::~COO_matrix() {

}
//////////////

void mat_vect_mul(COO_matrix& mat, std::vector<double>& vect, std::vector<double>& result) {
    for (int i = 0; i < mat.rows; ++i)
    {
        result[i] = 0;
    }
    for (int i = 0; i < mat.nnz; ++i)
    {
        //printf("i: %d res node: %d %f inp node: %d %f weight: %f\n", i, mat.cooColInd[i], result[mat.cooColInd[i]], mat.cooRowInd[i], vect[mat.cooRowInd[i]], mat.cooValues[i]);
        result[mat.cooRowInd[i]] += vect[mat.cooColInd[i]] * mat.cooValues[i];
    }
    //printf("%f\n",result[2]);
}

double randomdouble()
{
    return double(rand() - RAND_MAX / 2) / RAND_MAX;
}

int sign(double val)
{
    return (int(0) < val) - (val < int(0));
}

int max_pos(int ini, int end, const std::vector<double> &values)
{
    int maxi = ini;
    for(int i = ini; i < end; ++i)
        if (values[maxi] < values[i])
            maxi = i;
    return maxi;
}

double sum(const std::vector<double> &v)
{
    double sum = 0;
    for (double val : v)
        sum += val;
    return sum;
}


// multiply sparse matrix and dense vector
void smpv( COO_matrix &mat, std::vector<double> &vect, std::vector<double> &result)
{
    //--------------------------------------------------------------------------
    // Device memory management
    int* dA_rows, * dA_columns;
    double* dA_values, * dX, * dY;
    cudaMalloc((void**)&dA_rows, mat.nnz * sizeof(int));
    cudaMalloc((void**)&dA_columns, mat.nnz * sizeof(int));
    cudaMalloc((void**)&dA_values, mat.nnz * sizeof(double));
    cudaMalloc((void**)&dX, mat.cols * sizeof(double));
    cudaMalloc((void**)&dY, mat.rows * sizeof(double));

    cudaMemcpy(dA_rows, mat.cooRowInd.data(), mat.nnz * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, mat.cooColInd.data(), mat.nnz * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, mat.cooValues.data(), mat.nnz * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dX, vect.data(), mat.cols * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(dY, result.data(), mat.rows * sizeof(double),
        cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in COO format
    cusparseCreateCoo(&matA, mat.rows, mat.cols, mat.nnz,
        dA_rows, dA_columns, dA_values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    // Create dense vector X
    cusparseCreateDnVec(&vecX, mat.cols, dX, CUDA_R_32F);
    // Create dense vector y
    cusparseCreateDnVec(&vecY, mat.rows, dY, CUDA_R_32F);
    double alpha = 1.0f;
    double beta = 0.0f;
    // allocate an external buffer if needed
    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_COO_ALG1, &bufferSize);
    //std::cout << bufferSize<< std::endl;
    cudaMalloc(&dBuffer, bufferSize);
     
    // execute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_COO_ALG1, dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    //--------------------------------------------------------------------------
    //get result
    cudaMemcpy(result.data(), dY, mat.rows * sizeof(double),
        cudaMemcpyDeviceToHost);
    cudaFree(dBuffer);
    cudaFree(dA_rows);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dX);
    cudaFree(dY);
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_a, 1, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
