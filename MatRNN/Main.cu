
#include "Prune_evolve.cuh"

int main()
{

    cudaError_t cudaStatus;
    COO_matrix mat;
    // CUSPARSE APIs
    /*mat.rows = mat.cols = 10;
    mat.nnz = 10;
    mat.cooRowInd = { 0, 0, 0, 1, 2, 2, 2, 3, 3, 9 };//std::vector<int>(rowInd, rowInd + sizeof rowInd / sizeof rowInd[0]);
    mat.cooColInd = { 0, 2, 3, 1, 0, 2, 3, 1, 3, 9 };
    mat.cooValues = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };

    std::vector<double> hX = { 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f };
    std::vector<double> hY = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    std::vector<double> hY_result = { 19.0f, 8.0f, 51.0f, 52.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    smpv(mat, hX, hY); 
    mat_vect_mul(mat, hX, hY_result);

    for (int i = 0; i < mat.rows; i++)
        printf("%f %f\n", hY_result[i], hY[i]);

    */
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
    std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;

    prune_mnist_problem();
    //std::ifstream stream("train_caltech101.txt");

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}