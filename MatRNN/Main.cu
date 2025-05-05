#include "Chess.cuh"
#include "sqlite3.h"

std::vector<Vector> inp;
Vector res;

static int callback(void* data, int argc, char** argv, char** azColName) {
    int i;
    //fprintf(stderr, "%s: ", (const char*)data);

    /*for (i = 0; i < argc; i++) {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");

    }*/

    inp.push_back(chess_board(argv[1]).board_to_ai_inp());
    res.push_back(std::stof(argv[3]));
   // std::cout << inp.size() << " " << res.size() << std::endl;
    //printf("\n");
    return 0;
}
void chess_problem()
{
    sqlite3* db;
    // Save the connection result
    int exit = 0;
    exit = sqlite3_open("D:/test.db", &db);

    char* zErrMsg = 0;
    const char* data = "Callback function called";

    // Test if there was an error
    if (exit)
        std::cout << "DB Open Error: " << sqlite3_errmsg(db) << std::endl;
    else
        std::cout << "Opened Database Successfully!" << std::endl;

    /* Create merged SQL statement */
    auto sql = "SELECT * from evaluations LIMIT 100000";

    /* Execute SQL statement */
    auto rc = sqlite3_exec(db, sql, callback, (void*)data, &zErrMsg);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    }
    else {
        fprintf(stdout, "Operation done successfully\n");
    }
    sqlite3_close(db);

    std::cout << inp.size() << " " << res.size()<<std::endl;

    prune_chess_problem(inp, res);

}

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

    Vector hX = { 1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f };
    Vector hY = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    Vector hY_result = { 19.0f, 8.0f, 51.0f, 52.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    smpv(mat, hX, hY); 
    mat_vect_mul(mat, hX, hY_result);

    for (int i = 0; i < mat.rows; i++)
        printf("%f %f\n", hY_result[i], hY[i]);

    */
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
    std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;
    //RNN_mnist_autoencode_problem();
    prune_mnist_problem();
    //prune_caltech_problem();
    //chess_problem();
    //test_fen();
    //std::ifstream stream("train_caltech101.txt");

    cudaStatus = cudaDeviceReset();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}