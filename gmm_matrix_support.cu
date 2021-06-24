/**
 * @file gmm_matrix_support.cu
 * @author jonson (jxzhn@jxzhn.com)
 * @brief 一些高斯混合模型会用到的线性代数函数实现
 * @version 0.1
 * @date 2021-06-22
 * @copyright Copyright (c) 2021
 */

# include "gmm_matrix_support.h"

# include <cuda_runtime.h>

# ifdef TIME_INFO
# include <stdio.h>
# include <sys/time.h>

inline double wall_time() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}
# endif

constexpr int BLOCK_DIM_1D = 256;
constexpr int BLOCK_DIM_2D = 16;

__global__ void dataAverageCovarianceKernel(const double* xSubMu, const double* weights, double* buf, int m, int dim) {
    int i = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    double covar0 = 0.0, covar1 = 0.0;
    double scale = 0.0;

    __shared__ double tile1[BLOCK_DIM_2D][BLOCK_DIM_2D];
    __shared__ double tile2[BLOCK_DIM_2D][BLOCK_DIM_2D];

    int nTiles = (m + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D;

    // 分块乘法
    for (int t = 0; t < nTiles; t++)
    {
        // 载入分块到共享内存，每个线程负责两个元素
        int x = t * BLOCK_DIM_2D + tx;

        tile1[ty][tx] = (i < dim && x < m) ? xSubMu[x * dim + i] : 0.0;
        tile1[ty + BLOCK_DIM_2D / 2][tx] = ((i + BLOCK_DIM_2D / 2) < dim && x < m) ? xSubMu[x * dim + (i + BLOCK_DIM_2D / 2)] : 0.0;
    
        int y = t * BLOCK_DIM_2D + ty;

        tile2[ty][tx] = (y < m && j < dim) ? xSubMu[y * dim + j] : 0.0;
        tile2[ty + BLOCK_DIM_2D / 2][tx] = ((y + BLOCK_DIM_2D / 2) < m && j < dim) ? xSubMu[(y + BLOCK_DIM_2D / 2) * dim + j] : 0.0;
        
        __syncthreads();

        // 计算分块乘积
        for (int k = 0; k < BLOCK_DIM_2D; k++)
        {
            int kAbs = t * BLOCK_DIM_2D + k;
            if (kAbs < m)
            {
                scale += weights[kAbs];
                covar0 += weights[kAbs] * tile1[ty][k] * tile2[k][tx];
                covar1 += weights[kAbs] * tile1[ty + BLOCK_DIM_2D / 2][k] * tile2[k][tx];
            }
        }

        __syncthreads();
    }

    // 写入计算结果
    if (i < dim && j < dim)
    {
        buf[i * dim + j] = covar0 / (scale + 10 * __DBL_EPSILON__);
        buf[(i + BLOCK_DIM_2D / 2) * dim + j] = covar1 / (scale + 10 * __DBL_EPSILON__);
    }
}

/**
 * @brief 求数据的加权协方差
 * 
 * @param xSubMu 按行逐个存放的数据（已减去均值），大小为 m 行 dim 列
 * @param weights 数据对应的权重（未归一化），大小为 m
 * @param buf 协方差结果，大小为 dim 行 dim 列
 * @param m 
 * @param dim 
 */
void dataAverageCovariance(const double* xSubMu, const double* weights, double* buf, int m, int dim) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    constexpr dim3 blockSize(BLOCK_DIM_2D, BLOCK_DIM_2D / 2);
    dim3 gridSize((dim + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (dim + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);

    dataAverageCovarianceKernel<<<gridSize, blockSize>>>(xSubMu, weights, buf, m, dim);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("dataAverageCovariance finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void dataCovarianceKernel(const double* xSubMu, double* buf, int m, int dim) {
    int i = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    double covar0 = 0.0, covar1 = 0.0;

    __shared__ double tile1[BLOCK_DIM_2D][BLOCK_DIM_2D];
    __shared__ double tile2[BLOCK_DIM_2D][BLOCK_DIM_2D];

    int nTiles = (m + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D;

    // 分块乘法
    for (int t = 0; t < nTiles; t++)
    {
        // 载入分块到共享内存，每个线程负责两个元素
        int x = t * BLOCK_DIM_2D + tx;

        tile1[ty][tx] = (i < dim && x < m) ? xSubMu[x * dim + i] : 0.0;
        tile1[ty + BLOCK_DIM_2D / 2][tx] = ((i + BLOCK_DIM_2D / 2) < dim && x < m) ? xSubMu[x * dim + (i + BLOCK_DIM_2D / 2)] : 0.0;

        int y = t * BLOCK_DIM_2D + ty;

        tile2[ty][tx] = (y < m && j < dim) ? xSubMu[y * dim + j] : 0.0;
        tile2[ty + BLOCK_DIM_2D / 2][tx] = ((y + BLOCK_DIM_2D / 2) < m && j < dim) ? xSubMu[(y + BLOCK_DIM_2D / 2) * dim + j] : 0.0;
        
        __syncthreads();

        // 计算分块乘积
        for (int k = 0; k < BLOCK_DIM_2D; k++)
        {
            covar0 += tile1[ty][k] * tile2[k][tx];
            covar1 += tile1[ty + BLOCK_DIM_2D / 2][k] * tile2[k][tx];
        }

        __syncthreads();
    }

    // 写入计算结果
    if (i < dim && j < dim)
    {
        buf[i * dim + j] = covar0 / (double)(m - 1);
        buf[(i + BLOCK_DIM_2D / 2) * dim + j] = covar1 / (double)(m - 1);
    }
}

/**
 * @brief 求数据的协方差
 * 
 * @param xSubMu 按行逐个存放的数据（已减去均值），大小为 m 行 dim 列 
 * @param buf 协方差结果，大小为 dim 行 dim 列
 * @param m 
 * @param dim 
 */
void dataCovariance(const double* xSubMu, double* buf, int m, int dim) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    constexpr dim3 blockSize(BLOCK_DIM_2D, BLOCK_DIM_2D / 2);
    dim3 gridSize((dim + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (dim + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);

    dataCovarianceKernel<<<gridSize, blockSize>>>(xSubMu, buf, m, dim);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("dataCovariance finished in %lf seconds.\n", t2 - t1);
# endif
}


__global__ void matCholeskyDiagKernel(const double* mat, double* buf, int m, int k) {
    // 这个核函数只能使用 1 个线程！！
    double sum = 0.0;
    for (int i = 0; i < k; i++)
    {
        sum += buf[k * m +i] * buf[k * m + i];
    }
    
    buf[k * m + k] = sqrt(mat[k * m + k] - sum);
}

__global__ void matCholeskyColumnKernel(const double* mat, double* buf, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < k)
    {
        buf[i * m + k] = 0.0;
    }
    else if (k < i && i < m)
    {  
        double sum = 0.0;
        for (int j = 0; j < k; j++)
        {
            sum += buf[i * m + j] * buf[k * m + j];
        }

        buf[i * m + k] = (mat[i * m + k] - sum) / buf[k * m + k];
    }
}

/**
 * @brief 对正定的对称方阵进行 Cholesky 分解
 * 
 * @param mat 正定的对称方阵，大小为 m 行 m 列
 * @param buf 下三角矩阵输出，大小为 m 行 m 列
 * @param m 
 * @param n 
 */
void matCholesky(const double* mat, double* buf, int m) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    for (int k = 0; k < m; k++)
    {
        matCholeskyDiagKernel<<<1, 1>>>(mat, buf, m, k);
        matCholeskyColumnKernel<<<m, 1>>>(mat, buf, m, k);
    }

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matCholesky finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void solveLowerKernel(const double* lower, const double* b, double* buf, int dim, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n)
    {
        for (int i = 0; i < dim; i++)
        {
            double val = b[k * dim + i];
            for (int j = 0; j < i; j++)
            {
                val -= lower[i * dim + j] * buf[k * dim + j];
            }

            buf[k * dim + i] = val / lower[i * dim + i];
        }
    }
}

/**
 * @brief 求解下三角线性方程组 Ly = b
 * 
 * @param lower 下三角矩阵 L，大小为 dim 行 dim 列
 * @param b n 个待求解的 b 组成的矩阵, 每行为一个 b 向量的转置（大小为 dim）
 * @param buf n 个解 y 组成的结果矩阵，每行为一个 y 向量的转置（大小为 dim）
 * @param dim 
 * @param n 
 */
void solveLower(const double* lower, const double* b, double* buf, int dim, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    solveLowerKernel<<<numBlocks, BLOCK_DIM_1D>>>(lower, b, buf, dim, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("solveLower finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matMulKernel(const double* mat1, const double* mat2, double* buf, int m, int n, int k) {
    int i = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    double val0 = 0.0, val1 = 0.0;

    __shared__ double matTile1[BLOCK_DIM_2D][BLOCK_DIM_2D];
    __shared__ double matTile2[BLOCK_DIM_2D][BLOCK_DIM_2D];

    int nTiles = (n + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D;

    // 分块乘法
    for (int t = 0; t < nTiles; t++)
    {
        // 载入分块到共享内存，一个线程负责两个位置
        int x = t * BLOCK_DIM_2D + tx;

        matTile1[ty][tx] = (i < m && x < n) ? mat1[i * n + x] : 0.0;
        matTile1[ty + BLOCK_DIM_2D / 2][tx] = ((i + BLOCK_DIM_2D / 2) < m && x < n) ? mat1[(i + BLOCK_DIM_2D / 2) * n + x] : 0.0;

        int y = t * BLOCK_DIM_2D + ty;

        matTile2[ty][tx] = (y < n && j < k) ? mat2[y * k + j] : 0.0;
        matTile2[ty + BLOCK_DIM_2D / 2][tx] = ((y + BLOCK_DIM_2D / 2) < n && j < k) ? mat2[(y + BLOCK_DIM_2D / 2) * k + j] : 0.0;

        __syncthreads();

        // 计算分块乘积
        for (int l = 0; l < BLOCK_DIM_2D; l++)
        {
            val0 += matTile1[ty][l] * matTile2[l][tx];
            val1 += matTile1[ty + BLOCK_DIM_2D / 2][l] * matTile2[l][tx];
        }

        __syncthreads();
    }

    // 写入计算结果
    if (i < m && j < k)
    {
        buf[i * k + j] = val0;
        buf[(i + BLOCK_DIM_2D / 2) * k + j] = val1;
    }
}


/**
 * @brief 矩阵乘法
 * 
 * @param mat1 矩阵 1，大小为 m 行 n 列
 * @param mat2 矩阵 2，大小为 n 行 k 列
 * @param buf 矩阵相乘结果，大小为 m 行 k 列
 * @param m 
 * @param n 
 * @param k 
 */
void matMul(const double* mat1, const double* mat2, double* buf, int m, int n, int k) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    constexpr dim3 blockSize(BLOCK_DIM_2D, BLOCK_DIM_2D / 2);
    dim3 gridSize((k + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D, (m + BLOCK_DIM_2D - 1) / BLOCK_DIM_2D);

    matMulKernel<<<gridSize, blockSize>>>(mat1, mat2, buf, m, n, k);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matMul finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void colLog2SumExp2Kernel(const double* mat, double* buf, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n)
    {
        // 先找出最大值，这样求指数的时候更不容易溢出
        double maximum = mat[j];
        for (int i = 1; i < m; i++)
        {
            if (mat[i * n + j] > maximum)
            {
                maximum = mat[i * n + j];
            }
        }

        double res = 0.0;
        for (int i = 0; i < m; i++)
        {
            res += exp2(mat[i * n + j] - maximum);
        }

        buf[j] = log2(res) + maximum;
    }
}

/**
 * @brief 计算矩阵各列的元素的指数之和的对数（指数和对数均以 2 为底）
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各列元素的指数之和的对数结果，大小为 n
 * @param m 
 * @param n 
 */
void colLog2SumExp2(const double* mat, double* buf, int m, int n){
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    colLog2SumExp2Kernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, buf, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("colLog2SumExp2 finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matDiagAddInplaceKernel(double * mat, double alpha, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < dim)
    {
        mat[i * dim + i] += alpha;
    }
}

/**
 * @brief 为方阵对角线上元素加上 alpha
 * 
 * @param mat 方阵，大小为 dim 行 dim 列
 * @param alpha 一个浮点数
 * @param dim 
 */
void matDiagAddInplace(double* mat, double alpha, int dim){
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    matDiagAddInplaceKernel<<<dim, 1>>>(mat, alpha, dim);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matDiagAddInplace finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matPerRowDivInplaceKernel(double* mat, const double* alphas, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n)
    {
        int i = index / n;
        int j = index % n;

        mat[i * n + j] /= (alphas[i] + 10 * __DBL_EPSILON__);
    }
}

/**
 * @brief 矩阵原地各行除以各自的一个常数
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param alphas 各行对应的常数组成的数组，共 m 个常数
 * @param m 
 * @param n 
 */
void matPerRowDivInplace(double* mat, const double* alphas, int m, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (m * n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    matPerRowDivInplaceKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, alphas, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matPerRowDivInplace finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matVecColAddInplaceKernel(double* mat, const double* vec, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n)
    {
        int i = index / n;
        int j = index % n;

        mat[i * n + j] += vec[i];
    }
}

/**
 * @brief 矩阵向量原地按列加法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 m 行 1 列
 * @param m 
 * @param n 
 */
void matVecColAddInplace(double* mat, const double* vec, int m, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (m * n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    matVecColAddInplaceKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, vec, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matVecColAddInplace finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matVecRowSubKernel(const double* mat, const double* vec, double* buf, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n)
    {
        int i = index / n;
        int j = index % n;

        buf[i * n + j] = mat[i * n + j] - vec[j];
    }
}

/**
 * @brief 矩阵向量按行减法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 1 行 n 列
 * @param buf 按行减法结果，大小为 m 行 n 列
 * @param m 
 * @param n 
 */
void matVecRowSub(const double* mat, const double* vec, double* buf, int m, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (m * n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    matVecRowSubKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, vec, buf, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matVecRowSub finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void matVecRowSubInplaceKernel(double* mat, const double* vec, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n)
    {
        int i = index / n;
        int j = index % n;

        mat[i * n + j] -= vec[j];
    }
}

/**
 * @brief 矩阵向量原地按行减法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 1 行 n 列
 * @param m 
 * @param n 
 */
void matVecRowSubInplace(double* mat, const double* vec, int m, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (m * n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    matVecRowSubInplaceKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, vec, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("matVecRowSubInplace finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void rowSumSquareKernel(const double* mat, double* buf, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double sum = 0.0;

    for (int j = 0; j < n; j++)
    {
        double a = mat[i * n + j];
        sum += a * a;
    }

    buf[i] = sum;
}

/**
 * @brief 计算矩阵各行的元素平方之和
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各行元素平方之和结果，大小为 m
 * @param m 
 * @param n 
 */
void rowSumSquare(const double* mat, double* buf, int m, int n) {
# ifdef TIME_INFO
    double t1 = wall_time();
# endif

    int numBlocks = (m + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    rowSumSquareKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, buf, m, n);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("rowSumSquare finished in %lf seconds.\n", t2 - t1);
# endif
}