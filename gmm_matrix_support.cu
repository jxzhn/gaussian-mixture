/**
 * @file gmm_matrix_support.cpp
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

__global__ void dataAverageCovarianceKernel(const double* xSubMu, const double* weights, double* buf, int m, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < dim * dim) {
        int i = index / dim;
        int j = index % dim;

        double scale = 0.0;
        double covar = 0.0;
        
        for (int k = 0; k < m; k++)
        {
            scale += weights[k];
            covar += weights[k] * xSubMu[k * dim + i] * xSubMu[k * dim + j];
        }

        buf[i * dim + j] = covar  / (scale + 10 * __DBL_EPSILON__);
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

    int numBlocks = (dim * dim + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    dataAverageCovarianceKernel<<<numBlocks, BLOCK_DIM_1D>>>(xSubMu, weights, buf, m, dim);

# ifdef TIME_INFO
    cudaDeviceSynchronize();

    double t2 = wall_time();
    printf("dataAverageCovariance finished in %lf seconds.\n", t2 - t1);
# endif
}

__global__ void dataCovarianceKernel(const double* xSubMu, double* buf, int m, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < dim * dim) {
        int i = index / dim;
        int j = index % dim;

        double covar = 0.0;
        
        for (int k = 0; k < m; k++)
        {
            covar += xSubMu[k * dim + i] * xSubMu[k * dim + j];
        }

        buf[i * dim + j] = covar  / (double)(m - 1);
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

    int numBlocks = (dim * dim + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
    dataCovarianceKernel<<<numBlocks, BLOCK_DIM_1D>>>(xSubMu, buf, m, dim);

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