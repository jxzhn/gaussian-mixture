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
