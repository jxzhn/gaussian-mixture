/**
 * @file gmm_matrix_support.cpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief 一些高斯混合模型会用到的线性代数函数实现
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include "gmm_matrix_support.hpp"

/**
 * @brief 求矩阵每一列的均值
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 每一列的均值结果，大小为 n
 * @param m 
 * @param n 
 */
void matColMean(const float* mat, float* buf, int m, int n) {
    for (int j = 0; j < n; ++j) {
        buf[j] = 0.0;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            buf[j] += mat[i * n + j];
        }
    }
    for (int j = 0; j < n; ++j) {
        buf[j] *= (1.0 / m);
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
void dataCovariance(const float* xSubMu, float* buf, int m, int dim) {
    float scale = 1.0 / (m - 1);
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float covar = 0.0;
            for (int k = 0; k < m; ++k) {
                covar += xSubMu[k * dim + i] * xSubMu[k * dim + j];
            }
            buf[i * dim + j] = covar * scale;
        }
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
void dataAverageCovariance(const float* xSubMu, const float* weights, float* buf, int m, int dim) {
    float scale = 1.0 / (arrSum(weights, m) + 10 * __FLT_EPSILON__);
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float covar = 0.0;
            for (int k = 0; k < m; ++k) {
                covar += weights[k] * xSubMu[k * dim + i] * xSubMu[k * dim + j];
            }
            buf[i * dim + j] = covar * scale;
        }
    }
}