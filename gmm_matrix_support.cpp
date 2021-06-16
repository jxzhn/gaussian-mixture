/**
 * @file gmm_matrix_support.cpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief 一些高斯混合模型会用到的线性代数函数实现
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include "gmm_matrix_support.h"

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
    float scale = 0.0;
    for (int k = 0; k < m; ++k) {
        scale += weights[k];
    }
    scale = 1.0 / (scale + 10 * __FLT_EPSILON__);
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



//TODO: 这里开始 by cc
/**
 * @brief 为方阵对角线上元素加上 alpha
 * 
 * @param mat 方阵，大小为 dim 行 dim 列
 * @param alpha 一个浮点数
 * @param dim 
 */
void matDiagAddInplace(float* mat, float alpha, int dim){
    for(int i = 0; i < dim; i++){
        mat[i * dim + i] += alpha;
    }
}


/**
 * @brief 对矩阵进行 Cholesky 分解
 * 
 * @param mat 矩阵，大小为 m 行 m 列
 * @param buf 下三角矩阵输出，大小为 m 行 m 列
 * @param m 
 * @param n 
 */
void matCholesky(const float* mat, float* buf, int m){
// TODO: 测试时只能输入对称正定矩阵
    for(int k = 0; k < m; k++){
        float sum = 0.0f;
        for(int i = 0; i < k; i++){
            sum += powf(buf[k * m + i], 2);
        }
        sum = mat[k * m + k] - sum;
        buf[k * m + k] = sqrtf(sum);
        for(int i = k + 1; i < m; i++){
            sum = 0.0f;
            for(int j = 0; j < k; j++){
                sum += buf[i * m + j] * buf[k * m + j];
            }
            buf[i * m + k] = (mat[i * m + k] - sum) / buf[k * m + k];
        }
        for(int j = 0; j < k; j++){
            buf[j * m + k] = 0;
        }
    }
}

/**
 * @brief 计算一个方阵对角线上元素的对数（以 2 为底）之和
 * 
 * @param mat 矩阵，大小为 dim 行 dim 列
 * @param dim 
 * @return float 对角线上元素的对数之和
 */
float sumLog2Diag(const float* mat, int dim){
    float sum = 0.0f;
    for(int i = 0; i < dim; i++){
        sum += log(mat[i * dim + i]);
    }
    return sum / log(2);
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
void matVecRowSub(const float* mat, const float* vec, float* buf, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            buf[i * n + j] = mat[i * n + j] - vec[j];
        }
    }
}


/**
 * @brief 计算矩阵各行的元素平方之和
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各行元素平方之和结果，大小为 m
 * @param m 
 * @param n 
 */
void rowSumSquare(const float* mat, float* buf, int m, int n){
    for(int i = 0; i < m; i++){
        buf[i] = 0.0f;
        for(int j = 0; j < n; j++){
            buf[i] += powf(mat[i * n + j], 2);
        }
    }
}

/**
 * @brief 为数组中所有元素加上 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allAddInplace(float* arr, float alpha, int n){
    for(int i = 0; i < n; i++){
        arr[i] += alpha;
    }
}

/**
 * @brief 为数组中所有元素乘上 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allMulInplace(float* arr, float alpha, int n){
     for(int i = 0; i < n; i++){
        arr[i] *= alpha;
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
void colLog2SumExp2(const float* mat, float* buf, int m, int n){
    //TODO:这是按行扫描的，如果要拆成并行的话就改成按列？
    float* max = (float*)malloc(sizeof(float) * n);
    memset(max, 0, sizeof(float) * n);
    //找每列最大值
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(mat[i * n + j] > max[j]){
                max[j] = mat[i * n + j];
            }
        }
    }
    //计算logsumexp
    memset(buf, 0, sizeof(float) * n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            buf[j] += powf(2, mat[i * n + j] - max[j]);
        }
    }
    for(int j = 0; j < n; j++){
        buf[j] = log(buf[j])/log(2) + max[j];
    }
}

/**
 * @brief 对数组中所有元素取对数（以 2 为底）
 * 
 * @param arr 数组，大小为 n
 * @param buf 对数结果，大小为 n
 * @param n 
 */
void allLog2(const float* arr, float* buf, int n){
    for(int i = 0; i < n; i++){
        buf[i] = log(arr[i]) / log(2);
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
void matVecColAddInplace(float* mat, const float* vec, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            mat[i * n + j] += vec[i];
        }
    }
}