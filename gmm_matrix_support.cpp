/**
 * @file gmm_matrix_support.cpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief 一些高斯混合模型会用到的线性代数函数实现
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include "gmm_matrix_support.h"

# include <math.h>

/**
 * @brief 求矩阵每一列的均值
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 每一列的均值结果，大小为 n
 * @param m 
 * @param n 
 */
void matColMean(const double* mat, double* buf, int m, int n) {
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < m; ++i) {
            sum += mat[i * n + j];
        }
        buf[j] = sum * (1.0 / m);
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
    double scale = 1.0 / (m - 1);
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            double covar = 0.0;
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
void dataAverageCovariance(const double* xSubMu, const double* weights, double* buf, int m, int dim) {
    double scale = 0.0;
    for (int k = 0; k < m; ++k) {
        scale += weights[k];
    }
    scale = 1.0 / (scale + 10 * __DBL_EPSILON__);
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            double covar = 0.0;
            for (int k = 0; k < m; ++k) {
                covar += weights[k] * xSubMu[k * dim + i] * xSubMu[k * dim + j];
            }
            buf[i * dim + j] = covar * scale;
        }
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
    for(int i = 0; i < dim; i++){
        mat[i * dim + i] += alpha;
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
void matCholesky(const double* mat, double* buf, int m){
    // 拷贝矩阵的下三角部分
    for(int i = 0; i < m; ++i) // rows
    {
        for(int j = 0; j <= i; ++j) // cols
        {
            buf[i * m + j] = mat[i * m + j];
        }
        for(int j = i + 1; j < m; ++j) // cols
        {
            buf[i * m + j] = 0.0;
        }
    }
    for(int k = 0; k < m; ++k) // cols
    {
        buf[k * m + k] = sqrt(buf[k * m + k] > 0.0 ? buf[k * m + k] : 0.0);

        for(int i = k + 1; i < m; ++i) // rows
        {
            buf[i * m + k] /= buf[k * m + k];

            for(int j = k + 1; j <= i; ++j)
            { 
                buf[i * m + j] -= buf[i * m + k] * buf[j * m + k];
            }
        }
    }
}

/**
 * @brief 计算一个方阵对角线上元素的对数（以 2 为底）之和
 * 
 * @param mat 矩阵，大小为 dim 行 dim 列
 * @param dim 
 * @return double 对角线上元素的对数之和
 */
double sumLog2Diag(const double* mat, int dim){
    double sum = 0.0;
    for(int i = 0; i < dim; i++){
        sum += log2(mat[i * dim + i]);
    }
    return sum;
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
void matVecRowSub(const double* mat, const double* vec, double* buf, int m, int n){
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
void rowSumSquare(const double* mat, double* buf, int m, int n){
    for(int i = 0; i < m; i++){
        buf[i] = 0.0;
        for(int j = 0; j < n; j++){
            buf[i] += mat[i * n + j] * mat[i * n + j];
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
void allAddInplace(double* arr, double alpha, int n){
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
void allMulInplace(double* arr, double alpha, int n){
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
void colLog2SumExp2(const double* mat, double* buf, int m, int n){
    // TODO: 这个 CPU 访存连续性不是很好，但 CUDA 应该要用这种方式
    for (int j = 0; j < n; j++) {
        double maximum = -INFINITY;
        for (int i = 0; i < m; i++) {
            if (mat[i * n + j] > maximum) {
                maximum = mat[i * n + j];
            }
        }
        buf[j] = maximum;
    }
    // 计算 logsumexp
    for (int j = 0; j < n; j++) {
        double res = 0.0;
        for (int i = 0; i < m; i++) {
            res += exp2(mat[i * n + j] - buf[j]);
        }
        buf[j] += log2(res);
    }
}

/**
 * @brief 对数组中所有元素取对数（以 2 为底）
 * 
 * @param arr 数组，大小为 n
 * @param buf 对数结果，大小为 n
 * @param n 
 */
void allLog2(const double* arr, double* buf, int n){
    for(int i = 0; i < n; i++){
        buf[i] = log2(arr[i]);
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
void matVecColAddInplace(double* mat, const double* vec, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            mat[i * n + j] += vec[i];
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
    for (int k = 0; k < n; ++k)
    {
        for (int i = 0; i < dim; ++i)
        {
            double val = b[k * dim + i];
            for (int j = 0; j < i; ++j)
            {
                val -= (lower[i * dim + j] * buf[k * dim + j]);
            }
            buf[k * dim + i] = val / lower[i * dim + i];
        }
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
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n + j] -= vec[j];
        }
    }
}

/**
 * @brief 对数组中所有元素取指数(以 2 为底）
 * 
 * @param arr 数组，大小为 n
 * @param n 
 */
void allExp2Inplace(double* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = exp2(arr[i]);
    }
}

/**
 * @brief 求数组中所有元素平均值
 * 
 * @param arr 数组，大小为 n
 * @param n 
 * @return double 所有元素的平均值
 */
double arrMean(double* arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum/n;
}

/**
 * @brief 计算矩阵各行的元素之和
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各行的元素之和，大小为 m
 * @param m 
 * @param n 
 */
void rowSum(const double* mat, double* buf, int m, int n) {
    for (int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += mat[i*n + j];
        }
        buf[i] = sum;
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
    for (int i = 0; i < m*k; i++) {
        buf[i] = 0.0;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int p = 0; p < n; p++) {
                buf[i*k + j] += mat1[i*n + p] * mat2[p*k + j];
            }
        }
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
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat[i*n + j] /= (alphas[i] + 10 * __DBL_EPSILON__);
        }
    }
}

/**
 * @brief 为数组中所有元素除以 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allDivInplace(double* arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] /= alpha;
    }    
}
