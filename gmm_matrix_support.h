/**
 * @file gmm_matrix_support.hpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief 一些高斯混合模型会用到的线性代数函数实现（头文件）
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# ifndef GMM_MATRIX_SUPORT_H_
# define GMM_MATRIX_SUPORT_H_


# ifdef GPU_VERSION

# ifdef __cplusplus
constexpr int BLOCK_DIM_1D = 256;
constexpr int BLOCK_DIM_2D = 16;
# else // __cplusplus
# define BLOCK_DIM_1D = 256;
# define BLOCK_DIM_2D = 16;
# endif // __cplusplus

# endif // GPU_VERSION


# ifdef __cplusplus
extern "C" {
# endif


/**
 * @brief 求矩阵每一列的均值
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 每一列的均值结果，大小为 n
 * @param m 
 * @param n 
 */
void matColMean(const double* mat, double* buf, int m, int n);

/**
 * @brief 求数据的协方差
 * 
 * @param data 按行逐个存放的数据（已减去均值），大小为 m 行 dim 列 
 * @param buf 协方差结果，大小为 dim 行 dim 列
 * @param m 
 * @param dim 
 */
void dataCovariance(const double* xSubMu, double* buf, int m, int dim);

/**
 * @brief 为方阵对角线上元素加上 alpha
 * 
 * @param mat 方阵，大小为 dim 行 dim 列
 * @param alpha 一个浮点数
 * @param dim 
 */
void matDiagAddInplace(double* mat, double alpha, int dim);

/**
 * @brief 对正定的对称方阵进行 Cholesky 分解
 * 
 * @param mat 正定的对称方阵，大小为 m 行 m 列
 * @param buf 下三角矩阵输出，大小为 m 行 m 列
 * @param m 
 * @param n 
 */
void matCholesky(const double* mat, double* buf, int m);

/**
 * @brief 计算一个方阵对角线上元素的对数（以 2 为底）之和
 * 
 * @param mat 矩阵，大小为 dim 行 dim 列
 * @param dim 
 * @return double 对角线上元素的对数之和
 */
double sumLog2Diag(const double* mat, int dim);

/**
 * @brief 矩阵向量按行减法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 1 行 n 列
 * @param buf 按行减法结果，大小为 m 行 n 列
 * @param m 
 * @param n 
 */
void matVecRowSub(const double* mat, const double* vec, double* buf, int m, int n);

/**
 * @brief 求解下三角线性方程组 Ly = b
 * 
 * @param lower 下三角矩阵 L，大小为 dim 行 dim 列
 * @param b n 个待求解的 b 组成的矩阵, 每行为一个 b 向量的转置（大小为 dim）
 * @param buf n 个解 y 组成的结果矩阵，每行为一个 y 向量的转置（大小为 dim）
 * @param dim 
 * @param n 
 */
void solveLower(const double* lower, const double* b, double* buf, int dim, int n);

/**
 * @brief 计算矩阵各行的元素平方之和
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各行元素平方之和结果，大小为 m
 * @param m 
 * @param n 
 */
void rowSumSquare(const double* mat, double* buf, int m, int n);

/**
 * @brief 为数组中所有元素加上 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allAddInplace(double* arr, double alpha, int n);

/**
 * @brief 为数组中所有元素乘上 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allMulInplace(double* arr, double alpha, int n);

/**
 * @brief 计算矩阵各列的元素的指数之和的对数（指数和对数均以 2 为底）
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各列元素的指数之和的对数结果，大小为 n
 * @param m 
 * @param n 
 */
void colLog2SumExp2(const double* mat, double* buf, int m, int n);

/**
 * @brief 对数组中所有元素取对数（以 2 为底）
 * 
 * @param arr 数组，大小为 n
 * @param buf 对数结果，大小为 n
 * @param n 
 */
void allLog2(const double* arr, double* buf, int n);

/**
 * @brief 矩阵向量原地按列加法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 m 行 1 列
 * @param m 
 * @param n 
 */
void matVecColAddInplace(double* mat, const double* vec, int m, int n);

/**
 * @brief 矩阵向量原地按行减法
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param vec 向量，大小为 1 行 n 列
 * @param m 
 * @param n 
 */
void matVecRowSubInplace(double* mat, const double* vec, int m, int n);

/**
 * @brief 对数组中所有元素取指数(以 2 为底）
 * 
 * @param arr 数组，大小为 n
 * @param n 
 */
void allExp2Inplace(double* arr, int n);


# ifdef GPU_VERSION

/**
 * @brief 求数组中所有元素平均值
 * 
 * @param arr 数组，大小为 n
 * @param n 
 * @param tmp 一个用来存储中间规约结果的临时数组，大小至少应为 (n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D
 * @return double 所有元素的平均值
 */
double arrMean(double* arr, int n, double* tmp);

# else // GPU_VERSION

/**
 * @brief 求数组中所有元素平均值
 * 
 * @param arr 数组，大小为 n
 * @param n 
 * @return double 所有元素的平均值
 */
double arrMean(double* arr, int n);

# endif // GPU_VERSION


/**
 * @brief 计算矩阵各行的元素之和
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param buf 各行的元素之和，大小为 m
 * @param m 
 * @param n 
 */
void rowSum(const double* mat, double* buf, int m, int n);

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
void matMul(const double* mat1, const double* mat2, double* buf, int m, int n, int k);

/**
 * @brief 矩阵原地各行除以各自的一个常数
 * 
 * @param mat 矩阵，大小为 m 行 n 列
 * @param alphas 各行对应的常数组成的数组，共 m 个常数
 * @param m 
 * @param n 
 */
void matPerRowDivInplace(double* mat, const double* alphas, int m, int n);

/**
 * @brief 为数组中所有元素除以 alpha
 * 
 * @param arr 数组，大小为 n
 * @param alpha 一个浮点数
 * @param n 
 */
void allDivInplace(double* arr, double alpha, int n);

/**
 * @brief 求数据的加权协方差
 * 
 * @param xSubMu 按行逐个存放的数据（已减去均值），大小为 m 行 dim 列
 * @param weights 数据对应的权重（未归一化），大小为 m
 * @param buf 协方差结果，大小为 dim 行 dim 列
 * @param m 
 * @param dim 
 */
void dataAverageCovariance(const double* xSubMu, const double* weights, double* buf, int m, int dim);

# ifdef __cplusplus
} // extern "C"
# endif

# endif