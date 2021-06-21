/**
 * @file gmm.hpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief C++实现的高斯混合模型（头文件）
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include <stdio.h>
# include <stdlib.h>
# include <sys/time.h>
# include <memory.h>
# include <math.h>
# include "gmm_matrix_support.h"

inline double wall_time() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

/**
 * @brief 高斯混合模型实现类
 */
class GaussianMixture {
private:
    static constexpr float minCovar = 1e-4f; // 方差下限（防止浮点溢出）

    int dim; // 数据维度
    float* weights; // 各个聚类的权重，大小为 n_component
    float* means; // 各个聚类的均值，大小为 n_component * dim
    float* covariances; // 各个聚类的协方差矩阵，大小为 n_compoment * dim * dim

    int nComponent; // 聚类数量
    float tol; // 收敛条件（对数似然值变化小于 tol）
    int maxIter; // 最大迭代次数

    bool memoryMalloced; // 判断内存是由构造函数创建的还是外部传进来的

    /**
     * @brief 使用随机法初始化高斯混合模型的参数
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param numData 见上
     */
    void initParameter(const float* data, int numData);

    /**
     * @brief 计算所有数据对应各个聚类的对数概率密度
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param logDensity 对数概率密度输出，大小为 nComponent 行 numData 列
     * @param numData 见上
     */
    void logProbabilityDensity(const float* data, float* logDensity, int numData);

public:
    /**
     * @brief 构造一个高斯混合模型对象
     * 
     * @param dim 数据维度
     * @param nComponent 聚类数量
     * @param tol 收敛条件（对数似然值变化小于 tol）
     * @param maxIter 最大迭代次数
     */
    GaussianMixture(int dim, int nComponent, float tol = 1e-3, int maxIter = 100);

    /**
     * @brief 构造一个高斯混合模型对象（使用外部指针）
     * 
     * @param dim 数据维度
     * @param nComponent 聚类数量
     * @param weights 聚类权重所使用的内存地址
     * @param means 聚类均值所使用的内存地址
     * @param covariances 聚类协方差所使用的内存地址
     * @param tol 收敛条件（对数似然值变化小于 tol）
     * @param maxIter 最大迭代次数
     */
    GaussianMixture(int dim, int nComponent, float* weights, float* means, float* covariances, float tol = 1e-3, int maxIter = 100);

    /**
     * @brief 根据数据估计高斯混合模型参数
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param numData 见上
     */
    void fit(const float* data, int numData);

    /**
     * @brief 析构高斯混合模型对象，释放 malloc 的空间
     */
    virtual ~GaussianMixture();
};