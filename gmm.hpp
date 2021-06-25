/**
 * @file gmm.hpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief C++实现的高斯混合模型（头文件）
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

/**
 * @brief 高斯混合模型实现类
 */
class GaussianMixture {
private:
    static constexpr double minCovar = 1e-4f; // 方差下限（防止浮点溢出）

    int dim; // 数据维度
    double* weights; // 各个聚类的权重，大小为 n_component
    double* means; // 各个聚类的均值，大小为 n_component * dim
    double* covariances; // 各个聚类的协方差矩阵，大小为 n_compoment * dim * dim

    int nComponent; // 聚类数量
    double tol; // 收敛条件（对数似然值变化小于 tol）
    int maxIter; // 最大迭代次数

    bool memoryMalloced; // 判断内存是由构造函数创建的还是外部传进来的

    /**
     * @brief 初始化高斯混合模型的参数
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param numData 见上
     * @param xSubMuBuf 临时存放 xSubMu 的 buffer，大小为 numData 行 dim 列
     * @param meanBuf 临时存放 mean 的 buffer，大小为 dim
     * @param reduceBuf 并行规约需要的临时空间
     */
    void initParameter(const double* data, int numData, double* xSubMuBuf, double* meanBuf, double* reduceBuf);

    /**
     * @brief 计算所有数据对应各个聚类的对数概率密度
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param logDensity 对数概率密度输出，大小为 nComponent 行 numData 列
     * @param numData 见上
     * @param lowerMatBuf 临时存放 cholsky 分解得到的下三角矩阵的 buffer，大小为 dim 行 dim 列
     * @param xSubMuBuf 临时存放 x - mu 的 buffer，大小为 numData 行 dim 列
     * @param covSolBuf 临时存放 Ly = x - mu 的解的 buffer，大小为 numData 行 dim 列
     * @param reduceBuf 并行规约需要的临时空间
     */
    void logProbabilityDensity(const double* data, double* logDensity, int numData, double* lowerMatBuf, double* xSubMuBuf, double* covSolBuf, double* reduceBuf);

public:
    /**
     * @brief 构造一个高斯混合模型对象
     * 
     * @param dim 数据维度
     * @param nComponent 聚类数量
     * @param tol 收敛条件（对数似然值变化小于 tol）
     * @param maxIter 最大迭代次数
     */
    GaussianMixture(int dim, int nComponent, double tol = 1e-3, int maxIter = 100);

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
    GaussianMixture(int dim, int nComponent, double* weights, double* means, double* covariances, double tol = 1e-3, int maxIter = 100);

    /**
     * @brief 根据数据估计高斯混合模型参数
     * 
     * @param data 拟合数据，大小为 numData 行 dim 列
     * @param numData 见上
     */
    void fit(const double* data, int numData);

    /**
     * @brief 析构高斯混合模型对象，释放 malloc 的空间
     */
    virtual ~GaussianMixture();
};