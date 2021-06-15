/**
 * @file gmm.cpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief C++实现的高斯混合模型
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include "gmm.hpp"

/**
 * @brief 构造一个高斯混合模型对象
 * 
 * @param dim 数据维度
 * @param nComponent 聚类数量
 * @param tol 收敛条件（对数似然值变化小于 tol）
 * @param maxIter 最大迭代次数
 */
GaussianMixture::GaussianMixture(int dim, int nComponent, float tol = 1e-3, int maxIter = 100)
    : dim(dim), nComponent(nComponent), tol(tol), maxIter(maxIter)
{
    this->weights = (float*)malloc(sizeof(float) * nComponent);
    this->means = (float*)malloc(sizeof(float) * nComponent * dim);
    this->covariances = (float*)malloc(sizeof(float) * nComponent * dim * dim);
}

/**
 * @brief 初始化高斯混合模型的参数
 * 
 * @param data 拟合数据，大小为 numData 行 dim 列
 * @param numData 见上
 */
void GaussianMixture::initParameter(const float* data, int numData) {
    // 权重使用均匀分布初始化
    for (int c = 0; c < this->nComponent; ++c) {
        this->weights[c] = 1.0 / this->nComponent;
    }

    // 选择前 nComponent 个数据作为聚类均值 !! 注意，这里没使用随机法，最好把数据 shuffle 好
    memcpy(this->means, data, sizeof(float) * this->nComponent * this->dim);

    // TODO: 使用事先分配的 buffer 优化 malloc
    float* mean = (float*)malloc(sizeof(float) * this->dim);
    float* xSubMu = (float*)malloc(sizeof(float) * numData * this->dim);

    matColMean(data, mean, numData, this->dim);
    matVecRowSub(data, mean, xSubMu, numData, this->dim);
    // 使用所有数据的协方差初始化聚类协方差
    dataCovariance(xSubMu, this->covariances, numData, this->dim);
    // 加上 minCovar 以保证最小方差
    matDiagAddInplace(this->covariances, this->minCovar, this->dim);

    for (int c = 1; c < this->nComponent; ++c) {
        memcpy(this->covariances + c * this->dim * this->dim, this->covariances, sizeof(float) * this->dim * this->dim);
    }

    free(mean);
    free(xSubMu);
}

/**
 * @brief 计算所有数据对应各个聚类的对数概率密度
 * 
 * @param data 拟合数据，大小为 numData 行 dim 列
 * @param logDensity 对数概率密度输出，大小为 nComponent 行 numData 列
 * @param numData 见上
 */
void GaussianMixture::logProbabilityDensity(const float* data, float* logDensity, int numData) {
    // TODO: 使用事先分配的 buffer 优化 malloc

    // lowerMat 用来临时保存 cholsky 分解得到的下三角矩阵
    float* lowerMat = (float*)malloc(sizeof(float) * this->dim * this->dim);
    // xSubMu 用来保存 x - mu
    float* xSubMu = (float*)malloc(sizeof(float) * numData * this->dim);
    // covSol 用来临时保存 Ly = x - mu 的解
    float* covSol = (float*)malloc(sizeof(float) * numData * this->dim);

    // TODO: 这里可以并行计算，上面的中间内存分配每个聚类都要
    for (int c = 0; c < this->nComponent; ++c) {
        // 使用 cholesky 分解得到下三角矩阵
        matCholesky(this->covariances + c * this->dim * this->dim, lowerMat, this->dim);

        // 协方差矩阵的行列式的对数等于 cholesky 分解的下三角矩阵对角线上元素的对数求和
        float covLogDet = sumLog2Diag(lowerMat, this->dim);

        // 求解 y 满足 Ly = x - mu，则 (x - mu)^T Sigma^(-1) (x - mu) = y^T y
        matVecRowSub(data, this->means + c * this->dim, xSubMu, numData, this->dim);
        solveLower(lowerMat, xSubMu, covSol, this->dim, numData);

        // 计算概率密度
        float* logDensityOfComponent = logDensity + c * numData;
        rowSumSquare(covSol, logDensityOfComponent, numData, this->dim);
        allAddInplace(logDensityOfComponent, this->dim * (1 + log2f(M_PI)), numData);
        allAddInplace(logDensityOfComponent, covLogDet, numData);
        allMulInplace(logDensityOfComponent, -0.5, numData);
    }

    free(lowerMat);
    free(xSubMu);
    free(covSol);
}

/**
 * @brief 根据数据估计高斯混合模型参数
 * 
 * @param data 拟合数据，大小为 numData 行 dim 列
 * @param numData 见上
 */
void GaussianMixture::fit(const float* data, int numData) {
    this->initParameter(data, numData);

    // 聚类权重对数
    float* logWeights = (float*)malloc(sizeof(float) * this->nComponent);
    // 对数概率密度矩阵
    float* logProb = (float*)malloc(sizeof(float) * this->nComponent * numData);
    // logProbSum 用于保存上面矩阵的各列求和结果
    float* logProbSum = (float*)malloc(sizeof(float) * numData);
    // responsiblities 是簇分配结果，因为 logProb 和它不会同时用到，直接用一块空间就好了
    float* responsibilities = logProb;
    // xSubMu 在 M 步中计算协方差矩阵时需要使用
    float* xSubMu = (float*)malloc(sizeof(float) * numData * this->dim);

    // 对数似然值，比较两次迭代对数似然值变化用于判断迭代是否收敛
    float logLikelihood = INFINITY;

    for (int numIter = 0; numIter < this->maxIter; ++numIter) {
        float prevLogLikelihood = logLikelihood;

        // E 步
        this->logProbabilityDensity(data, logProb, numData);
        // 概率密度乘上聚类权重，相当于对数相加
        allLog2(this->weights, logWeights, this->nComponent);
        matVecColAddInplace(logProb, logWeights, this->nComponent, numData);
        // 各列求和便于概率密度归一化
        colLog2SumExp2(logProb, logProbSum, this->nComponent, numData);
        // 簇分配（soft assignment），注意 responsibilities 和 logProb 是同一块空间，直接用原地算法
        matVecRowSubInplace(responsibilities, logProbSum, this->nComponent, numData);
        allExp2Inplace(responsibilities, this->nComponent * numData);

        // 计算对数似然值变化
        logLikelihood = arrMean(logProbSum, numData);
        float diff = abs(logLikelihood - prevLogLikelihood);

        printf("iteration %d, log likelihood difference: %f\n", numIter, diff);

        // 判断是否收敛
        if (diff < this->tol) {
            break;
        }

        // M 步
        // 将簇分配结果按行求和，即得到每个簇被分配到的数据点数量
        rowSum(responsibilities, this->weights, this->nComponent, numData);
        // 将 responsibilities 和 data 做矩阵乘法，即得到 data 在不同聚类下由簇分配系数加权求和之和
        matMul(responsibilities, data, this->means, this->nComponent, numData, this->dim);
        // this->means 各行相应除以各聚类分配到的数据点数量，得到均值
        matPerRowDivInplace(this->means, this->weights, this->nComponent, this->dim);

        // TODO: 这里可以并行计算，但 xSubMu 的内存分配每个聚类都要
        for (int c = 0; c < this->nComponent; ++c) {
            // 将 this->weights 归一化
            this->weights[c] /= (float)numData;

            // resp 是样本点对当前聚类的簇分配结果
            float* resp = responsibilities + c * numData;
            matVecRowSub(data, this->means + c * this->dim, xSubMu, numData, this->dim);
            dataAverageCovariance(xSubMu, resp, this->covariances + c * this->dim * this->dim, numData, this->dim);
            // 加上 minCovar 以保证最小方差
            matDiagAddInplace(this->covariances + c * this->dim * this->dim, this->minCovar, this->dim);
        }
    }

    free(logWeights);
    free(logProb);
    free(logProbSum);
    free(xSubMu);
}

/**
 * @brief 析构高斯混合模型对象，释放 malloc 的空间
 */
GaussianMixture::~GaussianMixture() {
    free(this->weights);
    free(this->means);
    free(this->covariances);
}