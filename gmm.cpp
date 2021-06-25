/**
 * @file gmm.cpp
 * @author jonson (jxzhn@jxzhn.com)
 * @brief C++实现的高斯混合模型
 * @version 0.1
 * @date 2021-06-12
 * @copyright Copyright (c) 2021
 */

# include "gmm.hpp"
# include "gmm_matrix_support.h"
# include <stdio.h>
# include <sys/time.h>

inline double wall_time() {
    timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

# ifdef GPU_VERSION

# include <cuda_runtime.h>

# else // GPU_VERSION

# include <stdlib.h>
# include <memory.h>
# include <math.h>

# endif // GPU_VERSION

/**
 * @brief 构造一个高斯混合模型对象
 * 
 * @param dim 数据维度
 * @param nComponent 聚类数量
 * @param tol 收敛条件（对数似然值变化小于 tol）
 * @param maxIter 最大迭代次数
 */
GaussianMixture::GaussianMixture(int dim, int nComponent, double tol, int maxIter)
    : dim(dim), nComponent(nComponent), tol(tol), maxIter(maxIter), memoryMalloced(true)
{
# ifdef GPU_VERSION

    cudaMalloc(&this->weights, sizeof(double) * nComponent);
    cudaMalloc(&this->means, sizeof(double) * nComponent * dim);
    cudaMalloc(&this->covariances, sizeof(double) * nComponent * dim * dim);

# else // GPU_VERSION

    this->weights = (double*)malloc(sizeof(double) * nComponent);
    this->means = (double*)malloc(sizeof(double) * nComponent * dim);
    this->covariances = (double*)malloc(sizeof(double) * nComponent * dim * dim);

# endif // GPU_VERSION
}

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
GaussianMixture::GaussianMixture(int dim, int nComponent, double* weights, double* means, double* covariances, double tol, int maxIter)
: dim(dim), nComponent(nComponent), weights(weights), means(means), covariances(covariances), tol(tol), maxIter(maxIter), memoryMalloced(false)
{}

/**
 * @brief 初始化高斯混合模型的参数
 * 
 * @param data 拟合数据，大小为 numData 行 dim 列
 * @param numData 见上
 * @param xSubMuBuf 临时存放 xSubMu 的 buffer，大小为 numData 行 dim 列
 * @param meanBuf 临时存放 mean 的 buffer，大小为 dim
 * @param reduceBuf 并行规约需要的临时空间
 */
void GaussianMixture::initParameter(const double* data, int numData, double* xSubMuBuf, double* meanBuf, double* reduceBuf) {
    printf("initializing parameters\n");
    double t1 = wall_time();
    
# ifdef GPU_VERSION

    // 权重使用均匀分布初始化
    // 使用函数完成，方便操作 GPU 显存
    allMulInplace(this->weights, 0.0, this->nComponent);
    allAddInplace(this->weights, 1.0 / this->nComponent, this->nComponent);

    // 选择前 nComponent 个数据作为聚类均值 !! 注意，这里没使用随机法，最好把数据 shuffle 好
    cudaMemcpy(this->means, data, sizeof(double) * this->nComponent, cudaMemcpyDeviceToDevice);

    matColMean(data, meanBuf, numData, this->dim, reduceBuf);

# else // GPU_VERSION

    // 权重使用均匀分布初始化
    for (int c = 0; c < this->nComponent; ++c) {
        this->weights[c] = 1.0 / this->nComponent;
    }

    // 选择前 nComponent 个数据作为聚类均值 !! 注意，这里没使用随机法，最好把数据 shuffle 好
    memcpy(this->means, data, sizeof(double) * this->nComponent * this->dim);

    matColMean(data, meanBuf, numData, this->dim);

# endif // GPU_VERSION

    matVecRowSub(data, meanBuf, xSubMuBuf, numData, this->dim);
    // 使用所有数据的协方差初始化聚类协方差
    dataCovariance(xSubMuBuf, this->covariances, numData, this->dim);
    // 加上 minCovar 以保证最小方差
    matDiagAddInplace(this->covariances, this->minCovar, this->dim);

    for (int c = 1; c < this->nComponent; ++c) {
# ifdef GPU_VERSION
        cudaMemcpy(this->covariances + c * this->dim * this->dim, this->covariances, sizeof(double) * this->dim * this->dim, cudaMemcpyDeviceToDevice);
# else
        memcpy(this->covariances + c * this->dim * this->dim, this->covariances, sizeof(double) * this->dim * this->dim);
# endif
    }

    double t2 = wall_time();
    printf("initializing finished in %lf seconds\n", t2 - t1);
}

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
void GaussianMixture::logProbabilityDensity(const double* data, double* logDensity, int numData, double* lowerMatBuf, double* xSubMuBuf, double* covSolBuf, double* reduceBuf) {
    
    for (int c = 0; c < this->nComponent; ++c) {
        // 使用 cholesky 分解得到下三角矩阵
        matCholesky(this->covariances + c * this->dim * this->dim, lowerMatBuf, this->dim);

        // 协方差矩阵的行列式的对数等于 cholesky 分解的下三角矩阵对角线上元素的对数求和
# ifdef GPU_VERSION
        double covLogDet = 2 * sumLog2Diag(lowerMatBuf, this->dim, reduceBuf);
# else
        double covLogDet = 2 * sumLog2Diag(lowerMatBuf, this->dim);
# endif

        // 求解 y 满足 Ly = x - mu，则 (x - mu)^T Sigma^(-1) (x - mu) = y^T y
        matVecRowSub(data, this->means + c * this->dim, xSubMuBuf, numData, this->dim);
        solveLower(lowerMatBuf, xSubMuBuf, covSolBuf, this->dim, numData);

        // 计算概率密度
        double* logDensityOfComponent = logDensity + c * numData;
        rowSumSquare(covSolBuf, logDensityOfComponent, numData, this->dim);
        allAddInplace(logDensityOfComponent, this->dim * log2(2 * M_PI), numData);
        allAddInplace(logDensityOfComponent, covLogDet, numData);
        allMulInplace(logDensityOfComponent, -0.5, numData);
    }

}

/**
 * @brief 根据数据估计高斯混合模型参数
 * 
 * @param data 拟合数据，大小为 numData 行 dim 列
 * @param numData 见上
 */
void GaussianMixture::fit(const double* data, int numData) {
    // 分配一些各个函数会用到的 buffer
# ifdef GPU_VERSION

    double* xSubMu;
    cudaMalloc(&xSubMu, sizeof(double) * numData * this->dim);

    double* mean;
    cudaMalloc(&mean, sizeof(double) * this->dim);

    double* lowerMat, * covSol;
    cudaMalloc(&lowerMat, sizeof(double) * this->dim * this->dim);
    cudaMalloc(&covSol, sizeof(double) * numData * this->dim);

    double* logWeights, * logProb, * logProbSum, * responsibilities;
    cudaMalloc(&logWeights, sizeof(double) * this->nComponent);
    cudaMalloc(&logProb, sizeof(double) * this->nComponent * numData);
    cudaMalloc(&logProbSum, sizeof(double) * numData);
    responsibilities = logProb;

    // GPU 规约需要的空间
    double* reduceBuf;
    cudaMalloc(&reduceBuf, sizeof(double) * (this->dim > this->nComponent ? this->dim : this->nComponent) * (numData + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D);

# else // GPU_VERSION

    // xSubMu 各个函数都用了
    double* xSubMu = (double*)malloc(sizeof(double) * numData * this->dim);

    // initParameters
    double* mean = (double*)malloc(sizeof(double) * this->dim);

    // logProbabilityDensity
    double* lowerMat = (double*)malloc(sizeof(double) * this->dim * this->dim);
    double* covSol = (double*)malloc(sizeof(double) * numData * this->dim);

    // 聚类权重对数
    double* logWeights = (double*)malloc(sizeof(double) * this->nComponent);
    // 对数概率密度矩阵
    double* logProb = (double*)malloc(sizeof(double) * this->nComponent * numData);
    // logProbSum 用于保存上面矩阵的各列求和结果
    double* logProbSum = (double*)malloc(sizeof(double) * numData);
    // responsiblities 是簇分配结果，因为 logProb 和它不会同时用到，直接用一块空间就好了
    double* responsibilities = logProb;

    double* reduceBuf = nullptr;

# endif // GPU_VERSION


    this->initParameter(data, numData, xSubMu, mean, reduceBuf);


    // 对数似然值，比较两次迭代对数似然值变化用于判断迭代是否收敛
    double logLikelihood = INFINITY;

    for (int numIter = 0; numIter < this->maxIter; ++numIter) {
        double t1 = wall_time();

        double prevLogLikelihood = logLikelihood;

        // E 步
        this->logProbabilityDensity(data, logProb, numData, lowerMat, xSubMu, covSol, reduceBuf);
        // 概率密度乘上聚类权重，相当于对数相加
        allLog2(this->weights, logWeights, this->nComponent);
        matVecColAddInplace(logProb, logWeights, this->nComponent, numData);
        // 各列求和便于概率密度归一化
        colLog2SumExp2(logProb, logProbSum, this->nComponent, numData);
        // 簇分配（soft assignment），注意 responsibilities 和 logProb 是同一块空间，直接用原地算法
        matVecRowSubInplace(responsibilities, logProbSum, this->nComponent, numData);
        allExp2Inplace(responsibilities, this->nComponent * numData);

        // 计算对数似然值变化
# ifdef GPU_VERSION
        logLikelihood = arrMean(logProbSum, numData, reduceBuf);
# else
        logLikelihood = arrMean(logProbSum, numData);
# endif
        double diff = abs(logLikelihood - prevLogLikelihood);

        printf("iteration %d, log likelihood difference: %f\n", numIter, diff);

        // 判断是否收敛
        if (diff < this->tol) {
            break;
        }

        // M 步
        // 将簇分配结果按行求和，即得到每个簇被分配到的数据点数量
# ifdef GPU_VERSION
        rowSum(responsibilities, this->weights, this->nComponent, numData, reduceBuf);
# else
        rowSum(responsibilities, this->weights, this->nComponent, numData);
# endif
        // 将 responsibilities 和 data 做矩阵乘法，即得到 data 在不同聚类下由簇分配系数加权求和之和
        matMul(responsibilities, data, this->means, this->nComponent, numData, this->dim);
        // this->means 各行相应除以各聚类分配到的数据点数量，得到均值
        matPerRowDivInplace(this->means, this->weights, this->nComponent, this->dim);
        // 将 this->weights 归一化
        allDivInplace(this->weights, (double)numData, this->nComponent);

        // TODO: 这里可以并行计算，但 xSubMu 的内存分配每个聚类都要
        for (int c = 0; c < this->nComponent; ++c) {
            // resp 是样本点对当前聚类的簇分配结果
            double* resp = responsibilities + c * numData;
            matVecRowSub(data, this->means + c * this->dim, xSubMu, numData, this->dim);
            dataAverageCovariance(xSubMu, resp, this->covariances + c * this->dim * this->dim, numData, this->dim);
            // 加上 minCovar 以保证最小方差
            matDiagAddInplace(this->covariances + c * this->dim * this->dim, this->minCovar, this->dim);
        }

        double t2 = wall_time();
        printf("iteration %d finished in %lf seconds\n", numIter, t2 - t1);
    }

# ifdef GPU_VERSION

    cudaFree(xSubMu);
    
    cudaFree(mean);

    cudaFree(lowerMat);
    cudaFree(covSol);
    
    cudaFree(logWeights);
    cudaFree(logProb);
    cudaFree(logProbSum);

    cudaFree(reduceBuf);

# else // GPU_VERSION

    free(xSubMu);

    free(mean);

    free(lowerMat);
    free(covSol);

    free(logWeights);
    free(logProb);
    free(logProbSum);

# endif // GPU_VERSION
}

/**
 * @brief 析构高斯混合模型对象，释放 malloc 的空间
 */
GaussianMixture::~GaussianMixture() {
    if (this->memoryMalloced) {
# ifdef GPU_VERSION
        cudaFree(this->weights);
        cudaFree(this->means);
        cudaFree(this->covariances);
# else
        free(this->weights);
        free(this->means);
        free(this->covariances);
# endif
    }
}