# Guassian Mixture Model

## 优化过程记录：dataAverageCovariance

测试规模：60000 x 784

### CPU 版本：

```c++
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
```

运行时间：295.033056s

### CUDA V1

使用一维线程块，每个线程负责计算输出矩阵的一个元素

```c++
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
```

运行时间：0.223474s
