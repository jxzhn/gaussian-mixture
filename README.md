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

**运行时间：295.033056s**

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

**运行时间：0.223474s**

## 优化过程记录：dataCovariance

这个跟上面的是基本上是一样的

测试规模：60000 x 784

### CPU 版本

```c++
void dataCovariance(const double* xSubMu, double* buf, int m, int dim) {
    double scale = 1.0 / (m - 1);
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
```

**运行时间：286.136057s**

### CUDA V1

```c++
__global__ void dataCovarianceKernel(const double* xSubMu, double* buf, int m, int dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < dim * dim) {
        int i = index / dim;
        int j = index % dim;

        double covar = 0.0;
        
        for (int k = 0; k < m; k++)
        {
            covar += xSubMu[k * dim + i] * xSubMu[k * dim + j];
        }

        buf[i * dim + j] = covar  / (double)(m - 1);
    }
}
```

**运行时间：0.242587s**

## 优化过程记录：matCholesky

原理参考：[矩阵的 Cholesky 分解](https://blog.csdn.net/baimafujinji/article/details/6512460)

在 MNIST 数据集中的实际规模是 784 x 784，为了凸显区别，这里使用 4096 x 4096 进行测试。

### CPU 版本

```c++
void matCholesky(const double* mat, double* buf, int m){
    for(int k = 0; k < m; k++) {
        double sum = 0.0;

        for(int i = 0; i < k; i++) {
            sum += buf[k * m + i] * buf[k * m + i];
        }
        buf[k * m + k] = sqrt(mat[k * m + k] - sum);
        
        for(int i = k + 1; i < m; i++) {
            sum = 0.0;
            for(int j = 0; j < k; j++) {
                sum += buf[i * m + j] * buf[k * m + j];
            }

            buf[i * m + k] = (mat[i * m + k] - sum) / buf[k * m + k];
        }
        
        for(int j = 0; j < k; j++) {
            buf[j * m + k] = 0;
        }
    }
}
```

**运行时间：14.647272s**

### CUDA V1

Cholesky 分解的算法中，最外层的逐列计算循环之间存在数据依赖。另外，计算每一列对角线下方元素时，该列对角线上的元素需要已经算好。

考虑到 CUDA 核函数中没有全局同步的机制，我使用了两个核函数分别计算对角线上的元素和对角线下的元素，然后在 CPU 上使用 `for` 循环逐列调用核函数计算。

核函数：

```c++
__global__ void matCholeskyDiagKernel(const double* mat, double* buf, int m, int k) {
    // 这个核函数只能使用 1 个线程！！
    double sum = 0.0;
    for (int i = 0; i < k; i++)
    {
        sum += buf[k * m +i] * buf[k * m + i];
    }
    
    buf[k * m + k] = sqrt(mat[k * m + k] - sum);
}

__global__ void matCholeskyColumnKernel(const double* mat, double* buf, int m, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < k)
    {
        buf[i * m + k] = 0.0;
    }
    else if (k < i && i < m)
    {  
        double sum = 0.0;
        for (int j = 0; j < k; j++)
        {
            sum += buf[i * m + j] * buf[k * m + j];
        }

        buf[i * m + k] = (mat[i * m + k] - sum) / buf[k * m + k];
    }
}
```

CPU 代码：

```c++
int numBlocks = (m + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;

for (int k = 0; k < m; k++)
{
    matCholeskyDiagKernel<<<1, 1>>>(mat, buf, m, k);
    matCholeskyColumnKernel<<<numBlocks, BLOCK_DIM_1D>>>(mat, buf, m, k);
}
```

**运行时间：1.819354s**

### CUDA V2

上面的核函数 `matCholeskyColumnKernel` 中存在一个 `for` 循环，在该 `for` 循环中同一个 warp 中的线程会访问矩阵不同行的（同一列）数据，访存事务无法合并，只能串行访存。然而，CUDA 的不同线程块间是可以并行访存的（通过调度掩盖延迟），如果我们将线程分散到不同的线程块中，串行访存带来的开销将能够被有效减小。

修改 CPU 代码为：

```c++
for (int k = 0; k < m; k++)
{
    matCholeskyDiagKernel<<<1, 1>>>(mat, buf, m, k);
    matCholeskyColumnKernel<<<m, 1>>>(mat, buf, m, k);
}
```

**运行时间：1.059844s**

## 优化过程记录：solveLower

测试规模：784 x 784, 60000 x 784

### CPU 版本

```c++
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
```

**运行时间：20.320882s**

### CUDA V1

每个线程计算一个向量

```c++
__global__ void solveLowerKernel(const double* lower, const double* b, double* buf, int dim, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n)
    {
        for (int i = 0; i < dim; i++)
        {
            double val = b[k * dim + i];
            for (int j = 0; j < i; j++)
            {
                val -= lower[i * dim + j] * buf[k * dim + j];
            }

            buf[k * dim + i] = val / lower[i * dim + i];
        }
    }
}
```

**运行时间：0.831100s**

因为输出 buf 还要不断读进来，本来想用共享内存优化一下，但仔细一想共享内存只有 48kB，一个 784 维的双精度向量就是 6.125kB 了，可能不太行。

## 优化过程记录