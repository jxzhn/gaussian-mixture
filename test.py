import ctypes
import ctypes
import numpy as np

test = ctypes.cdll.LoadLibrary('./build/libtest.so')
# 设置参数类型
test.gmmFit.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_int
]

# # 生成 2 个三维高斯分布的数据
# dist1_mean = np.array([-1, 1, -1], dtype=np.float32)
# dist1_covar = np.array(
#     [[ 3, -2,  0],
#     [-2,  2,  0],
#     [ 0,  0,  2]]
# , dtype=np.float32)
# dist1_data = np.random.multivariate_normal(dist1_mean, dist1_covar, size=7000).astype(np.float32)

# dist2_mean = np.array([2, -1.5, 3], dtype=np.float32)
# dist2_covar = np.array(
#     [[ 3,  1, -5],
#     [ 1,  1, -1],
#     [-5, -1, 10]]
# , dtype=np.float32)
# dist2_data = np.random.multivariate_normal(dist2_mean, dist2_covar, size=3000).astype(np.float32)

# # 权重是 7:3
# data = np.concatenate([dist1_data, dist2_data])
# np.random.shuffle(data)

# weights = np.empty(2, dtype=np.float32)
# means = np.empty((2, 3), dtype=np.float32)
# covariances = np.empty((2, 3, 3), dtype=np.float32)

# test.gmmFit(data, weights, means, covariances, data.shape[0], data.shape[1], 2, 1e-4, 300)

data = np.random.randn(50000, 784).astype(np.float32)

weights = np.empty(10, dtype=np.float32)
means = np.empty((10, 784), dtype=np.float32)
covariances = np.empty((10, 784, 784), dtype=np.float32)

test.gmmFit(data, weights, means, covariances, data.shape[0], data.shape[1], 10, 0.0, 100)

print(weights)
print(means)
print(covariances)