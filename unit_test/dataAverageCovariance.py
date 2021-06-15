import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.dataAverageCovariance.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

testcase = np.random.randn(1000, 784).astype(np.float32) * 2.33 + 0.66
weights = np.random.random(1000).astype(np.float32)

xSubMu = testcase - testcase.mean(axis=0)
answer = np.dot(weights * xSubMu.T, xSubMu) / (weights.sum() + 10 * np.finfo(np.float32).eps)

output = np.empty((784, 784), dtype=np.float32)

gmm_matrix_support.dataAverageCovariance(xSubMu, weights, output, xSubMu.shape[0], xSubMu.shape[1])

diff = np.abs(output - answer).max()
if diff < 1e-5:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))