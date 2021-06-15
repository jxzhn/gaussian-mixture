import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matColMean.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

testcase = np.random.randn(1000, 784).astype(np.float32) * 2.33 + 0.66
output = np.empty(784, dtype=np.float32)

gmm_matrix_support.matColMean(testcase, output, testcase.shape[0], testcase.shape[1])

diff = np.abs(output - testcase.mean(axis=0)).max()
if diff < 1e-5:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
