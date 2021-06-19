import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matMul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

testcase1 = np.random.randn(1000, 784).astype(np.float32) * 2.33 + 0.66
testcase2 = np.random.randn(784, 1000).astype(np.float32) * 2.33 + 0.66


answer = np.matmul(testcase1, testcase2)


output = np.empty((1000,1000), dtype=np.float32)

gmm_matrix_support.matMul(testcase1, testcase2, output, testcase1.shape[0], testcase1.shape[1], testcase2.shape[1])


diff = np.abs(output - answer).max()
if diff < 1e-3:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
