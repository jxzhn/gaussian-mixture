import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matPerRowDivInplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

testcase = np.random.randn(1000, 784) * 2.33 + 0.66
vec = np.random.randn(1000) * 2.33 + 0.66


answer = testcase.T / (vec + 10 * np.finfo(np.float64).eps)
answer = answer.T


gmm_matrix_support.matPerRowDivInplace(testcase, vec, testcase.shape[0], testcase.shape[1])


diff = np.abs(testcase - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
