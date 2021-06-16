import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.allMulInplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_float,
    ctypes.c_int
]

testcase = np.random.randn(10000).astype(np.float32) * 2.33 + 0.66
answer = testcase * 8.12138

gmm_matrix_support.allMulInplace(testcase, 8.12138, testcase.shape[0])

diff = np.abs(testcase - answer).max()
if diff < 1e-3:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
