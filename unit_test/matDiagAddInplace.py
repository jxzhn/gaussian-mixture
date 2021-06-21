import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matDiagAddInplace.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_double,
    ctypes.c_int
]

testcase = np.random.randn(1000, 1000) * 2.33 + 0.66
answer = testcase + np.eye(1000) * 8.12138

gmm_matrix_support.matDiagAddInplace(testcase, 8.12138, testcase.shape[0])

diff = np.abs(testcase - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
