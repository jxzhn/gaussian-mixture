import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.arrSum.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int
]

gmm_matrix_support.arrSum.restype = ctypes.c_double

testcase = np.random.randn(10000).astype(np.float64) * 2.33 + 0.66
answer = np.sum(testcase)

output = gmm_matrix_support.arrSum(testcase,testcase.shape[0])

diff = np.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
