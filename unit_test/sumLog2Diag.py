import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.sumLog2Diag.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int
]
gmm_matrix_support.sumLog2Diag.restype = ctypes.c_double

testcase = np.random.rand(1000, 1000) * 2.33 + 0.66
answer = np.sum(np.log2(np.diag(testcase)))

output = gmm_matrix_support.sumLog2Diag(testcase, testcase.shape[0])

diff = np.abs(output - answer)
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
