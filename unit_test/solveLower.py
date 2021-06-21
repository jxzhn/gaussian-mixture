import ctypes
import numpy as np
import scipy.linalg

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.solveLower.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

X = np.random.randn(1000, 784) * 2.33 + 0.66
covarianceX = np.cov(X, rowvar=False)

testcase = np.ascontiguousarray(scipy.linalg.cholesky(covarianceX, lower=True))

answer = scipy.linalg.solve_triangular(testcase, X.T, lower=True).T

output = np.empty((1000, 784), dtype=np.float64)
gmm_matrix_support.solveLower(testcase, X, output, X.shape[1], X.shape[0])


diff = np.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))

