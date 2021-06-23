import ctypes
import cupy
import cupyx.scipy.linalg

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.solveLower.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]

X = cupy.random.randn(60000, 784) * 2.33 + 0.66
covar = cupy.cov(X, rowvar=False)
L = cupy.linalg.cholesky(covar)

answer = cupyx.scipy.linalg.solve_triangular(L, X.T, lower=True).T

output = cupy.empty((60000, 784), dtype=cupy.float64)
gmm_matrix_support.solveLower(
    ctypes.cast(L.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(X.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    X.shape[1],
    X.shape[0]
)


diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))
