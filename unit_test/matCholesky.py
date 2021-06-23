import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matCholesky.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]

X = cupy.random.randn(5000, 4096) * 2.33 + 0.66
testcase = X.T @ X
answer = cupy.linalg.cholesky(testcase)

output = cupy.random.random((4096, 4096), dtype=cupy.float64)

gmm_matrix_support.matCholesky(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0]
)

diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))
