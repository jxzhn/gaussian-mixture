import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.dataCovariance.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]

testcase = cupy.random.randn(60000, 784) * 2.33 + 0.66
answer = cupy.cov(testcase, rowvar=False)

xSubMu = testcase - testcase.mean(axis=0)
output = cupy.empty((784, 784), dtype=cupy.float64)

gmm_matrix_support.dataCovariance(
    ctypes.cast(xSubMu.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    xSubMu.shape[0],
    xSubMu.shape[1]
)

diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))