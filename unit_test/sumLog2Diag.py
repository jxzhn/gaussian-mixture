import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.sumLog2Diag.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
gmm_matrix_support.sumLog2Diag.restype = ctypes.c_double

testcase = cupy.random.rand(1000, 1000) * 2.33 + 0.66
answer = cupy.sum(cupy.log2(cupy.diag(testcase)))

tmp = cupy.empty(1000, dtype=cupy.float64)
output = gmm_matrix_support.sumLog2Diag(ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)), testcase.shape[0], ctypes.cast(tmp.data.ptr, ctypes.POINTER(ctypes.c_double)))


diff = cupy.abs(output - answer)
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))
