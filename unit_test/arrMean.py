import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.arrMean.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

gmm_matrix_support.arrMean.restype = ctypes.c_double

testcase = cupy.random.randn(60000000) * 2.33 + 0.66
tmp = cupy.random.randn(256000) * 2.33 + 0.66
answer = cupy.mean(testcase)
output = gmm_matrix_support.arrMean(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0],
    ctypes.cast(tmp.data.ptr, ctypes.POINTER(ctypes.c_double))
)
diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
