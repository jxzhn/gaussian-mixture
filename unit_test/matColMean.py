import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matColMean.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
row = 60000
col = 784
testcase = cupy.random.randn(row, col) * 2.33 + 0.66
# print(testcase)
output = cupy.empty(col, dtype=cupy.float64)
tmp = cupy.empty(row*col, dtype=cupy.float64)
gmm_matrix_support.matColMean(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0], 
    testcase.shape[1],
    ctypes.cast(tmp.data.ptr, ctypes.POINTER(ctypes.c_double)),
)
diff = cupy.abs(output - testcase.mean(axis=0)).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
