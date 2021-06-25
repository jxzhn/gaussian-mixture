import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.rowSum.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
row = 10
col = 60000
testcase = cupy.random.randn(row, col) * 2.33 + 0.66
output = cupy.empty(row, dtype=cupy.float64)
answer = cupy.sum(testcase,axis=1)
tmp = cupy.empty(row*col, dtype=cupy.float64)

gmm_matrix_support.rowSum(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0], 
    testcase.shape[1],
    ctypes.cast(tmp.data.ptr, ctypes.POINTER(ctypes.c_double)),
)
# print(testcase)
# print(output)
# print(answer)
diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
