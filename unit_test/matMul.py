import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matMul.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

testcase1 = cupy.random.randn(10, 60000) * 2.33 + 0.66
testcase2 = cupy.random.randn(60000, 784) * 2.33 + 0.66


answer = cupy.matmul(testcase1, testcase2)


output = cupy.empty((10, 784), dtype=cupy.float64)

gmm_matrix_support.matMul(
    ctypes.cast(testcase1.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(testcase2.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase1.shape[0],
    testcase1.shape[1],
    testcase2.shape[1]
)


diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))
