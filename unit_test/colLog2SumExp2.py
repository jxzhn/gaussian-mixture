import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.colLog2SumExp2.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]

testcase = cupy.random.rand(10, 60000) * 2.33 + 0.66

maximum = testcase.max(axis=0)
answer = cupy.log2(cupy.sum(cupy.exp2(testcase), axis=0))

output = cupy.empty(60000, dtype=cupy.float64)

gmm_matrix_support.colLog2SumExp2(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(output.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0],
    testcase.shape[1]
)

diff = cupy.abs(output - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff.item()))
