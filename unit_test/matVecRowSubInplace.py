import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.matVecRowSubInplace.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int
]

testcase = cupy.random.randn(1000, 784) * 2.33 + 0.66
vec = cupy.random.randn(784) * 2.33 + 0.66
answer = testcase - vec

gmm_matrix_support.matVecRowSubInplace(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    ctypes.cast(vec.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0],
    testcase.shape[1]
)

diff = cupy.abs(testcase - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
