import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.allExp2Inplace.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]

testcase = cupy.random.rand(600000) * 2.33 + 0.66
answer = cupy.exp2(testcase)


gmm_matrix_support.allExp2Inplace(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    testcase.shape[0])

diff = cupy.abs(testcase - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
