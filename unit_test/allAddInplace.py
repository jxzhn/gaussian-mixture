import ctypes
import cupy

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.allAddInplace.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int
]

testcase = cupy.random.randn(60000) * 2.33 + 0.66
answer = testcase + 8.12138
gmm_matrix_support.allAddInplace(
    ctypes.cast(testcase.data.ptr, ctypes.POINTER(ctypes.c_double)),
    8.12138, 
    testcase.shape[0]
)
diff = cupy.abs(testcase - answer).max()
if diff < 1e-8:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))
