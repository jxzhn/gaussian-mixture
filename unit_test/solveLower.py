import ctypes
import numpy as np

gmm_matrix_support = ctypes.cdll.LoadLibrary('./libgmm_matrix_support.so')
# 设置参数类型
gmm_matrix_support.solveLower.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

testcase = np.random.randn(1000,1000).astype(np.float32) * 2.33 + 0.66
testcase
testcase = np.tril(testcase, k=0)

bmat = np.random.randn(500,1000).astype(np.float32) * 2.33 + 0.66

output = np.empty((500,1000), dtype=np.float32)

answer = np.linalg.solve(testcase,bmat.T)
# print((answer == answer).all())

gmm_matrix_support.solveLower(testcase, bmat, output, testcase.shape[0],bmat.shape[0])


diff = np.abs(output - answer.T).max()
if diff < 1e-3:
    print('test passed.')
else:
    print('test wrong! maximum difference: {:.6g}'.format(diff))

