import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve



class DiscretePoissonSolver:
    def __init__(self, omega, max_iter, tol):
        self.omega = omega
        self.max_iter = max_iter
        self.tol = tol


    def gauss_seidel_sor(A, b, omega, x0, max_iter=100, tol=1e-6):
        '''Solve equation (7) using Gauss-Seidel iteration with successive overrelaxation.'''
        n = A.shape[0]
        x = x0.copy()
        num_iter = 0
        residual = np.inf

        while num_iter < max_iter and residual > tol:
            x_old = x.copy()
            for i in range(n):
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                row_values = A.data[row_start:row_end]
                row_indices = A.indices[row_start:row_end]

                # 计算Ax中除去对角元素的部分
                off_diagonal_sum = np.dot(row_values, x[row_indices])

                # 更新解向量x的第i个分量
                x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - off_diagonal_sum)

            residual = np.linalg.norm(A.dot(x) - b)
            num_iter += 1

        return x, num_iter

# 示例用法

# 打印结果
# print("Solution:")
# print(x)
# print("Number of iterations:", num_iter)
