import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from scipy.sparse import linalg as splinalg


def solve(A: csr_matrix, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    num_channels = b.shape[-1]
    assert A.shape == (n, n), f"{A.shape=}"
    assert b.shape == (n, num_channels), f"{b.shape=}"
    assert isspmatrix_csr(A)
    print("Solving the linear system ...")
    # solve over all channels
    f = np.zeros((n, num_channels), dtype=np.float32)
    error = np.zeros(num_channels)
    for i in range(b.shape[-1]):
        f_i = splinalg.spsolve(A, b[:, i])
        f[:, i] = f_i
        error[i] = np.linalg.norm(A @ f_i - b[:, i])

    f[f < 0] = 0
    f[f > 255] = 255

    return f, error
