from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix

"""
In the reference paper, we have a discrete Poisson equation:

    |N_p| f_p - \sum_{q \in N_p \cap \Omega} f_q = \
    \sum_{q \in N_p \cap \partial \Omega} f^{\star}_q + \sum_{q \in N_p} v_{pq}

Where `f` is the unknown variable, we need to compute following matrix/vector:
1. A: Index matrix of |N_p| -  \sum_{q \in N_p \cap \Omega}
2. b: \sum_{q \in N_p \cap \partial \Omega} f^{\star}_q + \sum_{q \in N_p} v_{pq}
Then we can solve the equation `A * f = b`.
"""


def generate_A(
    mask: np.ndarray, mask_indices: np.ndarray, index_to_id: np.ndarray
) -> csr_matrix:
    N_p = generate_neighbour_size(mask, mask_indices)
    A_omega = generate_neighbour_omega(
        mask, mask_indices, index_to_id
    )

    A = N_p - A_omega

    return A


def generate_neighbour_size(mask: np.ndarray, mask_indices: np.ndarray) -> csr_matrix:
    n = mask_indices.shape[0]
    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]

    N_up = (x_indices + 1) < mask.shape[0]
    N_down = (x_indices - 1) >= 0
    N_left = (y_indices - 1) >= 0
    N_right = (y_indices + 1) < mask.shape[1]

    N_p_data = (
        N_up.astype(np.int32)
        + N_down.astype(np.int32)
        + N_left.astype(np.int32)
        + N_right.astype(np.int32)
    )

    N_p_indices = np.arange(n)
    N_p_indptr = np.arange(n+1)

    N_p = csr_matrix((N_p_data, N_p_indices, N_p_indptr), shape=(n, n), dtype=np.int32)

    return N_p



def generate_neighbour_omega(
    mask: np.ndarray, mask_indices: np.ndarray, index_to_id: np.ndarray
) -> csr_matrix:
    n = mask_indices.shape[0]
    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]
    x_max = mask.shape[0]
    y_max = mask.shape[1]

    A_omega = dok_matrix((n, n), dtype=np.int32)

    # Up
    up = np.where((x_indices < mask.shape[0] - 1) & mask[(x_indices + 1) % x_max, y_indices])
    up_neighbour_ids = index_to_id[x_indices[up] + 1, y_indices[up]]
    valid_up_neighbours = up_neighbour_ids != -1
    A_omega[up[0][valid_up_neighbours], up_neighbour_ids[valid_up_neighbours]] = 1

    # Down
    down = np.where((x_indices > 0) & mask[x_indices - 1, y_indices])
    down_neighbour_ids = index_to_id[x_indices[down] - 1, y_indices[down]]
    valid_down_neighbours = down_neighbour_ids != -1
    A_omega[down[0][valid_down_neighbours], down_neighbour_ids[valid_down_neighbours]] = 1

    # Left
    left = np.where((y_indices > 0) & mask[x_indices, y_indices - 1])
    left_neighbour_ids = index_to_id[x_indices[left], y_indices[left] - 1]
    valid_left_neighbours = left_neighbour_ids != -1
    A_omega[left[0][valid_left_neighbours], left_neighbour_ids[valid_left_neighbours]] = 1

    # Right
    right = np.where((y_indices < mask.shape[1] - 1) & mask[x_indices, (y_indices + 1) % y_max])
    right_neighbour_ids = index_to_id[x_indices[right], y_indices[right] + 1]
    valid_right_neighbours = right_neighbour_ids != -1
    A_omega[right[0][valid_right_neighbours], right_neighbour_ids[valid_right_neighbours]] = 1

    return A_omega.tocsr()


def generate_b(
    mask: np.ndarray,
    mask_indices: np.ndarray,
    target: np.ndarray,
    sum_v_pq: np.ndarray,
) -> np.ndarray:
    """
    Note that target is a multi-channel image
    """
    n = mask_indices.shape[0]
    num_channels = target.shape[2]

    assert sum_v_pq.shape == (n, num_channels), f"{sum_v_pq.shape=}"

    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]
    x_max = mask.shape[0]
    y_max = mask.shape[1]

    target = target.astype(np.float32)
    b_up = np.zeros((n, num_channels), dtype=np.float32)
    b_down = np.zeros((n, num_channels), dtype=np.float32)
    b_left = np.zeros((n, num_channels), dtype=np.float32)
    b_right = np.zeros((n, num_channels), dtype=np.float32)

    # Up
    up = np.where((x_indices < mask.shape[0] - 1) & ~mask[(x_indices + 1) % x_max, y_indices])
    b_up[up] = target[x_indices[up] + 1, y_indices[up]]

    # Down
    down = np.where((x_indices > 0) & ~mask[x_indices - 1, y_indices])
    b_down[down] = target[x_indices[down] - 1, y_indices[down]]

    # Left
    left = np.where((y_indices > 0) & ~mask[x_indices, y_indices - 1])
    b_left[left] = target[x_indices[left], y_indices[left] - 1]

    # Right
    right = np.where((y_indices < mask.shape[1] - 1) & ~mask[x_indices, (y_indices + 1) % y_max])
    b_right[right] = target[x_indices[right], y_indices[right] + 1]

    b = b_up + b_down + b_left + b_right + sum_v_pq

    return b
