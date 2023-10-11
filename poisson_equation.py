from typing import Tuple
import numpy as np

""" In the reference paper, we have a discrete Poisson equation:
    |N_p| f_p - \sum_{q \in N_p \cap \Omega} f_q = \
    \sum_{q \in N_p \cap \partial \Omega} f^{\star}_q + \sum_{q \in N_p} v_{pq}
    `f` is the unknown variable, we need to compute following matrices:
    1. A: |N_p| - Index of \sum_{q \in N_p \cap \Omega}
    2. B: \sum_{q \in N_p \cap \partial \Omega} f^{\star}_q + \sum_{q \in N_p} v_{pq}
    Then we can solve the equation by `A * f = B`.
"""


def get_mask_indices(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask_indices = np.nonzero(mask)
    mask_indices = np.stack(mask_indices).transpose()

    pixel_id = np.arange(mask_indices.shape[0])
    index_to_id = np.zeros(mask.shape, dtype=np.int32) - 1
    index_to_id[mask_indices[:, 0], mask_indices[:, 1]] = pixel_id

    return mask_indices, index_to_id


def generate_A(
    mask: np.ndarray, mask_indices: np.ndarray, index_to_id: np.ndarray
) -> np.ndarray:
    neighbour_matrices = generate_neighbour_matrices(
        mask, mask_indices, index_to_id
    )



def generate_neighbour_size(mask: np.ndarray, mask_indices: np.ndarray)->np.ndarray:
    n = mask_indices.shape[0]
    N_p = np.zeros((n, 1), dtype=np.float32)
    


def generate_neighbour_matrices(
    mask: np.ndarray, mask_indices: np.ndarray, index_to_id: np.ndarray
):
    n = mask_indices.shape[0]

    A_up = np.zeros((n, n), dtype=np.float32)
    A_down = np.zeros((n, n), dtype=np.float32)
    A_left = np.zeros((n, n), dtype=np.float32)
    A_right = np.zeros((n, n), dtype=np.float32)

    for pixel_id, coord in enumerate(mask_indices):
        x, y = coord
        # Up
        if x > 0 and mask[x - 1, y]:
            neighbor_id = index_to_id[x - 1, y]
            if neighbor_id != -1:
                A_up[pixel_id, neighbor_id] = 1.0
        # Down
        if x < mask.shape[0] - 1 and mask[x + 1, y]:
            neighbor_id = index_to_id[x + 1, y]
            if neighbor_id != -1:
                A_down[pixel_id, neighbor_id] = 1.0
        # Left
        if y > 0 and mask[x, y - 1]:
            neighbor_id = index_to_id[x, y - 1]
            if neighbor_id != -1:
                A_left[pixel_id, neighbor_id] = 1.0
        # Right
        if y < mask.shape[1] - 1 and mask[x, y + 1]:
            neighbor_id = index_to_id[x, y + 1]
            if neighbor_id != -1:
                A_right[pixel_id, neighbor_id] = 1.0

    return A_up, A_down, A_left, A_right


def generate_neighbour_matrices_vec(mask, mask_indices, index_to_id):
    n = mask_indices.shape[0]

    A_up = np.zeros((n, n), dtype=np.float32)
    A_down = np.zeros((n, n), dtype=np.float32)
    A_left = np.zeros((n, n), dtype=np.float32)
    A_right = np.zeros((n, n), dtype=np.float32)

    x, y = mask_indices[:, 0], mask_indices[:, 1]

    # Up
    up_indices = np.where((x > 0) & mask[x - 1, y])
    up_neighbour_ids = index_to_id[x[up_indices] - 1, y[up_indices]]
    valid_up_neighbours = up_neighbour_ids != -1
    A_up[up_indices[0][valid_up_neighbours], up_neighbour_ids[valid_up_neighbours]] = 1.0

    # Down
    down_indices = np.where((x < mask.shape[0] - 1) & mask[x + 1, y])
    down_neighbour_ids = index_to_id[x[down_indices] + 1, y[down_indices]]
    valid_down_neighbours = down_neighbour_ids != -1
    A_down[down_indices[0][valid_down_neighbours], down_neighbour_ids[valid_down_neighbours]] = 1.0

    # Left
    left_indices = np.where((y > 0) & mask[x, y - 1])
    left_neighbour_ids = index_to_id[x[left_indices], y[left_indices] - 1]
    valid_left_neighbours = left_neighbour_ids != -1
    A_left[left_indices[0][valid_left_neighbours], left_neighbour_ids[valid_left_neighbours]] = 1.0

    # Right
    right_indices = np.where((y < mask.shape[1] - 1) & mask[x, y + 1])
    right_neighbour_ids = index_to_id[x[right_indices], y[right_indices] + 1]
    valid_right_neighbours = right_neighbour_ids != -1
    A_right[right_indices[0][valid_right_neighbours], right_neighbour_ids[valid_right_neighbours]] = 1.0

    return A_up, A_down, A_left, A_right