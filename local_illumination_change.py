import os
import math
import numpy as np

from args import get_parser
from image_utils import (
    read_image,
    write_image,
    get_mask_indices,
    gradient_over_mask,
    fill_target,
)
from poisson_equation import generate_A, generate_b
from discrete_possion_solver import solve


ALPHA = 0.2
BETA = 0.2


def compute_gradints(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray
) -> np.ndarray:
    v_source = gradient_over_mask(mask, mask_indices, source)
    g_norm = np.linalg.norm(v_source, axis=(0, 2))

    v = ALPHA**BETA * v_source / np.power(g_norm, BETA)[None, :, None]

    sum_v_pq = v.sum(-1)

    return sum_v_pq


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"{args.source} not found")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"{args.mask} not found")
    if os.path.exists(args.output):
        print(f"Warning! {args.output} already existed")

    source = read_image(args.source)
    mask = read_image(args.mask)

    mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.uint8)

    mask_indices, index_to_id = get_mask_indices(mask)

    A = generate_A(mask, mask_indices, index_to_id)

    sum_v_pq = compute_gradints(mask, mask_indices, source)

    b = generate_b(mask, mask_indices, source, sum_v_pq)

    f, error = solve(A, b)
    print(f"Linear system solving error: {error}")
    result = fill_target(source, mask_indices, f)

    write_image(args.output, result)
