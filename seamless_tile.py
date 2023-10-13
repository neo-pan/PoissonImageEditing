import os
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
from discrete_poisson_solver import solve


def compute_gradints(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray
) -> np.ndarray:
    v = gradient_over_mask(mask, mask_indices, source)

    sum_v_pq = v.sum(-1)

    return sum_v_pq


def build_boundary_mask(image: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image) * 255
    mask[1:-1, 1:-1] = 255
    mask = mask.astype(np.uint8)
    return mask


def modify_target_boundary(target: np.ndarray) -> np.ndarray:
    target = target.copy().astype(np.float32)
    t_north = target[0, :]
    t_south = target[-1, :]
    t_west = target[:, 0]
    t_east = target[:, -1]

    target[0, :] = 0.5 * (t_north + t_south)
    target[-1, :] = 0.5 * (t_north + t_south)
    target[:, 0] = 0.5 * (t_west + t_east)
    target[:, -1] = 0.5 * (t_west + t_east)

    corner_point = 0.25 * (
        target[0, 0] + target[0, -1] + target[-1, 0] + target[-1, -1]
    )
    target[0, 0] = corner_point
    target[0, -1] = corner_point
    target[-1, 0] = corner_point
    target[-1, -1] = corner_point

    target[target < 0] = 0
    target[target > 255] = 255
    target = target.astype(np.uint8)

    return target


def replicate_image(
    image: np.ndarray, horizontal_copies: int, vertical_copies: int
) -> np.ndarray:
    new_image = np.tile(image, (vertical_copies, horizontal_copies, 1))

    return new_image


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"{args.source} not found")
    if os.path.exists(args.output):
        print(f"Warning! {args.output} already existed")

    source = read_image(args.source)
    target = source.copy()
    target = modify_target_boundary(target)
    mask = build_boundary_mask(source)

    mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.uint8)
    mask_indices, index_to_id = get_mask_indices(mask)

    A = generate_A(mask, mask_indices, index_to_id)
    sum_v_pq = compute_gradints(mask, mask_indices, source)
    b = generate_b(mask, mask_indices, target, sum_v_pq)

    f, error = solve(A, b)
    print(f"Linear system error: {error}")

    result = fill_target(target, mask_indices, f)
    result = replicate_image(result, 3, 2)

    write_image(args.output, result)
