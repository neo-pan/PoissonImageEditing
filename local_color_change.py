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
from discrete_possion_solver import solve


R_FACTOR = 0.5
G_FACTOR = 1.5
B_FACTOR = 1.5


def compute_gradints(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray
) -> np.ndarray:
    v = gradient_over_mask(mask, mask_indices, source)

    sum_v_pq = v.sum(-1)

    return sum_v_pq


def adjust_color_channels(
    image: np.ndarray, b_factor: float, g_factor: float, r_factor: float
) -> np.ndarray:
    result = image.copy().astype(np.float32)

    result[..., 0] = image[..., 0] * b_factor
    result[..., 1] = image[..., 1] * g_factor
    result[..., 2] = image[..., 2] * r_factor

    return result


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "-b", "--blue", type=float, default=B_FACTOR, help="blue channel factor"
    )
    parser.add_argument(
        "-g", "--green", type=float, default=G_FACTOR, help="green channel factor"
    )
    parser.add_argument(
        "-r", "--red", type=float, default=R_FACTOR, help="red channel factor"
    )

    args = parser.parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"{args.source} not found")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"{args.mask} not found")
    if os.path.exists(args.output):
        print(f"Warning! {args.output} already existed")

    source = read_image(args.source)
    mask = read_image(args.mask)
    target = source.copy()
    source = adjust_color_channels(source, args.blue, args.green, args.red)

    mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.uint8)
    mask_indices, index_to_id = get_mask_indices(mask)

    A = generate_A(mask, mask_indices, index_to_id)
    sum_v_pq = compute_gradints(mask, mask_indices, source)
    b = generate_b(mask, mask_indices, target, sum_v_pq)

    f, error = solve(A, b)
    print(f"Linear system error: {error}")

    result = fill_target(target, mask_indices, f)
    write_image(args.output, result)
