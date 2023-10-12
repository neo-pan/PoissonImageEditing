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


def import_gradients(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray
) -> np.ndarray:
    v = gradient_over_mask(mask, mask_indices, source)

    sum_v_pq = v.sum(-1)

    return sum_v_pq


def mix_gradints(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    v_source = gradient_over_mask(mask, mask_indices, source)
    v_target = gradient_over_mask(mask, mask_indices, target)

    v = v_source.copy()
    target_indices = np.abs(v_target) > np.abs(v_source)
    v[target_indices] = v_target[target_indices]

    sum_v_pq = v.sum(-1)

    return sum_v_pq


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "-g",
        "--gradient",
        type=str,
        choices=["src", "mix"],
        default="src",
        help="method to calculate gradient",
    )
    args = parser.parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"{args.source} not found")
    if not os.path.exists(args.target):
        raise FileNotFoundError(f"{args.target} not found")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"{args.mask} not found")
    if os.path.exists(args.output):
        print(f"Warning! {args.output} already existed")

    source = read_image(args.source)
    target = read_image(args.target)
    mask = read_image(args.mask)
    mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.uint8)
    mask_indices, index_to_id = get_mask_indices(mask)

    A = generate_A(mask, mask_indices, index_to_id)
    if args.gradient == "src":
        sum_v_pq = import_gradients(mask, mask_indices, source)
    elif args.gradient == "mix":
        sum_v_pq = mix_gradints(mask, mask_indices, source, target)
    else:
        raise NotImplementedError
    b = generate_b(mask, mask_indices, target, sum_v_pq)
    f, error = solve(A, b)
    print(f"Linear system solving error: {error}")

    result = fill_target(target, mask_indices, f)
    write_image(args.output, result)
