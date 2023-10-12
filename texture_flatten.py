import os
import cv2
import numpy as np
from scipy import ndimage

from args import get_parser
from image_utils import read_image, write_image, get_mask_indices, gradient_over_mask, fill_target
from poisson_equation import generate_A, generate_b
from discrete_possion_solver import solve


def edge_detection(image: np.ndarray, method:str='sobel', threshold: int=50)->np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'sobel':
        edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1)
        edges = (edges > threshold).astype(np.uint8)
    elif method == 'canny':
        edges = cv2.Canny(gray_image, threshold, threshold*2)
        edges = edges.astype(np.uint8)
    else:
        raise ValueError("Unsupported edge detection method. Supported methods: sobel, canny")

    return edges


def compute_gradints(
    mask: np.ndarray, mask_indices: np.ndarray, source: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    v_source = gradient_over_mask(mask, mask_indices, source)
    n = mask_indices.shape[0]

    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]

    edge_up = np.zeros((n,), dtype=np.float32)
    edge_down = np.zeros((n,), dtype=np.float32)
    edge_left = np.zeros((n,), dtype=np.float32)
    edge_right = np.zeros((n,), dtype=np.float32)

    up = np.where(x_indices < mask.shape[0] - 1)
    edge_up[up] = (
        edges[x_indices[up], y_indices[up]] != edges[x_indices[up] + 1, y_indices[up]]
    )

    down = np.where(x_indices > 0)
    edge_down[down] = (
        edges[x_indices[down], y_indices[down]] != edges[x_indices[down] - 1, y_indices[down]]
    )

    left = np.where(y_indices > 0)
    edge_left[left] = (
        edges[x_indices[left], y_indices[left]] != edges[x_indices[left], y_indices[left] - 1]
    )

    right = np.where(y_indices < mask.shape[1] - 1)
    edge_right[right] = (
        edges[x_indices[right], y_indices[right]] != edges[x_indices[right], y_indices[right] + 1]
    )

    edge_mask = np.stack([edge_up, edge_down, edge_left, edge_right], axis=-1)
    edge_mask = edge_mask[:, None, :]

    v = edge_mask * v_source
    sum_v_pq = v.sum(-1)

    return sum_v_pq


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument(
        "-e",
        "--edge",
        type=str,
        choices=["sobel", "canny"],
        default="canny",
        help="method to calculate gradient",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="threshold value for edge detection"
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
    
    mask = mask.mean(-1)
    mask = (mask >= 128).astype(np.uint8)

    mask_indices, index_to_id = get_mask_indices(mask)

    A = generate_A(mask, mask_indices, index_to_id)

    edges = edge_detection(source, args.edge, args.threshold)
    sum_v_pq = compute_gradints(mask, mask_indices, source, edges)
    
    b = generate_b(mask, mask_indices, source, sum_v_pq)

    f, error = solve(A, b)
    print(f"Linear system solving error: {error}")
    result = fill_target(source, mask_indices, f)

    write_image(args.output, result)