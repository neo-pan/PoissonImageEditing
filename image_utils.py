from typing import Tuple
import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
    image = cv2.imread(name)
    assert len(image.shape) == 3

    return image


def write_image(name: str, image: np.ndarray) -> None:
    if image.dtype == np.float32:
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype == np.int32:
        image = image.astype(np.uint8)
    else:
        assert image.dtype == np.uint8

    assert len(image.shape) == 3

    cv2.imwrite(name, image.squeeze())


def get_mask_indices(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask_indices = np.nonzero(mask)
    mask_indices = np.stack(mask_indices).transpose()

    pixel_id = np.arange(mask_indices.shape[0])
    index_to_id = np.zeros(mask.shape, dtype=np.int32) - 1
    index_to_id[mask_indices[:, 0], mask_indices[:, 1]] = pixel_id

    return mask_indices, index_to_id


def gradient_over_mask(
    mask: np.ndarray,
    mask_indices: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """
    Note that image is a multi-channel image
    """
    n = mask_indices.shape[0]
    num_channels = image.shape[2]

    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]

    image = image.astype(np.float32)
    v_up = np.zeros((n, num_channels), dtype=np.float32)
    v_down = np.zeros((n, num_channels), dtype=np.float32)
    v_left = np.zeros((n, num_channels), dtype=np.float32)
    v_right = np.zeros((n, num_channels), dtype=np.float32)

    # Up
    up = np.where(x_indices < mask.shape[0] - 1)
    v_up[up] = (
        image[x_indices[up], y_indices[up]] - image[x_indices[up] + 1, y_indices[up]]
    )

    # Down
    down = np.where(x_indices > 0)
    v_down[down] = (
        image[x_indices[down], y_indices[down]]
        - image[x_indices[down] - 1, y_indices[down]]
    )

    # Left
    left = np.where(y_indices > 0)
    v_left[left] = (
        image[x_indices[left], y_indices[left]]
        - image[x_indices[left], y_indices[left] - 1]
    )

    # Right
    right = np.where(y_indices < mask.shape[1] - 1)
    v_right[right] = (
        image[x_indices[right], y_indices[right]]
        - image[x_indices[right], y_indices[right] + 1]
    )

    # Gradients of 4 directions
    v = np.stack([v_up, v_down, v_left, v_right], axis=-1)

    return v


def fill_target(
    target: np.ndarray, mask_indices: np.ndarray, f: np.ndarray
) -> np.ndarray:
    result = target.copy()
    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]

    result[x_indices, y_indices] = f.astype(target.dtype)

    return result


def boundary_on_image(image: np.ndarray, mask_indices: np.ndarray) -> np.ndarray:
    result = image.copy()
    x_indices = mask_indices[:, 0]
    y_indices = mask_indices[:, 1]

    result[x_indices, y_indices] = 255

    return result

def gradient_magnitude(image: np.ndarray):
    image = image.astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    return gradient_magnitude