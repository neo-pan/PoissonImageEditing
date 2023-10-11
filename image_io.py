import cv2
import numpy as np


def read_image(name: str) -> np.ndarray:
    img = cv2.imread(name)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    return img


def write_image(name: str, image: np.ndarray) -> None:
    cv2.imwrite(name, image)
