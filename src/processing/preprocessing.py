import numpy as np
import cv2 as cv

from copy import deepcopy as dc

def preprocess_image(
        image: np.ndarray, 
        h: int, 
        w: int
    ) -> np.ndarray:
    r"""Preprocess the image by resizing and padding it to 224x224.
    Args:
        image (np.ndarray): The input image to be preprocessed.
        h (int): The height of the input image.
        w (int): The width of the input image.
    Returns:
        np.ndarray: The preprocessed image of size 224x224.
    """
    max_hw = max(h, w)
    pad_h = (max_hw - h) // 2
    pad_w = (max_hw - w) // 2
    pad_h_extra = (max_hw - h) % 2
    pad_w_extra = (max_hw - w) % 2
    image = np.pad(
        image,
        pad_width=((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)),
        mode='constant',
        constant_values=0
    )
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_NEAREST)
    return image


def preprocess_volume(volume: np.ndarray) -> np.ndarray:
    r"""Preprocess the 3D volume by resizing and padding each slice to 224x224.
    Args:
        volume (np.ndarray): The input 3D volume to be preprocessed.stre
    Returns:
        np.ndarray: The preprocessed 3D volume with each slice of size 224x224.
    """
    volume = dc(volume)
    height, width, slices, phases = volume.shape
    processed_volume = np.zeros((224, 224, slices, phases))

    for i in range(slices):
        for j in range(phases):
            processed_volume[:, :, i, j] = preprocess_image(volume[:, :, i, j], height, width)

    return processed_volume