import numpy as np
import cv2 as cv

from copy import deepcopy as dc

def postprocess_image(
        image: np.ndarray, 
        h: int, 
        w: int
    ) -> np.ndarray:
    r"""Postprocess the image to restore its original dimensions.
    Args:
        image (np.ndarray): The input image to be postprocessed.
        h (int): The original height of the image.
        w (int): The original width of the image.
    Returns:
        np.ndarray: The postprocessed image with restored dimensions.
    """
    image = dc(image)

    max_hw = max(h, w)
    image = cv.resize(image, (max_hw, max_hw), interpolation=cv.INTER_NEAREST)

    pad_h = (max_hw - h) // 2
    pad_w = (max_hw - w) // 2
    pad_h_extra = (max_hw - h) % 2
    pad_w_extra = (max_hw - w) % 2
    image = image[pad_h:max_hw - pad_h - pad_h_extra, pad_w:max_hw - pad_w - pad_w_extra]
    return image
    

def postprocess_volume(
        volume: np.ndarray, 
        original_h: int, 
        original_w: int
    ) -> np.ndarray:
    r"""Postprocess the 3D volume to restore its original dimensions.
    Args:
        volume (np.ndarray): The input 3D volume to be postprocessed.
        original_h (int): The original height of the volume.
        original_w (int): The original width of the volume.
    Returns:
        np.ndarray: The postprocessed 3D volume with restored dimensions.
    """
    volume = dc(volume)
    _, _, slices, phases = volume.shape
    processed_volume = np.zeros((original_h, original_w, slices, phases))
    for i in range(slices):
        for j in range(phases):
            processed_volume[:, :, i, j] = postprocess_image(volume[:, :, i, j], original_h, original_w)
    
    return processed_volume