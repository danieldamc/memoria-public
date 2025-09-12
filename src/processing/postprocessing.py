import numpy as np
import cv2 as cv

from copy import deepcopy as dc

def postprocess_image(image, h, w):
        image = dc(image)

        max_hw = max(h, w)
        image = cv.resize(image, (max_hw, max_hw), interpolation=cv.INTER_NEAREST)

        pad_h = (max_hw - h) // 2
        pad_w = (max_hw - w) // 2
        pad_h_extra = (max_hw - h) % 2
        pad_w_extra = (max_hw - w) % 2
        image = image[pad_h:max_hw - pad_h - pad_h_extra, pad_w:max_hw - pad_w - pad_w_extra]
        return image
    

def postprocess_volume(volume, original_h, original_w):    
    volume = dc(volume)
    _, _, slices, phases = volume.shape
    processed_volume = np.zeros((original_h, original_w, slices, phases))
    for i in range(slices):
        for j in range(phases):
            processed_volume[:, :, i, j] = postprocess_image(volume[:, :, i, j], original_h, original_w)
    
    return processed_volume