import numpy as np


def shift_array(
        image: np.ndarray, 
        shift: int, 
        axis=0
    ) -> np.ndarray:
    """Function to roll the elements of a multi-Dimensional array with zero padding.
    Args:
        image (np.ndarray): Input array to be shifted.
        shift (int): Number of places by which elements are shifted. 
                     Positive values shift right/down, negative values shift left/up.
        axis (int, optional): Axis along which to shift. Defaults to 0.
    Returns:
        np.ndarray: The shifted array with zero padding.
    """
    if shift == 0:
        return image
    
    result = np.zeros_like(image)
    sl1, sl2 = [slice(None)] * image.ndim, [slice(None)] * image.ndim

    if shift > 0:
        sl1[axis], sl2[axis] = slice(shift, None), slice(None, -shift)
    else:
        sl1[axis], sl2[axis] = slice(None, shift), slice(-shift, None)
    result[tuple(sl1)] = image[tuple(sl2)]

    return result