import numpy as np


def shift_array(image: np.ndarray, shift: int, axis=0) -> np.ndarray:
    """
    Function to roll the elements of a multi-Dimensional array with zero padding.

    :param image: (np.array) Image to shift
    :param shift: (int) Number of positions to shift
    :param axis: (int) Axis to shift
    :return: (np.array) Shifted image
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