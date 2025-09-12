import numpy as np
from .nifti import load_nii

def open_nifti(file_path: str) -> tuple:
    """ Open a NIfTI file and return the volume, affine, and header.
    
    Args:
        file_path (str): Path to the NIfTI file.
    Returns:
        volume (np.ndarray): The image volume.
        affine (np.ndarray): The affine transformation matrix.
        header (nibabel.Nifti1Header): The NIfTI header.
    Raises:
        ValueError: If the volume has less than 3 dimensions.
    """
    volume, affine, header = load_nii(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 3:
        raise ValueError('Volume must have at least 3 dimensions')
    if len(volume.shape) == 3:
        volume = volume[..., None]
    return volume, affine, header


def open_numpy(file_path: str) -> np.ndarray:
    """ Open a NumPy file and return the volume.
    Args:
        file_path (str): Path to the NumPy file.
    Returns:
        volume (np.ndarray): The image volume.
    Raises:
        ValueError: If the volume has less than 3 dimensions.
    """
    volume = np.load(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 3:
        raise ValueError('Volume must have at least 3 dimensions')
    if len(volume.shape) == 3:
        volume = volume[..., None]
    return volume
