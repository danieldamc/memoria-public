import nibabel as nib
import numpy as np

def load_nii(file_path: str) -> tuple:
    """Function to load a 'nii' or 'nii.gz' file
    Args:
        file_path (str): Path to the 'nii' or 'nii.gz' file.
    Returns:
        volume (np.ndarray): The image volume.
        affine (np.ndarray): The affine transformation matrix.
        header (nibabel.Nifti1Header): The NIfTI header.
    """
    nimg = nib.load(file_path)
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header

def save_nii(file_path, image, affine, header=None):
    """Function to save a 'nii' or 'nii.gz' file
    Args:
        file_path (str): Path to save the 'nii' or 'nii.gz' file.
        image (np.ndarray): The image volume to save.
        affine (np.ndarray): The affine transformation matrix.
        header (nibabel.Nifti1Header, optional): The NIfTI header
    Returns:
        None    
    """
    if header is None:
        nifti_image = nib.Nifti1Image(image, affine)
        nib.save(nifti_image, file_path)
        return None
        
    nifti_image = nib.Nifti1Image(image, affine, header)
    nib.save(nifti_image, file_path)
    return None