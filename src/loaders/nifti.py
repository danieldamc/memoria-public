import nibabel as nib
import numpy as np

def load_nii(file_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header
    :param file_path: (string) Path of the 'nii' or 'nii.gz' image file name
    :return: Three element, the first is a numpy array of the image values (height, width, slices, phases),
             ## (No) the second is the affine transformation of the image, and the
             ## (No) last one is the header of the image.
    """

    nimg = nib.load(file_path)
    return np.asanyarray(nimg.dataobj), nimg.affine, nimg.header

def save_nii(file_path, image, affine, header=None):
    """
    Function to save a 'nii' or 'nii.gz' file
    :param file_path: (string) Path of the 'nii' or 'nii.gz' image file name
    :param data: (numpy array) Image data to save
    :param affine: (numpy array) Affine transformation of the image
    :param header: (nibabel.nifti1.Nifti1Header) Header of the image
    :return: None
    """
    if header is None:
        nifti_image = nib.Nifti1Image(image, affine)
        nib.save(nifti_image, file_path)
        return None
        
    nifti_image = nib.Nifti1Image(image, affine, header)
    nib.save(nifti_image, file_path)
    return None