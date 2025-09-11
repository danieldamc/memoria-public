import os
import argparse

import numpy as np

from tqdm import tqdm

from src.mri.alignment import Alignment
from src.loaders.nifti import load_nii, save_nii


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

def process_mri(
        folder_path: str, 
        folder: str, 
        shift: str,
        input_format: str = 'nifti',
        input_suffix: list[str] = ['_sa', '_sa_seg'],
        output_suffix: list[str] = ['_sa_aligned', '_sa_seg_aligned'],
        keep_type: bool = False,
    ) -> None:
    """ Process a single MRI image and its segmentation.
    Args:
        folder_path (str): Path to the folder containing the MRI images.
        folder (str): Name of the folder containing the MRI images.
        shift (str): Type of shift to apply to the image.
        output_format (str): Output format for the aligned images. Options are 'nifti
        output_suffix (list[str]): Suffixes for the output files.
        input_suffix (list[str]): Suffixes for the input files.
    Returns:
        None
    """
    path = os.path.join(folder_path, folder)

    file_format = ".nii.gz" if input_format == 'nifti' else ".npy"
    mri_path = os.path.join(path, f"{folder}{input_suffix[0]}{file_format}")
    segmentation_path = os.path.join(path, f"{folder}{input_suffix[1]}{file_format}")

    if not os.path.exists(mri_path) or not os.path.exists(segmentation_path):
        print(f"Skipping {folder} as files are missing, check input suffixes and format")
        return None

    if input_format == "nifti":
        mri_image, mri_affine, mri_header = open_nifti(mri_path)
        segmentation_image, segmentation_affine, segmentation_header = open_nifti(segmentation_path)
    else:
        mri_image = open_numpy(mri_path)
        segmentation_image = open_numpy(segmentation_path)
    
    mri_affine = np.eye(4) if input_format != "nifti" else mri_affine.tolist()
    segmentation_affine = np.eye(4) if input_format != "nifti" else segmentation_affine.tolist()

    Alignment_ = Alignment(original=mri_image, segmentation=segmentation_image, shift_type=shift)
    aligned_mri = Alignment_.get_volume(type_image='original', aligned=True)
    aligned_segmentation = Alignment_.get_volume(type_image='segmentation', aligned=True)

    keep_type = f'_{shift}' if keep_type else ''

    if input_format == "nifti":
        save_nii(os.path.join(path, f'{folder}{output_suffix[0]}{keep_type}.nii.gz'), aligned_mri, mri_affine)
        save_nii(os.path.join(path, f'{folder}{output_suffix[1]}{keep_type}.nii.gz'), aligned_segmentation, segmentation_affine)
    elif input_format == "numpy":
        np.save(os.path.join(path, f'{folder}{output_suffix[0]}{keep_type}.npy'), aligned_mri)
        np.save(os.path.join(path, f'{folder}{output_suffix[1]}{keep_type}.npy'), aligned_segmentation)
    return None


def main(args: argparse.Namespace) -> None:
    folder_path = args.folder_path
    for folder in tqdm(os.listdir(folder_path)):
        process_mri(folder_path=folder_path, 
                    folder=folder, 
                    shift=args.shift, 
                    input_format=args.input_format,
                    input_suffix=args.input_suffix,
                    output_suffix=args.output_suffix,
                    keep_type=args.keep_type)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRI Image Segmentation')

    parser.add_argument(
        '-i', '--input', 
        dest='folder_path',
        type=str, 
        required=True, 
        help='Path to the folder containing the MRI images'
    )
    parser.add_argument(
        '-s', '--shift',
        dest='shift',
        type=str,
        default='cm_hv',
        choices=['cm_hv', 'xcorr_hv', 'cm_4ch', 'xcorr_4ch'],
        help='Type of shift to apply to the image'
    )
    parser.add_argument(
        '-is', '--input_suffix',
        dest='input_suffix',
        type=str,
        nargs=2,
        default=['_sa', '_sa_seg'],
        help='Suffixes for the input files'
    )
    parser.add_argument(
        '-os', '--output_suffix',
        dest='output_suffix',
        type=str,
        nargs=2,
        default=['_sa_aligned', '_sa_seg_aligned'],
        help='Suffixes for the output files'
    )
    parser.add_argument(
        '-k', '--keep_type',
        action='store_true',
        help='Add the shift type to the output suffix'
    )
    parser.add_argument(
        '-f', '--format',
        dest='input_format',
        type=str,
        default='nifti',
        choices=['nifti', 'numpy'],
        help='Format of the input files'
    )
    parser.add_argument(
        '-o', '--output_format',
        dest='output_format',
        type=str,
        default='same',
        choices=['nifti', 'numpy', 'same'],
        help='Output format for the aligned images'
    )
    args = parser.parse_args()
    folder_path = args.folder_path

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exist")

    main(args)