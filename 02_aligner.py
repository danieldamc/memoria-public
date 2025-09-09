import os
import argparse

import numpy as np

from tqdm import tqdm

from src.mri.alignment import Alignment
from src.loaders.nifti import load_nii, save_nii

def open_nifti(file_path: str):
    volume, affine, header = load_nii(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 3:
        raise ValueError('Volume must have at least 3 dimensions')
    if len(volume.shape) == 3:
        volume = volume[..., None]
    return volume, affine, header


def open_numpy(file_path: str):
    volume = np.load(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 3:
        raise ValueError('Volume must have at least 3 dimensions')
    if len(volume.shape) == 3:
        volume = volume[..., None]
    return volume

def XAND(a, b):
    return a and b

def process_mri(
        folder_path, 
        folder, 
        shift,
        output_format='same',
        output_suffix = ['_mov_aligned_xcorr_hv', '_mov_seg_aligned_xcorr_hv'],
        input_suffix=['_mov', '_mov_seg']
    ) -> None:
    
    path = os.path.join(folder_path, folder)

    fileformat = ".nii.gz" 

    mri_path = os.path.join(path, f"{folder}{input_suffix[0]}.nii.gz")
    mri_is_nifti = os.path.isfile(mri_path)
    segmentation_path = os.path.join(path, f"{folder}{input_suffix[1]}.nii.gz")
    segmentation_is_nifti = os.path.isfile(segmentation_path)

    same_format = XAND(mri_is_nifti, segmentation_is_nifti)
    # print(mri_is_nifti, segmentation_is_nifti)
    # if not same_format:
    #     raise ValueError("original and segmentation files are not in the same format")
    is_nifti = mri_is_nifti

    if is_nifti:
        mri_path = os.path.join(path, f"{folder}{input_suffix[0]}.nii.gz")
        segmentation_path = os.path.join(path, f"{folder}{input_suffix[1]}.nii.gz")

        mri_image, mri_affine, mri_header = open_nifti(mri_path)
        segmentation_image, segmentation_affine, segmentation_header = open_nifti(segmentation_path)
    else:
        mri_path = os.path.join(path, f"{folder}{input_suffix[0]}.npy")
        segmentation_path = os.path.join(path, f"{folder}{input_suffix[1]}.npy")

        mri_image = open_numpy(mri_path)
        segmentation_image = open_numpy(segmentation_path)
    
    mri_affine = np.eye(4) if not is_nifti else mri_affine.tolist()
    segmentation_affine = np.eye(4) if not is_nifti else segmentation_affine.tolist()

    Alignment_ = Alignment(original=mri_image, segmentation=segmentation_image, shift_type=shift)
    aligned_mri = Alignment_.get_volume(type_image='original', aligned=True)
    aligned_segmentation = Alignment_.get_volume(type_image='segmentation', aligned=True)


    if output_format == 'numpy':
        np.save(os.path.join(path, f'{folder}{output_suffix[0]}.npy'), aligned_mri)
        np.save(os.path.join(path, f'{folder}{output_suffix[1]}.npy'), aligned_segmentation) 
    elif output_format == 'nifti':
        save_nii(os.path.join(path, f'{folder}{output_suffix[0]}.nii.gz'), aligned_mri, mri_affine)
        save_nii(os.path.join(path, f'{folder}{output_suffix[1]}.nii.gz'), aligned_segmentation, segmentation_affine)
    elif output_format == 'same':
        if is_nifti:
            save_nii(os.path.join(path, f'{folder}{output_suffix[0]}.nii.gz'), aligned_mri, mri_affine)
            save_nii(os.path.join(path, f'{folder}{output_suffix[1]}.nii.gz'), aligned_segmentation, segmentation_affine)
        else:
            np.save(os.path.join(path, f'{folder}{output_suffix[0]}.npy'), aligned_mri)
            np.save(os.path.join(path, f'{folder}{output_suffix[1]}.npy'), aligned_segmentation)
    
def main(args: argparse.Namespace) -> None:
    folder_path = args.folder_path

    for folder in tqdm(os.listdir(folder_path)):
        process_mri(folder_path, folder, args.shift, args.output_format)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRI Image Segmentation')

    parser.add_argument(
        '--folder_path', 
        dest='folder_path',
        type=str, 
        required=True, 
        help='Path to the folder containing the MRI images'
    )
    parser.add_argument(
        '-s', '--shift_type',
        dest='shift',
        type=str,
        default='cm',
        choices=['cm', 'xcorr', '4ch_cm', '4ch_xcorr'],
        help='Type of shift to apply to the image'
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
