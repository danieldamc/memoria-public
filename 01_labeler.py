import albumentations
import math
import torch
import os
import argparse

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.models.unet import ResUnetv4
from src.loaders.nifti import load_nii, save_nii

# Scripts that Segment 3D or 4D CMR images using a 2D segmentation model
# this is based in the code from:
# https://research.monash.edu/en/publications/ventseg-efficient-open-source-framework-for-ventricular-segmentat


def convert_multiclass_mask(mask: torch.Tensor) -> torch.Tensor:
    """ Transform multiclass mask [batch, num_classes, h, w] to [batch, h, w]
    Args:
        mask (torch.Tensor): Multiclass mask
    Returns:
        torch.Tensor: Transformed multiclass mask
    
    """
    return mask.max(1)[1]


def apply_normalization(
        image: np.ndarray, 
        normalization_type: str
    ) -> np.ndarray:
    """ Applies normalization to the image https://www.statisticshowto.com/normalized/
    Args:
        image (np.ndarray): Image to normalize
        normalization_type (str): Type of normalization to apply. Options are: 'none', 'reescale', 'standardize'
    Returns:
        np.ndarray: Normalized image
    """
    if normalization_type == "none":
        return image
    elif normalization_type == "reescale":
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)
        return image
    elif normalization_type == "standardize":
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std
        return image
    assert False, "Unknown normalization: '{}'".format(normalization_type)


def common_test_augmentation(img_size: int) -> list:
    """ Common test augmentation for all datasets
    Args:
        img_size (int): Size of the image
    Returns:
        list: List of augmentations
    """
    return [
        albumentations.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True),
        albumentations.CenterCrop(height=img_size, width=img_size, always_apply=True),
        albumentations.Resize(img_size, img_size, always_apply=True)
    ]


# If error with shapes is found check this function, 
# it was made by previous contributor and can have some issues
def reshape_masks(
        ndarray: np.ndarray, 
        to_shape: tuple
    ) -> np.ndarray: 
    """Reshapes a center cropped (or padded) array back to its original shape.
    Args:
        ndarray (np.array): Mask Array to reshape
        to_shape (tuple): Final desired shape
    Returns: 
        np.array: Reshaped array to desired shape
    """
    h_in, w_in = ndarray.shape
    h_out, w_out = to_shape
    
    if h_in > h_out:
        h_offset = math.ceil((h_in - h_out) / 2)
        ndarray = ndarray[h_offset:(h_offset + h_out), :]
    else: 
        pad_h = (h_out - h_in)
        rem = pad_h % 2
        pad_dim_h = (math.floor(pad_h / 2), math.floor(pad_h / 2 + rem))
        npad = (pad_dim_h, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)

    if w_in > w_out: 
        w_offset = math.ceil((w_in - w_out) / 2)
        ndarray = ndarray[:, w_offset:(w_offset + w_out)]
    else: 
        pad_w = (w_out - w_in)
        rem = pad_w % 2
        pad_dim_w = (math.floor(pad_w / 2), math.floor(pad_w / 2 + rem))
        npad = ((0, 0), pad_dim_w)
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
    return ndarray 


def predict(
        model: torch.nn.Module, 
        image: np.ndarray
    ) -> np.ndarray:
    """Predict segmentation for a 2D MRI image
    Args:
        model (torch.nn.Module): Segmentation model
        image (np.ndarray): 2D MRI image
    Returns:
        np.ndarray: Segmented 2D MRI image
    """

    image = image.copy()
    image_shape = image.shape

    common_reshape = common_test_augmentation(224)
    image = albumentations.Compose(common_reshape)(image=image)["image"]
    image = apply_normalization(image,'standardize')

    image=np.expand_dims(image,axis=0)
    image=torch.from_numpy(image)

    image=np.expand_dims(image,axis=0)
    data = DataLoader(image, batch_size=1, shuffle=False, drop_last=False)

    model.eval()

    with torch.no_grad():
        for _, img in enumerate(data):
            img = img.type(torch.float).cpu()
            prob_pred = model(img)

    for _, single_pred in enumerate(prob_pred):
        pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
        pred_mask = reshape_masks(pred_mask.squeeze(0), image_shape)

    return pred_mask


def mri_predict(
        mri_image: np.ndarray,
        model: torch.nn.Module
    ) -> np.ndarray:
    """ Predict segmentation for a 3D or 4D MRI image slice by slice

    Args:
        mri_image (np.ndarray): 3D or 4D MRI image
        model (torch.nn.Module): Segmentation model
    
    Returns:
        np.ndarray: Segmented MRI image
    """
    mri_image = mri_image.copy()
    n_dim = mri_image.ndim

    if n_dim == 3:
        mri_image = np.expand_dims(mri_image, axis=-1)

    result_image = np.empty_like(mri_image)

    _, _, slice_count, phase_count = mri_image.shape

    for slice_number in range(0, slice_count):
        for phase_number in range(0, phase_count):
            result_image[:,:,slice_number, phase_number] = predict(model, mri_image[:, :, slice_number, phase_number])

    return result_image


def main(
        args: argparse.Namespace
    ) -> None:

    model = ResUnetv4('resnet34_unet_scratch', pretrained=None, num_classes=4, classification=False, in_channels=1).cpu()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    state_dict = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(state_dict, strict=False)

    for folder in tqdm(os.listdir(args.input)):
        mri_path = os.path.join(args.input, folder, f"{folder}{args.input_suffix}.{'nii.gz' if args.format == 'nifti' else 'npy'}")
        output_path = os.path.join(args.input, folder, f"{folder}{args.output_suffix}.{'nii.gz' if args.format == 'nifti' else 'npy'}")

        if not os.path.isfile(mri_path):
            print(f"File {mri_path} does not exist (supported formats are '.nii.gz' and '.npy'), skipping...")
            continue
        
        if args.format == 'nifti':
            original_image, affine, header = load_nii(file_path=mri_path)
        else:
            original_image = np.expand_dims(np.load(mri_path), axis=-1)
        result_image = mri_predict(original_image, model)

        if args.format == 'nifti':
            save_nii(output_path, result_image, affine, header) 
        else:
            np.save(output_path, result_image)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMR Image Segmentator')

    parser.add_argument(
        '-i', '--input',
        dest='input', 
        type=str, 
        required=True, 
        help='Path to the folder containing the CMR images'
    )
    parser.add_argument(
        '-m', '--model_path',
        dest='model_path',
        type=str, 
        default="weights/segmentation_model.pt",
        help='Path to the model file'
    )
    parser.add_argument(
        '-is', '--input_suffix',
        dest='input_suffix',
        type=str,
        default='_sa',
        help='Suffix of the input file'
    )
    parser.add_argument(
        '-os', '--output_suffix',
        dest='output_suffix',
        type=str,
        default='_sa_seg',
        help='Suffix to add to the output file'
    )
    parser.add_argument(
        '-f', '--format',
        dest='format',
        type=str,
        default='nifti',
        choices=['nifti', 'numpy'],
        help='Format of the input file (nifti/numpy)'
    )

    args = parser.parse_args()

    model_path = args.model_path
    input_path = args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Folder path {input_path} does not exist")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    main(args)