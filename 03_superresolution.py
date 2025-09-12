import os
import argparse
import torch

import numpy as np
import cv2 as cv

from torch import nn
from copy import deepcopy as dc
from tqdm import tqdm
from skimage.exposure import match_histograms

from src.loaders.nifti import save_nii
from src.models.resunet import ResUnet
from src.loaders.mri import open_nifti, open_numpy
from src.processing.preprocessing import preprocess_image, preprocess_volume
from src.processing.postprocessing import postprocess_image

# Script to apply super-resolution on MRI images using a pre-trained model.

def downsample_volume(volume: np.ndarray) -> np.ndarray:
    """Function to downsample the volume by a factor of 2 in the slice dimension.
    Args:
        volume (np.ndarray): Input volume of shape (H, W, S, P).
    Returns:
        volume (np.ndarray): Downsampled volume of shape (H, W, S/2, P).
    """
    height, width, slice_count, phase_count = volume.shape
    indices_idxs = np.arange(0, slice_count, 2)
    return volume[:, :, indices_idxs, :]


def predict(
        input: np.ndarray, 
        model: nn.Module,
        device: str
    ) -> np.ndarray:
    """Function to predict the output of a model given an input.
    Args:
        input (np.ndarray): Input image of shape (2, H, W).
        model (nn.Module): The model to use for prediction.
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        output (np.ndarray): The predicted output image of shape (1, H, W).
    """

    model.eval()
    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(input)
    
    return output.cpu().numpy()


def volume_prediction(
    volume: np.ndarray, 
    model: nn.Module, 
    input_size: int,
    device: str
    ) -> np.ndarray:

    volume = dc(volume)

    h_original, w_original, slice_count, phase_count = volume.shape
    new_slice_count = slice_count * 2 - 1
    new_volume = np.zeros((h_original, w_original, new_slice_count, phase_count))

    for j in range(phase_count):
        for i in range(slice_count-1):
            slice_1 = volume[:, :, i, j]
            slice_2 = volume[:, :, i+1, j]

            slice_1_preprocessed = preprocess_image(slice_1, h_original, w_original)
            slice_2_preprocessed = preprocess_image(slice_2, h_original, w_original)

            input_image = np.stack([slice_1_preprocessed, slice_2_preprocessed], axis=0)
            output_image = predict(input_image, model, device).squeeze(0).squeeze(0)
            output_image = postprocess_image(output_image, h_original, w_original)

            output_image = np.clip(output_image, 0, 1)
            match_slice_1 = match_histograms(output_image, slice_1)
            match_slice_2 = match_histograms(output_image, slice_2)

            output_image = match_slice_1 * 0.5 + match_slice_2 * 0.5

            new_volume[:, :, i*2, j] = slice_1
            new_volume[:, :, i*2+1, j] = output_image

            if i == slice_count - 2:
                new_volume[:, :, i*2+2, j] = slice_2

    return new_volume.astype(np.float32)


def init_model(
    model_path: str, 
    device: str
    ) -> nn.Module:
    """Function to initialize the model from a given path.
    Args:
        model_path (str): Path to the model file.
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        model (nn.Module): The initialized model.
    Raises:
        ValueError: If the model file is not found.
    """

    model = ResUnet(in_channels=2,
                    out_channels=1,
                    num_convs=3,
                    kernel_size=3,
                    skip_kernel_size=3,
                    activation=nn.ReLU,
                    dropout=None,
                    normalization=nn.BatchNorm2d,
                    features=[32, 64, 128, 256, 512]).to(device)

    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def process_file(
    filename: str, 
    format_: str,
    input_suffix: str,
    output_suffix: str,
    model: nn.Module,
    model_input_size: int = 224,
    downsample: bool = True, 
    device: str = 'cpu'
    ) -> None:
    """Function to process a single file and save the output.
    Args:
        filename (str): Path to the input file.
        format_ (str): Format of the input file ('nifti' or 'numpy').
        input_suffix (str): Suffix of the input file to be processed.
        output_suffix (str): Suffix to be added to the output file.
        model (nn.Module): The model to use for prediction.
        model_input_size (int): Size of the input image for the model.
        downsample (bool): Whether to downsample the image in the spatial dimension.
        device (str): Device to run the model on ('cpu' or 'cuda').
    Returns:
        None
    """

    if format_ == 'nifti':
        volume, affine, header = open_nifti(filename)
        output_filename = filename.removesuffix('.nii.gz').removesuffix(input_suffix)
    else:
        volume = open_numpy(filename)
        output_filename = filename.removesuffix('.npy').removesuffix(input_suffix)

    volume = (volume / volume.max()).astype(np.float16)

    if downsample:
        volume = downsample_volume(volume)

    volume = preprocess_volume(volume)
    new_volume = volume_prediction(volume, model, model_input_size, device)

    if format_ == 'numpy':
        np.save(f'{output_filename}{output_suffix}.npy', new_volume)
    else:
        affine = np.eye(4) if format_ != "nifti" else affine.tolist()
        save_nii(f'{output_filename}{output_suffix}.nii.gz', new_volume, affine=affine)


def main(
        args: argparse.Namespace
    ) -> None:

    is_file = args.file

    if not os.path.exists(args.input):
        raise ValueError('Input folder does not exist')

    if os.path.isfile(args.model_path):
        model = init_model(args.model_path, args.device)
    else:
        raise ValueError('Model file not found')

    if is_file:
        process_file(
            args.input, 
            args.file_format,
            args.input_suffix,
            args.output_suffix,
            model=model,
            model_input_size=args.model_input_size,
            downsample=args.downsample,
            device=args.device
        )
    else:
        pbar = tqdm(sorted(os.listdir(args.input)), desc='Processing files')
        for filename in pbar:
            input_file = os.path.join(args.input, filename, f'{filename}{args.input_suffix}{".nii.gz" if args.file_format == "nifti" else ".npy"}')
            pbar.set_description(f'Processing {input_file}')
            process_file(
                input_file, 
                args.file_format,
                args.input_suffix,
                args.output_suffix,
                model=model,
                model_input_size=args.model_input_size,
                downsample=args.downsample,
                device=args.device
            )
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Super resolution Utility for MRI in Numpy or Nifti format'
    )
    parser.add_argument(
        '-i', '--input', 
        type=str,
        required=True,
        help='Path to the folder with the input files or to a single file'
    )
    parser.add_argument(
        '-f', '--format',
        dest='file_format',
        choices=['nifti', 'numpy'],
        default='nifti',
        help='Specify the format of the input files'
    )
    parser.add_argument(
        '-s', '--single',
        dest='file',
        action='store_true',
        help='Specify if the input is a file'
    )
    parser.add_argument(
        '-d', '--downsample',
        dest='downsample', 
        action='store_true',
        help='Downsample the image in the spatial dimension'
    )
    parser.add_argument(
        '-is', '--input_suffix',
        dest='input_suffix',
        type=str,
        default='_sa',
        help='suffix of the input file to be processed'
    )
    parser.add_argument(
        '-os', '--output_suffix',
        dest='output_suffix',
        type=str,
        default='_sa_sr',
        help='suffix to be added to the output file'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run the model'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="weights/superresolution_model.pth", 
        help='Path to the model file'
    )
    parser.add_argument(
        '--model_input_size', 
        type=int, 
        default=224, 
        help='size of the input image for the model'
    )
    
    args = parser.parse_args()

    main(args)


