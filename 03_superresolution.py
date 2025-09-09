import os
import argparse
import torch

import numpy as np
import cv2 as cv

from torch import nn
from copy import deepcopy as dc
from tqdm import tqdm

from src.loaders.nifti import load_nii, save_nii
from models.resunet import ResUnet

def open_nifti(file_path: str):
    volume, affine, header = load_nii(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 4:
        raise ValueError('Volume must have at least 4 dimensions')
    return volume, affine, header


def open_numpy(file_path: str):
    volume = np.load(file_path)
    v_n_dims = len(volume.shape)
    if v_n_dims < 3:
        raise ValueError('Volume must have at least 3 dimensions')
    if len(volume.shape) == 3:
        volume = volume[..., None]
    return volume


def downsample_volume(volume):
    height, width, slice_count, phase_count = volume.shape
    indices_idxs = np.arange(0, slice_count, 2)
    return volume[:, :, indices_idxs, :]


def preprocess_volume(volume):
    def preprocess_image(image, h, w):
        max_hw = max(h, w)
        pad_h = (max_hw - h) // 2
        pad_w = (max_hw - w) // 2
        pad_h_extra = (max_hw - h) % 2
        pad_w_extra = (max_hw - w) % 2
        image = np.pad(
            image,
            pad_width=((pad_h, pad_h + pad_h_extra), (pad_w, pad_w + pad_w_extra)),
            mode='constant',
            constant_values=0
        )
        image = cv.resize(image, (224, 224), interpolation=cv.INTER_NEAREST)
        return image

    volume = dc(volume)

    height, width, slices, phases = volume.shape
    processed_volume = np.zeros((224, 224, slices, phases))

    for i in range(slices):
        for j in range(phases):
            processed_volume[:, :, i, j] = preprocess_image(volume[:, :, i, j], height, width)

    return processed_volume


def predict(
        input: np.ndarray, 
        model: nn.Module,
        device: str
    ) -> np.ndarray:

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

    _, _, slice_count, phase_count = volume.shape
    new_slice_count = slice_count * 2 - 1
    new_volume = np.zeros((input_size, input_size, new_slice_count, phase_count))

    for j in range(phase_count):
        for i in range(slice_count-1):
            slice_1 = volume[:, :, i, j]
            slice_2 = volume[:, :, i+1, j]

            input_image = np.stack([slice_1, slice_2], axis=0)
            output_image = predict(input_image, model, device)

            new_volume[:, :, i*2, j] = slice_1
            new_volume[:, :, i*2+1, j] = output_image.squeeze().squeeze()

            if i == slice_count - 2:
                new_volume[:, :, i*2+2, j] = slice_2

    return new_volume.astype(np.float32)


def init_model(
    model_path: str, 
    device: str
    ) -> nn.Module:

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
    output_folder: str,
    output_format: str,
    output_suffix: str,
    model: nn.Module,
    model_input_size: int = 224,
    downsample: bool = True, 
    device: str = 'cpu'
    ) -> None:

    is_nifti = filename.endswith('.nii.gz')
    if is_nifti:
        volume, affine, header = open_nifti(filename)
        output_filename = filename.removesuffix('.nii.gz').split('/')[-1]
    else:
        volume = open_numpy(filename)
        output_filename = filename.removesuffix('.npy').split('/')[-1]
    
    volume = (volume / volume.max()).astype(np.float16)

    if downsample:
        volume = downsample_volume(volume)

    volume = preprocess_volume(volume)
    new_volume = volume_prediction(volume, model, model_input_size, device)

    if output_format == 'numpy':
        np.save(os.path.join(output_folder, f'{output_filename}{output_suffix}.npy'), new_volume)
    elif output_format == 'nifti':
        affine = np.eye(4) if not is_nifti else affine.tolist()
        save_nii(os.path.join(output_folder, f'{output_filename}{output_suffix}.nii.gz'), new_volume, affine=affine)
    else:
        raise ValueError('Output format not supported')


def main(
        args: argparse.Namespace
    ) -> None:

    is_file = args.file

    if os.path.isfile(args.model_path):
        model = init_model(args.model_path, args.device)
    else:
        raise ValueError('Model file not found')

    if not os.path.exists(args.input):
        raise ValueError('Input folder does not exist')

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    if is_file:
        process_file(
            args.input, 
            args.output, 
            args.output_format, 
            args.suffix,
            model=model,
            model_input_size=args.model_input_size,
            downsample=args.downsample,
            device=args.device
        )
    else:
        pbar = tqdm(sorted(os.listdir(args.input)), desc='Processing files')
        for filename in pbar:
            pbar.set_description(f'Processing {filename}')
            process_file(
                os.path.join(args.input, filename), 
                args.output, 
                args.output_format, 
                args.suffix,
                model=model,
                model_input_size=args.model_input_size,
                downsample=args.downsample,
                device=args.device
            )
    

if __name__ == '__main__':

    # command 1: python superresolution.py --input dataset/Original/Custom_Split/Test/A1E9Q1/A1E9Q1_sa.nii.gz --output test_output/ -o numpy -f -d
    # command 2: python superresolution.py --input dataset/Alignment/processed/8mm_dataset/processed/npy/volumes/  --output test_output/ -o numpy -d

    parser = argparse.ArgumentParser(
        description='Super resolution Utility for MRI in Numpy or Nifti format'
    )

    parser.add_argument(
        '--input', 
        type=str,
        required=True,
        help='Path to the folder with the input files'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='path to the folder where the output files will be saved'
    )
    parser.add_argument(
        '-o', '--output_format',
        choices=['nifti', 'numpy'],
        default='numpy',
        help='Specify the output format'
    )
    parser.add_argument(
        '-f', '--file',
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
        '-sg', '--segment',
        dest='segment',
        action='store_true',
        help='Specify if the output should be segmented'
    )
    parser.add_argument(
        '-s', '--suffix',
        dest='suffix',
        type=str,
        default='_sr',
        help='suffix to be added to the output file'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        help='Device to run the model'
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="models/ResUnet_32-512_FINAL_PRELIMINAR_model_10E.pth", 
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


