# thesis-public
This is the public repository for my undergrad thesis titled:

Superresolución y alineamiento de imágenes de eje corto del corazón obtenidas por resonancia magnética cardiovascular

[Manuscript](https://repositorio.usm.cl/entities/tesis/090340cd-f955-4b98-91e5-2c340599f218)

This is a continuation of previous work on [VentSeg](https://research.monash.edu/en/publications/ventseg-efficient-open-source-framework-for-ventricular-segmentat).

# Setup
### Requisites
- Linux OS
- Nvidia GPU + CUDA CuDNN
- Python 3

### Getting Started
- Clone this repo
```bash
git clone git@github.com:danieldamc/thesis-public.git
cd thesis-public
```
or
```bash
git clone https://github.com/danieldamc/thesis-public.git
cd thesis-public
```
- Install the dependencies, For Conda users, you can create a new environment with:
```bash
conda env create -f environment.yml
```
and then activate the environment with:
```bash
conda activate thesis
```
- Download the dataset from [M&M's Challenge](https://www.ub.edu/mnms/)
- Download models weights:
```bash
python weights/download_weights.py
```
or download them manually from [here](https://drive.google.com/drive/folders/1Yd-B0w-bpcqYKBeqUYIltK3q-MvKHWlG?usp=drive_link).
# Scripts
This repository contains three scripts (`01_labeler.py`, `02_aligner.py`, and `03_superresolucion.py`). Each script works independently and addresses a specific problem.

## `01_labeler.py`
This script is designed to segment short-axis Cardiac MRI images. It is an adaptation of the methodology presented in the VentSeg [Paper](https://research.monash.edu/en/publications/ventseg-efficient-open-source-framework-for-ventricular-segmentat), tailored for processing whole datasets. The model segments the left ventricle, right ventricle, and myocardium.

### Usage
```bash
python 01_labeler.py --input <path_to_mri_images_dataset> [--model_path <path_to_model_file>] [--input_suffix <input_suffix>] [--output_suffix <output_suffix>] [--format <input_format>]
```
**Arguments**
- `--input` or `-i` (required):
  - Description: Specifies the path to the folder containing the MRI images you want to segment.  
  The folder containing the MRI must look like the following example:
    ```
    mri_scans/  
    ├── PATIENT1/  
    │   └── PATIENT1_sa.nii.gz  
    └── PATIENT2/  
        └── PATIENT2_sa.nii.gz
    ```
    Where the **folder name is the same as the filename of the `.nii.gz`**
  - Example: If your images are located in a folder called `mri_scans`, you would use `--input /path/to/mri_scans`
- `--model_path` or `-m` (optional):
  - Description: Specifies the path to the pre-trained model file that will be used for segmentation.
  - Default: `weights/segmentation_model.pt`
- `--input_suffix` or `-is` (optional):
  - Description: Defines the suffix that the input files have.
  - Default: `_sa`
  - Example: If you set `--input_suffix _sa`, an image named `PATIENT1_sa.nii.gz` will be processed.
- `--output_suffix` or `-os` (optional):
  - Description: Defines the suffix to add to the output file names after segmentation.
  - Default: `_sa_seg`
  - Example: If you set `--suffix _segmented`, an image named `PATIENT1.nii.gz` will have its segmented version saved as `PATIENT1_segmented.nii.gz`
- `--format` or `-f` (optional):
  - Description: Specifies the input format for the segmented images.
  - Default: `nifti`
  - Options: `nifti` or `numpy`

## `02_aligner.py`
This script is designed to align short-axis Cardiac MRI images and their segmentations. It utilizes cross-correlation and image processing techniques to achieve accurate alignment.

### Usage
```bash
python 02_aligner.py --input <path_to_mri_images_dataset> [--shift <shift_type>] [--input_suffix <input_suffix>] [--output_suffix <output_suffix>] [--keep_type] [--format <input_format>]
```

**Arguments**
- `--input` or `-i` (required):
  - Description: Specifies the path to the folder containing the MRI images and segmentations you want to align.  
  The folder containing the MRI must look like the following example:
    ```
    mri_scans/  
    ├── PATIENT1/  
    │   ├── PATIENT1_sa.nii.gz  
    │   └── PATIENT1_sa_seg.nii.gz  
    └── PATIENT2/  
        ├── PATIENT2_sa.nii.gz  
        └── PATIENT2_sa_seg.nii.gz
    ```
    Where the **folder name is the same as the filename of the `.nii.gz`**
  - Example: If your images are located in a folder called `mri_scans`, you would use `--input /path/to/mri_scans`
- `--shift` or `-s` (optional):
  - Description: Type of shift to apply to the image. Options are:
    - `cm_hv`: Center of Mass alignment using the horizontal and vertical axes as reference.
    - `xcorr_hv`: Cross-Correlation alignment using the horizontal and vertical axes as reference.
    - `cm_4ch`: Center of Mass alignment using the 4-chamber view as reference.
    - `xcorr_4ch`: Cross-Correlation alignment using the 4-chamber view.
  - Default: `cm_hv`
  - Example: If you want to align using cross-correlation with the 4-chamber view, you would use `--shift xcorr_4ch`
- `--input_suffix` or `-is` (optional):
  - Description: Suffixes of the CMR to be processed. The first suffix is for the CMR images, and the second is for the segmentations.
  - Default: `_sa _sa_seg`
  - Example: If you set `--input_suffix _sa _sa_seg`, an image named `PATIENT1_sa.nii.gz` and its segmentation `PATIENT1_sa_seg.nii.gz` will be processed.
- `--output_suffix` or `-os` (optional):
  - Description: Suffixes to add to the output file names after alignment. The first suffix is for the aligned CMR images, and the second is for the aligned segmentations.
  - Default: `_sa_aligned _sa_seg_aligned`
  - Example: If you set `--output_suffix _sa_aligned _sa_seg_aligned`, an image named `PATIENT1_sa.nii.gz` will have its aligned version saved as `PATIENT1_sa_aligned.nii.gz`, and its segmentation will be saved as `PATIENT1_sa_seg_aligned.nii.gz`
- `--keep_type` or `-k` (optional):
  - Description: If this flag is set, the type of shift applied to the image will be added to the output suffix.
  - Default: `False`
  - Example: If you set `--keep_type` and the shift type is `cm`, an image named `PATIENT1_sa.nii.gz` will have its aligned version saved as `PATIENT1_sa_cm.nii.gz`
- `--format` or `-f` (optional):
  - Description: Specifies the input format for the images to be aligned.
  - Default: `nifti`
  - Options: `nifti` or `numpy`

## `03_superresolucion.py`
This script is designed to enhance the resolution of short-axis Cardiac MRI images using a pre-trained super-resolution model.

### Usage
```bash
python 03_superresolucion.py --input <path_to_mri_images_dataset_or_single_image> [--format <input_format>] [--single] [--downsample] [--input_suffix <input_suffix>] [--output_suffix <output_suffix>] [--device <device>] [--model_path <path_to_model_file>] [--model_input_size <input_size>]
```

**Arguments**
- `--input` or `-i` (required):
  - Description: Specifies the path to the folder containing the MRI images you want to super-resolve.  
  The folder containing the MRI must look like the following example:
    ```
    mri_scans/  
    ├── PATIENT1/  
    │   └── PATIENT1_sa.nii.gz  
    └── PATIENT2/  
        └── PATIENT2_sa.nii.gz
    ```
    Where the **folder name is the same as the filename of the `.nii.gz`**
  - Note: If you are processing a single image, provide the full path to the image file and use the `--single` flag.
  - Example: If your images are located in a folder called `mri_scans`, you would use `--input /path/to/mri_scans`
- `--format` or `-f` (optional):
  - Description: Specifies the input format for the images to be super-resolved.
  - Default: `nifti`
  - Options: `nifti` or `numpy`
- `--single` or `-s` (optional):
  - Description: If this flag is set, the script will process a single 3D image file instead of a folder containing multiple images.
  - Default: `False`
  - Example: If you want to process a single image file named `PATIENT1_sa.nii.gz`, you would use `--single` and provide the path to the file with `--input /path/to/PATIENT1_sa.nii.gz`
- `--downsample` or `-d` (optional):
  - Description: Option to downsample the input images before applying super-resolution. This can be useful for testing the model's performance on lower-resolution images. The procedure to do this is shown in the manuscript.
  - Default: `False`
- `--input_suffix` or `-is` (optional):
  - Description: Specifies the suffix that the input files have.
  - Default: `_sa`
  - Example: If your input files are named `PATIENT1_sa.nii.gz`, you would use `--input_suffix _sa` to correctly identify them.
- `--output_suffix` or `-os` (optional):
  - Description: Specifies the suffix to be added to the output files.
  - Default: `_sa_sr`
  - Example: If you set `--output_suffix _sa_sr`, an input file named `PATIENT1_sa.nii.gz` will have its super-resolved version saved as `PATIENT1_sa_sr.nii.gz`
- `--device` (optional):
  - Description: Specifies the device to be used for computation. Options are `cpu` or `cuda`.
  - Default: `cuda`
- `--model_path` (optional):
  - Description: Specifies the path to the pre-trained super-resolution model file.
  - Default: `weights/superres_model.pt`
- `--model_input_size` (optional):
  - Description: Specifies the input size for the super-resolution model.
  - Default: `224`

### Example
Processing a folder of MRI images in NIfTI format:
```bash
python 03_superresolucion.py --input /path/to/mri_scans --input_suffix _sa --output_suffix _sa_sr --format nifti
``` 
Processing a single MRI image in NIfTI format:
```bash
python 03_superresolucion.py --input /path/to/mri_scans/PATIENT1/PATIENT1_sa.nii.gz --single --input_suffix _sa --output_suffix _sa_sr --format nifti
```
