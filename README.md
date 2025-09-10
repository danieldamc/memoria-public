# memoria-public
This is the public repository for my undergrad thesis titled:

Superresolución y alineamiento de imágenes de eje corto del corazón obtenidas por resonancia magnética cardiovascular

[Manuscript](https://repositorio.usm.cl/entities/tesis/090340cd-f955-4b98-91e5-2c340599f218)

# Setup
### Requisites
- Linux OS
- Nvidia GPU + CUDA CuDNN
- Python 3

### Getting Started
- Clone this repo
```bash
git@github.com:danieldamc/memoria-public.git
```
- Install the dependencies, For Conda users, you can create a new enviorment with:
```bash
conda env create -f environment.yml
```
and then activate the environment with:
```bash
conda activate memoria
```
- Download the dataset from [M&M's Challenge](https://www.ub.edu/mnms/)
- Download models weights:
```bash
python weights/download_weights.py
```
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
    │   └── PATIENT1.nii.gz  
    └── PATIENT2/  
        └── PATIENT2.nii.gz
    ```
    Where the **folder name is the same as the filename of the `.nii.gz`**
  - Example: If your images are located in a folder called `mri_scans`, you would use `--folder_path /path/to/mri_scans`
- `--model_path` or `-m` (optional):
  - Description: Specifies the path to the pre-trained model file that will be used for segmentation.
  - Default: `weights/segmentation_model.pt`
- `--input_suffix` or `-is` (optional):
  - Description: Defines the suffix to add to the input file names before segmentation.
  - Default: `_sa`
  - Example: If you set `--suffix _raw`, an image named `PATIENT1_raw.nii.gz` will be processed.
- `--output_suffix` or `-os` (optional):
  - Description: Defines the suffix to add to the output file names after segmentation.
  - Default: `_sa_seg`
  - Example: If you set `--suffix _segmented`, an image named `PATIENT1.nii.gz` will have its segmented version saved as `PATIENT1_segmented.nii.gz`
- `--format` or `-f` (optional):
  - Description: Specifies the input format for the segmented images.
  - Default: `nifti`
  - Options: `nifti` or `numpy`

## `02_aligner.py`