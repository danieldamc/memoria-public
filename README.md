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