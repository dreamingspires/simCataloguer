# simCataloguer

# Installation

## Clone repo
`git clone --recursive git@github.com:dreamingspires/simCataloguer.git`

## Get NVIDIA GPU Computing Toolkit, CUDA
https://developer.nvidia.com/cuda-11-7-0-download-archive

## Get cuDNN
https://developer.nvidia.com/rdp/cudnn-download

### Install cuDNN
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

## Install pip
https://pip.pypa.io/en/stable/installation/

## Install Poetry
https://python-poetry.org/docs/#installation

## Install deps with poetry
`poetry install` (You may need to run this more than once, it fails to parallel process on occasion)

## Windows specific

`poetry run pip install --extra-index-url https://download.pytorch.org/whl/cu117 --no-deps --force-reinstall torch torchvision`