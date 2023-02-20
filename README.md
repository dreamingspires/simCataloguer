# simCataloguer

# Installation

## Get NVIDIA GPU Computing Toolkit, CUDA
https://developer.nvidia.com/cuda-toolkit

## Get cuDNN
https://developer.nvidia.com/rdp/cudnn-download

## Install Poetry
https://python-poetry.org/docs/#installation

## Install deps with poetry
`poetry install`

# Windows

`git clone --recursive git@github.com:dreamingspires/MAKE-Writer.git`

get pip

get poetry

`poetry install` (You may need to run this more than once, it fails to parallel process on occasion)

`poetry run pip install --extra-index-url https://download.pytorch.org/whl/cu117 --no-deps --force-reinstall torch torchvision`