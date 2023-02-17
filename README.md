# MAKE Writer

# Installation

## Get NVIDIA GPU Computing Toolkit, CUDA
https://developer.nvidia.com/cuda-toolkit

## Get cuDNN
https://developer.nvidia.com/rdp/cudnn-download

## Install Poetry
https://python-poetry.org/docs/#installation

## Install deps with poetry
`poetry install`

# For WSL users

Install cuda 11 here not 12
## Step 1

https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2

## Step 2
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#cudnn-package-manager-installation-overview
## Step 3
`sudo apt-get -y install cuda`

# Windows

`git clone --recursive git@github.com:dreamingspires/MAKE-Writer.git`

get pip

get poetry

`poetry install` (You may need to run this more than once, it fails to parallel process on occasion)

`poetry run pip uninstall -y torch torchvision`

`poetry run pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117`

Same for tensorflow
poetry run pip install tensorflow==2.10