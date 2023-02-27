# simCataloguer

# System Requirements

Please note that you may require a large amount of VRAM to use this library. Data we have gathered so far: 8Gb is not enough but 24Gb is sufficient.

# Installation

## Clone repo
`git clone --recursive git@github.com:dreamingspires/simCataloguer.git`

You may need to install zlib, and graphics driver

## Get NVIDIA GPU Computing Toolkit, CUDA
https://developer.nvidia.com/cuda-11-7-0-download-archive

## Get cuDNN
https://developer.nvidia.com/rdp/cudnn-download

### Install cuDNN
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

## Install python 3.9
https://www.python.org/downloads/release/python-3913/

## Install pip (may not be required, check with pip --version)
https://pip.pypa.io/en/stable/installation/

## Install Poetry
https://python-poetry.org/docs/#installation

### Add poetry to path using on screen instructions

### Reboot shell

### Check with poetry --version

`poetry config virtualenvs.in-project true`

## Install deps with poetry
`poetry install` (You may need to run this more than once, it fails to parallel process on occasion)

## Windows specific

`poetry run pip install --extra-index-url https://download.pytorch.org/whl/cu117 --no-deps --force-reinstall torch torchvision`

# Run test

Go into the examples folder and run:

`poetry run python test_writer.py`

# Credits

Developed by Dreaming Spires Software Development Ltd

For more details of the type of projects we develop, please contact:

contact@dreamingspires.dev

or visit:

https://dreamingspires.dev/
