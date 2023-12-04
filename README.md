# McMini

Miniscope data processing pipeline for the McDonald Lab (CCBN @ ULethbridge). This repo is a fork of _CaImAn_, with extra functionalities provided under the module `McMini`.

## Installation
Follow [this guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install Conda.
If not done so already, install mamba in the base env:
```
conda install -n base -c conda-forge mamba
```
Clone and install a dev version of this repo:
```
git clone https://github.com/LelouchLamperougeVI/CaImAn
cd CaImAn/
mamba env create -f environment.yml -n caiman
source activate caiman
pip install -e .
```

## Usage
### Preprocessing
Imaging data preprocessing follows from _CaImAn_. Users are directed to [this page](https://caiman.readthedocs.io/en/latest/index.html) for excellent instructions on the preprocessing steps. A tutorial has been provided under `tutorials/preprocess.ipynb`.

### Spatial analyses
The two modules `McMini.behav` and `McMini.analyzer` provide a behavioural video and a spatial processing pipeline, respectively. A tutorial on their usage has been provided under `tutorials/analyse.ipynb`.

## Notes
This is a barebones pipeline built using a single recording session kindly provided by Dr J. Quinn Lee.
The intention is to provide a foundation on which future analyses can be built once data becomes available.
Please contact HaoRan Chang regarding details concerning the `McMini` module.