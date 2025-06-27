# cdt_alicante311 Environment Setup

This guide provides instructions on how to install and use the `cdt_alicante311` Conda environment.

## Prerequisites

Ensure that you have Conda installed on your system. If Conda is not installed, you can install Miniconda or Anaconda:

### Install Miniconda (Recommended)
Miniconda is a minimal Conda installer that includes only Conda and essential dependencies:

**Windows:**
```sh
winget install Anaconda.Miniconda3
```

**Mac/Linux:**
```sh
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh
```

Alternatively, install Anaconda from [Anaconda's official website](https://www.anaconda.com/products/distribution).

## Creating the Environment

Once Conda is installed, create the `cdt_alicante311` environment using the provided YAML file:

```sh
conda env create -f environment.yml
```

This will create a Conda environment with Python 3.11 and the required packages.

## Activating the Environment

To activate the environment, run:
```sh
conda activate cdt_alicante311
```

## Verifying Installation

To check if the installation was successful, run:
```sh
python -c "import numpy, scipy, tqdm, cobyqa, matplotlib; print('All packages imported successfully!')"
```

If there are no errors, the environment is set up correctly.

## Deactivating and Removing the Environment

To deactivate the environment, use:
```sh
conda deactivate
```

To remove the environment completely:
```sh
conda env remove -n cdt_alicante311
```
