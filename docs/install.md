# Install

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment setup:

```
conda create -n forla python=3.10.12
conda activate forla
```

Then install PyTorch which is compatible with your cuda setting.
In our experiments, we use PyTorch 2.6.0 and CUDA 12.6:

```
conda install pytorch==2.6.0 torchvision==12.6 torchaudio==2.6.0 cudatoolkit=12.6 -c pytorch -c conda-forge
pip install pytorch-lightning==2.5.1 torchmetrics==1.7.1
```
 

This will automatically install packages necessary for the project.
Additional packages are listed as follows:

```
pip install pycocotools scikit-image lpips chardet omegaconf
pip install pytorch-fid einops transformers
```
  

```
Optional:

We use visdom for visualization:
pip install visdom

## Possible Issues

-   In case you encounter any environmental issues (e.g., package version compatibility), you can chech out to the env file exported from my server [requirements.txt](https://github.com/PCASOlab/FORLA/blob/main/docs/requirements.txt).
    You can use this file to install specific version of a package.