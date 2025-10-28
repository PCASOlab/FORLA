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
  
Optional:

We use visdom for visualization:
```
pip install visdom
```

We for foundation model including DINO, CLIP, MAE the code will handle the download of the models, as for segment-anything model (SAM), you will need to download it mannually from [SAM offical repo](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

Place it under [./data/SAM/](./data/SAM/), or you can change the dir defination in the code.



Possible Issues

-   In case you encounter any environmental issues (e.g., package version compatibility), you can check out the env file exported from our server [requirements.txt](https://github.com/PCASOlab/FORLA/blob/main/docs/requirements.txt).
    You can use this file to install specific version of a package.