# Dataset Preparation

We curated train/test data seperated for each dataset seperately, and stored them as pkl files to load for training.

in [data](./data/) folder we sampled a small number of data points to demonstrate the curated data structure, and the usage of training code. 

The steps on preparing full large scale data for each dataset is listed below.


## PASCAL VOC 2012

 
Please download the processed dataset from [PASCAL official data host](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). 

Unzip the downloaded `tgz` file. We do not need the folders with `saliency` in the name.
Please only take `images/`, `SegmentationClass/`, `SegmentationClassAug/`, `sets/` folders.

Finally, we also need the instance segmentation masks for evaluation.
Please download this [file](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), unzip it, take the `VOCdevkit/VOC2012/SegmentationObject/` folder.

The run the data precuration code [data_pascal.py](../data_pre_curation/data_pascal.py)
