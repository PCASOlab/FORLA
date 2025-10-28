# Dataset Preparation

We curated train/test data seperated for each dataset seperately, and stored them as pkl files to load for training.

in [data](../data/) folder we sampled a small number of data points to demonstrate the curated data structure, and the usage of training code. For demo on training using sampled small size data, the batch size is set to 1 in the training config of each specific dataset, for instance in  [./working_para/working_dir_root_train_pascal_p.py](./working_para/working_dir_root_train_pascal_p.py), please change the batch size for actual experiment.

The steps on preparing full large scale data for each dataset is listed below.


## Abdominal surgery dataset
Download link with preprocessing code & instruction in package [Download](https://upenn.box.com/s/493licnenrssjukuvok5zkvc5cqmx1nh) 

following the instruction, generated PKLS for train and test, and put them in [/data/MICCAI/video_clips_pkl/](../data/MICCAI/video_clips_pkl/) and [/data/MICCAI_selected_GT/pkl/](../data/MICCAI_selected_GT/pkl/) 

## Thoracic surgery dataset
Download link with preprocessing code & instruction in package [Download](https://upenn.box.com/s/rxqoi81j5ar4l343ob6otdxxeusc3iwg)

following the instruction, generated PKLS for train and test, and put them in [/data/Thoracic/pkl/](../data/Thoracic/pkl/) and [/data/Thoracic/annotated/pkl/](../data/Thoracic/annotated/pkl/) 


## Cholec surgery dataset
Download link with preprocessing code & instruction in package [Download](https://upenn.box.com/s/ree79lv9fbibjbs2b8mkwzz207oqu6jj)

following the instruction, generated PKLS for train and test, and put them in [/data/cholec80/output_pkl_croped/](../data/cholec80/output_pkl_croped/) and [/data/cholecseg8k_working/output_pkl_croped/](../data/cholecseg8k_working/output_pkl_croped/) 

## PASCAL VOC 2012

Please download the processed dataset from [PASCAL official data host](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html). 

Unzip the downloaded `tgz` file. We do not need the folders with `saliency` in the name.
Please only take `images/`, `SegmentationClass/`, `SegmentationClassAug/`, `sets/` folders.

Finally, we also need the instance segmentation masks for evaluation.
Please download this [file](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), unzip it, take the `VOCdevkit/VOC2012/SegmentationObject/` folder.

Then run the data precuration code [data_pascal.py](../data_pre_curation/data_pascal.py) to generate PKLS for train and test, and put them in [/data/PASCAL/VOCtrainval_11-May-2012/pkl/train](../data/PASCAL/VOCtrainval_11-May-2012/pkl/train/) and [/data/PASCAL/VOCtrainval_11-May-2012/pkl/val](../data/PASCAL/VOCtrainval_11-May-2012/pkl/val/) 


## MS COCO 2017

Please download the data from their [website](https://cocodataset.org/#download).
Specifically, we need `2017 Train images [118K/18GB]`, `2017 Val images [5K/1GB]`, `2017 Train/Val annotations [241MB]`.

Unzip them and you will get 2 images folders `train2017/` and `val2017/`, and 2 annotation files `instances_train2017.json` and `instances_val2017.json`.
Please put the image folders under `./data/COCO/images/`, and the annotations json files under `./data/COCO/annotations/`.

Then run the data precuration code [data_coco.py](../data_pre_curation/data_coco.py) to generate PKLS for train and test, and put them in [/data/COCO/pkl/train](../data/COCO/pkl/train/) and [/data/COCO/pkl/val](../data/COCO/pkl/val/) 

## YTVIS data

Download the instance segmentation (2019) version from Youtube object segmentation chanllenge official website [Download](https://youtube-vos.org/dataset/).

Then run the data precuration code [data_ytvis.py](../data_pre_curation/data_ytvis.py) and [data_ytvis_test.py](../data_pre_curation/data_ytvis.py) seperately to generate PKLS for train and test, and put them in [/data/YTVOS/pkl/](../data/YTVOS/pkl/) and [/data/YTVOS/pkl_test/](../data/YTVOS/pkl_test/) 

## YTOBJ data

Please download YTOBJ data from the official which contains videos for 10 objects and annotation with .mat format (https://calvin-vision.net/datasets/youtube-objects-dataset/). 

Unzip the files and use [data_ytobj.py](../data_pre_curation/data_ytobj.py) and [data_ytobj_box_test.py](../data_pre_curation/data_ytobj_box_test.py) seperately to generate PKLS for train and test, and put them in [/data/YTOBJ/pkl/](../data/YTOBJ/pkl/) and [/data/YTOBJ/pkl_test/](../data/YTOBJ/pkl_test/) 
