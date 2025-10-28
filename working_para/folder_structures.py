# PC
# working_root = "/media/guiqiu/Installation/database/surgtoolloc2022_dataset/"
#
# Dataset_video_root =  working_root + "_release/training_data/video_clips/"
# Dataset_video_pkl_root = working_root + "_release/training_data/video_clips_pkl/"
# Dataset_label_root =  working_root + "_release/training_data/"
# config_root =   working_root + "config/"
# Output_root =   "/media/guiqiu/Installation/database/output/"
# #
import json
import os
# Remote
Linux_computer= True
working_root = "C:/2data/"
 
if Linux_computer == True:
    working_root = "/home/guiqiu/GQ_project/weakly_supervised/Data/"
    working_root = "/data/"

 
from .machine_configs import get_machine_path

working_pcaso_raid = get_machine_path("base")


Dataset_video_root =  working_root + "training_data/video_clips/"

Dataset_video_pkl_merge_root  = working_root  + "training_data/video_clips_pkl_merge/"
Dataset_video_pkl_flow_root = working_root + "training_data/video_clips_pkl_flow/"
Dataset_video_pkl_cholec = working_root + "training_data/video_clips_pkl_cholec/"
Dataset_video_pkl_cholec = working_root + "cholec80/output_pkl/"
Dataset_video_pkl_thoracic =  'C:/2data/Raw_data_Chrocic/data/' + 'output_pkl/'
Dataset_video_pkl_endovis = "C:/2data/endovis/2017/pkl/"


if Linux_computer == True:
    # Dataset_video_pkl_cholec = working_pcaso_raid + "cholec80/output_pkl/"
    Dataset_video_pkl_cholec = working_pcaso_raid + "cholec80/output_pkl_croped/"


Dataset_label_root =  working_root + "training_data/"
config_root =   working_root + "config/"
Output_root =   working_pcaso_raid+"output/"
SAM_pretrain_root = working_pcaso_raid+"SAM/"
DINO_pretrain_root = working_root + "output_vdst/DINO/"

output_folder_sam_feature = working_root+ "cholec80/output_sam_features/"
# Dataset_video_pkl_cholec8k =  working_root+ "cholecseg8k_working/output_pkl/"
Dataset_video_pkl_cholec8k =  working_root+ "cholecseg8k_working/output_pkl_croped/"


output_folder_sam_feature_cholec8k = working_root + "cholecseg8k_working/output_sam_features/"
output_folder_sam_masks = Output_root + "sam_masks"

train_test_list_dir = working_root + "output_vdst/train_test_list/"
train_sam_feature_dir = working_root+ "cholec80/train_sam_feature/"
sam_feature_OLG_dir= working_root+ "cholec80/sam_feature_OLG/"





if Linux_computer == True:
    output_folder_sam_feature = working_pcaso_raid+ "cholec80/output_sam_features/"
    # Dataset_video_pkl_cholec8k =  working_pcaso_raid+ "cholecseg8k_working/output_pkl/"
    Dataset_video_pkl_cholec8k =  working_pcaso_raid+ "cholecseg8k_working/output_pkl_croped/"

    Dataset_video_pkl_root = working_pcaso_raid + "MICCAI/video_clips_pkl/"
    Dataset_label_root =  working_pcaso_raid + "MICCAI/"
    
    Dataset_video_pkl_MICCAIv2 =  "/media/guiqiu/surgvu24pkl/"
    Dataset_video_pkl_endovis = working_pcaso_raid + "Endovis/pkl/"
    Dataset_video_pkl_MICCAI_test = working_pcaso_raid + "MICCAI_selected_GT/pkl/"
    # Dataset_video_pkl_thoracic = working_pcaso_raid + "choracic/pkl/"
    Dataset_video_pkl_thoracic = working_pcaso_raid + "Thoracic/pkl/"
    Thoracic_select =  working_pcaso_raid + "Thoracic/selected/"
    Dataset_video_pkl_thoracic_test =  working_pcaso_raid + "Thoracic/annotated/pkl/"


    Davis_root = working_pcaso_raid + "DAVIS/"
    root_YTVOS = working_pcaso_raid + "YTVOS/"
    root_YTOBJ = working_pcaso_raid + "YTOBJ/"
    COCO_root = working_pcaso_raid + "COCO/"
    PASCAL_root = working_pcaso_raid + "PASCAL/VOCtrainval_11-May-2012/"
    POEM_root = working_pcaso_raid + "POEM/data/"

    Dataset_video_pkl_pascal = PASCAL_root + "pkl/train/"
    # PASCAL_root = working_pcaso_raid + "PASCAL/VOCtrainval_11-May-2012/"
    Dataset_video_pkl_coco = COCO_root + "pkl/train/"
    Dataset_video_pkl_poem = POEM_root + "pkl/train/"


    # root_YTOBJ = "C:/2data/TCAMdata/youtube video/download2/"
    Dataset_video_pkl_Davis = working_pcaso_raid + "DAVIS/pkl/"
    Dataset_video_pkl_YTVOS = working_pcaso_raid + "YTVOS/pkl/"
    Dataset_video_pkl_MOVIE = working_pcaso_raid + "MOVi/MOVi-E/pkl/train/"
    Dataset_video_pkl_MOVID = working_pcaso_raid + "MOVi/MOVi-D/pkl/train/"

    Dataset_video_pkl_YTOBJ = working_pcaso_raid + "YTOBJ/pkl/"


    Slots_prototype_pkl  =  working_pcaso_raid +  "out/pkl/"
    Slots_label_dir  =  working_pcaso_raid +  "out/labels/"

    

    output_folder_sam_feature_cholec8k = working_pcaso_raid + "cholecseg8k_working/output_sam_features/"
    train_sam_feature_dir = working_pcaso_raid+ "cholec80/train_sam_feature/"
    # sam_feature_OLG_dir= working_pcaso_raid+ "cholec80/sam_feature_OLG/"
    sam_feature_OLG_dir= working_pcaso_raid+ "sam_feature_OLG/"
    sam_feature_OLG_dir2= working_pcaso_raid+ "sam_feature_OLG2/"
    sam_feature_OLG_dir3= working_pcaso_raid+ "sam_feature_OLG3/"
    # sam_feature_OLG_dir = sam_feature_OLG_dir2


    # video_saur_pretrain = working_pcaso_raid + "model_trained/videosaur-ytvis.ckpt"
    video_saur_pretrain = working_pcaso_raid + "model_trained/videosaur-movi-e.ckpt"

    # video_saur_pretrain = working_pcaso_raid + "model_trained/videosaur_dinov2.ckpt"


    slot_out_dir = working_pcaso_raid + "MICCAI_slots_merged_mask"