# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import pickle
# from model import CE_build3  # the mmodel
from time import time
import os
print("Current working directory:", os.getcwd())
import shutil
# from train_display import *
# the model
# import arg_parse
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from model import  model_experiement, model_infer,model_infer_T,model_infer_MCT
from working_dir_root import Output_root
from dataset.dataset import myDataloader,MICCAI_data
from display import Display
import torch.nn.parallel
import torch.distributed as dist
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature,train_test_list_dir
from working_dir_root import train_sam_feature_dir
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import learningR,learningR_res,SAM_pretrain_root
# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'
Creat_balance_set = False
Save_sam_feature = False
MedSAM_flag = True
Create_sam_feature = False
GPU_mode = True
if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")
# sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
# sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
# self.inter_bz =1
# model_type = "vit_h"
# model_type = "vit_l"
model_type = "vit_b"

# model_type = "vit_t"
# sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

# mobile SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# self.predictor = SamPredictor(self.sam) 
Vit_encoder = sam.image_encoder
Vit_encoder.to(device)



if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
   
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
if GPU_mode ==True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")

   
# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= False,Save_pkl = True,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='train')
 
read_id = 0
 
epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(GPU_mode)
epoch =0
features = None
label_sum =0
all_frame_sum = 0
label_sum_appended = 0
training_list=[]
# Initialize counters for each category
category_counters = {category: 0 for category in dataLoader.categories}
sorted_all= np.argsort(dataLoader.categories_count) 
while (1):
    start_time = time()
    input_videos, labels= dataLoader.read_a_batch()
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    label_sum += labels
    if MICCAI_data == False:
        all_counter= np.sum( dataLoader.this_raw_labels,axis=0)
        all_frame_sum+=all_counter
    label_none =0
     
        
    if dataLoader.all_read_flag ==1:
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        print("all_labels")
        print(label_sum)
        print("all framez")
        if MICCAI_data == False:

            print(all_frame_sum)
        dataLoader.all_read_flag = 0
        break
        read_id=0
     
    if read_id % 1== 0   :
        print(" epoch" + str (epoch) )
    

    read_id+=1
    # print(labels)

    # pass





















