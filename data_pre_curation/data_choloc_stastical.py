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
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature,train_test_list_dir
from working_dir_root import train_sam_feature_dir
# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'
Creat_balance_set = False
Save_sam_feature = False

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

# dataroot = "../dataset/CostMatrix/"
torch.set_num_threads(2)
 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')

def is_external_drive(drive_path):
    # Check if the drive is a removable drive (usually external)
    return os.path.ismount(drive_path) and shutil.disk_usage(drive_path).total > 0

def find_external_drives():
    # List all drives on the system
    drives = [d for d in os.listdir('/') if os.path.isdir(os.path.join('/', d))]

    # Filter out external drives and exclude certain paths
    external_drives = [drive for drive in drives if is_external_drive(os.path.join('/', drive))
                       and not drive.startswith(('media', 'run', 'dev'))]

    return external_drives

# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    pass
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model_infer = model_infer_MCT._Model_infer(GPU_mode,num_gpus)
# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='train')
 
read_id = 0
print(Model_infer.resnet)
print(Model_infer.VideoNets)

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
    all_counter= np.sum( dataLoader.this_raw_labels,axis=0)
    all_frame_sum+=all_counter
    label_none =0
    if Save_sam_feature == True:
        this_features= dataLoader.this_features
        sam_pkl_file_name = dataLoader.this_file_name
        sam_pkl_file_path = os.path.join(train_sam_feature_dir, sam_pkl_file_name)

        with open(sam_pkl_file_path, 'wb') as file:
            pickle.dump(this_features, file)
            print("sam Pkl file created:" +sam_pkl_file_name)


    if Creat_balance_set == True:

        if read_id ==0:
            training_list.append(dataLoader.this_file_name)
            label_sum_appended += dataLoader.this_label
        elif (dataLoader.this_label is not None):
            sorted_indices_asc = np.argsort(label_sum_appended) # find waht is lacking in the set
            if dataLoader.this_label[sorted_indices_asc[dataLoader.obj_num-1]] == 0 or dataLoader.this_label[sorted_all[0]]==1 or dataLoader.this_label[sorted_indices_asc[0]] == 1: # if not  the most case:
                training_list.append(dataLoader.this_file_name)
                label_sum_appended += dataLoader.this_label
        print(label_sum_appended)
        if dataLoader.this_label is  None:
             label_none+=1
             print('None label' + str(label_none))
        
    if dataLoader.all_read_flag ==1:
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        print("all_labels")
        print(label_sum)
        print("all framez")
        print(all_frame_sum)
        dataLoader.all_read_flag = 0
        read_id=0
    if Creat_balance_set == True:
        if np.all(label_sum_appended > 800):
            
                    
                pkl_file_name = "train_set_balance_no8K.pkl"
                pkl_file_path = os.path.join(train_test_list_dir, pkl_file_name)
                with open(pkl_file_path, 'wb') as file:
                                pickle.dump(training_list, file)
                                print("sam Pkl file created:" + pkl_file_name)

        # break
    if read_id % 1== 0   :
        print(" epoch" + str (epoch) )
    

    read_id+=1
    # print(labels)

    # pass





















