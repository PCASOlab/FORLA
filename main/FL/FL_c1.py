# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
import time
import os
# os.environ['REQUESTS_CA_BUNDLE'] = '/home/guiqiu/GQ_project/huggingface-co-chain.pem'
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_miccai'  # Change this to your target mode

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_coco'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_poem'  # Change this to your target mode

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_cholec'  # Change this to your target mode
os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_mix2'  # Change this to your target mode

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_coco'  # Change this to your target mode

# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_pascal'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_ytobj'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_ytobj'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_ytvos'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_ytvos'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_pascal'  # Change this to your target mode


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
import eval_slots
from model import  model_experiement, model_infer_slot_att
# from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
import scheduler
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Batch_size,selected_data,Display_down_sample, Data_percentage,Evaluation_slots,Max_epoch
from dataset import io
import pathlib
import argparse
from main.FL.FL_client import Fed_iter,ClientFedHelper,use_fedprox,mu,COMPONENTS_avg,Output_root,Enable_student
from main.FL.FL_server import Average_student
# Output_root = Output_root + "FL/Dino/"
Gpu_selection ='0'

CLIENT_ID = 1
DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_en.yml"
DEFAULT_CONFIG_s = "videosaur_m/configs/videosaur/ytvis3_MLP_en_s.yml"

Fed_student = Average_student



dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
Output_root = Output_root+ "MI+Cho" + dataset_tag + str(CLIENT_ID) + "/"
io.self_check_path_create(Output_root)


RESULT_FINISHED = 0
RESULT_TIMEOUT = 1

CHECKPOINT_SUBDIR = "checkpoints"
TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"
 


DEFAULT_LOG_DIR = "./logs"

parser = argparse.ArgumentParser()
parser.add_argument(
    "config", nargs="?", default=DEFAULT_CONFIG, help="Configuration to run"
)
 
parser.add_argument("--config_overrides_file", help="Configuration to override")
 
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")


parser_s = argparse.ArgumentParser()
parser_s.add_argument(
    "config", nargs="?", default=DEFAULT_CONFIG_s, help="Configuration to run"
)
 
parser_s.add_argument("--config_overrides_file", help="Configuration to override")
 
parser_s.add_argument("config_overrides", nargs="*", help="Additional arguments")



import pickle

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
# torch.set_num_threads(8)
 # create the model

if Visdom_flag == True:
    from visual import VisdomLinePlotter

    plotter = VisdomLinePlotter(env_name='path finding training Plots')
 
############ for the linux to find the extenral drive
 
############ for the linux to find the extenral drive
# Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,slot_ini="random",gpu_selection=Gpu_selection,pooling="max",TPC=True)
# Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,slot_ini="random",Foundation_M="ensemble",gpu_selection=Gpu_selection,pooling="max",TPC=True)
Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),parser_s.parse_args(),GPU_mode,num_gpus,slot_ini="random",Foundation_M="ensemble",foundation_list  = ["DINO","SAM","MAE","CLIP"],fusion_method='moe_layer',gpu_selection=Gpu_selection,TPC=True)

# Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,Using_contrast=False,Using_SP_regu = False,Using_SP = True,Using_slot_bert=True,slot_ini= "binder+merger",cTemp=1.1,gpu_selection=Gpu_selection,pooling="max",TPC=True)
device = Model_infer.device

# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',starting_percentage=0.0,Device=device)

if Continue_flag == False:
    pass
    # Model_infer.VideoNets.apply(weights_init)
else:
    
    
    pass
    # Model_infer.model.initializer.load_state_dict(torch.load(Output_root + 'initializer' + loadmodel_index,map_location='cuda:0'))
    # Model_infer.model.encoder.load_state_dict(torch.load(Output_root + 'encoder' + loadmodel_index,map_location='cuda:0' ))
    # Model_infer.model.processor.load_state_dict(torch.load(Output_root + 'processor' + loadmodel_index,map_location='cuda:0' ))
    # Model_infer.model.presence_nn.load_state_dict(torch.load(Output_root + 'presence_nn' + loadmodel_index,map_location='cuda:0' ))
    # Model_infer.model.decoder.load_state_dict(torch.load(Output_root + 'decoder' + loadmodel_index,map_location='cuda:0' ))
    # Model_infer.model.temporal_binder.load_state_dict(torch.load(Output_root + 'temporal_binder' + loadmodel_index,map_location='cuda:0' ))
 
read_id = 0
# print(Model_infer.resnet)
# print(Model_infer.VideoNets)

epoch = 0
# transform = BaseTransform(  Resample_size,(104/256.0, 117/256.0, 123/256.0))
# transform = BaseTransform(  Resample_size,[104])  #gray scale data
iteration_num = 0
#################
#############training
saver_id =0
displayer = Display(GPU_mode,Model_infer)
epoch =0
features =None
visdom_id=0

# Modified training loop with federated learning integration
round_number = 0
last_global_version = 0
last_fed_version = 0
client = ClientFedHelper(client_id=CLIENT_ID)
if not client.load_client_model(Model_infer.model) :
    # If no client model exists, load global model
    print("Don't have local client models yet!!!!")
    
    proceed = input("Do you want to load the global model instead? (y/n): ").strip().lower()
    if proceed == 'y':
        client.load_global_model(Model_infer.model)
    else:
        print("Aborting model loading.")
if not client.load_client_model_s(Model_infer.model_s):
    # If no client model exists, load global model
    print("Don't have local client student models yet!!!!")
    
    proceed = input("Do you want to load the global model instead? (y/n): ").strip().lower()
    if proceed == 'y':
        client.load_global_model(Model_infer.model_s)
    else:
        print("Aborting model loading.")
while (1):
    
    
    # Original training iteration
    start_time = time.time()
    start_time = time.time()
    input_videos, labels= dataLoader.read_a_batch(this_epoch= epoch)
    input_videos_GPU = torch.from_numpy(np.float32(input_videos))
    labels_GPU = torch.from_numpy(np.float32(labels))
    input_videos_GPU = input_videos_GPU.to (device)
    labels_GPU = labels_GPU.to (device)
    input_flows = dataLoader.input_flows*1.0/ 255.0
    input_flows_GPU = torch.from_numpy(np.float32(input_flows))  
    input_flows_GPU = input_flows_GPU.to (device)
    if Load_feature ==True:
        features = dataLoader.features.to (device)
    Model_infer.forward(input_videos_GPU,  dataLoader.batch_files ,features,epoch=epoch,read_id=read_id,Output_root=Output_root,Enable_student=Enable_student)
    end_time = time.time()

    print(f"Time taken for forward pass: {end_time - start_time:.4f} seconds")
   
    lr=scheduler.cyclic_learning_rate(current_epoch=epoch,max_lr=Max_lr,min_lr=learningR,cycle_length=4)
    print("learning rate is :" + str(lr))
    if Evaluation == False and Evaluation_slots==False:
        # Modify the optimization call in your training loop
        if use_fedprox:
            Model_infer.optimization(
                labels_GPU, 
                global_state_dict=client.global_state_dict,
                components_avg=COMPONENTS_avg,
                mu=mu,
                epoch=epoch,read_id=read_id,
                Enable_student=Enable_student,
            )
        else:
            Model_infer.optimization(labels_GPU,epoch=epoch,read_id=read_id,Enable_student=Enable_student,)

    if  Save_feature_OLG== True:
        this_features= Model_infer.f[Batch_size-1].permute(1,0,2,3).half()
        sam_pkl_file_name = dataLoader.this_file_name
        sam_pkl_file_path = os.path.join(sam_feature_OLG_dir, sam_pkl_file_name)

        with open(sam_pkl_file_path, 'wb') as file:
            pickle.dump(this_features, file)
            print("sam Pkl file created:" +sam_pkl_file_name)
    if Save_sam_mask == True:
         
        this_mask= Model_infer.sam_mask.half()
        mask_pkl_file_name = dataLoader.this_file_name
        mask_pkl_file_path = os.path.join(output_folder_sam_masks, mask_pkl_file_name)

        with open(mask_pkl_file_path, 'wb') as file:
            pickle.dump(this_mask, file)
            print("sam Pkl file created:" +mask_pkl_file_name)


    if Display_flag == True and read_id%Display_down_sample == 0:
        dataLoader.labels  = dataLoader.labels * 0 +1
        displayer.train_display(Model_infer,dataLoader,read_id,Output_root)
        print(" epoch" + str (epoch) )
        

    
        

        # break
    

    if Evaluation == False and Evaluation_slots == False:
        
        if read_id % 50== 0 and Visdom_flag == True  :
            
            plotter.plot(str(CLIENT_ID)+'l0', str(CLIENT_ID)+'l0', str(CLIENT_ID)+'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            # if Enable_student:
            plotter.plot(str(CLIENT_ID)+'1ls',str(CLIENT_ID)+ '1ls', str(CLIENT_ID)+'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
            plotter.plot(str(CLIENT_ID)+'1lp', str(CLIENT_ID)+'1lp', str(CLIENT_ID)+'l1p', visdom_id, Model_infer.lossDisplay_p.cpu().detach().numpy())

        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

  
    # Modified model saving with presence_nn
    if (read_id % Fed_iter) == 0 and not Evaluation_slots:
        # Save client model
        client.save_client_model(Model_infer.model)
        client.save_client_model_s(Model_infer.model_s)

        saver_id = 0 if saver_id > 1 else saver_id + 1
        if Fed_student:
            if client.load_global_model(Model_infer.model_s):
                    print("Loaded new global model!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            if client.load_global_model(Model_infer.model):
                    print("Loaded new global model!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    read_id+=1
    visdom_id+=1
    # Federated averaging logic at epoch end
    if dataLoader.all_read_flag == 1:
        
        
        Save_feature_OLG = False
        #remove this for none converting mode
        epoch +=1
        # Model_infer.scheduler.step()
        # Model_infer.schedulers.step()
        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        if Evaluation_slots:
            read_id = 0
        # read_id=0

        if Evaluation:
            break
        if Save_feature_OLG: 
            break
        if epoch > Max_epoch  :
            output_file = eval_slots.process_metrics_from_excel(Output_root + "/metrics_video.xlsx", Output_root)

            break
    # ... [Rest of original training loop] ...

 
    # print(labels)

    # pass
 