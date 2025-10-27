# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
import time
import os
os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_miccai'  # Change this to your target mode

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
from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample, Data_percentage,Evaluation_slots,Max_epoch
# from 
from dataset import io
import pathlib
import argparse
from main.FL.FL_server import Output_root
# Add these flags at the top of your client code
use_fedprox = False  # Set to False to use FedAvg


Enable_student = True

if Evaluation_slots:
    Enable_student = False
mu = 0.01  # Proximal term weight

# Output_root = Output_root + "FL/Dino/"
# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'
# Add these lines at the top of the original script
Fed_iter=100
FED_MAX_ATTEMPTS = 5  # Max attempts for federation
FED_RETRY_DELAY = 1  # Seconds between retries
SHARED_ROOT = Output_root + "shared_models/"
STATUS_DIR = os.path.join(SHARED_ROOT, "status")
os.makedirs(SHARED_ROOT, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)
import copy
# Modified model loading function
GLOBAL_VERSION_FILE = os.path.join(SHARED_ROOT, "global", "version.txt")
# COMPONENTS = ["initializer", "adapter", "processor", "decoder"]
# COMPONENTS = ["initializer",   "processor", "decoder"]
COMPONENTS_load = ["initializer", "adapter", "processor", "decoder"]
COMPONENTS_load_s = ["initializer", "adapter", "processor", "decoder" ]
COMPONENTS = ["initializer", "adapter", "processor", "decoder"]
#** can asllow decdoer if the fed apert is reconstructing the adapted feature


# COMPONENTS_avg = [  "adapter", "processor","decoder"]
# COMPONENTS_avg = [   "processor","decoder"]
COMPONENTS_avg = [ "adapter",  "processor"]




class ClientFedHelper:
    def __init__(self, client_id):
        self.global_state_dict = None  # Add this line
        self.client_id = client_id
        self.client_dir = os.path.join(SHARED_ROOT, f"client_{client_id}")
        self.client_dir_s = os.path.join(SHARED_ROOT, f"client_{client_id}","student")

        self.global_version = self._get_global_version()
        os.makedirs(self.client_dir, exist_ok=True)
        os.makedirs(self.client_dir_s, exist_ok=True)



    def _get_global_version(self):
        try:
            with open(os.path.join(SHARED_ROOT, "global", "version.txt"), "r") as f:
                return int(f.read())
        except:
            return 0

    def save_client_model(self, model,participate=True):
        """Save client model with status timestamp"""
        tmp_dir = os.path.join(self.client_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            for comp in COMPONENTS:
                state_dict = getattr(model, comp).state_dict()
                torch.save(state_dict, os.path.join(tmp_dir, f"{comp}.pth"))
            
            # Atomic replace
            for fname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(self.client_dir, fname)
                os.replace(src, dst)
            
            # Update status
            if participate:
                with open(os.path.join(self.client_dir, "status.txt"), "w") as f:
                    f.write(str(time.time()))
                
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    def save_client_model_s(self, model,participate=True):
        """Save client model with status timestamp"""
        tmp_dir = os.path.join(self.client_dir_s, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            for comp in COMPONENTS:
                state_dict = getattr(model, comp).state_dict()
                torch.save(state_dict, os.path.join(tmp_dir, f"{comp}.pth"))
            
            # Atomic replace
            for fname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, fname)
                dst = os.path.join(self.client_dir_s, fname)
                os.replace(src, dst)
            
            # Update status
            if participate:
                with open(os.path.join(self.client_dir_s, "status.txt"), "w") as f:
                    f.write(str(time.time()))
                
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def load_global_model(self, model):
        """Load global model if updated version exists"""
        current_global = self._get_global_version()
        if current_global <= self.global_version:
            return False
            
        global_dir = os.path.join(SHARED_ROOT, "global")
        try:
            for comp in COMPONENTS_avg:
                path = os.path.join(global_dir, f"{comp}.pth")
                state_dict = torch.load(path, map_location='cpu')
                getattr(model, comp).load_state_dict(state_dict)
            
            # Save global model state dict for FedProx
            # self.global_state_dict = model.state_dict()
            self.global_state_dict = copy.deepcopy(model.state_dict())
            self.global_version = current_global
            return True
        except Exception as e:
            print(f"Error loading global model: {e}")
            return False
        # Add to ClientFedHelper class
    def load_client_model(self, model):
        """Load client-specific model if exists"""
        try:
            # Check if all components exist
            if not all(os.path.exists(os.path.join(self.client_dir, f"{comp}.pth")) 
                    for comp in COMPONENTS_load):
                return False
                
            # Load components
            for comp in COMPONENTS_load:
                path = os.path.join(self.client_dir, f"{comp}.pth")
                state_dict = torch.load(path, map_location='cpu')
                getattr(model, comp).load_state_dict(state_dict)
                
            print("Loaded existing client model")
            return True
        except Exception as e:
            print(f"Error loading client model: {str(e)}")
            return False
    def load_client_model_s(self, model):
        """Load client-specific model if exists"""
        try:
            # Check if all components exist
            if not all(os.path.exists(os.path.join(self.client_dir_s, f"{comp}.pth")) 
                    for comp in COMPONENTS_load_s):
                return False
                
            # Load components
            for comp in COMPONENTS_load_s:
                path = os.path.join(self.client_dir_s, f"{comp}.pth")
                state_dict = torch.load(path, map_location='cpu')
                getattr(model, comp).load_state_dict(state_dict)
                
            print("Loaded existing client model")
            return True
        except Exception as e:
            print(f"Error loading client model: {str(e)}")
            return False
