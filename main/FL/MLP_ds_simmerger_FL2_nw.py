# update on 26th July
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
# from model import CE_build3  # the mmodel
import time
import os
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_miccai'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_thoracic'  # Change this to your target mode
# os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_cholec'  # Change this to your target mode
os.environ['WORKING_DIR_IMPORT_MODE'] = 'eval_cholec'  # Change this to your target mode




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
from working_dir_root import Output_root,Linux_computer
from dataset.dataset import myDataloader
from display import Display
import torch.nn.parallel
import torch.distributed as dist
import scheduler
from working_dir_root import GPU_mode ,Continue_flag ,Visdom_flag ,Display_flag ,loadmodel_index  ,img_size,Load_flow,Load_feature
from working_dir_root import Max_lr, learningR,learningR_res,Save_feature_OLG,sam_feature_OLG_dir, Evaluation,Save_sam_mask,output_folder_sam_masks
from working_dir_root import Enable_student,Batch_size,selected_data,Display_down_sample, Data_percentage,Gpu_selection,Evaluation_slots,Max_epoch
from dataset import io
import pathlib
import argparse

# GPU_mode= True
# Continue_flag = True
# Visdom_flag = False
# Display_flag = False
# loadmodel_index = '3.pth'

# Add these lines at the top of the original script
Fed_iter=100
CLIENT_ID = 2
FED_MAX_ATTEMPTS = 5  # Max attempts for federation
FED_RETRY_DELAY = 1  # Seconds between retries
SHARED_ROOT = Output_root + "shared_models/"
STATUS_DIR = os.path.join(SHARED_ROOT, "status")
os.makedirs(SHARED_ROOT, exist_ok=True)
os.makedirs(STATUS_DIR, exist_ok=True)
# Modified model loading function
GLOBAL_VERSION_FILE = os.path.join(SHARED_ROOT, "global", "version.txt")
def get_global_version():
    try:
        with open(GLOBAL_VERSION_FILE, "r") as f:
            return int(f.read())
    except:
        return 0

def update_global_version():
    version = get_global_version() + 1
    with open(GLOBAL_VERSION_FILE, "w") as f:
        f.write(str(version))
    return version
def load_global_model():
    """Load global model with version checking and retries"""
    global_dir = os.path.join(SHARED_ROOT, "global")
    # components = ["initializer", "encoder", "processor",
    #              "temporal_binder", "presence_nn"]
    components = ["initializer", "encoder", "processor",
                  "presence_nn"]
    
    for attempt in range(FED_MAX_ATTEMPTS):
        try:
            # Check if all components exist
            if not all(os.path.exists(os.path.join(global_dir, f"{comp}.pth")) 
                      for comp in components):
                continue
                
            # Load all components
            for comp in components:
                path = os.path.join(global_dir, f"{comp}.pth")
                state_dict = torch.load(path, map_location='cpu')
                getattr(Model_infer.model, comp).load_state_dict(state_dict)
                
            return True
        except Exception as e:
            print(f"Load error (attempt {attempt+1}): {str(e)}")
            time.sleep(FED_RETRY_DELAY)
    
    return False

# Modified client saving function
def save_client_model():
    """Save client model with retries and versioning"""
    client_dir = os.path.join(SHARED_ROOT, f"client_{CLIENT_ID}")
    os.makedirs(client_dir, exist_ok=True)
    
    # Save with temporary files first
    tmp_dir = os.path.join(client_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        components = ["initializer", "encoder", "processor", 
                     "decoder", "temporal_binder", "presence_nn"]
        for comp in components:
            # Get state dict and move to CPU properly
            state_dict = getattr(Model_infer.model, comp).state_dict()
            # Convert tensors to CPU individually
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            torch.save(cpu_state_dict, os.path.join(tmp_dir, f"{comp}.pth"))
        
        # Atomically replace old files
        for fname in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(client_dir, fname)
            os.replace(src, dst)
            
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
def attempt_federation():
    """Attempt federation with partner client with retries"""
    partner_id = 2 if CLIENT_ID == 1 else 1
    lock_file = os.path.join(SHARED_ROOT, "fed.lock")
    
    for attempt in range(FED_MAX_ATTEMPTS):
        try:
            # Create lock file with atomic operation
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.close(fd)
            break
        except FileExistsError:
            if attempt == FED_MAX_ATTEMPTS - 1:
                return False
            time.sleep(FED_RETRY_DELAY)
            continue

    try:
        # Load both client models
        client_models = {}
        for cid in [CLIENT_ID, partner_id]:
            client_dir = os.path.join(SHARED_ROOT, f"client_{cid}")
            if not os.path.exists(client_dir):
                return False
            
            client_models[cid] = {}
            components = ["initializer", "encoder", "processor",
                         "decoder", "temporal_binder", "presence_nn"]
            
            for comp in components:
                path = os.path.join(client_dir, f"{comp}.pth")
                if not os.path.exists(path):
                    return False
                client_models[cid][comp] = torch.load(path, map_location='cpu')

        # Average weights
        avg_weights = {}
        components = ["initializer", "encoder", "processor",
                     "decoder", "temporal_binder", "presence_nn"]
        
        for comp in components:
            avg_weights[comp] = {}
            for key in client_models[CLIENT_ID][comp]:
                avg = (client_models[CLIENT_ID][comp][key] + 
                       client_models[partner_id][comp][key]) / 2
                avg_weights[comp][key] = avg

        # Save global model with temp files
        global_dir = os.path.join(SHARED_ROOT, "global")
        tmp_global = os.path.join(global_dir, "tmp")
        os.makedirs(tmp_global, exist_ok=True)
        
        try:
            for comp in components:
                torch.save(avg_weights[comp], os.path.join(tmp_global, f"{comp}.pth"))
            
            # Atomic replacement
            for fname in os.listdir(tmp_global):
                src = os.path.join(tmp_global, fname)
                dst = os.path.join(global_dir, fname)
                os.replace(src, dst)
                
            return True
        finally:
            shutil.rmtree(tmp_global, ignore_errors=True)
            
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)
# Output_root = Output_root+ "MLP_dynamic_slots_simmerger_FLMi+Cho" + selected_data + str(Data_percentage) + "/"
# loadmodel_index = '3.pth'
dataset_tag = "+".join(selected_data) if isinstance(selected_data, list) else selected_data
Output_root = Output_root+ "MLP_dynamic_slots_simmerger_FLMi+Cho" + dataset_tag + str(Data_percentage) + "/"
io.self_check_path_create(Output_root)
RESULT_FINISHED = 0
RESULT_TIMEOUT = 1

CHECKPOINT_SUBDIR = "checkpoints"
TENSORBOARD_SUBDIR = "tb"
METRICS_SUBDIR = "metrics"

# Define default configuration values
# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_dinov2.yml"
DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_ds_simmerger.yml"
# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3_MLP_cholec.yml"

# DEFAULT_CONFIG = "videosaur_m/configs/videosaur/ytvis3.yml"


DEFAULT_LOG_DIR = "./logs"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
group.add_argument("-q", "--quiet", action="store_true", help="Suppress outputs")
parser.add_argument("-n", "--dry", action="store_true", help="Dry run (no logfiles)")
parser.add_argument(
    "--no-interactive", action="store_true", help="If running in non-interactive environment"
)
parser.add_argument("--no-tensorboard", action="store_true", help="Do not write tensorboard logs")
parser.add_argument(
    "--check-validation", action="store_true", help="Run correctness checks on data used during eval"
)
parser.add_argument(
    "--run-eval-after-training", action="store_true", help="Evaluate after training has stopped"
)
parser.add_argument(
    "--use-optimizations", action="store_true", help="Enable Pytorch performance optimizations"
)
parser.add_argument("--timeout", help="Stop training after this time (format: DD:HH:MM:SS)")
parser.add_argument("--data-dir", help="Path to data directory")
parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Path to log directory")
parser.add_argument(
    "--no-sub-logdirs", action="store_true", help="Directly use log dir to store logs"
)
parser.add_argument(
    "--continue",
    dest="continue_from",
    type=pathlib.Path,
    help="Continue training from this log folder or checkpoint path",
)
parser.add_argument("--config_overrides_file", help="Configuration to override")
parser.add_argument(
    "config", nargs="?", default=DEFAULT_CONFIG, help="Configuration to run"
)
parser.add_argument("config_overrides", nargs="*", help="Additional arguments")

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
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
def add_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = 'module.' + key
        new_state_dict[new_key] = value
    return new_state_dict
# weight init
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
     
############ for the linux to find the extenral drive
external_drives = find_external_drives()

if external_drives:
    print("External drives found:")
    for drive in external_drives:
        print(drive)
else:
    print("No external drives found.")
############ for the linux to find the extenral drive

Model_infer = model_infer_slot_att._Model_infer(parser.parse_args(),GPU_mode,num_gpus,Using_contrast=False,Using_SP_regu = False,Using_SP = True,Using_slot_bert=True,Sim_threshold=0.80,slot_ini= "binder+merger",cTemp=1.1,gpu_selection=Gpu_selection,pooling="max",TPC=True)
device = Model_infer.device

# if GPU_mode == True:
#     if num_gpus > 1:
#         Model_infer.VideoNets = torch.nn.DataParallel(Model_infer.VideoNets)
#     Model_infer.VideoNets.to(device)

# Model.cuda()
dataLoader = myDataloader(img_size = img_size,Display_loading_video = False,Read_from_pkl= True,Save_pkl = False,Load_flow=Load_flow, Load_feature=Load_feature,Train_list='else',Device=device)

if Continue_flag == False:
    pass
    # Model_infer.VideoNets.apply(weights_init)
else:
    # torch.save(Model_infer.model.initializer.state_dict(), Output_root + "initializer" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.encoder.state_dict(), Output_root + "encoder" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.processor.state_dict(), Output_root + "processor" + str(saver_id) + ".pth")
    #     torch.save(Model_infer.model.decoder.state_dict(), Output_root + "decoder" + str(saver_id) + ".pth")
    # Model_infer.model.load_state_dict(torch.load(Output_root + 'model' + loadmodel_index ))
    
    Model_infer.model.initializer.load_state_dict(torch.load(Output_root + 'initializer' + loadmodel_index,map_location='cuda:0'))
    Model_infer.model.encoder.load_state_dict(torch.load(Output_root + 'encoder' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.processor.load_state_dict(torch.load(Output_root + 'processor' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.presence_nn.load_state_dict(torch.load(Output_root + 'presence_nn' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.decoder.load_state_dict(torch.load(Output_root + 'decoder' + loadmodel_index,map_location='cuda:0' ))
    Model_infer.model.temporal_binder.load_state_dict(torch.load(Output_root + 'temporal_binder' + loadmodel_index,map_location='cuda:0' ))

        # Load the entire state dictionary for the encoder
    # Load the entire state dictionary for the encoder
    # state_dict = torch.load(Output_root + 'encoder' + loadmodel_index, map_location='cuda:0')

    # # # Filter the state dictionary to only include keys related to the backbone
    # backbone_state_dict = {k[len("module.backbone."):]: v for k, v in state_dict.items() if k.startswith("module.backbone.")}

    # # # Load the filtered state dictionary into the module.backbone
    # Model_infer.model.encoder.module.backbone.load_state_dict(backbone_state_dict)
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
displayer = Display(GPU_mode)
epoch =0
features =None
visdom_id=0

# Modified training loop with federated learning integration
round_number = 0
last_global_version = 0
last_fed_version = 0
while (1):
    
    # if load_global_model():
    #             print("Global model updated successfully!")
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
    Model_infer.forward(input_videos_GPU,input_flows_GPU,features,Enable_student,epoch=epoch)
    lr=scheduler.cyclic_learning_rate(current_epoch=epoch,max_lr=Max_lr,min_lr=learningR,cycle_length=4)
    print("learning rate is :" + str(lr))
    if Evaluation == False and Evaluation_slots==False:
        Model_infer.optimization(labels_GPU,Enable_student) 

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
            
            plotter.plot('l0', 'l0', 'l0', visdom_id, Model_infer.lossDisplay.cpu().detach().numpy())
            # if Enable_student:
            plotter.plot('1ls', '1ls', 'l1s', visdom_id, Model_infer.lossDisplay_s.cpu().detach().numpy())
            plotter.plot('1lp', '1lp', 'l1p', visdom_id, Model_infer.lossDisplay_p.cpu().detach().numpy())

        if read_id % 1== 0   :
            print(" epoch" + str (epoch) )
            print(" loss" + str (Model_infer.lossDisplay.cpu().detach().numpy()) )
            if Enable_student:
                print(" loss_SS" + str (Model_infer.lossDisplay_s.cpu().detach().numpy()) )

  
    # Modified model saving with presence_nn
    if (read_id % Fed_iter) == 0 and not Evaluation_slots:
        # Save client model
        save_client_model()
        components = [
            "initializer", "encoder", "processor",
              "decoder","temporal_binder","presence_nn"
        ]
        for comp in components:
            torch.save(
                getattr(Model_infer.model, comp).state_dict(),
                Output_root + f"{comp}{saver_id}.pth"
            )
        # Attempt federation with retries
        for fed_attempt in range(FED_MAX_ATTEMPTS):
            if attempt_federation():
                print("Federation successful!")
                break
            print(f"Federation attempt {fed_attempt+1} failed, retrying...")
            time.sleep(FED_RETRY_DELAY)
        
        # Load updated global model with retries
        for load_attempt in range(FED_MAX_ATTEMPTS):
            if load_global_model():
                print("Global model updated successfully!")
                break
            print(f"Global model load attempt {load_attempt+1} failed, retrying...")
            time.sleep(FED_RETRY_DELAY)
        # round_number += 1
        saver_id = 0 if saver_id > 1 else saver_id + 1
    read_id+=1
    visdom_id+=1
    # Federated averaging logic at epoch end
    if dataLoader.all_read_flag == 1:
        
        
        Save_feature_OLG = False
        #remove this for none converting mode
        epoch +=1

        print("finished epoch" + str (epoch) )
        dataLoader.all_read_flag = 0
        read_id=0

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
 