import os
# import matplotlib.pyplot as plt
import cv2
# import pandas as pd
import random
import copy
import shutil
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import SAM_pretrain_root
import numpy as np
Show_img =False
Create_sam_feature = False
GPU_mode = True
if GPU_mode ==True:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device("cpu")
sam_checkpoint = SAM_pretrain_root+"sam_vit_h_4b8939.pth"
sam_checkpoint = SAM_pretrain_root+"sam_vit_l_0b3195.pth"
sam_checkpoint = SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
# self.inter_bz =1
model_type = "vit_h"
model_type = "vit_l"
model_type = "vit_b"

# model_type = "vit_t"
# sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

# mobile SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# self.predictor = SamPredictor(self.sam) 
Vit_encoder = sam.image_encoder
Vit_encoder.to(device)
color_class_mapping={(127, 127, 127): 0,
                    (210, 140, 140): 1,
                    (255, 114, 114): 2,
                    (231, 70, 156): 3,
                    (186, 183, 75): 4,
                    (170, 255, 0): 5,
                    (255, 85, 0): 6,
                    (255, 0, 0): 7,
                    (255, 255, 0): 8,
                    (169, 255, 184): 9,
                    (255, 160, 165): 10,
                    (0, 50, 128): 11,
                    (111, 74, 0): 12}
class_name_mapping={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}
# import ipywidgets as widgets
# import wandb
# Defining input images path, output path, masks path and labels path
img_size = (256, 256)  # Specify the desired size
# video_buffer_len = 29
downsample_len = 29
# Function to read labels from a text file
def read_labels(file_path):
    labels = np.genfromtxt(file_path, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=int)
    return labels
def find_bounding_box_keep_height(image, threshold=20):
    """
    Find the left and right bounds (W1 and W2) for cropping the actual image, 
    while keeping the full height of the image. The cropping is based on the 
    average pixel intensities of the top and bottom lines of the image.
    
    Parameters:
    - image: np.ndarray, input image of shape (H, W, 3)
    - threshold: int, pixel intensity threshold to differentiate between black borders and the actual image.
    
    Returns:
    - bounding_box: list, [H1, H2, W1, W2] bounding box to crop the actual image, 
                    with H1 and H2 keeping the full height and W1, W2 based on top/bottom lines.
    """
    # Get the height and width of the image
    H, W, _ = image.shape

    # Convert the image to grayscale by averaging the color channels
    grayscale_image = np.mean(image, axis=2).astype(np.uint8)

    # Calculate the horizontal projection (average intensity across each column)
    horizontal_projection = np.mean(grayscale_image, axis=0)  # Average each column

    # Calculate Wm1 (left) and Wm2 (right) based on the horizontal projection
    Wm1 = np.argmax(horizontal_projection > threshold)+100
    Wm2 = W - np.argmax(horizontal_projection[::-1] > threshold)-100

    # Get the top and bottom lines (first and last row) of pixels
    top_line = grayscale_image[10, :]  # Near first row (top line)
    bottom_line = grayscale_image[H-10, :]  # Near last row (bottom line)

    # Find crop points based on non-black columns in top and bottom lines
    Wt1 = np.argmax(top_line > threshold)  # First non-black column in the top line
    Wt2 = W - np.argmax(top_line[::-1] > threshold)  # Last non-black column in the top line

    Wb1 = np.argmax(bottom_line > threshold)  # First non-black column in the bottom line
    Wb2 = W - np.argmax(bottom_line[::-1] > threshold)  # Last non-black column in the bottom line

    # Set W1 and W2 based on the crop points closest to Wm1 and Wm2
    W1 = min(Wt1, Wb1) if abs(Wt1 - Wm1) < abs(Wb1 - Wm1) else Wb1
    W2 = max(Wt2, Wb2) if abs(Wt2 - Wm2) < abs(Wb2 - Wm2) else Wb2

    # Return the bounding box keeping the full height of the image (H1=0, H2=H)
    return [0, H, W1, W2]


def crop_frame(frame,bounding_box):
    """
    Crop a frame based on the bounding box determined by the find_bounding_box_keep_height function.
    
    Parameters:
    - frame: np.ndarray, input image
    
    Returns:
    - cropped_frame: np.ndarray, cropped frame
    """
    # bounding_box = find_bounding_box_keep_height(frame)
    H1, H2, W1, W2 = bounding_box
    cropped_frame = frame[H1:H2, W1:W2]
    return cropped_frame
# Function to read frames from a video folder and resize
def read_frames(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [cv2.resize(cv2.imread(frame_path), img_size) for frame_path in frame_paths]
    return frames

path='/media/guiqiu/Weakly_supervised_data/cholecseg8k_working/cholecseg8k/'
op_path='/media/guiqiu/Weakly_supervised_data/cholecseg8k_working/'
output_folder_pkl = op_path +"output_pkl_croped/"
output_folder_sam_feature = op_path + "output_sam_features/"
rawimages_path=os.path.join(op_path, 'raw_images')
maskimages_path=os.path.join(op_path, 'mask_images')
labels_path=os.path.join(op_path, 'labels')
# os.makedirs(rawimages_path)
# os.makedirs(maskimages_path)
# os.makedirs(labels_path)
# Defining output train,val and test paths
imgtrainpath = os.path.join(op_path,'images','train')
imgvalpath=os.path.join(op_path,'images','validation')
imgtestpath=os.path.join(op_path,'images','test')

labeltrainpath=os.path.join(op_path,'labels','train')
labelvalpath=os.path.join(op_path,'labels','validation')
labeltestpath=os.path.join(op_path,'labels','test')

# os.makedirs(imgtrainpath)
# os.makedirs(imgvalpath)
# os.makedirs(imgtestpath)

# os.makedirs(labeltrainpath)
# os.makedirs(labelvalpath)
# os.makedirs(labeltestpath)

m=0 # variable for counting total sub-directories
n=0 # variable for counting total images
o=0 # variable for counting total raw images
p=0 # variable for counting total masks
q=0 # variable for counting total directories
file_counter= 0 # for counting total videos
for directory in os.listdir(path):
    dir_path=os.path.join(path, directory)
    m=m+len(os.listdir(dir_path))
    q=q+1
    for sub_dir in os.listdir(dir_path):
        sub_dir_path=os.path.join(dir_path, sub_dir)
        n=n+len(os.listdir(sub_dir_path))
        file_list =  os.listdir(sub_dir_path)
        sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))
        video_images = []
        video_masks = []
        for image in sorted_file_list:
            src_path=os.path.join(sub_dir_path, image)
            # Rename the image based on sub-directory to distinguish images with same names from different directories
            newname=sub_dir+image
            if 'mask' not in image:   # Criterion for raw image             
                # newpath=os.path.join(rawimages_path, newname)
                # dest_path=os.path.join(rawimages_path, image)
                # shutil.copy(src_path, dest_path) # Copying raw image to output directory
                # os.rename(dest_path, newpath) # Renaming raw image
                o=o+1
                this_image = cv2.imread(src_path)
                bounding_box = find_bounding_box_keep_height(this_image)
                this_image = crop_frame(this_image, bounding_box)
                this_image=cv2.resize(this_image, img_size)
                video_images.append(this_image)
            if 'color_mask' in image:  # Criterion for color mask  
                # newpath=os.path.join(maskimages_path, newname)
                # dest_path=os.path.join(maskimages_path, image)
                # shutil.copy(src_path, dest_path) # Copying mask to output directory
                # os.rename(dest_path, newpath) # Renaming mask
                p=p+1
                this_mask= cv2.imread(src_path)
                # bounding_box = find_bounding_box_keep_height(this_image)
                this_mask = crop_frame(this_mask, bounding_box)
                this_mask=cv2.resize(this_mask, img_size,cv2.INTER_AREA)
                this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2RGB)

                mask_12_channel = np.zeros((len(class_name_mapping),*img_size), dtype=np.uint8)  # Initialize 12-channel mask

                # Create a mask for each color class
                for idx, color in enumerate(color_class_mapping.keys()):
                    class_mask = np.all(this_mask == color, axis=-1)
                    mask_12_channel[idx, :, :] = class_mask.astype(np.uint8) 
                    # if Show_img:
                    #     cv2.imshow(class_name_mapping[idx],class_mask.astype(np.uint8) *255)
                    #     cv2.waitKey(1)

                # this_mask = cv2.cvtColor(this_mask, cv2.COLOR_BGR2RGB)

                video_masks.append(mask_12_channel)
                if Show_img == True:
                    cv2.imshow('color maks',this_mask.astype(np.uint8) *255)

                    cv2.imshow('Grasp',mask_12_channel[5].astype(np.uint8) *255)
                    cv2.imshow('hook',mask_12_channel[9].astype(np.uint8) *255)

                    cv2.waitKey(1)

        print(len(video_images))    
        print(len(video_masks))   
        video_images = np.array(video_images)
        video_masks = np.array(video_masks)

        # Ensure the downsampled video has exact length of downsample_len
        if len(video_images) > downsample_len:
            step = len(video_images) // downsample_len
            video_images = video_images[:downsample_len * step:step]
        if len(video_masks) > downsample_len:
            step = len(video_masks) // downsample_len
            video_masks = video_masks[:downsample_len * step:step]
        video_images  = np.transpose(video_images , (3, 0, 1, 2))  # Reshape to (3, 29, 64, 64)
        video_masks  = np.transpose(video_masks , (1, 0, 2, 3))  # Reshape to (13, 29, 64, 64)


        print(video_images.shape)
        print(video_masks.shape) 
        data_dict = {'frames': video_images,
                    'labels': video_masks}


        pkl_file_name = f"clip_{file_counter:06d}.pkl"
        pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)

        with open(pkl_file_path, 'wb') as file:
            pickle.dump(data_dict, file)
            print("Pkl file created:" +pkl_file_name)
        if Create_sam_feature == True:
                    this_video_buff = data_dict['frames'] 
                    video_buff_GPU = torch.from_numpy(np.float32(this_video_buff)).to (device)
                    video_buff_GPU = video_buff_GPU.permute(1,0,2,3) # Reshape to (29, 3, 64, 64)
                    input_resample =   F.interpolate(video_buff_GPU,  size=(1024,  1024), mode='bilinear', align_corners=False)
                    
                    bz,  ch, H, W = input_resample.size()
                    predicted_tensors =[]
                    with torch.no_grad():

                        for i in range(bz):
                            
                            input_chunk = (input_resample[i:i+1] -124.0)/60.0
                            output_chunk = Vit_encoder(input_chunk)
                            predicted_tensors.append(output_chunk)
                        
                        # Concatenate predicted tensors along batch dimension
                        concatenated_tensor = torch.cat(predicted_tensors, dim=0)
                        
                    
                    features = concatenated_tensor.half()
                    sam_pkl_file_name = f"clip_{file_counter:06d}.pkl"
                    sam_pkl_file_path = os.path.join(output_folder_sam_feature, sam_pkl_file_name)

                    with open(sam_pkl_file_path, 'wb') as file:
                        pickle.dump(features, file)
                        print("sam Pkl file created:" +sam_pkl_file_name)
        file_counter +=1  
        



print("Total number of sub-directories:", m)
print("Total number of images:", n)
print("Total number of raw images:", o)
print("Total number of masks:", p)

len(os.listdir(rawimages_path)), len(os.listdir(maskimages_path))



# for directory in os.listdir(path):
#     dir_path=os.path.join(path, directory)
#     m=m+len(os.listdir(dir_path))
#     q=q+1
#     for sub_dir in os.listdir(dir_path):
#         sub_dir_path=os.path.join(dir_path, sub_dir)
#         n=n+len(os.listdir(sub_dir_path))
#         file_list =  os.listdir(sub_dir_path)
#         sorted_file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))
#         for image in sorted_file_list:
#             src_path=os.path.join(sub_dir_path, image)
#             # Rename the image based on sub-directory to distinguish images with same names from different directories
#             newname=sub_dir+image
#             if 'mask' not in image:   # Criterion for raw image             
#                 newpath=os.path.join(rawimages_path, newname)
#                 dest_path=os.path.join(rawimages_path, image)
#                 shutil.copy(src_path, dest_path) # Copying raw image to output directory
#                 os.rename(dest_path, newpath) # Renaming raw image
#                 o=o+1
#             if 'color_mask' in image:  # Criterion for color mask  
#                 newpath=os.path.join(maskimages_path, newname)
#                 dest_path=os.path.join(maskimages_path, image)
#                 shutil.copy(src_path, dest_path) # Copying mask to output directory
#                 os.rename(dest_path, newpath) # Renaming mask
#                 p=p+1