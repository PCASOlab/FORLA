import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
from visdom import Visdom
import cv2
import json
import pickle
from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import root_YTVOS
# from working_dir_root import SAM_pretrain_root,sam_feature_OLG_dir3
import torch
import torch.nn as nn
import torch.nn.functional as F
# sam_feature_OLG_dir3=[]
# SAM_pretrain_root=[]

# output_folder_sam_feature = sam_feature_OLG_dir3
# sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
# # self.inter_bz =1
# model_type = "vit_h"
# model_type = "vit_l"
# model_type = "vit_b"

# # model_type = "vit_t"
# # sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

# # mobile SAM
# Create_sam_feature = True
# Update_PKL = False
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# # self.predictor = SamPredictor(self.sam) 
# Vit_encoder = sam.image_encoder
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Vit_encoder.to(device)
# root_YTVOS = "C:/2data/YT VOS/" 

Update_PKL = True
Create_sam_feature = False
img_size =(256,256)
video_len=10

def read_frames(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [   cv2.resize(cv2.imread(frame_path), img_size) for frame_path in frame_paths]
    return frames
def read_masks(video_folder, img_size):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = [cv2.cvtColor(  cv2.resize(cv2.imread(frame_path), img_size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB) for frame_path in frame_paths]
    return frames
def get_categories(seq, category_map):
    """Get the list of categories for a given sequence."""
    return list(set(category_map[seq].values()))
def convert_mask_to_onehot(mask, categories):
    """Convert a segmentation mask to a one-hot presence vector."""
    pass
def one_hot_vector(categories, selected_categories):
    vector = np.zeros(len(categories))
    for category in selected_categories:
        if category in categories:
            vector[categories[category] - 1] = 1
    return vector
 
def get_frame_categories(frame_unique_colors, video_unique_color, seq, category_map):
    categories = []
    for color in frame_unique_colors:
        category_id = get_category_id(color, video_unique_color)
        category_id = int(np.clip(category_id,0,len(category_map[seq])))
        if category_id>0:
            # if category_id>len(category_map[seq]):
                
            category = category_map[seq][str(category_id)]
            categories.append(category)
    return categories

def get_category_id(color, video_unique_color):
    for idx, unique_color in enumerate(video_unique_color):
        if np.array_equal(color, unique_color):
            return idx
    return None
def get_unique_colors(masks):
    mask_reshaped = masks.reshape(-1, 3)
    unique_colors, counts = np.unique(mask_reshaped, axis=0, return_counts=True)
    # sorted_indices = np.lexsort(([unique_colors[:, i] for i in range(unique_colors.shape[1] - 1, -1, -1)]))
    # unique_colors_sorted = unique_colors[sorted_indices]
    # video_unique_color = unique_colors[np.lexsort(unique_colors.T[[0, 1,2]])]
    video_unique_color = unique_colors[np.lexsort(unique_colors.T[[2, 1,0]])]

    return video_unique_color

def get_categories_for_folder(json_data, folder_name):
    videos = json_data.get('videos', {})
    
    # Extract categories for the specific folder
    folder_categories = []
    if folder_name in videos:
        for obj_id, obj_data in videos[folder_name].get('objects', {}).items():
            folder_categories.append(obj_data['category'])
    
    return folder_categories

def get_all_unique_categories(json_data):
    videos = json_data.get('videos', {})
    all_categories = set()
    
    for video_data in videos.values():
        for obj_data in video_data.get('objects', {}).values():
            all_categories.add(obj_data['category'])
    
    return sorted(list(all_categories))

def convert_to_one_hot(folder_categories, all_categories):
    category_to_index = {category: idx for idx, category in enumerate(all_categories)}
    
    # Convert folder categories to one-hot vectors
    one_hot_vectors = np.zeros((len(folder_categories), len(all_categories)), dtype=int)
    for idx, category in enumerate(folder_categories):
        category_index = category_to_index[category]
        one_hot_vectors[idx, category_index] = 1
    
    return one_hot_vectors
def merge_one_hot_vectors(one_hot_vectors):
    merged_vector = np.any(one_hot_vectors, axis=0).astype(int)
    return merged_vector

color_list = [[236, 95, 103],[249, 145, 87]]
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    output_folder_pkl = root_YTVOS + "pkl/"
    subsets = ['train', 'val']
    subsets = ['train']
    Json_map_dir = root_YTVOS + 'meta.json'
    with open(Json_map_dir, 'r') as f:
        category_map = json.load(f)
    all_categories = get_all_unique_categories(category_map)
    file_counter=0
    for sub in subsets:
        annotation_list  = os.listdir( os.path.join(root_YTVOS, sub,"Annotations") )
        # convert one folder by one folder:
        for folder in annotation_list:
            # folder = "0a7b27fde9"
            images = read_frames ( os.path.join(root_YTVOS, sub,"JPEGImages",folder), img_size)
            masks = read_masks ( os.path.join(root_YTVOS, sub,"Annotations",folder), img_size)
            folder_categories = get_categories_for_folder(category_map, folder)
            one_hot_vectors = convert_to_one_hot(folder_categories, all_categories)
            merged_one_hot_vector = merge_one_hot_vectors(one_hot_vectors)
            print(" folder:", folder_categories)

            print("  One-Hot Vector:", one_hot_vectors)
            print("  One-Hot Vector:", merged_one_hot_vector)
          
            # video_unique_color =get_unique_colors(np.array(masks)) 
            all_data =[]
            for img, msk in zip(images, masks):
                # Organize as a dictionary or structure
               
                data_pair = {'frame': img, 'label': merged_one_hot_vector,'instance_label': one_hot_vectors, 'mask':msk}
                all_data.append(data_pair)

                # Check if buffer is not empty and has reached the desired length
                if len(all_data) > 0 and len(all_data) == video_len:
                    # Convert list of dictionaries to a dictionary of arrays
                    data_dict = {'frames': np.array([pair['frame'] for pair in all_data]),
                                    'labels': np.array([pair['label'] for pair in all_data]),
                                    'instance_label': np.array([pair['instance_label'] for pair in all_data]) ,
                                     'masks': np.array([pair['mask'] for pair in all_data])  }

                    # Perform "or" operation to merge labels
                    # merged_labels = merge_labels(data_dict['labels'])

                    # Reshape arrays
                    data_dict['frames'] = np.transpose(data_dict['frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
                    data_dict['masks'] = np.transpose(data_dict['masks'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)

                    # data_dict['labels'] = np.transpose(data_dict['labels'], (1, 0, 2, 3))  # Reshape to (10 29, 256, 256)
                
                    pkl_file_name = f"clip_{file_counter:06d}.pkl"
                    pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)
                    if Update_PKL:
                        with open(pkl_file_path, 'wb') as file:
                            pickle.dump(data_dict, file)
                            print("Pkl file created:" +pkl_file_name)
                    # if Create_sam_feature == True:
                    #     this_video_buff = data_dict['frames'] 
                    #     video_buff_GPU = torch.from_numpy(np.float32(this_video_buff)).to (device)
                    #     video_buff_GPU = video_buff_GPU.permute(1,0,2,3) # Reshape to (29, 3, 64, 64)
                    #     input_resample =   F.interpolate(video_buff_GPU,  size=(1024,  1024), mode='bilinear', align_corners=False)
                        
                    #     bz,  ch, H, W = input_resample.size()
                    #     predicted_tensors =[]
                    #     with torch.no_grad():

                    #         for i in range(bz):
                                
                    #             input_chunk = (input_resample[i:i+1] -124.0)/60.0
                    #             output_chunk = Vit_encoder(input_chunk)
                    #             predicted_tensors.append(output_chunk)
                            
                    #         # Concatenate predicted tensors along batch dimension
                    #         concatenated_tensor = torch.cat(predicted_tensors, dim=0)
                            
                        
                    #     features = concatenated_tensor.half()
                    #     sam_pkl_file_name = f"clip_{file_counter:06d}.pkl"
                    #     sam_pkl_file_path = os.path.join(output_folder_sam_feature, sam_pkl_file_name)

                    #     with open(sam_pkl_file_path, 'wb') as file:
                    #         pickle.dump(features, file)
                    #         print("sam Pkl file created:" +sam_pkl_file_name)
                        
                    file_counter += 1

                    # Clear data for the next batch
                    all_data = []
 
    print("Total files created:", file_counter)

