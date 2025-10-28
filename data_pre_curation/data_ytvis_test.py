import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
from visdom import Visdom
import cv2
import json
import pickle
# from SAM.segment_anything import  SamPredictor, sam_model_registry
from working_dir_root import root_YTVOS
# from working_dir_root import SAM_pretrain_root,sam_feature_OLG_dir3
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_datasets as tfds
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
img_size =(224,224)
video_len=10
def download_and_extract_ytvos_test(output_dir):
    """Downloads and extracts YouTube-VIS test data with proper mask handling"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load dataset
     # Use the correct configuration (480_640_full contains 2019 data)
    builder = tfds.builder('youtube_vis/480_640_full')
    builder.download_and_prepare()
    
    # YouTube-VIS uses 'validation' as the test split
    ds = builder.as_dataset(split='validation')

    meta = {'videos': {}}
    
    for example in tfds.as_numpy(ds):
        video_id = example['metadata']['video_name'].decode('utf-8')
        video_dir = os.path.join(output_dir, 'test', 'JPEGImages', video_id)
        ann_dir = os.path.join(output_dir, 'test', 'Annotations', video_id)
        
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        # Store metadata
        meta['videos'][video_id] = {
            'objects': {
                str(obj_id): {'category': cat.decode('utf-8')}
                for obj_id, cat in zip(example['objects']['track_ids'], example['objects']['category_labels'])
            }
        }

        # Process frames
        for i, (frame, mask) in enumerate(zip(example['video']['frames'], example['video']['segmentations'])):
            # Save frame
            img_path = os.path.join(video_dir, f"{i:05d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Save mask as 16-bit grayscale to preserve instance IDs
            mask_path = os.path.join(ann_dir, f"{i:05d}.png")
            cv2.imwrite(mask_path, mask.squeeze().astype(np.uint16))

    # Save metadata
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

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
def color_mask_to_instance_id(color_mask, max_instances=20):
    """Convert RGB color mask to instance ID mask (0=background, 1-20=instances)"""
    # Reshape to pixel array
    h, w, _ = color_mask.shape
    pixels = color_mask.reshape(-1, 3)
    
    # Find unique colors and sort them consistently
    unique_colors = np.unique(pixels, axis=0)
    
    # Sort colors lexographically (BGR order)
    sorted_indices = np.lexsort((unique_colors[:, 0],  # B
                                 unique_colors[:, 1],  # G
                                 unique_colors[:, 2]))  # R
    unique_colors = unique_colors[sorted_indices]
    
    # Truncate to max instances
    if len(unique_colors) > max_instances:
        unique_colors = unique_colors[:max_instances]
    
    # Create instance mask
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Assign instance IDs (start from 1)
    for instance_id, color in enumerate(unique_colors, start=1):
        # Create mask for this color
        color_mask = (color_mask == color).all(axis=-1)
        instance_mask[color_mask] = instance_id
    
    return instance_mask
def read_instance_masks_train(video_folder, img_size):
    """Read masks and convert to instance ID masks"""
    frame_paths = sorted(glob(os.path.join(video_folder, "*.png")))
    masks = []
    for frame_path in frame_paths:
        # Read and resize color mask
        color_mask = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        color_mask = cv2.resize(color_mask, img_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to instance mask
        instance_mask = color_mask_to_instance_id(color_mask)
        masks.append(instance_mask)
    return masks
def read_instance_masks(video_folder, img_size):
    """Read masks from instance folders and combine into instance ID masks."""
    # Get valid instance folders (even-numbered)
    valid_folders = []
    for d in os.listdir(video_folder):
        dir_path = os.path.join(video_folder, d)
        if os.path.isdir(dir_path) and d.isdigit():
            folder_num = int(d)
            if folder_num % 2 == 0:
                valid_folders.append(d)
    # Sort the folders numerically
    valid_folders = sorted(valid_folders, key=lambda x: int(x))
    # Assign instance IDs starting from 1
    instance_id_map = {folder: idx + 1 for idx, folder in enumerate(valid_folders)}
    
    # Collect all frame names across all instance folders
    frame_names = set()
    for folder in valid_folders:
        folder_path = os.path.join(video_folder, folder)
        frames = glob(os.path.join(folder_path, "*.png"))
        for f in frames:
            frame_name = os.path.basename(f)
            frame_names.add(frame_name)
    frame_names = sorted(frame_names)
    
    masks = []
    for frame_name in frame_names:
        # Initialize RGB mask with shape (height, width, 3)
        height, width = img_size[::-1]  # img_size is (width, height)
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
        # Set Green channel (index 1) to 1 for all pixels
        rgb_mask[..., 1] = 1
        
        for folder in valid_folders:
            instance_id = instance_id_map[folder]
            mask_path = os.path.join(video_folder, folder, frame_name)
            if os.path.exists(mask_path):
                # Read the mask in grayscale
                bw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Resize using nearest neighbor interpolation
                bw_mask = cv2.resize(bw_mask, img_size, interpolation=cv2.INTER_NEAREST)
                # Create a binary mask where white pixels (255) are True
                binary_mask = (bw_mask > 160)
                # Assign instance ID to the Red channel (index 0) for active pixels
                rgb_mask[binary_mask, 0] = instance_id
        
        masks.append(rgb_mask)
    
    return masks
color_list = [[236, 95, 103],[249, 145, 87]]
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # download_and_extract_ytvos_test(root_YTVOS)
    
    output_folder_pkl = root_YTVOS + "pkl_test/"
    subsets = ['train', 'val']
    subsets = ['valid']
    Json_map_dir = root_YTVOS + 'valid/meta_expressions_challenge.json'
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
            masks = read_instance_masks ( os.path.join(root_YTVOS, sub,"Annotations",folder), img_size)
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

