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
from working_dir_root import root_YTOBJ
import mat73
# from working_dir_root import root_YTOBJ
# root_YTOBJ = "C:/2data/TCAMdata/youtube video/download2/"

# from working_dir_root import SAM_pretrain_root,sam_feature_OLG_dir3
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
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
video_len=29
def apply_mask(resized_image, scaled_bbox):
  """
  Applies a mask to a resized image based on the provided scaled bounding box.

  Args:
      resized_image (np.ndarray): The resized image (OpenCV format with BGR channels).
      scaled_bbox (list[4]): List containing [xmin, ymin, xmax, ymax] for the scaled bounding box.

  Returns:
      np.ndarray: The resized image with the mask applied.
  """

  # Create a mask based on the scaled bounding box and resized image shape
  mask = np.zeros(resized_image.shape[:2], dtype=np.uint8)  # Extract height & width
  xmin, ymin, xmax, ymax = scaled_bbox

  # Ensure coordinates are within image boundaries
  xmin = max(0, xmin)
  ymin = max(0, ymin)
  xmax = min(resized_image.shape[1], xmax)  # Use width from image shape
  ymax = min(resized_image.shape[0], ymax)  # Use height from image shape

  # Fill the mask within the scaled bounding box coordinates
  mask[ymin:ymax, xmin:xmax] = 1

  # Apply the mask to the resized image (assuming BGR channels)
  masked_image = resized_image.copy()
  masked_image[mask == 0] = (0, 0, 0)  # Set masked pixels to black (BGR)

  return masked_image,mask
def scale_bounding_box(bbox, image_orig, image_new):
  """
  Scales a bounding box based on the original and new image sizes.

  Args:
      bbox (list[4]): List containing [xmin, ymin, xmax, ymax] for the bounding box.
      image_orig (np.ndarray): The original image (OpenCV format with BGR channels).
      image_new (np.ndarray): The new image (OpenCV format with BGR channels).

  Returns:
      list[4]: Scaled bounding box coordinates in the new image format.
  """

  # Get dimensions from the image shapes
  H_orig, W_orig, _ = image_orig.shape
  H_new, W_new, _ = image_new.shape

  # Calculate scaling factors
  scale_x = W_new / W_orig
  scale_y = H_new / H_orig

  # Scale coordinates
  xmin, ymin, xmax, ymax = bbox
  scaled_xmin = int(round(xmin * scale_x))
  scaled_ymin = int(round(ymin * scale_y))
  scaled_xmax = int(round(xmax * scale_x))
  scaled_ymax = int(round(ymax * scale_y))

  return [scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax]

def decode_bbx(target_string,GT):
#     data = {
#     "apple": [1, 2, 3],
#     "banana": [4, 5, 6],
#     "cherry": [7, 8, 9],
# }

# target_string = "banana"
    # target_string='aeroplane00002166'
    matching_item = [item for item in GT if item[0][0]== target_string]

    if matching_item:
        # index = GT.index(matching_item[0])
        vector = matching_item[0][1][0]
        print("vector of :" +target_string )
        print(f"Corresponding vector: {vector}")
        return vector
    else:
        # print(f"String '{target_string}' not found in the data")
        return None
def load_image_sequence(folder_path, start_num, end_num, flows,GT,ctg,downsample_factor=1):
  """
  Loads an image sequence with a downsample factor.

  Args:
      folder_path (str): Path to the folder containing the images.
      start_num (int): Starting frame number (inclusive).
      end_num (int): Ending frame number (inclusive).
      downsample_factor (int, optional): The factor by which to downsample the sequence. Defaults to 1 (load all frames).

  Returns:
      tuple: A tuple containing two lists:
          - images (list): A list of loaded images.
          - image_names (list): A list of corresponding image names (without extension).
  """

  # List all files in the directory
  all_files = os.listdir(folder_path)
  
  # Filter out files that don't match the pattern "XXXXXXX.jpg" where X is a digit
  image_files = [f for f in all_files if f.endswith('.jpg') and f[:-4].isdigit()]
  
  # Sort the files numerically based on the number in the file name
  image_files.sort(key=lambda f: int(f[:-4]))
  
  # Load images within the specified range with downsampling
  images = []
  image_names = []
  flow_masks = []
  masks = []
  boxs =[]
  for i, image_file in enumerate(image_files):
    if (start_num <= int(image_file[:-4]) < end_num) and (i % downsample_factor == 0):
        image_path = os.path.join(folder_path, image_file)
        image_OG = cv2.imread(image_path)
        image = cv2.resize(image_OG, img_size)
        if image is not None:
                target_box_name = ctg+image_file[:-4]
                bbx = decode_bbx(target_box_name,GT)
                mask=None
                if bbx is not None:
                    bbx =scale_bounding_box(bbx, image_OG, image)
                    masked_image,mask = apply_mask(image,bbx)
                    cv2.imwrite(root_YTOBJ + "maskedimages/"+image_file+".jpg", masked_image)
                # flow_id = int(image_file[:-4])-start_num
                # flow_id =np.clip(flow_id,0, len(flows)-1)
                # this_flow = flows[int(flow_id)]
                # magnitude = np.sqrt(this_flow[..., 0]**2 + this_flow[..., 1]**2)

                #                     # Normalize and scale the magnitude to the range [0, 255]
                # # magnitude = magnitude>3
                # # magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude)  
                # # magnitude = (magnitude - np.min(magnitude)) / np.max(magnitude) * 255

                # # magnitude = np.clip(magnitude,0,1)
                # magnitude = cv2.resize(magnitude, dsize=img_size, interpolation=cv2.INTER_AREA)
                # magnitude = magnitude>3
                
                images.append(image)
                image_names.append(image_file[:-4])
                masks.append(mask)
                boxs.append(bbx)
                # mask = magnitude[..., np.newaxis] * np.ones_like(image[..., :1])
                # masked_image = image * mask
                # masked_images.append(masked_image)
                # flow_masks.append(mask)

   

  
  return images, image_names, masks,boxs

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
    one_hot_vectors = np.zeros(len(all_categories), dtype=int)
   
    category_index = category_to_index[folder_categories]
    one_hot_vectors[ category_index] = 1
    
    return one_hot_vectors
def merge_one_hot_vectors(one_hot_vectors):
    merged_vector = np.any(one_hot_vectors, axis=0).astype(int)
    return merged_vector

color_list = [[236, 95, 103],[249, 145, 87]]
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    output_folder_pkl = root_YTOBJ + "pkl8/"
    subsets = ['train', 'val']
    subsets = ['train']
    all_categories = sorted(os.listdir(root_YTOBJ+ "GroundTruth/"))
    mat_contents = loadmat(root_YTOBJ + 'Ranges/ranges_aeroplane.mat')

# Inspect the contents
    print(mat_contents.keys())

    # Access a specific variable (replace 'variable_name' with the actual variable name)
    data = mat_contents['ranges']
    all_count_vector =0
    
# Print the data
    # print(data)
    file_counter =3000 # start from 3000 as the train is below 
    for category  in sorted(os.listdir(root_YTOBJ+ "videos/")):
        mat_contents_bbx = loadmat(root_YTOBJ + 'GroundTruth/'+category+'/bb_gtTest_'+category+'.mat')

# Inspect the contents
        print(mat_contents.keys())

        # Access a specific variable (replace 'variable_name' with the actual variable name)
        GT = mat_contents_bbx['bb_gtTest'][0]
        # category ='horse'
        mat_contents = loadmat(root_YTOBJ + 'Ranges/ranges_'+category+'.mat')

# Inspect the contents
        print(mat_contents.keys())

        # Access a specific variable (replace 'variable_name' with the actual variable name)
        data_range = mat_contents['ranges']

        # mat_contents_bbx = loadmat(root_YTOBJ + 'GroundTruth/'+category+'/bb_gtTraining_'+category+'.mat')
        one_hot_vectors = convert_to_one_hot(category, all_categories)

        print(mat_contents.keys())

    # Access a specific variable (replace 'variable_name' with the actual variable name)
        # data_GT = mat_contents_bbx['bb_gtTraining']
        flow_mat_list = (os.listdir(root_YTOBJ+ "OpticalFlow/"+category +"_flow/" + category +"/" +"broxPAMI2011/"))
        # convert one folder by one folder:
        for index, this_range in enumerate(data_range.T):
            this_video_dir = root_YTOBJ + 'videos/'+category + "/"
 
            flows = mat73.loadmat(root_YTOBJ+ "OpticalFlow/"+category +"_flow/" + category +"/" +"broxPAMI2011/flowShot"+str(index+1)+'.mat')
            images, image_names, masks, boxs= load_image_sequence(this_video_dir,this_range[0],this_range[1],flows['flow'],GT,category)

            # video_unique_color =get_unique_colors(np.array(masks)) 
            all_data =[]
            if any(boxs):
                for img, name ,mask, box in zip(images, image_names, masks, boxs):
                    # Organize as a dictionary or structure
                    
                    data_pair = {'frame': img, 'label': one_hot_vectors,'mask':mask,'box':box,'image_names': name}
                    all_data.append(data_pair)
                    # cv2.imwrite(root_YTOBJ + "maskedimages/"+name+".jpg", masked_image)
                    # Check if buffer is not empty and has reached the desired length
                    if len(all_data) > 0 and len(all_data) == video_len:
                        # Convert list of dictionaries to a dictionary of arrays
                        data_dict = {'frames': np.array([pair['frame'] for pair in all_data]),
                                        'labels': np.array([pair['label'] for pair in all_data]),
                                        'boxs': np.array([pair['box'] for pair in all_data]),
                                        'masks': np.array([pair['mask'] for pair in all_data]),
                                        'image_names': np.array([pair['image_names'] for pair in all_data]) ,
                                        }
                        # if any(data_dict['boxs']):
                        # Perform "or" operation to merge labels
                        # merged_labels = merge_labels(data_dict['labels'])

                        # Reshape arrays
                        data_dict['frames'] = np.transpose(data_dict['frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
                        # data_dict['masks'] = np.transpose(data_dict['masks'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
                        # data_dict['masked_frames'] = np.transpose(data_dict['masked_frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)

                        # data_dict['labels'] = np.transpose(data_dict['labels'], (1, 0, 2, 3))  # Reshape to (10 29, 256, 256)
                    
                        pkl_file_name = f"clip_{file_counter:06d}.pkl"
                        pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)
                        if Update_PKL:
                            all_count_vector += one_hot_vectors

                            with open(pkl_file_path, 'wb') as file:
                                pickle.dump(data_dict, file)
                                print("Pkl file created:" +pkl_file_name)
                                print(all_count_vector)
                        
                            
                        file_counter += 1

                            # Clear data for the next batch
                        all_data = []
                else:
                    print("nobox" +category + str(this_range[0])+"-"+str(this_range[1]) )
 
    print("Total files created:", file_counter)

