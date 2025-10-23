import cv2
import os
import  numpy as np
# from working_dir_root import Dataset_video_root, Dataset_label_root
# import csv
import re
import json
from PIL import Image, ImageDraw
from dataset.io import save_a_image
from dataset.io import save_a_pkl_w_create as save_a_pkl
import pickle

Video_format =  ".mp4"
Video_format =  ".mpg"


Jsonfile_dir = 'C:/2data/Raw_data_Chrocic/raw/label/' # the folder for all jsons
Json_to_decode = ['labelsB','labelsA'] 
data_root_dir = 'C:/2data/endovis/2017/all/' # folder for all videos folders with number
Decoded_data_dir= 'C:/2data/Raw_data_Chrocic/data/interim/' # output folder
output_folder_pkl = "C:/2data/endovis/2017/pkl/"
output_folder_sam_feature = 'C:/2data/Raw_data_Chrocic/data/' + "output_sam_features/"
Using_normalized = False
image_resize = 256

# json_data = file.read()
categories_merge = [
    'Prograsp_Forceps_labels',
    'Large_Needle_Driver_labels',
    'Grasping_Retractor_labels',
    'Bipolar_Forceps_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Other_labels'
]
categories = [
    'Left_Prograsp_Forceps_labels',
    'Right_Prograsp_Forceps_labels',
    'Maryland_Bipolar_Forceps_labels',
    'Left_Large_Needle_Driver_labels',
    'Right_Large_Needle_Driver_labels',
    'Prograsp_Forceps_labels',
    'Bipolar_Forceps_labels',
    'Grasping_Retractor_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Left_Grasping_Retractor_labels',
    'Right_Grasping_Retractor_labels',
    'Other_labels'
]
category_colors = {
    'Lymph node': (0, 0, 255),        # Red
    'Vagus nereve': (0, 255, 0),      # Green
    'Bronchus': (255, 0, 0),          # Blue
    'Lung parenchyma': (255, 255, 0),  # Yellow
    'Instruments': (255, 0, 255),      # Magenta
}
# Function to save sampled clips and masks as pickle files
img_size = (256, 256)  # Specify the desired size
video_buffer_len = 5
num_category = len(categories_merge)
# Function to read labels from a text file
def find_index(categories, target):
    for index, category in enumerate(categories):
        if category in target:
            return index
    return -1  # Return -1 if no match is found
def read_labels(file_path):
    labels = np.genfromtxt(file_path, skip_header=1, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=int)
    return labels
# Function to read frames from a video folder, crop, resize, and optionally display
def read_frames(video_folder, img_size, display=False):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        origin = frame
        H, W, _ = frame.shape  # Get the actual height (H) and width (W) of the image
        # crop_coords = (192,0 , 1088, H-80)  # Dynamically crop based on actual image height H
        frame = frame[32:H-32, 336:1590]
        frame = cv2.resize(frame, img_size)  # Resize the cropped image
        frames.append(frame)

        if display:  # Display image if display flag is True
            cv2.imshow(" OG Image", origin)

            cv2.imshow("Cropped and Resized Image", frame)
            cv2.waitKey(1)  # Wait for a key press to close the displayed image
            # cv2.destroyAllWindows()  # Close the display window after key press
    return frames

# Function to read grayscale frames from a video folder, crop, resize, and optionally display
def read_frames_gray(video_folder, img_size, display=False):
    frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        H, W, _ = frame.shape  # Get the actual height (H) and width (W) of the image
        frame = frame[32:H-32, 336:1590]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, img_size)  # Resize the cropped grayscale image
        frames.append(frame_resized)

        if display:  # Display image if display flag is True
            cv2.imshow("Cropped and Resized Grayscale Image", frame_resized)
            cv2.waitKey(1)  # Wait for a key press to close the displayed image
            # cv2.destroyAllWindows()  # Close the display window after key press
    return frames
# Function to perform "or" operation on a group of labels
def merge_labels(label_group):
    return np.max(label_group, axis=0)

# Counter for naming files
file_counter = 0

# Iterate through text files
for sub_set in sorted(os.listdir(data_root_dir)):
    sub_set_imgs_dir = os.path.join(data_root_dir,sub_set, "left_frames")
    frames = read_frames(sub_set_imgs_dir, img_size)
    Masks_1_folder = np.zeros((len(frames),num_category,256,256))

    for annotated_folder in os.listdir(os.path.join(data_root_dir,sub_set, "ground_truth")):
        annotated_folder_dir = os.path.join(data_root_dir,sub_set, "ground_truth",annotated_folder)
        masks_instrument = read_frames_gray(annotated_folder_dir, img_size)
        index = find_index(categories_merge, annotated_folder)
        Masks_1_folder[:,index,:,:] = np.array(masks_instrument) # Does this need to be reshaped or transposed, can be seen in the evaluation 
        pass

    
    Mask_list = [Masks_1_folder[i] for i in range(Masks_1_folder.shape[0])]  
 
    all_data = []
    for this_frame, this_label in zip(frames, Mask_list):
        # Organize as a dictionary or structure
        data_pair = {'frame': this_frame, 'label': this_label}
        all_data.append(data_pair)

        # Check if buffer is not empty and has reached the desired length
        if len(all_data) > 0 and len(all_data) == video_buffer_len:
            # Convert list of dictionaries to a dictionary of arrays
            data_dict = {'frames': np.array([pair['frame'] for pair in all_data]),
                            'labels': np.array([pair['label'] for pair in all_data])}

            # Perform "or" operation to merge labels
            # merged_labels = merge_labels(data_dict['labels'])

            # Reshape arrays
            data_dict['frames'] = np.transpose(data_dict['frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
            data_dict['labels'] = np.transpose(data_dict['labels'], (1, 0, 2, 3))  # Reshape to (10 29, 256, 256)
            #video_masks  = np.transpose(video_masks , (1, 0, 2, 3))  # Reshape to (13, 29, 64, 64)

            # merged_labels = np.transpose(merged_labels)  # Reshape to (3, 29)
            

            # Save frames and labels to HDF5 file
            # hdf5_file_name = f"clip_{file_counter:06d}.h5"
            # hdf5_file_path = os.path.join(output_folder_hdf5, hdf5_file_name)

            # with h5py.File(hdf5_file_path, 'w') as file:
            #     for key, value in data_dict.items():
            #         file.create_dataset(key, data=value)

            # Save frames and labels to PKL file
            pkl_file_name = f"clip_{file_counter:06d}.pkl"
            pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)

            with open(pkl_file_path, 'wb') as file:
                pickle.dump(data_dict, file)
                print("Pkl file created:" +pkl_file_name)

                
            file_counter += 1

            # Clear data for the next batch
            all_data = []

# Example: Print the total number of files created
print("Total files created:", file_counter)