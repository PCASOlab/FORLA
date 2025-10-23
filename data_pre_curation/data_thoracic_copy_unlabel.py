import cv2
import os
import  numpy as np
# from working_dir_root import Dataset_video_root, Dataset_label_root
# import csv
import re
import json
from PIL import Image, ImageDraw
# from dataset.io import save_a_image
# from dataset.io import save_a_pkl_w_create as save_a_pkl
import pickle
from working_dir_root import working_pcaso_raid
Crop_flag = True
Video_format =  ".mp4"
from visdom import Visdom
viz = Visdom(port=8097)
import shutil
# Video_format =  ".mpg"


# Jsonfile_dir = working_pcaso_raid +'Thoracic/label/' # the folder for all jsons
# Json_to_decode = ['labelsB','labelsA'] 
# Raw_video_dir = working_pcaso_raid +'Thoracic/video#7/' # folder for all videos folders with number
# Decoded_data_dir= working_pcaso_raid +'Thoracic/interim/' # output folder
# output_folder_pkl = working_pcaso_raid +'Thoracic/pkl/'
# output_folder_sam_feature = working_pcaso_raid +'Thoracic/output_sam_features/'
# Using_normalized = True
# video_buff_size = 30 
# image_resize = 256

# json_data = file.read()


def copy_unannotated_pkls(source_folder, annotated_folder, destination_folder):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get list of all PKL files in source folder
    source_files = [f for f in os.listdir(source_folder) if f.endswith('.pkl')]
    
    for filename in source_files:
        # Construct possible annotated filenames
        annotated_a = f"labelsA_{filename}"
        annotated_b = f"labelsB_{filename}"
        
        # Check if either annotated version exists
        a_exists = os.path.exists(os.path.join(annotated_folder, annotated_a))
        b_exists = os.path.exists(os.path.join(annotated_folder, annotated_b))
        
        if not (a_exists or b_exists):
            # Copy the file if no annotations exist
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(destination_folder, filename)
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename}")

# Example usage
source_folder = working_pcaso_raid +'Thoracic/pkl/'
annotated_folder = working_pcaso_raid +'Thoracic/pkl backup annotation/pkl/'
destination_folder = working_pcaso_raid +'Thoracic/unannotated/pkl/'

copy_unannotated_pkls(source_folder, annotated_folder, destination_folder)