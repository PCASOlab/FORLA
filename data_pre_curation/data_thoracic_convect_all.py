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
from working_dir_root import working_pcaso_raid
Crop_flag = True
Video_format =  ".mp4"
from visdom import Visdom
viz = Visdom(port=8097)

# Video_format =  ".mpg"


Jsonfile_dir = working_pcaso_raid +'Thoracic/label/' # the folder for all jsons
Json_to_decode = ['labelsB','labelsA'] 
Raw_video_dir = working_pcaso_raid +'Thoracic/video#7/' # folder for all videos folders with number
Decoded_data_dir= working_pcaso_raid +'Thoracic/interim/' # output folder
output_folder_pkl = working_pcaso_raid +'Thoracic/pkl/'
output_folder_sam_feature = working_pcaso_raid +'Thoracic/output_sam_features/'
Using_normalized = True
video_buff_size = 30 
image_resize = 256

# json_data = file.read()
categories = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
]
category_colors = {
    'Lymph node': (0, 0, 255),        # Red
    'Vagus nereve': (0, 255, 0),      # Green
    'Bronchus': (255, 0, 0),          # Blue
    'Lung parenchyma': (255, 255, 0),  # Yellow
    'Instruments': (255, 0, 255),      # Magenta
}
# Function to save sampled clips and masks as pickle files

def get_specific_frame(video_path,frame_id):
     # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = int(frame_id)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
        exit()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set the frame number you want to extract (41st frame in this case)
    # frame_number = 41
    # Set the capture object's position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number )

    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        exit()

    # Display or process the frame as needed
    # For example, you can save the frame to an image file
    # cv2.imwrite(f"frame_{frame_number}.png", frame)

    # Release the capture object
    cap.release()

    print(f"Frame {frame_number} extracted successfully.")
    return frame

def find_symmetric_bounding_box(image, threshold=20):
     
    H, W, _ = image.shape
    # Convert the image to grayscale to simplify the analysis
    grayscale_image = np.mean(image, axis=2).astype(np.uint8)
    # Create vertical and horizontal projections by averaging pixel intensities along each axis
    vertical_projection = np.mean(grayscale_image, axis=1)  # Average each row
    horizontal_projection = np.mean(grayscale_image, axis=0)  # Average each column
    # For robustness, we focus on the top half and left half for calculating H1 and W1, assuming symmetry
    H_half = H // 2
    W_half = W // 2
    # Calculate H1 (top border) by analyzing the top half
    H1 = np.argmax(vertical_projection[:H_half] > threshold)
    # Since the border is symmetric, H2 can be calculated as:
    H2 = H - H1
    # Calculate W1 (left border) by analyzing the left half
    W1 = np.argmax(horizontal_projection[:W_half] > threshold)
    # Similarly, W2 can be calculated as:
    W2 = W - W1

    H1 = min(H1, H // 4)  # Enforce top limit
    H2 = max(H2, 3 * H // 4)  # Enforce bottom limit
    W1 = min(W1, W // 4)  # Enforce left limit
    W2 = max(W2, 3 * W // 4)  # Enforce right limit
    return [H1, H2, W1, W2]
def load_a_video_buffer(video_path, video_buff_size, image_size  ):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_down_sample = int((total_frames - 1) / video_buff_size)
    mid_id = total_frames//2
    mid_frame = get_specific_frame(video_path,mid_id)
    annotation_masks_full_buffer= []
    if   mid_frame is not None:
                bounding_coord=find_symmetric_bounding_box(mid_frame, threshold=20)
                print("crop:")
                print(bounding_coord)
                 
    frame_count = 0
    buffer_count = 0
    video_buffer = np.zeros((3, video_buff_size, image_size, image_size))
    
    Valid_video = False
    if total_frames <= video_buff_size:  
        return video_buffer, annotation_masks_full_buffer, Valid_video

    while True:
        if frame_count % video_down_sample == 0:
            ret, frame = cap.read()

            if ret == True:
                H, W, _ = frame.shape
                if bounding_coord is not None:
                    [h1,h2,w1,w2]= bounding_coord
                    frame = frame[h1:h2,w1:w2]
                this_resize = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                reshaped = np.transpose(this_resize, (2, 0, 1))
                video_buffer[:, buffer_count, :, :] = reshaped
                buffer_count += 1

            if buffer_count >= video_buff_size:
                Valid_video = True
                break
        else:
            ret = cap.grab()

        if not ret:
            break

        frame_count += 1
 
    return video_buffer.astype(np.uint8), annotation_masks_full_buffer, Valid_video
def save_sampled_clip_and_masks(pkl_file_name, video_images, video_masks, output_folder_pl):
    pkl_file_name = pkl_file_name + '.pkl'
    data_dict = {'frames': video_images, 'labels': video_masks}
    # pkl_file_name = f"clip_{file_counter:06d}.pkl"
    pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data_dict, file)
        print("Pkl file created:", pkl_file_name)

def main():
    video_clips = []

    # Walk through the subdirectories and find all .mp4 files
    for root, _, files in os.walk(Raw_video_dir):
        for file in files:
            if file.endswith(".mp4"):
                # Append the full path of each video to the list
                video_clips.append(os.path.join(root, file))

    # Print the list of video clips
    for video_path in video_clips:
        # video_path = '/media/guiqiu/Weakly_supervised_data/Thoracic/video#7/video2024/__MACOSX/._Kazu_RATS_RUL_#7_94_phase2_6_normalized.mp4'
        video_buffer, annotation_masks_full_buffer, valid_video = load_a_video_buffer(video_path, video_buff_size, image_resize)
        if valid_video:
            # index += 1
            video_name = os.path.basename(video_path)
    
            # Remove '_normalized' if present before '.mp4'
            video_name = video_name.replace('_normalized', '')

            # Remove the '.mp4' extension
            video_name = os.path.splitext(video_name)[0]
            save_pkl_name = video_name
            print("Got a valid video", save_pkl_name, video_buffer.shape)
            # sam_and_save_features(index, video_buffer, Vit_encoder, output_folder_sam_feature)
            save_sampled_clip_and_masks(save_pkl_name, video_buffer, annotation_masks_full_buffer, output_folder_pkl)
        else:
            print("Error: Video is not valid", video_path)
if __name__ == '__main__':
   main()