import os
import csv
# from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import dataset.io as io

import pickle
align_video_length = 100
# Function to merge video clips with the same labels
def merge_sequential_videos_with_same_labels(video_dict, video_clip_dir, output_folder, output_csv_folder):
    new_labels_data = []
    current_label = None
    current_clip_group = []

    for clip_name, label_string in video_dict.items():
        # Check if the label is the same as the current group
        if label_string == current_label:
            clip_path = os.path.join(video_clip_dir, clip_name)
            current_clip_group.append((clip_name, clip_path))
        else:
            # Merge clips in the current group
            if current_clip_group:
                video_arrays = [read_a_pkl(path) for _, path in current_clip_group]
                merged_clip = concatenate_videos([clip for clip in video_arrays if clip is not None])
                adjusted_clip = adjust_video_length(merged_clip, align_video_length)
                output_path = os.path.join(output_folder, f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}.pkl")
                save_as_pkl(adjusted_clip, output_path)
                new_labels_data.append([f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}", current_label])

            # Start a new group with the current label
            current_label = label_string
            current_clip_group = [(clip_name, os.path.join(video_clip_dir, clip_name))]

    # Merge clips in the last group
    if current_clip_group:
        video_arrays = [read_a_pkl(path) for _, path in current_clip_group]
        merged_clip = concatenate_videos([clip for clip in video_arrays if clip is not None])
        adjusted_clip = adjust_video_length(merged_clip, align_video_length)
        output_path = os.path.join(output_folder, f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}.pkl")
        save_as_pkl(adjusted_clip, output_path)
        new_labels_data.append([f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}", current_label])

    # Write new labels to CSV file
    new_csv_file_path = os.path.join(output_csv_folder, "new_labels.csv")
    with open(new_csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['index', 'clip_name', 'tools_present'])  # Updated header
        for idx, row in enumerate(new_labels_data, start=1):  # Start counting from 1
            csvwriter.writerow([idx] + row)  # Add index to each row

# Load all labels and video clip paths from the CSV file
def load_all_lables(csv_file_path): # load all labels and save then as dictionary format
         

        # Initialize an empty list to store the data from the CSV file
        data = []
       
        # Open the CSV file and read its contents
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                csvreader = csv.reader(csvfile)

                # Read the header row (if any)
                header = next(csvreader)

                # Read the remaining rows and append them to the 'data' list
                for row in csvreader:
                    data.append(row)
                    
        except FileNotFoundError:
            print(f"File not found at path: {csv_file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

        # Now you have the data from the CSV file in the 'data' list
        # You can manipulate or process the data as needed

        # Example: Printing the first few rows
        for row in data[:5]:
            print("all data is loaded and here are some samples:")
            print(row)
        labels = data
        # conver label list into dictionary that can used key for fast lookingup
        label_dict = {label_info[1]: label_info[2] for label_info in labels}  # use the full name as the dictionary key
        label_dict_number = {label_info[0]: label_info[2] for label_info in
                             labels}  # using the number and dictionary keey instead

        all_labels = label_dict
        return all_labels
def adjust_video_length(video_array, target_length):
    """
    Adjust the length of the video array to match the target length.
    
    Parameters:
        video_array (numpy.ndarray): Input video array of shape (channels, frames, height, width).
        target_length (int): Desired length of the output video.
        
    Returns:
        numpy.ndarray: Video array with adjusted length.
    """
    current_length = video_array.shape[1]
    if current_length == target_length:
        return video_array  # No adjustment needed
    
    if current_length < target_length:  # Upsample (copy frames)
        repetitions = int(np.ceil(target_length / current_length))
        adjusted_video = np.concatenate([video_array] * repetitions, axis=1)
        return adjusted_video[:, :target_length, :, :]
    else:  # Downsample (remove frames)
        indices = np.round(np.linspace(0, current_length - 1, target_length)).astype(int)
        adjusted_video = video_array[:, indices, :, :]
        return adjusted_video
# Function to concatenate numpy arrays along the temporal dimension
def concatenate_videos(video_arrays):
    return np.concatenate(video_arrays, axis=1)

# Function to read a video from a pickle file
# Function to read a video from a pickle file
def read_a_pkl(path):
    try:
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

# Function to save a numpy array as a pickle file
def save_as_pkl(video_array, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(video_array, f)

# Main function
def main():
    video_clip_dir = "C:/2data/training_data/video_clips_pkl/"
    output_folder = "C:/2data/training_data/new_video_clips/"
    csv_file_path = "C:/2data/training_data/labels.csv"
    output_csv_folder= "C:/2data/training_data/"
    os.makedirs(output_folder, exist_ok=True)

    # Load all labels and video clip paths
    all_labels = load_all_lables(csv_file_path)

    # Merge sequential video clips with the same labels
    merge_sequential_videos_with_same_labels(all_labels, video_clip_dir, output_folder,output_csv_folder)

if __name__ == "__main__":
    main()