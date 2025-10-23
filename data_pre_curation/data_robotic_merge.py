import os
import csv
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np

# read from mp4
# Function to merge video clips with the same labels
def merge_sequential_videos_with_same_labels(video_dict, video_clip_dir, output_folder, output_csv_folder,compression_factor=0.5):
    new_labels_data = []
    current_label = None
    current_clip_group = []

    for clip_name, label_string in video_dict.items():
       

        # Check if the label is the same as the current group
        if label_string == current_label:
            clip_path = os.path.join(video_clip_dir, clip_name)
            clip_num = int(clip_name.split('_')[1])
            current_clip_group.append((clip_num, clip_path))
        else:
            # Merge clips in the current group
            if current_clip_group:
                merged_clip = concatenate_videoclips([VideoFileClip(path + ".mp4") for _, path in current_clip_group])
                compressed_clip = merged_clip.resize(height=int(merged_clip.size[1] * compression_factor))
                output_path = os.path.join(output_folder, f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}.mp4")
                compressed_clip.write_videofile(output_path,   threads = 32,fps=2, audio=False)

                # Add entry to the new labels CSV file
                new_labels_data.append([f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}", current_label])
            
            # Start a new group with the current label
            current_label = label_string
            current_clip_group = [(int(clip_name.split('_')[1]), os.path.join(video_clip_dir, clip_name))]

    # Merge clips in the last group
    if current_clip_group:
        merged_clip = concatenate_videoclips([VideoFileClip(path) for _, path in current_clip_group])
        output_path = os.path.join(output_folder, f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}.mp4")
        merged_clip.write_videofile(output_path)

        # Add entry to the new labels CSV file
        new_labels_data.append([f"{current_clip_group[0][0]}_{current_clip_group[-1][0]}", current_label])

    # Write new labels to CSV file
    new_csv_file_path = os.path.join(output_csv_folder, "new_labels.csv")
    with open(new_csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['clip_name', 'label'])
        csvwriter.writerows(new_labels_data)

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

# Main function
def main():
    video_clip_dir = "C:/2data/training_data/video_clips/"
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