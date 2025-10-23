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
import pandas as pd
# from working_dir_root import SAM_pretrain_root,sam_feature_OLG_dir3
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from moviepy.editor import VideoFileClip
data_root_dir = "/media/guiqiu/surgvu24/"

def time_to_seconds(time_str):
    """
    Convert a time string in the format 'HH:MM:SS' or 'HH:MM:SS.microseconds' to seconds.
    
    :param time_str: Time string in the format 'HH:MM:SS' or 'HH:MM:SS.microseconds'.
    :return: Total time in seconds as a float.
    """
    try:
        # Try parsing with microseconds
        time_format = "%H:%M:%S.%f"
        time_obj = datetime.strptime(time_str, time_format)
    except ValueError:
        # If it fails, parse without microseconds
        time_format = "%H:%M:%S"
        time_obj = datetime.strptime(time_str, time_format)
    
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6
    return total_seconds
def seconds_to_time(seconds):
    """
    Convert seconds to a time string in the format 'HH:MM:SS.microseconds'.
    
    :param seconds: Time in seconds.
    :return: Time string in the format 'HH:MM:SS.microseconds'.
    """
    microseconds = int((seconds - int(seconds)) * 1e6)
    time_str = datetime.utcfromtimestamp(seconds).strftime(f"%H:%M:%S.{microseconds:06d}")
    return time_str


def get_video_length(video_path):
    """
    Get the length of a video in seconds.
    
    :param video_path: Path to the video file.
    :return: Length of the video in seconds.
    """
    with VideoFileClip(video_path) as video:
        video_length = video.duration  # Duration in seconds
        return video_length
def adjust_time_for_part(time_in_seconds, part, part1_length):
    """
    Adjust the time based on the video part.

    :param time_in_seconds: Original time in seconds.
    :param part: The part of the video (1 or 2).
    :param part1_length: The length of the first part of the video.
    :return: Adjusted time in the merged video.
    """
    return time_in_seconds + (part - 1) * part1_length

def are_intervals_overlapping(pair1, pair2):
    start1, end1 = pair1
    start2, end2 = pair2

    if start1 < end2 and start2 < end1:
        return True
    return False


def get_tool_names(binary_vector, categories):
    """
    Convert a binary vector into a comma-separated string of tool names.
    
    :param binary_vector: Binary vector indicating the presence of tools.
    :param categories: List of tool categories.
    :return: Comma-separated string of tool names.
    """
    tool_names = [categories[i] for i, presence in enumerate(binary_vector) if presence]
    return ', '.join(tool_names)

# Tool categories and their binary vector positions
categories = [
    'bipolar dissector', 
    'bipolar forceps', 
    'cadiere forceps', 
    'clip applier', 
    'force bipolar',
    'grasping retractor',
    'monopolar curved scissors',
    'needle driver',
    'permanent cautery hook/spatula', 
    'prograsp forceps',
    'stapler', 
    'suction irrigator', 
    'tip-up fenestrated grasper', 
    'vessel sealer' 
]
for sub_set in sorted(os.listdir(data_root_dir)):
    if "case" in sub_set:
        sub_case_dir = os.path.join(data_root_dir, sub_set)
        
        # Load the tasks.csv file
        task_file_path = os.path.join(sub_case_dir, 'tasks.csv')
        tasks_df = pd.read_csv(task_file_path)

        # Load the tools.csv file
        tool_file_path = os.path.join(sub_case_dir, 'tools.csv')
        tools_df = pd.read_csv(tool_file_path)
        
        # Remove rows with negative times
        tasks_df = tasks_df[(tasks_df['start_time'] >= 0) & (tasks_df['stop_time'] >= 0)]
        # tools_df = tools_df[(tools_df['install_case_time'] >= 0) & (tools_df['uninstall_case_time'] >= 0)]

        # Remove rows where end_time is smaller than start_time only if they are within the same part
        tasks_df = tasks_df[(tasks_df['start_part'] != tasks_df['stop_part']) | (tasks_df['stop_time'] >= tasks_df['start_time'])]
        # tools_df = tools_df[(tools_df['install_case_part'] != tools_df['uninstall_case_part']) | (tools_df['uninstall_case_time'] >= tools_df['install_case_time'])]



        # Get the video parts
        video_parts = [f for f in os.listdir(sub_case_dir) if f.endswith('.mp4')]
        video_parts = sorted(video_parts)
        
        # Get the length of the first part
        part1_path = os.path.join(sub_case_dir, video_parts[0])
        part1_length = get_video_length(part1_path)
        part2_path = os.path.join(sub_case_dir, video_parts[1]) if len(video_parts) > 1 else None

        if part2_path:
            part2_length = get_video_length(part2_path)
        else:
            part2_length =0
        
        # Convert and adjust times in tasks DataFrame
        tasks_df['start_time'] = tasks_df.apply(lambda row: adjust_time_for_part(row['start_time'], row['start_part'], part1_length), axis=1)
        tasks_df['stop_time'] = tasks_df.apply(lambda row: adjust_time_for_part(row['stop_time'], row['stop_part'], part1_length), axis=1)

        # Convert and adjust times in tools DataFrame
        tools_df['install_case_time'] = tools_df.apply(lambda row: adjust_time_for_part(time_to_seconds(row['install_case_time']), row['install_case_part'], part1_length), axis=1)
        tools_df['uninstall_case_time'] = tools_df.apply(lambda row: adjust_time_for_part(time_to_seconds(row['uninstall_case_time']), row['uninstall_case_part'], part1_length), axis=1)

        # Print the adjusted DataFrames
        print("Adjusted Tasks DataFrame:")
        print(tasks_df)
        print("\nAdjusted Tools DataFrame:")
        print(tools_df)
    
        # Initialize a binary vector for each task
        tasks_df['tool_presence'] = tasks_df.apply(lambda row: [0] * len(categories), axis=1)
        
        # Check tool presence for each task
        for tool_idx, tool_row in tools_df.iterrows():
            tool_name = tool_row['groundtruth_toolname']
            if tool_name in categories:
                tool_install_time = tool_row['install_case_time']
                tool_uninstall_time = tool_row['uninstall_case_time']
                tool_interval = [tool_install_time, tool_uninstall_time]

                # Determine which category index corresponds to this tool
                tool_category_index = categories.index(tool_name)

                for task_idx, task_row in tasks_df.iterrows():
                    task_interval = [task_row['start_time'], task_row['stop_time']]

                    # Check if tool interval overlaps with task interval
                    if are_intervals_overlapping(task_interval, tool_interval):
                        # Update the binary vector to indicate tool presence
                        tasks_df.at[task_idx, 'tool_presence'][tool_category_index] = 1
        # Add a column for comma-separated tool names
        tasks_df['tools_string'] = tasks_df['tool_presence'].apply(lambda x: get_tool_names(x, categories))
        
        # Add columns for time in 'HH:MM:SS.microseconds' format
        tasks_df['start_time_str'] = tasks_df['start_time'].apply(seconds_to_time)
        tasks_df['stop_time_str'] = tasks_df['stop_time'].apply(seconds_to_time)
        tools_df['install_case_time_str'] = tools_df['install_case_time'].apply(seconds_to_time)
        tools_df['uninstall_case_time_str'] = tools_df['uninstall_case_time'].apply(seconds_to_time)
        # Save the updated DataFrame to a new CSV file
        output_task_file_path = os.path.join(sub_case_dir, 'tasks_with_tools.csv')
        tasks_df.to_csv(output_task_file_path, index=False)
        output_tool_file_path = os.path.join(sub_case_dir, 'tool_in_seconds_merged.csv')
        tools_df.to_csv(output_tool_file_path, index=False)



        for task_idx, task_row in tasks_df.iterrows():
            task_interval = [task_row['start_time'], task_row['stop_time']]
            original_start = task_row['start_time']
            original_end = task_row['stop_time']
            
            old_presence_vector = tasks_df.at[task_idx, 'tool_presence']
            
            
            for extension_time in [180, 150, 120, 90,60,30,0]:
                extended_start = max(original_start - extension_time, 0)
                extended_end = min (original_end + extension_time, part2_length + part1_length)
                task_interval = [extended_start,extended_end]
                new_presence_vector = [0] * len(categories)

                
                for tool_idx, tool_row in tools_df.iterrows():
                    tool_name = tool_row['groundtruth_toolname']
                    if tool_name in categories:
                            tool_install_time = tool_row['install_case_time']
                            tool_uninstall_time = tool_row['uninstall_case_time']
                            tool_category_index = categories.index(tool_row['groundtruth_toolname'])
                            tool_interval = [tool_install_time,tool_uninstall_time]

                            # Check if tool interval overlaps with task interval
                            if are_intervals_overlapping(task_interval, tool_interval):
                                    new_presence_vector[tool_category_index] =1
                if new_presence_vector == old_presence_vector:
                    tasks_df.at[task_idx, 'start_time'] = extended_start
                    tasks_df.at[task_idx, 'stop_time'] = extended_end
                    break
                else:
                    print("this extension is too long try next")  # Reset the new presence vector to try the next extension

        tasks_df['start_time_str'] = tasks_df['start_time'].apply(seconds_to_time)
        tasks_df['stop_time_str'] = tasks_df['stop_time'].apply(seconds_to_time)
        output_task_file_path = os.path.join(sub_case_dir, 'extend_tasks_with_tools.csv')
        tasks_df.to_csv(output_task_file_path, index=False)

        tasks = tasks_df[['groundtruth_taskname', 'start_time', 'stop_time','start_part','stop_part','tool_presence']]
        tools = tools_df[['groundtruth_toolname', 'install_case_time', 'uninstall_case_time', 'install_case_part','uninstall_case_part' ]]


    print("fin!")



    
# for sub_set in sorted(os.listdir(data_root_dir)):
#     sub_case_dir = os.path.join(data_root_dir,sub_set)
#     # Load the spreadsheet (replace 'your_file_path.xlsx' with the actual file path)
#     task_file_path = os.path.join(sub_case_dir,'tasks.op_time','start_part','stop_part'csv') # Replace with your actual file path
#     df = pd.read_csv(task_file_path)







#     # Extracting the tasks with their corresponding start and stop times
#     tasks = df[['groundtruth_taskname', 'start_time', 'st]]
    

#     tool_file_path = os.path.join(sub_case_dir,'tools.csv') # Replace with your actual file path
#     df = pd.read_csv(tool_file_path)

#     # Extracting the tasks with their corresponding start and stop times
#     tools = df[['groundtruth_toolname', 'install_case_time', 'uninstall_case_time', 'install_case_part','uninstall_case_part' ]]

#     video_parts = [f for f in os.listdir(sub_case_dir) if f.endswith('.mp4')]
#     video_parts = sorted (video_parts)
#     part1_path =  os.path.join(sub_case_dir, video_parts[0])
#     part1len =   get_video_length(part1_path)
#     for video_part in video_parts:
#         video_path = os.path.join(sub_case_dir, video_part)
#         # extract_frames(video_path, output_folder, fps)
#      # Convert the time columns to seconds
#     tools['install_case_time'] = tools['install_case_time'].apply(time_to_seconds)
#     tools['uninstall_case_time'] = tools['uninstall_case_time'].apply(time_to_seconds)



#     # Print the tasks and their times
#     print(tools)
     
 