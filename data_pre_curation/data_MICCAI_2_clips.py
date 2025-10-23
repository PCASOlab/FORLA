import os
import cv2
import pickle
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
output_dir = "/media/guiqiu/surgvu24pkl/"
# os.makedirs(output_dir, exist_ok=True)

data_root_dir = "/media/guiqiu/surgvu24/"

def get_video_length(video_path):
    """
    Get the length of a video in seconds.
    
    :param video_path: Path to the video file.
    :return: Length of the video in seconds.
    """
    with VideoFileClip(video_path) as video:
        video_length = video.duration  # Duration in seconds
        return video_length
def extract_frames_from_task(video_path, start_time, stop_time, resize_dim=(224, 224), fps=0.5):
    """
    Extract frames from a task between start_time and stop_time at a specified frequency and resize them.
    
    :param video_path: Path to the video file.
    :param start_time: Start time of the task in seconds.
    :param stop_time: Stop time of the task in seconds.
    :param resize_dim: Tuple with the desired dimensions (width, height) for resizing the frames.
    :param fps: Frames per second to extract (e.g., 1 frame every second, 1 frame every 2 seconds).
    :return: 4D numpy array of resized extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    frame_interval = int(1000 / fps)  # Calculate the interval in milliseconds based on fps
    current_time = start_time
    while current_time <= stop_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, frame = cap.read()
        if not success:
            break
        # Resize the frame
        H, W, _ = frame.shape

        frame = frame[0:H-80, 192:1088]
        frame = cv2.resize(frame, resize_dim)
        frames.append(frame)
        current_time += frame_interval / 1000  # Increment time by frame interval

    cap.release()
    return np.array(frames)

def save_task_as_pkl(output_dir, file_counter, frames, tool_presence, clip_length):
    """
    Save task data (frames, tool presence, clip length) as a .pkl file.
    
    :param output_dir: Directory to save the .pkl file.
    :param task_name: Name of the task.
    :param frames: 4D numpy array of extracted frames.
    :param tool_presence: Binary vector indicating tool presence.
    :param clip_length: Length of the task clip in seconds.
    """
    task_data = {
        'frames': frames,
        'tool_presence': tool_presence,
        'clip_length': clip_length
    }
    task_data['clip_length'] = len(frames)
    task_data['frames'] = np.transpose(task_data['frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 64, 64)

    pkl_file_name = f"clip_{file_counter:06d}.pkl"
    output_path = os.path.join(output_dir, pkl_file_name)
    with open(output_path, 'wb') as f:
        pickle.dump(task_data, f)
    print(pkl_file_name)


def check_already_generated(output_dir, file_counter ):
      
    pkl_file_name = f"clip_{file_counter:06d}.pkl"
    output_path = os.path.join(output_dir, pkl_file_name)
    return os.path.isfile(output_path)

file_counter =0
for sub_set in sorted(os.listdir(data_root_dir)):
    if "case" in sub_set and sub_set!= "case_071":
        sub_case_dir = os.path.join(data_root_dir, sub_set)
        
        task_file_path = os.path.join(sub_case_dir, 'extend_tasks_with_tools.csv')
        tasks_df = pd.read_csv(task_file_path)
        
        video_parts = [f for f in os.listdir(sub_case_dir) if f.endswith('.mp4')]
        video_parts = sorted(video_parts)
        
        part1_path = os.path.join(sub_case_dir, video_parts[0])
        part1_length = get_video_length(part1_path)
        part2_path = os.path.join(sub_case_dir, video_parts[1]) if len(video_parts) > 1 else None
        
        for task_idx, task_row in tasks_df.iterrows():
            if check_already_generated(output_dir,file_counter) ==False:
                start_time = task_row['start_time']
                stop_time = task_row['stop_time']
                tool_presence = task_row['tool_presence']
                clip_length = stop_time - start_time
                
                if stop_time <= part1_length:
                    # Task is entirely within part 1
                    frames = extract_frames_from_task(part1_path, start_time, stop_time)
                elif start_time >= part1_length and part2_path:
                    # Task is entirely within part 2
                    start_time_in_part2 = start_time - part1_length
                    stop_time_in_part2 = stop_time - part1_length
                    frames = extract_frames_from_task(part2_path, start_time_in_part2, stop_time_in_part2)
                else:
                    # Task spans across part 1 and part 2
                    stop_time_in_part1 = part1_length
                    frames_part1 = extract_frames_from_task(part1_path, start_time, stop_time_in_part1)
                    
                    start_time_in_part2 = 0
                    stop_time_in_part2 = stop_time - part1_length
                    frames_part2 = extract_frames_from_task(part2_path, start_time_in_part2, stop_time_in_part2)
                    
                    frames = np.concatenate((frames_part1, frames_part2), axis=0)
                
                print(sub_set)
                
                save_task_as_pkl(output_dir, file_counter , frames, eval(tool_presence), clip_length)
            else:
                print (str(file_counter) + "already generated")
            file_counter+=1
