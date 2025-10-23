import cv2
import os

# Paths
input_folder = r"C:\2data\training_data\selected_video_frame_sequence"
output_folder = r"C:\2data\training_data\reconstructed_videos"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through the subfolders in the input folder
for video_folder in os.listdir(input_folder):
    video_folder_path = os.path.join(input_folder, video_folder)
    
    if os.path.isdir(video_folder_path):
        # Collect all frame files and sort them by name (to maintain frame order)
        frames = sorted([f for f in os.listdir(video_folder_path) if f.endswith('.jpg')])
        
        # Define the output video path
        output_video_path = os.path.join(output_folder, f"{video_folder}.mp4")
        
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        frame_rate = 1  # 1 frame per second
        frame_size = (256, 256)  # Frame size (256x256)
        
        video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)
        
        # Iterate over the frames and write them to the video
        for frame_file in frames:
            frame_path = os.path.join(video_folder_path, frame_file)
            frame = cv2.imread(frame_path)
            
            # Ensure the frame is correctly sized (256x256)
            if frame is not None and frame.shape[0:2] == frame_size:
                video_writer.write(frame)
            else:
                print(f"Skipping frame {frame_file} due to size mismatch or read error.")

        # Release the video writer once done
        video_writer.release()

print("Videos reconstructed from frames.")
