import cv2
import os

# Paths
selected_folder = '/data/Cholec_new_selected/mp4/'
output_folder = '/data/Cholec_new_selected/frame_sequence/'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through the selected MP4 videos
for video_file in os.listdir(selected_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(selected_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # Get the video name without extension

        # Create a subfolder for each video to store its frames
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the video's original FPS
        frame_rate = int(fps)  # Convert FPS to an integer
        sec_interval = frame_rate  # One frame per second

        frame_count = 0  # Counter for frames
        save_count = 0   # Counter for saved frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit the loop if no frame is captured

            # Save every second (1 frame per second)
            if frame_count % 1 == 0:
                # Get the height and width of the original frame
                H, W, _ = frame.shape

                # Apply the cropping
                # cropped_frame = frame[56:H-80, 192:1088]  # Cropping the frame

                # Resize the cropped frame to 256x256
                # resized_frame = cv2.resize(cropped_frame, (256, 256))

                # Save the resized frame as a JPEG image
                frame_filename = f"{save_count:05d}.jpg"
                frame_path = os.path.join(video_output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                save_count += 1
            
            frame_count += 1

        cap.release()  # Release the video file

print("Frame extraction, cropping, and resizing complete.")
