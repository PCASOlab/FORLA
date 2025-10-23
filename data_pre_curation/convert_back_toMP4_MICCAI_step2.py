import os
import subprocess

# Paths
input_folder = r"C:\2data\training_data\reconstructed_videos"
output_folder = r"C:\2data\training_data\re_encoded_videos"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through the videos in the input folder
for video_file in os.listdir(input_folder):
    if video_file.endswith('.mp4'):
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, video_file)
        
        # FFmpeg command to re-encode the video with H.264 codec and fixed frame rate
        ffmpeg_command = [
            'ffmpeg', '-i', input_video_path,
            '-c:v', 'libx264',   # Use H.264 video codec
            '-preset', 'slow',   # Encoding speed vs quality balance
            '-crf', '22',        # Quality level (lower is better, range 0-51, 18-23 is reasonable)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility with browsers
            '-r', '1',           # Set frame rate to 1 FPS
            '-movflags', '+faststart',  # Optimize for web streaming
            output_video_path
        ]

        # Run the FFmpeg command
        subprocess.run(ffmpeg_command)

print("Re-encoding complete.")
