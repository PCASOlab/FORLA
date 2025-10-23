import cv2
import numpy as np
import os
import random
# from matplotlib.pyplot import *
# # from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# import matplotlib.pyplot as plt
# # PythonETpackage for xml file edition
import shutil
import tempfile
import pickle
import subprocess
def write_mp4_from_tensor_last_in_batch(
    input_videos_np,
    out_path,
    fps=1,
    codec='mp4v',
    quality=None
):
    """
    Save the last video in a batch to an MP4 file, then re-encode to
    H.264 (yuv420p,+faststart) for maximum compatibility.
    """
    assert input_videos_np.ndim == 5 and input_videos_np.shape[1] == 3, \
        f"Expected (B,3,T,H,W), got {input_videos_np.shape}"

    # Last item -> (3, T, H, W) -> (T, H, W, C)
    vid_last = input_videos_np[-1]
    frames_rgb = np.transpose(vid_last, (1, 2, 3, 0))

    # To uint8 [0,255]
    if frames_rgb.dtype != np.uint8:
        if frames_rgb.max() <= 1.5:
            frames_rgb = (frames_rgb * 255.0).clip(0, 255)
        frames_rgb = frames_rgb.astype(np.uint8)

    T, H, W, C = frames_rgb.shape

    # Ensure output folder exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Pad to even dims if needed (required by yuv420p/H.264)
    pad_h = H % 2
    pad_w = W % 2
    if pad_h or pad_w:
        padded = np.zeros((T, H + pad_h, W + pad_w, C), dtype=np.uint8)
        padded[:, :H, :W, :] = frames_rgb
        frames_rgb = padded
        H, W = frames_rgb.shape[1:3]

    # Write an intermediate file with OpenCV
    with tempfile.TemporaryDirectory() as td:
        temp_path = os.path.join(td, "intermediate.mp4")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            # fallback to MJPG AVI (very reliable intermediate)
            temp_path = os.path.join(td, "intermediate.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            raise IOError(f"Failed to open VideoWriter for path: {temp_path}")

        if quality is not None:
            try:
                writer.set(cv2.CAP_PROP_BITRATE, int(quality))
            except Exception as e:
                print(f"Warning: Unable to set CAP_PROP_BITRATE: {e}")

        try:
            for t in range(T):
                bgr = frames_rgb[t] 
                # bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)

                writer.write(bgr)
        finally:
            writer.release()

        # If ffmpeg missing, just copy intermediate and return
        if not shutil.which('ffmpeg'):
            print("Warning: ffmpeg not found; returning intermediate file as-is.")
            shutil.copy2(temp_path, out_path)
            return out_path

        # Build ffmpeg command:
        # -y: overwrite
        # -r <fps> BEFORE -i is for input, AFTER -i is output; here we just set output to your fps
        # -pix_fmt yuv420p for compatibility, +faststart for better playback
        # Safety net: force even dims again via scale in case of unexpected metadata
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_path,
            '-c:v', 'libx264',
            '-preset', 'slow',
            '-crf', '22',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-r', str(fps),
            '-an',
            out_path
        ]
        try:
            # Capture stderr so we can show errors
            res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b'').decode('utf-8', errors='ignore')
            print("FFmpeg conversion failed. stderr:\n", err)
            # As a fallback, deliver the intermediate file (better than nothing)
            shutil.copy2(temp_path, out_path)
        # temp files auto-removed by context manager

    return out_path
def self_check_path_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_a_pkl(dir,name,object):
    with open(dir + name +'.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
    pass

def save_a_pkl_w_create(dir,name,object):
    self_check_path_create(dir)
    with open(dir + name +'.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)
    pass

def read_a_pkl(dir, name):
  """
  Reads a pickled object from a file, handling potential missing files and encoding issues.

  Args:
      dir (str): The directory containing the pickle file.
      name (str): The name of the pickle file (without the .pkl extension).

  Returns:
      object: The loaded object from the pickle file, or None if the file is missing.
  """

  # Check if the file exists before attempting to load
  if not os.path.isfile(dir + name + '.pkl'):
    print(f"File '{dir + name + '.pkl'}' not found. Returning None.")
    return None

  try:
    # Attempt to load the object with 'iso-8859-1' encoding (optional)
    object = pickle.load(open(dir + name + '.pkl', 'rb'), encoding='iso-8859-1')
  except (EOFError, pickle.UnpicklingError) as e:
    print(f"Error loading pickle file '{dir + name + '.pkl'}: {e}. Returning None.")
    return None

  return object


def save_img_to_folder(this_save_dir,ID,img):
    # this_save_dir = Output_root + "1out_img/" + Model_key + "/ground_circ/"
    if not os.path.exists(this_save_dir):
        os.makedirs(this_save_dir)
    cv2.imwrite(this_save_dir +
                str(ID) + ".jpg", img)
    
def save_a_image(dir,name,image):
    self_check_path_create(dir)
    cv2.imwrite(dir+name, image)


def load_a_video_buffer(video_path,video_buff_size,image_size,annotated_frame_ID,Display_loading_video ):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_down_sample = int((total_frames-1)/video_buff_size)
        # Read frames from the video clip
        frame_count = 0
        buffer_count = 0
        # Read frames from the video clip
        video_buffer = np.zeros((3, video_buff_size,   image_size,  image_size))
         
        frame_number =0
        Valid_video=False
        this_frame = 0
        previous_frame = 0
        previous_count =0
        while True:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if (frame_count %  video_down_sample==0):
                # start_time = time()

                ret, frame = cap.read()
      
                if ret == True:
                    H, W, _ = frame.shape
                  
                    
                     
                    this_resize = cv2.resize(frame, ( image_size,  image_size), interpolation=cv2.INTER_AREA)
                    reshaped = np.transpose(this_resize, (2, 0, 1))


                    # if frame_count %  video_down_sample==0:
                    video_buffer[:, buffer_count, :, :] = reshaped
                        
                    
                   
                    if buffer_count >=  video_buff_size:
                        buffer_count = 0
                        Valid_video =True
                        break
            else:
                ret = cap.grab()
                # counter += 1
            if not ret:
                break
            frame_count += 1
            frame_number +=1

        
       

        
        for annotated_id in annotated_frame_ID:
    # Set the position of the video capture object to the original frame index
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotated_id)

            # Read the frame at the annotated frame index
            ret, frame = cap.read()

            if ret:
                H, W, _ = frame.shape
               

                this_resize = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                reshaped = np.transpose(this_resize, (2, 0, 1))

                # Calculate the corresponding index in the downsampled video buffer
                closest_frame_index = annotated_id * video_buff_size // total_frames
                closest_frame_index = min(closest_frame_index, video_buff_size - 1)

                # Replace the frame at the calculated index in the video buffer
                video_buffer[:, closest_frame_index, :, :] = reshaped
        cap.release()
        # return video_buffer, squeezed,Valid_video
        return video_buffer,Valid_video