import os
import pickle
import cv2
import numpy as np

# Paths
pkl_folder = "/media/guiqiu/Weakly_supervised_data/MICCAI_selected_GT/pkl"  # Folder containing original PKL files
video_folder = "/media/guiqiu/Weakly_supervised_data/MICCAI/selected"  # Folder containing original videos
output_pkl_folder = "/media/guiqiu/Weakly_supervised_data/MICCAI_selected_GT/pkl30fps"  # New folder for updated PKLs

# Ensure output folder exists
os.makedirs(output_pkl_folder, exist_ok=True)

FPS_TARGET = 30  # Target frame rate

def load_pkl(pkl_path):
    """Load PKL file and return video frames and masks."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data['frames'], data['labels']


def extend_video_and_masks(video_name, video_frames, mask_frames):
    """Extend video stack to 30 FPS and align masks correctly."""
    C_video, num_original_frames, H, W = video_frames.shape
    C_mask, num_mask_frames, _, _ = mask_frames.shape

    # Load the original video
    video_path = os.path.join(video_folder, f"{video_name}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None

    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize lists to store frames and masks
    frame_list = []
    mask_list = []

    # Read frames from video and store them
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 2 == 0:  # Downsample by selecting every 2nd frame
            H_i, _, _ = frame.shape

                # Apply the cropping
            cropped_frame = frame[56:H_i-80, 192:1088]  # Cropping the frame
            resized_frame = cv2.resize(cropped_frame, (W, H))
            frame_list.append(np.transpose(resized_frame, (2, 0, 1)))  # Convert HWC -> CHW
        frame_idx += 1
    cap.release()

    total_frames = len(frame_list)  # Number of frames actually read from video

    # Directly assign to extended_video_stack
    extended_video_stack = np.array(frame_list).transpose(1, 0, 2, 3)  # (N, C, H, W) -> (C, N, H, W)

    # Initialize extended_mask_stack based on the shape of extended_video_stack
    extended_mask_stack = np.full((C_mask, total_frames, H, W), np.nan, dtype=np.float32)

    # Map masks from 60 FPS to 30 FPS and align them
    mask_indices_60fps = np.arange(0, num_mask_frames * 60, 60)
    mask_indices_30fps = mask_indices_60fps // 2  # Convert to 30 FPS index

    for mask_idx_30fps, mask_idx_60fps in zip(mask_indices_30fps, range(num_mask_frames)):
        if mask_idx_30fps < total_frames:
            extended_mask_stack[:, mask_idx_30fps] = mask_frames[:, mask_idx_60fps]

    return extended_video_stack, extended_mask_stack
# Process all PKLs
for pkl_file in os.listdir(pkl_folder):
    if pkl_file.endswith(".pkl"):
        pkl_path = os.path.join(pkl_folder, pkl_file)
        video_name = os.path.splitext(pkl_file)[0]  # Extract video name
        
        print(f"Processing {video_name}...")

        # Load original PKL data
        video_frames, mask_frames = load_pkl(pkl_path)

        # Extend video and masks
        updated_video, updated_masks = extend_video_and_masks(video_name, video_frames, mask_frames)
        if updated_video is None or updated_masks is None:
            continue  # Skip if video failed to load

        # Save updated PKL
        updated_pkl_path = os.path.join(output_pkl_folder, pkl_file)
        with open(updated_pkl_path, "wb") as f:
            pickle.dump({'frames': updated_video, 'labels': updated_masks}, f)
        
        print(f"Updated PKL saved: {updated_pkl_path}")

print("All PKLs updated successfully.")
