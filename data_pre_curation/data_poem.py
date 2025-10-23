import os
import cv2
import numpy as np
import pickle
import random
from tqdm import tqdm

# ================== CONFIGURATION ==================
base_dir = "/data/POEM/data/Frames"
output_dir = "/data/POEM/data/pkl"
img_size = 224
train_ratio = 0.8  # 80% training, 20% validation
crop_threshold = 20  # Intensity threshold for blank area detection

# Visualization settings
vis_display_flag = True  # Set to True to enable visualization
vis_display_sample = 100 # Display every Nth image (to avoid flooding)
# ===================================================

# Create output directories
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

# Setup Visdom if enabled
vis = None
if vis_display_flag:
    try:
        from visdom import Visdom
        vis = Visdom(env='image_processing')
        print("Visdom visualization enabled")
    except ImportError:
        print("Visdom not installed. Visualization disabled.")
        vis_display_flag = False
    except ConnectionError:
        print("Visdom server not running. Visualization disabled.")
        vis_display_flag = False

# ================== BIDIRECTIONAL CROPPING FUNCTIONS ==================
def get_crop_boundaries(gray_image, threshold=crop_threshold):
    """
    Detect crop boundaries in both horizontal and vertical directions
    Returns: crop_start (x1, y1), crop_end (x2, y2)
    """
    H, W = gray_image.shape
    
    # 1. HORIZONTAL CROPPING (left/right)
    # Select 3 horizontal lines near the vertical center
    num_h_lines = 3
    v_center = H // 2
    h_lines = [v_center - 1, v_center, v_center + 1]
    
    # Sum pixel values along horizontal lines
    h_sum = np.zeros(W, dtype=np.int32)
    for line in h_lines:
        if 0 <= line < H:
            h_sum += gray_image[line]
    

    # Find horizontal boundaries
    h_nonzero = np.where(h_sum > threshold)[0]
    if h_nonzero.size > 0:
        left = h_nonzero[0]
        right = h_nonzero[-1] + 1  # inclusive end
    else:
        left, right = 0, W
    
    # 2. VERTICAL CROPPING (top/bottom)
    # Select 3 vertical lines near the horizontal center
    num_v_lines = 3
    h_center = (left + right) // 2
    v_lines = [h_center - 1, h_center, h_center + 1]
    
    # Sum pixel values along vertical lines
    v_sum = np.zeros(H, dtype=np.int32)
    for col in v_lines:
        if left <= col < right:
            v_sum += gray_image[:, col]
    
    # Find vertical boundaries
    v_nonzero = np.where(v_sum > threshold)[0]
    if v_nonzero.size > 0:
        top = v_nonzero[0]
        bottom = v_nonzero[-1] + 1  # inclusive end
    else:
        top, bottom = 0, H
    
    # Ensure boundaries are within image dimensions
    #additional aggressive cropping
    w_margin = int(W * 0.08)  # 5% margin
    h_margin = int(H * 0.08)  # 5% margin

    left = max(0, left+ w_margin)
    right = min(W, right- w_margin)
    top = max(0, top+ h_margin)
    bottom = min(H, bottom- h_margin)
    
    return (left, top), (right, bottom)

def crop_image(image, crop_start, crop_end):
    """Crop image using detected boundaries"""
    x1, y1 = crop_start
    x2, y2 = crop_end
    return image[y1:y2, x1:x2, :]
# ========================================================

# Collect all image paths
image_paths = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images")

# Shuffle and split into train/val
random.shuffle(image_paths)
split_idx = int(len(image_paths) * train_ratio)
train_paths = image_paths[:split_idx]
val_paths = image_paths[split_idx:]

print(f"Train images: {len(train_paths)}, Val images: {len(val_paths)}")

# Initialize processing counter
processing_counter = 0

def process_image(img_path, is_validation):
    """Process and save single image with bidirectional cropping"""
    global processing_counter
    processing_counter += 1
    
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load: {img_path}")
        
        # Store original for visualization
        original_img = img.copy()
        
        # Convert to grayscale for cropping detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get crop boundaries (both horizontal and vertical)
        crop_start, crop_end = get_crop_boundaries(gray)
        
        # Apply cropping
        cropped_img = crop_image(img, crop_start, crop_end)
        
        # Resize and convert to RGB
        resized_img = cv2.resize(cropped_img, (img_size, img_size))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # Format for PyTorch (CHW with batch dimension)
        img_array = rgb_img.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=1)  # Add dimension [3, 1, 224, 224]
        img_array = img_array.astype(np.uint8)
        
        # Create data dictionary
        data_dict = {
            'image': img_array,
            'mask': None  # No masks in this dataset
        }
        
        # Generate unique filename based on relative path
        rel_path = os.path.relpath(img_path, base_dir)
        safe_name = rel_path.replace(os.sep, "_").replace(".", "_") + ".pkl"
        
        # Save to appropriate folder
        folder = 'val' if is_validation else 'train'
        save_path = os.path.join(output_dir, folder, safe_name)
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        # Visualization with Visdom
        if vis_display_flag and processing_counter % vis_display_sample == 0:
            # Create visualization images
            orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            # Get filename for title
            filename = os.path.basename(img_path)
            
            # Draw crop boundaries on original image
            annotated_img = original_img.copy()
            x1, y1 = crop_start
            x2, y2 = crop_end
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Visualize original image with crop boundaries
            # vis.image(
            #     cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1),
            #     opts=dict(title=f'{filename} - Crop Area', caption=f'X:{x1}-{x2}, Y:{y1}-{y2}'),
            #     win=f'crop_area_{processing_counter}'
            # )
            
            # # Visualize cropped image
            # vis.image(
            #     cropped_rgb.transpose(2, 0, 1),
            #     opts=dict(title=f'{filename} - Cropped', caption=f'Size: {cropped_img.shape[1]}x{cropped_img.shape[0]}'),
            #     win=f'cropped_{processing_counter}'
            # )
            
            # Visualize resized image
            vis.image(
                img_array.squeeze(1),  # Remove extra dimension [3, 224, 224]
                opts=dict(title=f'{filename} - Resized', caption=f'Size: {img_size}x{img_size}'),
                win=f'resized_{processing_counter}'
            )
            
            print(f"Displayed visualization for {filename}")
            
        return True
    
    except Exception as e:
        print(f"\nSkipped {img_path}: {str(e)}")
        return False

# Process datasets with progress bars
print("\nProcessing validation set:")
val_success = 0
for path in tqdm(val_paths):
    val_success += 1 if process_image(path, True) else 0

print("\nProcessing training set:")
train_success = 0
for path in tqdm(train_paths):
    train_success += 1 if process_image(path, False) else 0

print(f"\nProcessing complete! "
      f"\nTrain: {train_success}/{len(train_paths)} images"
      f"\nVal: {val_success}/{len(val_paths)} images")