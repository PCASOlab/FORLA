import os
import cv2
import numpy as np
import pickle

os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_pascal'  # Change this to your target mode
print("Current working directory:", os.getcwd())
from working_dir_root import PASCAL_root
img_size = 224
# Configuration
base_dir = os.path.join(PASCAL_root, 'VOCdevkit/VOC2012')
output_dir = os.path.join(PASCAL_root, 'pkl')
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

# Get all image IDs from JPEGImages
all_images = [f.split('.')[0] for f in os.listdir(os.path.join(base_dir, 'JPEGImages')) if f.endswith('.jpg')]

# Get official validation IDs
with open(os.path.join(base_dir, 'ImageSets/Segmentation/val.txt'), 'r') as f:
    val_ids = set(f.read().splitlines())

# Split into train/val
train_ids = [img_id for img_id in all_images if img_id not in val_ids]

def process_image(img_id, is_validation):
    print(str(img_id) + str(is_validation))
    """Process single image/mask pair and save as pickle"""
    try:
        # Load image with OpenCV
        img_path = os.path.join(base_dir, 'JPEGImages', f'{img_id}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR to RGB and resize
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        
        # Convert to CHW format and add dimension
        img_array = img.transpose(2, 0, 1)  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=1)  # Shape becomes [3, 1, 224, 224]
        img_array = img_array.astype(np.uint8)

        mask_array = None
        mask_path = os.path.join(base_dir, 'SegmentationObject', f'{img_id}.png')

        if is_validation or os.path.exists(mask_path):
            # Load mask with OpenCV
            mask = cv2.imread(mask_path)
            if mask is None:
                print("no mask")
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            # Convert BGR to RGB and resize
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            
            # Convert to CHW format and add dimension
            mask_array = mask.transpose(2, 0, 1)  # HWC to CHW
            mask_array = np.expand_dims(mask_array, axis=1)  # Shape becomes [3, 1, 224, 224]
            mask_array = mask_array.astype(np.uint8)

        # Create data dict
        data_dict = {
            'image': img_array,
            'mask': mask_array
        }

        # Save to appropriate folder
        folder = 'val' if is_validation else 'train'
        save_path = os.path.join(output_dir, folder, f'{img_id}.pkl')
        if (folder == "train" and img_array is not None) or (folder == "val" and img_array is not None and mask_array is not None):
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            
        return True
    
    except Exception as e:
        print(f"Skipped {img_id}: {str(e)}")
        return False

# Process datasets
val_success = sum(process_image(img_id, True) for img_id in val_ids)

# train_success = sum(process_image(img_id, False) for img_id in train_ids)

print(f"Processing complete! \nTrain: {train_success} images (masks included when available)"
      f"\nVal: {val_success} images (with color masks)")