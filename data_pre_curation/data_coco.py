import os
import cv2
from PIL import Image
import numpy as np
import pickle
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import defaultdict

os.environ['WORKING_DIR_IMPORT_MODE'] = 'train_coco'
print("Current working directory:", os.getcwd())
from working_dir_root import COCO_root

# Configuration
base_dir = COCO_root
train_images_dir = os.path.join(base_dir, 'train2017')
val_images_dir = os.path.join(base_dir, 'val2017')
annotations_dir = os.path.join(base_dir, 'annotations_trainval2017', 'annotations')
output_dir = os.path.join(COCO_root, 'pkl/')

# Create output directories
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
def decode_instance_mask(mask_tensor):
    """
    Convert RGB mask tensor to class/instance IDs
    Input shape: [batch_size, 3, height, width]
    Returns: (class_ids, instance_ids)
    """
    # Squeeze batch dimension if needed
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor.squeeze(0)
    
    # Extract channels
    class_ids = mask_tensor[0]    # Red channel
    instance_ids = mask_tensor[1] # Green channel
    
    return class_ids, instance_ids
def process_coco_split(annotation_file, images_dir, split_name):
    """Process a COCO split with instance-aware color encoding"""
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    
    for img_id in tqdm(img_ids, desc=f'Processing {split_name}'):
        try:
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(images_dir, img_info['file_name'])
            
            # Load and process image
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Convert to CHW format and add dimension
            img_array = img.transpose(2, 0, 1)[:, np.newaxis, :, :]  # Shape [3, 1, 224, 224]
            img_array = img_array.astype(np.uint8)

            
            # Initialize mask_array as None
            mask_array = None
            
            # Process annotations
            ann_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(ann_ids)
            
            if annotations:
                original_mask = np.zeros((img_info['height'], img_info['width'], 3), dtype=np.uint8)
                class_counts = defaultdict(int)
                
                # Process each instance with unique encoding
                for ann in annotations:
                    class_id = ann['category_id']
                    class_counts[class_id] += 1
                    instance_num = class_counts[class_id]
                    if instance_num > 1:
                        print("more than 1")
                    # Encode class and instance into RGB
                    color = (
                        class_id % 256,           # Red channel: class ID
                        instance_num % 256,       # Green channel: instance number
                        0                          # Blue channel: unused
                    )
                    
                    # Apply instance mask
                    ann_mask = coco.annToMask(ann)
                    original_mask[ann_mask == 1] = color

                # Resize and format mask
                # Resize and format mask with OpenCV
                mask = cv2.resize(original_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                
                # Convert to CHW format and add dimension
                mask_array = mask.transpose(2, 0, 1)[:, np.newaxis, :, :]  # Shape [3, 1, 224, 224]
                mask_array = mask_array.astype(np.uint8)

            # Create data dict
            data_dict = {
                'image': img_array,
                'mask': mask_array
            }

            # Save to appropriate folder
            save_path = os.path.join(output_dir, split_name, f"{img_id}.pkl")
            if (split_name == "train" and img_array is not None) or (split_name == "val" and img_array is not None and mask_array is not None):
                with open(save_path, 'wb') as f:
                    pickle.dump(data_dict, f)
                
        except Exception as e:
            print(f"Skipped {img_id}: {str(e)}")


# Process validation data
process_coco_split(
    annotation_file=os.path.join(annotations_dir, 'instances_val2017.json'),
    images_dir=val_images_dir,
    split_name='val'
)

# Process training data
process_coco_split(
    annotation_file=os.path.join(annotations_dir, 'instances_train2017.json'),
    images_dir=train_images_dir,
    split_name='train'
)

print("Processing complete!")