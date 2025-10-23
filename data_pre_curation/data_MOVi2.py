import os
import cv2
import json
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm

# Configuration
MOVi_ROOT = "/data/MOVi"
DATASETS = ["MOVi-D", "MOVi-E"]
SPLITS = ["train", "validation", "test"]
IMG_SIZE = (224, 224)  # Target resolution
VIDEO_LEN = 24        # Frames per clip

def load_metadata(dataset_path):
    """Load MOVi metadata JSON file"""
    dataset_name = dataset_path.split('/')[-1]
    json_file = f"movi_{dataset_name.split('-')[-1].lower()}.json"
    json_path = os.path.join(dataset_path, json_file)
    
    with open(json_path, 'r') as f:
        return json.load(f)

def get_unique_categories(metadata):
    """Extract all unique categories from metadata"""
    all_categories = set()
    for video_data in metadata['videos'].values():
        for obj in video_data['objects']:
            all_categories.add(obj['category'])
    return sorted(list(all_categories))

def create_rgb_mask(mask, img_size):
    """Create RGB mask with instance IDs in red channel"""
    # Resize mask first
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask
    rgb_mask = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    rgb_mask[..., 0] = mask  # Instance IDs in red channel
    rgb_mask[..., 1] = 1     # Green channel set to 1 (as in YTVOS)
    return rgb_mask

def process_dataset():
    """Main processing function for MOVi datasets"""
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        dataset_path = os.path.join(MOVi_ROOT, dataset)
        # metadata = load_metadata(dataset_path)
        # all_categories = get_unique_categories(metadata)
        # category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
        # num_categories = len(all_categories)
        
        for split in SPLITS:
            print(f"  {split.capitalize()} split:")
            split_path = os.path.join(dataset_path, split)
            sequence_folders = sorted([d for d in os.listdir(split_path) 
                                      if os.path.isdir(os.path.join(split_path, d))])
            
            # Create output directory
            output_dir = os.path.join(dataset_path, "pkl", split)
            os.makedirs(output_dir, exist_ok=True)
            
            global_counter = 0
            for seq_folder in tqdm(sequence_folders, desc="Processing sequences"):
                seq_path = os.path.join(split_path, seq_folder)
                
                # Get sequence metadata
                # seq_metadata = metadata['videos'].get(seq_folder)
                # if not seq_metadata:
                #     print(f"Metadata not found for sequence {seq_folder}, skipping")
                #     continue
                
                # # Get objects and categories for this sequence
                # objects = seq_metadata['objects']
                # num_objects = len(objects)
                
                # Create one-hot vectors
                # one_hot_vectors = np.zeros((num_objects, num_categories), dtype=int)
                # for i, obj in enumerate(objects):
                #     cat_idx = category_to_idx.get(obj['category'])
                #     if cat_idx is not None:
                #         one_hot_vectors[i, cat_idx] = 1
                # merged_one_hot_vector = np.any(one_hot_vectors, axis=0).astype(int)
                
                # Get sorted frame files
                frame_files = sorted(glob(os.path.join(seq_path, "*.jpg")))
                frame_files = [f for f in frame_files if not f.endswith('_mask.png')]
                
                # Process frames in chunks of VIDEO_LEN
                for start_idx in range(0, len(frame_files), VIDEO_LEN):
                    end_idx = start_idx + VIDEO_LEN
                    if end_idx > len(frame_files):
                        break
                    
                    frames = []
                    masks = []
                    
                    for frame_idx in range(start_idx, end_idx):
                        img_path = frame_files[frame_idx]
                        mask_path = img_path.replace('.jpg', '_mask.png')
                        
                        # Read and resize image
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Failed to read image: {img_path}")
                            img = np.zeros((*IMG_SIZE[::-1], 3), dtype=np.uint8)
                        else:
                            img = cv2.resize(img, IMG_SIZE)
                        frames.append(img)
                        
                        # Read and process mask
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is None:
                            print(f"Failed to read mask: {mask_path}")
                            mask = np.zeros(IMG_SIZE[::-1], dtype=np.uint8)
                        rgb_mask = create_rgb_mask(mask, IMG_SIZE)
                        masks.append(rgb_mask)
                    
                    # Create data dictionary
                    frames_array = np.array(frames)  # Shape: (T, H, W, 3)
                    masks_array = np.array(masks)     # Shape: (T, H, W, 3)
                    
                    # Transpose to channel-first format
                    frames_array = frames_array.transpose(3, 0, 1, 2)  # (3, T, H, W)
                    masks_array = masks_array.transpose(3, 0, 1, 2)     # (3, T, H, W)
                    
                    # Create labels arrays
                    # labels_array = np.tile(merged_one_hot_vector, (VIDEO_LEN, 1))
                    # instance_array = np.tile(one_hot_vectors, (VIDEO_LEN, 1, 1))
                    
                    data_dict = {
                        'frames': frames_array.astype(np.uint8),
                        # 'labels': labels_array.astype(np.int32),
                        # 'instance_label': instance_array.astype(np.int32),
                        'masks': masks_array.astype(np.uint8)
                    }
                    
                    # Save as PKL
                    pkl_path = os.path.join(output_dir, f"clip_{global_counter:06d}.pkl")
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(data_dict, f)
                    
                    global_counter += 1
            
            print(f"Created {global_counter} PKL files for {dataset}/{split}")

if __name__ == '__main__':
    process_dataset()