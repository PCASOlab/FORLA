import os
import pickle
import numpy as np
import cv2
from glob import glob

# ==== CONFIG ====
img_size = (224, 224)
video_len = 10

data_roots = {
    'MOVi-D': '/data/MOVi/MOVi-D',
    'MOVi-E': '/data/MOVi/MOVi-E'
}
output_root = '/data/MOVi/pkls'

# ==== MASK ENCODING ====
def encode_mask(mask_img, instance_id=1):
    mask_img = cv2.resize(mask_img, img_size, interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_img > 160)
    h, w = img_size
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_mask[..., 1] = 1  # Green channel fixed to 1
    rgb_mask[binary_mask] = [instance_id, 1, 0]  # Red = instance ID
    return rgb_mask

# ==== READ FRAMES & MASKS ====
def read_sequence_folder(folder_path):
    frames = []
    masks = []
    jpg_files = sorted(glob(os.path.join(folder_path, '*.jpg')))
    frame_files = [f for f in jpg_files if not f.endswith('_mask.jpg')]

    for frame_path in frame_files:
        base_name = os.path.splitext(os.path.basename(frame_path))[0]
        mask_path = os.path.join(folder_path, f"{base_name}_mask.png")

        # Read image
        img = cv2.imread(frame_path)
        img = cv2.resize(img, img_size)

        # Read and encode mask
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        rgb_mask = encode_mask(mask_img)

        frames.append(img)
        masks.append(rgb_mask)

    return frames, masks

# ==== CONVERT FUNCTION ====
def convert_dataset(dataset_name, root_path):
    subsets = ['train', 'validation', 'test']
    for split in subsets:
        split_path = os.path.join(root_path, split)
        if not os.path.exists(split_path):
            print(f"Skipping {dataset_name}/{split} (folder missing)")
            continue

        output_dir = os.path.join(output_root, dataset_name, split)
        os.makedirs(output_dir, exist_ok=True)

        folders = sorted(os.listdir(split_path))
        file_counter = 0
        for seq_folder in folders:
            seq_path = os.path.join(split_path, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            frames, masks = read_sequence_folder(seq_path)
            if len(frames) < video_len:
                continue  # skip short sequences

            all_data = []
            for i in range(video_len):
                data_pair = {
                    'frame': frames[i],
                    'label': np.zeros(10),  # Dummy labels
                    'instance_label': np.zeros((1, 10)),
                    'mask': masks[i]
                }
                all_data.append(data_pair)

            data_dict = {
                'frames': np.transpose(np.array([x['frame'] for x in all_data]), (3, 0, 1, 2)),
                'labels': np.array([x['label'] for x in all_data]),
                'instance_label': np.array([x['instance_label'] for x in all_data]),
                'masks': np.transpose(np.array([x['mask'] for x in all_data]), (3, 0, 1, 2))
            }

            pkl_name = f"{dataset_name.lower()}_{split}_clip_{file_counter:06d}.pkl"
            pkl_path = os.path.join(output_dir, pkl_name)
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_dict, f)
                print(f"Saved: {pkl_path}")

            file_counter += 1

        print(f"Total PKLs for {dataset_name}/{split}: {file_counter}")

# ==== EXECUTE ====
if __name__ == '__main__':
    for dataset_name, dataset_path in data_roots.items():
        convert_dataset(dataset_name, dataset_path)
