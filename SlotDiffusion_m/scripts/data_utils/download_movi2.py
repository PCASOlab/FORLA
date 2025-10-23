import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from torchvision import transforms
import torchvision.utils as vutils
import tensorflow_io as tfio
parser = argparse.ArgumentParser()

parser.add_argument('--out_path', default='./data/MOVi/')
parser.add_argument('--level', default='c')
parser.add_argument('--image_size', type=int, default=128)

args = parser.parse_args()

args.out_path = os.path.join(args.out_path, f'MOVi-{args.level.upper()}')
ds, ds_info = tfds.load(
    f"movi_{args.level}/{args.image_size}x{args.image_size}:1.0.0",
    # data_dir="C:/1projects/codes/Slot_Difussion/SlotDiffusion-main/SlotDiffusion-main/data/",
    data_dir="/data/MOVi/",
    with_info=True
)
# file_path = "gs://kubric-public/tfds/movi_e/128x128/1.0.0/features.json"
# with tf.io.gfile.GFile(file_path, 'r') as f:
#     ds = f.read()

to_tensor = transforms.ToTensor()


def save_one_split(split):
    b = 0
    all_paths = []
    data_iter = iter(tfds.as_numpy(ds[split]))
    for record in tqdm(data_iter):
        video = record['video']
        masks = record['segmentations']
        T, *_ = video.shape
        assert masks.shape[0] == T

        # setup dirs
        path_vid = os.path.join(args.out_path, split, f"{b:08}")
        os.makedirs(path_vid, exist_ok=True)

        for t in range(T):
            img = video[t]
            img = to_tensor(img)
            vutils.save_image(img, os.path.join(path_vid, f"{t:06}.jpg"))

            mask = masks[t, ..., 0].astype(np.uint8)
            cv2.imwrite(os.path.join(path_vid, f"{t:06}_mask.png"), mask)

        b += 1
        all_paths.append(os.path.dirname(path_vid))

    return all_paths

save_one_split('test')
save_one_split('validation')
save_one_split('train')
