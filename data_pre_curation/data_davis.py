import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
from visdom import Visdom
import cv2
import json
import pickle
from SAM.segment_anything import  SamPredictor, sam_model_registry

from working_dir_root import SAM_pretrain_root,sam_feature_OLG_dir3
import torch
import torch.nn as nn
import torch.nn.functional as F
output_folder_sam_feature = sam_feature_OLG_dir3
sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"
# self.inter_bz =1
model_type = "vit_h"
model_type = "vit_l"
model_type = "vit_b"

# model_type = "vit_t"
# sam_checkpoint = "./MobileSAM/weights/mobile_sam.pt"

# mobile SAM
Create_sam_feature = True
Update_PKL = False
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# self.predictor = SamPredictor(self.sam) 
Vit_encoder = sam.image_encoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Vit_encoder.to(device)
class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, root, task='unsupervised', subset='val', sequences='all', resolution='480p', codalab=False):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')
        self.img_size =(256,256)
        self.video_len=15
        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root, annotations_folder, resolution)
        year = '2019' if task == 'unsupervised' and (subset == 'test-dev' or subset == 'test-challenge') else '2017'
        self.imagesets_path = os.path.join(self.root, 'ImageSets', year)

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'DAVIS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found, download the missing subset '
                                    f'for the {self.task} task from {self.DATASET_WEB}')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder for the {self.task} task not found, download it from {self.DATASET_WEB}')
    def read_frames(self,video_frame_list, img_size):
        # frame_paths = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder))]
        frames = [cv2.cvtColor(  cv2.resize(cv2.imread(frame_path), img_size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB) for frame_path in video_frame_list]
        # frames = [cv2.cvtColor(   cv2.imread(frame_path) ,cv2.COLOR_BGR2RGB) for frame_path in video_frame_list]

        return frames
    def get_stacks(self,sequence):

        image_stack = self.read_frames(self.sequences[sequence]['images'],self.img_size)
        mask_stack = self.read_frames(self.sequences[sequence]['masks'],self.img_size)
        return image_stack, mask_stack
         
    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask
    

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq
def get_categories(seq, category_map):
    """Get the list of categories for a given sequence."""
    return list(set(category_map[seq].values()))
def convert_mask_to_onehot(mask, categories):
    """Convert a segmentation mask to a one-hot presence vector."""
    pass
def one_hot_vector(categories, selected_categories):
    vector = np.zeros(len(categories))
    for category in selected_categories:
        if category in categories:
            vector[categories[category] - 1] = 1
    return vector

def super_category_one_hot_vector(categories, super_categories, super_category_list, selected_categories):
    super_category_vector = np.zeros(len(super_category_list))
    for category in selected_categories:
        if category in super_categories:
            super_category = super_categories[category]
            index = super_category_list.index(super_category)
            super_category_vector[index] = 1
    return super_category_vector
def get_frame_categories(frame_unique_colors, video_unique_color, seq, category_map):
    categories = []
    for color in frame_unique_colors:
        category_id = get_category_id(color, video_unique_color)
        category_id = int(np.clip(category_id,0,len(category_map[seq])))
        if category_id>0:
            # if category_id>len(category_map[seq]):
                
            category = category_map[seq][str(category_id)]
            categories.append(category)
    return categories

def get_category_id(color, video_unique_color):
    for idx, unique_color in enumerate(video_unique_color):
        if np.array_equal(color, unique_color):
            return idx
    return None
def get_unique_colors(masks):
    mask_reshaped = masks.reshape(-1, 3)
    unique_colors, counts = np.unique(mask_reshaped, axis=0, return_counts=True)
    # sorted_indices = np.lexsort(([unique_colors[:, i] for i in range(unique_colors.shape[1] - 1, -1, -1)]))
    # unique_colors_sorted = unique_colors[sorted_indices]
    video_unique_color = unique_colors[np.lexsort(unique_colors.T[[0, 1,2]])]
    return video_unique_color
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    root_davis = "/media/guiqiu/Weakly_supervised_data/DAVIS/"
    output_folder_pkl = root_davis + "pkl/"

    Json_map_dir = root_davis + 'Usupervised/DAVIS/davis_semantics.json'
    with open(Json_map_dir, 'r') as f:
        category_map = json.load(f)
    Categories_json = root_davis +  'Usupervised/DAVIS/categories.json'
    with open(Categories_json, 'r') as f:
        category_list_json= json.load(f)
    categories_list = {key: value['id'] for key, value in category_list_json.items()}
    super_categories = {key: value['super_category'] for key, value in category_list_json.items()}
    super_category_list = list(set(super_categories.values()))
    super_category_list.sort()
    print("Categories:", categories_list)
    print("Super Categories:", super_category_list)
    only_first_frame = True
    subsets = ['train', 'val']
    subsets = ['train']

    viz = Visdom(port=8097)
    file_counter =0
    for s in subsets:
        dataset = DAVIS(root=root_davis+ 'Usupervised/DAVIS', subset=s)
        print("subsets"+s)
        for seq in dataset.get_sequences(): # seq= "bear" or "bike-packing" or other name of a sequence 
            # seq="longboard"#test

            # g = dataset.get_frames(seq)
            images, masks = dataset.get_stacks(seq)
            print(seq)
            video_categories = get_categories(seq, category_map)
            print(video_categories)
            
            all_data = []
            video_unique_color =get_unique_colors(np.array(masks)) 
            for img, msk in zip(images, masks):
                # Organize as a dictionary or structure
                mask_reshaped = msk.reshape(-1, 3)
  # Find unique elements along axis 0 (considering each row as a color)
                frame_unique_colors = np.unique(mask_reshaped, axis=0)
                frame_categories = get_frame_categories(frame_unique_colors,video_unique_color,seq, category_map)
                category_vector = one_hot_vector(categories_list, frame_categories)
                super_category_vector = super_category_one_hot_vector(categories_list, super_categories, super_category_list, frame_categories)
                print("Category One-Hot Vector:", category_vector)
                print("Super Category One-Hot Vector:", super_category_vector)
                data_pair = {'frame': img, 'label': category_vector,'super_label': super_category_vector, 'mask':msk}
                all_data.append(data_pair)

                # Check if buffer is not empty and has reached the desired length
                if len(all_data) > 0 and len(all_data) == dataset.video_len:
                    # Convert list of dictionaries to a dictionary of arrays
                    data_dict = {'frames': np.array([pair['frame'] for pair in all_data]),
                                    'labels': np.array([pair['label'] for pair in all_data]),
                                    'super_labels': np.array([pair['super_label'] for pair in all_data]) ,
                                     'masks': np.array([pair['mask'] for pair in all_data])  }

                    # Perform "or" operation to merge labels
                    # merged_labels = merge_labels(data_dict['labels'])

                    # Reshape arrays
                    data_dict['frames'] = np.transpose(data_dict['frames'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)
                    data_dict['masks'] = np.transpose(data_dict['masks'], (3, 0, 1, 2))  # Reshape to (3, 29, 256, 256)

                    # data_dict['labels'] = np.transpose(data_dict['labels'], (1, 0, 2, 3))  # Reshape to (10 29, 256, 256)
                
                    pkl_file_name = f"clip_{file_counter:06d}.pkl"
                    pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)
                    if Update_PKL:
                        with open(pkl_file_path, 'wb') as file:
                            pickle.dump(data_dict, file)
                            print("Pkl file created:" +pkl_file_name)
                    if Create_sam_feature == True:
                        this_video_buff = data_dict['frames'] 
                        video_buff_GPU = torch.from_numpy(np.float32(this_video_buff)).to (device)
                        video_buff_GPU = video_buff_GPU.permute(1,0,2,3) # Reshape to (29, 3, 64, 64)
                        input_resample =   F.interpolate(video_buff_GPU,  size=(1024,  1024), mode='bilinear', align_corners=False)
                        
                        bz,  ch, H, W = input_resample.size()
                        predicted_tensors =[]
                        with torch.no_grad():

                            for i in range(bz):
                                
                                input_chunk = (input_resample[i:i+1] -124.0)/60.0
                                output_chunk = Vit_encoder(input_chunk)
                                predicted_tensors.append(output_chunk)
                            
                            # Concatenate predicted tensors along batch dimension
                            concatenated_tensor = torch.cat(predicted_tensors, dim=0)
                            
                        
                        features = concatenated_tensor.half()
                        sam_pkl_file_name = f"clip_{file_counter:06d}.pkl"
                        sam_pkl_file_path = os.path.join(output_folder_sam_feature, sam_pkl_file_name)

                        with open(sam_pkl_file_path, 'wb') as file:
                            pickle.dump(features, file)
                            print("sam Pkl file created:" +sam_pkl_file_name)
                        
                    file_counter += 1

                    # Clear data for the next batch
                    all_data = []
                    # img, mask = next(g)
                    # Send images to Visdom
                    # viz.image(np.transpose(mask, (2, 0, 1)), opts=dict(title=f'{seq} - Mask'))
                    
                    # plt.subplot(2, 1, 1)
                    # plt.title(seq)
                    # plt.imshow(img)
                    # plt.subplot(2, 1, 2)
                    # plt.imshow(mask)
                    # plt.show(block=True)
    print("Total files created:", file_counter)

