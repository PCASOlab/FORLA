import cv2
import numpy as np
import json
import argparse
import os
from PIL import Image, ImageDraw
from files_io import save_a_image, save_a_pkl

video_format = ".mp4"



def load_parameters(task):
    with open('./src/data/parameters.json', 'r') as file:
        data = json.load(file)
    return data[task]


def save_decoded_images(decoded_data_dir, original, color_mask, onehot_mask,
                        frame_id, video_name, json_file_name):
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'original/', video_name + '_' +
                 frame_id + '.jpg', original)
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'color_mask/', video_name + '_' +
                 frame_id + '.jpg', color_mask)
    save_a_pkl(decoded_data_dir + json_file_name + '/' +
               'onehot_mask/', video_name + '_' +
               frame_id, onehot_mask)
    alpha = 0.5
    overlay = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0)
    original_mask_overlay = np.hstack((original, overlay))
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'original_plus_color_mask_overlay/',
                 video_name + '_' + frame_id +
                 '.jpg', original_mask_overlay)
    save_a_image(decoded_data_dir + json_file_name + '/' +
                 'color_mask_overlay/', video_name + '_'
                 + frame_id + '.jpg', overlay)


def create_mask(polygon_points, image_size=(100, 100),
                fill_color=255, outline_color=0):
    image = Image.new("L", image_size, outline_color)
    draw = ImageDraw.Draw(image)
    scaled_polygon = [(int(point['x'] * image_size[0]), int(point['y']
                       * image_size[1])) for point in polygon_points.values()]
    draw.polygon(scaled_polygon, fill=fill_color)
    return image


def get_specific_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    frame_number = int(frame_id)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        print(video_path)
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return None
    cap.release()
    print(f"Frame {frame_number} extracted successfully.")
    return frame


def decode_json(json_data, json_file_name, categories,
                category_colors, raw_video_dir, decoded_data_dir):
    decoded_data = json.loads(json_data)
    index = 0
    for frame_data in decoded_data:
        labels = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("labels", {})
        H = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("height", {})
        W = frame_data.get("data_units", {}).get(
            frame_data["data_hash"], {}).get("width", {})
        image_size = (W, H)
        image_sizeR = (H, W)
        video_full_name = frame_data['data_title']
        video_name = video_full_name
        this_full_video_path = raw_video_dir + video_name
        print(video_full_name)
        print(this_full_video_path)
        for key, value in labels.items():
            this_frame_id = key
            this_label = value
            category_masks = {category: np.zeros(
                image_sizeR, dtype=np.uint8) for category in categories}
            for i in range(len(this_label['objects'])):
                this_object = this_label['objects'][i]
                this_category_type = this_object['name']
                if this_category_type in categories:
                    this_polygon = this_object['polygon']
                    this_mask = create_mask(
                        this_polygon, image_size=image_size)
                    this_mask_array = np.array(this_mask)
                    category_masks[this_category_type] += this_mask_array
            color_image = np.zeros((H, W, 3), dtype=np.uint8)
            for category, mask in category_masks.items():
                color_image[mask > 0] = category_colors[category]
            this_image = get_specific_frame(
                this_full_video_path, this_frame_id)
            if this_image is not None and (len(this_label['objects']) > 0):
                padded_frame_id = this_frame_id.zfill(4)
                save_decoded_images(decoded_data_dir,
                                    this_image,
                                    color_image,
                                    category_masks,
                                    padded_frame_id,
                                    video_name,
                                    json_file_name)
            print(this_frame_id)
        index += 1
        print(index)


def main():
    # parser = argparse.ArgumentParser(
    #     description='Process Encord JSON files and videos.')
    # parser.add_argument('--task', type=str, required=True,
    #                     help='Task to determine which parameters to use')
    # args = parser.parse_args()
    # task = args.task
    # parameters = load_parameters(task)
    # jsonfile_dir = parameters['json_dir']
    jsonfile_dir = "/data/Cholec_new_selected/json_label/"
    
    # json_to_decode = parameters['json_files']
    json_to_decode =['labels']
    raw_video_dir = '/data/Cholec_new_selected/mp4/'
    decoded_data_dir = '/data/Cholec_new_selected/pkl/'
    categories = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6                  
    ]
    category_colors = {
    'Grasper': (0, 0, 255),        # Blue
    'Bipolar': (0, 255, 0),        # Green
    'Hook': (255, 0, 0),           # Red
    'Scissors': (255, 255, 0),     # Yellow
    'Clipper': (255, 0, 255),      # Magenta
    'Irrigator': (255, 165, 0),    # Orange
    'SpecimenBag': (128, 0, 128)   # Purple
    }
    json_to_decode = os.listdir(jsonfile_dir)


    for json_file_name in json_to_decode:
        with open(jsonfile_dir + json_file_name , 'r') as file:
            json_data = file.read()
        decode_json(json_data, json_file_name, categories,
                    category_colors, raw_video_dir, decoded_data_dir)


if __name__ == '__main__':
    main()