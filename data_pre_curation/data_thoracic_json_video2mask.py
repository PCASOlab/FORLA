import cv2
# import os
import  numpy as np
# from working_dir_root import Dataset_video_root, Dataset_label_root
# import csv
import re
import json
from PIL import Image, ImageDraw
from dataset.io import save_a_image
from dataset.io import save_a_pkl_w_create as save_a_pkl
Video_format =  ".mp4"
Video_format =  ".mpg"


Jsonfile_dir = 'C:/2data/Raw_data_Chrocic/raw/label/' # the folder for all jsons
Json_to_decode = ['labelsB','labelsA'] 
Raw_video_dir = 'C:/2data/Raw_data_Chrocic/raw/video#7/' # folder for all videos folders with number
Decoded_data_dir= 'C:/2data/Raw_data_Chrocic//data/interim/' # output folder
Using_normalized = False
# json_data = file.read()
categories = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
]
category_colors = {
    'Lymph node': (0, 0, 255),        # Red
    'Vagus nereve': (0, 255, 0),      # Green
    'Bronchus': (255, 0, 0),          # Blue
    'Lung parenchyma': (255, 255, 0),  # Yellow
    'Instruments': (255, 0, 255),      # Magenta
}
def save_decoded_images (Decoded_data_dir,original,color_mask,onehot_mask,frame_id,video_name,json_file_name):
    #original
    save_a_image (Decoded_data_dir+json_file_name + '/' +'original/',video_name+'_'+frame_id+'.jpg',original)
    #color_mask
    save_a_image (Decoded_data_dir+json_file_name + '/' +'color_mask/',video_name+'_'+frame_id+'.jpg',color_mask)
    #pkl_onehot mask
    save_a_pkl (Decoded_data_dir+json_file_name + '/' +'onehot_mask/',video_name+'_'+frame_id ,onehot_mask)

    # Create the overlay image
    alpha= 0.5
    overlay = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0)
    original_mask_overlay = np.hstack((original, overlay))
    save_a_image (Decoded_data_dir+json_file_name + '/' +'original_plus_color_mask_overlay/',video_name+'_'+frame_id+'.jpg',original_mask_overlay)

    #color_mask
    save_a_image (Decoded_data_dir+json_file_name + '/' +'color_mask_overlay/',video_name+'_'+frame_id+'.jpg',overlay)

    pass
def create_mask(polygon_points, image_size=(100, 100), fill_color=255, outline_color=0):
    image = Image.new("L", image_size, outline_color)
    draw = ImageDraw.Draw(image)

    # Scale polygon points to image size
    scaled_polygon = [(int(point['x'] * image_size[0]), int(point['y'] * image_size[1])) for point in polygon_points.values()]

    # Draw filled polygon on the image
    draw.polygon(scaled_polygon, fill=fill_color)

    return image
def get_specific_frame(video_path,frame_id):
     # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_number = int(frame_id)
    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
        exit()

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set the frame number you want to extract (41st frame in this case)
    # frame_number = 41
    # Set the capture object's position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number )

    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        exit()

    # Display or process the frame as needed
    # For example, you can save the frame to an image file
    # cv2.imwrite(f"frame_{frame_number}.png", frame)

    # Release the capture object
    cap.release()

    print(f"Frame {frame_number} extracted successfully.")
    return frame
     
def decode_json(json_data,json_file_name, categories):
    decoded_data = json.loads(json_data)
    index = 0
    for frame_data in decoded_data:
        labels = frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("labels", {})
        H = frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("height", {})
        W= frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("width", {})
        image_size = (W, H)
        image_sizeR = (H, W)

        video_full_name = frame_data['data_title']
        underscore_index = video_full_name.find("_")

# Extract the substring before the underscore
        video_name = video_full_name[:underscore_index] if underscore_index != -1 else video_full_name
        numeric_values = [int(word) for word in video_name.split() if re.match(r'\d+', word)]
        # if video_name == "Kazu RATS RUL #7 88 time9":
        #     error = True
        # Get the first numeric value (if any)
        numeric_value = numeric_values[0] if numeric_values else None

        video_name_format = video_name + Video_format
        if Using_normalized == True:
            video_name_format = video_name + '_normalized'+Video_format

        this_full_video_path = Raw_video_dir+str(numeric_value) +"/" + video_name_format
        print(video_full_name)
        for key, value in labels.items():
            this_frame_id = key
            this_label = value
            category_masks = {category: np.zeros(image_sizeR, dtype=np.uint8) for category in categories}

            for i in range(len(this_label['objects'])):
                this_object = this_label['objects'][i]
                this_category_type = this_object['name']
                # this_polygon = this_object['polygon']
                # this_mask =  create_mask(this_polygon, image_size=image_size)
                # this_mask_array = np.array(this_mask)
                if this_category_type in categories:
                    this_polygon = this_object['polygon']
                    this_mask = create_mask(this_polygon, image_size=image_size)
                    this_mask_array = np.array(this_mask)

                    # Accumulate the mask in the corresponding channel
                    category_index = categories.index(this_category_type)
                    category_masks[this_category_type] += this_mask_array
                # this_mask.show()
            color_image = np.zeros((H, W, 3), dtype=np.uint8)

        # Fill the color image with the corresponding colors based on the masks
            for category, mask in category_masks.items():
                color_image[mask > 0] = category_colors[category]
            this_image=get_specific_frame(this_full_video_path,this_frame_id)
            # Display the color-coded image using OpenCV
            if this_image is not None and (len(this_label['objects'])>0):
                            
                # save_decoded_images()
                padded_frame_id = this_frame_id.zfill(4)
                save_decoded_images (Decoded_data_dir,this_image,color_image,category_masks, padded_frame_id,video_name,json_file_name)
                # cv2.imshow('Color Mask', color_image)
                # cv2.imshow('Color Image', this_image)
                # cv2.waitKey(10)
            print(this_frame_id)
            pass
            # cv2.destroyAllWindows()
        index +=1
        print(index)
def main():
     for json_file_name in Json_to_decode:
        
        
        with open(Jsonfile_dir + json_file_name + '.json', 'r') as file:
                json_data = file.read()
        # the list of labels
        # Define the path to your CSV file
        #############3 read all the labels##########################
        decode_json(json_data,json_file_name, categories)
if __name__ == '__main__':
   main()