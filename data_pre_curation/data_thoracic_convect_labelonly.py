import cv2
import os
import  numpy as np
# from working_dir_root import Dataset_video_root, Dataset_label_root
# import csv
import re
import json
from PIL import Image, ImageDraw
from dataset.io import save_a_image
from dataset.io import save_a_pkl_w_create as save_a_pkl
import pickle
from working_dir_root import working_pcaso_raid
Crop_flag = True
Video_format =  ".mp4"
from visdom import Visdom
viz = Visdom(port=8097)

# Video_format =  ".mpg"


Jsonfile_dir = working_pcaso_raid +'Thoracic/label/' # the folder for all jsons
Json_to_decode = ['labelsB','labelsA'] 
Raw_video_dir = working_pcaso_raid +'Thoracic/video#7/' # folder for all videos folders with number
Decoded_data_dir= working_pcaso_raid +'Thoracic/interim/' # output folder
output_folder_pkl = working_pcaso_raid +'Thoracic/pkl/'
output_folder_sam_feature = working_pcaso_raid +'Thoracic/output_sam_features/'
Using_normalized = True
video_buff_size = 29 
image_resize = 256

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
# Function to save sampled clips and masks as pickle files

def find_symmetric_bounding_box(image, threshold=20):
     
    H, W, _ = image.shape
    # Convert the image to grayscale to simplify the analysis
    grayscale_image = np.mean(image, axis=2).astype(np.uint8)
    # Create vertical and horizontal projections by averaging pixel intensities along each axis
    vertical_projection = np.mean(grayscale_image, axis=1)  # Average each row
    horizontal_projection = np.mean(grayscale_image, axis=0)  # Average each column
    # For robustness, we focus on the top half and left half for calculating H1 and W1, assuming symmetry
    H_half = H // 2
    W_half = W // 2
    # Calculate H1 (top border) by analyzing the top half
    H1 = np.argmax(vertical_projection[:H_half] > threshold)
    # Since the border is symmetric, H2 can be calculated as:
    H2 = H - H1
    # Calculate W1 (left border) by analyzing the left half
    W1 = np.argmax(horizontal_projection[:W_half] > threshold)
    # Similarly, W2 can be calculated as:
    W2 = W - W1
    return [H1, H2, W1, W2]
def save_sampled_clip_and_masks(pkl_file_name, video_images, video_masks, output_folder_pl):
    pkl_file_name = pkl_file_name + '.pkl'
    data_dict = {'frames': video_images, 'labels': video_masks}
    # pkl_file_name = f"clip_{file_counter:06d}.pkl"
    pkl_file_path = os.path.join(output_folder_pkl, pkl_file_name)
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data_dict, file)
        print("Pkl file created:", pkl_file_name)
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
def show_images_to_visdom (original,color_mask):
    alpha= 0.5
    overlay = cv2.addWeighted(original, 1 - alpha, color_mask, alpha, 0)
    original_mask_overlay = np.hstack((original, overlay))
    viz.image(np.transpose(original_mask_overlay, (2, 0, 1)), opts=dict(title=f'{1} - Mask'))
def create_mask(polygon_points, image_size=(100, 100), fill_color=255, outline_color=0):
    image = Image.new("L", image_size, outline_color)
    draw = ImageDraw.Draw(image)

    # Scale polygon points to image size
    scaled_polygon = [(int(point['x'] * image_size[0]), int(point['y'] * image_size[1])) for point in polygon_points.values()]

    # Draw filled polygon on the image
    draw.polygon(scaled_polygon, fill=fill_color)

    return image
def resize_mask(mask, target_size):
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

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

def crop_and_resize_mask(mask, bbox, original_shape, target_size=(256, 256)):
  
    H, W = original_shape
    target_H, target_W = target_size

    # Calculate scaling factors for height and width
    scale_h = target_H / H
    scale_w = target_W / W

    # Scale the bounding box coordinates
    h1, h2, w1, w2 = bbox
    h1, h2 = int(h1 * scale_h), int(h2 * scale_h)
    w1, w2 = int(w1 * scale_w), int(w2 * scale_w)

    # Crop the mask using the scaled bounding box
    cropped_mask = mask[:, h1:h2, w1:w2].astype(np.uint8)  # Convert to uint8 for resizing

    # Resize each channel of the cropped mask to the target size
    resized_mask = np.array([
        cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
        for m in cropped_mask
    ])

    # Convert back to boolean
    resized_mask = resized_mask.astype(bool)

    return resized_mask
def load_a_video_buffer(video_path, video_buff_size, image_size, annotated_frame_ID, annotation_masks,annotated_frames,bounding_coord, Display_loading_video):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_down_sample = int((total_frames - 1) / video_buff_size)

    frame_count = 0
    buffer_count = 0
    video_buffer = np.zeros((3, video_buff_size, image_size, image_size))
    annotation_masks_full_buffer = np.ones((len(categories), video_buff_size, image_size, image_size))
    annotation_masks_full_buffer = (annotation_masks_full_buffer*np.nan) # assign all value to -1 first
    Valid_video = False
    if total_frames == 0:
        return video_buffer,annotation_masks_full_buffer, Valid_video

    while True:
        if frame_count % video_down_sample == 0:
            ret, frame = cap.read()

            if ret == True:
                H, W, _ = frame.shape
                if bounding_coord is not None:
                    [h1,h2,w1,w2]= bounding_coord
                    frame = frame[h1:h2,w1:w2]
                this_resize = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                reshaped = np.transpose(this_resize, (2, 0, 1))
                video_buffer[:, buffer_count, :, :] = reshaped
                buffer_count += 1

            if buffer_count >= video_buff_size:
                Valid_video = True
                break
        else:
            ret = cap.grab()

        if not ret:
            break

        frame_count += 1

    for k in range(len(annotated_frame_ID)):
        annotated_id = int(annotated_frame_ID[k])
        annotated_frame = annotated_frames[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotated_id)
        ret, frame = cap.read()

        if ret:
            H, W, _ = frame.shape
            masks =  annotation_masks[k]

            # if bounding_coord is not None:
            #         [h1,h2,w1,w2]= bounding_coord
            #         frame = annotated_frame[h1:h2,w1:w2]
            #         this_mask_arr = crop_and_resize_mask(mask=masks,bbox=bounding_coord,original_shape=(H,W))
            # else:
            frame = annotated_frame
            this_mask_arr = masks


            this_resize = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            reshaped = np.transpose(this_resize, (2, 0, 1))

            closest_frame_index = annotated_id * video_buff_size // total_frames
            closest_frame_index = int(min(closest_frame_index, video_buff_size - 1))

            video_buffer[:, closest_frame_index, :, :] = reshaped
            this_mask_arr = this_mask_arr>0.5
            annotation_masks_full_buffer[:, closest_frame_index, :, :] = this_mask_arr.astype(np.uint8)

    cap.release()
    return video_buffer.astype(np.uint8), annotation_masks_full_buffer, Valid_video
def decode_json(json_data,json_file_name, categories):
    decoded_data = json.loads(json_data)
    index = 0
    for frame_data in decoded_data:
        labels = frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("labels", {})
        H = frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("height", {})
        W= frame_data.get("data_units", {}).get(frame_data["data_hash"], {}).get("width", {})
        image_size = (W, H)
        image_sizeR = (H, W)
        image_size_down = (image_resize, image_resize)


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
        if (video_full_name == "Kazu RATS RUL #7 41 time3_normalized.mp4"):
            print(video_full_name)
        annotation_masks =[]
        annotated_frames_id =[]
        annotated_frames =[]

        for key, value in labels.items():
            this_frame_id = key
            this_label = value
            category_masks = {category: np.zeros(image_size_down, dtype=np.uint8) for category in categories}
            this_image=get_specific_frame(this_full_video_path,this_frame_id)
            bounding_coord = None
            if   this_image is not None:
                bounding_coord=find_symmetric_bounding_box(this_image, threshold=20)
                print("crop:")
                print(bounding_coord)
                [h1,h2,w1,w2]= bounding_coord
                this_image = this_image[h1:h2,w1:w2]


            


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
                    if Crop_flag and bounding_coord is not None:
                        [h1,h2,w1,w2]= bounding_coord
                        this_mask_array = this_mask_array[h1:h2,w1:w2]

                    this_mask_array = resize_mask(this_mask_array, image_size_down) 
                    # Accumulate the mask in the corresponding channel
                    category_index = categories.index(this_category_type)
                    category_masks[this_category_type] += this_mask_array
                # this_mask.show()
            color_image = np.zeros((image_resize, image_resize, 3), dtype=np.uint8)
            
        # Fill the color image with the corresponding colors based on the masks
            for category, mask in category_masks.items():
                color_image[mask > 0] = category_colors[category]


            # Display the color-coded image using OpenCV
            if this_image is not None and (len(this_label['objects'])>0):
                            
                # save_decoded_images()
                padded_frame_id = this_frame_id.zfill(4)
                one_hot_mask = np.stack([category_masks[category] for category in categories], axis=0) # Modified: Create multi-channel one-hot mask
                one_hot_mask = one_hot_mask >0.5
                annotation_masks.append(one_hot_mask)  # Added: Store one-hot mask in a list
                annotated_frames_id.append(this_frame_id)
                annotated_frames.append(this_image)
                # save_decoded_images (Decoded_data_dir,this_image,color_image,category_masks, padded_frame_id,video_name,json_file_name)
                # cv2.imshow('Color Mask', color_image)
                # cv2.imshow('Color Image', this_image)
                this_image = cv2.resize(this_image,image_size_down, interpolation=cv2.INTER_AREA)

                show_images_to_visdom(this_image, color_image)
                # cv2.waitKey(10)
            print(this_frame_id)
            pass
            # cv2.destroyAllWindows()
        index +=1
        annotated_frames_id = list(set(annotated_frames_id))
        video_buffer, annotation_masks_full_buffer, valid_video = load_a_video_buffer(this_full_video_path, video_buff_size, image_resize,
                                                    annotated_frames_id,annotation_masks,annotated_frames,bounding_coord, Display_loading_video=True)
        
        if valid_video:
            # index += 1
            save_pkl_name = json_file_name+'_'+video_name
            print("Got a valid video", this_full_video_path, video_buffer.shape)
            # sam_and_save_features(index, video_buffer, Vit_encoder, output_folder_sam_feature)
            save_sampled_clip_and_masks(save_pkl_name, video_buffer, annotation_masks_full_buffer, output_folder_pkl)
        else:
            print("Error: Video is not valid", this_full_video_path)

        #video_buffer + annotation_masks_buffer save to the folder#

        #Complete this code for generate SAM features and save to the folder#


        # print(video_buffer)
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