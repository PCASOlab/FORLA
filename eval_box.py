from eval import *
from dataset import io
from working_dir_root import Visdom_flag
import cv2
from visdom import Visdom
if Visdom_flag:
  viz = Visdom(port=8097)
from model.model_operator import post_process_softmask
from working_dir_root import Display_visdom_figure
from  data_pre_curation. data_ytobj_box_train import apply_mask
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
# import torch.nn.functional as F
# from scipy.spatial.distance import directed_hausdorff
def scale_heatmap(heatmap):
    """
    Scale down values larger than 0.5 and scale up values between 0.1 and 0.5 in the heatmap.
    
    Args:
    heatmap (numpy array): The input heatmap with values between 0 and 1.
    
    Returns:
    numpy array: The scaled heatmap.
    """
    scaled_heatmap = np.copy(heatmap)
    
    # Scale down values larger than 0.5
    mask_high = scaled_heatmap > 0.5
    scaled_heatmap[mask_high] = 0.5 + (scaled_heatmap[mask_high] - 0.5) * 0.5  # Example scaling factor 0.5
    
    # Scale up values between 0.1 and 0.5
    mask_mid = (scaled_heatmap > 0.1) & (scaled_heatmap <= 0.5)
    scaled_heatmap[mask_mid] = 0.1+ (scaled_heatmap[mask_mid] - 0.1) * 1.8  # Example scaling factor 2.0
    
    return scaled_heatmap
def get_bounding_box(mask, threshold=0.5):
   

  # Binarize the mask (optional for non-binary masks)
  if mask.dtype != np.bool_:
    mask = mask > threshold

  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(contours) == 0:
    return None

  # Get the first contour (assuming single object)
  largest_contour = max(contours, key=cv2.contourArea)

  x, y, w, h = cv2.boundingRect(largest_contour)
#   x, y, w, h = cv2.boundingRect(contour)

  return x, y, x + w, y + h
def plot_and_save_image_with_bboxes(plot_img, plot_GT_mask, plot_pr_mask):
 

  # Convert image to BGR format if needed (assuming plot_img is RGB)
  img = plot_img.copy()
  if img.shape[-1] == 3 and img.dtype == np.uint8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  # Get bounding boxes
  gt_bbox = get_bounding_box(plot_GT_mask)
  pr_bbox = get_bounding_box(plot_pr_mask)

  # Draw bounding boxes
  if gt_bbox is not None:
    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 1)  # Green for GT
  if pr_bbox is not None:
    cv2.rectangle(img, (pr_bbox[0], pr_bbox[1]), (pr_bbox[2], pr_bbox[3]), (0, 0, 255), 1)  # Red for predicted

  # Save the image
#   cv2.imwrite(output_filename, img)
  return img,gt_bbox,pr_bbox
def plot_and_save_image_with_heatmap(plot_img, plot_pr_mask, gt_bbox,pr_bbox):
  """
  Plots a heatmap overlay of the predicted mask and bounding boxes on the image.

  Args:
      plot_img: Input image (numpy array) of shape [3, H, W].
      plot_pr_mask: Predicted mask (numpy array) of shape [H, W].
      output_filename: Path to save the output image.
  """

  # Convert image to BGR format if needed (assuming plot_img is RGB)
  img = plot_img.copy()
  if img.shape[-1] == 3 and img.dtype == np.uint8:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#   plot_pr_mask= (plot_pr_mask>0)*plot_pr_mask
#   plot_pr_mask = plot_pr_mask -np.min(plot_pr_mask)
#   plot_pr_mask = plot_pr_mask /(np.max(plot_pr_mask)+0.0000001)*254 
  plot_pr_mask = plot_pr_mask *254
    # stack
    # stack = (stack>20)*stack
    # stack = (stack>0.5)*128
  plot_pr_mask = np.clip(plot_pr_mask,0,254)
  # Apply heatmap
  heatmap = cv2.applyColorMap(plot_pr_mask.astype((np.uint8)), cv2.COLORMAP_JET)  # Scale mask to 0-255
  heatmap = cv2.addWeighted(heatmap, 0.5, img.astype((np.uint8)), 0.5, 0)  # Blend heatmap with image

  # Get bounding box (optional)
#   pr_bbox = get_bounding_box(plot_pr_mask)

  # Draw bounding box (optional)
  if gt_bbox is not None:
    cv2.rectangle(heatmap, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 1)  # Green for GT
  if pr_bbox is not None:
    cv2.rectangle(heatmap, (pr_bbox[0], pr_bbox[1]), (pr_bbox[2], pr_bbox[3]), (0, 0, 255), 1)  # Red for predicted

  # Save the image
  return heatmap
def cal_dice_np(true, predict):
    # Dice coefficient
    intersection = np.sum(true * predict)
    union = np.sum(true) + np.sum(predict)
    s = 0.000000001
    dice = (2. * intersection + s) / (union + s)
    return dice
def select_valid_masks(M_T, M_P, V_P):
  """
  Selects valid masks from ground truth, corresponding predicted masks, 
  and channel with maximum value.

  Args:
      M_T: Ground truth masks with shape [Len_array, H, W]. (Can contain None values)
      M_P: Predicted multi-channel masks with shape [channel, Len_array, H, W].
      V_P: Predicted array indicating valid channels with shape [channel].

  Returns:
      valid_M_T: Valid ground truth masks with shape [num_valid, H, W].
      valid_M_P: Corresponding predicted masks with shape [num_valid, H, W].
      max_channel: Channel index with maximum value for each valid mask with shape [num_valid].

  """
  # Find non-None indices in ground truth
  valid_indices = [i for i, mask in enumerate(M_T) if mask is not None]

  # Select valid masks from ground truth
  valid_M_T = M_T[valid_indices]

  # Select corresponding predicted masks
  valid_M_P = M_P[:, valid_indices]  # Select all channels for valid indices

  # Find channel with maximum value for each valid mask
  max_values = np.argmax(valid_M_P, axis=0)  # Find max index along channel axis

  # Select valid channels based on V_P (optional)
  # If V_P indicates specific valid channels, uncomment this:
  # valid_channels = V_P[max_values]

  return valid_M_T, valid_M_P, max_values

def cal_all_metrics_box (read_id,Output_root, label_masks, frame_label, video_label, predic_mask_3D, output_video_label,output_frame_label, input_video, results):
    valid_indices = [i for i, mask in enumerate(label_masks) if mask is not None]
    if any(valid_indices):
    # Select valid masks from ground truth
        valid_M_T = label_masks[valid_indices]
        valid_M_T = np.stack(valid_M_T, axis=0)
        # Select corresponding predicted masks
        # valid_M_P = M_P[:, valid_indices]  # Select all channels for valid indices

        # device = label_masks.device
        # predic_mask_3D=predic_mask_3D.to(device)
        # output_video_label =output_video_label.to(device)
        _, H_i, W_i = valid_M_T.shape
        # predic_mask_3D =   F.interpolate(predic_mask_3D,  size=( H, W), mode='bilinear', align_corners=False)
        ch, D, H, W = predic_mask_3D.size()
        predic_mask_3D=(predic_mask_3D>0)*predic_mask_3D
        # for i in range(ch):
        #     predic_mask_3D[i,:,:,:] = predic_mask_3D[i,:,:,:] -torch.min(predic_mask_3D[i,:,:,:])
        
        #     predic_mask_3D[i,:,:,:] = predic_mask_3D[i,:,:,:] /(torch.max(predic_mask_3D[i,:,:,:] )+0.0000001)
        # predic_mask_3D = torch.clip(predic_mask_3D,0,1)
        predic_mask_soft = predic_mask_3D

        # predic_mask_3D = predic_mask_3D>0.05
        # predic_mask_3D = torch.clamp(predic_mask_3D,0,1)
        ch,D,_,_ = predic_mask_3D.size()

        output_video_label = (output_video_label > 0.5) * 1

        output_video_label_expanded = output_video_label.reshape(ch, 1, 1, 1) 
        output_video_label_expanded = output_video_label_expanded.repeat(1, D, H, W)
        predic_mask_3D = predic_mask_3D * output_video_label_expanded

        frame_label = frame_label.permute(1, 0)
        # Sum along the spatial dimensions to get frame-level predictions
        predic_frame = torch.sum(predic_mask_3D, dim=(-1, -2))
        # Sum along the frame dimension to get video-level predictions
        predic_video_from_cam = torch.max(predic_frame, dim=(-1))[0]
        predic_video_from_cam = (predic_video_from_cam > 1000) * 1
        # Calculate video-level AP from camera
        video_ap_from_cam = cal_ap_video(video_label, predic_video_from_cam)
        print("Video AP from cam:", video_ap_from_cam)
        
        # Calculate video-level AP from model output
        video_ap = cal_ap_video(video_label, output_video_label)
        print("Video AP from model output:", video_ap)
        predic_frame = (predic_frame > 100) * 1
        if output_frame_label is not None:
            predic_frame = (output_frame_label[0] > 0.5) * 1



        frame_ap = cal_ap_frame(frame_label, predic_frame)
        print("Frame AP from model output:", frame_ap)
        valid_channels =  np.argmax(video_label.cpu().numpy() )# Select channel with highest value in V_P avoid adapt batch
        target_channels =  np.argmax(video_label.cpu().numpy() )# Select channel with highest value in V_P
        # valid_M_P = M_P[valid_channels, valid_indices]  #
        select_predic_mask_3D = predic_mask_3D[valid_channels,valid_indices]
        select_mask_soft = predic_mask_soft [valid_channels,valid_indices]
        select_input_frame = input_video[:,valid_indices]

        
        
        # print(np.sum(counts))

        # Apply threshold to frame predictions
        # predic_frame = (predic_frame > 20) * 1.0
        
        

        plot_img = select_input_frame[:,0,:,:]
        # plot_GT_mask = valid_M_T[0]
        plot_GT_mask = valid_M_T[0]

        plot_pr_mask = select_predic_mask_3D[0].cpu().detach().numpy()
        plot_soft_masks = select_mask_soft[0].cpu().detach().numpy()
        plot_pr_mask =  cv2.resize(plot_pr_mask, (H_i, W_i), interpolation = cv2.INTER_LINEAR)
        plot_soft_masks =  cv2.resize(plot_soft_masks, (H_i, H_i), interpolation = cv2.INTER_LINEAR)

        plot_soft_masks= (plot_soft_masks>0)*plot_soft_masks
        plot_soft_masks = plot_soft_masks -np.min( plot_soft_masks )
        plot_soft_masks = plot_soft_masks /(np.max( plot_soft_masks )+0.0000001) 
        # plot_pr_mask=post_process_softmask(plot_soft_masks*4 ,plot_img )
        plot_soft_masks
        plot_pr_mask= post_process_softmask(scale_heatmap(plot_soft_masks)*3.0  ,plot_img )

        # plot_pr_mask = plot_soft_masks>0.05
        plot_img = plot_img.transpose(1, 2, 0)

        img_box, gt_bbx,pr_bbx =  plot_and_save_image_with_bboxes(plot_img, plot_GT_mask, plot_pr_mask)
        heat_box = plot_and_save_image_with_heatmap(plot_img, plot_soft_masks,gt_bbx,pr_bbx)
        heat_box2 = plot_and_save_image_with_heatmap(plot_img, plot_pr_mask,gt_bbx,pr_bbx)

        stitched_img = cv2.hconcat([img_box.astype((np.uint8)), heat_box.astype((np.uint8)),heat_box2.astype((np.uint8))])
        io.save_img_to_folder(Output_root + "image/bbx/" +str(valid_channels) + "/",  read_id, stitched_img.astype((np.uint8)) )
        if Display_visdom_figure:
            
            stitched_img =  cv2.cvtColor(stitched_img, cv2.COLOR_RGB2BGR)
            
            viz.image(np.transpose(stitched_img.astype((np.uint8)), (2, 0, 1)), opts=dict(title=f'{read_id} - _overlay'))
        # Calculate Dice coefficient for video-level predictions
        if pr_bbx is not None:
          _,pr_box_mask = apply_mask(plot_img,pr_bbx)
        else:
            pr_box_mask = plot_GT_mask*0
        dice = cal_dice_np(pr_box_mask, plot_GT_mask)
        dice = round(dice.item(), 4)
        print("Dice coefficient:", dice)
        this_corr_vect = video_label.cpu().numpy()  * dice
        print(this_corr_vect)
        this_corr_cnt_v = this_corr_vect>0.5
        results["score"].append(this_corr_vect)
        results["count"].append(video_label.cpu().numpy())
        results["cpr_cont"] .append(this_corr_cnt_v)
        scores = np.array(results["score"])
        counts = np.array ( results["count"])
        cor_cnt_v = np.array ( results["cpr_cont"])
        all_cate = np.sum(scores, axis=0) / (np.sum(counts, axis=0) + 0.000000001)
        all_corbox = np.sum(cor_cnt_v, axis=0) / (np.sum(counts, axis=0) + 0.000000001)
        print("avg dic per cat:")
        print(all_cate)  # Format all_cate with 4 decimals
        print("avg corb per cat:")
        print(all_corbox)  # Format all_cate with 4 decimals
        print("avg score:", "{:.4f}".format(np.sum(all_cate) / len(all_cate)))  # Format average with 4 decimals
        print("avg corbox:", "{:.4f}".format(np.sum(all_corbox) / len(all_corbox)))  # Format average with 4 decimals
          


        global metrics_video_data
        metrics_video_data.append({'read_id': read_id, 'Video_AP_Cam': video_ap_from_cam, 'Video_AP_Model': video_ap, 'Corr': dice})

        # Add frame-level accuracy scores to metrics_frame_data
        global metrics_frame_data
        new_frame_data = {'read_id': read_id}
        for i in range(len(all_corbox)):
            new_frame_data[f'Frame_{i+1}_AP'] = all_corbox[i]
        metrics_frame_data.append(new_frame_data)

        # Convert lists to DataFrames
        metrics_video = pd.DataFrame(metrics_video_data)
        metrics_frame = pd.DataFrame(metrics_frame_data)

        # Save to Excel files
        # metrics_video.to_excel(Output_root+'metrics_video.xlsx', index=False, float_format='%.4f')
        # metrics_frame.to_excel(Output_root+'metrics_frame.xlsx', index=False, float_format='%.4f')
        return results

# Example usage
# cal_all_metrics(...)