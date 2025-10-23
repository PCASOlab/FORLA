import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from torch.profiler import profile, record_function, ProfilerActivity
from ptflops import get_model_complexity_info
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from model.slots import temporal_slots_slide

# Create empty DataFrames to store metrics
metrics_video_data = []
metrics_frame_data = []
# Create empty DataFrames to store metrics
 

# NEW: per-class accumulators
metrics_video_class_data = []
metrics_frame_class_data = []

def get_model_flops(model, input_size):
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    flops = FlopCountAnalysis(model, dummy_input)
    return flops.total()
def get_component_flops(model_infer):
    """Compute FLOPs for each component including sliding window average"""
    # device = model_infer.device
    device = torch.device("cpu")
    results = {}
    batch_size = 1
    num_frames = 24
    num_slots = 7
    slot_dim = 64
    num_patches = 784  # 28x28 patches
    feature_dim = 64  # Feature dimension
    
    # 1. Encoder FLOPs
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            
        def forward(self, x):
            return self.encoder(x)["features"]
    
    encoder_wrapper = EncoderWrapper(model_infer.model.encoder).to(device)
    encoder_wrapper.eval()
    
    encoder_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    flops = FlopCountAnalysis(encoder_wrapper, encoder_input).total()
    results["encoder"] = flops

    # 2. Processor FLOPs
    class ProcessorWrapper(nn.Module):
        def __init__(self, processor, initializer):
            super().__init__()
            self.processor = processor
            self.initializer = initializer
            
        def forward(self, features):
            # features = features.permute(0, 3, 2, 1)  # (1, 128, 784, 24)
            slots_initial = self.initializer(batch_size=batch_size)
            # return self.processor(slots_initial, features, rnn=True,Next_state_predict="videosaur")["state"]
            return self.processor(slots_initial, features, rnn=False)["state"]
        # 
    
    processor_wrapper = ProcessorWrapper(
        model_infer.model.processor,
        model_infer.model.initializer
    ).to(device)
    processor_wrapper.eval()
    
    processor_input = torch.randn(batch_size, num_frames, num_patches, feature_dim).to(device)
    flops = FlopCountAnalysis(processor_wrapper, processor_input).total()
    results["processor"] = flops

    # 3. Sliding Window Average FLOPs
    class SlidingWindowWrapper(nn.Module):
        def __init__(self, temporal_binder):
            super().__init__()
            self.temporal_binder = temporal_binder
            
        def forward(self, slots):
            # slots shape: (B, T, N, D) = (1, 24, 7, 64)
            return temporal_slots_slide.apply_sliding_window_avg(self.temporal_binder, slots)
    
    # Check if temporal_binder exists
    if hasattr(model_infer.model, 'temporal_binder'):
        temporal_binder = model_infer.model.temporal_binder.to(device)
        window_wrapper = SlidingWindowWrapper(temporal_binder).to(device)
        window_wrapper.eval()
        
        # Input: (1, 24, 7, 64)
        window_input = torch.randn(batch_size, num_frames, num_slots, slot_dim).to(device)
        flops = FlopCountAnalysis(window_wrapper, window_input).total()
        results["sliding_window_avg"] = flops
    else:
        results["sliding_window_avg"] = 0

    # 4. Decoder FLOPs
    class DecoderWrapper(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            
        def forward(self, slots):
            return self.decoder(slots)["masks"]
    
    decoder_wrapper = DecoderWrapper(model_infer.model.decoder).to(device)
    decoder_wrapper.eval()
    
    decoder_input = torch.randn(batch_size, num_frames, num_slots, slot_dim).to(device)
    flops = FlopCountAnalysis(decoder_wrapper, decoder_input).total()
    results["decoder"] = flops

    return results
def calculate_model_flops_slot_difussion(model):
    """
    Calculate and print FLOPs for model.module.loss_function
    with input shape [1, 24, 3, 128, 128]
    
    Args:
        model: Your model instance
    """
    try:
        # Get device from model parameters
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_img = torch.randn(1, 24, 3, 128, 128).to(device)
        batch_data = {
                'img': dummy_img,
                'data_idx': torch.tensor([0], device=device)
            }
            
        # # Create wrapper for dictionary input
        # def loss_wrapper(img_tensor):
        #     batch_data = {
        #         'img': img_tensor,
        #         'data_idx': torch.tensor([0], device=device)
        #     }
            
        #     # Handle DataParallel/DistributedDataParallel
        #     # return model.loss_function(batch_data)
        #     return model(batch_data)
        
            # if hasattr(model, 'module'):
            #     return model.module.loss_function(batch_data)
            # return model.loss_function(batch_data)
        
        # Set model to eval mode
        model.eval()
        
        # Calculate FLOPs
        with torch.no_grad():
            flops = FlopCountAnalysis(model, batch_data)
            total_flops = flops.total()
        
        # Print results
        print(f"\n{' FLOPs Analysis ':=^60}")
        print(f"Input shape: {tuple(dummy_img.shape)}")
        print(f"Total FLOPs: {total_flops / 1e9:.4f} GFLOPs")
        # print(f"Detailed view:\n{flop_count_table(flops)}")
        print("=" * 60)
        
    except AttributeError:
        print("Error: Model doesn't have required loss_function method")
    except Exception as e:
        print(f"FLOPs calculation failed: {str(e)}")

# Usage example:
# calculate_model_flops(your_model)
def get_model_infer_flops(model_infer, input_size):
    """
    Compute FLOPs including slot BERT operations
    Args:
        model_infer: Your model instance
        input_size: Tuple representing input dimensions (C, T, H, W)
    Returns:
        Total FLOPs count
    """
    class ModelWrapper(nn.Module):
        def __init__(self, model, use_bert, slot_ini, Mask_feat, img_sim):
            super().__init__()
            self.model = model
            self.use_bert = use_bert
            self.slot_ini = slot_ini
            self.Mask_feat = Mask_feat
            self.img_sim = img_sim
            
        def forward(self, x):
            video_input = {"video": x.permute(0, 2, 1, 3, 4)}
            feature_stack = self.model.forward_feature_stack(
                video_input,
                self.use_bert,
                self.slot_ini,
                self.Mask_feat,
                self.img_sim
            )
            # Explicitly include slot BERT operations
            if self.use_bert:
                slots = feature_stack["slot_features"]
                bert_output = self.model.slot_bert(slots)  # Assuming slot_bert exists
                feature_stack["slot_features"] = bert_output
                
            output = self.model(
                video_input,
                feature_stack,
                self.use_bert,
                self.slot_ini,
                self.Mask_feat,
                self.img_sim
            )
            return output['decoder']['masks']

    # Create wrapper with BERT parameters
    wrapper = ModelWrapper(
        model_infer.model,
        model_infer.use_bert,
        model_infer.slot_ini,
        model_infer.Mask_feat,
        model_infer.img_sim
    )
    wrapper.eval()
    
    # Input dimensions: (batch, C, T, H, W)
    batch_size = 1
    input_tensor = torch.randn(batch_size, *input_size)
    
    # Profile FLOPs
    flops, _ = get_model_complexity_info(
        wrapper,
        tuple(input_size),
        as_strings=False,
        print_per_layer_stat=True,  # Enable to see layer-wise breakdown
        verbose=True
    )
    
    return flops
def get_model_infer_flops2(model_infer, input_size):
    # Create properly shaped input
    dummy_input = torch.randn(1, *input_size).to(model_infer.device)
    
    # Disable gradients globally
    with torch.no_grad():
        # Preprocessing steps
        bz, ch, D, H, W = dummy_input.size()
        input_resample = F.interpolate(
            dummy_input, size=(D, 224, 224), 
            mode='trilinear', align_corners=False
        )
        input_resample = (input_resample - 124.0) / 60.0
        video_input = {"video": input_resample.permute(0, 2, 1, 3, 4)}
        
        # Set models to evaluation mode
        # model_infer.model.eval()
        # model_infer.model_s.eval()
        
        # Use PyTorch profiler to measure FLOPs
        feature_stack =None

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with record_function("model_inference"):
                # Forward pass
                feature_stack =None
                # feature_stack = model_infer.model.forward_feature_stack(
                #     video_input, model_infer.use_bert, 
                #     slot_ini=model_infer.slot_ini,
                #     Mask_feat=model_infer.Mask_feat,
                #     img_sim=model_infer.img_sim
                # )
                output = model_infer.model(
                    video_input, feature_stack, Using_bert=model_infer.use_bert, 
                    slot_ini=model_infer.slot_ini,
                    Mask_feat=model_infer.Mask_feat,
                    img_sim=model_infer.img_sim
                )
                
                # Convert mask
                # b, f, n_slots, hw = output["decoder"]["masks"].shape
                # h = w = int(np.sqrt(hw))
                # masks_video = output["decoder"]["masks"].reshape(b, f, n_slots, h, w)
                # masks_video = masks_video.permute(0, 2, 1, 3, 4)
                # cam3D = F.interpolate(
                #     masks_video, size=(D, H, W), 
                #     mode='trilinear', align_corners=False
                # )
    
    # Calculate total FLOPs
    total_flops = 0
    for event in prof.key_averages():
        if event.flops > 0:
            total_flops += event.flops
    
    return total_flops
def cal_hausdorff(true, predict, class_indices):
    hausdorff_distances = []
    
    true_np = true.cpu().numpy()  # Convert to numpy array
    predict_np = predict.cpu().numpy()  # Convert to numpy array

    for class_idx in class_indices:
        for frame in range(true_np.shape[1]):  # Iterate over all frames
            true_frame = true_np[class_idx, frame]
            predict_frame = predict_np[class_idx, frame]
            
            # Get the coordinates of the true and predicted points
            true_points = np.argwhere(true_frame > 0)
            predict_points = np.argwhere(predict_frame > 0)
            
            if len(true_points) == 0 or len(predict_points) == 0:
                continue  # Skip frames with no points

            # Calculate the directed Hausdorff distance
            hausdorff_dist = max(directed_hausdorff(true_points, predict_points)[0],
                                 directed_hausdorff(predict_points, true_points)[0])
            hausdorff_distances.append(hausdorff_dist)

    if hausdorff_distances:
        average_hausdorff_distance = np.mean(hausdorff_distances)
    else:
        average_hausdorff_distance = 0  # If no valid frames, return infinity

    return average_hausdorff_distance
def cal_J(true, predict):
    # Intersection and Union for calculating Jaccard Index (Intersection over Union)
    AnB = true * predict  # Element-wise multiplication for intersection
    AuB = true + predict  # Element-wise addition for union
    AuB = torch.clamp(AuB, 0, 1)  # Clamp values between 0 and 1
    s = 0.000000001
    this_j = (torch.sum(AnB) + s) / (torch.sum(AuB) + s)  # Compute Jaccard Index
    return this_j

def cal_dice(true, predict):
    # Dice coefficient
    intersection = torch.sum(true * predict)
    union = torch.sum(true) + torch.sum(predict)
    s = 0.000000001
    dice = (2. * intersection + s) / (union + s)
    return dice

def cal_ap_video(true, predict):
    # Move tensors to CPU before conversion
    true_cpu = true.cpu().numpy()
    predict_cpu = predict.cpu().numpy()
    ap = accuracy_score(true_cpu, predict_cpu)
    return ap 

def cal_ap_frame(true, predict):
    average_precision_frame = []

    for i in range(len(true[0])):
        ap_frame = accuracy_score(true[:, i].cpu().numpy(), predict[:, i].cpu().numpy())
        average_precision_frame.append(ap_frame)

    return average_precision_frame

def cal_fpr(true, predict, class_indices):
    false_positives = 0
    true_negatives = 0
    true_np = true.cpu().numpy()
    predict_np = predict.cpu().numpy()

    for class_idx in class_indices:
        true_class = true_np[class_idx]
        predict_class = predict_np[class_idx]
        false_positives += np.sum((predict_class == 1) & (true_class == 0))
        true_negatives += np.sum((predict_class == 0) & (true_class == 0))

    fpr = false_positives / (false_positives + true_negatives + 1e-10)  # Adding a small value to avoid division by zero
    return fpr

def cal_tnr(true, predict, class_indices):
    true_negatives = 0
    false_positives = 0
    true_np = true.cpu().numpy()
    predict_np = predict.cpu().numpy()

    for class_idx in class_indices:
        true_class = true_np[class_idx]
        predict_class = predict_np[class_idx]
        true_negatives += np.sum((predict_class == 0) & (true_class == 0))
        false_positives += np.sum((predict_class == 1) & (true_class == 0))

    tnr = true_negatives / (true_negatives + false_positives + 1e-10)  # Adding a small value to avoid division by zero
    return tnr

def cal_all_metrics(read_id, Output_root, label_mask, frame_label, video_label,
                    predic_mask_3D, output_video_label, output_frame_label):
    device = label_mask.device
    predic_mask_3D = predic_mask_3D.to(device)
    output_video_label = output_video_label.to(device)

    ch, D, H, W = label_mask.size()
    predic_mask_3D = F.interpolate(predic_mask_3D, size=(H, W), mode='bilinear', align_corners=False)
    predic_mask_3D = (predic_mask_3D > 0) * predic_mask_3D
    predic_mask_3D = predic_mask_3D - torch.min(predic_mask_3D)
    predic_mask_3D = predic_mask_3D / (torch.max(predic_mask_3D) + 1e-7) * 1
    predic_mask_3D = predic_mask_3D > 0.1
    predic_mask_3D = torch.clamp(predic_mask_3D, 0, 1)

    output_video_label = (output_video_label > 0.5) * 1

    output_video_label_expanded = output_video_label.reshape(ch, 1, 1, 1).repeat(1, D, H, W)
    predic_mask_3D = predic_mask_3D * output_video_label_expanded

    # frame_label: originally (D, ch); you permute to (ch, D)
    frame_label = frame_label.permute(1, 0)

    # Frame-level predictions (counts per frame)
    predic_frame = torch.sum(predic_mask_3D, dim=(-1, -2))  # (ch, D)

    # Video-level from CAM
    predic_video_from_cam = torch.max(predic_frame, dim=-1)[0]  # (ch,)
    predic_video_from_cam = (predic_video_from_cam > 1000) * 1

    # Video-level AP
    video_ap_from_cam = cal_ap_video(video_label, predic_video_from_cam)
    print("Video AP from cam:", video_ap_from_cam)

    video_ap = cal_ap_video(video_label, output_video_label)
    print("Video AP from model output:", video_ap)

    # Use model's frame outputs if provided
    predic_frame_bin = (predic_frame > 100) * 1
    if output_frame_label is not None:
        predic_frame_bin = (output_frame_label[0] > 0.5) * 1

    frame_ap = cal_ap_frame(frame_label, predic_frame_bin)
    print("Frame AP from model output:", frame_ap)

    # Global IoU / Dice (all classes together)
    IoU = round(cal_J(label_mask[0], predic_mask_3D[0]).item(), 4)
    print("Intersection over Union (IoU):", IoU)

    dice = round(cal_dice(label_mask, predic_mask_3D).item(), 4)
    print("Dice coefficient:", dice)

    # Positive-frame mask expansion (per frame threshold already in predic_frame_bin)
    predic_frame_expanded = predic_frame_bin.reshape(ch, D, 1, 1).repeat(1, 1, H, W)

    IoU_maskout = round(cal_J(label_mask * predic_frame_expanded, predic_mask_3D * predic_frame_expanded).item(), 4)
    print("IoU for positive frames only:", IoU_maskout)

    Dice_maskout = round(cal_dice(label_mask * predic_frame_expanded, predic_mask_3D * predic_frame_expanded).item(), 4)
    print("Dice coefficient for positive frames only:", Dice_maskout)

    # ---- Save per-video aggregate (existing) ----
    global metrics_video_data
    metrics_video_data.append({
        'read_id': read_id,
        'Video_AP_Cam': video_ap_from_cam,
        'Video_AP_Model': video_ap,
        'IoU': IoU,
        'IoU_Positive_Frames': IoU_maskout,
        'Dice_Coefficient': dice,
        'Dice_Coefficient_Positive_Frames': Dice_maskout
    })

    # ---- Save per-frame (existing) ----
    global metrics_frame_data
    new_frame_data = {'read_id': read_id}
    for i in range(len(frame_ap)):
        new_frame_data[f'Frame_{i+1}_AP'] = frame_ap[i]
    metrics_frame_data.append(new_frame_data)

    # =========================
    # NEW: Per-class metrics
    # =========================
    global metrics_video_class_data, metrics_frame_class_data

    # Per-class video-level APs are simply computed per class from your vectors
    # For per-class IoU/Dice, compute on (D,H,W) volumes per class
    for c in range(ch):
        # Per-class video predictions
        ap_cam_c = accuracy_score(
            video_label[c].view(-1).cpu().numpy(),
            predic_video_from_cam[c].view(-1).cpu().numpy()
        )
        ap_model_c = accuracy_score(
            video_label[c].view(-1).cpu().numpy(),
            output_video_label[c].view(-1).cpu().numpy()
        )

        # Per-class IoU/Dice over all frames & pixels
        iou_c = cal_J(label_mask[c], predic_mask_3D[c]).item()
        dice_c = cal_dice(label_mask[c], predic_mask_3D[c]).item()

        # Masked by positive frames (for that class only)
        pf_exp_c = predic_frame_bin[c].reshape(D, 1, 1).repeat(1, H, W)
        iou_pos_c = cal_J(label_mask[c] * pf_exp_c, predic_mask_3D[c] * pf_exp_c).item()
        dice_pos_c = cal_dice(label_mask[c] * pf_exp_c, predic_mask_3D[c] * pf_exp_c).item()

        metrics_video_class_data.append({
            'read_id': read_id,
            'class_id': c,
            'Video_AP_Cam': round(ap_cam_c, 4),
            'Video_AP_Model': round(ap_model_c, 4),
            'IoU': round(iou_c, 4),
            'IoU_Positive_Frames': round(iou_pos_c, 4),
            'Dice_Coefficient': round(dice_c, 4),
            'Dice_Coefficient_Positive_Frames': round(dice_pos_c, 4),
        })

        # Per-class frame accuracy (across time) for this video
        fa_c = accuracy_score(
            frame_label[c, :].view(-1).cpu().numpy(),
            predic_frame_bin[c, :].view(-1).cpu().numpy()
        )
        metrics_frame_class_data.append({
            'read_id': read_id,
            'class_id': c,
            'Frame_Accuracy': round(float(fa_c), 4),
        })

    # =========================
    # Save all CSVs
    # =========================
    metrics_video = pd.DataFrame(metrics_video_data)
    metrics_frame = pd.DataFrame(metrics_frame_data)

    metrics_video.to_csv(Output_root + 'metrics_video.csv', index=False, float_format='%.4f')
    metrics_frame.to_csv(Output_root + 'metrics_frame.csv', index=False, float_format='%.4f')

    # Existing averages (your originals)
    video_means = metrics_video.drop(columns=['read_id']).mean(numeric_only=True).to_frame(name='mean').T
    video_means.to_csv(Output_root + 'metrics_video_average.csv', index=False, float_format='%.4f')

    if len(metrics_video_data) > 0:
        video_avg = metrics_video.mean(numeric_only=True)
        video_avg_df = pd.DataFrame([video_avg], index=['Average'])
        video_avg_df.to_csv(Output_root + 'metrics_video_average.csv', float_format='%.4f')

    if len(metrics_frame_data) > 0:
        frame_avg = metrics_frame.mean(numeric_only=True)
        frame_avg_df = pd.DataFrame([frame_avg], index=['Average'])
        frame_avg_df.to_csv(Output_root + 'metrics_frame_average.csv', float_format='%.4f')

        all_frame_values = metrics_frame.drop('read_id', axis=1).values.flatten()
        overall_frame_accuracy = np.mean(all_frame_values)
        overall_frame_df = pd.DataFrame({'Overall_Accuracy': [overall_frame_accuracy]})
        overall_frame_df.to_csv(Output_root + 'metrics_frame_overall_accuracy.csv', index=False, float_format='%.4f')

    # ---- NEW: save per-class CSVs + per-class averages ----
    if len(metrics_video_class_data) > 0:
        df_vc = pd.DataFrame(metrics_video_class_data)
        df_vc.to_csv(Output_root + 'metrics_video_per_class.csv', index=False, float_format='%.4f')

        # averages per class across videos
        vc_avg = df_vc.groupby('class_id').mean(numeric_only=True).reset_index()
        vc_avg.to_csv(Output_root + 'metrics_video_per_class_average_by_class.csv', index=False, float_format='%.4f')

    if len(metrics_frame_class_data) > 0:
        df_fc = pd.DataFrame(metrics_frame_class_data)
        df_fc.to_csv(Output_root + 'metrics_frame_per_class.csv', index=False, float_format='%.4f')

        fc_avg = df_fc.groupby('class_id').mean(numeric_only=True).reset_index()
        fc_avg.to_csv(Output_root + 'metrics_frame_per_class_average_by_class.csv', index=False, float_format='%.4f')


# Example usage
# cal_all_metrics(...)