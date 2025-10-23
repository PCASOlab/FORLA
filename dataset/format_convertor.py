import numpy as np
 

class_name_Cholec_8k={0: 'Black Background',
                    1: 'Abdominal Wall',
                    2: 'Liver',
                    3: 'Gastrointestinal Tract',
                    4: 'Fat',
                    5: 'Grasper',
                    6: 'Connective Tissue',
                    7: 'Blood',
                    8: 'Cystic Duct',
                    9: 'L-hook Electrocautery',
                    10: 'Gallbladder',
                    11: 'Hepatic Vein',
                    12: 'Liver Ligament'}

categories_cholec = [
        'Grasper', #0   
        'Bipolar', #1    
        'Hook', #2    
        'Scissors', #3      
        'Clipper',#4       
        'Irrigator',#5    
        'SpecimenBag',#6                  
    ]
categories_thoracic = [
    'Lymph node',
    'Vagus nereve',
    'Bronchus',
    'Lung parenchyma',
    'Instruments', 
    ]

categories_endovis =  [
    'Prograsp_Forceps_labels',
    'Large_Needle_Driver_labels',
    'Grasping_Retractor_labels',
    'Bipolar_Forceps_labels',
    'Vessel_Sealer_labels',
    'Monopolar_Curved_Scissors_labels',
    'Other_labels'
]
def color_mask_to_instance_mask_movi(color_mask, max_instances=15):
    # Transpose color mask to (Frame_num, H, W, 3)
    color_mask = np.transpose(color_mask, (1, 2, 3, 0))
    frame_number, H, W, _ = color_mask.shape

    # Convert RGB values to integer hashes for faster processing
    r = color_mask[..., 0].astype(np.int64)
    g = color_mask[..., 1].astype(np.int64)
    b = color_mask[..., 2].astype(np.int64)
    color_ints = r + (g << 8) + (b << 16)

    

    # Find unique colors (excluding background)
    unique_ints = np.unique(color_ints)
     
    unique_ints = unique_ints[:max_instances]  # Limit to max_instances
    num_instances = len(unique_ints)

    if num_instances == 0:
        return np.zeros((max_instances, frame_number, H, W), dtype=np.uint8)

    # Sort unique colors for binary search
    unique_sorted = np.sort(unique_ints)
    
    # Flatten and find indices for each pixel
    color_flat = color_ints.ravel()
    idx = np.searchsorted(unique_sorted, color_flat)
    valid = (idx < num_instances) & (unique_sorted[idx] == color_flat)
    labels = np.where(valid, idx, -1).reshape(frame_number, H, W)

    # Create instance masks via broadcasting
    instance_mask = (labels == np.arange(num_instances)[:, None, None, None]).astype(np.uint8)
    
    # Pad with zeros if fewer instances than max_instances
    if num_instances < max_instances:
        pad = ((0, max_instances - num_instances), (0, 0), (0, 0), (0, 0))
        instance_mask = np.pad(instance_mask, pad, mode='constant')
    
    return instance_mask
  
 
def color_mask_to_instance_mask_ytvos(color_mask, max_instances=15):
    # Transpose color mask to (Frame_num, H, W, 3)
    color_mask = np.transpose(color_mask, (1, 2, 3, 0))
    frame_number, H, W, _ = color_mask.shape

    # Convert RGB values to integer hashes for faster processing
    r = color_mask[..., 0].astype(np.int64)
    g = color_mask[..., 1].astype(np.int64)
    b = color_mask[..., 2].astype(np.int64)
    color_ints = r + (g << 8) + (b << 16)

    # Calculate background hash for [0, 1, 0]
    background_color = np.array([0, 1, 0], dtype=np.int64)
    background_hash = background_color[0] + (background_color[1] << 8) + (background_color[2] << 16)

    # Find unique colors (excluding background)
    unique_ints = np.unique(color_ints)
    unique_ints = unique_ints[unique_ints != background_hash]  # Exclude background
    unique_ints = unique_ints[:max_instances]  # Limit to max_instances
    num_instances = len(unique_ints)

    if num_instances == 0:
        return np.zeros((max_instances, frame_number, H, W), dtype=np.uint8)

    # Sort unique colors for binary search
    unique_sorted = np.sort(unique_ints)
    
    # Flatten and find indices for each pixel
    color_flat = color_ints.ravel()
    idx = np.searchsorted(unique_sorted, color_flat)
    valid = (idx < num_instances) & (unique_sorted[idx] == color_flat)
    labels = np.where(valid, idx, -1).reshape(frame_number, H, W)

    # Create instance masks via broadcasting
    instance_mask = (labels == np.arange(num_instances)[:, None, None, None]).astype(np.uint8)
    
    # Pad with zeros if fewer instances than max_instances
    if num_instances < max_instances:
        pad = ((0, max_instances - num_instances), (0, 0), (0, 0), (0, 0))
        instance_mask = np.pad(instance_mask, pad, mode='constant')
    
    return instance_mask
  
def color_mask_to_instance_mask(color_mask, max_instances=20):
    """
    Optimized version using vectorized operations and integer hashing
    - Excludes black [0,0,0] and colors where all components > 180
    """
    H, W, _ = color_mask.shape
    
    # Convert RGB values to integer hashes
    r = color_mask[..., 0].astype(np.int64)
    g = color_mask[..., 1].astype(np.int64)
    b = color_mask[..., 2].astype(np.int64)
    color_ints = r + (g << 8) + (b << 16)
    
    # Find unique colors (excluding black and high values)
    unique_ints = np.unique(color_ints)
    unique_ints = unique_ints[unique_ints != 0]  # Exclude black
    
    # Exclude colors where all components > 180
    rs = unique_ints % 256
    gs = (unique_ints // 256) % 256
    bs = (unique_ints // 65536) % 256
    high_mask = (rs > 180) & (gs > 180) & (bs > 180)
    unique_ints = unique_ints[~high_mask]
    
    # Limit to max instances
    unique_ints = unique_ints[:max_instances]
    num_instances = len(unique_ints)
    
    if num_instances == 0:
        return np.zeros((max_instances, H, W), dtype=np.uint8)
    
    # Create instance masks through vectorized operations
    unique_sorted = np.sort(unique_ints)
    color_flat = color_ints.ravel()
    
    idx = np.searchsorted(unique_sorted, color_flat)
     # CRITICAL FIX: Prevent index-out-of-bounds
    valid = (idx < num_instances)
    valid[valid] = unique_sorted[idx[valid]] == color_flat[valid]
    
    labels = np.where(valid, idx, -1).reshape(H, W)

    instance_mask = (labels == np.arange(num_instances)[:, None, None]).astype(np.uint8)
    
    # Pad with zeros if needed
    if num_instances < max_instances:
        instance_mask = np.pad(
            instance_mask, 
            ((0, max_instances - num_instances), (0, 0), (0, 0)),
            mode='constant'
        )
    
    return instance_mask

def color_mask_to_instance_mask_coco(color_mask, max_instances=20):
    """
    Optimized version for COCO-style masks
    - Only excludes black [0,0,0]
    """
    H, W, _ = color_mask.shape
    
    # Convert RGB values to integer hashes
    r = color_mask[..., 0].astype(np.int64)
    g = color_mask[..., 1].astype(np.int64)
    b = color_mask[..., 2].astype(np.int64)
    color_ints = r + (g << 8) + (b << 16)
    
    # Find unique colors (excluding black)
    unique_ints = np.unique(color_ints)
    unique_ints = unique_ints[unique_ints != 0]  # Exclude black
    unique_ints = unique_ints[:max_instances]    # Limit to max instances
    num_instances = len(unique_ints)
    
    if num_instances == 0:
        return np.zeros((max_instances, H, W), dtype=np.uint8)
    
    # Create instance masks through vectorized operations
    unique_sorted = np.sort(unique_ints)
    color_flat = color_ints.ravel()
    
    idx = np.searchsorted(unique_sorted, color_flat)
     # CRITICAL FIX: Prevent index-out-of-bounds
    valid = (idx < num_instances)
    valid[valid] = unique_sorted[idx[valid]] == color_flat[valid]
    
    labels = np.where(valid, idx, -1).reshape(H, W)
     
    
    instance_mask = (labels == np.arange(num_instances)[:, None, None]).astype(np.uint8)
    
    # Pad with zeros if needed
    if num_instances < max_instances:
        instance_mask = np.pad(
            instance_mask, 
            ((0, max_instances - num_instances), (0, 0), (0, 0)),
            mode='constant'
        )
    
    return instance_mask
def label_from_ytobj(inputmask, inputlabel, class_num=10):
     # inputmask : list contain frame_number of mask, some of them is None, some of them is of mask of size (H,W)
    # inputlabel# binary array of (10), indicateing the class of valid masks in inputmask 
    frame_number = len(inputmask)
    class_num = len(inputlabel)
    # Determine the mask shape (H, W) from the first non-None entry
    H, W = next((mask.shape for mask in inputmask if mask is not None), (0, 0))

    # Initialize the output mask array with NaNs
    mask = np.full((class_num, frame_number, H, W), np.nan, dtype=np.float32)

    # Find valid class indices from inputlabel
    class_indices = np.where(inputlabel == 1)[0]

    # Assign masks to the valid classes
    for t, frame_mask in enumerate(inputmask):
        if frame_mask is not None:
            mask[class_indices, t, :, :] = frame_mask  # Assign the mask to the active class indices

    return mask # (class_num, frame_number, H, W)
 

def label_from_endovis(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_endovis)
    new_label = inputlabel>5
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>1)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
def label_from_Miccaitest(inputlabel):  #(13,29,256,256)
    in_ch, in_D, H, W = inputlabel.shape
    inputlabel = np.transpose(inputlabel, (1, 0, 2, 3))  # Transpose dimensions
    
    lenth = len(categories_endovis)
    
    # Create new_label while preserving NaN values
    new_label = np.where(np.isnan(inputlabel), np.nan, (inputlabel > 20) * 1.0)  # Use np.where to preserve NaN
    
    # Calculate frame_label, handle NaN values by checking if NaN is in the frame
    frame_label = np.sum(new_label, axis=(2, 3))
    frame_label = np.where(np.isnan(frame_label), np.nan, (frame_label > 1) * 1.0)
    
    # Calculate video_label, handle NaN values similarly
    video_label = np.nanmax(frame_label, axis=0)  # Use nanmax to ignore NaN values
    
    # Revert the label back to the original shape
    mask = np.transpose(new_label, (1, 0, 2, 3))
    
    return mask, frame_label, video_label

def label_from_seg8k_2_cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_cholec)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label

def label_from_full_cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_cholec)
    new_label = np.zeros((in_D,lenth,H,W))
    # new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    new_label  = inputlabel # swap

    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label


def label_from_seg8k_full(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_cholec)
    new_label = np.zeros((in_D,lenth,H,W))
    # new_label[:,0,:,:] = inputlabel[:,5,:,:] # swap
    # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    new_label  = inputlabel # swap
    # new_label[:,2,:,:] = inputlabel[:,9,:,:] # swap
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
     
def label_from_thoracic(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_thoracic)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,4,:,:] # swap
    # new_label[:,0,:,:] = inputlabel[:,5,:,:]
    # new_label[:,2,:,:] = inputlabel[:,9,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
def label_from_thoracic_full(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_thoracic)
    new_label = np.zeros((in_D,lenth,H,W))
    # new_label = inputlabel  # swap
    new_label[:,0,:,:] = inputlabel[:,4,:,:]
    new_label[:,1,:,:] = inputlabel[:,3,:,:]
    new_label[:,2,:,:] = inputlabel[:,2,:,:]
    new_label[:,3,:,:] = inputlabel[:,0,:,:]

    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label
def label_from_pascal(inputlabel, instance_num=11):  # (3, frame_number, 224, 224)
    in_ch, in_D, H, W = inputlabel.shape  # 3, frame_number, 224, 224
    inputlabel = np.transpose(inputlabel, (1, 0, 2, 3))  # Frame_number, 3, 224, 224
    
    new_label = np.zeros((in_D, instance_num, H, W), dtype=np.uint8)
    
    for d in range(in_D):
        # Convert each frame's color mask to instance mask
        color_mask = inputlabel[d].transpose(1, 2, 0)  # Shape (H, W, 3)
        instance_mask = color_mask_to_instance_mask(color_mask, max_instances=instance_num)
        
        # Assign each instance to a separate channel
        new_label[d, :instance_mask.shape[0]] = instance_mask  
    
    frame_label = np.sum(new_label, axis=(2, 3))
    frame_label = (frame_label > 20).astype(np.float32)  # Threshold to determine instance presence in frame
    video_label = np.max(frame_label, axis=0)  # Aggregate presence across all frames
    mask = np.transpose(new_label, (1, 0, 2, 3))  # Transpose back to (instance_num, frame_number, H, W)
    
    return mask, frame_label, video_label

def label_from_movi(mask_og, instance_num=11):  # (3, frame_number, 224, 224)
    in_ch, in_D, H, W = mask_og.shape  # 3, frame_number, 224, 224
    mask_og = np.transpose(mask_og, (1, 0, 2, 3))  # Frame_number, 3, 224, 224
    
    new_label = np.zeros((in_D, instance_num, H, W), dtype=np.uint8)
    
    for d in range(in_D):
        # Convert each frame's color mask to instance mask
        color_mask = mask_og[d].transpose(1, 2, 0)  # Shape (H, W, 3)
        instance_mask = color_mask_to_instance_mask(color_mask, max_instances=instance_num)
        
        # Assign each instance to a separate channel
        new_label[d, :instance_mask.shape[0]] = instance_mask  
    
    frame_label = np.sum(new_label, axis=(2, 3))
    frame_label = (frame_label > 20).astype(np.float32)  # Threshold to determine instance presence in frame
    video_label = np.max(frame_label, axis=0)  # Aggregate presence across all frames
    mask = np.transpose(new_label, (1, 0, 2, 3))  # Transpose back to (instance_num, frame_number, H, W)
    
    return mask, frame_label, video_label


def label_from_coco(inputlabel,instance_num=11): #(13,29,256,256)
    in_ch, in_D, H, W = inputlabel.shape  # 3, frame_number, 224, 224
    inputlabel = np.transpose(inputlabel, (1, 0, 2, 3))  # Frame_number, 3, 224, 224
    
    new_label = np.zeros((in_D, instance_num, H, W), dtype=np.uint8)
    
    for d in range(in_D):
        # Convert each frame's color mask to instance mask
        color_mask = inputlabel[d].transpose(1, 2, 0)  # Shape (H, W, 3)
        instance_mask = color_mask_to_instance_mask_coco(color_mask, max_instances=instance_num)
        
        # Assign each instance to a separate channel
        new_label[d, :instance_mask.shape[0]] = instance_mask  
    
    frame_label = np.sum(new_label, axis=(2, 3))
    frame_label = (frame_label > 20).astype(np.float32)  # Threshold to determine instance presence in frame
    video_label = np.max(frame_label, axis=0)  # Aggregate presence across all frames
    mask = np.transpose(new_label, (1, 0, 2, 3))  # Transpose back to (instance_num, frame_number, H, W)
    
    return mask, frame_label, video_label

def label_from_thoracic2cholec(inputlabel): #(13,29,256,256)
    in_ch,in_D,H,W =  inputlabel.shape
    inputlabel=np.transpose(inputlabel , (1, 0, 2, 3)) 
    lenth = len(categories_cholec)
    new_label = np.zeros((in_D,lenth,H,W))
    new_label[:,0,:,:] = inputlabel[:,0,:,:]
    new_label[:,1,:,:] = inputlabel[:,1,:,:]
    new_label[:,2,:,:] = inputlabel[:,2,:,:]
    new_label[:,3,:,:] = inputlabel[:,3,:,:]
    new_label[:,4,:,:] = inputlabel[:,4,:,:]
    frame_label=np.sum(new_label,axis=(2,3))
    frame_label=(frame_label>20)*1.0
    video_label=np.max(frame_label, axis=0)
    mask = np.transpose(new_label , (1, 0, 2, 3)) 
    return mask,frame_label,video_label



