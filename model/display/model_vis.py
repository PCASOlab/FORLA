import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from visdom import Visdom
import torch.nn.functional as F
import cv2
import os
from PIL import Image
import matplotlib.cm as cm
# Initialize Visdom
viz = Visdom(port=8091)

def plot_all_frames_on_projection_vizdom_2d(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on a 2D plane using Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = reduced_vectors[:, 0]
    Y = reduced_vectors[:, 1]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "PCA 1",
        "ylabel": "PCA 2",
        "title": f"Latent Vectors in 2D Plane (All Frames Combined)"
    }
    viz.scatter(
        X=np.column_stack((X, Y)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
 
def plot_all_frames_on_hypersphere_vizdom2(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on the same hypersphere using Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 3D using PCA
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Project onto a unit sphere
    norms = np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
    sphere_vectors = reduced_vectors / norms  # Normalize to project onto the sphere

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = sphere_vectors[:, 0]
    Y = sphere_vectors[:, 1]
    Z = sphere_vectors[:, 2]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "Sphere X",
        "ylabel": "Sphere Y",
        "zlabel": "Sphere Z",
        "title": "Latent Vectors on a 3D Sphere (All Frames Combined)"
    }
    viz.scatter(
        X=np.column_stack((X, Y, Z)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
def plot_all_frames_on_projection_vizdom_2d_tsne(batch_frames_matrix):
    """
    Visualize latent vectors from all frames on a 2D plane using t-SNE and Visdom.

    Parameters:
        batch_frames_matrix: Tensor of shape (Batch, Frames, M, N)
    """
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = reduced_vectors[:, 0]
    Y = reduced_vectors[:, 1]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "t-SNE Dim 1",
        "ylabel": "t-SNE Dim 2",
        "title": "Latent Vectors in 2D (t-SNE)"
    }
    viz.scatter(
        X=np.column_stack((X, Y)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
def plot_all_frames_on_hypersphere_vizdom_tsne(batch_frames_matrix):
    
    batch_size, num_frames, M, N = batch_frames_matrix.size()

    # Select the first batch
    vectors = batch_frames_matrix[0]  # Shape (Frames, M, N)
    
    # Reshape to combine all frames
    reshaped_vectors = vectors.view(-1, N)  # Shape (Frames*M, N)

    # Normalize vectors to lie on the unit hypersphere
    normalized_vectors = reshaped_vectors / reshaped_vectors.norm(dim=1, keepdim=True)

    # Reduce dimensions to 3D using t-SNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(normalized_vectors.detach().cpu().numpy())

    # Project onto a unit sphere
    norms = np.linalg.norm(reduced_vectors, axis=1, keepdims=True)
    sphere_vectors = reduced_vectors / norms  # Normalize to project onto the sphere

    # Prepare colors: Each vector index across frames gets a unique color
    colors = np.arange(1, M + 1)  # Start labels from 1
    frame_colors = np.tile(colors, num_frames)  # Repeat colors for each frame

    # Prepare data for Visdom scatter plot
    X = sphere_vectors[:, 0]
    Y = sphere_vectors[:, 1]
    Z = sphere_vectors[:, 2]

    # Create scatter plot in Visdom
    opts = {
        "markersize": 5,
        "legend": [f"Slot {i+1}" for i in range(M)],
        "xlabel": "t-SNE Dim 1",
        "ylabel": "t-SNE Dim 2",
        "zlabel": "t-SNE Dim 3",
        "title": "Latent Vectors on 3D Sphere (t-SNE)"
    }
    viz.scatter(
        X=np.column_stack((X, Y, Z)),
        Y=frame_colors,  # Colors now start from 1
        opts=opts,
    )
if __name__ == '__main__':
    # Example usage
    Batch, Frames, M, N = 1, 5, 10, 50  # One batch, 5 frames, 10 vectors per frame, vector size 50
    batch_frames_matrix = torch.randn(Batch, Frames, M, N)

    # Plot all frames' vectors on the same hypersphere
    plot_all_frames_on_hypersphere_vizdom_tsne(batch_frames_matrix)

import math
def plot_PCA_per_batch(images,feature):
    images = images[:,:,0,:,:]
    patch_embeddings  = feature[:,0,:,:]
    # Infer dimensions from features
    batch_size, num_patches, embed_dim = patch_embeddings.shape
    grid_size = int(math.sqrt(num_patches))

    # Global PCA computation
    all_patches = patch_embeddings.reshape(-1, embed_dim)
    global_mean = all_patches.mean(dim=0, keepdim=True)
    all_patches_centered = all_patches - global_mean

    U, S, V = torch.linalg.svd(all_patches_centered, full_matrices=False)
    
    # Get first three principal components
    pc1 = V[0, :]
    pc2 = V[1, :]
    pc3 = V[2, :]

    # Project and normalize each component
    projections = []
    for pc in [pc1, pc2, pc3]:
        proj = torch.matmul(all_patches_centered, pc.unsqueeze(-1)).squeeze()
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min())
        projections.append(proj_norm.view(batch_size, grid_size, grid_size))

    # Visualization parameters
    upscale_factor = 224 // grid_size

    for i in range(batch_size):
    # for i in range(1):
    
        # Original image
        img = images[i].cpu().float()
        
        # Create RGB PCA map
        pca_maps = []
        for j in range(3):
            # Upsample each component
            upsampled = torch.repeat_interleave(
                torch.repeat_interleave(projections[j][i], upscale_factor, dim=0),
                upscale_factor, dim=1
            )
            pca_maps.append(upsampled)
        
        # Combine into RGB image
        rgb_pca = torch.stack(pca_maps, dim=0)  # Shape [3, H, W]
        
        # Visualization
        viz.image(
            img,
            win=f"image_{i}_original",
            opts={'title': f'Image {i+1} - Original', 'width': 300, 'height': 300}
        )
        
        viz.image(
            rgb_pca,
            win=f"image_{i}_pca_rgb",
            opts={
                'title': f'Image {i+1} - PCA RGB',
                'width': 300,
                'height': 300
            }
        )
def plot_patch_pca_per_image(images, features,title="PCA"):
    images = images[:, :, 0, :, :]
    patch_embeddings = features[:, 0, :, :]
    
    batch_size, num_patches, embed_dim = patch_embeddings.shape
    grid_size = int(math.sqrt(num_patches))
    upscale_factor = 224 // grid_size
    kernel_size = upscale_factor * 2 + 1

    # for i in range(batch_size):
    for i in range(1):

        img = images[i].cpu().float()
        patches = patch_embeddings[i]

        # Get device from patches tensor
        device = patches.device
        
        # Per-image PCA computation
        mean = patches.mean(dim=0, keepdim=True)
        patches_centered = patches - mean
        
        U, S, Vh = torch.linalg.svd(patches_centered, full_matrices=False)
        pc1, pc2, pc3 = Vh[0], Vh[1], Vh[2]

        projections = [
            torch.matmul(patches_centered, pc).view(grid_size, grid_size)
            for pc in [pc1, pc2, pc3]
        ]

        # Fixed device handling in normalization
        def percentile_normalize(x):
            x_flat = x.flatten()
            # Create quantile tensor on same device as input
            quantiles = torch.tensor([0.1, 0.9], device=x.device)
            q1, q9 = torch.quantile(x_flat, quantiles)
            return torch.clamp((x - q1) / (q9 - q1 + 1e-6), 0, 1)

        normalized_projections = [percentile_normalize(p) for p in projections]

        # Rest of the code remains the same...
        channel_maps = []
        for proj in normalized_projections:
            upsampled = F.interpolate(
                proj.unsqueeze(0).unsqueeze(0),
                scale_factor=upscale_factor,
                mode='bicubic',
                align_corners=False
            ).squeeze()
            
            # smoothed = F.gaussian_blur(
            #     upsampled.unsqueeze(0).unsqueeze(0),
            #     kernel_size=kernel_size,
            #     sigma=3
            # ).squeeze()
            smoothed = upsampled
            channel_maps.append(smoothed)

        rgb_pca = torch.stack(channel_maps, dim=0)
        rgb_pca = torch.clamp(rgb_pca, 0, 1) ** 0.6

        # Visualization code remains unchanged...
        # np.transpose(combine_stack_matched.astype((np.uint8)), (2, 0, 1))
        # Convert and process original image
        img_np = img.numpy()
        
        # 1. Handle channel order and BGR conversion
        # Transpose from (C, H, W) to (H, W, C) for OpenCV
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Scale if needed before type conversion
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
            
        # Convert RGB to BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Transpose back to (C, H, W) for visualization
        img_np = np.transpose(img_np, (2, 0, 1))
        # img_uint8 = (img * 255).clamp(0, 255).byte()  # Convert to 0-255 and uint8
        # img_np =  cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        viz.image(
            img_np,
            # np.transpose(img_np.astype((np.uint8)), (2, 0, 1)),
            win=f"image_{i}_original",
            opts={'title': f'Image {i+1} - Original', 'width': 300, 'height': 300}
        )
        viz.image(
            rgb_pca,
            win=f"image_{i}_pca_rgb"+title,
            opts={'title': f'Image {i+1} - Patch PCA'+title, 'width': 300, 'height': 300}
        )

def visdom_cosine_heatmap(slots, title="Slot Cosine Similarity"):
    """
    Args:
        slots: Tensor of shape (B, T, K, D)
        title: Base title for Visdom window
    """
    B, T, K, D = slots.shape

    for i in range(min(1, B)):  # Show up to 1 samples
        for t in range(min(3, T)):  # Show up to 2 frames
            with torch.no_grad():
                slot_matrix = slots[i, t]  # (K, D)
                slot_matrix = F.normalize(slot_matrix, dim=-1)  # (K, D)
                sim_matrix = torch.mm(slot_matrix, slot_matrix.T)  # (K, K)

                viz.heatmap(
                    X=sim_matrix.cpu(),
                    win=f"cosine_sim_b{i}_f{t}",
                    opts=dict(
                        colormap='Spectral',
                        title=f"{title} | B{i} F{t}",
                        xlabel="Slot Index",
                        ylabel="Slot Index",
                        xmin = -1,
                        xmax = 1
                    )
                )
def save_cosine_similarity_maps(
    slots,
    output_root,
    read_id,
    make_png=True,
    png_upscale=16,      # 16x upscaling so small K looks readable
    vmin=-1.0, vmax=1.0, # cosine range; keep consistent across frames
    csv_fmt="%.6f"
):
    """
    Args:
        slots: Tensor (B, T, K, D)
        output_root: base output directory
        read_id: subfolder name under similarity_map
        make_png: if True, also save PNG heatmaps (headless)
        png_upscale: integer upscale factor for PNG readability
        vmin, vmax: clamp range for color mapping
        csv_fmt: format for CSV values

    Outputs:
        CSV: {output_root}/similarity_map/{read_id}/sim_b{b}_f{t:03d}.csv
        PNG: {output_root}/similarity_map/{read_id}/sim_b{b}_f{t:03d}.png
    """
    assert slots.ndim == 4, f"Expected (B, T, K, D), got {slots.shape}"
    B, T, K, D = slots.shape

    save_dir = os.path.join(output_root, "similarity_map", str(read_id))
    os.makedirs(save_dir, exist_ok=True)

    # Prepare colormap (viridis) once (256 steps)
    viridis = cm.get_cmap("hsv", 256)

    with torch.no_grad():
        # Normalize features along D once per (B, T)
        slots_n = F.normalize(slots, dim=-1)  # (B, T, K, D)

        for b in range(B):
            for t in range(T):
                sm = torch.mm(slots_n[b, t], slots_n[b, t].T).clamp(vmin, vmax)  # (K, K)

                # ----- Save CSV -----
                csv_path = os.path.join(save_dir, f"sim_b{b}_f{t:03d}.csv")
                np.savetxt(csv_path, sm.cpu().numpy(), delimiter=",", fmt=csv_fmt)

                if make_png:
                    # Map to [0, 255] index for colormap
                    arr = sm.cpu().numpy()
                    arr01 = (arr - vmin) / (vmax - vmin)  # [0,1]
                    # arr01 = np.clip(arr01, 0.0, 1.0)
                    arr01 = -arr01
                    idx = (arr01 * 255.0).round().astype(np.uint8)

                    # Apply colormap → RGBA, then drop alpha
                    rgba = viridis(idx)  # shape (K, K, 4), float in [0,1]
                    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

                    # Make image and upscale for readability
                    im = Image.fromarray(rgb, mode="RGB")
                    if png_upscale and png_upscale > 1:
                        im = im.resize((K * png_upscale, K * png_upscale), resample=Image.NEAREST)

                    png_path = os.path.join(save_dir, f"sim_b{b}_f{t:03d}.png")
                    im.save(png_path, format="PNG")