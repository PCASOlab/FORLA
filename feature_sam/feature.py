import torch
import torch.nn as nn
# Import SAM modules (ensure the SAM library is in your PYTHONPATH)
from SAM.segment_anything import SamPredictor, sam_model_registry

from working_dir_root import learningR,learningR_res,SAM_pretrain_root
class SAMViT(nn.Module):
    def __init__(self, model_type="vit_b", checkpoint_path="path_to_your_sam_checkpoint.pth", device=None):
        super().__init__()
        # Determine device (GPU or CPU)
        sam_checkpoint =SAM_pretrain_root+ "sam_vit_b_01ec64.pth"

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SAM model and extract its image encoder
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_encoder = self.sam.image_encoder.to(self.device).eval()
        
        # Freeze the encoder parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Define an MLP to project features.
        # For vit_b, the encoder dimension is 768. Adjust these values if using a different model_type.
        self.output_transform = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )
    
    def forward(self, pixel_values):
        """
        Expects pixel_values to be a normalized tensor of shape [B, 3, H, W].
        Returns a dict with:
            - "features": the projected features reshaped as [B, num_patches, 64]
            - "backbone_features": the raw features from the SAM image encoder reshaped as [B, num_patches, 768]
        """
        with torch.no_grad():
            # Pass the input image through the SAM image encoder.
            # The encoder is expected to output a tensor of shape [B, num_tokens, feature_dim]
            encoder_outputs = self.image_encoder(pixel_values)
            # Remove the class token (assumed to be the first token)
            backbone_features = encoder_outputs.permute(0, 2, 3, 1).reshape(
        encoder_outputs.shape[0], 
        encoder_outputs.shape[2]*encoder_outputs.shape[3],  # Automatically computes w*h
        encoder_outputs.shape[1]  # 256
    )
        
        # Project the backbone features using the MLP
        project_features = self.output_transform(backbone_features)  # shape: [B, num_patches, 64]
        
        return {
            "features": project_features,
            "backbone_features": backbone_features,
        }

# Example usage:
if __name__ == "__main__":
    # Set device and checkpoint path accordingly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "path_to_your_sam_checkpoint.pth"  # e.g., SAM_pretrain_root + "sam_vit_b_01ec64.pth"
    
    # Instantiate the SAM feature extractor module.
    encoder = SAMViT(model_type="vit_b", checkpoint_path=checkpoint_path, device=device)
    
    # Load and preprocess an image.
    # (Note: SAM might require specific preprocessing; adjust the transforms if needed.)
    from PIL import Image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if SAM's image encoder requires it.
    ])
    image = Image.open("path_to_your_image.jpg")
    pixel_values = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        outputs = encoder(pixel_values)
        print("Projected features shape:", outputs["features"].shape)
        print("Backbone features shape:", outputs["backbone_features"].shape)
