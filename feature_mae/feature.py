import torch
import torch.nn as nn
from transformers import ViTModel, ViTImageProcessor

class MAEViT(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", device=None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the ViT model and its preprocessing function
        self.model = ViTModel.from_pretrained(model_name).to(self.device).eval()
        self.preprocess = ViTImageProcessor.from_pretrained(model_name)
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Define an MLP to project features
        # Input dimension is 768 (ViT embedding dimension),
        # intermediate dimension is 1536 (double of 768),
        # and output dimension is 64.
        self.output_transform = nn.Sequential(
            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Linear(1536, 64)
        )
    
    def forward(self, pixel_values):
        """
        Expects pixel_values to be a normalized tensor of shape [B, 3, H, W].
        Returns a dict with:
            - "features": the projected features reshaped as [B, num_patches, 64]
            - "backbone_features": the raw features from the MAE encoder reshaped as [B, num_patches, 768]
        """
        # pixel_values is assumed to be already on self.device and normalized.
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            # outputs.last_hidden_state shape: [B, num_patches + 1, 768]
            backbone_features = outputs.last_hidden_state
        
        # Remove the class token (first token)
        backbone_features = backbone_features[:, 1:, :]  # shape: [B, num_patches, 768]
        
        # Project the backbone features with the MLP
        project_features = self.output_transform(backbone_features)  # shape: [B, num_patches, 64]
        
        return {
            "features": project_features,
            "backbone_features": backbone_features,
        }

# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate the encoder module
    encoder = MAEViT(model_name="facebook/vit-base-patch16-224", device=device)
    
    # Load and preprocess an image
    from PIL import Image
    image = Image.open("path_to_your_image.jpg")
    
    # Extract features
    with torch.no_grad():
        outputs = encoder(image)
        print("Projected features shape:", outputs["features"].shape)
        print("Backbone features shape:", outputs["backbone_features"].shape)
