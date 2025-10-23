import torch
import torch.nn as nn
import clip
from PIL import Image

class CLIPViT(nn.Module):
    def __init__(self, model_name="ViT-B/16", device=None):
        super().__init__()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.output_transform = nn.Sequential(
            nn.Linear(768, 1536),
            nn.ReLU(),
            nn.Linear(1536, 64)
        )
    
    def forward(self, image):
        # Move image to correct device and convert to half-precision
        image = image.half()

        with torch.no_grad():
            visual = self.model.visual
            x = visual.conv1(image)
            B, C, H_patch, W_patch = x.shape
            x = x.reshape(B, C, H_patch * W_patch).permute(0, 2, 1)

            class_token = visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
            x = torch.cat([class_token, x], dim=1)
            
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)

            for block in visual.transformer.resblocks:
                x = block(x)
            
            # Apply final layer norm (ln_post) to all tokens
            x = visual.ln_post(x)
            backbone_features = x[:, 1:, :]  # Remove class token
            
            grid_size = int(backbone_features.shape[1] ** 0.5)
            backbone_features = backbone_features.reshape(B, grid_size, grid_size, -1)
            backbone_features = backbone_features.reshape(B, grid_size * grid_size, -1)
        
        # Project features
        backbone_features = backbone_features.float()
        project_feature = self.output_transform(backbone_features)
        
        return {
            "features": project_feature,
            "backbone_features": backbone_features,
            "vit_block12": backbone_features,  # Adjust if using intermediate blocks
        }

# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Instantiate the encoder module
    encoder = CLIPViT(model_name="ViT-B/16", device=device)
    
    # Load and preprocess an image using the CLIP preprocess function
    image = encoder.preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    image = image.half()  # Convert image to half precision to match model weights

    # Extract features
    with torch.no_grad():
        outputs = encoder(image)
        print("Projected features shape:", outputs["features"].shape)
        print("Backbone features shape:", outputs["backbone_features"].shape)
