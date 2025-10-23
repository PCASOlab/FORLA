
import torch
import torch.nn as nn

class EnsembleViT(nn.Module):
    def __init__(self, encoders, fusion_method="concat"):
        """
        Args:
            encoders (list): List of initialized encoder models (DINO, SAM, CLIP, MAE, etc.).
            fusion_method (str): Method to combine feature maps. Options: "concat", "attention", "moe".
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fusion_method = fusion_method
        self.temperature = 0.1  # Controls binarization sharpness
        # Get feature dimensions
        # feature_dims = [enc.image_encoder.out_channels for enc in encoders]
        total_dim = 768+768+256
        
        # Define fusion layers based on method
        if fusion_method == "concat":
            self.fusion_layer =nn.Sequential(
            nn.Linear(total_dim, total_dim*2),
            nn.ReLU(),
            nn.Linear(total_dim*2, 256)
            )
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        elif fusion_method == "moe":
            self.gate = nn.Linear(768, len(encoders))
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, 256)
            )
        # Fusion components
        elif fusion_method == "moe_layer":
            # Channel-wise gating network
            self.gate = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),  # Bottleneck layer
                nn.ReLU(),
                nn.Linear(total_dim * 2, total_dim),
                # nn.Sigmoid()  # Use sigmoid for mask-like behavior
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, 256)
            )
    def gumbel_sigmoid(self, logits):
        # Add Gumbel noise to logits
        noise = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbel) / self.temperature
        
        # Differentiable binary approximation
        binary = torch.sigmoid(noisy_logits)
        return binary  
    def forward(self, pixel_values):
        """Extract and fuse backbone features from all encoders."""
        features = [enc(pixel_values)["backbone_features"] for enc in self.encoders]
        
        if self.fusion_method == "concat":
            fused_features = torch.cat(features, dim=2)
            project_features = self.fusion_layer(fused_features)
        elif self.fusion_method == "attention":
            query = features[0].flatten(2).permute(2, 0, 1)  # Take first encoder as query
            key_value = torch.cat(features[1:], dim=1).flatten(2).permute(2, 0, 1)
            fused_features, _ = self.attention(query, key_value, key_value)
            fused_features = fused_features.permute(1, 2, 0).view_as(features[0])
        elif self.fusion_method == "moe":
            # Get weights using first encoder's global features
            global_features = features[0].mean(dim=1)  # [B, D]
            weights = torch.softmax(self.gate(global_features), dim=-1)  # [B, num_encoders]
            weights = self.gumbel_sigmoid(weights)
            # Weight and concatenate features
            weighted_features = []
            for i, feat in enumerate(features):
                # Add channel dimension for broadcasting
                w = weights[:, i][:, None, None]  # [B, 1, 1]
                weighted_features.append(feat * w)
                
            fused_features = torch.cat(weighted_features, dim=2)
            project_features = self.fusion_layer(fused_features)
        elif self.fusion_method == "moe_layer":
            # Concatenate all features along channel dimension
            concated = torch.cat(features, dim=2)  # [B, N, total_dim]
            B, N, _ = concated.shape
            
            # Compute global channel weights
            global_features = concated.mean(dim=1)  # [B, total_dim]
            channel_weights = self.gate(global_features)  # [B, total_dim]
            channel_weights = self.gumbel_sigmoid(channel_weights)
            # channel_weights = torch.sigmoid(channel_weights)
            
            # Expand and apply weights
            weights = channel_weights.unsqueeze(1).expand(B, N, -1)  # [B, N, total_dim]
            fused_features = concated * weights  # Element-wise multiplication
            
            # Final projection
            project_features = self.fusion_layer(fused_features)
        
        return {"features": project_features,
                "backbone_features": fused_features}
 