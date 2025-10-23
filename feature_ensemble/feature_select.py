import torch
import os
import torch.nn as nn
from working_dir_root import Evaluation_slots,Cathe_feature_dir,Catche_epoch,Cathe_feature
import torch.nn.functional as F  # <-- Add this import
import pickle 
Feature_cath =True
# Cathe_feature_dir =  working_pcaso_raid + "MICCAI/"
# Catche_epoch=30
# Cathe_feature = True
CATCH_EPOCH_MAP = {
    'miccai': 29,
    'cholec': 29,
    'thoracic': 29,
    'ytobj': 25,
    'ytvos': 10,
    'pascal': 1,
    'coco': 1
}
class EncoderStack(nn.Module):
    """Handles multiple foundation model encoders"""
    def __init__(self, encoders):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self._feature_cache = {}  # For RAM caching
        self.cache_use_half = True

    def _get_cache_key(self, epoch, file_info):
        """Generate flat filename key"""
        dataset_source = file_info[1]
        catche_epoch = CATCH_EPOCH_MAP.get(dataset_source, 1)  # default to 1 if not found
        return f"{epoch % catche_epoch}_{file_info[0]}_{file_info[1]}"  # epoch_dataset_source_filename

        # return f"{epoch % Catche_epoch}_{file_info[0]}_{file_info[1]}"  # epoch_dataset_source_filename

    def forward(self, pixel_values, epoch=None, batch_files=None, Output_root=None):
        """Returns list of backbone features from all encoders"""
        if Cathe_feature in ["ram", "disk"]:
            Output_root = os.path.join(Cathe_feature_dir, "features")
            os.makedirs(Output_root, exist_ok=True)
            
            batch_size = pixel_values.size(0)
            cached_features = [[] for _ in self.encoders]
            uncached_indices = []
            
            # Check cache status for each item
            for idx in range(batch_size):
                file_info = batch_files[idx]  # (filename, dataset_source)
                cache_key = self._get_cache_key(epoch, file_info)
                
                if Cathe_feature == "ram":
                    if cache_key in self._feature_cache:
                        cached = self._feature_cache[cache_key]
                        for i, f in enumerate(cached):
                            cached_features[i].append(f.to(pixel_values.device).float())
                    else:
                        uncached_indices.append((idx, cache_key))
                
                elif Cathe_feature == "disk":
                    file_path = os.path.join(Output_root, f"{cache_key}.pkl")
                    try:
                        with open(file_path, 'rb') as f:
                            loaded_features = pickle.load(f)
                            for i, feat in enumerate(loaded_features):
                                cached_features[i].append(feat.to(pixel_values.device).float())
                    except:
                        uncached_indices.append((idx, cache_key))

            # Early return if all cached
            if not uncached_indices:
                return [torch.stack(feats) for feats in cached_features]

            # Process uncached items
            uncached_pixels = pixel_values[[idx for idx, _ in uncached_indices]]
            with torch.no_grad():
                uncached_feats = [enc(uncached_pixels)["backbone_features"] for enc in self.encoders]

            # Store and organize results
            for sub_idx, (idx, cache_key) in enumerate(uncached_indices):
                # Get features for this item
                item_feats = [enc_feat[sub_idx] for enc_feat in uncached_feats]
                
                if Cathe_feature == "ram":
                    self._feature_cache[cache_key] = [f.cpu().half().clone() for f in item_feats]
                elif Cathe_feature == "disk":
                    file_path = os.path.join(Output_root, f"{cache_key}.pkl")
                    with open(file_path, 'wb') as f:
                        pickle.dump([f.cpu().half() for f in item_feats], f)
                
                # Update cached features list
                for i, f in enumerate(item_feats):
                    cached_features[i].append(f.to(pixel_values.device))

            # Reconstruct full batch in original order
            final_features = []
            for i in range(len(self.encoders)):
                # Get full feature shape including temporal dimension
                feat_shape = cached_features[i][0].shape if cached_features[i] else (0,)
                
                # Create tensor with correct dimensions (batch_size, T, D)
                full_feat = torch.zeros(
                    (batch_size,) + feat_shape,
                    device=pixel_values.device
                )
                
                # Fill with cached features
                for j, feat in enumerate(cached_features[i]):
                    full_feat[j] = feat  # Now matches dimensions
                    
                final_features.append(full_feat)

            return final_features
            
        # No caching
        return [enc(pixel_values)["backbone_features"] for enc in self.encoders]
class FusionModule(nn.Module):
    """Handles different feature fusion methods"""
    def __init__(self, fusion_method, feature_dims=[768,256,768], num_encoders=3):
        super().__init__()
        self.fusion_method = fusion_method
        self.feature_dims = feature_dims
        total_dim = sum(feature_dims)
        self.slot_dim = 256
        self.total_dim = total_dim
        self.temperature = 0.1
        num_encoders = len(feature_dims)
        self.testing = Evaluation_slots
        self.norm = nn.LayerNorm(total_dim)
        self.norm2 = nn.LayerNorm(self.slot_dim)
        self.dropout = nn.Dropout(p=0.1)
        if fusion_method == "proj":
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim*2),
                nn.ReLU(),
                nn.Linear(total_dim*2, 256)
            )
            
        elif fusion_method == "attention":
            self.common_dim = 256
            self.projection_layers = nn.ModuleList([
                nn.Linear(dim, self.common_dim) for dim in feature_dims
            ])
            self.attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=8)
            
        elif fusion_method == "moe":
            # self.gate = nn.Linear(feature_dims[0], num_encoders)
            self.gate = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, num_encoders)
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, 256)
            )
        
        elif fusion_method == "moe_layer":
            self.mask_logits = nn.Parameter(torch.zeros(total_dim))  # Learnable logits

            self.gate = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, total_dim)
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_dim, total_dim * 2),
                nn.ReLU(),
                nn.Linear(total_dim * 2, 256)
            )
            
    def gumbel_softmax(self, logits):
        if self.testing:
            # Deterministic (use argmax)
          
            noisy_logits = (logits ) / self.temperature
            return F.softmax(noisy_logits, dim=-1), logits.shape[-1]
        else:
            # Add Gumbel noise
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel) / self.temperature
            return F.softmax(noisy_logits, dim=-1), logits.shape[-1]
    def gumbel_sigmoid(self, logits):
        # if self.testing:
        #     # Inference mode: deterministic
        #     noisy_logits = logits / self.temperature
        # else:
            # Training mode: keep stochasticity
        noise = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbel) / self.temperature
        mask = (torch.rand_like(noisy_logits) > 0.1).float()  # 10% of elements will be 0
        # self.testing==False
        if self.testing == False:
            # Inference mode: deterministic
            # noisy_logits = logits / self.temperature
            noisy_logits = noisy_logits * mask  # Apply mask to weights
            
        return torch.sigmoid(noisy_logits)
    def forward(self, features):
        self.pruning_loss = torch.tensor(0).cuda()
        self.pruning_loss=self.pruning_loss.to(features[0].get_device())
        if self.fusion_method == "concat":
            concated = torch.cat(features, dim=2)
            concated = self.norm(concated)
            
            proj = concated
        elif self.fusion_method == "proj":
            concated_og = torch.cat(features, dim=2)
            concated = self.norm(concated_og)
          
            proj = self.fusion_layer(concated)
            # proj = self.norm2(proj)

            
        elif self.fusion_method == "attention":
            concated = torch.cat(features, dim=2)

            projected_features = [proj(feat) for proj, feat in zip(self.projection_layers, features)]
            # Concatenate projected features[1:] along the sequence dimension (dim=1)
            query = projected_features[0].flatten(2).permute(1, 0, 2)
            key_value = torch.cat([feat.flatten(2) for feat in projected_features[1:]], dim=1).permute(1, 0, 2)
            fused, _ = self.attention(query, key_value, key_value)
            proj = fused.permute(1,0 , 2) 
            proj = self.norm2(proj)

            
        elif self.fusion_method == "moe":
            concated_og = torch.cat(features, dim=2)
            concated = self.norm(concated_og)

            B, N, _ = concated.shape
            gate_out = self.gate(concated.mean(dim=1))
            weights = torch.sigmoid(gate_out) 
            # if weights[0,0]<0.5:
            #     print('no')
            weighted = [f * weights[:, i][:, None, None] for i, f in enumerate(features)]
            fused = torch.cat(weighted, dim=2)
            proj = self.fusion_layer(fused)
            proj = self.norm2(proj)
            mean_weight= weights.mean()   # Compute pruning loss

            self.pruning_loss = mean_weight + \
                         0.2 * F.relu(0.3 - mean_weight)  # Hinge loss

            
        elif self.fusion_method == "moe_layer":
            concated_og = torch.cat(features, dim=2)
            concated = self.norm(concated_og)
            # proj = concated
            B, N, _ = concated.shape
            gate_out = self.gate(concated.mean(dim=1))
            weights =  torch.sigmoid(gate_out).unsqueeze(1).expand(B, N, -1)
            # channel_weights = self.gumbel_sigmoid(self.mask_logits)  # [total_dim]
            # weights = channel_weights.view(1, 1, -1).expand(B, N, -1)
            fused = concated * weights
            mean_weight= weights.mean()   # Compute pruning loss
            # self.pruning_loss = mean_weight if mean_weight > 0.3 else 0.0 
            self.pruning_loss = mean_weight + \
                         0.2 * F.relu(0.3 - mean_weight)  # Hinge loss
            # fused = concated 

            proj = self.fusion_layer(fused)
            # proj = self.norm2(proj)
            # proj = self.fusion_layer(concated)

 
        return {"features": proj, "backbone_features": concated_og}