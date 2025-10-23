import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import scipy.optimize

import torch.nn.functional as F
from dataset.dataset import myDataloader
import model.base_models2 as block_buider
from dataset.dataset import Obj_num, Seperate_LR
import random
from working_dir_root import Evaluation,Fintune,Enable_student
# Seperate_LR = True # seperate left and right
from model import model_operator 
class MLPClassifier(nn.Module):
    def __init__(self, feature_dim, category_number, hidden_dim=None):
        super(MLPClassifier, self).__init__()
        
        # Define the hidden layer if needed, otherwise it's just a linear layer
        if hidden_dim:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(hidden_dim/2)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
                # nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(int(hidden_dim/4), category_number)
            )
        else:
            self.mlp = nn.Linear(feature_dim, category_number)
        
    def forward(self, x):
        # x is of shape [batch, slots_number, feature_dim]
        # Apply the MLP to each slot independently
        x = self.mlp(x)
        # activation = nn.Sigmoid()
        # x = activation(x)
        x = F.softmax(x, dim=-1)
        # x is now of shape [batch, slots_number, category_number]
        return x
    
def hungarian_huber_loss_gpt(x, y, delta=1.0):
    """
    Huber loss for sets, matching elements with the Hungarian algorithm.

    Args:
        x: Batch of sets of size [batch_size, n_points, dim_points]. Each set in the
           batch contains n_points many points, each represented as a vector of
           dimension dim_points.
        y: Batch of sets of size [batch_size, n_points, dim_points].
        delta: The point where the Huber loss function changes from a quadratic to linear.

    Returns:
        Average distance between all sets in the two batches.
    """
    batch_size, n_points, dim_points = x.size()

    # Compute pairwise Huber loss
    x_expanded = x.unsqueeze(2)  # [batch_size, n_points, 1, dim_points]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, n_points, dim_points]

    diff = x_expanded - y_expanded  # [batch_size, n_points, n_points, dim_points]
    abs_diff = torch.abs(diff)
    quadratic = torch.minimum(abs_diff, torch.tensor(delta, device=x.device))
    linear = abs_diff - quadratic
    pairwise_cost = 0.5 * quadratic**2 + delta * linear  # [batch_size, n_points, n_points, dim_points]
    pairwise_cost = pairwise_cost.sum(dim=-1)  # [batch_size, n_points, n_points]

    # Detach only before converting to numpy
    indices = [scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy()) for cost_matrix in pairwise_cost]
    indices = np.array(indices)
    
    actual_costs = []
    for b in range(batch_size):
        row_indices, col_indices = indices[b]
        actual_costs.append(pairwise_cost[b, row_indices, col_indices].sum())

    # Convert the list of costs back to a tensor without detaching
    actual_costs = torch.stack(actual_costs).to(x.device)
    return actual_costs.mean()

class HungarianHuberLoss(nn.Module):
    def __init__(self):
        super(HungarianHuberLoss, self).__init__()

    def forward(self, x, y):
        """
        Huber loss for set prediction using the Hungarian algorithm.

        Args:
            x: Predicted tensor of shape [batch_size, 7, 13].
            y: Ground truth tensor of shape [batch_size, n_target_slots, 13].

        Returns:
            Scalar loss value.
        """
        batch_size = x.size(0)
        n_slots = x.size(1)

        # Expand dimensions for pairwise comparison
        y_expanded = y.unsqueeze(2)  # Shape: [batch_size, n_target_slots, 1, 13]
        x_expanded = x.unsqueeze(1)  # Shape: [batch_size, 1, n_slots, 13]

        # Compute pairwise Huber loss
        pairwise_cost = F.smooth_l1_loss(x_expanded, y_expanded, reduction='none')  # Shape: [batch_size, n_target_slots, n_slots, 13]
        pairwise_cost = pairwise_cost.sum(dim=-1)  # Sum over the feature dimension: Shape: [batch_size, n_target_slots, n_slots]

        total_loss = 0
        for i in range(batch_size):
            cost_matrix = pairwise_cost[i].detach().cpu().numpy()  # Convert to numpy for scipy
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve the assignment problem

            total_loss += pairwise_cost[i, row_ind, col_ind].sum()  # Sum the matching costs

        return total_loss / batch_size


# Define batch size, number of elements in predictions and targets
 