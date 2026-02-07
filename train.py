"""Training script for HeteroFloodGNN with autoregressive forecasting.

This script implements:
1. Teacher forcing with scheduled decay
2. Standardized RMSE loss
3. Autoregressive prediction during validation/test
4. Checkpointing and logging
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple
import logging
from tqdm import tqdm
from pathlib import Path

from dataset import UrbanFloodDataset
from model import HeteroFloodGNN
from config import ModelConfig, TrainingConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandardizedRMSELoss(nn.Module):
    """Compute Standardized RMSE loss for predictions.
    
    RMSE is computed separately for each node type and normalized by
    pre-computed standard deviations.
    """
    
    def __init__(self, std_manhole: float, std_cell: float):
        super().__init__()
        self.std_manhole = std_manhole
        self.std_cell = std_cell
    
    def forward(self, pred_dict: Dict[str, torch.Tensor], 
                target_dict: Dict[str, torch.Tensor],
                mask_dict: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_dict: {'manhole': [N1, 1], 'cell': [N2, 1]}
            target_dict: {'manhole': [N1, 1], 'cell': [N2, 1]}
            mask_dict: Optional masks for valid nodes
        
        Returns:
            Averaged standardized RMSE across both node types
        """
        losses = []
        
        # Manhole RMSE
        pred_man = pred_dict['manhole'].squeeze(-1)
        target_man = target_dict['manhole'].squeeze(-1)
        if mask_dict and 'manhole' in mask_dict:
            mask = mask_dict['manhole']
            pred_man = pred_man[mask]
            target_man = target_man[mask]
        
        rmse_man = torch.sqrt(torch.mean((pred_man - target_man) ** 2))
        std_rmse_man = rmse_man / self.std_manhole
        losses.append(std_rmse_man)
        
        # Cell RMSE
        pred_cell = pred_dict['cell'].squeeze(-1)
        target_cell = target_dict['cell'].squeeze(-1)
        if mask_dict and 'cell' in mask_dict:
            mask = mask_dict['cell']
            pred_cell = pred_cell[mask]
            target_cell = target_cell[mask]
        
        rmse_cell = torch.sqrt(torch.mean((pred_cell - target_cell) ** 2))
        std_rmse_cell = rmse_cell / self.std_cell
        losses.append(std_rmse_cell)
        
        # Average across node types (equal weight)
        return torch.stack(losses).mean()


def compute_std_from_events(dataset: UrbanFloodDataset, event_list: list, device: str) -> Tuple[float, float]:
    """Compute standard deviations of water levels across training events."""
    logger.info("Computing standard deviations from training data...")
    
    manhole_levels = []
    cell_levels = []
    
    # 【注意】这里不需要 data = dataset.get(0)，因为我们只用 dataset.load_event
    
    for event_name in tqdm(event_list, desc="Loading events for std"):
        # 【修复】使用 dataset.load_event
        event_data = dataset.load_event(event_name)
        
        # manhole_dyn: [T, N1, 2] where [:, :, 0] is water_level
        manhole_levels.append(event_data['manhole'][:, :, 0].reshape(-1).numpy())
        
        # cell_dyn: [T, N2, 3] where [:, :, 1] is water_level
        cell_levels.append(event_data['cell'][:, :, 1].reshape(-1).numpy())
    
    std_manhole = float(np.concatenate(manhole_levels).std())
    std_cell = float(np.concatenate(cell_levels).std())
    
    logger.info(f"Computed std_manhole={std_manhole:.4f}, std_cell={std_cell:.4f}")
    
    return std_manhole, std_cell

def get_teacher_forcing_ratio(epoch: int, config: TrainingConfig) -> float:
    """Compute teacher forcing ratio with linear decay."""
    if epoch >= config.teacher_forcing_decay_epochs:
        return config.teacher_forcing_ratio_end
    
    progress = epoch / config.teacher_forcing_decay_epochs
    ratio = config.teacher_forcing_ratio_start + progress * (
        config.teacher_forcing_ratio_end - config.teacher_forcing_ratio_start
    )
    return ratio


def train_one_event(model: HeteroFloodGNN, data, event_data: Dict, 
                   criterion: nn.Module, device: str,
                   teacher_forcing_ratio: float = 1.0) -> float:
    """Train on one event with autoregressive teacher forcing.
    
    Args:
        model: HeteroFloodGNN model
        data: HeteroData with static features
        event_data: Dict with 'manhole' [T, N1, D1], 'cell' [T, N2, D2]
        criterion: Loss function
        device: torch device
        teacher_forcing_ratio: Probability of using ground truth
    
    Returns:
        Average loss for this event
    """
    model.train()
    
    manhole_seq = event_data['manhole'].to(device)  # [T, N1, D1]
    cell_seq = event_data['cell'].to(device)  # [T, N2, D2]
    T = manhole_seq.shape[0]
    
    total_loss = 0.0
    h_dict = None
    
    # Autoregressive loop: predict t+1 given 0..t
    for t in range(T - 1):
        # Current timestep features
        manhole_dyn_t = manhole_seq[t]  # [N1, D1]
        cell_dyn_t = cell_seq[t]  # [N2, D2]
        
        # Predict next timestep
        pred_dict, h_dict = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
        
        # Ground truth for t+1
        target_dict = {
            'manhole': manhole_seq[t+1, :, 0:1],  # water_level is first feature
            'cell': cell_seq[t+1, :, 1:2]  # water_level is second feature for cells
        }
        
        # Compute loss
        loss = criterion(pred_dict, target_dict)
        total_loss += loss.item()
        
        # Backprop per step
        loss.backward()
        
        # 【关键修复】Detach hidden state to prevent backpropagating through time indefinitely
        # This cuts the graph connection to the previous timestep
        h_dict = {k: v.detach() for k, v in h_dict.items()}
        
        # Teacher forcing: decide whether to use ground truth or prediction for next step
        with torch.no_grad():
            use_teacher = np.random.rand() < teacher_forcing_ratio
            
            if use_teacher:
                # Use ground truth for next iteration
                pass  # manhole_seq[t+1] already has ground truth
            else:
                # Use model prediction for next iteration
                # Update the water_level feature in the sequence
                # .detach() here prevents gradient flow through the input features of the next step
                manhole_seq[t+1, :, 0] = pred_dict['manhole'].squeeze(-1).detach()
                cell_seq[t+1, :, 1] = pred_dict['cell'].squeeze(-1).detach()
    
    return total_loss / (T - 1)


@torch.no_grad()
def validate_one_event(model: HeteroFloodGNN, data, event_data: Dict,
                      criterion: nn.Module, device: str) -> float:
    """Validate on one event with full autoregressive prediction.
    
    No teacher forcing - always use model predictions.
    """
    model.eval()
    
    manhole_seq = event_data['manhole'].to(device)
    cell_seq = event_data['cell'].to(device)
    T = manhole_seq.shape[0]
    
    total_loss = 0.0
    h_dict = None
    
    # Use ground truth for the first step only
    manhole_dyn_t = manhole_seq[0].clone()
    cell_dyn_t = cell_seq[0].clone()
    
    for t in range(T - 1):
        pred_dict, h_dict = model(data, manhole_dyn_t, cell_dyn_t, h_dict)
        
        target_dict = {
            'manhole': manhole_seq[t+1, :, 0:1],
            'cell': cell_seq[t+1, :, 1:2]
        }
        
        loss = criterion(pred_dict, target_dict)
        total_loss += loss.item()
        
        # Always use predictions for next step (autoregressive)
        manhole_dyn_t = manhole_seq[t+1].clone()
        cell_dyn_t = cell_seq[t+1].clone()
        manhole_dyn_t[:, 0] = pred_dict['manhole'].squeeze(-1)
        cell_dyn_t[:, 1] = pred_dict['cell'].squeeze(-1)
    
    return total_loss / (T - 1)


def train(model_config: ModelConfig, train_config: TrainingConfig):
    """Main training loop."""
    
    # Setup
    device = torch.device(train_config.device)
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset for Model {train_config.model_id}...")
    dataset = UrbanFloodDataset(root="./", model_id=train_config.model_id)
    data = dataset.get(0)
    
    # Compute standard deviations if needed
    if train_config.use_standardized_rmse:
        std_manhole, std_cell = compute_std_from_events(
            dataset, train_config.train_events, device
        )
        train_config.std_manhole = std_manhole
        train_config.std_cell = std_cell
    
    # Initialize model
    logger.info("Initializing model...")
    model = HeteroFloodGNN(
        config=model_config,
        manhole_static_dim=4,  # depth, invert, surface, area
        cell_static_dim=6,  # area, rough, min_elev, elev, aspect, curv
        manhole_dynamic_dim=2,  # water_level, inlet_flow
        cell_dynamic_dim=3  # rainfall, water_level, water_volume
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, 
                     weight_decay=train_config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.num_epochs)
    
    # Loss function
    criterion = StandardizedRMSELoss(train_config.std_manhole, train_config.std_cell)
    
    # Move data to device
    data = data.to(device)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(train_config.num_epochs):
        # Get teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(epoch, train_config)
        
        # Train
        model.train()
        train_losses = []
        
        for event_name in tqdm(train_config.train_events, desc=f"Epoch {epoch+1}/{train_config.num_epochs}"):
            event_data = dataset.load_event(event_name)
            
            optimizer.zero_grad()
            loss = train_one_event(model, data, event_data, criterion, device, tf_ratio)
            train_losses.append(loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        val_losses = []
        for event_name in train_config.val_events:
            event_data = dataset.load_event(event_name)
            val_loss = validate_one_event(model, data, event_data, criterion, device)
            val_losses.append(val_loss)
        
        avg_val_loss = np.mean(val_losses)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                   f"Val Loss={avg_val_loss:.4f}, TF Ratio={tf_ratio:.2f}, "
                   f"LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(train_config.checkpoint_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': model_config,
                'train_config': train_config,
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        
        if (epoch + 1) % train_config.save_every_n_epochs == 0:
            checkpoint_path = Path(train_config.checkpoint_dir) / f"model_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    # Example configuration
    model_config = ModelConfig(
        hidden_dim=128,
        num_gnn_layers=3,
        num_recurrent_steps=3,
        use_gatv2=True,
        num_heads=4,
        dropout=0.1
    )
    
    train_config = TrainingConfig(
        model_id=1,
        learning_rate=1e-3,
        num_epochs=16,
        teacher_forcing_ratio_start=1.0,
        teacher_forcing_ratio_end=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    train(model_config, train_config)
