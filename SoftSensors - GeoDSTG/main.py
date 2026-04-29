"""
Soft Sensor Modeling for SRU Process using GeoDSTG
==================================================
"""

import os
import time
import random
import warnings
import logging
import copy
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Local imports
from models import GeoDSTG


class Config:
    
    # Data parameters
    DATA_PATH = "./Data/SRU_data.txt"
    HISTORY_STEPS = 15
    BATCH_SIZE = 32
    
    # Model parameters
    HIDDEN_SIZE = 64
    NUM_HEADS = 4
    DROPOUT = 0.2
    NUM_NODES = 6
    
    # Training parameters
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    PATIENCE = 20
    # KNN parameters
    KNN_K = 4  # Optimal parameter from original paper
    
    # Random seed for reproducibility
    # 112, 113, 114, 115, 116, 117, 118, 123, 329, 331
    SEED = 331


def setup_environment() -> str:
    # Suppress warnings
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Setup logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    
    return device


def seed_everything(seed: int = Config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)


class KNNGraphBuilder:
    """K-Nearest Neighbors graph construction strategy."""
    
    def __init__(self, k: int = Config.KNN_K, device: str = 'cpu'):
        self.k = k
        self.device = device
    
    def build_graph(self, node_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to numpy for KNN calculation
        features_np = node_features.cpu().numpy()
        num_nodes = features_np.shape[0]
        
        # Use KNN to find nearest neighbors
        try:
            knn = NearestNeighbors(n_neighbors=min(self.k + 1, num_nodes), 
                                 metric='euclidean', algorithm='ball_tree')
        except:
            knn = NearestNeighbors(n_neighbors=min(self.k + 1, num_nodes), 
                                 metric='euclidean', algorithm='brute')
        
        knn.fit(features_np)
        distances, indices = knn.kneighbors(features_np)
        
        edge_index = []
        edge_weight = []
        
        for i in range(num_nodes):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Add bidirectional edges
                edge_index.append([i, neighbor_idx])
                edge_index.append([neighbor_idx, i])
                
                # Use inverse distance as weight
                weight = 1.0 / (distance + 1e-8)  # Avoid division by zero
                edge_weight.append(weight)
                edge_weight.append(weight)
        
        if len(edge_index) == 0:
            # If no edges, add self-loop
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_weight = torch.ones(1, dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).T
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        return edge_index.to(self.device), edge_weight.to(self.device)


def load_and_preprocess_data(data_path: str = Config.DATA_PATH) -> Tuple:
    # Load data
    data = pd.read_csv(data_path, sep=r"\s+", header=None, skiprows=2)
    data.columns = ['u1', 'u2', 'u3', 'u4', 'u5', 'y1', 'y2']
    
    # Extract features and target
    u = data[['u1', 'u2', 'u3', 'u4', 'u5']].values
    y = data[['y2']].values.reshape(-1, 1)
    
    # Split data
    indices = np.arange(len(u))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.4, random_state=Config.SEED, shuffle=False
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=Config.SEED, shuffle=False
    )
    
    train_u, val_u, test_u = u[train_indices], u[val_indices], u[test_indices]
    train_y, val_y, test_y = y[train_indices], y[val_indices], y[test_indices]
    
    # Standardize data
    scaler_u = StandardScaler().fit(train_u)
    scaler_y = StandardScaler().fit(train_y)
    
    return (train_u, val_u, test_u, train_y, val_y, test_y, scaler_u, scaler_y)


def create_graph_data(u_data: torch.Tensor, y_data: torch.Tensor, 
                     history_steps: int = Config.HISTORY_STEPS, 
                     device: str = 'cpu') -> List[Data]:

    knn_builder = KNNGraphBuilder(k=Config.KNN_K, device=device)
    data_list = []
    num_samples = u_data.shape[0]
    
    for t in range(history_steps, num_samples - 1):
        # Create sliding window
        u_window = u_data[t - history_steps + 1: t + 1]
        y_window = y_data[t - history_steps + 1: t + 1]
        
        # Create node features
        node_features = torch.cat([u_window.T, y_window.T], dim=0)
        
        # Build graph
        edge_index, edge_weight = knn_builder.build_graph(node_features)
        
        # Create graph data object
        data = Data(
            x=node_features.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_weight.to(device),
            y=y_data[t + 1].unsqueeze(0).to(device)
        )
        data_list.append(data)
    
    return data_list


def evaluate_model(model: nn.Module, loader: DataLoader, 
                  scaler_y: StandardScaler, criterion: nn.Module,
                  device: str = 'cpu') -> Tuple[float, float, float, float]:

    model.eval()
    loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            
            loss += criterion(out, batch.y).item()
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Inverse standardization
    all_preds_orig = scaler_y.inverse_transform(all_preds.reshape(-1, 1))
    all_targets_orig = scaler_y.inverse_transform(all_targets.reshape(-1, 1))
    
    mse = mean_squared_error(all_targets_orig, all_preds_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets_orig, all_preds_orig)
    r2 = r2_score(all_targets_orig, all_preds_orig)
    
    return loss / len(loader), rmse, mae, r2


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               optimizer: torch.optim.Optimizer, criterion: nn.Module,
               scaler_y: StandardScaler,
               epochs: int = Config.EPOCHS, patience: int = Config.PATIENCE,
               device: str = 'cpu') -> Tuple:

    best_loss = float('inf')
    wait = 0
    train_losses = []
    val_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        seed_everything(Config.SEED + epoch)
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_targets = []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_train_preds.append(out.detach().cpu().numpy())
            all_train_targets.append(batch.y.cpu().numpy())
        
        # Calculate training set metrics
        all_train_preds = np.concatenate(all_train_preds)
        all_train_targets = np.concatenate(all_train_targets)
        
        # Inverse standardization for training metrics
        train_preds_orig = scaler_y.inverse_transform(all_train_preds.reshape(-1, 1))
        train_targets_orig = scaler_y.inverse_transform(all_train_targets.reshape(-1, 1))
        
        train_rmse = np.sqrt(mean_squared_error(train_targets_orig, train_preds_orig))
        train_mae = mean_absolute_error(train_targets_orig, train_preds_orig)
        train_r2 = r2_score(train_targets_orig, train_preds_orig)
        
        # Calculate training loss
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_rmse, val_mae, val_r2 = evaluate_model(model, val_loader, scaler_y, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1:3d}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
        print(f"  Val   - Loss: {val_loss:.4f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
        print("-" * 80)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    
    return model, train_losses, val_losses


def main():
    
    # Setup environment
    device = setup_environment()
    seed_everything()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_u, val_u, test_u, train_y, val_y, test_y, scaler_u, scaler_y = load_and_preprocess_data()
    
    # Convert to tensors
    train_u = torch.tensor(scaler_u.transform(train_u), dtype=torch.float32).to(device)
    val_u = torch.tensor(scaler_u.transform(val_u), dtype=torch.float32).to(device)
    test_u = torch.tensor(scaler_u.transform(test_u), dtype=torch.float32).to(device)
    
    train_y = torch.tensor(scaler_y.transform(train_y), dtype=torch.float32).to(device)
    val_y = torch.tensor(scaler_y.transform(val_y), dtype=torch.float32).to(device)
    test_y = torch.tensor(scaler_y.transform(test_y), dtype=torch.float32).to(device)
    
    print(f"Data split - Training: {len(train_u)}, Validation: {len(val_u)}, Test: {len(test_u)}")
    
    # Create graph data
    print("Creating graph data...")
    train_data = create_graph_data(train_u, train_y, device=device)
    val_data = create_graph_data(val_u, val_y, device=device)
    test_data = create_graph_data(test_u, test_y, device=device)
    
    print(f"Graph data - Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Create data loaders
    def worker_init_fn(worker_id):
        np.random.seed(Config.SEED + worker_id)
        random.seed(Config.SEED + worker_id)
    
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True,
                             worker_init_fn=worker_init_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE,
                           worker_init_fn=worker_init_fn, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE,
                            worker_init_fn=worker_init_fn, num_workers=0)
    
    # Initialize model
    print("Initializing GeoDSTG model...")
    model = GeoDSTG(
        history_steps=Config.HISTORY_STEPS,
        out_channel=1,
        hidden_size=Config.HIDDEN_SIZE,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        num_nodes=Config.NUM_NODES
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Train model
    print("Starting training...")
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                                 optimizer, criterion, scaler_y, device=device)
    
    # Final evaluation
    print("\nFinal evaluation:")
    test_loss, test_rmse, test_mae, test_r2 = evaluate_model(model, test_loader, scaler_y, criterion, device=device)
    print(f"Test Results - Loss: {test_loss:.4f}, RMSE: {test_rmse:.6f}, "
          f"MAE: {test_mae:.6f}, R²: {test_r2:.6f}")
    
    # Save model and results
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Save model
    save_path = os.path.join(run_folder, f"model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save loss history
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses
    })
    excel_path = os.path.join(run_folder, f"loss_record_{timestamp}.xlsx")
    loss_df.to_excel(excel_path, index=False)
    print(f"Loss history saved to {excel_path}")


if __name__ == "__main__":
    main()