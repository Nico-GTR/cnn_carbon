import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10, device="cpu"):
    """
    Execute the training loop with validation, checkpointing, and early stopping.
    
    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Initial learning rate for the Adam optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        device (str): Computation device ('cpu', 'cuda', or 'mps').
        
    Returns:
        torch.nn.Module: The trained model.
    """
    # 1. Setup Loss function (Mean Squared Error for regression) and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    # 2. Setup Checkpointing
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    output_dir = Path("models/weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model_carbon.pth"

    print(f"Starting training on device: {device.upper()}")

    # 3. Main Epoch Loop
    for epoch in range(num_epochs):
        
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode (enables dropout, batchnorm updates)
        train_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            # Move tensors to the targeted device and format target for regression
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1) 

            # Forward pass
            optimizer.zero_grad() # Clear previous gradients
            outputs = model(batch_x)
            
            # Compute loss and Backpropagation
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step() # Update weights

            # Accumulate loss
            train_loss += loss.item() * batch_x.size(0)
            
        # Calculate average training loss for the epoch
        train_loss /= len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode (freezes dropout, batchnorm)
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient computation to save memory and speed up
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        # Calculate average validation loss for the epoch
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- EARLY STOPPING & CHECKPOINTING ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            
            # Save the model state dictionary (weights only)
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Model checkpoint saved! New best Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs due to no validation improvement.")
            break

    print(f"\nTraining completed. Best model weights are saved at: {best_model_path}")
    
    # Load the best weights before returning the model
    model.load_state_dict(torch.load(best_model_path))
    return model