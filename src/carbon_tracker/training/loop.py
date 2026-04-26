import torch
import copy
from tqdm import tqdm
from pathlib import Path
import pandas as pd

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, device='cpu'):
    """
    Trains the PyTorch model with Early Stopping and logs the training history.
    
    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (DataLoader): Training data iterator.
        val_loader (DataLoader): Validation data iterator.
        criterion: The loss function (e.g., MSELoss).
        optimizer: The optimization algorithm (e.g., Adam).
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before early stopping.
        device (str): Compute device ('cpu' or 'cuda').
        
    Returns:
        torch.nn.Module: The model loaded with the best weights.
    """
    print("\n--- Starting Training Process ---")
    print(f"Starting training on device: {device}")
    
    # Initialize variables for Early Stopping and checkpointing
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Initialize dictionary to store metrics for the learning curve
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    # Ensure the output directory exists
    output_dir = Path("models/weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # -----------------------------------------
        # 1. Training Phase
        # -----------------------------------------
        model.train()
        running_train_loss = 0.0
        
        # Setup progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        for inputs, labels in train_bar:
            # Move tensors to configured device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape labels to match outputs: (batch_size, 1)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_train_loss += loss.item() * inputs.size(0)
            
        # Calculate average loss over the epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # -----------------------------------------
        # 2. Validation Phase
        # -----------------------------------------
        model.eval()
        running_val_loss = 0.0
        
        # Disable gradient calculation for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        # -----------------------------------------
        # 3. Logging Metrics
        # -----------------------------------------
        # Added spaces at the end to clean up the tqdm progress bar overlap
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}      ")
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        # -----------------------------------------
        # 4. Early Stopping & Checkpoint Logic
        # -----------------------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the current best weights in memory
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save weights to disk
            torch.save(best_model_wts, output_dir / "best_model_carbon.pth")
            print(f"  --> Model checkpoint saved! New best Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement for {epochs_no_improve} epoch(s).")
            
            # Stop training if no improvement for 'patience' epochs
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # -----------------------------------------
    # 5. Finalize and Export
    # -----------------------------------------
    print(f"\nTraining completed. Best Val Loss: {best_val_loss:.4f}")
    
    # Export metrics to CSV for the plot_training.py script
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f"Training history successfully saved to {output_dir / 'training_history.csv'}")

    # Load the best weights back into the model before returning it
    model.load_state_dict(best_model_wts)
    return model