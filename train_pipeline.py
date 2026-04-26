import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Import custom model architecture and training loop
from carbon_tracker.models.resnet import get_multispectral_resnet
from carbon_tracker.training.loop import train_model

# Import your data loaders (adjust this path if your function is named differently)
from carbon_tracker.data.dataset import get_dataloaders

def main():
    print("--- Environment Status ---")
    # Automatically detect hardware (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device.type.upper()}\n")

    print("--- Preparing Data ---")
    # Load the training and validation datasets
    data_dir = Path("data/raw")
    x_path = data_dir / "X_train.npy"
    y_path = data_dir / "y_train.npy"
    train_loader, val_loader = get_dataloaders(x_path, y_path) 
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}\n")

    print("--- Initializing Multispectral ResNet-18 ---")
    # Initialize the model and send it to the designated device
    model = get_multispectral_resnet()
    model = model.to(device)

    # -------------------------------------------------------------
    # CRITICAL ADDITION: Define Loss Function and Optimizer
    # -------------------------------------------------------------
    # Mean Squared Error for regression tasks
    criterion = nn.MSELoss()
    # Adam optimizer with standard learning rate
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Execute the training loop with all required arguments
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        patience=10,
        device=device
    )

if __name__ == "__main__":
    main()