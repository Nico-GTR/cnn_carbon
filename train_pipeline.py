import torch
from pathlib import Path
from carbon_tracker.data.dataset import get_dataloaders
from carbon_tracker.models.resnet import get_multispectral_resnet
from carbon_tracker.training.loop import train_model

def main():
    """
    Main execution pipeline for model training.
    Assembles data loaders, initializes the model, and starts the training loop.
    """
    # 1. Configuration
    data_dir = Path("data/raw")
    x_path = data_dir / "X_train.npy"
    y_path = data_dir / "y_train.npy"
    
    # Hardware detection: Automatically fall back to CPU if no NVIDIA GPU is found
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Environment Status ---")
    print(f"Target Compute Device: {device.upper()}")
    
    if not x_path.exists() or not y_path.exists():
        print(f"Error: Data files not found in {data_dir}.")
        print("Please run 'python src/carbon_tracker/data/extractor.py' first.")
        return

    # 2. Data Preparation
    print("\n--- Preparing Data ---")
    # Batch size of 16 is safe for CPUs and 4-channel images
    train_loader, val_loader = get_dataloaders(x_path, y_path, batch_size=16)
    print(f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}")

    # 3. Model Initialization
    print("\n--- Initializing Multispectral ResNet-18 ---")
    model = get_multispectral_resnet()

    # 4. Training Execution
    print("\n--- Starting Training Process ---")
    # FOR LOCAL TESTING: Set num_epochs=1 or 2. 
    # FOR FULL TRAINING (Colab/GPU): Set num_epochs=50.
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,         # <--- Changed to 2 for local CPU test
        learning_rate=0.001,
        patience=5,
        device=device
    )

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()