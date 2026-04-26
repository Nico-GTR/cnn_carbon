import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_learning_curves(csv_path="models/weights/training_history.csv", output_dir="plots"):
    """
    Reads training history from a CSV and generates a professional learning curve plot.
    This graph is useful for analyzing model convergence and potential overfitting.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: Could not find {csv_path}.")
        print("Make sure you have run the training pipeline completely to generate the CSV history.")
        return

    # Load training metrics
    df = pd.read_csv(csv_file)

    # Set up the plot style (requires seaborn)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot Train vs Validation Loss
    sns.lineplot(data=df, x='epoch', y='train_loss', label='Training Loss', marker='o', linewidth=2)
    sns.lineplot(data=df, x='epoch', y='val_loss', label='Validation Loss', marker='o', linewidth=2)

    # Identify the best epoch based on minimum validation loss
    best_epoch_idx = df['val_loss'].idxmin()
    best_epoch = df.loc[best_epoch_idx, 'epoch']
    best_val_loss = df.loc[best_epoch_idx, 'val_loss']

    # Highlight the best epoch with a vertical line and annotation
    plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label='Best Model Checkpoint')
    plt.scatter(best_epoch, best_val_loss, color='red', s=100, zorder=5)
    plt.annotate(f'Best Val Loss:\n{best_val_loss:.2f}', 
                 (best_epoch, best_val_loss),
                 textcoords="offset points",
                 xytext=(10, 10), 
                 ha='left',
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    # Formatting and labels
    plt.title('ResNet-18 Training Progress: Forest Carbon Estimation', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Squared Error (Loss)', fontsize=12)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Create output directory and save the figure
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / "learning_curves.png"
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Learning curve plot successfully saved to: {save_file}")

if __name__ == "__main__":
    plot_learning_curves()