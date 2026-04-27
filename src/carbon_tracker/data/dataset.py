import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CarbonDataset(Dataset):
    """
    Custom Dataset for Multispectral Sentinel-2 images and AGB labels.
    """
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (numpy.ndarray): Image data of shape (N, H, W, C).
            labels (numpy.ndarray): Target values of shape (N,).
            transform (callable, optional): Optional transform to be applied.
        """
        # PyTorch expects (Channels, Height, Width), so we transpose (N, H, W, C) -> (N, C, H, W)
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        self.labels = torch.from_numpy(labels).float()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Basic normalization for satellite data (Min-Max or Z-score)
        # Here we scale to [0, 1] assuming 16-bit or typical Sentinel-2 ranges
        image = image / 10000.0 

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(x_path, y_path, batch_size=32, train_ratio=0.8):
    """
    Load .npy files and return training and validation DataLoaders.
    """
    # Load raw data
    x_data = np.load(x_path)
    y_data = np.load(y_path)

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_data, train_size=train_ratio, random_state=42
    )

    # Create Dataset objects
    train_dataset = CarbonDataset(x_train, y_train)
    val_dataset = CarbonDataset(x_val, y_val)

    # Create DataLoaders
    # shuffle=True is set for training to prevent ordering bias
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader