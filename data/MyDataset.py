from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the features and target label for the given index
        features = self.dataframe.iloc[idx, 1:].values  # Exclude the first column (target) from the features
        target = self.dataframe.iloc[idx, 0]           # First column contains the target label

        # Convert features to tensor (assuming they are numerical)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Apply transformations if specified
        if self.transform is not None:
            features = self.transform(features)

        return {'features': features, 'target': target}