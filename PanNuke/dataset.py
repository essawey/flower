# pannuke/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

class SemanticSegmentationDataset(Dataset):
    """Semantic Segmentation Dataset"""

    def __init__(self, df, transform=None):
        """
        Args:
            df (Dataframe): Dataframe with all the Directorys needed.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.train_path = df.iloc[0].data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Data Path
        row = self.df.iloc[idx]
        image_path = os.path.join("/", row.data_path, row.png)
        mask_path = os.path.join("/", row.data_path, row.npy)

        # Load a RGB image
        image = cv2.imread(image_path)  # (h, w, c)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and One-Hot encode the mask
        mask = np.load(mask_path)  # (h, w)
        mask = np.eye(6)[mask]  # (h, w, c)

        # Apply Data Augmentation
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # To Tensor
        to_tensor = torchvision.transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        return image, mask


def getMeansAndStds():

    train_path = os.path.join(os.getcwd(), 'PanNuke', 'data', 'Patched', 'Train')

    train_data = torchvision.datasets.ImageFolder(
        root=train_path, 
        transform=torchvision.transforms.ToTensor()
    )

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for image, _ in train_data:
        means += torch.mean(image, dim=(1, 2))
        stds += torch.std(image, dim=(1, 2))

    means /= len(train_data)
    stds /= len(train_data)

    return means.tolist(), stds.tolist()
