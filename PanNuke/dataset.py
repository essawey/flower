# pannuke/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import glob

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



def class_weights():
    mask_dir = os.path.join(os.getcwd(), 'PanNuke', 'data', 'Patched', 'Train', '**', '*.npy')
    mask_file_list = glob.glob(mask_dir, recursive=True)
    mask_file_list = sorted(mask_file_list)

    class_pixel_counts = {}  # Dictionary to accumulate pixel counts across all files

    for path in mask_file_list:
        # Load the numpy array
        array = np.load(path)
        
        # Get unique class labels and their pixel counts
        unique_classes, counts = np.unique(array, return_counts=True)
        
        # Update the global class pixel count dictionary
        for cls, count in zip(unique_classes, counts):
            if cls in class_pixel_counts:
                class_pixel_counts[cls] += count
            else:
                class_pixel_counts[cls] = count

    # Calculate class weights based on the total pixel count
    total_pixels = sum(class_pixel_counts.values())
    class_weights = {cls: total_pixels / count for cls, count in class_pixel_counts.items()}

    return class_weights

# # Call the function and print the computed class weights
# weights = class_weights()
