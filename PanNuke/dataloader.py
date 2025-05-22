# pannuke/dataloader.py
from torch.utils.data import DataLoader
from .dataset import SemanticSegmentationDataset, getMeansAndStds
from .transforms import get_data_augmentation
import pandas as pd
import glob
import os

def load_data(batch_size):
    
    image_dir = os.path.join(os.getcwd(), 'PanNuke', 'data', 'Patched', '**', '*.png')
    mask_dir = os.path.join(os.getcwd(), 'PanNuke', 'data', 'Patched', '**', '*.npy')

    image_file_list = glob.glob(image_dir, recursive=True)
    mask_file_list = glob.glob(mask_dir, recursive=True)
    
    png = sorted(image_file_list)
    npy = sorted(mask_file_list)
    
    # Paths DataFrame
    df = pd.DataFrame({
        'data_path': [os.path.dirname(path) for path in png],
        'png': [os.path.basename(path) for path in png],
        'npy': [os.path.basename(path) for path in npy],
        'clientNo': [next((char for char in path if char.isdigit()), "Test") for path in png],
        'split': ['Train' if 'Train' in path else ('Validation' if 'Validation' in path else 'Test') for path in png],
        'organName': [path.split(os.sep)[-2] for path in png]
    })

    # Splitting the DataFrame into individual client datasets for train and validation
    train_client_0_df = df[df['split'].str.contains("Train") & df['clientNo'].str.contains("0")].head(1)
    train_client_1_df = df[df['split'].str.contains("Train") & df['clientNo'].str.contains("1")].head(1)
    train_client_2_df = df[df['split'].str.contains("Train") & df['clientNo'].str.contains("2")].head(1)
    train_client_3_df = df[df['split'].str.contains("Train") & df['clientNo'].str.contains("3")].head(1)

    val_client_0_df = df[df['split'].str.contains("Validation") & df['clientNo'].str.contains("0")].head(1)
    val_client_1_df = df[df['split'].str.contains("Validation") & df['clientNo'].str.contains("1")].head(1)
    val_client_2_df = df[df['split'].str.contains("Validation") & df['clientNo'].str.contains("2")].head(1)
    val_client_3_df = df[df['split'].str.contains("Validation") & df['clientNo'].str.contains("3")].head(1)

    server_test_df = df[df['split'].str.contains("Test")].head(1)

    # Data augmentation transforms
    means, stds = getMeansAndStds()
    data_augmentation = get_data_augmentation(means, stds)

    # Define datasets for Train, Validation, and Test
    image_datasets = {
        'Train': {
            0: SemanticSegmentationDataset(df=train_client_0_df, transform=data_augmentation['Train']),
            1: SemanticSegmentationDataset(df=train_client_1_df, transform=data_augmentation['Train']),
            2: SemanticSegmentationDataset(df=train_client_2_df, transform=data_augmentation['Train']),
            3: SemanticSegmentationDataset(df=train_client_3_df, transform=data_augmentation['Train']),
        },
        'Validation': {
            0: SemanticSegmentationDataset(df=val_client_0_df, transform=data_augmentation['Validation']),
            1: SemanticSegmentationDataset(df=val_client_1_df, transform=data_augmentation['Validation']),
            2: SemanticSegmentationDataset(df=val_client_2_df, transform=data_augmentation['Validation']),
            3: SemanticSegmentationDataset(df=val_client_3_df, transform=data_augmentation['Validation']),
        },
        'Test': SemanticSegmentationDataset(df=server_test_df, transform=data_augmentation['Test'])
    }

    # Create DataLoaders for clients
    dataloaders = {
        'Train': [
            DataLoader(image_datasets['Train'][0], batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(image_datasets['Train'][1], batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(image_datasets['Train'][2], batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(image_datasets['Train'][3], batch_size=batch_size, shuffle=True, drop_last=True),
        ],
        'Validation': [
            DataLoader(image_datasets['Validation'][0], batch_size=batch_size, shuffle=False, drop_last=True),
            DataLoader(image_datasets['Validation'][1], batch_size=batch_size, shuffle=False, drop_last=True),
            DataLoader(image_datasets['Validation'][2], batch_size=batch_size, shuffle=False, drop_last=True),
            DataLoader(image_datasets['Validation'][3], batch_size=batch_size, shuffle=False, drop_last=True),
        ],
        'Test': DataLoader(image_datasets['Test'], batch_size=batch_size, shuffle=False, drop_last=False),
    }

    return dataloaders

