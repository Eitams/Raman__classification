import numpy as np
from .MyDataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torchvision import transforms
from .utils import preprocess, preprocess_cv


def get_dataloaders(df_dir,params):
    
    """
    This function returns the train and val dataloaders.
    """

    # create the datasets
    df = pd.read_csv(df_dir)

    ## preprocess data
    train_df, val_df = preprocess(df, params, normalization = params["normalization"])


    logging.info(f'Train samples={len(train_df)}, Validation samples={len(val_df)}')
    
        # Define data transformations (optional)
    transform = transforms.Compose([
        # Add any necessary transformations here
    ])
    train_ds = CustomDataset(train_df, transform=transform)
    val_ds = CustomDataset(val_df, transform=transform)


    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    # val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)
    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
   
    return train_dl, val_dl

def get_dataloaders_cv(df_dir,params):
    
    """
    This function returns the train and val dataloaders.
    """

    # create the datasets
    df = pd.read_csv(df_dir)


    ## preprocess data
    train_folds, val_folds = preprocess_cv(df, params, normalization = params["normalization"])


    # logging.info(f'Train samples={len(train_folds[1])}, Validation samples={len(val_folds[1])}')
   
    return train_folds, val_folds