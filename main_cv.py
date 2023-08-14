
from comet_ml import Experiment
import time
from project import Project
from logger import logging
from utils import device , plot_validation, plot_fold_acc, plot_fold_loss
from data import get_dataloaders_cv
from secretApi import comet_api_key
from pathlib import Path

from models.CNNEN import RamanModel
import torch.optim as optim
from torchsummary import summary
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import random
import numpy as np

from models.utils import train_val, calculate_average_metrics




if __name__ == '__main__':
    project = Project()

    # our hyperparameters
    params = {
        # project settings
        'model': '1dCNN-withElasticNet',
        'dataset': 'earLobe.csv',#'innerArm.csv',
        'seed': 42,
        'checkpoint_path': project.checkpoint_dir / 'EL_CV5p2.pt', #project.checkpoint_dir.as_posix() + "/Exp16.pt",

        # Name used to identify experiment in comet
        'Exp_name': "EL_CV5p2",

        # Model hyperparameters
        'lr': 0.0003,
        'batch_size': 4,
        'epochs': 500,
        'elasticnet': True,
        'L1': 0.001,
        'L2': 0.01,

        # convolution layer
        'out_channels':32,
        'kernel_size': 10,

        # Preprocessing
        'start_wave': 800,
        'end_wave': 1800,
        'test_size': 0.3,
        'normalization': True
        

    }

    logging.info(f'Using device={device} 🚀')

    print(project.data_dir)

    # Set seeds for reproducibility
    torch.manual_seed(params["seed"])
    random.seed(params["seed"])
    np.random.seed(params["seed"])

    # Importing data
    # get_dataloaders create 7 cross validation folds of the dataframe. After importing the data each fold is preprocess, arrange as a dataset and set as a dataloader in a dic
    train_folds, val_folds = get_dataloaders_cv(project.data_dir / params['dataset'], params) # Format: {1: {'features':fold1_torch["features"], 'target':fold1_torch["target"]}, 2:...}

    ## Examine dataloaders and extract shape of input data
    for batch in train_folds[1]:
        print(batch['features'].size())
        print(batch['target'])
        width = batch['features'].size(2)
        height = batch['features'].size(1)
        break


    # define our comet experiment
    experiment = Experiment(api_key=comet_api_key,
                            project_name="RamanPerDiabeticClassification",
                            workspace="eitams"
                            )
    experiment.set_name(params["Exp_name"])
    experiment.log_parameters(params)

    #----------------------------------------------------
    ## Train and validation of each fold
    #----------------------------------------------------

    fold_loss ={}
    fold_metric = {}
    for fold in train_folds.keys():
        print('----------------------------------------------------------')
        print("Fold: {}".format(fold))
        print('----------------------------------------------------------')

    
        # create model
        cnn1dEN = RamanModel(width, params).to(device)    
        # logging.info(summary(cnn1dEN, (height, width))) # print the model summary

        ################################
        ##  Training and validation loop
        ################################
        optimizer = optim.Adam(cnn1dEN.parameters(), lr=params['lr'])
        # optimizer = optim.SGD(cnn1dEN.parameters(), lr=params['lr'], momentum=0.9)

        params_train = {
            'epochs': params["epochs"],
            'optimizer': optimizer,
            'f_loss': nn.BCELoss(),
            'lr_change': ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=100,verbose=1),
            'train': train_folds[fold],
            'val': val_folds[fold],
            'experiment': experiment,
            'elasticnet': params["elasticnet"],
            'checkpoint_path': params["checkpoint_path"],
            'Exp_name': params["Exp_name"],
            'L1': params["L1"],
            'L2': params["L2"]

                    }

        ## train and validation function return the selected train model,
        ## the training and validation loss and  the metric history in a dic format
        cnn1dEN,loss_hist,metric_hist = train_val(model=cnn1dEN, params=params_train, verbose=True)
        fold_loss[fold] = loss_hist
        fold_metric[fold] = metric_hist

    #----------------------------------------------------
    ## Calculate average metrics across folds
    #----------------------------------------------------

    avg_train_acc = calculate_average_metrics(fold_metric, 'train')
    avg_val_acc = calculate_average_metrics(fold_metric, 'val')
    avg_train_loss = calculate_average_metrics(fold_loss, 'train')
    avg_val_loss = calculate_average_metrics(fold_loss, 'val')

    #----------------------------------------------------
    ## Plot 7 folds average loss and accuracy
    #----------------------------------------------------   
    # experiment.log_metrics(avg_train_acc)
    plot_fold_acc(params["epochs"], avg_train_acc , avg_val_acc, params)
    plot_fold_loss(params["epochs"], avg_train_loss , avg_val_loss, params)

    experiment.end()
