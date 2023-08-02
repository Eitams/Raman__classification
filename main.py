
from comet_ml import Experiment
import time
from project import Project
from logger import logging
from utils import device , show_dl, show_dataset, plot_validation
from data import get_dataloaders
from secretApi import comet_api_key
from pathlib import Path

from models.CNNEN import RamanModel
import torch.optim as optim
from torchsummary import summary

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from models.utils import train_val





if __name__ == '__main__':
  
    project = Project()
    
    # our hyperparameters
    params = {
        # project settings
        'model': '1dCNN-withElasticNet',
        'dataset': 'innerArm.csv',
        'seed': 42,
        'checkpoint_path': project.checkpoint_dir.as_posix() + "/Exp1.pt",

        'Exp_name': "2",

        # Model hyperparameters
        'lr': 0.0001,
        'batch_size': 5,
        'epochs': 500,
        'elasticnet': True,
        
        # convolution layer
        'out_channels':32,
        'kernel_size': 20,

        # Preprocessing
        'start_wave': 800,
        'end_wave': 1600,
        'test_size': 0.3,
        'normalization': True
        

    }
   
    logging.info(f'Using device={device} 🚀')

    print(project.data_dir)

    # Importing data
    # get_dataloaders import the data, perform initial preprocess, arrange it as a dataset and set it as a dataloader
    train_dl, val_dl = get_dataloaders(project.data_dir / params['dataset'], params)

    ## Examine dataloaders and extract shape of input data
    for batch in train_dl:
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

    # create model
    cnn1dEN = RamanModel(width, params).to(device)    
    logging.info(summary(cnn1dEN, (height, width))) # print the model summary
    
    ################################
    ##  Training and validation loop
    ################################
    optimizer = optim.Adam(cnn1dEN.parameters(), lr=params['lr'])
    params_train = {
        'epochs': params["epochs"],
        'optimizer': optimizer,
        'f_loss': nn.BCELoss(),
        'lr_change': ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=100,verbose=1),
        'train': train_dl,
        'val': val_dl,
        'weight_path': params['weights_path'],
        'experiment': experiment,
        'elasticnet': params["elasticnet"],
        'checkpoint_path': params["checkpoint_path"]
              }
   
    ## train and validation function return the selected train model,
    ## the training and validation loss and  the metric history in a dic format
    cnn1dEN,loss_hist,metric_hist = train_val(model=cnn1dEN, params=params_train, verbose=True)

    plot_validation(loss_hist, metric_hist, params_train)
    
    experiment.end()