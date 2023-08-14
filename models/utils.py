from tqdm import tqdm
import copy
import torch
from logger import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' Helper Functions'''

# Function to get the learning rate
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# Function to compute the loss value per batch of data
def loss_batch(loss_func, output, target, model,L1 = 0.001, L2 = 0.001, opt=None, elasticnet = False):
    ## opt = True is used only for train loop
    ## elasticnet is used if regulazation if needed

    loss = loss_func(output.float(), target.float()) # get loss

    if elasticnet:
        # Add L1 and L2 regularization to the loss function
        l1_lambda = L1  # Adjust this value for L1 regularization strength
        l2_lambda = L2 # Adjust this value for L2 regularization strength
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        # print(loss)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)  # L1 regularization (absolute value)
            l2_reg += torch.norm(param, 2)  # L2 regularization (squared value)

        loss = loss + l1_lambda * l1_reg + l2_lambda * l2_reg 
   
    threshold = 0.5
    pred = (output >= threshold).float()  # Convert True/False to 1/0 and cast to float
    metric_b=pred.eq(target.view_as(pred)).sum().item() # get performance metric (accuracy)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b

# Compute the loss value & performance metric for the entire dataset (epoch)
def loss_epoch(model,loss_func,dataset_dl, params, opt=None):
    
    run_loss=0.0 
    t_metric=0.0
    len_data=len(dataset_dl.dataset)

    # internal loop over dataset
    for batch in dataset_dl:
        xb = batch["features"]
        yb = batch["target"]
        # move batch to device
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb) # get model output
        loss_b,metric_b=loss_batch(loss_func, output, yb,  model = model, L1 = params["L1"], L2 = params["L2"], opt = opt, elasticnet=params["elasticnet"]) # get loss per batch
        run_loss+=loss_b        # update running loss

        if metric_b is not None: # update running metric
            t_metric+=metric_b    
    
    loss=run_loss/float(len_data)  # average loss value
    metric=t_metric/float(len_data) # average metric value
    
    return loss, metric

def train_val(model, params,verbose=False):
    
    # Get the parameters
    epochs=params["epochs"]
    loss_func=params["f_loss"]
    opt=params["optimizer"]
    train_dl=params["train"]
    val_dl=params["val"]
    lr_scheduler=params["lr_change"]
    checkpoint_path=params["checkpoint_path"]

    
    loss_history={"train": [],"val": []} # history of loss values in each epoch
    metric_history={"train": [],"val": []} # histroy of metric values in each epoch
    best_model_wts = copy.deepcopy(model.state_dict()) # a deep copy of weights for the best performing model
    best_loss=float('inf') # initialize best loss to a large value
    best_metric=0.0 # initialize best accuracy to 0

    best_epoch = 0
    
    ''' Train Model n_epochs '''
    
    for epoch in tqdm(range(epochs)):
        
        ''' Get the Learning Rate '''
        current_lr=get_lr(opt)
        if(verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
        
        '''
        
        Train Model Process
        
        '''
        
        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func,train_dl,params, opt)
        
        # Log results in comet
        params["experiment"].log_metric('train_acc', train_metric, epoch=epoch)
        params["experiment"].log_metric('train_loss', train_loss, epoch=epoch)
        

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        '''
        
        Evaluate Model Process
        
        '''
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func,val_dl, params)

            # Log results
            #logging.info(f'validation_acc=({val_metric})')
            params["experiment"].log_metric('val_acc', val_metric, epoch=epoch)
            params["experiment"].log_metric('val_loss', val_loss, epoch=epoch)

        ## store best model based on validation performance
        # if(val_loss < best_loss):
        #     best_loss = val_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        
        ## store best model based on accuracy performance
        if(val_metric > best_metric):
            best_metric = val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = val_loss
            best_epoch = epoch
            
            checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': val_loss,
            'acc': val_metric
                  }
            
            # store weights into a local file
            torch.save(checkpoint, checkpoint_path)
            if(verbose):
                print("Copied best model weights!")
        
        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if(verbose):
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 

        if(verbose):
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
            print('Best Epoch: {}, Validation loss {:.4f}, Validation accuracy: {:.4f}'.format(best_epoch, best_loss, best_metric*100))
            print("-"*10) 

    # load best model weights
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history

def calculate_average_metrics(fold_metric, metric_type):
    """
    Calculate average metrics across folds

    Parameters:
    - fold_metric (dict): A dictionary containing the metric values for each fold.
    - metric_type (str): Type of metric to calculate ('train' or 'val').
    """
    # Initialize a variable to store the sum of metric lists
    metric_key = 'train' if metric_type == 'train' else 'val'
    metric_sum = [0] * len(fold_metric[list(fold_metric.keys())[0]][metric_key])

    # Loop through each key-value pair in the dictionary
    for value in fold_metric.values():
        metric_sum = [x + y for x, y in zip(metric_sum, value[metric_key])]

    # Calculate the average by dividing the sum by the total number of folds
    num_folds = len(fold_metric)
    average_metric = [x / num_folds for x in metric_sum]

    return average_metric

