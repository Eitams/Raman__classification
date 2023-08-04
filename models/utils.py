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
def loss_batch(loss_func, output, target, model,reg_value = 0.001, opt=None, elasticnet = False):
    ## opt = True is used only for train loop
    ## elasticnet is used if regulazation if needed

    loss = loss_func(output.float(), target.float()) # get loss

    if elasticnet:
        # Add L1 and L2 regularization to the loss function
        l1_lambda = reg_value  # Adjust this value for L1 regularization strength
        l2_lambda = reg_value  # Adjust this value for L2 regularization strength
        l1_reg = torch.tensor(0.)
        l2_reg = torch.tensor(0.)
        # print(loss)
        for param in model.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)

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
        loss_b,metric_b=loss_batch(loss_func, output, yb,  model = model, reg_value = params["regularization"], opt = opt, elasticnet=params["elasticnet"]) # get loss per batch
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