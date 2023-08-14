import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)

def plot_validation(loss_hist, metric_hist, params_train, experiment):
    """
    Create a visualization of training and validation loss, as well as training and validation metric (accuracy)
    over the course of training epochs for a machine learning experiment.

    :param loss_hist: Dictionary containing training and validation loss history.
    :param metric_hist: Dictionary containing training and validation metric (accuracy) history.
    :param params_train: Dictionary containing training parameters.
    :param experiment: Object for logging the experiment.

    This function generates line plots using the seaborn library and displays them in a subplot layout.

    """
    
    import seaborn as sns; sns.set(style='whitegrid')

    epochs=params_train["epochs"]

    fig,ax = plt.subplots(1,2,figsize=(12,5))

    sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='loss_hist["train"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='loss_hist["val"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='metric_hist["train"]')
    sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='metric_hist["val"]')
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_ylabel("Accuracy")
    fig.suptitle('Convergence History')
    # Save the plot to a file
    plt.savefig("graphs/" + params_train["Exp_name"] +'.png')
    
    # Log the figure to your experiment
    #experiment.log_figure(figure_name="multiple_graphs", figure=plt)

    plt.show()


def plot_fold_acc(epochs, average_train_acc, average_val_acc, params):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 5))
    # plt.ylim(0, 1)
    sns.lineplot(x=[*range(1, epochs + 1)], y=average_train_acc, label='Train average accuracy')
    sns.lineplot(x=[*range(1, epochs + 1)], y=average_val_acc, label='Validation average accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Average Accuracy')
    plt.legend()
    plt.grid(True)

    plt.savefig("graphs/" + params["Exp_name"] +'_avg_acc.png')
    plt.show()

def plot_fold_loss(epochs, average_train_loss, average_val_loss, params):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 5))
    # plt.ylim(0, 0.8)
    sns.lineplot(x=[*range(1, epochs + 1)], y=average_train_loss, label='Train average loss')
    sns.lineplot(x=[*range(1, epochs + 1)], y=average_val_loss, label='Validation average loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Average loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/" + params["Exp_name"] +'_avg_loss.png')

    plt.show()