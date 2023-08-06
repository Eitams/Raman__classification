import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)

def plot_validation(loss_hist, metric_hist, params_train, experiment):
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