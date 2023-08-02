import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    

def show_dataset(df, n=6):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20,10))
    for i in range(0,len(df)):
        if df.loc[i+1,"diabetic"] == 1:
            sns.lineplot(df.iloc[i,800:1600], color='r', alpha=0.5)
        else:
            sns.lineplot(df.iloc[i,800:1600], color='b', alpha=0.5)
    plt.ylabel("Intensity")
    plt.xlabel("Raman shift")

    plt.show()


def show_dl(dl, n=6):

    import seaborn as sns
    import matplotlib.pyplot as plt
    batch = None
    for batch in dl:
        features = batch['features'] 
        target = batch['target']
        break
    
    plt.figure(figsize=(20,10))
    for i in range(0,len(features)):
        if target[i+1] == 1:
            sns.lineplot(features[i], color='r', alpha=0.5)
        else:
            sns.lineplot(features[i], color='b', alpha=0.5)
    plt.ylabel("Intensity")
    plt.xlabel("Raman shift")

    plt.show()

def plot_validation(loss_hist, metric_hist, params_train):
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
    plt.title('Convergence History')
    plt.show()