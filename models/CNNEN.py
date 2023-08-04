import numpy as np
import torch.nn as nn

def findConv1dOutShape(win, conv):
    '''
    The function automatically calculate the output width of a 1 d CNN
    '''
    # Get conv arguments
    kernel_size = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    dilation = conv.dilation[0]

    wout = np.floor((win - kernel_size + 2 * padding) / stride + 1)

    return int(wout)

def findpollingOutShape(win, conv):
    '''
    The function automatically calculate the output width of a pooling layer
    '''
    # Get conv arguments
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    wout = np.floor((win - kernel_size ) //stride + 1)

    return int(wout)



class RamanModel(nn.Module):
    def __init__(self, input_size, params):
        super(RamanModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=params["out_channels"], kernel_size=params["kernel_size"], stride=1, padding=1)
        w = findConv1dOutShape(input_size, self.conv1)
        # output shape (32, 1, 3142) when len(single_input.shape[2]==3159) output width = ((input_width - karnel_width + 2*padding)/stride) + 1
        self.relu1 = nn.ReLU()
        self.maxpoll1 = nn.MaxPool1d(kernel_size=4, stride=2)
        w = findpollingOutShape(w, self.maxpoll1)
        # output shape (32, 1, 1570) when len(single_input.shape[2]==3159)

        self.flatten1 = nn.Flatten()
        # output shape (1, 50240) when len(single_input.shape[2]==3159)
        # Calculate the input size for the fully connected (linear) layer 12480
        self.fc1 = nn.Linear(w * 32, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("1 {}".format(x.shape))
        x = self.conv1(x)
        # print("2 {}".format(x.shape))
        x = self.relu1(x)
        # print("3 {}".format(x.shape))
        x = self.maxpoll1(x)
        # print("4 {}".format(x.shape))
        x = self.flatten1(x)
        # print("5 {}".format(x.shape))
        logits = self.fc1(x)
        # print("6 {}".format(logits.shape))
        prob = self.sig(logits)
        
        # threshold = 0.5
        # predictions = (prob >= threshold).float()  # Convert True/False to 1/0 and cast to float

        return prob.squeeze(1) #logits # predictions.squeeze(1) # logits #, prob, predictions