import numpy as np

import torch
import torch.nn as nn

from scripts.utils import make_plots
from models.vgg import *
from models.two_layer_CNN import *

from tqdm import tqdm


def initialize_model(model, learning_rate, num_classes, device):
    """
    initialize the model
    define loss function and optimizer and move data to gpu if available
    
    return:
        model, loss function(criterion), optimizer
    """
    
    if isinstance(model, VGG) or isinstance(model, HybridVGG16):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer
    
# train
def train(train_loader, model, criterion, optimizer, experiment_name, device):
    """
    Move data to GPU memory
    Also plot the loss function and save it in `Figures/`
    Trained model is saved as `cnn.ckpt`
    """
    model.train()
    
    # for each training sample
    loss_hist = []
    step_hist = []
    for i, (images, covariates, label) in enumerate(tqdm(train_loader)):
    
        train_pred = list()
        train_true = list()

        # move to gpu if available
        images = images.to(device).float()
        covariates = covariates.to(device).float()
        label = label.to(device).float()

        # forward pass
        if isinstance(model, VGG) or isinstance(model, ConvNet_simple):
            outputs = model(images)               # baseline vgg
        else:
            outputs = model(images, covariates)   # hybrid model takes covariate here
        outputs = torch.squeeze(outputs)
        
        # calculate loss
        rmse_loss = torch.sqrt(criterion(outputs, label))

        # backprop
        optimizer.zero_grad()
        rmse_loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            train_true += label.cpu().detach().numpy().tolist()
            train_pred += outputs.cpu().detach().numpy().tolist()
            step_hist.append(i+1)
            loss_hist.append(rmse_loss.item())
            print('Iteration: {}, Train rmse: {}'.format(i+1, rmse_loss.item()))
            
    # plot
    make_plots(step_hist, loss_hist, experiment_name)

    torch.save(model.state_dict(), experiment_name+'.ckpt')