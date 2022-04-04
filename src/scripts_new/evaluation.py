import pandas as pd

import torch
import torch.nn as nn

from models.vgg import *
from models.two_layer_CNN import *

from scripts.utils import get_dataframe

from tqdm import tqdm


def evaluate(test_loader, data_dir, model, device):
    model.eval() 
    print('Making predictions...')
    test_pred = []    
    test_df = get_dataframe(data_dir, is_train=False)
    
    with torch.no_grad():
        for (test_images, covariates, test_labels) in tqdm(test_loader):
            test_images = test_images.to(device).float()
            # forward pass
            if isinstance(model, VGG) or isinstance(model, ConvNet_simple):
                outputs = model(test_images)               # baseline vgg
            else:
                outputs = model(test_images, covariates)    # hybrid model takes covariate here
  
            test_pred.extend(outputs.cpu().detach().squeeze().numpy().tolist())

        # write to file
        output_df = pd.DataFrame({"Id": test_df['Id'], "Pawpularity": test_pred})

        return output_df
