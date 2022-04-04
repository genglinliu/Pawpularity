import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from scripts.dataset import *

# utils for data loading

def get_dataframe(data_dir, is_train=True):
    
    if is_train:
        image_dir = os.path.join(data_dir, 'train')
        file_path = os.path.join(data_dir, 'train.csv')
    else:
        image_dir = os.path.join(data_dir, 'test')
        file_path = os.path.join(data_dir, 'test.csv')
    
    df = pd.read_csv(file_path)

    # set image filepath
    df['img_file_path'] = df['Id'].apply(lambda x: os.path.join(image_dir, f'{x}.jpg'))
    
    return df



def load_data(data_dir, batch_size, is_train=True, use_subset=False):
    """
    return the train dataloader
    """
    
    # images and targets
    if is_train:
        df = get_dataframe(data_dir, is_train=True)
        images = df['img_file_path'].to_numpy()
        targets = df['Pawpularity'].to_numpy()
    else:
        df = get_dataframe(data_dir, is_train=False)
        images = df['img_file_path'].to_numpy()
        targets = np.zeros_like(images)
    
    # covariates [2:13]
    # But here for computational complexity we will only choose a few
    selected_columns = ['Accessory', 'Collage', 'Human']
    covariates = df.loc[:, selected_columns].to_numpy()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    
    dataset = PawpularityDataset(image_filepaths=images, covariates=covariates, targets=targets, transform=transform)
    
    subset_ind = list(range(500))
    
    data_subset = Subset(dataset, subset_ind)

    # data loader
    data_loader = DataLoader(dataset=data_subset if use_subset else dataset, 
                                batch_size=batch_size,
                                shuffle=True)
    
    return data_loader


# plotting loss curve

def make_plots(step_hist, loss_hist, experiment_name):
    plt.plot(step_hist, loss_hist)
    plt.xlabel('train_iterations')
    plt.ylabel('Loss')
    plt.title(experiment_name)
    plt.savefig(experiment_name)
    plt.show()
    
# upzip
def unzip_dataset():
    with zipfile.ZipFile("petfinder-pawpularity-score.zip","r") as zip_ref:
        zip_ref.extractall()