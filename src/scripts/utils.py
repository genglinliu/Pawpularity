import pandas as pd
import os

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from dataset import *

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


def load_data(batch_size, is_train=True, use_subset=False):
    """
    return the train dataloader
    """
    
    if is_train:
        df = get_dataframe(data_dir, is_train=True)
        images = np.array(df['img_file_path'])
        targets = np.array(df['Pawpularity'])
    else:
        df = get_dataframe(data_dir, is_train=False)
        images = np.array(df['img_file_path'])
        targets = np.zeros_like(images)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    
    dataset = PetDataset(image_filepaths=images, targets=targets, transform=transform)
    
    subse_ind = list(range(500))
    
    data_subset = Subset(dataset, subse_ind)

    # data loader
    data_loader = DataLoader(dataset=data_subset if use_subset else dataset, 
                                batch_size=batch_size,
                                shuffle=True)
    
    return data_loader