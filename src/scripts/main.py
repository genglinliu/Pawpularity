# import necessary packages and utils

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset

import PIL.Image as Image
from tqdm import tqdm
import pickle

from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit

from models.vgg16 import *
from utils import *


def main():
    
    # config  
    data_dir = "../data/petfinder-pawpularity-score/"
    num_classes = 1 # for regression
    batch_size = 16
    learning_rate = 1e-3
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")    
    train_loader = load_data(batch_size, is_train=True, use_subset=True)
    test_loader = load_data(batch_size, is_train=False)
    
    print("Initializing model...")
    model, criterion, optimizer = initialize_model(model_name, learning_rate, num_classes)
   
    print("Start training... \n")
    train(train_loader, model, criterion, optimizer)
    
    print("Start evaluating... \n")
    evaluate(val_loader, model)    

if __name__ == "__main__":
    main()