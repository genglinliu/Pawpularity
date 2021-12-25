# import necessary packages and utils

import torch
from models.vgg16 import *
from models.two_layer_CNN import ConvNet_v1

from scripts.utils import load_data, get_dataframe
from scripts.train import initialize_model, train
from scripts.evaluation import evaluate




def main():
    
    # config  
    data_dir = "../data/petfinder-pawpularity-score/"
    num_classes = 1 # for regression
    batch_size = 16
    learning_rate = 1e-3
    model_name = ConvNet_v1()
    # model_name = vgg16_bn(pretrained=True)
    experiment_name = "simple_run"
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")    
    train_loader = load_data(data_dir, batch_size, is_train=True, use_subset=True)
    test_loader = load_data(data_dir, batch_size, is_train=False)
    
    print("Initializing model...")
    
    model, criterion, optimizer = initialize_model(model_name, learning_rate, num_classes, device)
   
    print("Start training... \n")
    train(train_loader, model, criterion, optimizer, experiment_name, device)
    
    print("Start evaluating... \n")
    output_df = evaluate(test_loader, data_dir, model, device)    
    
if __name__ == "__main__":
    main()