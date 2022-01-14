# import necessary packages and utils

import torch
from models.vgg import *
from models.two_layer_CNN import *

from scripts.utils import load_data, get_dataframe
from scripts.train import initialize_model, train
from scripts.evaluation import evaluate


def main():
    
    # config  
    data_dir = "../data/petfinder-pawpularity-score/"
    num_classes = 1 # for regression
    batch_size = 32
    learning_rate = 1e-5
    # model_name = ConvNet_hybrid()
    # experiment_name = "simple_cnn_hybrid"
     
    model_name = HybridVGG16()
    experiment_name = "vgg16_hybrid"
    
    # model_name = vgg16_bn(pretrained=True)
    # experiment_name = "vgg16_baseline"
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    print("Device = ", device)
    print("Loading data... ")    
    train_loader = load_data(data_dir, batch_size, is_train=True, use_subset=False)
    test_loader = load_data(data_dir, batch_size, is_train=False)
    
    print("Initializing model...")
    model, criterion, optimizer = initialize_model(model_name, learning_rate, num_classes, device)
   
    print("Start training... \n")
    train(train_loader, model, criterion, optimizer, experiment_name, device)
    
    print("Start evaluating... \n")
    output_df = evaluate(test_loader, data_dir, model, device)    
    output_df.to_csv('submission.csv', index = False)
    
    print("END OF PROGRAM")
    
if __name__ == "__main__":
    main()