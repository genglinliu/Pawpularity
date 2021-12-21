from dataset import load_data
from train import *


def main():
    # config   
    num_epochs = 1
    num_classes = 2
    batch_size = 32
    learning_rate = 1e-5
    model_name = HybridVGG16_v40()
    # model_name = vgg16_bn(pretrained=True) # baseline model
    
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(batch_size, use_subset=True)
    
    print("Initializing model...")
    model, criterion, optimizer = initialize_model(model_name, learning_rate, num_classes)
   
    print("Start training... \n")
    train(train_loader, model, criterion, optimizer, num_epochs)
    
    print("Start evaluating... \n")
    evaluate(val_loader, model)    

if __name__ == "__main__":
    main()