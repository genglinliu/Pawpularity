import torch
import torch.nn as nn

def initialize_model(model, learning_rate, num_classes, device):
    """
    initialize the model
    define loss function and optimizer and move data to gpu if available
    
    return:
        model, loss function(criterion), optimizer
    """
    
    if isinstance(model, VGG):
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer
    
# train

def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


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
    for i, (images, label) in (enumerate(train_loader)):

        train_pred = list()
        train_true = list()

        # move to gpu if available
        images = images.to(device).float()
        label = label.to(device).float()

        # forward pass
        out = model(images)
        loss = torch.sqrt(criterion(out, label))

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            train_true += label.cpu().detach().numpy().tolist()
            train_pred += out.cpu().detach().numpy().tolist()

            train_rmse = calc_rmse(np.array(train_pred), np.array(train_true))
            
            step_hist.append(i+1)
            loss_hist.append(train_rmse)
            print('Iteration: {}, Train rmse: {}'.format(i+1, train_rmse))
            
    # plot
    make_plots(step_hist, loss_hist)

    torch.save(model.state_dict(), experiment_name+'.ckpt')