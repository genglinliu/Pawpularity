def initialize_model(model, learning_rate, num_classes):
    """
    initialize the model (pretrained vgg16_bn)
    define loss function and optimizer and move data to gpu if available
    
    return:
        model, loss function(criterion), optimizer
    """
    
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()   # potential alternative: nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train(train_loader, model, criterion, optimizer, num_epochs):
    """
    Move data to GPU memory and train for specified number of epochs
    Also plot the loss function and save it in `Figures/`
    Trained model is saved as `cnn.ckpt`
    """
    for epoch in range(num_epochs): # repeat the entire training `num_epochs` times
        # for each training sample
        loss_hist = []
        step_hist = []
        for i, (images, labels) in tqdm(enumerate(train_loader)):
         
            zero_one_labels = (labels + 1) // 2       # map from {-1, 1} to {0, 1}
            
            label = zero_one_labels[:, 2]             # attractiveness label
            cov_attr_1 = zero_one_labels[:, 31]       # smiling   
            cov_attr_2 = zero_one_labels[:, 39]       # young
            cov_attr_3 = zero_one_labels[:, 19]       # high_cheeekbones
            
            cov_attrs = torch.stack((cov_attr_1, cov_attr_2, cov_attr_3)).T # (minibatch, num_cov) e.g. (32, 3)
            
            # move to gpu if available
            images = images.to(device)
            cov_attrs = cov_attrs.to(device)
            label = label.to(device)
            
            # forward pass
            if isinstance(model, VGG):
                outputs = model(images)               # baseline vgg
            
            else:
                outputs = model(images, cov_attrs)    # hybrid model takes covariate here
            
            loss = criterion(outputs, label) 
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 50 == 0:
                print('Epoch: [{}/{}], Step[{}/{}], Loss:{:.4f}' \
                        .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                with open(experiment_name+'.txt', 'a') as f:
                    print('Epoch: [{}/{}], Step[{}/{}], Loss:{:.4f}' \
                        .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()), file=f)
                loss_hist.append(loss.item())
                step_hist.append(i+1)
        
        make_plots(step_hist, loss_hist, epoch)
        
    torch.save(model.state_dict(), experiment_name+'.ckpt')
