def evaluate(val_loader, model):
    """
    Run the validation set on the trained model
    """
    # uncomment if you want to load from checkpoint
    # model_path = "{}.ckpt".format(experiment_name)
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)
    
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for images, labels in tqdm(val_loader):
            
            zero_one_labels = (labels + 1) // 2 # map from {-1, 1} to {0, 1}
            
            label = labels[:, 2]
            cov_attr_1 = zero_one_labels[:, 31]       # smiling   
            cov_attr_2 = zero_one_labels[:, 39]       # young
            cov_attr_3 = zero_one_labels[:, 19]       # high_cheeekbones
            
            cov_attrs = torch.stack((cov_attr_1, cov_attr_2, cov_attr_3)).T # (minibatch, num_cov) e.g. (32, 3)
            
            # move to device
            images = images.to(device)
            cov_attrs = cov_attrs.to(device)
            label = label.to(device)
            
            # forward pass
            if isinstance(model, VGG):
                outputs = model(images)               # baseline vgg
            
            else: 
                outputs = model(images, cov_attrs)    # hybrid model takes covariate here
                
            _, predicted = torch.max(outputs.data, dim=1)
            
            # accumulate stats
            y_true.append(label.cpu().numpy()) # in the one
            y_pred.append(predicted.cpu().numpy())
            total += label.size(0) # number of elements in the tensor
            correct += (label == predicted).sum().item()
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        print('F1 Score: {}'.format(f1_score(y_true, y_pred, average='macro')))
        print('Validation accuracy: {}'.format(correct / total))
        with open(experiment_name+'.txt', 'a') as f:
            print('F1 Score: {}'.format(f1_score(y_true, y_pred, average='macro')), file=f)
            print('Validation accuracy: {}'.format(correct / total), file=f)