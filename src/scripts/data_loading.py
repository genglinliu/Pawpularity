def load_data(batch_size, use_subset=True):
    """
    return the train/val/test dataloader
    """
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])
    
    train_dataset = datasets.CelebA(root='./data',
                                    split='train',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)
    val_dataset = datasets.CelebA(root='./data',
                                    split='valid',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)
    test_dataset = datasets.CelebA(root='./data',
                                    split='test',
                                    target_type='attr',
                                    transform=transform,
                                    download=False)
    
    indices_train = list(range(700))
    indices_val = list(range(150))    
    indices_test = list(range(150))
    
    train_subset = Subset(train_dataset, indices_train)
    val_subset = Subset(train_dataset, indices_val)
    test_subset = Subset(test_dataset, indices_test)

    # data loader
    train_loader = DataLoader(dataset=train_subset if use_subset else train_dataset, 
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = DataLoader(dataset=val_subset if use_subset else val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = DataLoader(dataset=test_subset if use_subset else test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    return train_loader, val_loader, test_loader