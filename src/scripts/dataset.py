class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb) / 255 # convert to 0-1

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        target = self.targets[idx]

        image = torch.tensor(image, dtype = torch.float)
        target = torch.tensor(target, dtype = torch.float)
        return image, target
    
def get_transforms(dim=224):
    return A.Compose(
        [            
            A.Resize(height=dim, width=dim)
        ]
    )