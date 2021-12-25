import torch
from torch.utils.data import Dataset
import PIL.Image as Image

# Image Dataset - might need modification later for covariates

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

        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(self.targets[idx])
        
        return image, target