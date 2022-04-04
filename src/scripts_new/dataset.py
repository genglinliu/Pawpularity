import torch
from torch.utils.data import Dataset
import PIL.Image as Image

# Image Dataset - might need modification later for covariates

class PawpularityDataset(Dataset):
    def __init__(self, image_filepaths, covariates, targets, transform):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform
        self.covaraites_all = covariates
    
    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        covaraites_per_image = torch.tensor(self.covaraites_all[idx])
        target = torch.tensor(self.targets[idx])
        
        with open(image_filepath, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            image = self.transform(image)
        
        return image, covaraites_per_image, target
    
    
# class PawpularityDataset(Dataset):
#     def __init__(self, main_dir, all_imgs, labels, meta, transform):
#         self.main_dir = main_dir
#         self.transform = transform
#         self.all_imgs = all_imgs
#         self.labels = labels
#         self.meta = meta

#     def __len__(self):
#         return len(self.all_imgs)

#     def __getitem__(self, idx):
#         img_location = os.path.join(self.main_dir, self.all_imgs[idx])
#         image = Image.open(img_location).convert("RGB")
#         tensor_image = self.transform(image)
#         return tensor_image, self.meta[idx], self.labels[idx]