import os 

from torch.utils.data import DataLoader, Dataset
from glob import glob

class CustomDataset(Dataset):
    """Load images from image folders"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = os.path.join(self.root_dir, '*.png')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imgs[idx] 
        img = Image.open(img_path).convert('RGB')
        label = 0 

        if self.transform:
            img = self.transform(img)

        return img, label