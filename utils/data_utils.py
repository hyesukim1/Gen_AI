from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = glob(os.path.join(data_dir, '*/*.png'))
        self.class_names = os.listdir(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img