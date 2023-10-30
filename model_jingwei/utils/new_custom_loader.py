import os
import torch.utils.data as data
from PIL import Image
import numpy as np


class CustomDataset(data.Dataset):
    def __init__(self, dataset_name, transform=None, target_transform=None):
        """
        Args:
            dataset_name (string): Name of the dataset folder under /data/.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.root = os.path.join("/Users/apagnoux/Downloads/Continuous_Learning_RM2-master/data/images", dataset_name)

        # Assuming images are directly under /data/<name_of_the_dataset>.
        # Adjust as necessary.
        self.image_paths = list(sorted(os.listdir(self.root)))

        # Since there are no labels, label every image with '0'
        self.targets = [0 for _ in self.image_paths]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_paths[idx])
        img = Image.open(img_path).convert("RGB")  # Convert to RGB. Adjust if your images are grayscale, etc.
        target = self.targets[idx]  # As all images are labeled with '0'

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target, self.image_paths[idx]

    def __len__(self):
        return len(self.image_paths)
