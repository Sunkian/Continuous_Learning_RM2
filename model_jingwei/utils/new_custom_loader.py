import torch.utils.data as data
from PIL import Image
import requests

class CustomDataset(data.Dataset):
    def __init__(self, dataset_name, transform=None, target_transform=None, api_url="http://localhost:8000"):
        """
        Args:
            dataset_name (string): Name of the dataset.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on the target.
            api_url (string): URL to the API that returns image paths.
        """
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_transform = target_transform

        # Call the API to get image paths
        response = requests.get(f"{api_url}/get_image_paths/{dataset_name}")
        image_paths_data = response.json()
        self.image_paths = image_paths_data.get("image_paths", [])

        # Since there are no labels, label every image with '0'
        self.targets = [0 for _ in self.image_paths]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)
