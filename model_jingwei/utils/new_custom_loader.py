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

        # Call the API to get image data
        response = requests.get(f"{api_url}/get_image_data/{dataset_name}")
        image_data = response.json()

        # Extract image paths and targets if provided, otherwise default to 0
        self.image_paths = []
        self.targets = []
        for item in image_data.get("image_data", []):
            self.image_paths.append(item.get('image_path'))
            # Assign a default target of 0 if not provided
            self.targets.append(item.get('target', None))

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
