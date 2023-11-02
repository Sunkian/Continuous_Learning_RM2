import requests
from PIL import Image
from io import BytesIO
import torch.utils.data as data

class CustomDataset(data.Dataset):
    def __init__(self, dataset_name, transform=None, target_transform=None):
        """
        Args:
            dataset_name (string): Name of the dataset folder.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on the target.
        """
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_transform = target_transform

        # Using the FastAPI endpoint to fetch image file names for the dataset
        response = requests.get(f"http://localhost:8000/list_files/{dataset_name}/")
        response_data = response.json()

        # Assuming your endpoint returns {"files": [list_of_files]}
        if 'files' in response_data:
            self.image_names = response_data['files']
        else:
            self.image_names = []

        # Since there are no labels, label every image with '0'
        self.targets = [0 for _ in self.image_names]

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        target = self.targets[idx]  # As all images are labeled with '0'

        # Using the FastAPI endpoint to fetch the image
        response = requests.get(f"http://localhost:8000/get_image/{image_name}/")
        image_data = BytesIO(response.content)
        img = Image.open(image_data).convert("RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_names)
