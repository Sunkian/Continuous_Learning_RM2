# import os
import torch.utils.data as data
from PIL import Image
import numpy as np
#
#
# class CustomDataset(data.Dataset):
#     def __init__(self, dataset_name, transform=None, target_transform=None):
#         """
#         Args:
#             dataset_name (string): Name of the dataset folder under /data/.
#             transform (callable, optional): Optional transform to be applied on an image.
#             target_transform (callable, optional): Optional transform to be applied on the target.
#         """
#         self.root = os.path.join("/Users/apagnoux/Downloads/Continuous_Learning_RM2-master/data/images", dataset_name)
#
#         # Assuming images are directly under /data/<name_of_the_dataset>.
#         # Adjust as necessary.
#         self.image_paths = list(sorted(os.listdir(self.root)))
#
#         # Since there are no labels, label every image with '0'
#         self.targets = [0 for _ in self.image_paths]
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root, self.image_paths[idx])
#         img = Image.open(img_path).convert("RGB")  # Convert to RGB. Adjust if your images are grayscale, etc.
#         target = self.targets[idx]  # As all images are labeled with '0'
#
#         if self.transform:
#             img = self.transform(img)
#         if self.target_transform:
#             target = self.target_transform(target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     import requests
#     from PIL import Image
#     from io import BytesIO
#     import torch.utils.data as data
#
#     class CustomDataset(data.Dataset):
#         def __init__(self, dataset_name, transform=None, target_transform=None):
#             """
#             Args:
#                 dataset_name (string): Name of the dataset folder.
#                 transform (callable, optional): Optional transform to be applied on an image.
#                 target_transform (callable, optional): Optional transform to be applied on the target.
#             """
#             self.dataset_name = dataset_name
#             self.transform = transform
#             self.target_transform = target_transform
#
#             # Using the FastAPI endpoint to fetch image file names for the dataset
#             response = requests.get(f"http://localhost:8000/list_files/{dataset_name}/")
#             response_data = response.json()
#
#             # Assuming your endpoint returns {"files": [list_of_files]}
#             if 'files' in response_data:
#                 self.image_names = response_data['files']
#             else:
#                 self.image_names = []
#
#             # Since there are no labels, label every image with '0'
#             self.targets = [0 for _ in self.image_names]
#
#         def __getitem__(self, idx):
#             image_name = self.image_names[idx]
#             target = self.targets[idx]  # As all images are labeled with '0'
#
#             # Using the FastAPI endpoint to fetch the image
#             # response = requests.get(f"http://localhost:8000/get_image/{image_name}/")
#             # image_data = BytesIO(response.content)
#             # img = Image.open(image_data).convert("RGB")
#
#             image_api_url = f"http://127.0.0.1:8000/get_image/{image_name}/"
#
#             # Send a request to the FastAPI service to get the image
#             response = requests.get(image_api_url)
#             if response.status_code == 200:
#                 img_data = BytesIO(response.content)
#                 img = Image.open(img_data).convert("RGB")
#             else:
#                 raise FileNotFoundError(f"Image {image_name} not found in database or path.")
#
#             if self.transform:
#                 img = self.transform(img)
#             if self.target_transform:
#                 target = self.target_transform(target)
#
#             return img, target
#
#         def __len__(self):
#             return len(self.image_names)
#



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
