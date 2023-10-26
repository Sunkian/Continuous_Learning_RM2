import os
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Optional, Tuple
# from API.api_helper import fetch_images
from PIL import Image
import io
import base64
import requests

BASE_API_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI server address


def fetch_images(dataset_name):
    """
    Fetches image details from the provided FastAPI endpoint.
    Args:
        dataset_name (str): Name of the dataset to fetch images from.

    Returns:
        list: List of dictionaries containing image details.
    """
    # Fetch the list of all image names in the dataset
    response = requests.get(f"{BASE_API_URL}/list_files/{dataset_name}/")
    response.raise_for_status()  # Raise an exception for HTTP errors
    image_names = response.json().get("files", [])

    # Fetch individual image details
    image_data = []
    for image_name in image_names:
        response = requests.get(f"{BASE_API_URL}/get_image/{image_name}/")
        response.raise_for_status()
        encoded_content = base64.b64encode(response.content).decode('utf-8')
        image_data.append({
            'name': image_name,
            'data': encoded_content
        })

    return image_data
class GenericImageDataset(Dataset):
    def __init__(self, source, mode='external', transform: Optional[Callable] = None):
        """
        Args:
            source (str): Path to the local directory or name of the dataset in the external database.
            mode (str): 'local' for local directory or 'external' for external databases.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform
        self.mode = mode

        if mode == 'local':
            self.image_files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
            self.root_dir = source

        elif mode == 'external':
            self.dataset_name = source
            self.image_data = fetch_images(self.dataset_name)
        else:
            raise ValueError("Mode should be either 'local' or 'external'.")

    def __len__(self):
        if self.mode == 'local':
            return len(self.image_files)
        else:
            return len(self.image_data)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        if self.mode == 'local':
            image_path = os.path.join(self.root_dir, self.image_files[index])
            image_name = self.image_files[index]
            image = Image.open(image_path)

        else:
            image_info = self.image_data[index]
            image_name = image_info['name']
            image_bytes = base64.b64decode(image_info['data'])
            image = Image.open(io.BytesIO(image_bytes))

        if self.transform:
            image = self.transform(image)

        return image, 0
