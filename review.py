import streamlit as st
import requests
from PIL import Image
import os

BASE_API_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI server address

def fetch_ood_images():
    """
    Fetches list of images with 'bool_ood' set to True from the FastAPI service.

    Returns:
        list: List of dictionaries containing file paths and names of the images.
    """
    response = requests.get(f"{BASE_API_URL}/get_ood_images/")
    response.raise_for_status()
    ood_images = response.json()
    return ood_images

def review():
    st.title("Review Out-of-Distribution Images")

    st.write("Instructions:")
    st.write("1. Select the files that you wish to re-classify")
    st.write("2. You can either select all the images or one by one")

    ood_images = fetch_ood_images()

    if not ood_images:
        st.write("No OOD images found.")
        return

    st.write(f"Total OOD images: {len(ood_images)}")

    # Add a 'select all' checkbox
    select_all = st.checkbox("Select all")

    selected_images = []  # List to hold the names of the selected images

    # Using the columns feature to display up to 4 images per row
    for i in range(0, len(ood_images), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(ood_images):
                img_info = ood_images[i + j]
                print(img_info)
                img_path = img_info["file_path"]
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    image = image.resize((400, 200))

                    # Create a checkbox for each image and default its value based on the 'select all' checkbox
                    is_selected = cols[j].checkbox("", value=select_all, key=img_info["file_name"])

                    if is_selected:
                        selected_images.append(img_info["file_name"])

                    dataset_name = img_info.get("dataset",
                                                "Unknown dataset")  # Use a default value if dataset is missing
                    cols[j].image(image, caption=f"{img_info['file_name']} ({dataset_name}) ", use_column_width=True)
                else:
                    cols[j].write(f"Image not found: {img_info['file_name']}")
    class_ground_truth = st.text_input("Enter new batch name:")


    # If the 'Update' button is pressed and there are selected images, update their class_ground_truth
    if st.button("Update Ground Truth") and selected_images:
        payload = {
            "file_names": selected_images,
            "class_ground_truth": class_ground_truth,
            "dataset": "FT_" + img_info["dataset"],
            "reviewed": True
        }
        response = requests.post(f"{BASE_API_URL}/update_ground_truth/", json=payload)
        if response.status_code == 200:
            st.success("Successfully updated class ground truth!")
        else:
            st.error("Failed to update class ground truth.")
