import streamlit as st
import requests
import io
from PIL import Image


# def send_images_to_fastapi(dataset_name, files):
#     url = "http://localhost:8000/upload_files/"
#     data = {"dataset_name": dataset_name}
#     response = requests.post(url, data=data, files=files)
#     return response
#
# def fetch_datasets():
#     """
#     Fetches list of datasets from the FastAPI service.
#
#     Returns:
#         list: List of dataset names.
#     """
#     # Send GET request to the FastAPI endpoint
#     response = requests.get("http://localhost:8000/list_datasets/")
#
#     # Raise an exception if there's an HTTP error
#     response.raise_for_status()
#
#     # Extract and return the list of datasets
#     datasets = response.json().get("datasets", [])
#     return datasets
#
# def upload():
#     st.title("Upload")
#
#     # Input form for images
#     uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png", "gif"],
#                                        accept_multiple_files=True)
#     dataset_name = st.text_input("Enter Dataset Name:")
#     if st.button("Upload Images..."):
#         if uploaded_images:
#             files = [("files", (f.name, f)) for f in uploaded_images]
#             response = send_images_to_fastapi(dataset_name, files)
#             if response.status_code == 200:
#                 st.success("Images uploaded successfully!")

def upload():
    st.title("Uploader")

    # Input for dataset name
    dataset_name = st.text_input("Enter batch name:")

    uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if st.button('Upload') and dataset_name:
        # Prepare the files to send in the correct format for the FastAPI endpoint
        files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]

        # Post the files to the FastAPI endpoint
        res = requests.post(url="http://127.0.0.1:8000/uploadfiles",
                            data={"dataset_name": dataset_name},
                            files=files_to_send)

        if res.status_code == 200:  # check if the request was successful
            st.success("Uploaded files")
        else:
            st.error("Error uploading files. Please try again.")



