import streamlit as st
import requests
import torch

from model_jingwei.exp.exp_OWL import Exp_OWL
from model_jingwei.utils.args_loader import get_args
from model_jingwei.utils.new_custom_loader import CustomDataset
import torchvision.transforms as transforms
import argparse
import os

BASE_API_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI server address


def fetch_datasets():
    """
    Fetches list of datasets from the FastAPI service.

    Returns:
        list: List of dataset names.
    """
    # Send GET request to the FastAPI endpoint
    response = requests.get(f"{BASE_API_URL}/list_datasets/")

    # Raise an exception if there's an HTTP error
    response.raise_for_status()

    # Extract and return the list of datasets
    datasets = response.json().get("datasets", [])
    return datasets


def fetch_files(dataset_name):
    """
    Fetches list of files for a given dataset from the FastAPI service.
    Returns:
        list: List of file names.
    """
    response = requests.get(f"{BASE_API_URL}/list_files/{dataset_name}/")
    response.raise_for_status()
    files = response.json().get("files", [])
    return files


def fetch_image(image_name):
    """
    Fetches an image by its name from the FastAPI service.

    Returns:
        bytes: Content of the image.
    """
    response = requests.get(f"{BASE_API_URL}/get_image/{image_name}/")
    response.raise_for_status()
    return response.content


def run():
    # st.title("Model Inference / Fine-Tuning")

    st.markdown("""
        
            <h4 style='text-align: left; color: #6F6F6F; margin-bottom: 20px;'>Model Inference / Fine-Tuning</h4>
            """, unsafe_allow_html=True)

    st.info('**Instructions:**\n'
            'Option 1 : Model Inference\n'
            '• Select a data batch on which you wish to run the model inference.\n'
            'Option 2 : Model Fine-Tuning\n'
            '• Select the generated labeled dataset to fine-tune the model\n')

    # Retrieve the list of datasets using FastAPI route
    # datasets_response = requests.get("http://127.0.0.1:8000/get_datasets/")
    # datasets = datasets_response.json()["datasets"]
    option = st.selectbox('Select an option :',
                          ('ID feature extraction',
                           'NS feature extraction',
                           'Out-Of-Distribution detection'))

    datasets = fetch_datasets()
    # st.write(datasets)

    # Let user select a dataset using Streamlit
    selected_dataset = st.selectbox("Select a dataset:", datasets)

    # Fetch and display the list of files for the selected dataset
    files = fetch_files(selected_dataset)
    # selected_file = st.selectbox("Select an Image:", files)

    if st.button('Run'):
        args = get_args()
        exp = Exp_OWL(args)
        if option == 'In-Distribution feature extraction':
            exp.id_feature_extract(exp.model, args.in_dataset)
            st.success('ID features extracted successfully !')
            # response = requests.post(f"{BASE_API_URL}/store_metadata/")
            # if response.status_code == 200:
            #     st.success(response.json().get("status", "Metadata stored successfully!"))
            # else:
            #     st.error("Failed to store metadata!")

        if option == 'NS feature extraction':
            transformations = transforms.Compose([
                transforms.Resize((32, 32)),  # Resize to the desired size
                transforms.ToTensor()
            ])
            args.out_datasets = [selected_dataset]
            dataset = CustomDataset(dataset_name=selected_dataset, transform=transformations)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            exp.ns_feature_extract(exp.model, data_loader, selected_dataset)

            st.success('NS features extracted successfully !')


        # elif option == 'New-Sample feature extraction':
        #     args.out_datasets = [selected_dataset]
        #     print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
        #     exp.ns_feature_extract(selected_dataset)
        #     st.success("New features extracted successfully !")
        #     print("New features extracted successfully !")
        #
        #     base_path = "/app/cache/"
        #     # file_name = "/app/cache/SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
        #     # original_file_name = "SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
        #     new_file_name = f"{selected_dataset}vsCIFAR-10_resnet18-supcon_out_alllayers.npz"
        #
        #     # Rename the file
        #     # os.rename(os.path.join(base_path, original_file_name), os.path.join(base_path, new_file_name))
        #
        #     # Set the new file path
        #     file_name = os.path.join(base_path, new_file_name)
        #
        #     # response = upload_large_npz_to_backend(file_name)
        #     # print(response.json())
        #     st.success(".npz file sucessfully uploaded in the file system")
        #     print(".npz file sucessfully uploaded in the file syste")
        #
        # elif option == 'Out-Of-Distribution detection':
        #     unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection(selected_dataset, K=50)
        #
        #     # Prepare data for API
        #     update_data = []
        #     for idx, file_name in enumerate(files):  # Assuming 'files' contains the list of file names in the dataset
        #         update_data.append({
        #             "file_name": file_name,
        #             "bool_ood": str(bool_ood[idx]),
        #             "scores_conf": str(scores_conf[idx]),
        #             "pred_scores": str(pred_scores[idx]),
        #             "pred_labels": str(pred_labels[idx]),
        #         })
        #
        #     # Send POST request to update the results
        #     response = requests.post(f"{BASE_API_URL}/update_results/", json=update_data)
        #     if response.status_code == 200:
        #         st.success("Data updated successfully!")
        #     else:
        #         st.error("Failed to update data!")


