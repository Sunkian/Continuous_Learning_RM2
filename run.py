import streamlit as st
import requests
from model_jingwei.exp.exp_OWL import Exp_OWL
from model_jingwei.utils.args_loader import get_args
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
    st.title("Run Resnet18-supcon")

    # Retrieve the list of datasets using FastAPI route
    # datasets_response = requests.get("http://127.0.0.1:8000/get_datasets/")
    # datasets = datasets_response.json()["datasets"]
    option = st.selectbox('Select an option :',
                          ('In-Distribution feature extraction',
                           'New-Sample feature extraction',
                           'Out-Of-Distribution detection'))

    datasets = fetch_datasets()
    # st.write(datasets)

    # Let user select a dataset using Streamlit
    selected_dataset = st.selectbox("Select a dataset:", datasets)

    # Fetch and display the list of files for the selected dataset
    files = fetch_files(selected_dataset)
    # selected_file = st.selectbox("Select an Image:", files)

    # Fetch and display the image for the selected file
    # if selected_file:
    #     image_content = fetch_image(selected_file)
    #     st.image(image_content, caption=selected_file, use_column_width=True)

    # If you need to retrieve and display the list of images for the selected dataset,
    # you can uncomment and use the commented code. However, I'm assuming
    # `list_files` is another FastAPI endpoint you have set up.

    # files_response = requests.get(f"http://127.0.0.1:8000/list_files/{selected_dataset}/")
    # files = files_response.json()["files"]
    # selected_file = st.selectbox("Select an Image:", files)



    if st.button('Run'):
        args = get_args()
        exp = Exp_OWL(args)
        if option == 'In-Distribution feature extraction':
            exp.id_feature_extract()
            st.success('In-distribution features extracted successfully !')
            response = requests.post(f"{BASE_API_URL}/store_metadata/")
            if response.status_code == 200:
                st.success(response.json().get("status", "Metadata stored successfully!"))
            else:
                st.error("Failed to store metadata!")


        elif option == 'New-Sample feature extraction':
            args.out_datasets = [selected_dataset]
            print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
            exp.ns_feature_extract(selected_dataset)
            st.success("New features extracted successfully !")
            print("New features extracted successfully !")

            base_path = "/app/cache/"
            # file_name = "/app/cache/SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
            # original_file_name = "SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
            new_file_name = f"{selected_dataset}vsCIFAR-10_resnet18-supcon_out_alllayers.npz"

            # Rename the file
            # os.rename(os.path.join(base_path, original_file_name), os.path.join(base_path, new_file_name))

            # Set the new file path
            file_name = os.path.join(base_path, new_file_name)

            # response = upload_large_npz_to_backend(file_name)
            # print(response.json())
            st.success(".npz file sucessfully uploaded in the file system")
            print(".npz file sucessfully uploaded in the file syste")

        elif option == 'Out-Of-Distribution detection':
            unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection(selected_dataset, K=50)

            # Prepare data for API
            update_data = []
            for idx, file_name in enumerate(files):  # Assuming 'files' contains the list of file names in the dataset
                update_data.append({
                    "file_name": file_name,
                    "bool_ood": str(bool_ood[idx]),
                    "scores_conf": str(scores_conf[idx]),
                    "pred_scores": str(pred_scores[idx]),
                    "pred_labels": str(pred_labels[idx]),
                })

            # Send POST request to update the results
            response = requests.post(f"{BASE_API_URL}/update_results/", json=update_data)
            if response.status_code == 200:
                st.success("Data updated successfully!")
            else:
                st.error("Failed to update data!")



    # if st.button('Start id feature extraction'):
    #     args = get_args()
    #
    #     # args.out_datasets = [selected_dataset, ]
    #
    #     exp = Exp_OWL(args)
    #
    #     exp.id_feature_extract()
    #     print('ID extracted')
    #
    # if st.button('Store Metadata'):
    #     # Send POST request to the FastAPI endpoint to store metadata
    #     response = requests.post(f"{BASE_API_URL}/store_metadata/")
    #
    #     # Check for successful response and display a message accordingly
    #     if response.status_code == 200:
    #         st.success(response.json().get("status", "Metadata stored successfully!"))
    #     else:
    #         st.error("Failed to store metadata!")
    #
    # if st.button("Start new sample feature extraction"):
    #     parser = argparse.ArgumentParser()
    #     args = get_args()
    #
    #     args.out_datasets = [selected_dataset, ]
    #
    #     exp = Exp_OWL(args)
    #
    #     print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
    #         args.out_datasets))
    #     # exp.ns_feature_extract('SVHN')
    #     # st.write("Feature extraction completed!")
    #
    #     exp.ns_feature_extract(selected_dataset)
    #     st.write("Feature extraction completed")
    #     print("Feature extraction completed")
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
    #     st.write("DONE")
    #     print("PUSHED TO DB !!!")
    #
    # if st.button("OOD "):
    #     parser = argparse.ArgumentParser()
    #     args = get_args()
    #
    #     args.out_datasets = [selected_dataset, ]
    #
    #     exp = Exp_OWL(args)
    #
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

