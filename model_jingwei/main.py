import os
import time
import argparse, configparser
from torchvision import transforms
import requests
from torch.utils.data import DataLoader
from .utils.args_loader import get_args
from .utils import metrics
import torch
import faiss
import numpy as np
from .exp.exp_OWL import Exp_OWL
from .utils.data_loader import get_loader_in, get_loader_out
import streamlit as st
# from .utils.custom_loader import GenericImageDataset
from .utils.new_custom_loader import  CustomDataset

import torchvision.transforms as transforms

transformations = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to the desired size
    transforms.ToTensor()
])


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


def new_code():
    st.title("Test on the new code")

    datasets = fetch_datasets()
    selected_dataset = st.selectbox("Select a dataset:", datasets)
    files = fetch_files(selected_dataset)

    imagesize = 32
    transform_test = transforms.Compose([
        transforms.Resize((imagesize, imagesize)),
        transforms.CenterCrop(imagesize),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
        #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])


    if st.button('Run'):

        parser = argparse.ArgumentParser()
        args = get_args()

        exp = Exp_OWL(args)  # set experiments

        ##### RUN ID_FEATURE EXTRACT ################################# --> Working
        # print('>>>>>>>start feature extraction on in-distribution data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
        #     args.in_dataset))
        # exp.id_feature_extract(exp.model, args.in_dataset)
        #

        ##### RUN NS_FEATURE EXTRACT #################################
        ##### HERE, CHANGE THE DATA LOADER TO MINE #################################
        # print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(selected_dataset))
        # # loader_out = get_loader_out(args, dataset=(None, 'SVHN'), split=('val'))
        # # val_loader_out = loader_out.val_ood_loader  # take the val/test batch of the ood data
        # # ood_data =  CustomDataset(source=selected_dataset, transform=transform_test)
        # # out_loader = DataLoader(ood_data, batch_size=32, shuffle=False,
        # #                                                 num_workers=2)
        #
        # # --->
        # dataset = CustomDataset(dataset_name=selected_dataset, transform=transformations)
        # data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        # # # <---
        # # # # # print(out_loader)
        # exp.ns_feature_extract(exp.model, data_loader, selected_dataset)

        print('>>>>>>>start ood detection on new-coming data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.out_datasets))
        unknown_idx,scores_conf, bool_ood = exp.ood_detection(selected_dataset, K=50)

        update_data = []
        for idx, file_name in enumerate(files):  # Assuming 'files' contains the list of file names in the dataset
            update_data.append({
                # "file_name": file_name,
                # # "unknown_idx" : str(unknown_idx),
                # "bool_ood": str(bool_ood[idx]),
                # "scores_conf": str(scores_conf[idx]),
                "file_name": file_name,
                "bool_ood": bool(bool_ood[idx]),
                "scores_conf": float(scores_conf[idx]),
            })

        # print(update_data)

        # Send POST request to update the results
        response = requests.post(f"{BASE_API_URL}/update_results/", json=update_data)
        print(response.content)
        if response.status_code == 200:
            st.success("Data updated successfully!")
        else:
            st.error("Failed to update data!")

        print(f'Total new samples: {len(bool_ood)} \nNumber of correctly detected ood samples: {len(unknown_idx)}')
        st.write(f'Total new samples: {len(bool_ood)} \nNumber of correctly detected ood samples: {len(unknown_idx)}')





        #
        print('>>>>>>>start incremental learning on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(selected_dataset))
        ood_class = [0, 1]  # select two classes in ood data as unrecognized/new classes
        n_ood = 50  # take 50 ood samples
        exp.train_global(selected_dataset, True, ood_class, n_ood)
        print('DONE')
        st.write('DONE')


        torch.cuda.empty_cache()
