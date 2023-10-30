import streamlit as st
import requests
import argparse
import requests
from model_jingwei.utils.args_loader import get_args
from model_jingwei.exp.exp_OWL import Exp_OWL
from model_jingwei.exp.exp_OWL import Exp_OWL

import torch

BASE_API_URL = "http://127.0.0.1:8000"


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


def run_ft():
    parser = argparse.ArgumentParser()
    args = get_args()

    exp = Exp_OWL(args)  # set experiments

    if st.button('Extract ID features'):
        exp.id_feature_extract(exp.model, args.in_dataset)
        # st.success('In-distribution features extracted successfully !')
        # response = requests.post(f"{BASE_API_URL}/store_metadata/")
        # if response.status_code == 200:
        #     st.success(response.json().get("status", "Metadata stored successfully!"))
        # else:
        #     st.error("Failed to store metadata!")
        st.write('Id features put in metadata')

    # option = st.selectbox('Select an option :',
    #                       ('Inference',
    #                        'Fine-Tune'))
    #
    # datasets = fetch_datasets()
    #
    # batch = st.selectbox('Select an data batch :',
    #                      datasets)
    # if st.button('Run'):
    #     if option == 'Inference':
    #         st.write('Inference selected')
