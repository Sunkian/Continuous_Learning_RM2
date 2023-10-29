import streamlit as st
import requests
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

def run_ft():
    option = st.selectbox('Select an option :',
                              ('Inference',
                               'Fine-Tune'))

    datasets = fetch_datasets()

    batch = st.selectbox('Select an data batch :',
                          datasets)


print('HELLO')