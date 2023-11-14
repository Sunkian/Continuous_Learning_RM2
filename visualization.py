import os
import pandas as pd
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap


from model_jingwei.utils.args_loader import get_args
from model_jingwei.exp.exp_OWL import Exp_OWL

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def view_data_tab():
    st.title("View Dataset Data")
    available_datasets = fetch_datasets()  # Assuming this function fetches the list of datasets
    selected_dataset = st.selectbox("Select a Dataset", available_datasets)

    if selected_dataset:
        # Fetch data for the selected dataset from your FastAPI backend
        response = requests.get(f"http://localhost:8000/get_data_by_dataset/{selected_dataset}")
        if response.status_code == 200:
            dataset_data = response.json()

            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(dataset_data)
            st.dataframe(df, use_container_width=True)  # Display the data in a DataFrame

        else:
            st.error("Failed to fetch data from the server.")

def fetch_data(dataset_name):
    """Fetches data from the FastAPI backend for a given dataset."""
    response = requests.get(f"http://localhost:8000/get_data_by_dataset/{dataset_name}")
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        return None

def plot_data(df, x_col, y_col):
    """Plots the data based on selected x and y columns."""
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    st.pyplot(fig)

def interactive_plot_tab():
    """Tab for interactive plotting."""
    st.title("Interactive Data Plot")

    available_datasets = fetch_datasets()  # Fetch datasets
    selected_dataset = st.selectbox("Select a Dataset", available_datasets)

    if selected_dataset:
        df = fetch_data(selected_dataset)  # Fetch data for the selected dataset

        if df is not None and not df.empty:
            # Dropdowns for selecting columns
            columns = df.columns
            x_col = st.selectbox("Select X Axis", columns, index=0)
            y_col = st.selectbox("Select Y Axis", columns, index=1)

            if st.button("Plot"):
                plot_data(df, x_col, y_col)
        else:
            st.error("Failed to fetch data or dataset is empty.")




# def load_local_npz(file_name):
#     with np.load("/Users/apagnoux/Downloads/Continuous_Learning_RM2-master/cache/CIFAR-10_val_resnet18-supcon.npz") as data:
#         return dict(data)
# def load_local_npz(file_name):
#     with np.load(f"cache/{file_name}") as data:
#         return dict(data)
#
# def load_id_data(file_name):
#     # Load ID data
#     data = load_local_npz(file_name)
#     id_feat = data['feat_log']
#     id_label = data['label']
#     return id_feat, id_label
#
# def load_ood_data(file_name):
#     # Load OOD data
#     data = load_local_npz(file_name)
#     ood_feat = data['ood_feat_log']
#     ood_score = data['ood_label']  # Note: This is a bit misleading, fthe name 'score' typically doesn't refer to labels
#     return ood_feat, ood_score
def plot_tsne(id_feat, id_label, ood_feat, ood_score):
    # Combine both ID and OOD features for UMAP fitting



    combined_features = np.vstack((id_feat, ood_feat))

    # Use UMAP for dimensionality reduction
    embedding = TSNE(n_components=2).fit_transform(combined_features)

    # Separate ID and OOD embeddings
    embedding_id = embedding[:len(id_feat)]
    embedding_ood = embedding[len(id_feat):]

    fig, ax = plt.subplots(figsize=(10, 8))



    # Plot ID
    scatter_id = ax.scatter(embedding_id[:, 0], embedding_id[:, 1], c=id_label, cmap='tab10', s=5, label='ID')
    # Plot OOD
    scatter_ood = ax.scatter(embedding_ood[:, 0], embedding_ood[:, 1], c=ood_score, cmap='viridis', s=5, label='OOD',
                             alpha=0.6)

    # Create a legend
    ax.legend(handles=[scatter_id, scatter_ood], loc='upper right')
    plt.colorbar(scatter_ood, ax=ax, label='OOD Score')
    ax.set_title("UMAP Visualization of ID vs OOD Features")

    for idx, class_name in enumerate(CLASS_NAMES):
        mean_x = np.mean(embedding_id[id_label == idx, 0])
        mean_y = np.mean(embedding_id[id_label == idx, 1])
        ax.text(mean_x, mean_y, class_name, fontsize=9, ha='center', va='center', backgroundcolor='white')

    return fig


ID_DATA_URL = "http://localhost:8000/get_id_data/"
OOD_DATA_URL = "http://localhost:8000/get_ood_data/"  # dataset_name will be appended

def load_id_data():
    response = requests.get(ID_DATA_URL)
    data = response.json()
    id_feat = np.array(data['id_feat'])
    id_label = np.array(data['id_label'])
    return id_feat, id_label

def load_ood_data(dataset_name):
    response = requests.get(f"{OOD_DATA_URL}{dataset_name}")
    data = response.json()
    # print(data)
    ood_feat = np.array(data['ood_feat'])
    ood_label = np.array(data['ood_label'])
    return ood_feat, ood_label



def plot_umap_v2(id_feat, id_label, ood_feat, ood_label):

    print(f"id_feat shape: {id_feat.shape}")
    print(f"ood_feat shape: {ood_feat.shape}")
    st.write(f"id_feat shape: {id_feat.shape}")
    st.write(f"ood_feat shape: {ood_feat.shape}")
    # Combine both ID and OOD features for UMAP fitting
    combined_features = np.vstack((id_feat, ood_feat))

    # Use UMAP for dimensionality reduction
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(combined_features)

    # Separate ID and OOD embeddings
    embedding_id = embedding[:len(id_feat)]
    embedding_ood = embedding[len(id_feat):]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ID
    scatter_id = ax.scatter(embedding_id[:, 0], embedding_id[:, 1], c='blue', s=5, label='ID')
    # Plot OOD
    scatter_ood = ax.scatter(embedding_ood[:, 0], embedding_ood[:, 1], c='pink', s=5, label='OOD', alpha=0.6)

    # Create a legend
    ax.legend(handles=[scatter_id, scatter_ood], loc='upper right')
    plt.colorbar(scatter_ood, ax=ax, label='OOD Score')
    ax.set_title("UMAP Visualization of ID vs OOD Features")

    for idx, class_name in enumerate(CLASS_NAMES):
        mean_x = np.mean(embedding_id[id_label == idx, 0])
        mean_y = np.mean(embedding_id[id_label == idx, 1])
        ax.text(mean_x, mean_y, class_name, fontsize=9, ha='center', va='center', backgroundcolor='white')

    unique_ood_labels = np.unique(ood_label)
    if len(unique_ood_labels) < 20:  # Example threshold to prevent clutter
        for ood_idx in unique_ood_labels:
            mean_x = np.mean(embedding_ood[ood_label == ood_idx, 0])
            mean_y = np.mean(embedding_ood[ood_label == ood_idx, 1])
            ax.text(mean_x, mean_y, f"OOD_{ood_idx}", fontsize=9, ha='center', va='center', backgroundcolor='white')



    return fig

def fetch_datasets(filter_prefix=None):
    """
    Fetches list of datasets from the FastAPI service.

    Returns:
        list: List of dataset names.
    """
    # Send GET request to the FastAPI endpoint
    response = requests.get("http://localhost:8000/list_datasets/")

    # Raise an exception if there's an HTTP error
    response.raise_for_status()

    # Extract and return the list of datasets
    datasets = response.json().get("datasets", [])

    # Filter the datasets based on the prefix if provided
    if filter_prefix:
        datasets = [dataset for dataset in datasets if dataset.startswith(filter_prefix)]

    return datasets

def visuuu():
    st.title(f"T-SNE Visualization of ID vs TEST Embeddings on the same graph")

    option = st.selectbox('Select an option :',
                          (
                           'Data frames',
                           'Graphs', 'Interactive Plot'))

    # available_files = [f for f in os.listdir('cache/') if f.endswith('.npz')]
    available_files = fetch_datasets()

    ood_files = [f for f in available_files if not f.startswith('CIFAR-10_')]

    if option == 'Graphs':

        st.write("Graphs Visualization")

        # Allow the user to select an OOD file
        selected_ood_file = st.selectbox('Select the OOD data file:', ood_files)



        visualization_options = ["UMAP", "T-SNE"]
        selected_visualizations = st.multiselect("Choose visualization methods:", visualization_options,
                                                 default=visualization_options)

        if st.button("Show Graph"):
            # Load and process the data
            id_feat, id_label = load_id_data()
            ood_feat, ood_score = load_ood_data(selected_ood_file)
            print('OOD FEAT', ood_feat)

            if "UMAP" in selected_visualizations:
                st.write("UMAP Visualization:")
                st.pyplot(plot_umap_v2(id_feat, id_label, ood_feat, ood_score))

            if "T-SNE" in selected_visualizations:
                st.write("T-SNE Visualization:")
                st.pyplot(plot_tsne(id_feat, id_label, ood_feat, ood_score))

    if option == 'Data frames':

        st.write("Tabs Visualization")
        view_data_tab()

    if option == 'Interactive Plot':
        interactive_plot_tab()
