import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap


from model_jingwei.utils.args_loader import get_args
from model_jingwei.exp.exp_OWL import Exp_OWL

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def load_local_npz(file_name):
    with np.load(f"/Users/apagnoux/PycharmProjects/pythonProject2/cache/{file_name}") as data:
        return dict(data)

def load_id_data(file_name):
    # Load ID data
    data = load_local_npz(file_name)
    id_feat = data['feat_log']
    id_label = data['label_log']
    return id_feat, id_label

def load_ood_data(file_name):
    # Load OOD data
    data = load_local_npz(file_name)
    ood_feat = data['ood_feat_log']
    ood_score = data['ood_score_log'][:, 0]
    return ood_feat, ood_score

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


def plot_umap_v2(id_feat, id_label, ood_feat, ood_score):
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
def visuuu():
    st.title(f"T-SNE Visualization of ID vs TEST Embeddings on the same graph")

    available_files = [f for f in os.listdir('/Users/apagnoux/PycharmProjects/pythonProject2/cache/') if f.endswith('.npz')]

    # Allow the user to select a file
    # selected_id_file = st.selectbox('Select the ID data file:', available_files, index=0)
    selected_id_file = 'CIFAR-10_train_resnet18-supcon_in_alllayers.npz'
    # selected_ood_file = st.selectbox('Select the OOD data file:', available_files, index=1)

    ood_files = [f for f in available_files if not f.startswith('CIFAR-10_')]

    # Allow the user to select an OOD file
    selected_ood_file = st.selectbox('Select the OOD data file:', ood_files)

    # Commented out the OOD Detection for now since it's not the focus
    # of your current request

    # if st.button("Show Graph"):
    #     # Load and process the data
    #     id_feat, id_label = load_id_data(selected_id_file)
    #     ood_feat, ood_score = load_ood_data(selected_ood_file)
    #
    #     # Display UMAP visualization
    #     # st.pyplot(plot_tsne(id_feat, id_label, ood_feat, ood_score))
    #     st.pyplot(plot_umap_v2(id_feat, id_label, ood_feat, ood_score))

    visualization_options = ["UMAP", "T-SNE"]
    selected_visualizations = st.multiselect("Choose visualization methods:", visualization_options,
                                             default=visualization_options)

    if st.button("Show Graph"):
        # Load and process the data
        id_feat, id_label = load_id_data(selected_id_file)
        ood_feat, ood_score = load_ood_data(selected_ood_file)

        if "UMAP" in selected_visualizations:
            st.write("UMAP Visualization:")
            st.pyplot(plot_umap_v2(id_feat, id_label, ood_feat, ood_score))

        if "T-SNE" in selected_visualizations:
            st.write("T-SNE Visualization:")
            st.pyplot(plot_tsne(id_feat, id_label, ood_feat, ood_score))

