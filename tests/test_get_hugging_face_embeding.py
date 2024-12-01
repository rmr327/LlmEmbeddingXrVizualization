import pytest
import torch
import numpy as np
import pandas as pd
from src.get_hugging_face_embeding import HuggingFaceEmbeddingViz


@pytest.fixture
def hf_embedding_viz():
    model_name = "dunzhang/stella_en_400M_v5"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return HuggingFaceEmbeddingViz(model_name, device)


def test_get_model_embeddings(hf_embedding_viz):
    words = ["Algorithm", "Regression", "Data Scientist"]
    embeddings = hf_embedding_viz.get_model_embeddings(words)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, hf_embedding_viz.model.config.hidden_size)


def test_generate_visualization_pca(hf_embedding_viz):
    words = ["Algorithm", "Regression", "Data Scientist"]
    embeddings = hf_embedding_viz.get_model_embeddings(words)
    reduced_embeddings_df = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, method="pca", plot=False
    )
    assert isinstance(reduced_embeddings_df, pd.DataFrame)
    assert reduced_embeddings_df.shape == (3, 3)  # 2 components + labels


def test_generate_visualization_tsne(hf_embedding_viz):
    words = ["Algorithm", "Regression", "Data Scientist"]
    embeddings = hf_embedding_viz.get_model_embeddings(words)
    reduced_embeddings_df = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, method="tsne", plot=False
    )
    assert isinstance(reduced_embeddings_df, pd.DataFrame)
    assert reduced_embeddings_df.shape == (3, 3)  # 2 components + labels


def test_generate_visualization_umap(hf_embedding_viz):
    words = ["Algorithm", "Regression", "Data Scientist"]
    embeddings = hf_embedding_viz.get_model_embeddings(words)
    reduced_embeddings_df = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, method="umap", plot=False
    )
    assert isinstance(reduced_embeddings_df, pd.DataFrame)
    assert reduced_embeddings_df.shape == (3, 3)  # 2 components + labels
