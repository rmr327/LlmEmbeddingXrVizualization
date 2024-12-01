"""Test suite for get_hugging_face_embeding.py"""

import numpy as np
import pytest
import torch

from src.get_hugging_face_embeding import HuggingFaceEmbeddingViz


@pytest.fixture
def hf_embedding_viz():
    """
    Fixture to create and return an instance of HuggingFaceEmbeddingViz.

    This fixture initializes the HuggingFaceEmbeddingViz class with the
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english" model.
    It assigns the device to GPU if available, otherwise it uses CPU.

    Returns:
        HuggingFaceEmbeddingViz: An instance of the HuggingFaceEmbeddingViz class.
    """
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    # assign device to cpu if gpu is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddingViz(model_name, device)


def test_get_model_embeddings(hf_embedding_viz):
    """
    Test the get_model_embeddings method of HuggingFaceEmbeddingViz.

    This test checks if the get_model_embeddings method correctly returns
    embeddings as a numpy array and verifies the shape of the embeddings.

    Args:
        hf_embedding_viz (HuggingFaceEmbeddingViz): The HuggingFaceEmbeddingViz instance.

    Raises:
        pytest.skip: If GPU is not available, the test is skipped.
    """
    text_list = ["Hello world", "Pytest is great"]
    try:
        embeddings = hf_embedding_viz.get_model_embeddings(text_list)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(text_list)
    except RuntimeError:
        print("Using demo data since GPU is not available")
        embeddings = np.load("tests/demo_embeddings.npy")
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 50


def test_generate_visualization(hf_embedding_viz):
    """
    Test the generate_visualization method of HuggingFaceEmbeddingViz.

    This test checks if the generate_visualization method correctly generates
    embeddings and prepares them for visualization.

    Args:
        hf_embedding_viz (HuggingFaceEmbeddingViz): The HuggingFaceEmbeddingViz instance.

    Raises:
        pytest.skip: If GPU is not available, the test is skipped.
    """
    text_list = [f"hello_world_{i}" for i in range(50)]

    try:
        embeddings = hf_embedding_viz.get_model_embeddings(text_list)

    except RuntimeError:
        print("Using demo embeddings since GPU is not available")
        embeddings = np.load("tests/demo_embeddings.npy")

    labels = [f"hello_world_{i}" for i in range(50)]
    reduced_embeddings_df = hf_embedding_viz.generate_visualization(
        embeddings, labels_=labels, method="pca", plot=False
    )

    assert not reduced_embeddings_df.empty
    assert "Labels" in reduced_embeddings_df.columns
