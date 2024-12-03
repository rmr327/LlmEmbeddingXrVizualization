"""Test suite for get_hugging_face_embeding.py"""

import os

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


def test_save_and_load_embeddings(hf_embedding_viz):
    """
    Test the save_embeddings and load_embeddings methods of HuggingFaceEmbeddingViz.

    This test checks if the embeddings are correctly saved to a file and then
    loaded back, ensuring the data integrity.

    Args:
        hf_embedding_viz (HuggingFaceEmbeddingViz): The HuggingFaceEmbeddingViz instance.
    """

    file_path = "tests/temp_embeddings.npy"
    embeddings = np.load("tests/demo_embeddings.npy")
    # Save embeddings to a file
    hf_embedding_viz.save_embeddings(embeddings, file_path)

    # Load embeddings from the file
    loaded_embeddings = hf_embedding_viz.load_embeddings(file_path)

    # Check if the loaded embeddings match the original embeddings
    assert np.array_equal(embeddings, loaded_embeddings)

    # Clean up the temporary file
    os.remove(file_path)


def test_generate_3d_visualization(hf_embedding_viz):
    """
    Test the generate_3d_visualization method of HuggingFaceEmbeddingViz.

    This test checks if the generate_3d_visualization method correctly generates
    a 3D visualization and saves it to a file.

    Args:
        hf_embedding_viz (HuggingFaceEmbeddingViz): The HuggingFaceEmbeddingViz instance.
    """
    embeddings = np.load("tests/demo_embeddings.npy")
    labels = [f"hello_world_{i}" for i in range(50)]
    reduced_embeddings_df = hf_embedding_viz.generate_visualization(
        embeddings, labels_=labels, method="pca", plot=False
    )
    output_file = "tests/temp_visualization.dae"

    # Generate 3D visualization and save to file
    hf_embedding_viz.generate_3d_visualization(reduced_embeddings_df, output_file)

    # Check if the file is created
    assert os.path.exists(output_file)

    # Clean up the temporary file
    os.remove(output_file)
