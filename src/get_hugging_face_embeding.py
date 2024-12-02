"""This module is used to extract embeddings from the provided Hugging Face model"""

import os
import warnings
from typing import List, Optional

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import plotly.express as px  # pylint: disable=import-error
import torch  # pylint: disable=import-error
import umap  # pylint: disable=import-error
from sklearn.decomposition import PCA  # pylint: disable=import-error
from sklearn.manifold import MDS, TSNE, Isomap  # pylint: disable=import-error
from transformers import AutoModel, AutoTokenizer  # pylint: disable=import-error

# stop future warning
warnings.filterwarnings("ignore")

# Constants
N_COMPONENTS = 2
PERPLEXITY = 30
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42


class HuggingFaceEmbeddingViz:
    """This class is used to extract embeddings from the provided Hugging Face model"""

    def __init__(self, model_name: str, device_: torch.device = torch.device("cpu")):
        """This class is used to extract embeddings from the provided Hugging Face model

        Args:
            model_name (str): The name of the Hugging Face model to use
            device_ (torch.device, optional): The device to use for the model.
            Defaults to torch.device("cpu")
        """

        try:
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(device_)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

        except (OSError, ValueError) as e_err:
            raise RuntimeError(
                f"Failed to load model or tokenizer for {model_name}: {e_err}"
            ) from e_err

        self.model_name = model_name
        self.device = device_

    def get_model_embeddings(self, text_list: List[str]) -> np.ndarray:
        """This function is used to extract embeddings for the passed text as
        defined by the LLM models embedding space

        Args:
            text_list (List[str]): The list of text to extract embeddings from

        Returns:
            np.ndarray: The embeddings of the text

        """
        if self.device.type == "cpu" and not os.getenv("TEST_ENV"):
            raise RuntimeError("This model requires a CUDA-enabled GPU to run.")

        embeddings = []
        # Dynamically set max_length based on the model
        max_length = self.tokenizer.model_max_length

        with torch.no_grad():
            for entry in text_list:
                input_data = self.tokenizer(
                    entry, return_tensors="pt", truncation=True, max_length=max_length
                ).to(self.device)
                input_data = {k: v.to(self.device) for k, v in input_data.items()}
                last_hidden_state = self.model(**input_data).last_hidden_state
                sentence_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings.append(sentence_embedding)

        return np.array(embeddings)

    def generate_visualization(
        self,
        embeddings: np.ndarray,
        labels_: Optional[List[str]] = None,
        color_: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """This function is used to generate a visualization of the
        embedding space of the LLM model, using the plotly library

        Args:
            embeddings (np.ndarray): The embeddings to visualize
            labels_ (Optional[List[str]], optional): Labels for the embeddings. Defaults to None.
            color_ (Optional[List[str]], optional): Color for the embeddings. Defaults to None.
            **kwargs: Additional keyword arguments
                - method (str): The method to use for visualization. Defaults to "umap".
                - plot (bool): Whether to plot the embeddings. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the reduced embeddings
        """

        method = kwargs.get("method", "umap")

        reducer, title, x_label, y_label = self._get_reducer(method, labels_)
        reduced_embeddings = reducer.fit_transform(embeddings)

        plot = kwargs.get("plot", False)

        if plot:
            self._plot_embeddings(
                reduced_embeddings,
                labels_,
                color_,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )

        reduced_embeddings_df = pd.DataFrame(
            reduced_embeddings, columns=[x_label, y_label]
        )
        if labels_ is not None:
            reduced_embeddings_df["Labels"] = labels_

        return reduced_embeddings_df

    def _get_reducer(self, method: str, labels_: Optional[List[str]]) -> tuple:
        """Helper function to get the appropriate reducer and labels for the visualization method"""
        if method == "pca":
            reducer = PCA(n_components=N_COMPONENTS)
            title = f"PCA of {self.model_name} embedding space"
            x_label = "Principal Component 1"
            y_label = "Principal Component 2"
        elif method == "tsne":
            if len(labels_) > PERPLEXITY:
                perplexity = PERPLEXITY
            else:
                perplexity = len(labels_) - 1

            reducer = TSNE(
                n_components=N_COMPONENTS,
                perplexity=perplexity,
                n_iter=300,
                random_state=RANDOM_STATE,
            )
            title = f"t-SNE of {self.model_name} embedding space"
            x_label = "Component 1"
            y_label = "Component 2"
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=N_COMPONENTS,
                n_neighbors=N_NEIGHBORS,
                min_dist=MIN_DIST,
                random_state=RANDOM_STATE,
            )
            title = f"UMAP of {self.model_name} Embeddings"
            x_label = "Component 1"
            y_label = "Component 2"
        elif method == "mds":
            reducer = MDS(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
            title = f"MDS of {self.model_name} embedding space"
            x_label = "Component 1"
            y_label = "Component 2"
        elif method == "isomap":
            reducer = Isomap(n_components=N_COMPONENTS)
            title = f"Isomap of {self.model_name} embedding space"
            x_label = "Component 1"
            y_label = "Component 2"
        else:
            raise ValueError(
                "Invalid method. Choose from 'pca', 'tsne', 'umap', 'mds', or 'isomap'."
            )
        return reducer, title, x_label, y_label

    @staticmethod
    def _plot_embeddings(
        embeddings: np.ndarray,
        labels_: Optional[List[str]],
        color_: Optional[List[str]],
        **kwargs,
    ) -> None:
        """Helper function to plot embeddings using Plotly

        Args:
            embeddings (np.ndarray): The embeddings to plot
            labels_ (Optional[List[str]]): Labels for the embeddings
            color_ (Optional[List[str]]): Color for the embeddings
            **kwargs: Additional keyword arguments
                - title (str): The title of the plot. Defaults to "Embeddings Visualization".
                - x_label (str): The x-axis label. Defaults to "Component 1".
                - y_label (str): The y-axis label. Defaults to "Component 2".

        Returns:
            None
        """

        # Set default values of the plot
        title = kwargs.get("title", "Embeddings Visualization")
        x_label = kwargs.get("x_label", "Component 1")
        y_label = kwargs.get("y_label", "Component 2")

        fig = px.scatter(
            embeddings,
            x=0,
            y=1,
            text=labels_,
            title=title,
            labels={"0": x_label, "1": y_label},
            color=color_,
        )
        fig.update_traces(marker={"size": 8})
        fig.show()


if __name__ == "__main__":

    # Assign device to make sure GPU is used when available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU. Performance may be slower.")

    # Words List (Organized by Domain)
    words = [
        # Data Science
        "Algorithm",
        "Regression",
        "Data Scientist",
        "Feature",
        "Model",
        "Prediction",
        "Clustering",
        "Neural Network",
        "Overfitting",
        "Normalization",
        # Finance
        "Equity",
        "Dividend",
        "Derivative",
        "Arbitrage",
        "Trader",
        "Bond",
        "Portfolio",
        "Asset",
        "Interest",
        "Yield",
        # Healthcare
        "Diagnosis",
        "Therapy",
        "Immunology",
        "Doctor",
        "Biopsy",
        "Vaccination",
        "Prescription",
        "Surgery",
        "Pathology",
        "Cardiology",
        # Technology
        "Encryption",
        "Protocol",
        "Hacker",
        "Processor",
        "Network",
        "Firewall",
        "Algorithm",
        "Data Center",
        "API",
        "Cloud Computing",
        # Sports
        "Basketball",
        "Football",
        "Soccer",
        "Fencing",
        "Badminton",
        "Tennis",
        "Swimming",
        "Running",
        "Cycling",
        "Volleyball",
    ]

    # Domains List
    domains = (
        ["Data Science"] * 10
        + ["Finance"] * 10
        + ["Healthcare"] * 10
        + ["Technology"] * 10
        + ["Sports"] * 10
    )

    # Hugging Face Model examples
    # HUGGING_MODEL = "dunzhang/stella_en_400M_v5"
    # HUGGING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    HUGGING_MODEL = "facebook/bart-large"
    # HUGGING_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    # Initialize the class
    hf_embedding_viz = HuggingFaceEmbeddingViz(HUGGING_MODEL, device)

    # Generate word embeddings for stella model
    embeddings_ = hf_embedding_viz.get_model_embeddings(words)

    # Generate PCA visualization
    reduced_embeddings_pca = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="pca", plot=False
    )

    # Generate TSNE visualization
    reduced_embeddings_tsne = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="tsne", plot=False
    )

    # Generate UMAP visualization
    reduced_embeddings_umap = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="umap", plot=False
    )

    # Generate MDS visualization
    reduced_embeddings_mds = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="mds", plot=False
    )

    # Generate Isomap visualization
    reduced_embeddings_isomap = hf_embedding_viz.generate_visualization(
        embeddings_, labels_=words, color_=domains, method="isomap", plot=True
    )
