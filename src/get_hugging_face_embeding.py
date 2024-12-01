"""This module is used to extract embeddings from the provided Hugging Face model"""

import warnings
from typing import List, Optional

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import plotly.express as px  # pylint: disable=import-error
import torch  # pylint: disable=import-error
import umap  # pylint: disable=import-error
from sklearn.decomposition import PCA  # pylint: disable=import-error
from sklearn.manifold import TSNE  # pylint: disable=import-error
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
        """This class is used to extract embeddings from the provided Hugging Face model"""

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
        defined by the LLM models embedding space"""
        if self.device.type == "cpu":
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

    def generate_visualization(  # pylint: disable=too-many-arguments
        self,  # pylint: disable=too-many-arguments
        embeddings: np.ndarray,  # pylint: disable=too-many-arguments
        labels_: Optional[List[str]] = None,  # pylint: disable=too-many-arguments
        color_: Optional[List[str]] = None,  # pylint: disable=too-many-arguments
        method: str = "pca",  # pylint: disable=too-many-arguments
        plot: bool = False,  # pylint: disable=too-many-arguments
    ) -> pd.DataFrame:  # pylint: disable=too-many-arguments
        """This function is used to generate a visualization of the
        embedding space of the LLM model, using the plotly library"""
        reducer, title, x_label, y_label = self._get_reducer(method, labels_)
        reduced_embeddings = reducer.fit_transform(embeddings)

        if plot:
            self._plot_embeddings(
                reduced_embeddings, labels_, color_, title, x_label, y_label
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
        else:
            raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")
        return reducer, title, x_label, y_label

    @staticmethod
    def _plot_embeddings(  # pylint: disable=too-many-arguments
        embeddings: np.ndarray,  # pylint: disable=too-many-arguments
        labels_: Optional[List[str]],  # pylint: disable=too-many-arguments
        color_: Optional[List[str]],  # pylint: disable=too-many-arguments
        title: str,  # pylint: disable=too-many-arguments
        x_label: str,  # pylint: disable=too-many-arguments
        y_label: str,  # pylint: disable=too-many-arguments
    ) -> None:  # pylint: disable=too-many-arguments
        """Helper function to plot embeddings using Plotly"""
        fig = px.scatter(
            embeddings,
            x=0,
            y=1,
            text=labels_,
            title=title,
            labels={"0": x_label, "1": y_label},
            color=color_,
        )
        fig.update_traces(marker=dict(size=8))
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
    # HUGGING_MODEL = "facebook/# bart-large"
    HUGGING_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

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
        embeddings_, labels_=words, color_=domains, method="umap", plot=True
    )
