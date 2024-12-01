from transformers import AutoModel, AutoTokenizer
import torch
from typing import List, Optional
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
import warnings

# stop future warning
warnings.filterwarnings("ignore")

# Constants
N_COMPONENTS = 2
PERPLEXITY = 30
N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_STATE = 42


class HuggingFaceEmbeddingViz:
    def __init__(self, model_name: str, device: torch.device = torch.device("cpu")):
        """This class is used to extract embeddings from the provided Hugging Face model"""
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model_name = model_name
        self.device = device

    def get_model_embeddings(self, text_list: List[str]) -> np.ndarray:
        """This function is used to extract embeddings for the passed text as defined by the LLM models embedding space"""
        embeddings = []

        with torch.no_grad():
            for entry in text_list:
                input_data = self.tokenizer(
                    entry, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)
                input_data = {k: v.cuda() for k, v in input_data.items()}
                last_hidden_state = self.model(**input_data).last_hidden_state
                sentence_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()
                embeddings.append(sentence_embedding)

        return np.array(embeddings)

    def generate_visualization(
        self,
        embeddings: np.ndarray,
        labels_: Optional[List[str]] = None,
        color_: Optional[List[str]] = None,
        method: str = "pca",
        plot: bool = False,
    ) -> pd.DataFrame:
        """This function is used to generate a visualization of the embedding space of the LLM model, using the plotly library"""
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

    def _plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels_: Optional[List[str]],
        color_: Optional[List[str]],
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
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

    # Initialize the class
    hugging_model = "dunzhang/stella_en_400M_v5"
    hf_embedding_viz = HuggingFaceEmbeddingViz(hugging_model, device)

    # Generate word embeddings for stella model
    embeddings = hf_embedding_viz.get_model_embeddings(words)

    # Generate PCA visualization
    reduced_embeddings_pca = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, color_=domains, method="pca", plot=False
    )

    # Generate TSNE visualization
    reduced_embeddings_tsne = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, color_=domains, method="tsne", plot=False
    )

    # Generate UMAP visualization
    reduced_embeddings_umap = hf_embedding_viz.generate_visualization(
        embeddings, labels_=words, color_=domains, method="umap", plot=True
    )
