from transformers import AutoModel, AutoTokenizer
import torch
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd

# stop future warning
import warnings

warnings.filterwarnings("ignore")


class HuggingFaceEmbeddingViz:
    def __init__(self, model_name, device=torch.device("cpu")):
        """This class is used to extract embeddings from the privded
        Hugging Face model"""
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model_name = model_name
        self.device = device

    def get_model_embeddings(self, text_list: List[str]) -> np.array:
        """This function is used to extract embeddings for the passed text as defined by the
        LLM models embedding space"""
        # Collect embeddings
        embeddings = []

        with torch.no_grad():
            for entry in text_list:
                # Tokenize the text entry
                input_data = self.tokenizer(
                    entry, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)

                input_data = {k: v.cuda() for k, v in input_data.items()}

                # Forward pass through the model
                last_hidden_state = self.model(**input_data).last_hidden_state

                # Extract the embedding representing the word/sentence
                sentence_embedding = last_hidden_state[:, 0, :].cpu().numpy().flatten()

                # Appending to the list of embeddings
                embeddings.append(sentence_embedding)

        return np.array(embeddings)

    def generate_pca_visualization(
        self, embeddings: np.array, labels_=None, color_=None
    ) -> pd.DataFrame:
        """This funcion is used to generate a PCA visualization of the embedding space
        of of the LLM model, using the plotly library"""
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # Plot PCA results using Plotly for interactivity
        fig_pca = px.scatter(
            reduced_embeddings,
            x=0,
            y=1,
            text=labels_,
            title=f"PCA of {self.model_name} embedding space",
            labels={"0": "Principal Component 1", "1": "Principal Component 2"},
            color=color_,
        )

        fig_pca.update_traces(marker=dict(size=8))

        # making the xaxis a little wider
        fig_pca.update_xaxes(range=[-20, 20])

        fig_pca.show()

        return reduced_embeddings


if __name__ == "__main__":
    # Assign device to make sur GPU is used when available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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

    # initialize the class
    hugging_model = "dunzhang/stella_en_400M_v5"

    # different models can be used
    # hugging_model = "bert-base-uncased"

    hf_embedding_viz = HuggingFaceEmbeddingViz(hugging_model, device)

    # generate  word embeddings for stella model
    embeddings = hf_embedding_viz.get_model_embeddings(words)

    # generate PCA visualization
    reduced_embeddings = hf_embedding_viz.generate_pca_visualization(
        embeddings, labels_=words, color_=domains
    )

    pd.DataFrame(reduced_embeddings).to_clipboard(index=False)
