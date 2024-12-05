"""Streamlit app for LLM Embedding Visualization."""

import re
import subprocess

import pandas as pd
import streamlit as st  # pylint: disable=wrong-import-order, import-error
import torch

# Constants
DEFAULT_MODEL = "dunzhang/stella_en_400M_v5"
DEFAULT_OUTPUT_FILE = "output.dae"
REDUCTION_METHODS = ["umap", "pca", "tsne", "isomap", "mds"]


# Function to remove extra spaces
def remove_extra_spaces(text):
    """Function to remove extra spaces from text."""
    return re.sub(r"\s+", " ", text)


# Main title
st.markdown(
    "<h1 style='text-align: center;'>🕶️ PlotVerseXR 🕶️</h1>", unsafe_allow_html=True
)
st.markdown(
    """<h2 style='text-align: center; color: #FF5733;'
    >Visualize 3D Data like  it's meant to be seen!</h2>""",
    unsafe_allow_html=True,
)

# Step 1: Model Input
st.header("Step 1: Model Input")
model_name = st.text_input(
    "Enter the Hugging Face model name or path:",
    value=DEFAULT_MODEL,
    max_chars=200,
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    st.warning("CUDA not available, using CPU. Performance may be slower.")

# Step 2: Output File Path
st.header("Step 2: Output File Path")
output_file_path = st.text_input(
    "Enter the output file path for the .dae file:",
    value=DEFAULT_OUTPUT_FILE,
)

# Step 3: Choose Input Type
st.header("Step 3: Choose Input Type")
input_type = st.selectbox(
    "Select the input type:",
    options=["Words", "Sentences", "Upload CSV"],
)

# Step 4: Dimensionality Reduction Algorithm
st.header("Step 4: Dimensionality Reduction Algorithm")
selected_algorithm = st.selectbox(
    "Select a dimensionality reduction algorithm:",
    options=REDUCTION_METHODS,
    index=REDUCTION_METHODS.index("umap"),
)


# Functions for processing input
def process_words_input():
    """Function to process words input in Streamlit."""
    st.subheader("Words Input")
    words = st.text_area(
        "Enter words (comma-separated, no spaces between):",
        value="""Algorithm,Regression,Data Scientist,Feature,Model,Prediction,\
Clustering,Neural-Network,Overfitting,Normalization,Equity,Dividend,Derivative,\
Arbitrage,Trader,Bond,Portfolio,Asset,Interest,Yield,\
Diagnosis,Therapy,Immunology,Doctor,Biopsy,Vaccination,Prescription,\
Surgery,Pathology,Cardiology,Encryption,Protocol,Hacker,Processor,\
Network,Firewall,Algorithm,Data Center,API,Cloud-Computing,Basketball,\
Football,Soccer,Fencing,Badminton,Tennis,Swimming,Running,Cycling,Volleyball""",
    )
    domains = st.text_area(
        "Enter domains (comma-separated, same order as words, no spaces between):",
        value="""Data Science,Data Science,Data Science,Data Science,Data Science,\
Data Science,Data Science,Data Science,Data Science,Data Science,Finance,Finance,\
Finance,Finance,Finance,Finance,Finance,Finance,Finance,Finance,Healthcare,Healthcare,\
Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,\
Technology,Technology,Technology,Technology,Technology,Technology,Technology,Technology,\
Technology,Technology,Sports,Sports,Sports,Sports,Sports,Sports,Sports,Sports,Sports,\
Sports""",
    )
    if st.button("Generate Embeddings and Export"):
        words_list = words.split(",")
        domains_list = domains.split(",")
        if len(words_list) != len(domains_list):
            st.error("The number of words and domains must match.")
        if len(domains_list) < 6:
            st.error("At least 5 words are required.")
        else:
            # run bash command to generate embeddings
            # Define the bash command as a string
            bash_command = f"""llm-embedding-viz -m {model_name} \
            -r {selected_algorithm} -o {output_file_path} -w '{words}' -d '{domains}' -s"""
            # Run the bash command
            bash_command = remove_extra_spaces(bash_command)
            process = subprocess.run(
                bash_command, shell=True, check=True, text=True, capture_output=True
            )

            # Print the output and error (if any)
            st.write("Error:", process.stderr)

            st.success(f"Output exported to {output_file_path}")


def process_sentences_input():
    """Function to process sentences input in Streamlit."""
    st.subheader("Sentences Input")
    text_input = st.text_area(
        "Enter sentences (comma-separated, no spaces between):",
        value="""data science is cool,data science is fun,data science is awesome,\
i will play basketball, I will play football, I will watch soccer""",
    )

    domains = st.text_area(
        "Enter domains (comma-separated, same order as words, no spaces between):",
        value="Data Science,Data Science,Data Science,Sports,Sports,Sports",
    )
    if st.button("Process Text and Export"):
        domains_s = domains.split(",")
        if len(text_input.split(",")) != len(domains_s):
            st.error("The number of words and domains must match.")
        if len(domains_s) < 6:
            st.error("At least 5 sentences are required.")
        else:
            bash_command = f"""llm-embedding-viz -m {model_name} \
            -r {selected_algorithm} -o {output_file_path} \
            -w '{text_input}' -d '{domains}' -s"""
            bash_command = remove_extra_spaces(bash_command)
            process = subprocess.run(
                bash_command, shell=True, check=True, text=True, capture_output=True
            )
            st.write("Error:", process.stderr)
            st.success(f"Output exported to {output_file_path}")


def process_csv_input():
    """Function to process CSV input in Streamlit."""
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.to_csv("uploaded_file.csv", index=False)  # pylint: disable=no-member
        st.write("Uploaded Data Head:", data.head())  # pylint: disable=no-member

        if data.shape[0] < 6:  # pylint: disable=no-member
            st.error("At least 5 rows are required.")

        # if uploaded file doesnt have words and domains columns give error
        if (
            "words" not in data.columns  # pylint: disable=no-member
            or "domains" not in data.columns  # pylint: disable=no-member
        ):
            st.error("CSV file must have 'words' and 'domains' columns.")

        if st.button("Process CSV and Export"):
            bash_command = f"""llm-embedding-viz -m {model_name} \
            -r {selected_algorithm} -o {output_file_path} -c uploaded_file.csv -s"""
            bash_command = remove_extra_spaces(bash_command)
            process = subprocess.run(
                bash_command, shell=True, check=True, text=True, capture_output=True
            )
            st.write("Error:", process.stderr)
            st.success(f"Output exported to {output_file_path}")


# Main logic based on input type
if input_type == "Words":
    process_words_input()
elif input_type == "Sentences":
    process_sentences_input()
elif input_type == "Upload CSV":
    process_csv_input()
