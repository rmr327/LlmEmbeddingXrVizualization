"""This file makes get_huggingface embedding to a command line tool"""

from typing import Optional

import click
import pandas as pd  # pylint: disable=import-error
import torch  # pylint: disable=import-error

from mylib.get_hugging_face_embeding import HuggingFaceEmbeddingViz


@click.command()
@click.option(
    "--model_name",
    "-m",
    default="dunzhang/stella_en_400M_v5",
    help="Name of the HuggingFace model to use.",
)
@click.option(
    "--reduction_method",
    "-r",
    default="umap",
    help="Select dimensionality reduction algo from: 'umap', 'pca', 'tsne', "
    "'isomap', 'mds'.",
)
@click.option(
    "--words",
    "-w",
    default=(
        """Algorithm,Regression,Data Scientist,Feature,Model,Prediction,Clustering,\
Neural Network,Overfitting,Normalization,Equity,Dividend,Derivative,\
Arbitrage,Trader,Bond,Portfolio,Asset,Interest,Yield,Diagnosis,Therapy,\
Immunology,Doctor,Biopsy,Vaccination,Prescription,Surgery,Pathology,\
Cardiology,Encryption,Protocol,Hacker,Processor,Network,Firewall,\
Algorithm,Data Center,API,Cloud Computing,Basketball,Football,Soccer,\
Fencing,Badminton,Tennis,Swimming,Running,Cycling,Volleyball"""
    ),
    help="Comma-separated list of words to embed.",
)
@click.option(
    "--domains",
    "-d",
    default=(
        """Data Science,Data Science,Data Science,Data Science,Data Science,\
Data Science,Data Science,Data Science,Data Science,Data Science,Finance,\
Finance,Finance,Finance,Finance,Finance,Finance,Finance,Finance,Finance,\
Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,\
Healthcare,Healthcare,Healthcare,Healthcare,Technology,Technology,\
Technology,Technology,Technology,Technology,Technology,Technology,\
Technology,Technology,Sports,Sports,Sports,Sports,Sports,Sports,Sports,\
Sports,Sports,Sports"""
    ),
    help="Domain associated with the words.",
)
@click.option(
    "--csv_file",
    "-c",
    default=None,
    help="Path to the CSV file containing words and domains.",
)
@click.option(
    "--output_file",
    "-o",
    default="output.dae",
    help="Path to save the output DAE file.",
)
@click.option(
    "--show_plot", "-s", is_flag=True, help="Show the plot if this flag is set."
)
def main(
    model_name: str,
    reduction_method: str,
    words: str,
    domains: str,
    csv_file: Optional[str],
    **kwargs,
) -> None:
    """Generate a 3D XR visualization of word embeddings from a provided Hugging Face model."""

    print(
        "🚀🚀🚀 Kicking off the transformation of 3D data into a cool XR dae object!🚀🚀🚀"
    )
    output_file = kwargs.get("output_file")

    if csv_file:
        print(
            "\n📄 Reading words and domains from CSV file. Ignoring words and domains arguments..."
        )
        input_df = pd.read_csv(csv_file)

        # if words and domains are not provided, error
        if (  # pylint: disable=no-member
            "words" not in input_df.columns
            or "domains" not in input_df.columns  # pylint: disable=no-member
        ):  # pylint: disable=no-member
            raise ValueError("CSV file must contain 'words' and 'domains' columns.")

        words_list = input_df["words"].tolist()
        domains_list = input_df["domains"].tolist()
    else:
        print("\n\n📝 Reading words and domains from arguments...")
        words_list = words.split(",")
        domains_list = domains.split(",")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("⚠️ Warning: CUDA not available, using CPU. Performance may be slower.⚠️")

    if len(words_list) != len(domains_list):
        raise ValueError("Number of words/sentences and domains must be the same.")

    if len(words_list) < 6:
        raise ValueError("Number of words/sentences must be at least 6.")

    print("🛠️ Initializing Hugging Face Model (expect some output below)...")
    print(f"\n        🔌 Using model: {model_name}🔌\n\n")
    viz = HuggingFaceEmbeddingViz(model_name=model_name, device_=device)
    print("\n\n🧠 Getting model embeddings for words/sentences...")
    embeddings = viz.get_model_embeddings(words_list)
    print(
        f"📈 Generating reduced embeddings for words/sentences using {reduction_method}..."
    )
    reduced_embeddings = viz.generate_visualization(
        embeddings,
        labels_=words_list,
        color_=domains_list,
        method=reduction_method,
        plot=kwargs.get("show_plot", False),
    )
    print(f"📊 Generating & Saving 3D Object to Dae file {output_file}... \n")
    # normalize the first three columns between 0 and 5
    viz.generate_3d_visualization(reduced_embeddings, output_file)
    print(f"\n💾 Generated Dae file saved to {output_file}🎁")
    print("✅ Done!\n")

    print(
        """ℹ️ please take the generated dae file and upload it to xr visualization \
    tool ("eg. Sketchup") to see the 3D object. If Sketchup is used, the \
    resulting XR environment can be uploaded to the cloud and viewed from Meta \
    Quest Headsets."""
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
