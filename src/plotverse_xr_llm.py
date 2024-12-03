"""This file makes get_huggingface embedding to a command line tool"""

import click
import torch  # pylint: disable=import-error
from get_hugging_face_embeding import HuggingFaceEmbeddingViz


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
    default="pca",
    help="Dimensionality reduction method to use.",
)
@click.option(
    "--words",
    "-w",
    default="Algorithm,Regression,Data Scientist,Feature,Model'\
        ',Prediction,Clustering,Neural Network,Overfitting,Normalization,'\
        'Equity,Dividend,Derivative,Arbitrage,Trader,Bond,Portfolio,Asset,'\
            'Interest,Yield,Diagnosis,Therapy,Immunology,Doctor,Biopsy,'\
                'Vaccination,Prescription,Surgery,Pathology,Cardiology,'\
                    'Encryption,Protocol,Hacker,Processor,Network,'\
                        'Firewall,Algorithm,Data Center,API,Cloud Computing,'\
                            'Basketball,Football,Soccer,Fencing,Badminton,'\
                                'Tennis,Swimming,Running,Cycling,Volleyball",
    help="Comma-separated list of words to embed.",
)
@click.option(
    "--domains",
    "-d",
    default="Data Science,Data Science,Data Science,Data Science,Data Science,'\
        'Data Science,Data Science,Data Science,Data Science,Data Science,'\
            'Finance,Finance,Finance,Finance,Finance,Finance,Finance,Finance,'\
                'Finance,Finance,Healthcare,Healthcare,Healthcare,Healthcare,'\
                    'Healthcare,Healthcare,Healthcare,Healthcare,Healthcare,'\
                        'Healthcare,Technology,Technology,Technology,'\
                            'Technology,Technology,Technology,Technology,'\
                                'Technology,Technology,Technology,Sports,'\
                                    'Sports,Sports,Sports,Sports,Sports,'\
                                        'Sports,Sports,Sports,Sports",
    help="Domain associated the words.",
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
    **kwargs,
) -> None:
    """Generate a 3D visualization of word embeddings."""
    show_plot = kwargs.get("show_plot")
    output_file = kwargs.get("output_file")

    words_list = words.split(",")
    domains_list = [domains] * len(words_list)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    viz = HuggingFaceEmbeddingViz(model_name=model_name, device_=device)
    embeddings = viz.get_model_embeddings(words_list)
    reduced_embeddings = viz.generate_visualization(
        embeddings,
        labels_=words_list,
        color_=domains_list,
        method=reduction_method,
        plot=show_plot,
    )
    viz.generate_3d_visualization(reduced_embeddings, output_file)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
