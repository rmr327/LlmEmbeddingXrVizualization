# LlmEmbeddingXrVizualization
[![Python package](https://github.com/rmr327/LlmEmbeddingXrVizualization/actions/workflows/python-package.yml/badge.svg)](https://github.com/rmr327/LlmEmbeddingXrVizualization/actions/workflows/python-package.yml)

A package for visualizing Large Language Model (LLM) embedding spacese from Hugging Face models with just the model name as input!

Inspired by the belief that data should be experienced, not just viewed, we're bridging the gap between 2D plots and spatial understanding in the LLM embeddings space. The fundamental limitation of 2D screens - trying to compress three dimensions into two - has always forced us to sacrifice either information or clarity. Our platform breaks free from these constraints, transforming raw datasets into immersive XR visualizations using nothing but the name of the model from Hugging Face. Every visualization is accessible on your Meta Quest XR Headsets. We're not just plotting data - we're creating a new way to discover insights through spatial exploration, one that respects the true dimensionality of our data.

Each word/sentece embedding is meticulously positioned in virtual space, ensuring perfect spatial accuracy and true-to-scale representation. This precision becomes particularly powerful when visualizing LLM embedding spaces - allowing users to physically explore how concepts are related within these models. By walking through the three-dimensional embedding space, researchers can intuitively verify if semantically similar concepts cluster together and identify unexpected relationships that traditional 2D visualizations might miss.

## Installation

```bash
pip install LlmEmbeddingXrVizualization
```

## Usage
```bash
llm-embedding-viz --help
```
![image](https://github.com/user-attachments/assets/4586bcf0-1d03-441d-9cda-cb4a7f6a43c0)

```bash
llm-embedding-viz
```
![image](https://github.com/user-attachments/assets/0ed8ddd7-be71-4724-b25e-90c53a100e8c)

> example website to open the generated 3d object ('.dae file').

![image](https://github.com/user-attachments/assets/8da4f88a-72ce-46c2-b699-048fb0d8d1d5)

> example experience on meta quest 3

![PlotVerseXR_Trailer](https://github.com/user-attachments/assets/7c76cee8-7476-45ec-b482-6213618176d0)

![PlotVerseXR_Trailer (1)](https://github.com/user-attachments/assets/26903be9-2e82-4421-98bb-ca8adfb96157)



```bash
llm-embedding-viz --model_name "distilbert/distilbert-base-uncased-finetuned-sst-2-english" -c path_to_ur_labels_domains.csv -r isomap -s"
```
> The csv file must have 'domains' and 'words' columns.

![image](https://github.com/user-attachments/assets/02749a52-cb1c-460b-8393-2ef347f65a70)

> generated plot for -s flag

![image](https://github.com/user-attachments/assets/1c332560-e9f8-463a-be2c-095c77f77a1c)

## References
This idea started in a Hacathon: https://devpost.com/software/plotversexr.

Generative Ai such as Github Copilot and Chat GPT was used extensively in this project. 

Duke University Xplainable Ai Class: AIPI 590.

