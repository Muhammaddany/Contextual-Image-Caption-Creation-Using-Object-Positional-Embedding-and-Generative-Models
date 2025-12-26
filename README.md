# Contextual Image Caption Creation Using Object Positional Embedding and Generative Models
## Overview
This research presents a novel approach to image captioning by integrating object detection, spatial relationship modeling, and advanced language generation. The system leverages YOLOv5 for precise object detection, constructs scene graphs to capture spatial relationships, and utilizes Generative model to generate contextually rich captions. In our scenario, caption generation is the assigned task. Multiple models are integrated with the state-of-the-art language generator transformer. Performance is evaluated using standard automatic metrics as well as a human expert survey to assess caption relevance and fluency. The survey link is included at the end of this repository.

**Getting Started**
To start working with this project, follow these steps:

## Dataset Selection & Reproducibility
This study is based on two publicly available benchmark datasets:
- MS COCO: https://cocodataset.org/#download
- Flickr8k: https://www.kaggle.com/datasets/adityajn105/flickr8k

To ensure controlled and reproducible evaluation, a subset of 500 images was curated. The corresponding ground-truth reference captions are provided in: `data/subset_images_reference_captions.csv/`

## Features
* **Object Detection:** Utilizes YOLOv5 for accurate object detection. 

* **Scene Graph Generation:** Constructs spatial relationships between detected objects. 

* **Caption Generation:** Transformer-based generative models for contextual captions.
  
* **Evaluation:** Automatic metrics and human expert assessment.

* **Cross-Platform Compatibility:** Windows, macOS, and Linux.

## Installation

### On Linux/macOS
    git clone https://github.com/Muhammaddany/Contextual-Image-Caption-Creation-Using-Object-Positional-Embedding-and-Generative-Models.git
    cd Contextual-Image-Caption-Creation-Using-Object-Positional-Embedding-and-Generative-Models
    pip install -r requirements.txt

    mkdir -p model && cd model
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names


### On Windows

    git clone https://github.com/Muhammaddany/Contextual-Image-Caption-Creation-Using-Object-Positional-Embedding-and-Generative-Models.git
    cd Contextual-Image-Caption-Creation-Using-Object-Positional-Embedding-and-Generative-Models
    pip install -r requirements.txt

    kdir model
    cd model
    curl -O https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_optimal/yolov5.weights
    curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov5.cfg
    curl -O https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names


## Dependencies
Key dependencies include:
 * numpy
 * pandas
 * torch, torchvision
 * opencv-python
 * networkx
 * nltk
 * rouge-score
 * pycocotools, pycocoevalcap
 * matplotlib, seaborn

## Implementation Details

### Object Detection

* **Model:** YOLOv5
* **Preprocessing:** Image resizing, blob conversion, RGB transformation
* **Parameters:**
    * Confidence threshold: 0.5
    * NMS threshold: 0.4

### Scene Graph Generation
    * Predict spatial and contextual relationships between objects.
    * Constructs directed graphs using NetworkX

### Caption Generation
* **Model:** Transformer-based GPT model
* **Parameters:** 
    * Prompt Length: 150 tokens
    * Max tokens: 100 tokens
    * Temperature: 0.7
    * Top_p: 1

## Usage                 
The complete experimental pipeline is demonstrated in: `model/captioning.ipynb/`

##  Results
The system generates:
    
1.  Object Detection Visualizations
2.	Scene Graph representations
3.	Context-aware natural language captions

Generated captions from the proposed method and baseline models are available in: `data/generated_captions_models/`

## Evaluation  
**Standard Marics**
Performance is evaluated using standard metrics:
 * BLEU
 * ROUGE
 * METEOR
 * CIDEr
 * SPICE
Evaluation scripts and inputs are available in: `evaluation/standard_matric.ipynb/`

**Human Expert Evaluation**
A human evaluation study was conducted using a five-point relevance scale via Google Forms: https://docs.google.com/forms/d/e/1FAIpQLSfkLlBj6UdrTU6Ixpb0UWGRRitybHly4YKqsxQ8nPMpE-chcA/viewform

Aggregated expert ratings are provided in: `data/expert rating.csv/` or `data/expert_rating_summary`

**Exploratory Data Analysis**
Exploratory analysis was performed, analyzed to assess how well three distinct models (M1, M2, and M3) performed, examining key aspects such as vocabulary richness, readability, sentiment alignment, similarity to ground-truth captions, and POS diversity. EDA notebooks are available at: `EDA/Exploratory_Data_Analysis.ipynb/`

## Data Availability
All minimal data required to reproduce the results reported in this study are publicly available in the `data/ directory of this repository`/.

