# Multimodal Image and Text Analysis Prototype

Final-phase proof of concept combining product images and text. The Phase III report proposes an IT hardware inventory and support-ticket use case, while the submitted notebook uses a fashion-product image/text dataset. This repository documents the implementation present in the code and does not claim the IT inventory integration was completed.

## Project highlights

- Generated image captions and visual question answers with BLIP.
- Classified text using TF-IDF features and Logistic Regression.
- Compared image-text compatibility with CLIP and calculated BLEU for generated text.
- The notebook reports 85% classifier accuracy on 55 test samples, a mean CLIP score of 26.32, and a BLEU score of 0.00. The small test set and metric outputs should be interpreted cautiously.

## Core competencies

Multimodal AI, image captioning, visual question answering, text classification, image-text similarity, model evaluation.

## Code

`src/multimodal_prototype.py` contains Python code cells exported from the Phase III notebook. Configure the dataset path and review model downloads before running.

## Setup

```bash
python -m pip install -r requirements.txt
```

Pretrained model weights and dataset files are downloaded separately; they are not included.

## Data and credentials

No dataset, model weights, API keys, access tokens, or personal data are included.
