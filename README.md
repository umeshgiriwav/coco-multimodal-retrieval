# Bidirectional Image-Text Retrieval with Multimodal Learning

## Objective

The project evaluates symmetric InfoNCE, hard negative triplet loss, and an early fusion two-tower MLP on their ability to align frozen image and text representations in a shared 256-dimensional space. Target performance across the 500-image test set is at least 10% Recall@1 and 25% Recall@5 for both retrieval directions.

## Dataset

The corpus contains 5,000 images from the MS-COCO 2017 validation split on Hugging Face. Removing whitespace and four empty placeholder strings left 25,010 valid captions. A fixed split partitions the images into 4,000 for training, 500 for validation, and 500 for testing, alongside a 500-image training probe for in sample comparisons.

## Methodology

All models rely on frozen feature extractors: 2,048-dimensional vectors from ResNet-50 and 768-dimensional vectors from BERT. Two dual-encoder models project both modalities into an L2-normalized 256-dimensional space and train with either symmetric InfoNCE or margin based triplet ranking. A two tower MLP concatenates the two 256-dimensional projections into a 512-dimensional vector and scores candidate pairs through a two layer network trained against circularly shifted negatives.

## Installation and execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the feature cache:**
   Place `coco_splits.json`, `image_features.pt`, and `text_features.pt` inside `multimodal_project/shared_data/`. If the `.pt` files are not present, running Cell 2 of `Multimodal_Project.ipynb` will download `lmms-lab/COCO-Caption2017` from Hugging Face and extract the ResNet-50 and BERT tensors automatically.

3. **Run the notebook:**
   Open `Multimodal_Project.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code and run all cells from top to bottom.