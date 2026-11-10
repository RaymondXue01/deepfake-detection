# Detecting AI-Generated Images via Self-Supervised Representations and Transfer Learning

## Repository Structure

`/src/`: Source code
- `/src/preprocess.py`: Image preprocessing — JPEG compression, Gaussian blur, Gaussian noise
- `/src/extract_embeddings.py`: DINO Vision Transformer (ViT-S/16) feature extraction
- `/src/train_classifiers.py`: Train Logistic Regression, SVM, Random Forest classifiers and generate ROC curves
- `/src/train_cnn.py`: ResNet-18 fine-tuning for end-to-end classification
- `/src/unsupervised_analysis.py`: PCA, t-SNE, UMAP dimensionality reduction and k-Means clustering
- `/src/pca_analysis.py`: Standalone PCA with explained variance analysis
- `/src/gradcam.py`: GradCAM interpretability — spatial attention heatmaps
- `/src/full_analysis.py`: Complete analysis pipeline orchestrating all methods

`/scripts/`: Helper scripts
- `/scripts/download_dataset.py`: Downloads the CIFake dataset from Kaggle (requires API token)
