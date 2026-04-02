# BraTS Active Learning: Uncertainty-Guided Segmentation

This repository contains the code for a Deep Learning Computer Vision project focusing on medical image analysis. It implements an Uncertainty-Guided Active Learning pipeline for Brain Tumor Segmentation using the BraTS 2020 dataset.

The central goal is to address the high costs associated with medical image annotation by simulating a clinical workflow: training a foundational model on a small fraction of available data, and engineering it to explicitly quantify its own diagnostic doubt to flag the most ambiguous cases for expert review.

## Overview
The project simulates a real-world scenario where ground-truth labels are expensive to acquire. To support this, the dataset is partitioned at the patient level via a central indexing system (`data_splits.csv`) into three subsets:
*   **Initial Set (10%)**: Seed data used to train a foundational baby model.
*   **Unlabeled Pool (70%)**: The core simulation component. The model evaluates these unseen scans and prioritizes cases that generate the highest uncertainty.
*   **Validation Set (20%)**: A fixed holdout set used to accurately gauge generalization and improvements across the active learning loop.

## Research Summary
Key technical features include:
*   **2.5D Pseudo-3D Approach:** Stacking sequential MRI slices (z-1, z, z+1) into a 12-channel input to provide the model with 3D spatial context without the massive RAM overhead of full 3D convolutions.
*   **Dynamic ROI Extraction:** A "Smart Crop" algorithm that locates the geometric center of the brain tissue to safely crop volumes to `192x192`, preventing accidental amputation of off-center anatomy while significantly speeding up training.
*   **Agile U-Net & BALD Uncertainty:** A lightweight U-Net with Monte Carlo Dropout at the bottleneck. By keeping dropout active during inference and computing BALD (Bayesian Active Learning by Disagreement), the pipeline isolates epistemic uncertainty to score and rank unlabeled patients.
*   **Active Learning Loop:** Three iterations of scan → select → retrain, each adding the 10 most uncertain patients to the labeled set. Independent pools and a pre-fixed random baseline ensure a clean comparison between intelligent and random selection strategies.
*   **Robust Optimization:** Patient-specific Z-Score normalization and a custom multi-class Dice Loss module engineered to handle severe voxel class imbalances.

## Repository Structure
```
├── notebooks/          # full pipeline notebooks 
├── artifacts/          # saved model checkpoints
├── docs/               # experiment logs 
├── requirements.txt
└── README.md
```

## Technology Stack
Language: Python 3.11.9. Key Libraries:

*   **Deep Learning & Modeling**: PyTorch, Torch-DirectML, Albumentations 
*   **Medical Imaging & Data Processing**: Nibabel, NumPy, Pandas
*   **Visualization**: Matplotlib
*   **Utilities**: Scikit-learn, tqdm

*(See `requirements.txt` for full list)*

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.11.9
    *   Git
    *   *Note: The raw BraTS 2020 dataset (`.nii` format) must be downloaded independently (e.g., from Kaggle) and placed in the `data/` directory prior to running the extraction scripts.*

2.  **Clone Repository:**
    ```bash
    git clone https://github.com/yourusername/brats-active-learning.git
    ```

3.  **Set Up Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Running the Analysis:**
    *   The repository contains Jupyter notebooks detailing the all operations performed, see `notebooks/`