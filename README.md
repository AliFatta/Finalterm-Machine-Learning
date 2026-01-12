# Machine Learning Projects Portfolio

This repository contains three machine learning projects covering fraud detection, regression prediction, and image classification using various advanced techniques and algorithms.

## Table of Contents
- [Project 1: Fraud Detection](#1-fraud-detection)
- [Project 2: Year Prediction (Regression)](#2-year-prediction-regression)
- [Project 3: Fish Image Classification](#3-fish-image-classification)
- [Requirements](#requirements)
- [Installation](#installation)

---

## 1. Fraud Detection

**File:** `Finalterm_ML_1_transaction_data.ipynb`

### Overview
This project aims to detect fraudulent transactions from a large and imbalanced financial transaction dataset. The model identifies potentially fraudulent activities to help prevent financial losses.

### Model & Methodology
- **Algorithm:** LightGBM (Light Gradient Boosting Machine)
- **Technique:** Gradient Boosting Decision Tree (GBDT) with class imbalance handling using `scale_pos_weight` parameter
- **Experiments:** Two model configurations were tested:
  - **Model A (Standard):** `num_leaves=31`, `learning_rate=0.05`
  - **Model B (Aggressive):** `num_leaves=63`, `learning_rate=0.1`

### Results
The model achieved excellent performance in distinguishing between normal and fraudulent transactions:

| Metric | Score |
|--------|-------|
| ROC-AUC Score | 0.9651 |
| Accuracy | 96% |
| Recall (Fraud) | 83% |
| Precision (Fraud) | 44% |

**Key Insight:** The model successfully captures most fraud cases (83% recall) while maintaining high overall accuracy.

---

## 2. Year Prediction (Regression)

**File:** `Finalterm_ML_2_regresi.ipynb`

### Overview
This project uses regression techniques to predict a target value (Year) based on 90 numerical features. The model learns patterns from anonymous feature data to make temporal predictions.

### Model & Methodology
- **Algorithm:** CatBoostRegressor
- **Configuration:** 
  - GPU acceleration (`task_type="GPU"`)
  - Tree depth: 8
  - Iterations: 5000
- **Dataset:** Regression dataset with 90 anonymous features (Feat_1 to Feat_90)
- **Split:** 80% training, 20% testing

### Results
Evaluation metrics on test data:

| Metric | Score |
|--------|-------|
| R² Score | 0.37853 |
| RMSE | 8.6002 |
| MAE | 6.0091 years |

**Key Insight:** The model provides reasonable predictions with an average error of approximately 6 years.

---

## 3. Fish Image Classification

**File:** `Finalterm_Task3_FishImageCNN.ipynb`

### Overview
A computer vision project that classifies fish species from images using deep learning and transfer learning techniques.

### Model & Methodology
- **Architecture:** Transfer Learning with MobileNetV2
- **Base Model:** MobileNetV2 (pre-trained on ImageNet) with frozen layers
- **Custom Head:** 
  - GlobalAveragePooling2D
  - Dropout layer
  - Dense layer (3 output classes)
- **Preprocessing:** 
  - Data augmentation (Flip, Rotation, Zoom, Contrast)
  - Class imbalance handling using `class_weights`
- **Training:** 10-20 epochs

### Results
Initial validation results after 10 epochs:

| Metric | Score |
|--------|-------|
| Validation Accuracy | ~38% |
| Loss | ~1.11 |

**Note:** These results represent early training stages. Further training and hyperparameter tuning may improve performance.

---

## Requirements

```
python>=3.8
numpy
pandas
scikit-learn
lightgbm
catboost
tensorflow>=2.8
keras
matplotlib
seaborn
pillow
jupyter
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open the desired notebook and run the cells sequentially.

## Usage

Each notebook is self-contained and includes:
- Data loading and preprocessing
- Model training and configuration
- Evaluation metrics and visualization
- Results interpretation

Simply open the notebook and run all cells to reproduce the results.

## Project Structure

```
.
├── Finalterm_ML_1_transaction_data.ipynb    # Fraud Detection
├── Finalterm_ML_2_regresi.ipynb             # Year Prediction
├── Finalterm_Task3_FishImageCNN.ipynb       # Fish Classification
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## Future Improvements

### Fraud Detection
- Experiment with ensemble methods combining multiple models
- Implement SMOTE or other advanced sampling techniques
- Feature engineering to improve precision

### Year Prediction
- Hyperparameter tuning to improve R² score
- Feature selection to identify most important predictors
- Try other algorithms (XGBoost, Neural Networks)

### Fish Classification
- Increase training epochs
- Fine-tune MobileNetV2 layers
- Experiment with other architectures (ResNet, EfficientNet)
- Collect more training data to improve accuracy
