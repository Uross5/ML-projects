# Heart Disease Prediction - Machine Learning Project

## Project Overview

This project implements a complete data science lifecycle to predict heart disease in patients using machine learning algorithms. The project follows all required stages of the Data Science Life Cycle, from problem definition through model evaluation.

## Problem Statement

Heart disease is the leading cause of death globally. This project aims to develop a machine learning model that can predict the presence of heart disease in patients based on various clinical, demographic, and lifestyle factors. This is a **binary classification problem** where we predict whether a patient has heart disease (1) or does not have heart disease (0).

## Dataset

**Source:** Heart Disease Health Indicators Dataset (Kaggle)
**Link:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
**Dataset Size:** 253,680 records (229,781 after duplicate removal), 22 features
**Usage:** Full dataset used for comprehensive analysis

**Dataset Description:**

This dataset is derived from the 2015 Behavioral Risk Factor Surveillance System (BRFSS) survey conducted by the Centers for Disease Control and Prevention (CDC). The dataset contains health indicators including BMI, smoking status, alcohol consumption, physical and mental health metrics, and various medical conditions.

## Project Structure

```
.
├── heart_disease_prediction.ipynb  # Main Jupyter notebook with complete analysis
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. **Clone or download this repository**
2. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset:**

   - Visit: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
   - Click "Download" button
   - Extract the ZIP file
   - Save `heart_disease_health_indicators_BRFSS2015.csv` in the project directory
4. **Open the Jupyter notebook:**

   ```bash
   jupyter notebook heart_disease_prediction.ipynb
   ```

## Running the Project

1. Open `heart_disease_prediction.ipynb` in Jupyter Notebook
2. Run all cells sequentially (Cell → Run All)
3. The notebook will:
   - Load and prepare the data
   - Perform exploratory data analysis
   - Train multiple machine learning models
   - Evaluate and compare model performance
   - Generate visualizations and reports

## Methodology

The project follows the complete Data Science Life Cycle:

### 1. Problem Definition

- Clear problem statement
- Defined as binary classification problem
- Business understanding and importance

### 2. Data Collection

- Dataset source and description
- Feature documentation
- Limitations identified

### 3. Data Preparation

- Missing value handling
- Duplicate removal
- Outlier detection and treatment (winsorization)
- Feature encoding and scaling
- Train-test split (80-20)

### 4. Exploratory Data Analysis (EDA)

- Target variable distribution
- Correlation analysis
- Key insights and questions

### 5. Modeling

- **Algorithms Used:**
  - Logistic Regression (baseline)
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machine (SVM)
- Cross-validation (3-fold, optimized for large dataset)
- Hyperparameter tuning using GridSearchCV

### 6. Model Evaluation

- Training set performance
- Test set performance
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- ROC curves
- Feature importance analysis
- Model comparison

## Results

**Best Performing Model:** Gradient Boosting Classifier achieved the best ROC-AUC score (0.8316) on the test set.

**Model Performance Summary:**
- **Gradient Boosting**: ROC-AUC: 0.8316, Accuracy: 89.79%
- **Logistic Regression**: ROC-AUC: 0.8292, Accuracy: 89.79%
- **Random Forest**: ROC-AUC: 0.8277, Accuracy: 89.74%
- **SVM**: ROC-AUC: 0.3851, Accuracy: 21.57% (poor performance due to class imbalance)

The notebook displays:
- Cross-validation scores for all models
- Best hyperparameters for tuned models
- Performance metrics on both training and test sets
- Visualizations including:
  - Confusion matrices
  - ROC curves
  - Feature importance plots
  - Model comparison charts

## Key Features

- Complete Data Science Life Cycle implementation
- Multiple machine learning algorithms
- Comprehensive EDA with visualizations
- Hyperparameter tuning
- Model evaluation with multiple metrics
- Well-documented code with markdown explanations
- Professional visualizations

## Dependencies

See `requirements.txt` for complete list. Main packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Notes

- This model is for **educational purposes only**
- Should not be used for actual medical diagnosis without proper clinical validation
- Results may vary based on dataset version and random seed

## Future Work

- Advanced feature engineering
- Deep learning models
- Model deployment as web application
- SHAP values for interpretability
- External validation

## Author

 Uros Cukic 


