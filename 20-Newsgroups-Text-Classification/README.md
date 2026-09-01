# 20 Newsgroups Text Classification

A machine learning project that implements a Multinomial Naive Bayes classifier from scratch to categorize text documents from the 20 Newsgroups dataset.

## Project Overview

The model classifies 18,774 documents into 20 topic categories, including technology, science, sports, politics, and religion.

The classifier was implemented without machine learning libraries. NumPy is used for matrix operations and efficient prediction.

## Dataset

* 11,269 training documents
* 7,505 testing documents
* 61,188 vocabulary words
* 20 text categories

## Model

The project uses a Multinomial Naive Bayes classifier with:

* Class prior probabilities
* Word likelihood probabilities
* Laplace smoothing
* Log probabilities for numerical stability
* Confusion matrices and per-class evaluation

## Results

| Dataset  | Accuracy |
| -------- | -------: |
| Training |   94.11% |
| Testing  |   78.11% |

The best-performing test category was `rec.sport.hockey` with 95.49% accuracy.

## Technologies

* Python
* NumPy
* CSV data processing

## Project Structure

```text
20-Newsgroups-Text-Classification/
├── 20newsgroups/
│   └── 20newsgroups/
│       ├── map.csv
│       ├── test_data.csv
│       ├── test_label.csv
│       ├── train_data.csv
│       ├── train_label.csv
│       └── vocabulary.txt
├── lab_report.txt
├── naive_bayes_20newsgroups.py
└── README.md
```

## How to Run

Install NumPy:

```bash
pip install numpy
```

Run the classifier from the project directory:

```bash
python naive_bayes_20newsgroups.py
```

The program prints class probabilities, overall accuracy, accuracy for each category, and confusion matrices for the training and testing datasets.
