# 🧠 Bayes to the Future: Predicting Heart Disease with Data!

## Overview
This project uses a Bayesian Network to predict heart disease from patient health records.

## Dataset
Used the `heart_disease.csv` dataset (from https://bit.ly/3T1A7Rs), cleaned and normalized.

## Tasks Completed
- Cleaned and normalized dataset
- Built Bayesian Network: age → fbs → target → chol, thalach
- Trained using Maximum Likelihood Estimation (MLE)
- Queried probabilities using `pgmpy`

## Setup
```bash
pip install pandas numpy scikit-learn pgmpy matplotlib networkx

python model_script.py

