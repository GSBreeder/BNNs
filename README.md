# Bayesian Neural Networks for Genomic Prediction
This is the code for manuscript "Bayesian Neural Networks for Genomic Prediction: Uncertainty Quantification and SNP Interpretation with SHAP and GWAS"

## Introduction
This repository contains the implementation for Bayesian Neural Networks (BNNs) with LASSO feature selection for genomic prediction of crop traits. The framework enables:
- Simultaneous processing of genotype (features) and phenotype (target traits) data
- Bayesian uncertainty quantification for predictions
- SHAP-based model interpretability
- Single or multi-trait prediction capabilities


## Data Preparation
We provide a dataset template `model_data.csv` in the repository. 

## Requirements
This code is based on pytorch.

- torch
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- shap
