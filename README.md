# Bayesian Neural Networks for Genomic Prediction: Uncertainty Quantification and SNP Interpretation with SHAP and GWAS
This is the code for manuscript "Bayesian Neural Networks for Genomic Prediction: Uncertainty Quantification and SNP Interpretation with SHAP and GWAS"

## Technical Overview: Bayesian Genetic Prediction Framework

This implementation features a robust Bayesian Neural Networks (BNNs) pipeline for high-dimensional genetic analysis, emphasizing statistical rigor and model interpretability.

**1. Probabilistic Weight Modeling (BNNs)**
Unlike deterministic MLPs, this model treats weights as learnable probability distributions ($W \sim q(W|\theta)$). By optimizing the **ELBO (Evidence Lower Bound)**, the model balances predictive accuracy with complexity (KL divergence), effectively acting as an advanced regularizer for sparse genetic datasets.

**2. Intentional Stochasticity & Uncertainty**
Variations in accuracy across runs are an inherent feature of the Bayesian approach. Through **Monte Carlo (MC) Sampling** during inference, the model generates a predictive distribution rather than a single point estimate. This allows for the quantification of **Epistemic Uncertainty** (model confidence), providing a "safety metric" for clinical or agricultural decision-making.

**3. Post-hoc Interpretability**
The framework integrates **SHAP (SHapley Additive exPlanations)** to decode the "black box." By leveraging the trained posterior, it attributes influence to specific SNPs, bridging the gap between deep learning performance and biological discovery.

**4. Hyperparameter Adaptability & Fine-tuning** 
Recognizing that different biological traits (e.g., fiber quality vs. yield components) possess distinct genetic architectures and heritability levels, this framework supports granular parameter tuning:
    
    Sparsity Control via Priors: The mixture prior parameters ($\sigma_1, \sigma_2, \pi$) can be adjusted to reflect the expected genetic sparsity. For traits controlled by major genes, narrowing the spike prior ($\sigma_2$) and decreasing the mixture probability ($\pi$) enforces stricter feature selection.
    
    Architecture Scalability: The network depth and width (e.g., [32, 16, 8] vs. [128, 64]) are fully configurable, allowing the model to adapt from simple additive genetic effects to highly complex, non-linear epistatic interactions.
    
    KL-Divergence Weighting: To manage the trade-off between the data likelihood and the structural prior, the KL term can be scaled. This is particularly effective for small-sample datasets where over-regularization via the prior is necessary to prevent overfitting.
    
    Sampling Fidelity: The Monte Carlo (MC) sampling depth during inference is adjustable. Increasing the sample_num provides more stable estimates of the posterior predictive distribution, essential for traits with high environmental variance.

    LASSO Hyperparameter $\alpha$ (Alpha): By default, $\alpha$ is set to 0.001. This parameter controls the strength of the shrinkage. A larger $\alpha$ increases the penalty on the coefficients, effectively forcing more SNP effects to zero and resulting in a more parsimonious (sparse) model.


## How to Interpret Results
Accuracy Fluctuation: Since the model uses Monte Carlo Sampling, slight variations in Pearson Correlation across runs are expected. For stable results, increase the sample_num in the inference functions.

Uncertainty Plots: The generated PosteriorPlot visualizes the 95% confidence interval. A wider shaded area indicates the model is less certain about that specific sample.

SHAP Summary: The SummaryPlot shows the top SNPs influencing the trait. Red indicates high allele values, and their position on the X-axis indicates their positive or negative impact on the prediction.


## Requirements
This code is based on pytorch.

- torch
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- shap

## Output Structure
Successful execution generates:
- results
  - {trait}_model.pth            # Trained BNNs model
  - {trait}_PredResult.csv      # Predictions with uncertainty estimates
- figures
  - PosteriorPlot               # Prediction vs actual plots
  - LossPlot                    # Training/validation curves
  - PearsonPlot                 # Correlation metrics
  - ShapPlot                    # Feature importance
    - SummaryPlot                # Global feature impacts
    - dependencePlot             # Feature interactions
    - forcePlot                  # Individual sample explanations
