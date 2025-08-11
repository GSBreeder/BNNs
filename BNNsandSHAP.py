import time
import math
import os
import shap
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from scipy.stats import mode
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

plt.rc('font', family='serif')  # Times New Roman
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Prior:
    """Prior distribution"""
    def __init__(self, sigma1=1, sigma2=0.00001, pi=0.5):
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)
        self.pi = pi

    def log_prob(self, inputs):
        """Compute log probability sum (independent variables)"""
        prob1 = self.normal1.log_prob(inputs).exp()
        prob2 = self.normal2.log_prob(inputs).exp()
        return (self.pi * prob1 + (1 - self.pi) * prob2).log().sum()

class VariationalPoster:
    """Variational Posterior"""
    def __init__(self):
        self.normal = Normal(0, 1)
        self.sigma = None

    def sample(self, mu, rho):
        self.mu = mu
        self.sigma = rho.exp().log1p()
        epsilon = self.normal.sample(mu.shape).to(mu.device)
        return self.mu + self.sigma * epsilon  # 

    def log_prob(self, inputs):
        """Log probability density of normal distribution"""
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((inputs - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class BayesLinear(nn.Module):
    """Bayesian Fully Connected Layer"""
    def __init__(self, in_features, out_features, prior, deterministic=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        self.b_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.prior = prior
        self.W_variational_post = VariationalPoster()
        self.b_variational_post = VariationalPoster()

    def sample_weight(self, deterministic=False):
        """Sample weights from variational posterior"""
        if deterministic:
            return self.W_mu, self.b_mu
        W = self.W_variational_post.sample(self.W_mu, self.W_rho)
        b = self.b_variational_post.sample(self.b_mu, self.b_rho)
        return W, b

    def forward(self, inputs, train=True):
        W, b = self.sample_weight()
        outputs = F.linear(inputs, W.to(inputs.device), b.to(inputs.device))

        # Prediction mode
        if not train:
            return outputs, 0, 0

        # Training mode
        log_prior = self.prior.log_prob(W).sum() + self.prior.log_prob(b).sum()
        log_va_poster = self.W_variational_post.log_prob(W) + self.b_variational_post.log_prob(b)
        return outputs, log_prior, log_va_poster

class BayesMLP(nn.Module):
    """Bayesian MLP Model"""
    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none'):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior))
            self.layers.append(nn.Dropout(p=0.5))  # Added dropout layer
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior))

        # Activation function selection
        self.act_fn = {
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'none': F.tanh
        }.get(activate, F.tanh)
        self.flatten = nn.Flatten()

    def run_sample(self, inputs, train):
        """Run single sampling pass"""
        if len(inputs.shape) >= 3:  # Handle matrix inputs (e.g., images)
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0
        for layer in self.layers:
            if isinstance(layer, BayesLinear):
                model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
                log_prior += layer_log_prior
                log_va_poster += layer_log_va_poster
                inputs = self.act_fn(model_preds)
            elif isinstance(layer, nn.Dropout):
                inputs = layer(inputs)
        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num=1):
        """Forward pass with multiple samples (Equation 29)"""
        log_prior_s, log_va_poser_s = 0, 0
        model_preds_s = []

        for _ in range(sample_num):
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior
            log_va_poser_s += log_va_poster
            model_preds_s.append(model_preds)

        return model_preds_s, log_prior_s / sample_num, log_va_poser_s / sample_num

class RegressionELBOLoss(nn.Module):
    """ELBO Loss for Regression"""
    def __init__(self, batch_num, noise_tol=0.1):
        super().__init__()
        self.batch_num = batch_num
        self.noise_tol = noise_tol

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out
        log_like_s = 0
        for model_preds in model_preds_s:
            dist = Normal(model_preds, self.noise_tol)
            log_like_s += dist.log_prob(targets).sum()
        return 1/self.batch_num * (log_va_poster - log_prior) - log_like_s/len(model_preds_s)

def train_model(X, Y, scaler_y, epochs, lr, weight_decay=1e-4):
    model = BayesMLP(X.shape[2], 1, [16, 8], activate='relu').to(device)
    criterion = RegressionELBOLoss(batch_num=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    pearson_corrs_train = []
    Y = Y.unsqueeze(-1).to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X.to(device), 1)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Calculate Pearson correlation
        pred_train = outputs[0][0].detach().cpu().numpy().flatten()
        Y_train_inv = scaler_y.inverse_transform(Y.cpu().numpy()).flatten()
        pred_train_inv = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        pearson_corr_train = pearsonr(pred_train_inv, Y_train_inv)[0]
        pearson_corrs_train.append(pearson_corr_train)

    return model, train_losses, pearson_corrs_train, time.time() - start



def main():
    # Ensure directories exist
    os.makedirs('BNN_saved_models', exist_ok=True)
    os.makedirs('BNN_preds_result', exist_ok=True)
    os.makedirs('BNN_figure/PosteriorPlot', exist_ok=True)
    os.makedirs('BNN_figure/LossPlot', exist_ok=True)
    os.makedirs('BNN_figure/PearsonPlot', exist_ok=True)
    os.makedirs('BNN_figure/ShapPlot/SummaryPlot', exist_ok=True)
    os.makedirs('BNN_figure/ShapPlot/dependencePlot', exist_ok=True)
    os.makedirs('BNN_figure/ShapPlot/forcePlot', exist_ok=True)

    data = pd.read_csv("model_data.csv")
    X = data.iloc[:, 60:].values
    feature_names = data.columns[60:]
    targets = data.iloc[:, 4:19]
    
    for tar in targets.columns:
        print(f"\n{'=='*30}\nProcessing target: {tar}\n{'=='*30}")
        Y = targets[tar].values
        
        # Feature selection and preprocessing
        X_datas, Y_datas, scaler_y, selected_feature_names, scaler_x_selected = \
            load_preprocess_select_features_with_lasso(
                X, Y, tar, feature_names, 
                n_features_to_select=1000, alpha=0.01
            )
        
        # Cross-validation
        model_path = f'BNN_saved_models/{tar}_model.pth'
        X_train, X_test, Y_test, fold_results, avg_pearson = cross_validate(
            X_datas, Y_datas, scaler_y, model_path, 
            n_splits=10, epochs=300, lr=0.001
        )
        
        # Uncertainty estimation
        preds_path = f'BNN_preds_result/{tar}_PredResult.csv'
        post_plot = f'BNN_figure/PosteriorPlot/{tar}_PosteriorPlot.png'
        uncertainty_estimation(
            X_test, Y_test, scaler_y, model_path, 
            preds_path, post_plot, num_samples=100
        )
        
        # Plotting results
        loss_plot = f'BNN_figure/LossPlot/{tar}_LossPlot.png'
        pearson_plot = f'BNN_figure/PearsonPlot/{tar}_PearsonPlot.png'
        plot_results(fold_results, loss_plot)
        plot_pearson_correlations(fold_results, pearson_plot)
        
        # SHAP explanations
        summary_plot = f'BNN_figure/ShapPlot/SummaryPlot/{tar}_SummaryPlot.png'
        dep_plot_dir = f'BNN_figure/ShapPlot/dependencePlot/{tar}'
        force_plot = f'BNN_figure/ShapPlot/forcePlot/{tar}_forcePlot.html'
        ShapExplainer(
            X_train, X_test, model_path, summary_plot, 
            dep_plot_dir, force_plot, selected_feature_names, scaler_x_selected
        )

if __name__ == '__main__':
    main()