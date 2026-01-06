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
from scipy.stats import norm
import matplotlib.pyplot as plt

plt.rc('font', family='serif')  # Times New Roman
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Prior:
    """Prior Distribution (Equation 38)"""

    def __init__(self, sigma1=1, sigma2=0.00001, pi=0.5):
        """
        Args:
            sigma1: sigma1 in Eq. 38
            sigma2: sigma2 in Eq. 38
            pi:     pi in Eq. 38
        """
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)
        self.pi = pi

    def log_prob(self, inputs):
        """
        Calculates the log probability sum. Since values are independent, 
        the joint distribution is the product of individual distributions, 
        which becomes a sum in log space.
        """
        prob1 = self.normal1.log_prob(inputs).exp()  # Probability density
        prob2 = self.normal2.log_prob(inputs).exp()  # Probability density
        return (self.pi * prob1 + (1 - self.pi) * prob2).log().sum()  # Based on Eq. 38


class VariationalPoster:
    """Variational Posterior Distribution"""

    def __init__(self):
        self.normal = Normal(0, 1)
        self.sigma = None

    def sample(self, mu, rho):
        self.mu = mu
        self.sigma = rho.exp().log1p()
        epsilon = self.normal.sample(mu.shape).to(mu.device)  # Algorithm 2: Line 5
        return self.mu + self.sigma * epsilon  # Eq. 33 | Algorithm 2: Line 6

    def log_prob(self, inputs):
        """
        Log probability density of normal distribution:
        log(N(x|mu, sigma)) = -log(sqrt(2*pi)) - log(sigma) - (x-mu)^2/(2*sigma^2)
        """
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((inputs - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class BayesLinear(nn.Module):
    """
    Bayesian Fully Connected Layer
    """

    def __init__(self, in_features, out_features, prior, deterministic=False):
        """
        Args:
            in_features: Input dimensions
            out_features: Output dimensions
            prior: Prior distribution
        """
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
        """Samples weights and biases from the variational posterior"""
        if deterministic:
            # Use mean directly without sampling
            W = self.W_mu
            b = self.b_mu
        else:
            # Sample from the distribution
            W = self.W_variational_post.sample(self.W_mu, self.W_rho)  # Algorithm 2: Line 6
            b = self.b_variational_post.sample(self.b_mu, self.b_rho)  # Algorithm 2: Line 6
        return W, b

    def forward(self, inputs, train=True):
        W, b = self.sample_weight()  # Sample weights and biases
        outputs = F.linear(inputs, W.to(inputs.device), b.to(inputs.device))  # Wx + b

        # -- Inference
        if not train:
            return outputs, 0, 0

        # -- Training
        # Log Prior
        log_prior = self.prior.log_prob(W).sum() + self.prior.log_prob(b).sum()  # Algorithm 2: Line 7
        # Log Variational Posterior
        log_va_poster = self.W_variational_post.log_prob(W) + self.b_variational_post.log_prob(b)  # Algorithm 2: Line 7
        return outputs, log_prior, log_va_poster


class BayesMLP(nn.Module):
    """
    Bayesian Multi-Layer Perceptron (MLP) Model
    """

    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none'):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior))
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior))

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid
        self.flatten = nn.Flatten()

    def run_sample(self, inputs, train):
        """Performs a single sample run, returns prediction, log prior, and log variational posterior"""
        if len(inputs.shape) >= 3:  # Handle cases where inputs are matrices (e.g., images)
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0
        for layer in self.layers:
            if isinstance(layer, BayesLinear):
                model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
                log_prior += layer_log_prior
                log_va_poster += layer_log_va_poster
                inputs = self.act_fn(model_preds)

        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num=1):
        """
        Args:
            inputs: Model input
            sample_num: Number of MC samples (m in Eq. 29)
        """
        log_prior_s = 0
        log_va_poser_s = 0
        model_preds_s = []

        for _ in range(sample_num):  # Algorithm 2: Line 4
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior
            log_va_poser_s += log_va_poster
            model_preds_s.append(model_preds)

        return model_preds_s, log_prior_s / sample_num, log_va_poser_s / sample_num


class BayesMLP_copy(nn.Module):
    """
    Copy version of Bayesian MLP for deterministic inference (SHAP)
    """

    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none',
                 deterministic=True):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior, deterministic))
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior, deterministic))

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid
        self.flatten = nn.Flatten()

    def run_sample(self, inputs, train):
        if len(inputs.shape) >= 3:
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0
        for layer in self.layers:
            if isinstance(layer, BayesLinear):
                model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
                log_prior += layer_log_prior
                log_va_poster += layer_log_va_poster
                inputs = self.act_fn(model_preds)

        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num=1):
        log_prior_s = 0
        log_va_poser_s = 0
        model_preds_s = []

        for _ in range(sample_num):
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior
            log_va_poser_s += log_va_poster
            model_preds_s.append(model_preds)

        model_preds_s = torch.stack(model_preds_s)  # Convert list to tensor

        if not self.training:
            for layer in self.layers:
                if isinstance(layer, nn.Dropout):
                    layer.eval()  # Disable Dropout during evaluation

        return model_preds_s.mean(dim=0)


class RegressionELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) Loss for regression problems
    """

    def __init__(self, batch_num, noise_tol=0.1):
        super().__init__()
        self.batch_num = batch_num
        self.noise_tol = noise_tol

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out
        log_like_s = 0
        for model_preds in model_preds_s:  # Algorithm 2: Line 7, Part 3
            # Regression assumes model outputs follow Gaussian distribution centered at prediction
            dist = Normal(model_preds, self.noise_tol)
            log_like_s += dist.log_prob(targets).sum()
        # Algorithm 2: Line 8
        return 1 / self.batch_num * (log_va_poster - log_prior) - log_like_s / len(model_preds_s)


def train_model(X, Y, scaler_y, epochs, lr, weight_decay=1e-4):
    model = BayesMLP(X.shape[2], 1, [32, 16, 8], activate='relu').to(device)
    criterion = RegressionELBOLoss(batch_num=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    pearson_corrs_train = []
    Y = Y.unsqueeze(-1).to(device)
    start = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X.to(device), 1)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Calculate and record Pearson correlation for current epoch
        pred_train = outputs[0][0].detach().cpu().numpy().reshape(-1, 1).flatten()
        Y_train_inv = scaler_y.inverse_transform(Y.cpu().numpy().reshape(-1, 1)).flatten()
        pred_train_inv = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        pearson_corr_train = pearsonr(pred_train_inv, Y_train_inv)[0]
        pearson_corrs_train.append(pearson_corr_train)

    training_time = time.time() - start
    return model, train_losses, pearson_corrs_train, training_time


def uncertainty_estimation(best_X_test, best_Y_test, scaler_y, model_saved_path, preds_saved_path,
                           PosteriorPlot_saved_path, num_samples=200):
    print('Loading model for uncertainty estimation...')
    best_model = BayesMLP(best_X_test.shape[2], 1, [32, 16, 8], activate='relu').to(device)
    saved_model_params = torch.load(model_saved_path)
    best_model.load_state_dict(saved_model_params)

    best_model.eval()  # Ensure model is in evaluation mode

    all_predictions = []
    print('Start predicting for uncertainty analysis...')
    with torch.no_grad():
        for _ in range(num_samples):
            output = best_model(best_X_test.to(device), 1)
            predictions = output[0][0].cpu().numpy()
            predictions = scaler_y.inverse_transform(predictions)
            all_predictions.append(predictions)

    all_predictions = np.array(all_predictions).squeeze(2)

    results_list = []
    idx_list = []
    mean_predictions = []
    true_values = []
    ci_lowers = []
    ci_uppers = []

    # Calculate statistics for each sample
    best_Y_test_inv = scaler_y.inverse_transform(best_Y_test.numpy().reshape(-1, 1)).flatten()
    for idx in range(best_X_test.shape[0]):
        sample_preds = all_predictions[:, idx]
        mean_pred = np.mean(sample_preds)
        median_pred = np.median(sample_preds)
        mode_pred = mode(sample_preds, keepdims=True).mode[0]
        variance_pred = np.var(sample_preds)
        std_dev_pred = np.std(sample_preds)
        
        # 95% Confidence Interval using Std Dev
        ci_lower1 = mean_pred - 1.96 * std_dev_pred
        ci_upper1 = mean_pred + 1.96 * std_dev_pred
        # 95% Confidence Interval using Percentiles
        ci_lower2 = np.percentile(sample_preds, 2.5)
        ci_upper2 = np.percentile(sample_preds, 97.5)
        
        true_value = best_Y_test_inv[idx]

        idx_list.append(idx)
        mean_predictions.append(mean_pred)
        true_values.append(true_value)
        ci_lowers.append(ci_lower2)
        ci_uppers.append(ci_upper2)

        results_dict = {
            'Sample Index': idx,
            'True Value': true_value,
            'Mean Prediction': mean_pred,
            'Median Prediction': median_pred,
            'Mode Prediction': mode_pred,
            'Variance': variance_pred,
            'Standard Deviation': std_dev_pred,
            '95% CI Lower (Std)': ci_lower1,
            '95% CI Upper (Std)': ci_upper1,
            '2.5% Percentile': ci_lower2,
            '97.5% Percentile': ci_upper2
        }
        results_list.append(results_dict)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(preds_saved_path, mode='w', header=True, index=False)

    # Plot predictions vs ground truth with confidence interval
    plt.figure(figsize=(12, 9))
    plt.plot(idx_list, mean_predictions, color='#9dc3e7', marker='.', linestyle='-', linewidth=2, markersize=2, label='Mean Predictions')
    plt.plot(idx_list, true_values, color='#f18180', marker='.', linestyle='-', linewidth=2, markersize=2, label='True Values')
    plt.fill_between(idx_list, ci_lowers, ci_uppers, color='gray', alpha=0.12, label='95% Confidence Interval')
    plt.xlabel('Sample Index', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Posterior Predictive Uncertainty', fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)
    
    if os.path.exists(PosteriorPlot_saved_path):
        os.remove(PosteriorPlot_saved_path)
    plt.savefig(PosteriorPlot_saved_path, bbox_inches='tight')
    plt.close()

    print('Uncertainty estimation finished!')
    return results_df


def cross_validate(X_raw, Y_raw, feature_names, model_saved_path, n_splits, epochs, lr):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Global normalization for Y
    scaler_y = MinMaxScaler()
    Y_sc = scaler_y.fit_transform(Y_raw.reshape(-1, 1)).flatten()
    Y_datas = torch.tensor(Y_sc, dtype=torch.float32)
    
    criterion = RegressionELBOLoss(batch_num=1)
    fold_results = []
    total_pearson_test = 0 
    max_pearson_test = float('-inf')
    
    best_model_params = None
    best_X_train = None
    best_X_test = None
    best_Y_test = None
    best_feature_names = None   
    best_scaler_x = None        

    for fold, (train_index, test_index) in enumerate(kf.split(X_raw), 1):
        fold_start_time = time.time()

        # 1. Split data
        X_tr_raw, X_te_raw = X_raw[train_index], X_raw[test_index]
        Y_train, Y_test = Y_datas[train_index], Y_datas[test_index]

        # 2. Internal normalization and Lasso feature selection 
        scaler_x_pre = MinMaxScaler()
        X_tr_sc = scaler_x_pre.fit_transform(X_tr_raw)
        
        lasso = Lasso(alpha=0.001, random_state=42)
        selector = SelectFromModel(lasso, max_features=3000)
        selector.fit(X_tr_sc, Y_train.numpy())
        
        select_idx = selector.get_support(indices=True)
        cur_feat_names = feature_names[select_idx] 
        
        # 3. Build Scaler specifically for selected features
        scaler_x_selected = MinMaxScaler()
        X_tr_final_np = scaler_x_selected.fit_transform(X_tr_raw[:, select_idx])
        X_te_final_np = scaler_x_selected.transform(X_te_raw[:, select_idx])
        
        X_train_tensor = torch.tensor(X_tr_final_np, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_te_final_np, dtype=torch.float32).unsqueeze(1)

        # 4. Train Model
        model, train_losses, pearson_corrs_train, _ = train_model(X_train_tensor, Y_train, scaler_y, epochs=epochs, lr=lr)

        # 5. Prediction and correlation calculation
        model.eval()
        with torch.no_grad():
            pred_train_out = model(X_train_tensor.to(device), sample_num=100)
            pred_test_out = model(X_test_tensor.to(device), sample_num=100)
            test_loss = criterion(pred_test_out, Y_test.to(device)).item()

        # Average Monte Carlo sampling results
        p_tr = pd.DataFrame(np.array([s.detach().cpu().numpy() for s in pred_train_out[0]]).squeeze(-1).T).apply(lambda r: r.mean(), axis=1).values
        p_te = pd.DataFrame(np.array([s.detach().cpu().numpy() for s in pred_test_out[0]]).squeeze(-1).T).apply(lambda r: r.mean(), axis=1).values
        
        # Inverse transform to calculate real Pearson correlation
        p_tr_inv = scaler_y.inverse_transform(p_tr.reshape(-1, 1)).flatten()
        p_te_inv = scaler_y.inverse_transform(p_te.reshape(-1, 1)).flatten()
        y_tr_inv = scaler_y.inverse_transform(Y_train.numpy().reshape(-1, 1)).flatten()
        y_te_inv = scaler_y.inverse_transform(Y_test.numpy().reshape(-1, 1)).flatten()

        pearson_train = pearsonr(p_tr_inv, y_tr_inv)[0]
        pearson_test = pearsonr(p_te_inv, y_te_inv)[0]
        total_pearson_test += pearson_test 

        fold_time = time.time() - fold_start_time
        print(f"Fold {fold} - Corr - Train: {pearson_train:.4f}, Test: {pearson_test:.4f}, Features: {len(select_idx)}, Time: {fold_time:.4f}s")

        if pearson_test > max_pearson_test:
            max_pearson_test = pearson_test
            best_model_params = model.state_dict()
            best_X_train, best_X_test, best_Y_test = X_train_tensor, X_test_tensor, Y_test
            best_feature_names, best_scaler_x = cur_feat_names, scaler_x_selected

        fold_results.append({
            "fold": fold,
            "train_losses": train_losses,
            "test_loss": test_loss,
            "pearson_train": pearson_corrs_train,
            "pearson_test": pearson_test
        })

    avg_pearson_test = total_pearson_test / n_splits
    print(f"\nAverage Pearson Correlation Coefficient on Test Set: {avg_pearson_test:.4f}")
    torch.save(best_model_params, model_saved_path)

    return best_X_train, best_X_test, best_Y_test, fold_results, avg_pearson_test, scaler_y, best_feature_names, best_scaler_x


def plot_results(fold_results, LossPlot_saved_path):
    plt.figure(figsize=(12, 9))
    train_losses = [fold_result['train_losses'][-1] for fold_result in fold_results]
    test_losses = [fold_result['test_loss'] for fold_result in fold_results]
    folds = range(1, len(fold_results) + 1)
    plt.plot(folds, train_losses, color='#faccb0', marker='.', linestyle='-', linewidth=3, markersize=2, label='train_loss')
    plt.plot(folds, test_losses, color='#f5b0b0', marker='.', linestyle='-', linewidth=3, markersize=2, label='test_loss')
    plt.title('Loss per Fold', fontsize=16)
    plt.xlabel('Fold', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=True, framealpha=0.6)
    if os.path.exists(LossPlot_saved_path):
        os.remove(LossPlot_saved_path)
    plt.savefig(LossPlot_saved_path, bbox_inches='tight')
    plt.close()


def plot_pearson_correlations(fold_results, PearsonPlot_saved_path):
    plt.figure(figsize=(12, 9))
    pearson_train = [fold_result['pearson_train'][-1] for fold_result in fold_results]
    pearson_test = [fold_result['pearson_test'] for fold_result in fold_results]
    folds = range(1, len(fold_results) + 1)
    plt.plot(folds, pearson_train, color='#92A5D1', marker='.', linestyle='-', linewidth=4, markersize=2, label='pearson_train')
    plt.plot(folds, pearson_test, color='#D9B9D4', marker='.', linestyle='-', linewidth=4, markersize=2, label='pearson_test')
    plt.title('Pearson Correlation per Fold', fontsize=16)
    plt.xlabel('Fold', fontsize=16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)
    if os.path.exists(PearsonPlot_saved_path):
        os.remove(PearsonPlot_saved_path)
    plt.savefig(PearsonPlot_saved_path, bbox_inches='tight')
    plt.close()


def ShapExplainer(best_X_train, best_X_test, model_saved_path, shap_values_file_path, SummaryPlot_saved_path, dependencePlot_saved_path,
                  forcePlot_saved_path, selected_feature_names, scaler_x_selected):
    print('Start SHAP analysis... Loading deterministic model copy...')
    best_model = BayesMLP_copy(best_X_test.shape[2], 1, [32, 16, 8], activate='relu', deterministic=True).to(device)
    saved_model_params = torch.load(model_saved_path)
    best_model.load_state_dict(saved_model_params)

    # Move tensors and handle shapes
    best_X_train_np = best_X_train.cpu().numpy().reshape(best_X_train.shape[0], -1)
    best_X_test_np = best_X_test.cpu().numpy().reshape(best_X_test.shape[0], -1)
    
    # Inverse transform to original scale for interpretation
    best_X_train_orig = scaler_x_selected.inverse_transform(best_X_train_np)
    best_X_test_orig = scaler_x_selected.inverse_transform(best_X_test_np)
    
    best_X_train_tensor = torch.tensor(best_X_train_orig, dtype=torch.float32).to(device)
    best_X_test_tensor = torch.tensor(best_X_test_orig, dtype=torch.float32).to(device)

    best_model.eval()
    
    # Create GradientExplainer
    explainer = shap.GradientExplainer(best_model, best_X_train_tensor)
    shap_values = explainer.shap_values(best_X_test_tensor)
    shap_values = np.squeeze(shap_values)
    
    print(f"Reshaped SHAP values shape: {np.shape(shap_values)}")
    best_X_test_reshaped = best_X_test_orig
    feature_label = selected_feature_names

    # Save Mean SHAP values to CSV
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    mean_shap_values = shap_values.mean(axis=0)
    shap_df = pd.DataFrame({'Feature': feature_label, 'Mean_SHAP_Value': mean_shap_values})
    shap_df.to_csv(shap_values_file_path, index=False)
    print(f"SHAP values saved to {shap_values_file_path}")
    
    # Get top 4 features by importance
    top_4_indices = np.argsort(mean_abs_shap_values)[-4:][::-1]
    top_4_feature_names = [feature_label[i] for i in top_4_indices]

    # Plot Summary Plot
    shap.summary_plot(shap_values, best_X_test_reshaped, show=False, feature_names=feature_label, cmap='coolwarm', max_display=20)
    if os.path.exists(SummaryPlot_saved_path):
        os.remove(SummaryPlot_saved_path)
    plt.savefig(SummaryPlot_saved_path, bbox_inches='tight')
    plt.close()

    # Plot Dependence Plots for top interactions
    print("Generating Dependence Plots for Top Features...")
    for idx, feature_name in enumerate(top_4_feature_names):
        for jdx in range(idx + 1, 4):
            interaction_feature = top_4_feature_names[jdx]
            print(f"Interaction: {feature_name} <-> {interaction_feature}")
            suffix = f'_dependencePlot_{idx}_{jdx}.png'
            new_path = dependencePlot_saved_path + suffix
            
            shap.dependence_plot(feature_name, shap_values, best_X_test_reshaped,
                                 show=False, interaction_index=interaction_feature, feature_names=feature_label)
            if os.path.exists(new_path):
                os.remove(new_path)
            plt.savefig(new_path, bbox_inches='tight', pad_inches=0.3)
            plt.close()

    # Generate Force Plot for the first test sample
    sample_index = 0
    shap_single = shap_values[sample_index]
    feat_single = best_X_test_orig[sample_index]
    
    with torch.no_grad():
        base_value = best_model(best_X_train_tensor).cpu().numpy().mean()
        
    force_plot = shap.force_plot(base_value=base_value, shap_values=shap_single, features=feat_single, 
                                 feature_names=feature_label, show=False)
    shap.save_html(forcePlot_saved_path, force_plot)


def main():
    data = pd.read_csv("cotton_data2.csv")
    X_raw = data.iloc[:, 21:].values  # Original features
    feature_names = data.columns[21:] # Original feature names
    targets = data.iloc[:, 5:13]

    for tar in targets.columns:
        Y_raw = targets[tar].values
        print('==' * 30)
        print(f"Processing Target: {tar}")

        model_saved_path = f'BNN_saved_models/{tar}_model.pth'
        
        # Run Cross Validation
        (best_X_train, best_X_test, best_Y_test, fold_results, 
         avg_pearson_test, scaler_y, best_feat_names, best_scaler_x) = cross_validate(
            X_raw, Y_raw, feature_names,
            model_saved_path, n_splits=10,
            epochs=3000, lr=0.001
        )

        # Uncertainty Estimation
        preds_saved_path = f'BNN_preds_result/{tar}_PredResult.csv'
        PosteriorPlot_saved_path = f'BNN_figure/PosteriorPlot/{tar}_PosteriorPlot.png'
        uncertainty_estimation(best_X_test, best_Y_test, scaler_y, model_saved_path, preds_saved_path, PosteriorPlot_saved_path, num_samples=100)
        
        # Visualizing results
        LossPlot_saved_path = f'BNN_figure/LossPlot/{tar}_LossPlot.png'
        PearsonPlot_saved_path = f'BNN_figure/PearsonPlot/{tar}_PearsonPlot.png'
        plot_results(fold_results, LossPlot_saved_path)
        plot_pearson_correlations(fold_results, PearsonPlot_saved_path)

        # SHAP Interpretation
        shap_values_file_path = f'BNN_SHAP_num/{tar}_SNP_SHAP_values.csv'
        SummaryPlot_saved_path = f'BNN_figure/ShapPlot/SummaryPlot/{tar}_SummaryPlot.png'
        dependencePlot_saved_path = f'BNN_figure/ShapPlot/dependencePlot/{tar}'
        forcePlot_saved_path = f'BNN_figure/ShapPlot/forcePlot/{tar}_forcePlot.html'
        
        ShapExplainer(best_X_train, best_X_test, model_saved_path, shap_values_file_path, 
                      SummaryPlot_saved_path, dependencePlot_saved_path,
                      forcePlot_saved_path, best_feat_names, best_scaler_x)
        
        print(f'Finished all analysis for {tar}')
        print('==' * 30)


if __name__ == '__main__':
    main()
