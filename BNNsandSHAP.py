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
# Ignore warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Prior:
    """Prior distribution"""

    def __init__(self, sigma1=1, sigma2=0.00001, pi=0.5):
        """
        Args:
            sigma1
            sigma2 
            pi     
        """
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)
        self.pi = pi

    def log_prob(self, inputs):
        """Compute the logarithmic probabilities and sum them."""
        prob1 = self.normal1.log_prob(inputs).exp()  
        prob2 = self.normal2.log_prob(inputs).exp()  
        return (self.pi * prob1 + (1 - self.pi) * prob2).log().sum()  


class VariationalPoster:
    """Variational posterior"""

    def __init__(self):
        self.normal = Normal(0, 1)
        self.sigma = None

    def sample(self, mu, rho):
        self.mu = mu
        self.sigma = rho.exp().log1p()
        epsilon = self.normal.sample(mu.shape).to(mu.device)  
        return self.mu + self.sigma * epsilon          

    def log_prob(self, inputs):
        """
        Log probability density of the normal distribution
        log(N(x|mu, sigma)) = -log(sqrt(2*pi)) - log(sigma) - (x-mu)^2/(2*sigma^2)
        """
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((inputs - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class BayesLinear(nn.Module):
    """
    Bayesian dense layer
    """

    def __init__(self, in_features, out_features, prior, deterministic=False):
        """
        Args:
            in_features:  Input dimension
            out_features: Output dimension
            prior:        Prior distribution
            mu:           mu
            rho:          rho
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
        """Sample the weight matrix and bias vector of the fully connected layer from the variational posterior"""
        if deterministic:
            # Return the mean directly without sampling
            W = self.W_mu
            b = self.b_mu
        else:
            # Sample from the distribution
            W = self.W_variational_post.sample(self.W_mu, self.W_rho)  
            b = self.b_variational_post.sample(self.b_mu, self.b_rho)  
        return W, b

    def forward(self, inputs, train=True):
        W, b = self.sample_weight()  # Sample the weight matrix and bias vector
        outputs = F.linear(inputs, W.to(inputs.device), b.to(inputs.device))  # Wx + b

        # --Predict
        if not train:
            return outputs, 0, 0

        # --Train
        # Log prior
        log_prior = self.prior.log_prob(W).sum() + self.prior.log_prob(b).sum()  
        # Log variational posterior
        log_va_poster = self.W_variational_post.log_prob(W) + self.b_variational_post.log_prob(b)  
        return outputs, log_prior, log_va_poster


class BayesMLP(nn.Module):
    """
    Bayesian MLP model
    """

    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none'):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior))
            self.layers.append(nn.Dropout(p=0.5))  
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior))

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid
        self.flatten = nn.Flatten()

    def run_sample(self, inputs, train):
        """Perform one sampling iteration and return the model prediction, log prior, and log variational posterior."""
        if len(inputs.shape) >= 3:  # The case where the sample is a matrix rather than a vector (e.g., images).
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0  # Log prior, log variational posterior.
        for layer in self.layers:
            if isinstance(layer, BayesLinear):
                model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
                log_prior += layer_log_prior
                log_va_poster += layer_log_va_poster
                inputs = self.act_fn(model_preds)
            elif isinstance(layer, nn.Dropout):
                inputs = layer(inputs)  # Correctly call the Dropout layer without passing additional parameters.

        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num=1):
        """
        Args:
            inputs: Model input
            sample: Number of samples
        """
        log_prior_s = 0
        log_va_poser_s = 0
        model_preds_s = []

        for _ in range(sample_num):  
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior  
            log_va_poser_s += log_va_poster  
            model_preds_s.append(model_preds)  

        return model_preds_s, log_prior_s / sample_num, log_va_poser_s / sample_num


class BayesMLP_copy(nn.Module):
    """
    Bayesian MLP model
    """

    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none',
                 deterministic=True):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior, deterministic))
            self.layers.append(nn.Dropout(p=0.5)) 
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
            elif isinstance(layer, nn.Dropout):
                inputs = layer(inputs)  

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

       
        model_preds_s = torch.stack(model_preds_s)  # Convert the list to a tensor and return the mean or a single tensor

        # If in evaluation mode (eval), disable dropout and randomness
        if not self.training:
            for layer in self.layers:
                if isinstance(layer, nn.Dropout):
                    layer.eval()  
        return model_preds_s.mean(dim=0)


class RegressionELBOLoss(nn.Module):
    """
    ELBO loss for regression problems
    """

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
        return 1 / self.batch_num * (log_va_poster - log_prior) - log_like_s / len(model_preds_s) 


def train_model(X, Y, scaler_y, epochs, lr, weight_decay=1e-4):
#    model = BayesMLP(X.shape[2], 1, [16, 16], activate='relu').to(device)
    model = BayesMLP(X.shape[2], 1, [16, 8], activate='relu').to(device)
    #     criterion = nn.MSELoss()
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

        # Calculate and record the correlation coefficient for the current epoch.
        pred_train = outputs[0][0].detach().cpu().numpy().reshape(-1, 1).flatten()
        Y_train_inv = scaler_y.inverse_transform(Y.cpu().numpy().reshape(-1, 1)).flatten()
        pred_train_inv = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        pearson_corr_train = pearsonr(pred_train_inv, Y_train_inv)[0]
        pearson_corrs_train.append(pearson_corr_train)

    Time = time.time() - start
    return model, train_losses, pearson_corrs_train, Time


# 10-fold cross-validation
def cross_validate(X_datas, Y_datas, scaler_y, model_saved_path, n_splits, epochs, lr):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    criterion = RegressionELBOLoss(batch_num=1)
    fold_results = []
    total_pearson_test = 0  
    max_pearson_test = float('-inf')
    best_model_params = None
    best_X_test = None
    best_Y_test = None

    for fold, (train_index, test_index) in enumerate(kf.split(X_datas), 1):
        X_train, X_test = X_datas[train_index], X_datas[test_index]
        Y_train, Y_test = Y_datas[train_index], Y_datas[test_index]

        # Train the model and obtain the training loss and correlation coefficient.
        model, train_losses, pearson_corrs_train, Time = train_model(X_train, Y_train, scaler_y, epochs=epochs, lr=lr)

        # Predict and calculate the test set loss and correlation coefficient
        model.eval()
        with torch.no_grad():
            pred_train = model(X_train.to(device), sample_num=100)
            pred_test = model(X_test.to(device), sample_num=100)
            test_loss = criterion(pred_test, Y_test.to(device)).item()

        pred_train = pd.DataFrame(np.array([s.detach().cpu().numpy() for s in pred_train[0]]).squeeze(-1).T)
        pred_train = np.array(pred_train.apply(lambda row: row.mean(), axis=1))
        pred_test = pd.DataFrame(np.array([s.detach().cpu().numpy() for s in pred_test[0]]).squeeze(-1).T)
        pred_test = np.array(pred_test.apply(lambda row: row.mean(), axis=1))
        # Denormalization
        pred_train_inv = scaler_y.inverse_transform(pred_train.reshape(-1, 1)).flatten()
        pred_test_inv = scaler_y.inverse_transform(pred_test.reshape(-1, 1)).flatten()
        Y_train_inv = scaler_y.inverse_transform(Y_train.numpy().reshape(-1, 1)).flatten()
        Y_test_inv = scaler_y.inverse_transform(Y_test.numpy().reshape(-1, 1)).flatten()


        # Calculate Pearson correlation coefficient
        pearson_train = pearsonr(pred_train_inv, Y_train_inv)[0]
        pearson_test = pearsonr(pred_test_inv, Y_test_inv)[0]
        total_pearson_test += pearson_test  

        if pearson_test > max_pearson_test:
            max_pearson_test = pearson_test
            best_model_params = model.state_dict()
            best_X_train = X_train
            best_X_test = X_test
            best_Y_test = Y_test

        print(f"Fold {fold} - Corr - Train: {pearson_train:.4f}, Test: {pearson_test:.4f}, Time: {Time:.4f}s")

        fold_results.append({
            "fold": fold,
            "train_losses": train_losses,
            "test_loss": test_loss,
            "pearson_train": pearson_corrs_train,
            "pearson_test": pearson_test
        })

    # Calculate and print the average correlation coefficient
    avg_pearson_test = total_pearson_test / n_splits
    print('saving model...')
    torch.save(best_model_params, model_saved_path)
    print(f"Average Pearson Correlation Coefficient on Test Set: {avg_pearson_test}")

    return best_X_train, best_X_test, best_Y_test, fold_results, avg_pearson_test


def uncertainty_estimation(best_X_test, best_Y_test, scaler_y, model_saved_path, preds_saved_path,
                           PosteriorPlot_saved_path, num_samples=200):
    predictions = []
    print('loading model...')
#     best_model = BayesMLP(best_X_test.shape[2], 1, [16, 16], activate='relu').to(device)
    best_model = BayesMLP(best_X_test.shape[2], 1, [16, 8], activate='relu').to(device)
    saved_model_params = torch.load(model_saved_path)
    best_model.load_state_dict(saved_model_params)

    best_model.eval()  


    # collect the prediction results for each sample
    all_predictions = []
    print('Start predicting...')
    with torch.no_grad():
        for _ in range(num_samples):
            output = best_model(best_X_test.to(device), 1) 
            predictions = output[0][0].cpu().numpy()
            predictions = scaler_y.inverse_transform(predictions)
            all_predictions.append(predictions)

    all_predictions = np.array(all_predictions).squeeze(2)  # Reshape the array

    # Initialize the DataFrame to store results.
    results_list = []
    idx_list = []
    mean_predictions = []
    true_values = []
    ci_lowers = []
    ci_uppers = []

    # Perform statistical calculations for each sample
    best_Y_test = scaler_y.inverse_transform(best_Y_test.numpy().reshape(-1, 1)).flatten()
    for idx in range(best_X_test.shape[0]):
        sample_preds = all_predictions[:, idx]  
        mean_pred = np.mean(sample_preds)
        median_pred = np.median(sample_preds)
        mode_pred = mode(sample_preds, keepdims=True).mode[0]
        variance_pred = np.var(sample_preds)
        std_dev_pred = np.std(sample_preds)
        ci_lower1 = mean_pred - 1.96 * std_dev_pred
        ci_upper1 = mean_pred + 1.96 * std_dev_pred
        ci_lower2 = np.percentile(sample_preds, 2.5)
        ci_upper2 = np.percentile(sample_preds, 97.5)
        true_value = best_Y_test[idx].item() if torch.is_tensor(best_Y_test) else best_Y_test[idx]

        idx_list.append(idx)
        mean_predictions.append(mean_pred)
        true_values.append(true_value)
        ci_lowers.append(ci_lower2)
        ci_uppers.append(ci_upper2)

        # Store the results in a dictionary.
        results_dict = {
            'Sample Index': idx,
            'True Value': true_value,
            'Mean Prediction': mean_pred,
            'Median Prediction': median_pred,
            'Mode Prediction': mode_pred,
            'Variance': variance_pred,
            'Standard Deviation': std_dev_pred,
            '95% CI Lower': ci_lower1,
            '95% CI Upper': ci_upper1,
            '2.5% Percentile': ci_lower2,
            '97.5% Percentile': ci_upper2
        }
        results_list.append(results_dict)

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    # Write to a CSV file, writing one row per iteration
    results_df.to_csv(preds_saved_path, mode='w', header=True, index=False)

    # Plot the predicted values, true values, and confidence intervals
    plt.figure(figsize=(12, 9))
    plt.plot(idx_list, mean_predictions, color='#9dc3e7', marker='.', linestyle='-', linewidth=2, markersize=2,
             label='Mean Predictions')
    plt.plot(idx_list, true_values, color='#f18180', marker='.', linestyle='-', linewidth=2, markersize=2,
             label='True Values')
    plt.fill_between(idx_list, ci_lowers, ci_uppers, color='gray', alpha=0.12, label='95% Confidence Interval')
    plt.xlabel('Sample Index', fontsize=16)
    plt.ylabel('Predicted/True Value', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Posterior Predictive', fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)

    if os.path.exists(PosteriorPlot_saved_path):
        os.remove(PosteriorPlot_saved_path)
    plt.savefig(PosteriorPlot_saved_path, bbox_inches='tight')
    plt.close()  

    print('Finish predict!')

    return results_df


def load_preprocess_select_features_with_lasso(X, Y, tar, feature_names, n_features_to_select, alpha=0.01):
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)

    # Use Lasso as the feature selection method
    lasso = Lasso(alpha=alpha)
    selector = SelectFromModel(lasso, max_features=n_features_to_select)
    X_selected = selector.fit_transform(X_scaled, Y)


    selected_features_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_features_indices]  



    X_selected = scaler_x.inverse_transform(X_scaled)[:, selected_features_indices]
    scaler_x_selected = MinMaxScaler()
    X_selected_scaled = scaler_x_selected.fit_transform(X_selected)

    scaler_y = MinMaxScaler()
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1)).flatten()

    X_tensors = torch.tensor(X_selected_scaled, dtype=torch.float32).unsqueeze(1)
    Y_tensors = torch.tensor(Y_scaled, dtype=torch.float32)


    return X_tensors, Y_tensors, scaler_y, selected_feature_names , scaler_x_selected


# Plot the loss curve for each fold
def plot_results(fold_results, LossPlot_saved_path):
    plt.figure(figsize=(12, 9))
    train_losses = [fold_result['train_losses'][-1] for fold_result in fold_results]  
    test_losses = [fold_result['test_loss'] for fold_result in fold_results]  
    folds = range(1, len(fold_results) + 1)
    plt.plot(folds, train_losses, color='#faccb0', marker='.', linestyle='-', linewidth=3, markersize=2,
             label='train_loss')
    plt.plot(folds, test_losses, color='#f5b0b0', marker='.', linestyle='-', linewidth=3, markersize=2,
             label='test_loss')
    plt.title('Loss  Fold', fontsize=16)
    plt.xlabel('Fold', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=True, framealpha=0.6)
 
    if os.path.exists(LossPlot_saved_path):
        os.remove(LossPlot_saved_path)
  
    plt.savefig(LossPlot_saved_path, bbox_inches='tight')
    plt.close()  


# Plot the correlation coefficient graph for each fold
def plot_pearson_correlations(fold_results, PearsonPlot_saved_path):
    plt.figure(figsize=(12, 9))
    pearson_train = [fold_result['pearson_train'][-1] for fold_result in fold_results]  
    pearson_test = [fold_result['pearson_test'] for fold_result in fold_results]  
    folds = range(1, len(fold_results) + 1)
    plt.plot(folds, pearson_train, color='#92A5D1', marker='.', linestyle='-', linewidth=4, markersize=2,
             label='pearson_train')
    plt.plot(folds, pearson_test, color='#D9B9D4', marker='.', linestyle='-', linewidth=4, markersize=2,
             label='pearson_test')
    plt.title('Pearson Correlation Coefficient over Epochs by Fold', fontsize=16)
    plt.xlabel('Fold', fontsize=16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper left', fontsize=16, frameon=False)
 
    if os.path.exists(PearsonPlot_saved_path):
        os.remove(PearsonPlot_saved_path)
   
    plt.savefig(PearsonPlot_saved_path, bbox_inches='tight')
    plt.close()  


def ShapExplainer(best_X_train, best_X_test, model_saved_path, SummaryPlot_saved_path, dependencePlot_saved_path,
                  forcePlot_saved_path, selected_feature_names,scaler_x_selected):
    print('Start SHAP!!! Loading model for ShapExplainer...')

    best_model = BayesMLP_copy(best_X_test.shape[2], 1, [16, 8], activate='relu', deterministic=True).to(device)
    saved_model_params = torch.load(model_saved_path)
    best_model.load_state_dict(saved_model_params)

    best_X_train = best_X_train.cpu().numpy()  
    best_X_test = best_X_test.cpu().numpy()
    best_X_train = best_X_train.reshape(best_X_train.shape[0], -1)
    best_X_test = best_X_test.reshape(best_X_test.shape[0], -1)
    best_X_train = scaler_x_selected.inverse_transform(best_X_train)
    best_X_test = scaler_x_selected.inverse_transform(best_X_test)
    best_X_train = torch.tensor(best_X_train, dtype=torch.float32).to(device)
    best_X_test = torch.tensor(best_X_test, dtype=torch.float32).to(device)


    best_model.eval()
    # Create SHAP explainer
    explainer = shap.GradientExplainer(best_model, best_X_train)
    # Calculate SHAP values.
    shap_values = explainer.shap_values(best_X_test)
    # print(f"shap_values shape: {np.shape(shap_values)}")
    # Remove redundant dimensions
    shap_values = np.squeeze(shap_values)  
    # 打Print the transformed shape.
    print(f"reshaped shap_values shape: {np.shape(shap_values)}")

    # Use squeeze to remove redundant dimensions from best_X_test.
    best_X_test_reshaped = best_X_test.cpu().numpy().squeeze()
    print(f"reshaped best_X_test shape: {np.shape(best_X_test_reshaped)}")
    feature_label = selected_feature_names

    # Step 1: Calculate the mean absolute SHAP value for each feature.
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    # Step 2: Sort the SHAP values and obtain the indices of the top 4 features.
    top_4_indices = np.argsort(mean_abs_shap_values)[-4:][::-1]
    # Step 3: Obtain the names of the top 4 features.
    top_4_feature_names = [feature_label[i] for i in top_4_indices]

    # summaryplot
    shap.summary_plot(shap_values, best_X_test_reshaped, show=False,
                      feature_names=feature_label, cmap='coolwarm', max_display=20) # cmap='RdYlBu'


    if os.path.exists(SummaryPlot_saved_path):
        os.remove(SummaryPlot_saved_path)

    plt.savefig(SummaryPlot_saved_path, bbox_inches='tight')  # bbox_inches='tight' 
    plt.close()  

    # dependencePlot
    print("Top 4 features by impact:")
    for idx, feature_name in enumerate(top_4_feature_names, start=1):
        for idxx in range(idx, 4):
            demo = idxx+1
            interaction_feature=top_4_feature_names[idxx]
            print(feature_name + '<--' + top_4_feature_names[idxx])
            suffix = f'_dependencePlot{idx}-{demo}.png'
            new_saved_path = dependencePlot_saved_path + suffix
            shap.dependence_plot(feature_name, shap_values, best_X_test_reshaped,
                                  show=False, interaction_index=interaction_feature,feature_names=feature_label) #, cmap='RdBu' =feature_name
      
            if os.path.exists(new_saved_path):
                os.remove(new_saved_path)
            plt.savefig(new_saved_path, bbox_inches='tight', pad_inches=0.3)  # bbox_inches='tight'
            plt.close()  
        print("Top 4 features by impact:")
     

        # forcePlot
        # Assume selecting the first sample for force plot visualization
        sample_index = 1 
        shap_value_single_sample = shap_values[sample_index]  
        feature_value_single_sample = best_X_train[sample_index].cpu().numpy()  
       
        with torch.no_grad():
            #  Input the training data into the model to obtain the output mean, which serves as the base value
            model_predictions = best_model(best_X_train.to(device)).cpu().numpy()
            base_value = model_predictions.mean()
        # Plot the force plot
        force_plot = shap.force_plot(
             base_value=base_value, 
             shap_values=shap_value_single_sample,  
             features=feature_value_single_sample,  
             feature_names=feature_label,  
             show=False
         )
        shap.save_html(forcePlot_saved_path, force_plot)

def main():
    data = pd.read_csv("model_data.csv")
    X = data.iloc[:, 60:].values
    # Obtain the original feature names (starting from the 60th column).
    feature_names = data.columns[60:]
    targets = data.iloc[:, 4:19]
    for tar in targets.columns:
        Y = targets[tar].values
        print('==' * 30)
        print(tar)

        # Perform dimensionality reduction using Lasso
        X_datas, Y_datas, scaler_y, selected_feature_names , scaler_x_selected = load_preprocess_select_features_with_lasso(X, Y, tar,
                                                                                                        feature_names,
                                                                                                        n_features_to_select=1000,
                                                                                                        alpha=0.01)  # n_features_to_select=300

        # Perform model training and cross-validation
        model_saved_path = f'BNN_saved_models/{tar}_model.pth'
        X_train, X_test, Y_test, fold_results, avg_pearson_test = cross_validate(X_datas, Y_datas, scaler_y,
                                                                                 model_saved_path, n_splits=10,
                                                                                 epochs=300, lr=0.001) 

        # Perform uncertainty analysis
        preds_saved_path = f'BNN_preds_result/{tar}_PredResult.csv'
        PosteriorPlot_saved_path = f'BNN_figure/PosteriorPlot/{tar}_PosteriorPlot.png'
        uncertainty_estimation(X_test, Y_test, scaler_y, model_saved_path, preds_saved_path, PosteriorPlot_saved_path,
                               num_samples=100)
        print(f'Finish {tar}_PosteriorPlot.png')
        # Generate and output uncertainty-related images.
        LossPlot_saved_path = f'BNN_figure/LossPlot/{tar}_LossPlot.png'
        PearsonPlot_saved_path = f'BNN_figure/PearsonPlot/{tar}_PearsonPlot.png'
        plot_results(fold_results, LossPlot_saved_path)
        print(f'Finish {tar}_LossPlot.png')
        plot_pearson_correlations(fold_results, PearsonPlot_saved_path)
        print(f'Finish {tar}_PearsonPlot.png')

        # Perform SHAP model interpretation.
        SummaryPlot_saved_path = f'BNN_figure/ShapPlot/SummaryPlot/{tar}_SummaryPlot.png'
        dependencePlot_saved_path = f'BNN_figure/ShapPlot/dependencePlot/{tar}'
        forcePlot_saved_path = f'BNN_figure/ShapPlot/forcePlot/{tar}_forcePlot.html'
        ShapExplainer(X_train, X_test, model_saved_path, SummaryPlot_saved_path, dependencePlot_saved_path,
                      forcePlot_saved_path,selected_feature_names,scaler_x_selected)
        print(f'Finish {tar}_SummaryPlot.png')
        print(f'Finish {tar}_dependencePlot.png')
        print(f'Finish {tar}_forcePlot.html')

        print('==' * 30)


if __name__ == '__main__':
    main()
