import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

# For DOE generation using Latin Hypercube Sampling
from scipy.stats import qmc
import itertools

###########################################
#    Functions for Data & DOE Generation  #
###########################################

def load_training_data(use_csv=False, file_path='data/energy_efficiency.csv'):
    """
    Load the training dataset either from a CSV file or from the California Housing dataset.
    """
    if use_csv:
        data = pd.read_csv(file_path)
        print(f"Training data loaded from CSV file: {file_path}")
    else:
        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        print("California Housing Data Loaded Successfully!")
    print("Training data shape:", data.shape)
    return data

def load_or_generate_inputs(X_train, feature_cols, method='latin_hypercube', n_samples=20, levels=3):
    """
    If an 'input.csv' file exists, load new input data from it.
    Otherwise, generate a DOE design table based on the bounds of X_train.
    The returned DataFrame will contain only the feature columns.
    
    If the loaded CSV has extra columns (beyond the feature columns),
    they are assumed to be the ground truth outputs.
    """
    if os.path.exists("input.csv"):
        new_inputs = pd.read_csv("input.csv")
        print("Loaded new inputs from input.csv")
    else:
        # Determine bounds for each input variable from X_train
        bounds = []
        for col in feature_cols:
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            bounds.append((min_val, max_val))
        print("Generating DOE design using", method)
        
        if method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=len(bounds))
            sample = sampler.random(n=n_samples)
            l_bounds = [b[0] for b in bounds]
            u_bounds = [b[1] for b in bounds]
            new_inputs_array = qmc.scale(sample, l_bounds, u_bounds)
            new_inputs = pd.DataFrame(new_inputs_array, columns=feature_cols)
        elif method == 'full_factorial':
            grids = []
            for (min_val, max_val) in bounds:
                grids.append(np.linspace(min_val, max_val, levels))
            full_grid = list(itertools.product(*grids))
            new_inputs = pd.DataFrame(full_grid, columns=feature_cols)
        else:
            raise ValueError("Unknown DOE method")
    return new_inputs

###########################################
#       Functions for Training Models     #
###########################################

def train_models(X_train, y_train, multi_output):
    """
    Train a set of nine regression models on the training data.
    For models that do not natively support multi-output, wrap them using MultiOutputRegressor.
    """
    if multi_output:
        svr_model = MultiOutputRegressor(SVR())
        gbr_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        abr_model = MultiOutputRegressor(AdaBoostRegressor(random_state=42))
    else:
        svr_model = SVR()
        gbr_model = GradientBoostingRegressor(random_state=42)
        abr_model = AdaBoostRegressor(random_state=42)
        
    models = {
        #"Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Regression": svr_model,
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Gradient Boosting": gbr_model,
        #"AdaBoost": abr_model,
        "Extra Trees": ExtraTreesRegressor(random_state=42),
        "Ridge Regression": Ridge()
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    return models

###########################################
#         Functions for Prediction        #
###########################################

def predict_new_inputs(models, new_inputs):
    """
    Use the trained models to predict outputs for the new input data.
    Returns a dictionary with model names as keys and predictions as values.
    """
    predictions = {}
    for name, model in models.items():
        pred = model.predict(new_inputs)
        predictions[name] = pred
    return predictions

###########################################
#  Functions for Plotting and Reporting   #
###########################################

def plot_predictions_with_truth(new_inputs, predictions, truth, target_cols):
    """
    For each model, plot predicted vs. actual (if ground truth is available),
    along with residuals and error distributions.
    """
    for model_name, pred in predictions.items():
        if truth.ndim == 1 or truth.shape[1] == 1:
            plt.figure(figsize=(14,4))
            # Predicted vs. Actual
            plt.subplot(1, 3, 1)
            plt.scatter(truth, pred, edgecolor='k', alpha=0.7)
            plt.plot([truth.min(), truth.max()], [truth.min(), truth.max()], 'r--')
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{model_name}: Predicted vs Actual")
            
            # Residual plot
            residuals = truth - pred
            plt.subplot(1, 3, 2)
            plt.scatter(pred, residuals, edgecolor='k', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel("Predicted")
            plt.ylabel("Residuals")
            plt.title(f"{model_name}: Residuals")
            
            # Error distribution
            plt.subplot(1, 3, 3)
            plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
            plt.xlabel("Residuals")
            plt.ylabel("Frequency")
            plt.title(f"{model_name}: Error Distribution")
            plt.tight_layout()
            plt.show()
        else:
            n_outputs = truth.shape[1]
            for i, col in enumerate(target_cols):
                plt.figure(figsize=(14,4))
                # Predicted vs. Actual
                plt.subplot(1, 3, 1)
                plt.scatter(truth.iloc[:, i], pred[:, i], edgecolor='k', alpha=0.7)
                plt.plot([truth.iloc[:, i].min(), truth.iloc[:, i].max()],
                         [truth.iloc[:, i].min(), truth.iloc[:, i].max()], 'r--')
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{model_name} ({col}): Predicted vs Actual")
                
                # Residual plot
                residuals = truth.iloc[:, i] - pred[:, i]
                plt.subplot(1, 3, 2)
                plt.scatter(pred[:, i], residuals, edgecolor='k', alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Predicted")
                plt.ylabel("Residuals")
                plt.title(f"{model_name} ({col}): Residuals")
                
                # Error distribution
                plt.subplot(1, 3, 3)
                plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
                plt.xlabel("Residuals")
                plt.ylabel("Frequency")
                plt.title(f"{model_name} ({col}): Error Distribution")
                plt.tight_layout()
                plt.show()

def plot_prediction_distributions(predictions, target_cols):
    """
    When no ground truth is available, for each output variable, produce box plots and histograms
    to compare the predictions from different models.
    """
    for out_idx, col in enumerate(target_cols):
        plt.figure(figsize=(10, 6))
        data = {}
        for model_name, pred in predictions.items():
            if pred.ndim == 1:
                data[model_name] = pred
            else:
                data[model_name] = pred[:, out_idx]
        df_box = pd.DataFrame(data)
        df_box.boxplot()
        plt.title(f"Box Plot of Predictions for {col}")
        plt.ylabel("Predicted Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for model_name, values in data.items():
            plt.hist(values, bins=20, alpha=0.5, label=model_name, edgecolor='k')
        plt.title(f"Histogram of Predictions for {col}")
        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

def write_prediction_report(predictions, truth, target_cols, filename="prediction_report.txt"):
    """
    Write a text report summarizing prediction performance.
    If ground truth is available, include error metrics;
    otherwise, include summary statistics (mean, std, etc.) for each model.
    """
    with open(filename, "w") as f:
        f.write("Prediction Performance Report\n")
        f.write("="*40 + "\n\n")
        if truth is not None:
            if truth.ndim == 1 or truth.shape[1] == 1:
                for model_name, pred in predictions.items():
                    r2 = r2_score(truth, pred)
                    mse = mean_squared_error(truth, pred)
                    f.write(f"Model: {model_name}\n")
                    f.write(f"  R2 Score: {r2:.4f}\n")
                    f.write(f"  Mean Squared Error: {mse:.4f}\n")
                    f.write("-"*40 + "\n")
            else:
                for model_name, pred in predictions.items():
                    f.write(f"Model: {model_name}\n")
                    for i, col in enumerate(target_cols):
                        r2 = r2_score(truth.iloc[:, i], pred[:, i])
                        mse = mean_squared_error(truth.iloc[:, i], pred[:, i])
                        f.write(f"  {col} - R2 Score: {r2:.4f}, MSE: {mse:.4f}\n")
                    f.write("-"*40 + "\n")
        else:
            for model_name, pred in predictions.items():
                f.write(f"Model: {model_name}\n")
                if pred.ndim == 1:
                    stats = pd.Series(pred).describe()
                    f.write(str(stats) + "\n")
                else:
                    for i, col in enumerate(target_cols):
                        stats = pd.Series(pred[:, i]).describe()
                        f.write(f"{col} statistics:\n")
                        f.write(str(stats) + "\n")
                f.write("-"*40 + "\n")
    print(f"Prediction report written to {filename}")

###########################################
# New Feature: Scatter Plots of Inputs vs. Outputs
###########################################

def plot_input_vs_output_scatter_all_models(new_inputs, predictions, target_cols):
    """
    For each output variable, create scatter plots of each input variable versus the predicted output,
    overlaying the predictions from all models in the same subplot.
    
    Parameters:
      - new_inputs: DataFrame containing input features.
      - predictions: Dictionary with model names as keys and their prediction arrays as values.
      - target_cols: List of target variable names.
    """
    import math
    import matplotlib.pyplot as plt

    # Determine the number of input features.
    n_inputs = new_inputs.shape[1]
    n_cols = 3  # You can adjust the grid layout as needed.
    n_rows = math.ceil(n_inputs / n_cols)

    # For each output variable...
    for out_idx, target_name in enumerate(target_cols):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        # Flatten axes in case of a grid.
        axes = axes.flatten() if n_inputs > 1 else [axes]

        for i, col in enumerate(new_inputs.columns):
            ax = axes[i]
            # Plot each model's predictions in the same subplot.
            for model_name, pred in predictions.items():
                # Check if the prediction is multi-output or single-output.
                if pred.ndim == 1:
                    y_vals = pred
                else:
                    y_vals = pred[:, out_idx]
                ax.scatter(new_inputs[col], y_vals, alpha=0.7, edgecolor='k', label=model_name)
            ax.set_xlabel(col)
            ax.set_ylabel(f"Predicted {target_name}")
            ax.set_title(f"{col} vs Predicted {target_name}")
            ax.legend(fontsize='small', loc='best')
        # Hide any extra subplots.
        for j in range(i+1, n_rows*n_cols):
            axes[j].set_visible(False)
        plt.suptitle(f"Overlay: Input Features vs Predicted {target_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def plot_output_relationships_all_models(predictions, target_cols):
    """
    If multiple outputs are predicted, produce scatter plots for each pair of outputs
    overlaying predictions from all models. For each model, also fit a linear regression
    (using np.polyfit) on the predicted outputs and plot the line as a dashed line.
    
    Parameters:
      - predictions: Dictionary with model names as keys and prediction arrays as values.
      - target_cols: List of target variable names.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    
    if len(target_cols) < 2:
        print("Only one output variable provided; no output relationships to plot.")
        return
    
    # Define markers and colors for clarity.
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x']
    colors = plt.cm.tab10.colors  # A set of 10 distinct colors
    
    # Create scatter plots for each pair of outputs.
    output_pairs = list(itertools.combinations(range(len(target_cols)), 2))
    for (i, j) in output_pairs:
        plt.figure(figsize=(8, 6))
        for k, (model_name, pred) in enumerate(predictions.items()):
            # Ensure the prediction is multi-output.
            if pred.ndim > 1:
                x_vals = pred[:, i]
                y_vals = pred[:, j]
            else:
                continue
            marker = markers[k % len(markers)]
            color = colors[k % len(colors)]
            plt.scatter(x_vals, y_vals, alpha=0.7, edgecolor='k',
                        marker=marker, color=color, label=model_name)
            # Fit a linear regression on x_vals and y_vals.
            sorted_idx = np.argsort(x_vals)
            x_sorted = x_vals[sorted_idx]
            y_sorted = y_vals[sorted_idx]
            coeffs = np.polyfit(x_sorted, y_sorted, 1)
            poly_eqn = np.poly1d(coeffs)
            y_fit = poly_eqn(x_sorted)
            plt.plot(x_sorted, y_fit, linestyle='--', color=color)
        plt.xlabel(f"Predicted {target_cols[i]}")
        plt.ylabel(f"Predicted {target_cols[j]}")
        plt.title(f"Predicted {target_cols[i]} vs. {target_cols[j]} (All Models)")
        plt.legend(fontsize='small', loc='best')
        plt.tight_layout()
        plt.show()

###########################################
#             Main Prediction             #
###########################################

if __name__ == "__main__":
    # ----- Load training data and define features/targets -----
    use_csv = True  # Use Energy Efficiency dataset
    train_file = 'data/energy_efficiency.csv'
    data = load_training_data(use_csv=use_csv, file_path=train_file)
    
    # For Energy Efficiency: first 8 columns are features, last 2 are outputs.
    if use_csv:
        feature_cols = data.columns[:8].tolist()
        target_cols = data.columns[8:].tolist()
    else:
        feature_cols = data.columns.drop('MedHouseVal').tolist()
        target_cols = ['MedHouseVal']
    
    X = data[feature_cols]
    y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
    multi_output = isinstance(y, pd.DataFrame)
    
    # Use the entire X for bounds in DOE generation.
    X_train = X.copy()
    y_train = y.copy()
    
    # ----- Train models -----
    models = train_models(X_train, y_train, multi_output)
    
    # ----- Load or generate new inputs -----
    new_inputs = load_or_generate_inputs(X_train, feature_cols, method='latin_hypercube', n_samples=40, levels=3)
    
    # Check if ground truth outputs are provided with new inputs.
    provided_truth = None
    extra_cols = [col for col in new_inputs.columns if col not in feature_cols]
    if extra_cols:
        provided_truth = new_inputs[extra_cols]
        new_inputs = new_inputs[feature_cols]  # Keep only feature columns for prediction
        print("Ground truth outputs found in input.csv:", extra_cols)
    else:
        print("No ground truth outputs provided with new inputs.")
    
    # ----- Make predictions using the trained models -----
    predictions = predict_new_inputs(models, new_inputs)
    
    # Write predictions to a CSV file.
    results = new_inputs.copy()
    for model_name, pred in predictions.items():
        if pred.ndim == 1:
            results[model_name] = pred
        else:
            for i in range(pred.shape[1]):
                results[f"{model_name}_out{i+1}"] = pred[:, i]
    results.to_csv("predictions.csv", index=False)
    print("Predictions written to predictions.csv")
    
    # ----- Generate Plots and Report -----
    if provided_truth is not None:
        # Ground truth available: plot predicted vs. actual for each model.
        if not isinstance(provided_truth, pd.DataFrame):
            provided_truth = pd.DataFrame(provided_truth)
        for model_name, pred in predictions.items():
            plot_predictions_with_truth(new_inputs, {model_name: pred}, provided_truth, target_cols)
    else:
        # No ground truth: compare prediction distributions across models.
        plot_prediction_distributions(predictions, target_cols)
    
    # Write a prediction report.
    write_prediction_report(predictions, provided_truth, target_cols, filename="prediction_report.txt")
    
    # Example of calling the new function (to be added after predictions are generated):
    # Assuming new_inputs, predictions, and target_cols are already defined.
    #plot_input_vs_output_scatter_all_models(new_inputs, predictions, target_cols)
    # --- At the end of your Prediction Script, after generating predictions ---
    # Assuming new_inputs, predictions, and target_cols are already defined
    plot_output_relationships_all_models(predictions, target_cols)