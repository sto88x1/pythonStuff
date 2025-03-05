import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Regression models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Import MultiOutputRegressor to enable multi-output support for models that need it
from sklearn.multioutput import MultiOutputRegressor

# Data loading function
def load_data(use_csv=False, file_path='data/energy_efficiency.csv'):
    """
    Load data either from a CSV file or from the California Housing dataset.
    
    Parameters:
    - use_csv (bool): If True, load data from a CSV file.
                      If False, load the California Housing dataset.
    - file_path (str): Path to the CSV file (if use_csv is True).
    
    Returns:
    - data (DataFrame): Loaded data.
    """
    if use_csv:
        data = pd.read_csv(file_path)
        print(f"Data loaded from CSV file: {file_path}")
    else:
        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        print("California Housing Data Loaded Successfully!")
    
    print("Data Shape:", data.shape)
    print("First 5 rows:\n", data.head())
    return data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using R² score and Mean Squared Error (MSE).
    Returns predictions along with the computed metrics.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mse, y_pred

def plot_model_results(model_name, y_test, y_pred, output_names):
    """
    Group plots by model. For each output (or single output) the function plots:
    - Predicted vs. Actual values
    - Residuals vs. Predicted values
    - Error distribution histogram
    
    If there are multiple outputs, a row is created for each output.
    """
    if isinstance(y_test, pd.DataFrame):
        n_outputs = len(output_names)
        fig, axes = plt.subplots(n_outputs, 3, figsize=(18, 5 * n_outputs))
        if n_outputs == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, col in enumerate(output_names):
            # Predicted vs. Actual
            ax = axes[i, 0]
            ax.scatter(y_test[col], y_pred[:, i], edgecolor='k', alpha=0.7)
            ax.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{model_name} - {col}: Predicted vs Actual")
            
            # Residual Plot
            residuals = y_test[col] - y_pred[:, i]
            ax = axes[i, 1]
            ax.scatter(y_pred[:, i], residuals, edgecolor='k', alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residuals")
            ax.set_title(f"{model_name} - {col}: Residual Plot")
            
            # Error Distribution
            ax = axes[i, 2]
            ax.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
            ax.set_xlabel("Residuals")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model_name} - {col}: Error Distribution")
    else:
        # Single output
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Predicted vs. Actual
        ax = axes[0]
        ax.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name}: Predicted vs Actual")
        
        # Residual Plot
        residuals = y_test - y_pred
        ax = axes[1]
        ax.scatter(y_pred, residuals, edgecolor='k', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{model_name}: Residual Plot")
        
        # Error Distribution
        ax = axes[2]
        ax.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model_name}: Error Distribution")
    
    plt.tight_layout()
    plt.show()

def write_report(results, filename="model_report.txt"):
    """
    Write a detailed report of model performance to a text file.
    """
    with open(filename, "w") as f:
        f.write("Model Performance Report\n")
        f.write("=" * 30 + "\n")
        for name, r2, mse in results:
            f.write(f"Model: {name}\n")
            f.write(f"  R2 Score: {r2:.4f}\n")
            f.write(f"  Mean Squared Error: {mse:.4f}\n")
            f.write("-" * 30 + "\n")
    print(f"Report written to {filename}")

if __name__ == "__main__":
    # Choose data source:
    # Set use_csv = True to use your CSV file (e.g., the Energy Efficiency dataset).
    # Set use_csv = False to use the built-in California Housing dataset.
    use_csv = True
    file_path = 'data/energy_efficiency.csv'  # Ensure this CSV is available if use_csv is True.
    
    # Load the data
    data = load_data(use_csv=use_csv, file_path=file_path)
    
    # Define features and targets based on the dataset structure
    if use_csv:
        # For the Energy Efficiency dataset: first 8 columns are inputs and last 2 are outputs.
        feature_cols = data.columns[:8].tolist()
        target_cols = data.columns[8:].tolist()
        print("\nUsing Energy Efficiency dataset with multiple outputs:")
        print("Input features:", feature_cols)
        print("Target outputs:", target_cols)
    else:
        # For the California Housing dataset, the target is 'MedHouseVal'
        feature_cols = data.columns.drop('MedHouseVal').tolist()
        target_cols = ['MedHouseVal']
    
    # Split data into features (X) and target (y)
    X = data[feature_cols]
    y = data[target_cols] if len(target_cols) > 1 else data[target_cols[0]]
    
    # Create an 80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Determine if the task is multi-output
    multi_output = isinstance(y, pd.DataFrame)
    
    # Wrap models that do not natively support multi-output.
    if multi_output:
        svr_model = MultiOutputRegressor(SVR())
        gbr_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        abr_model = MultiOutputRegressor(AdaBoostRegressor(random_state=42))
    else:
        svr_model = SVR()
        gbr_model = GradientBoostingRegressor(random_state=42)
        abr_model = AdaBoostRegressor(random_state=42)
    
    # Define nine models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Support Vector Regression": svr_model,
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Gradient Boosting": gbr_model,
        "AdaBoost": abr_model,
        "Extra Trees": ExtraTreesRegressor(random_state=42),
        "Ridge Regression": Ridge()
    }
    
    results = []  # To store performance metrics for each model
    
    # Loop over each model: train, evaluate, and generate grouped plots
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Uncomment below for hyperparameter tuning with GridSearchCV if needed in the future
        # if name == "Decision Tree":
        #     param_grid = {'max_depth': [None, 5, 10, 20]}
        #     grid_search = GridSearchCV(model, param_grid, cv=5)
        #     grid_search.fit(X_train, y_train)
        #     model = grid_search.best_estimator_
        # else:
        #     model.fit(X_train, y_train)
        
        model.fit(X_train, y_train)
        r2, mse, y_pred = evaluate_model(model, X_test, y_test)
        results.append((name, r2, mse))
        print(f"{name} Performance:")
        print("  R2 Score:", r2)
        print("  Mean Squared Error:", mse)
        
        # Generate grouped plots by model.
        if isinstance(y_test, pd.DataFrame):
            plot_model_results(name, y_test, y_pred, target_cols)
        else:
            plot_model_results(name, y_test, y_pred, [target_cols[0]])
    
    # Write detailed performance report to a file.
    write_report(results, filename="model_report.txt")
