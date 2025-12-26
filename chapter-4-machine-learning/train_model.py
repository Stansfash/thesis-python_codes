"""
Model Training Script for Dairy Milk Yield Prediction
Creates the best_my_pred_model.joblib file needed for optimization
Author: Cleaned version for compatibility
"""

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, make_scorer, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = 'new_data.csv'
RESULTS_DIR = 'results'
MODEL_FILENAME = 'best_my_pred_model.joblib'

# Data configuration
SELECTED_COLUMNS = ["breed", "sol", "dim", "lact", "conc", "vp", "min_temp", "max_temp", "rh_tmin", "rh_tmax"]
TARGET_COLUMN = 'my'
TEST_SIZE = 0.2
RANDOM_STATE = 47819283

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_process_data(file_path):
    """
    Loads data from the specified file path and processes it by converting 
    specified columns to object types and dates to datetime.

    Args:
        file_path (str): The path to the CSV file containing the data.

    Returns:
        DataFrame: The processed DataFrame with the specified conversions.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"  ✓ Loaded {len(data)} records from {file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Convert specified columns to object types
    object_columns = ['cow_id', 'breed', 'breed_c', 'sol', 'lact', 'nvisits', 'IsYieldValid']
    data[object_columns] = data[object_columns].astype('object')

    # Convert date columns to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['calving_date'] = pd.to_datetime(data['calving_date'], errors='coerce')

    return data


def data_preprocessing(data, selected_columns, target_column, test_size=0.2, random_state=42):
    """
    Performs feature engineering and splits the DataFrame into training and testing sets.

    Args:
        data: The DataFrame to be processed.
        selected_columns: A list of columns to select from the DataFrame.
        target_column: The name of the target column.
        test_size: The proportion of the data to be used for testing.
        random_state: A random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    # Select the desired columns
    processed_data = data[selected_columns].copy()
    
    # Sort by date before splitting
    data_sorted = data.sort_values('date')

    X = processed_data
    y = data_sorted[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def create_pipeline(model, X_train):
    """
    Creates a machine learning pipeline with preprocessing steps.

    Args:
        model: The machine learning model to be included in the pipeline.
        X_train: Training data to determine feature types.

    Returns:
        Pipeline object that includes preprocessing and the model.
    """
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def evaluate(true, predicted):
    """
    Calculate evaluation metrics.
    
    Args:
        true: True target values
        predicted: Predicted values
        
    Returns:
        Tuple of metrics: (MAE, MAPE, MSE, MSLE, RMSE, RMSLE, R2)
    """
    mae = round(mean_absolute_error(true, predicted), 4)
    mape = round(mean_absolute_percentage_error(true, predicted), 4)
    mse = round(mean_squared_error(true, predicted), 4)
    msle = round(mean_squared_log_error(true, predicted), 4)
    rmse = round(np.sqrt(mse), 4)
    rmsle = round(np.sqrt(msle), 4)
    r2 = round(r2_score(true, predicted), 4)
    
    return mae, mape, mse, msle, rmse, rmsle, r2


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_best_model():
    """
    Train the best model using GridSearchCV and save it.
    """
    print("="*80)
    print("MODEL TRAINING FOR DAIRY MILK YIELD PREDICTION")
    print("="*80)
    
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"\n  ✓ Created directory: {RESULTS_DIR}/")
    
    # Load and process data
    print("\n[1/5] Loading and processing data...")
    data = load_and_process_data(DATA_FILE)
    if data is None:
        print("ERROR: Failed to load data. Exiting.")
        return
    
    # Prepare training and testing sets
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = data_preprocessing(
        data, SELECTED_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
    )
    print(f"  ✓ Training set: {len(X_train)} samples")
    print(f"  ✓ Testing set: {len(X_test)} samples")
    
    # Define models and hyperparameters
    print("\n[3/5] Setting up models and hyperparameters...")
    models = {
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(),
            'params': {
                'model__n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [5, 10],
                'model__min_samples_split': [5],
                'model__min_samples_leaf': [5]
            }
        },
        'XGBRegressor': {
            'model': XGBRegressor(),
            'params': {
                'model__n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [5, 10],
                'model__reg_alpha': [0.5],
                'model__reg_lambda': [0.5]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(),
            'params': {
                'model__n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
                'model__max_depth': [5, 10],
                'model__min_samples_split': [5],
                'model__min_samples_leaf': [5]
            }
        }
    }
    
    cv_strategies = {
        '5-Fold': KFold(n_splits=5, shuffle=False),
        '10-Fold': KFold(n_splits=10, shuffle=False)
    }
    
    print(f"  ✓ Models to evaluate: {len(models)}")
    print(f"  ✓ CV strategies: {len(cv_strategies)}")
    
    # Grid search for best model
    print("\n[4/5] Performing grid search (this will take a while)...")
    start_time = time.time()
    
    best_score = -np.inf
    best_model_info = None
    best_params = None
    best_pipeline = None
    
    total_combinations = len(models) * len(cv_strategies)
    current = 0
    
    for model_name, model_info in models.items():
        for cv_name, cv in cv_strategies.items():
            current += 1
            print(f"\n  [{current}/{total_combinations}] Training {model_name} with {cv_name}...")
            
            # Create pipeline
            pipeline = create_pipeline(model_info['model'], X_train)
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=model_info['params'],
                cv=cv,
                scoring={
                    'R2': 'r2',
                    'MSE': make_scorer(mean_squared_error, greater_is_better=False),
                    'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
                },
                refit='MSE',
                verbose=0,
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train.values.ravel() if hasattr(y_train, 'values') else y_train)
            
            # Check if this is the best model
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model_info = (model_name, cv_name)
                best_params = grid_search.best_params_
                best_pipeline = grid_search.best_estimator_
                print(f"    → New best model! MSE: {-best_score:.4f}")
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n  Grid search completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Evaluate best model
    print("\n[5/5] Evaluating best model...")
    print(f"\n  Best Model: {best_model_info[0]}")
    print(f"  Best CV: {best_model_info[1]}")
    print(f"  Best Parameters: {best_params}")
    
    # Make predictions
    train_pred = best_pipeline.predict(X_train)
    test_pred = best_pipeline.predict(X_test)
    
    # Calculate metrics
    train_metrics = evaluate(y_train, train_pred)
    test_metrics = evaluate(y_test, test_pred)
    
    # Display metrics
    print("\n  Performance Metrics:")
    print("  " + "-"*70)
    metric_names = ['MAE', 'MAPE', 'MSE', 'MSLE', 'RMSE', 'RMSLE', 'R2']
    print(f"  {'Metric':<10} {'Train':<15} {'Test':<15}")
    print("  " + "-"*70)
    for i, metric in enumerate(metric_names):
        print(f"  {metric:<10} {train_metrics[i]:<15} {test_metrics[i]:<15}")
    print("  " + "-"*70)
    
    # Save model
    model_path = os.path.join(RESULTS_DIR, MODEL_FILENAME)
    joblib.dump(best_pipeline, model_path)
    print(f"\n  ✓ Model saved to: {model_path}")
    
    # Save model info
    model_info_path = os.path.join(RESULTS_DIR, 'model_info.txt')
    with open(model_info_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Model: {best_model_info[0]}\n")
        f.write(f"Best CV Strategy: {best_model_info[1]}\n")
        f.write(f"Best MSE: {-best_score:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Metric':<10} {'Train':<15} {'Test':<15}\n")
        f.write("-"*40 + "\n")
        for i, metric in enumerate(metric_names):
            f.write(f"{metric:<10} {train_metrics[i]:<15} {test_metrics[i]:<15}\n")
        f.write("\n")
        f.write(f"Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        
        # Add sklearn version info
        import sklearn
        f.write(f"Scikit-learn Version: {sklearn.__version__}\n")
    
    print(f"  ✓ Model info saved to: {model_info_path}")
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nYou can now run the optimization scripts:")
    print(f"  1. python multi_algorithm_optimization.py")
    print(f"  2. python statistical_analysis.py")
    print("="*80)


if __name__ == "__main__":
    # Print sklearn version info
    import sklearn
    print(f"\nUsing scikit-learn version: {sklearn.__version__}")
    print("This version will be saved with the model for future reference.\n")
    
    train_best_model()
