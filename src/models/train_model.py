"""
Model Training for US Traffic Accidents.

This script performs:
1. Data preparation
2. Model training with XGBoost
3. Model evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette('viridis')

# Define paths
DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports/figures/eda')
MODELS_DIR = Path('models')

def load_data():
    """Load the processed dataset with features."""
    df = pd.read_csv(DATA_DIR / 'accidents_with_features.csv')
    return df

def prepare_data(df):
    """Prepare data for modeling."""
    # Select features for modeling
    feature_cols = [
        # Temporal features
        'start_hour', 'start_day_of_week', 'start_month', 'is_weekend',
        'is_rush_hour', 'is_holiday', 'time_of_day', 'season',
        
        # Weather features
        'temperature_category', 'wind_speed_category', 'humidity_category',
        'visibility_category', 'weather_severity',
        
        # Road features
        'is_intersection', 'has_traffic_control', 'is_complex_intersection',
        'road_feature_count',
        
        # Location features
        'is_urban', 'region'
    ]
    
    # Target variable: Severity (1-4)
    X = df[feature_cols]
    y = df['Severity'] - 1  # Convert to 0-based indexing
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with proper class imbalance handling."""
    # Calculate class weights
    class_counts = y_train.value_counts()
    class_weights = {i: sum(class_counts) / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters for full dataset
    params = {
        'objective': 'multi:softmax',
        'num_class': 4,
        'max_depth': 8,  # Increased for more complex patterns
        'eta': 0.05,     # Reduced learning rate for stability
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        # 'scale_pos_weight': [class_weights[i] for i in range(4)], # Often better handled by sample_weight if supported directly or adjusting objectives
        'eval_metric': 'merror',
        'tree_method': 'hist',  # Use histogram-based algorithm for efficiency
        'grow_policy': 'lossguide'  # Grow trees based on loss reduction
    }
    
    # Dictionary to store evaluation results
    evals_result = {}
    
    # Train model with more rounds for full dataset
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,  # Increased number of rounds
        evals=[(dtrain, 'train'), (dtest, 'test')],
        evals_result=evals_result, # Store evaluation results
        early_stopping_rounds=20,  # Increased patience
        verbose_eval=20  # Less frequent logging
    )
    
    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODELS_DIR / 'xgb_model.json')
    
    return model, evals_result # Return model and training history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)
    
    # Convert predictions back to 1-based indexing for reporting
    y_test_1based = y_test + 1
    y_pred_1based = y_pred + 1
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_1based, y_pred_1based))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_1based, y_pred_1based)
    
    # Define the axis labels (1-4)
    axis_labels = [1, 2, 3, 4]
    
    # Plot heatmap with custom axis labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=axis_labels, yticklabels=axis_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Severity')
    plt.ylabel('Actual Severity')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'confusion_matrix.png')
    plt.close() # Close the plot figure

def plot_training_history(evals_result):
    """Plots the training and validation classification error curves."""
    train_error = evals_result['train']['merror']
    test_error = evals_result['test']['merror']
    epochs = range(1, len(train_error) + 1)
    
    # --- Debugging Print Statements --- 
    print("--- Training Error Rate Values ---")
    print(f"First 5: {train_error[:5]}")
    print(f"Last 5: {train_error[-5:]}")
    print("--- Validation Error Rate Values ---")
    print(f"First 5: {test_error[:5]}")
    print(f"Last 5: {test_error[-5:]}")
    print("-----------------------------")
    # --- End Debugging --- 
    
    plt.figure(figsize=(12, 6))
    # Use different styles to ensure visibility
    plt.plot(epochs, train_error, 'b--', label='Training Error Rate')
    plt.plot(epochs, test_error, 'r-', label='Validation Error Rate')
    plt.title('Training and Validation Classification Error')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Classification Error Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'training_performance.png')
    plt.close() # Close the plot figure
    print(f"Training performance plot saved to {REPORTS_DIR / 'training_performance.png'}")

def main():
    """Main function to run the model training and evaluation."""
    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data()
    
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    print("Training model...")
    model, evals_result = train_model(X_train, y_train, X_test, y_test)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Plotting training history...")
    plot_training_history(evals_result)
    
    print("Model training and evaluation complete! Check the reports/figures/eda directory for visualizations.")

if __name__ == "__main__":
    main() 