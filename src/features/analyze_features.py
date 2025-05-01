"""
Feature Analysis for US Traffic Accidents.

This script performs detailed feature analysis including:
1. SHAP value analysis
2. Feature importance visualization
3. Feature relationship analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette('viridis')

# Define paths
DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports/figures')
MODELS_DIR = Path('models')

def load_data_and_model():
    """Load the processed data and trained model."""
    print("Loading data and model...")
    
    # Load data
    df = pd.read_csv(DATA_DIR / 'accidents_with_features.csv')
    
    # Load model
    model = xgb.Booster()
    model.load_model(MODELS_DIR / 'xgb_model.json')
    
    return df, model

def prepare_data(df):
    """Prepare data for feature analysis."""
    # Select features for analysis
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
    
    # Prepare features
    X = df[feature_cols]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X[col] = pd.factorize(X[col])[0]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols

def analyze_shap_values(model, X, feature_cols, sample_size=1000):
    """Analyze feature importance using SHAP."""
    print("Calculating SHAP values...")
    
    # Take a random sample for SHAP analysis
    sample_size = min(sample_size, len(X))
    sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for the sample
    shap_values = explainer.shap_values(X_sample)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar")
    plt.title('Feature Importance from XGBoost')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_importance.png')
    
    # Save raw SHAP values and sample indices
    np.save(REPORTS_DIR / 'shap_values.npy', shap_values)
    np.save(REPORTS_DIR / 'sample_indices.npy', sample_indices)
    
    print("SHAP analysis complete!")
    return shap_values

def analyze_feature_relationships(df):
    """Analyze relationships between features and target."""
    print("Analyzing feature relationships...")
    
    # 1. Temporal Patterns
    plt.figure(figsize=(15, 10))
    
    # Time of day vs Severity
    plt.subplot(2, 2, 1)
    sns.boxplot(x='time_of_day', y='Severity', data=df)
    plt.title('Accident Severity by Time of Day')
    plt.xticks(rotation=45)
    
    # Season vs Severity
    plt.subplot(2, 2, 2)
    sns.boxplot(x='season', y='Severity', data=df)
    plt.title('Accident Severity by Season')
    
    # Weather Severity vs Accident Severity
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='weather_severity', y='Severity', data=df, alpha=0.1)
    plt.title('Weather Severity vs Accident Severity')
    
    # Road Features vs Severity
    plt.subplot(2, 2, 4)
    road_features = ['is_intersection', 'has_traffic_control', 'is_complex_intersection']
    road_severity = df.groupby(road_features)['Severity'].mean().reset_index()
    sns.heatmap(road_severity.pivot_table(index='is_intersection', 
                                        columns='has_traffic_control', 
                                        values='Severity'),
                annot=True, fmt='.2f')
    plt.title('Average Severity by Road Features')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_relationships.png')
    
    # 2. Regional Analysis
    plt.figure(figsize=(12, 6))
    region_severity = df.groupby('region')['Severity'].mean().sort_values()
    sns.barplot(x=region_severity.index, y=region_severity.values)
    plt.title('Average Accident Severity by Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'regional_severity.png')
    
    print("Feature relationship analysis complete!")

def main():
    """Main function to run the feature analysis."""
    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data and model
    df, model = load_data_and_model()
    
    # Prepare data
    X, feature_cols = prepare_data(df)
    
    # Analyze SHAP values
    shap_values = analyze_shap_values(model, X, feature_cols)
    
    # Analyze feature relationships
    analyze_feature_relationships(df)
    
    print("Feature analysis complete! Check the reports/figures directory for visualizations.")

if __name__ == "__main__":
    main() 