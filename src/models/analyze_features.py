"""
Feature Analysis and Model Building for US Traffic Accidents.

This script performs:
1. Feature importance analysis using various methods
2. Initial model building with different algorithms
3. Visualization of feature relationships and model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette('viridis')

# Define paths
DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports/figures')

def load_data():
    """Load the processed dataset with features."""
    df = pd.read_csv(DATA_DIR / 'accidents_with_features.csv')
    # Take a 10% sample for faster processing
    df = df.sample(frac=0.1, random_state=42)
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
    y = df['Severity']
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols

def analyze_feature_importance(X_train, y_train, feature_cols):
    """Analyze feature importance using Random Forest."""
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title('Feature Importance from Random Forest')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_importance.png')
    
    return rf, importances

def visualize_feature_relationships(df):
    """Create visualizations of feature relationships."""
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

def build_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Build and evaluate multiple models."""
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'confusion_matrix.png')
    
    # SHAP analysis
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'shap_summary.png')
    
    return rf

def main():
    """Main function to run the analysis."""
    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_data()
    
    print("Preparing data for modeling...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    print("Analyzing feature importance...")
    rf, importances = analyze_feature_importance(X_train, y_train, feature_cols)
    
    print("Visualizing feature relationships...")
    visualize_feature_relationships(df)
    
    print("Building and evaluating models...")
    model = build_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    print("Analysis complete! Check the reports/figures directory for visualizations.")

if __name__ == "__main__":
    main() 