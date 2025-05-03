"""
Comprehensive Feature Importance Analysis for Traffic Accident Severity.

This script implements multiple methods to analyze feature importance:
1. SHAP values (global and local)
2. Permutation importance
3. Feature correlation analysis
4. Feature interaction analysis
5. Feature value impact analysis
6. Weather severity impact analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette('viridis')

# Define paths
DATA_DIR = Path('data/processed')
REPORTS_DIR = Path('reports/figures')
MODELS_DIR = Path('models')

# Check for GPU availability
def gpu_available():
    """Check if GPU is available for XGBoost."""
    try:
        # First check if NVIDIA GPU is available using os.system
        gpu_check = os.system("nvidia-smi > /dev/null 2>&1")
        if gpu_check != 0:
            print("No NVIDIA GPU detected by nvidia-smi")
            return False
            
        # Check if XGBoost can use GPU
        print("Attempting to set up XGBoost with GPU...")
        # Set XGBoost GPU parameters before creating DMatrix
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        test_data = np.random.rand(10, 5)
        test_label = np.random.randint(0, 2, 10)
        dtrain = xgb.DMatrix(test_data, label=test_label)
        
        # Try to train a small model with GPU
        bst = xgb.train(params, dtrain, num_boost_round=1)
        
        print("GPU configuration successful for XGBoost!")
        return True
    except Exception as e:
        print(f"GPU available but XGBoost GPU setup failed: {str(e)}")
        print("Installing correct XGBoost GPU support might help...")
        return False

# Set the compute device
USE_GPU = gpu_available()
DEVICE = 'cuda:0' if USE_GPU else None
print(f"{'Using GPU' if USE_GPU else 'GPU not available, using CPU'} for calculations")

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
    y = df['Severity'] - 1  # Convert to 0-based indexing
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        X[col] = pd.factorize(X[col])[0]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, y, feature_cols

def analyze_shap_values(model, X, feature_cols, sample_size=1000):
    """Perform comprehensive SHAP analysis."""
    print("Performing SHAP analysis...")
    
    # Take a random sample for SHAP analysis
    sample_size = min(sample_size, len(X))
    sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Create DMatrix for GPU acceleration if available
    dmatrix = xgb.DMatrix(X_sample)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values with GPU if available
    if USE_GPU:
        print("Using GPU-accelerated SHAP analysis")
        # Force XGBoost to use GPU
        model.set_param({'predictor': 'gpu_predictor'})
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample if not USE_GPU else dmatrix)
    
    # Debug shapes
    print(f"X_sample shape: {X_sample.shape}")
    print(f"SHAP values type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"SHAP values length: {len(shap_values)}")
        print(f"First element shape: {shap_values[0].shape}")
    else:
        print(f"SHAP values shape: {shap_values.shape}")
    
    # Convert to numpy array if it's a list
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    # The SHAP values shape is (samples, features, classes) = (1000, 19, 4)
    
    # 1. Global feature importance
    plt.figure(figsize=(12, 8))
    # Calculate mean absolute SHAP values for each feature (average across samples and classes)
    # First get absolute values, then average across samples (axis=0), then across classes (axis=1)
    mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
    
    # Create a bar plot of feature importance
    sorted_idx = np.argsort(mean_abs_shap)
    plt.barh(range(len(feature_cols)), mean_abs_shap[sorted_idx])
    plt.yticks(range(len(feature_cols)), np.array(feature_cols)[sorted_idx])
    plt.title('Global Feature Importance (SHAP)')
    plt.xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'shap_global_importance.png')
    plt.close()
    
    # 2. Feature importance by class
    for i in range(4):  # For each severity class
        plt.figure(figsize=(12, 8))
        # Get SHAP values for this class, average across samples
        class_shap = np.abs(shap_values[:, :, i]).mean(axis=0)  # Shape: (19,)
        
        # Create a bar plot of feature importance for this class
        sorted_idx = np.argsort(class_shap)
        plt.barh(range(len(feature_cols)), class_shap[sorted_idx])
        plt.yticks(range(len(feature_cols)), np.array(feature_cols)[sorted_idx])
        plt.title(f'Feature Importance for Severity {i+1}')
        plt.xlabel('|SHAP value|')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f'shap_class_{i+1}_importance.png')
        plt.close()
    
    # 3. Feature interactions
    plt.figure(figsize=(15, 10))
    # Calculate interaction values
    interaction_values = np.zeros((len(feature_cols), len(feature_cols)))
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            # Calculate interaction strength across all samples and classes
            interaction = np.abs(shap_values[:, i, :] * shap_values[:, j, :]).mean()
            interaction_values[i, j] = interaction
            interaction_values[j, i] = interaction
    
    # Plot interaction matrix
    sns.heatmap(interaction_values, annot=False, 
                xticklabels=feature_cols, yticklabels=feature_cols,
                cmap='viridis')
    plt.title('Feature Interaction Strength')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_interactions.png')
    plt.close()
    
    return shap_values

def analyze_permutation_importance(model, X, y, feature_cols):
    """Analyze feature importance using GPU-optimized permutation importance."""
    print("Calculating permutation importance (GPU-optimized)...")
    
    # Create results directory if it doesn't exist
    results_dir = REPORTS_DIR / 'intermediate_results'
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if we have cached results to avoid recomputation
    cache_file = results_dir / 'permutation_importance.npz'
    if cache_file.exists():
        print(f"Loading cached permutation importance results from {cache_file}")
        cached_results = np.load(cache_file)
        importances = cached_results['importances']
        importances_mean = cached_results['importances_mean']
        importances_std = cached_results['importances_std']
        
        # Create result object
        class PermutationImportanceResult:
            def __init__(self, importances, importances_mean, importances_std):
                self.importances = importances
                self.importances_mean = importances_mean
                self.importances_std = importances_std
        
        result = PermutationImportanceResult(importances, importances_mean, importances_std)
    else:
        # Convert to numpy arrays for speed
        X_np = X.copy()
        y_np = np.array(y)
        
        # Create DMatrix for baseline score
        dmatrix_baseline = xgb.DMatrix(X_np)
        
        # Get baseline predictions
        if USE_GPU:
            model.set_param({'predictor': 'gpu_predictor'})
        baseline_preds = model.predict(dmatrix_baseline)
        
        # For multi-class, get class predictions
        if len(baseline_preds.shape) > 1:
            baseline_preds = np.argmax(baseline_preds, axis=1)
        
        # Calculate baseline accuracy
        baseline_score = np.mean(baseline_preds == y_np)
        print(f"Baseline accuracy: {baseline_score:.4f}")
        
        # Parameters for permutation
        n_repeats = 3
        random_state = np.random.RandomState(42)
        
        # Store results
        importances = np.zeros((n_repeats, len(feature_cols)))
        
        # Function to get accuracy score after permutation
        def get_permutation_score(X_permuted):
            dmatrix = xgb.DMatrix(X_permuted)
            preds = model.predict(dmatrix)
            if len(preds.shape) > 1:
                preds = np.argmax(preds, axis=1)
            return np.mean(preds == y_np)
        
        print("Starting permutation importance calculation...")
        print(f"Testing {len(feature_cols)} features with {n_repeats} repeats each...")
        
        # Perform permutation importance calculation
        start_time = time.time()
        for i, feature_idx in enumerate(range(len(feature_cols))):
            feature_name = feature_cols[feature_idx]
            print(f"Permuting feature {i+1}/{len(feature_cols)}: {feature_name}", end="", flush=True)
            
            # For each repeat
            for r in range(n_repeats):
                # Copy the data
                X_permuted = X_np.copy()
                
                # Permute the feature
                perm_idx = random_state.permutation(len(X_permuted))
                X_permuted[:, feature_idx] = X_permuted[perm_idx, feature_idx]
                
                # Calculate score after permutation
                permuted_score = get_permutation_score(X_permuted)
                
                # The importance is the drop in score
                importance = baseline_score - permuted_score
                importances[r, feature_idx] = importance
                
                print(".", end="", flush=True)
            
            # Show average importance after all repeats
            avg_importance = np.mean(importances[:, feature_idx])
            print(f" Done. Importance: {avg_importance:.4f}")
            
            # Save intermediate results after each feature (in case of crash)
            temp_importances_mean = np.mean(importances, axis=0)
            temp_importances_std = np.std(importances, axis=0)
            np.savez(
                results_dir / 'permutation_importance_temp.npz',
                importances=importances,
                importances_mean=temp_importances_mean,
                importances_std=temp_importances_std,
                feature_names=np.array(feature_cols)
            )
        
        # Calculate mean and std of importance scores
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        
        print(f"Permutation importance calculation complete in {time.time() - start_time:.2f}s")
        
        # Save final results
        np.savez(
            cache_file,
            importances=importances,
            importances_mean=importances_mean,
            importances_std=importances_std,
            feature_names=np.array(feature_cols)
        )
        
        # Create a named result object similar to sklearn's
        class PermutationImportanceResult:
            def __init__(self, importances, importances_mean, importances_std):
                self.importances = importances
                self.importances_mean = importances_mean
                self.importances_std = importances_std
        
        result = PermutationImportanceResult(importances, importances_mean, importances_std)
    
    # Plot results - fixing the boxplot issue
    try:
        plt.figure(figsize=(12, 8))
        
        # Sort by mean importance
        sorted_idx = result.importances_mean.argsort()
        sorted_features = np.array(feature_cols)[sorted_idx]
        
        # Ensure data for boxplot has the right shape
        boxplot_data = [result.importances[:, i] for i in sorted_idx]
        
        # Create boxplot with correct data format
        plt.boxplot(boxplot_data, vert=False, labels=sorted_features)
        
        plt.title("Permutation Importance (GPU-optimized)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'permutation_importance.png')
        plt.close()
        
        # Also create a simpler bar chart with error bars (more reliable)
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_features)), result.importances_mean[sorted_idx], 
                xerr=result.importances_std[sorted_idx], align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.title("Permutation Importance (GPU-optimized)")
        plt.xlabel("Mean Importance (with std. dev.)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'permutation_importance_bars.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting permutation importance: {str(e)}")
        print("Generating simplified plot instead...")
        
        # Create simplified bar chart as fallback
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(feature_cols)), result.importances_mean, align='center')
        plt.yticks(range(len(feature_cols)), feature_cols)
        plt.title("Permutation Importance (GPU-optimized) - Simplified")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'permutation_importance_simple.png')
        plt.close()
    
    return result

def analyze_feature_correlations(df, feature_cols):
    """Analyze correlations between features and target."""
    print("Analyzing feature correlations...")
    
    # Create a copy of the relevant columns
    corr_df = df[feature_cols + ['Severity']].copy()
    
    # Convert categorical columns to numeric
    for col in corr_df.columns:
        if corr_df[col].dtype == 'object' or corr_df[col].dtype.name == 'category':
            print(f"Converting categorical column '{col}' to numeric")
            corr_df[col] = pd.factorize(corr_df[col])[0]
    
    # Calculate correlations
    corr_matrix = corr_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_correlations.png')
    
    # Plot correlations with target
    plt.figure(figsize=(12, 8))
    target_corr = corr_matrix['Severity'].drop('Severity')
    target_corr.sort_values().plot(kind='barh')
    plt.title('Feature Correlations with Severity')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'target_correlations.png')
    plt.close()

def analyze_feature_interactions(model, X, feature_cols, top_n=5):
    """Analyze feature interactions using SHAP, skipping specific pairs."""
    print("Analyzing feature interactions...")
    
    # Define pairs to exclude from plotting
    excluded_pairs = {
        ('has_traffic_control', 'start_hour'), ('start_hour', 'has_traffic_control'),
        ('has_traffic_control', 'region'), ('region', 'has_traffic_control'),
        ('has_traffic_control', 'start_month'), ('start_month', 'has_traffic_control'),
        ('has_traffic_control', 'weather_severity'), ('weather_severity', 'has_traffic_control'),
        ('region', 'start_month'), ('start_month', 'region'),
        ('region', 'weather_severity'), ('weather_severity', 'region'),
        ('start_hour', 'region'), ('region', 'start_hour'),
        ('start_hour', 'start_month'), ('start_month', 'start_hour'),
        ('start_hour', 'weather_severity'), ('weather_severity', 'start_hour'),
        ('weather_severity', 'start_month'), ('start_month', 'weather_severity')
    }
    
    # Use consistent sampling for SHAP values
    sample_size = min(1000, len(X))
    sample_indices = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Get SHAP values specifically for this sample
    if USE_GPU:
        model.set_param({'predictor': 'gpu_predictor'})
        print("Using GPU acceleration for feature interaction analysis")
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Convert to numpy array if it's a list
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    print(f"SHAP values shape for interactions: {shap_values.shape}")
    
    # Calculate mean importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
    
    # Get top features
    top_indices = np.argsort(mean_abs_shap)[-top_n:]
    top_features = [feature_cols[i] for i in top_indices]
    
    print(f"Analyzing interactions for top {top_n} features: {top_features}")
    
    # For each top feature, plot its interaction with other top features
    for i, idx1 in enumerate(top_indices):
        feature1 = feature_cols[idx1]
        for j, idx2 in enumerate(top_indices[i+1:], i+1):
            feature2 = feature_cols[idx2]
            
            # Skip excluded pairs
            if (feature1, feature2) in excluded_pairs:
                print(f"Skipping excluded interaction plot: {feature1} and {feature2}")
                continue
                
            print(f"Analyzing interaction between {feature1} and {feature2}")
            
            try:
                # Create plot
                plt.figure(figsize=(10, 8))
                
                # Get feature columns for the plot
                if isinstance(X, pd.DataFrame):
                    feature_values = {
                        feature1: X.iloc[sample_indices][feature1].values,
                        feature2: X.iloc[sample_indices][feature2].values
                    }
                else:
                    feature_values = {
                        feature1: X_sample[:, idx1],
                        feature2: X_sample[:, idx2]
                    }
                
                # Plot feature interaction
                plt.scatter(
                    feature_values[feature1], 
                    feature_values[feature2],
                    c=np.abs(shap_values[:, idx1, :]).mean(axis=1) if shap_values.shape[0] == sample_size else np.random.rand(len(X_sample)),
                    cmap='viridis',
                    s=50,
                    alpha=0.7
                )
                plt.colorbar(label='|SHAP value|')
                
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                plt.title(f'Interaction between {feature1} and {feature2}')
                plt.tight_layout()
                plt.savefig(REPORTS_DIR / f'interaction_{feature1}_{feature2}.png')
                plt.close()
            except Exception as e:
                print(f"Error generating interaction plot for {feature1} and {feature2}: {str(e)}")
    
    print("Feature interaction analysis complete.")

def analyze_feature_value_impact(df, feature_cols):
    """Analyze the impact of different feature values on severity.
    (Currently, this function only prints status messages as plotting was moved to eda.py)
    """
    print("Analyzing feature value impact...")
    # value_impact_dir = REPORTS_DIR / 'feature_value_impact'
    # value_impact_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze categorical features like time_of_day, season, weather categories
    # categorical_features = [
    #     'time_of_day', 'season', 'temperature_category', 'wind_speed_category',
    #     'humidity_category', 'visibility_category', 'region'
    # ]
    
    # for feature in categorical_features:
    #     if feature in df.columns:
    #         # plt.figure(figsize=(12, 6))
    #         # sns.boxplot(data=df, x=feature, y='Severity', palette='viridis')
    #         # plt.title(f'Severity Distribution by {feature}')
    #         # plt.xticks(rotation=45, ha='right')
    #         # plt.tight_layout()
    #         # plt.savefig(value_impact_dir / f'{feature}_severity_boxplot.png')
    #         # plt.close()
    #         pass # Plotting removed
            
    # Analyze boolean features
    # boolean_features = [
    #     'is_weekend', 'is_rush_hour', 'is_holiday', 'is_intersection', 
    #     'has_traffic_control', 'is_complex_intersection', 'is_urban'
    # ]
    
    # for feature in boolean_features:
    #     if feature in df.columns:
    #         # plt.figure(figsize=(8, 5))
    #         # sns.barplot(data=df, x=feature, y='Severity', palette='viridis', ci=None) # ci=None to avoid error bars for binary
    #         # plt.title(f'Mean Severity by {feature}')
    #         # plt.xticks([0, 1], ['False', 'True'])
    #         # plt.tight_layout()
    #         # plt.savefig(value_impact_dir / f'{feature}_severity_barplot.png')
    #         # plt.close()
    #         pass # Plotting removed
            
    # Numerical features (use binned plots or scatter if appropriate)
    # Example: Road feature count
    # if 'road_feature_count' in df.columns:
    #     # plt.figure(figsize=(12, 6))
    #     # sns.boxplot(data=df, x='road_feature_count', y='Severity')
    #     # plt.title('Severity by Road Feature Count')
    #     # plt.tight_layout()
    #     # plt.savefig(value_impact_dir / 'road_feature_count_severity_boxplot.png')
    #     # plt.close()
    #     pass # Plotting removed
        
    print("Feature value impact analysis complete (Plotting moved to eda.py).")

def analyze_weather_severity_impact(df):
    """Create detailed visualizations for weather severity impact on accidents."""
    print("Creating detailed weather severity impact visualizations...")
    
    # Create directory for weather severity visualizations
    weather_dir = REPORTS_DIR / 'feature_value_impact' / 'weather_detailed'
    weather_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Weather severity by accident severity (stacked bar)
    plt.figure(figsize=(14, 8))
    severity_by_weather = pd.crosstab(df['weather_severity'], df['Severity'], normalize='index') * 100
    severity_by_weather.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Accident Severity Distribution by Weather Severity')
    plt.xlabel('Weather Severity')
    plt.ylabel('Percentage (%)')
    plt.legend(title='Accident Severity')
    plt.tight_layout()
    plt.savefig(weather_dir / 'weather_severity_stacked.png')
    plt.close()
    
    # 2. Mean accident severity by weather severity (bar)
    plt.figure(figsize=(14, 8))
    mean_severity = df.groupby('weather_severity')['Severity'].mean().sort_values()
    counts = df.groupby('weather_severity').size()
    
    ax = mean_severity.plot(kind='bar', color='teal')
    plt.title('Mean Accident Severity by Weather Severity')
    plt.xlabel('Weather Severity')
    plt.ylabel('Mean Severity')
    
    # Add count annotations
    for i, (val, count) in enumerate(zip(mean_severity, counts[mean_severity.index])):
        ax.text(i, val + 0.05, f'n={count}', ha='center')
    
    plt.tight_layout()
    plt.savefig(weather_dir / 'mean_severity_by_weather.png')
    plt.close()
    
    # 3. Weather severity heatmap with time of day
    if 'time_of_day' in df.columns:
        plt.figure(figsize=(14, 8))
        pivot = df.pivot_table(values='Severity', index='weather_severity', 
                              columns='time_of_day', aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Mean Severity by Weather Severity and Time of Day')
        plt.tight_layout()
        plt.savefig(weather_dir / 'weather_time_heatmap.png')
        plt.close()
    
    # 4. Weather severity heatmap with day of week
    plt.figure(figsize=(14, 8))
    pivot = df.pivot_table(values='Severity', index='weather_severity', 
                          columns='start_day_of_week', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Mean Severity by Weather Severity and Day of Week')
    plt.tight_layout()
    plt.savefig(weather_dir / 'weather_day_heatmap.png')
    plt.close()
    
    # 5. Weather conditions breakdown if available
    if 'Weather_Condition' in df.columns:
        # Get top 10 weather conditions by frequency
        top_weather = df['Weather_Condition'].value_counts().head(10).index
        
        # Create a bar chart of mean severity by weather condition
        plt.figure(figsize=(16, 8))
        weather_severity = df[df['Weather_Condition'].isin(top_weather)].groupby('Weather_Condition')['Severity'].mean().sort_values()
        weather_severity.plot(kind='bar', color='purple')
        plt.title('Mean Severity by Weather Condition (Top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(weather_dir / 'weather_condition_severity.png')
        plt.close()
    
    print("Weather severity impact visualizations complete!")

def main():
    """Main function to run all feature importance analyses."""
    # Create reports directory if it doesn't exist
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure XGBoost for GPU if available
    if USE_GPU:
        print("Configuring XGBoost to use GPU acceleration with CUDA")
        # These environment variables can help with CUDA initialization
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # This sometimes helps with XGBoost GPU errors
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    else:
        print("GPU not available, using CPU for calculations")
    
    # Load data and model
    df, model = load_data_and_model()
    
    # If GPU is available, configure the model to use it
    if USE_GPU:
        print("Setting XGBoost model parameters for GPU prediction")
        model.set_param({'predictor': 'gpu_predictor'})
    
    # Prepare data
    X_scaled, X, y, feature_cols = prepare_data(df)
    
    # Run all analyses
    print("\nRunning comprehensive feature importance analysis...")
    
    # Add this line to analyze feature value impact first - it's faster and gives more direct insights
    analyze_feature_value_impact(df, feature_cols)
    
    # Add detailed weather severity analysis
    analyze_weather_severity_impact(df)
    
    # SHAP analysis
    shap_values = analyze_shap_values(model, X_scaled, feature_cols, sample_size=1000)
    
    # Permutation importance
    perm_importance = analyze_permutation_importance(model, X_scaled, y, feature_cols)
    
    # Feature correlations
    analyze_feature_correlations(df, feature_cols)
    
    try:
        # Feature interactions
        analyze_feature_interactions(model, X_scaled, feature_cols, top_n=5)
    except Exception as e:
        print(f"Error in feature interactions analysis: {str(e)}")
    
    print("\nFeature importance analysis complete! Check the reports/figures directory for visualizations.")

if __name__ == "__main__":
    main() 