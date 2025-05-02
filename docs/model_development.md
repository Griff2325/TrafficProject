# Traffic Accident Severity Prediction Model Development

## Overview
This document outlines the development process of a machine learning model to predict traffic accident severity (levels 1-4) in the US using various temporal, geographical, weather, and road features. The project involved data preparation, feature engineering, model training (XGBoost), model evaluation, and feature importance analysis.

## Project Structure (Current)
```
TrafficProject/
├── data/
│   ├── raw/                  # Original data files (e.g., US_Accidents_March23.csv)
│   ├── processed/            # Cleaned and feature-engineered data (e.g., cleaned_accidents.csv, accidents_with_features.csv)
│   └── geospatial/           # Shapefiles for geographic analysis (e.g., tl_2020_us_uac20.shp)
├── models/                   # Saved model files (e.g., xgb_model.json)
├── reports/
│   └── figures/
│       └── eda/              # Generated visualizations from EDA, model eval, and feature importance
├── src/
│   ├── data/                 # Data processing scripts (e.g., clean_data.py)
│   ├── features/             # Feature engineering & importance scripts (e.g., analyze_features.py, feature_importance_analysis.py)
│   ├── models/               # Model development scripts (e.g., train_model.py)
│   └── visualization/        # Visualization scripts (e.g., eda.py)
├── docs/
│   └─── model_development.md
├── .venv/                    # Python virtual environment
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Development Process & Scripts

### 1. Data Cleaning (`src/data/clean_data.py`)
- Loads raw data.
- Handles missing values, duplicates, or initial inconsistencies.
- Saves cleaned data (e.g., `data/processed/cleaned_accidents.csv`).

### 2. Feature Engineering (`src/features/build_features.py`)

Feature engineering involves creating new variables from the existing data to improve model performance. Key steps include:

*   **Temporal Features:** Extracted hour, day of week, month, year, season, time of day categories (Morning, Afternoon, etc.), weekend flag, rush hour flag, and holiday flag.
*   **Duration:** Calculated accident duration in minutes.
*   **Weather Features:** Created categorical features for temperature, wind speed, humidity, visibility, and precipitation. Added a numerical `weather_severity` score based on `Weather_Condition`.
*   **Road Features:** Created flags for intersections, traffic controls, complex intersections, and a count of nearby road features (Amenity, Bump, Stop, etc.).
*   **Location Features:**
    *   Mapped states to broader US regions (Northeast, Midwest, South, West).
    *   **Urban/Suburban Classification:** Implemented a more accurate classification using `geopandas` and the [2020 TIGER/Line Urban Areas shapefile](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Urban+Areas) (`tl_2020_us_uac20.shp`). This involves performing a spatial join between accident coordinates and urban area polygons. If the shapefile is not found in `data/geospatial/`, the script falls back to a simpler, less accurate state-based estimation. This spatial join significantly increases the runtime of the feature engineering script.
*   **Missing Value Handling:** Imputed missing numerical weather features using the median value grouped by state and month, falling back to the global median if necessary. Added indicator columns for originally missing weather values.

The final feature set used for modeling contains 75 features.

### 3. Model Training (`src/models/train_model.py`)
- **Input:** Feature-engineered data (`accidents_with_features.csv`).
- **Data Preparation:**
    - Selects features used for modeling.
    - Encodes categorical variables (e.g., LabelEncoder).
    - Splits data into training/testing sets (stratified).
    - Scales numerical features (StandardScaler).
- **Model:** XGBoost Classifier (`xgb.train`).
    - Objective: `multi:softmax` (4 classes).
    - Handles class imbalance using `scale_pos_weight` (calculated from training data).
    - Uses early stopping to prevent overfitting.
- **Evaluation:**
    - Predicts on the test set.
    - Prints a classification report.
    - Generates and saves a confusion matrix (`reports/figures/eda/confusion_matrix.png`).
- **Output:** Trained model (`models/xgb_model.json`).

### 4. Feature Importance Analysis (`src/features/feature_importance_analysis.py`)
- **Input:** Feature-engineered data (`accidents_with_features.csv`) and trained model (`models/xgb_model.json`).
- **Analysis Methods:**
    - **SHAP:** Calculates SHAP values (using `shap.TreeExplainer`). Generates global and per-class importance bar plots. Attempts GPU acceleration.
    - **Permutation Importance:** Calculates importance using `sklearn.inspection.permutation_importance` (with a custom wrapper for the XGBoost booster). Generates a bar plot with error bars.
    - **Feature Interactions:** Analyzes SHAP interaction values for top features, plotting scatter plots for selected pairs (skips pre-defined excluded pairs).
- **Output:** Importance plots saved to `reports/figures/eda/`.

### 5. Exploratory Data Analysis (`src/visualization/eda.py`)
- **Input:** Cleaned data (`cleaned_accidents.csv`) and Feature-engineered data (`accidents_with_features.csv`).
- **Analyses Performed:**
    - Basic distributions (accidents by hour, day, month, state).
    - Geographical density heatmap (using sampling for performance).
    - Weather patterns, road feature counts.
    - Correlation analysis (heatmap and target correlation).
    - Detailed analysis of intersection features and combinations (counts vs. severity).
    - Urban vs. Suburban comparisons (counts by time of day/month, severity by time of day/month).
- **Output:** Various plots saved to `reports/figures/eda/`.

## Results Analysis

### Model Performance
```
Classification Report:
              precision    recall  f1-score   support

           1       0.66      0.01      0.03     13473
           2       0.80      0.99      0.89   1231396
           3       0.54      0.03      0.07    259868
           4       0.59      0.01      0.01     40942

    accuracy                           0.80   1545679
   macro avg       0.65      0.26      0.25   1545679
weighted avg       0.75      0.80      0.72   1545679
```
*   **Confusion Matrix:** Visualizes prediction accuracy across the 4 severity classes (saved in `reports/figures/eda/`).
*   **Training Performance Plot:** Shows Training vs. Validation Classification Error (`merror`) over boosting rounds (saved as `training_performance.png` in `reports/figures/eda/`).
*   **Key Observations:**
    *   The model generally achieves high accuracy (80%) due to the prevalence of the majority class (Severity 2).
    *   Performance (precision/recall) on minority classes (Severity 1, 3, 4) is typically much lower due to severe class imbalance (Recall: ~1% for Sev 1, ~3% for Sev 3, ~1% for Sev 4).
    *   Training and validation error rates track very closely, indicating minimal overfitting according to this metric.
    *   Final Training Error Rate (`merror`): `~0.20155`
    *   Final Validation Error Rate (`merror`): `~0.20181`

### Feature Importance
*   SHAP plots (global and per-class) identify the most influential features for the model's predictions overall and for each severity level.
*   Permutation importance provides an alternative measure based on performance degradation when features are shuffled.
*   Key drivers often include weather severity, time of day, specific road features (like Junction, Traffic_Signal), and location (region, urban/suburban).

## Challenges and Solutions

1.  **Class Imbalance:**
    *   **Problem:** Severity level 2 dominates the dataset, making it hard for the model to learn patterns for less frequent severity levels.
    *   **Solution:** Class weights (`scale_pos_weight`) were applied during XGBoost training to give more importance to minority classes. Stratified splitting was used.
    *   **Future:** Could explore SMOTE, ADASYN, or other advanced sampling techniques; potentially different model architectures or cost-sensitive learning.
2.  **Computational Resources / Runtime:**
    *   **Problem:** Certain analyses (KDE plots, SHAP, Permutation Importance) can be slow on large datasets.
    *   **Solution:** Implemented sampling for KDE plot in EDA. Utilized GPU acceleration where possible (XGBoost training, SHAP). Permutation importance uses a wrapper but might still be CPU-bound depending on the scikit-learn implementation.
    *   **Future:** Further optimization, exploring Dask or other parallel processing frameworks if needed.
3.  **Feature Interpretation:**
    *   **Problem:** Understanding *why* a feature is important requires careful analysis.
    *   **Solution:** Used SHAP for local and global explanations. EDA plots help visualize relationships (e.g., urban/suburban differences, intersection feature impacts).

## Technical Notes

*   **Dependencies:** See `requirements.txt`. Key libraries include pandas, scikit-learn, xgboost, shap, matplotlib, seaborn.
*   **GPU Acceleration:** Attempts to use GPU via XGBoost parameters (`'predictor': 'gpu_predictor'`, `'tree_method': 'gpu_hist'`) and SHAP if available.
*   **Configuration:** Feature interaction plots to exclude are defined in `src/features/feature_importance_analysis.py`.

## Next Steps

1. **Model Improvement**:
   - Implement more sophisticated handling of class imbalance
   - Conduct hyperparameter optimization
   - Experiment with different model architectures

2. **Feature Engineering**:
   - Develop more sophisticated feature interactions
   - Incorporate additional data sources
   - Create time-series features

## File Structure Changes
1. Split original `analyze_features.py` into:
   - `train_model.py` for model training
   - `analyze_features.py` for feature analysis
2. Created proper directory structure for:
   - Data storage
   - Model storage
   - Report generation
   - Documentation

## Key Code Changes
1. Added StandardScaler import to feature analysis script
2. Implemented proper model saving and loading
3. Created comprehensive visualization pipeline
4. Added proper error handling and logging 