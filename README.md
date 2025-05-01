# US Traffic Accident Analysis Project

This project analyzes US traffic accident data to identify patterns, correlations, and insights related to the **severity** of accidents. The primary goal is to build and interpret a machine learning model that predicts accident severity (levels 1-4) based on various temporal, geographical, weather, and road conditions.

## Project Structure

```
TrafficProject/
├── data/
│   ├── raw/                  # Original data files (e.g., US_Accidents_March23.csv)
│   ├── processed/            # Cleaned and feature-engineered data (e.g., cleaned_accidents.csv, accidents_with_features.csv)
│   └── interim/              # Intermediate data files (optional)
├── models/                   # Saved model files (e.g., xgb_model.json)
├── reports/
│   └── figures/
│       └── eda/              # Generated visualizations from EDA, model eval, and feature importance
├── notebooks/                # Jupyter notebooks (optional, for exploratory analysis)
├── src/
│   ├── data/                 # Data processing scripts (e.g., clean_data.py)
│   ├── features/             # Feature engineering & importance scripts (e.g., analyze_features.py, feature_importance_analysis.py)
│   ├── models/               # Model development scripts (e.g., train_model.py)
│   └── visualization/        # Visualization scripts (e.g., eda.py)
├── docs/                     # Project documentation
│   ├── model_development.md
│   ├── project_completion.md
│   └── poster_guidelines.md
├── .venv/                    # Python virtual environment
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url> TrafficProject
cd TrafficProject
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The dataset used in this project is typically `US_Accidents_March23.csv`. Download it from a source like:
- [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- [Alternative Source](https://smoosavi.org/datasets/us_accidents)

Place the downloaded file in the `data/raw/` directory.

## Running the Project

The recommended execution sequence is:

### 1. Data Cleaning (if applicable)

If you have a separate cleaning script (`src/data/clean_data.py`):

```bash
python3 src/data/clean_data.py
```
*   **Input:** `data/raw/US_Accidents_March23.csv`
*   **Output:** `data/processed/cleaned_accidents.csv` (or similar)

### 2. Feature Engineering

Run the script to create engineered features:

```bash
# Use the appropriate script name if different
python3 src/features/analyze_features.py
```
*   **Input:** `data/processed/cleaned_accidents.csv` (or similar)
*   **Output:** Creates `data/processed/accidents_with_features.csv`

### 3. Model Training and Evaluation

Train the severity prediction model:

```bash
python3 src/models/train_model.py
```
*   **Input:** `data/processed/accidents_with_features.csv`
*   **Output:** Saves the trained model to `models/xgb_model.json` and the confusion matrix to `reports/figures/eda/confusion_matrix.png`.

### 4. Feature Importance Analysis

Analyze the importance of features using the trained model:

```bash
python3 src/features/feature_importance_analysis.py
```
*   **Input:** `data/processed/accidents_with_features.csv` and `models/xgb_model.json`.
*   **Output:** Generates SHAP and permutation importance plots in `reports/figures/eda/`. Skips specific interaction plots as configured.

### 5. Exploratory Data Analysis (EDA)

Run the consolidated EDA script:

```bash
python3 src/visualization/eda.py
```
*   **Input:** `data/processed/cleaned_accidents.csv` and `data/processed/accidents_with_features.csv`.
*   **Output:** Generates various EDA plots (temporal, geographical, correlations, road features, urban/suburban comparisons) in `reports/figures/eda/`. Uses sampling for the density heatmap.

## Key Analyses Performed

*   **Exploratory Data Analysis (EDA):** Analysis of accident distributions over time (hour, day, month), geography (state, density heatmap), weather patterns, road features, and correlations. Includes specific analysis of intersection features and combinations.
*   **Urban vs. Suburban Analysis:** Comparison of accident counts and severity patterns between urban and suburban areas based on time of day and month.
*   **Model Training:** Training an XGBoost classifier to predict accident severity (4 levels), including handling class imbalance.
*   **Model Evaluation:** Assessing model performance using a classification report and confusion matrix.
*   **Feature Importance:** Evaluating feature importance using SHAP values (global and per-class) and Permutation Importance. Specific interaction plots are excluded based on configuration.

## Dependencies

The project requires the following Python packages (see `requirements.txt`):

*   **Data Processing**: pandas, numpy
*   **Machine Learning**: scikit-learn, xgboost, shap
*   **Visualization**: matplotlib, seaborn
*   **Utilities**: pathlib

## Troubleshooting

*   **Long EDA Runtime:** The geographical density heatmap uses sampling (`sample_size=100000` in `eda.py`) to improve performance. Adjust this value if needed.
*   **Feature Importance Errors:** Ensure the model file (`models/xgb_model.json`) and the feature data (`data/processed/accidents_with_features.csv`) exist before running `feature_importance_analysis.py`.
*   **GPU Usage:** XGBoost and SHAP analysis attempt to use the GPU if available and properly configured. If issues arise, check CUDA/driver compatibility or rely on the automatic CPU fallback.

## Project Documentation

For more details on the project methodology and findings, please refer to:
- `docs/model_development.md` - Detailed documentation of the model development process.
- `docs/project_completion.md` - Summary of project achievements, findings, and challenges.
- `docs/poster_guidelines.md` - Suggestions for creating a project presentation poster.

## Contributing

Standard contribution guidelines apply (fork, branch, commit, PR).

## License

This project is licensed under the MIT License - see the LICENSE file for details (if available). 