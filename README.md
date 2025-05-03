# Predicting US Traffic Accident Severity using XGBoost and Feature Analysis

This project analyzes US traffic accident data to identify patterns, correlations, and insights related to the **severity** of accidents. The primary goal is to build and interpret a machine learning model that predicts accident severity (levels 1-4) based on various temporal, geographical, weather, and road conditions. Additionally, Exploratory Data Analysis will be used to identify times of the year, week, and day that accidents are most prevelent as well as most severe.

Note that this project was initially run through Jetstream2 on the flavor g3.large; running all the scripts in this environment will take around 30 minutes. Results may be much slower if run on a local machine.

Additionally, data files are large, with the raw data being ~ 3.0GB, processed data files being around another 6.5GB, and US Census Shape files being around 120MB. Ensure adequate storage is aviailable on your system.

## Project Structure

```
TrafficProject/
├── .venv/                    # Python virtual environment (ignored by git)
├── data/
│   ├── geospatial/           # Shapefiles for geographic analysis (tl_2020_us_uac20.shp with supporting files)
│   ├── processed/            # Cleaned and feature-engineered data (cleaned_accidents.csv, accidents_with_features.csv)
│   └── raw/                  # Original data files (US_Accidents_March23.csv)
├── docs/
│   ├── dataset_description.md # Description of the dataset columns.
│   ├── eda_results.md        # Summary of EDA findings (source markdown - ignored by git).
│   ├── eda_results.pdf       # Generated PDF report of EDA findings.
│   └── model_development.md  # Detailed documentation of the model development process.
├── models/                   # Saved model files (xgb_model.json - ignored by git)
├── reports/
│   └── figures/
│       └── eda/              # Generated visualizations (ignored by git)
├── src/
│   ├── data/                 # Data processing scripts (e.g., clean_data.py)
│   ├── features/             # Feature engineering & importance scripts (e.g., build_features.py, feature_importance_analysis.py)
│   ├── models/               # Model development scripts (e.g., train_model.py)
│   └── visualization/        # Visualization scripts (e.g., eda.py)
├── .gitignore                # Specifies intentionally untracked files
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone Griff2325/TrafficProject
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

### 5. Geospatial Data (Required for Accurate Urban/Suburban Classification)

This project uses geographic boundaries to classify accidents as occurring in urban or suburban areas more accurately. You need to download the necessary shapefile from the US Census Bureau:

1.  **Download the Shapefile:** Go to the [2020 TIGER/Line® Shapefiles: Urban Areas](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Urban+Areas) page.
2.  Download the **2020 Urban Areas National File** shapefile.
3.  **Unzip the File:** Extract the contents of the downloaded zip file.
4.  **Place the Files:** Move all of the extracted files (including `.shp`, `.shx`, `.dbf`, `.prj`, etc.) into the `data/geospatial/` directory within this project.
The feature engineering script expects the main shapefile to be located at `data/geospatial/tl_2020_us_uac20.shp`.

If the shapefile is not found, the feature engineering script will fall back to a less accurate, state-based estimation for the urban/suburban classification.

### 6. Environment Variables

## Running the Project

The recommended execution sequence is:

### 1. Data Cleaning

```bash
python3 src/data/clean_data.py
```
*   **Input:** `data/raw/US_Accidents_March23.csv`
*   **Output:** `data/processed/cleaned_accidents.csv`

### 2. Feature Engineering

Run the script to create engineered features:

```bash
python3 src/features/build_features.py
```
*   **Input:** `data/processed/cleaned_accidents.csv`
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
*   **Output:** Generates SHAP and permutation importance plots in `reports/figures/eda/`. Skips unneccesary interaction plots as configured.

### 5. Exploratory Data Analysis (EDA)

Run the EDA script:

```bash
python3 src/visualization/eda.py
```
*   **Input:** `data/processed/cleaned_accidents.csv` and `data/processed/accidents_with_features.csv`.
*   **Output:** Generates various EDA plots (temporal, geographical, correlations, road features, urban/suburban comparisons) in `reports/figures/eda/`. Uses sampling for the density heatmap due to compute restraints.

## Key Analyses Performed

*   **Exploratory Data Analysis (EDA):** Analysis of accident distributions over time (hour, day, month), geography (state, density heatmap), weather patterns, road features, and correlations. Includes specific analysis of intersection features and combinations.
*   **Urban vs. Suburban Analysis:** Comparison of accident counts and severity patterns between urban and suburban areas based on time of day, week, and month.
*   **Model Training:** Training an XGBoost classifier to predict accident severity (4 levels), including an attempt to handle class imbalance.
*   **Model Evaluation:** Assessing model performance using a classification report and confusion matrix.
*   **Feature Importance:** Evaluating feature importance using SHAP values (global and per-class) and Permutation Importance. Unneccesary interaction plots are excluded based on configuration.

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
- `docs/dataset_description.md` - Detailed documentaion of dataset columns and features added
- `docs/eda_results.md` - Summarizes key findings from EDA

## Contributing

Standard contribution guidelines apply (fork, branch, commit, PR).

## License

This project is licensed under the MIT License

## AI Use Disclaimer

Cursor was used for general assistance throughout the development process and documentation.

## Acknowledgments

- Jetsteam2 through Indiana University for the computing resources provided
- The University of Tennessee Knoxville
- The teaching team for COSC426
