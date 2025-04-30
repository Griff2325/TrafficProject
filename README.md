# US Traffic Accident Analysis Project

This project analyzes US traffic accident data to identify patterns, correlations, and insights related to time, location, and seasonality of accidents. The goal is to build a system that can understand, analyze, and predict the probability of traffic accidents occurring given specific conditions.

## Project Structure

```
TrafficProject/
├── data/
│   ├── raw/                  # Original data files
│   ├── processed/            # Processed and cleaned data
│   └── interim/              # Intermediate data files
├── notebooks/                # Jupyter notebooks for analysis
├── src/
│   ├── data/                 # Data processing scripts
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Model development scripts
│   └── visualization/        # Visualization scripts
├── .venv/                    # Python virtual environment
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TrafficProject.git
cd TrafficProject
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

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

The dataset used in this project is the US_Accidents_March23.csv file. You can download it from:
- [Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- [Alternative Source](https://smoosavi.org/datasets/us_accidents)

Place the downloaded file in the `data/raw/` directory.

## Running the Project

### 1. Data Cleaning

Run the data cleaning script to process the raw data:

```bash
python src/data/clean_data.py
```

This will:
- Load the raw dataset
- Check for missing values and duplicates
- Clean the data
- Save the cleaned data to `data/processed/cleaned_accidents.csv`

### 2. Feature Engineering

Run the feature engineering script to create derived features:

```bash
python src/features/build_features.py
```

This will:
- Load the cleaned data
- Create temporal features (time of day, season, etc.)
- Create road features (crossings, junctions, etc.)
- Create weather features (temperature, wind, etc.)
- Create light condition features
- Create location features
- Handle missing values
- Save the processed data to `data/processed/accidents_with_features.csv`

## Data Processing Pipeline

1. **Data Cleaning** (`src/data/clean_data.py`):
   - Loads raw data
   - Handles missing values
   - Removes duplicates
   - Converts data types
   - Saves cleaned data

2. **Feature Engineering** (`src/features/build_features.py`):
   - Creates temporal features
   - Creates road features
   - Creates weather features
   - Creates light condition features
   - Creates location features
   - Handles missing values
   - Saves processed data

## Dependencies

The project requires the following Python packages (see `requirements.txt` for specific versions):

- **Data Processing**: pandas, numpy, scipy
- **Geospatial Processing**: geopy, geopandas, folium, shapely
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, shap
- **Time Series**: statsmodels, prophet

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 