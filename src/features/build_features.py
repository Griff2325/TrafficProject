"""
Feature Engineering for US Traffic Accidents dataset.

This script creates derived features from the cleaned dataset, including:
- Temporal features (time of day, season, holidays, etc.)
- Weather features (temperature categories, visibility impact, etc.)
- Road features (intersection types, traffic controls, etc.)
- Location features (urban/rural, region, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any
from datetime import datetime
import holidays
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cleaned_data(data_path: str) -> pd.DataFrame:
    """
    Load the cleaned dataset from the specified path.
    
    Args:
        data_path (str): Path to the cleaned data file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # First read without parsing dates to check the format
        df = pd.read_csv(data_path)
        
        # Convert datetime columns with mixed format
        datetime_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='mixed')
        
        logger.info(f"Successfully loaded cleaned data from {data_path}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading cleaned data: {str(e)}")
        raise

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from timestamps."""
    # Basic temporal features
    df['start_hour'] = df['Start_Time'].dt.hour
    df['start_day_of_week'] = df['Start_Time'].dt.dayofweek
    df['start_month'] = df['Start_Time'].dt.month
    df['start_year'] = df['Start_Time'].dt.year
    df['duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    # Time of day categories
    df['time_of_day'] = pd.cut(df['start_hour'], 
                              bins=[0, 6, 12, 18, 24],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    # Season
    df['season'] = pd.cut(df['start_month'],
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Fall'],
                         include_lowest=True)
    
    # Weekend flag
    df['is_weekend'] = df['start_day_of_week'].isin([5, 6]).astype(int)
    
    # Rush hour flag
    df['is_rush_hour'] = ((df['start_hour'].between(7, 9)) | 
                         (df['start_hour'].between(16, 18))).astype(int)
    
    # Holiday flag
    us_holidays = holidays.US()
    df['is_holiday'] = df['Start_Time'].dt.date.apply(lambda x: x in us_holidays).astype(int)
    
    logger.info("Extracted temporal features")
    return df

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create weather-related features."""
    # Temperature categories
    df['temperature_category'] = pd.cut(df['Temperature(F)'],
                                      bins=[-np.inf, 32, 50, 70, 85, np.inf],
                                      labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'])
    
    # Wind speed categories
    df['wind_speed_category'] = pd.cut(df['Wind_Speed(mph)'],
                                     bins=[-np.inf, 5, 10, 20, 30, np.inf],
                                     labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong'])
    
    # Humidity categories
    df['humidity_category'] = pd.cut(df['Humidity(%)'],
                                   bins=[-np.inf, 30, 50, 70, 90, np.inf],
                                   labels=['Very Dry', 'Dry', 'Moderate', 'Humid', 'Very Humid'])
    
    # Visibility categories
    df['visibility_category'] = pd.cut(df['Visibility(mi)'],
                                     bins=[-np.inf, 0.1, 1, 3, 10, np.inf],
                                     labels=['Zero', 'Very Poor', 'Poor', 'Moderate', 'Good'])
    
    # Precipitation categories
    df['precipitation_category'] = pd.cut(df['Precipitation(in)'],
                                        bins=[-np.inf, 0, 0.1, 0.5, 1, np.inf],
                                        labels=['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy'])
    
    # Weather severity index
    weather_severity = {
        'Clear': 1,
        'Partly Cloudy': 1,
        'Mostly Cloudy': 2,
        'Overcast': 2,
        'Light Rain': 3,
        'Rain': 3,
        'Heavy Rain': 4,
        'Light Snow': 4,
        'Snow': 5,
        'Heavy Snow': 5,
        'Fog': 4,
        'Thunderstorm': 5
    }
    df['weather_severity'] = df['Weather_Condition'].map(weather_severity).fillna(3)
    
    logger.info("Created weather features")
    return df

def create_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create road-related features."""
    # Road feature combinations
    df['is_intersection'] = (df['Junction'] | df['Crossing'] | df['Traffic_Signal']).astype(int)
    df['has_traffic_control'] = (df['Traffic_Signal'] | df['Stop'] | df['Give_Way']).astype(int)
    df['is_complex_intersection'] = ((df['Junction'] & df['Traffic_Signal']) | 
                                   (df['Crossing'] & df['Traffic_Signal'])).astype(int)
    
    # Road feature counts
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                    'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                    'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    df['road_feature_count'] = df[road_features].sum(axis=1)
    
    logger.info("Created road features")
    return df

def create_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create location-related features."""
    # Region mapping
    region_map = {
        'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
        'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast', 'RI': 'Northeast',
        'VT': 'Northeast',
        'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest',
        'MI': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
        'ND': 'Midwest', 'OH': 'Midwest', 'SD': 'Midwest', 'WI': 'Midwest',
        'AL': 'South', 'AR': 'South', 'DE': 'South', 'FL': 'South',
        'GA': 'South', 'KY': 'South', 'LA': 'South', 'MD': 'South',
        'MS': 'South', 'NC': 'South', 'OK': 'South', 'SC': 'South',
        'TN': 'South', 'TX': 'South', 'VA': 'South', 'WV': 'South',
        'AK': 'West', 'AZ': 'West', 'CA': 'West', 'CO': 'West',
        'HI': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
        'NM': 'West', 'OR': 'West', 'UT': 'West', 'WA': 'West', 'WY': 'West'
    }
    df['region'] = df['State'].map(region_map)
    
    # Urban/rural classification based on population density
    # This is a simplified version - in practice, you might want to use actual population data
    urban_states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    df['is_urban'] = df['State'].isin(urban_states).astype(int)
    
    logger.info("Created location features")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Weather-related missing values
    weather_cols = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 
                   'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                   'Precipitation(in)']
    
    # Fill missing weather values with median by state and month
    for col in weather_cols:
        df[col] = df.groupby(['State', 'start_month'])[col].transform(
            lambda x: x.fillna(x.median()))
    
    # Fill any remaining missing values with overall median
    for col in weather_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Create missing value indicators
    for col in weather_cols:
        df[f'{col}_missing'] = df[col].isnull().astype(int)
    
    logger.info("Handled missing values")
    return df

def main():
    """Main function to run feature engineering."""
    print("Starting feature engineering...")
    
    # Define paths
    data_dir = Path('data')
    cleaned_data_path = data_dir / 'processed' / 'cleaned_accidents.csv'
    features_data_path = data_dir / 'processed' / 'accidents_with_features.csv'
    
    try:
        print("Loading data...")
        df = load_cleaned_data(cleaned_data_path)
        
        print("Creating temporal features...")
        df = create_temporal_features(df)
        
        print("Creating weather features...")
        df = create_weather_features(df)
        
        print("Creating road features...")
        df = create_road_features(df)
        
        print("Creating location features...")
        df = create_location_features(df)
        
        print("Handling missing values...")
        df = handle_missing_values(df)
        
        print("Saving processed data...")
        df.to_csv(features_data_path, index=False)
        logger.info(f"Processed data with features saved to {features_data_path}")
        
        # Log feature summary
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info("New features created:")
        new_features = [
            # Temporal features
            'start_hour', 'start_day_of_week', 'start_month', 'start_year',
            'start_quarter', 'duration_minutes', 'time_of_day', 'season',
            'is_weekend', 'is_rush_hour', 'is_holiday',
            
            # Road features
            'is_intersection', 'has_traffic_control', 'is_complex_intersection',
            'road_feature_count',
            
            # Weather features
            'temperature_category', 'wind_speed_category', 'humidity_category',
            'visibility_category', 'weather_severity',
            
            # Location features
            'is_urban', 'region'
        ]
        for feature in new_features:
            if feature in df.columns:
                logger.info(f"- {feature}: {df[feature].dtype}")
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 