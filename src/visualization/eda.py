"""
Exploratory Data Analysis for US Traffic Accidents dataset.

This script performs comprehensive EDA on the US traffic accidents dataset,
generating visualizations and saving them to the reports/figures directory.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_style('whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    os.makedirs('reports/figures', exist_ok=True)

def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv('data/processed/cleaned_accidents.csv')
    
    # Convert timestamp columns with a specific format that includes microseconds
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')
    
    # Create temporal features
    df['start_hour'] = df['Start_Time'].dt.hour
    df['start_day_of_week'] = df['Start_Time'].dt.dayofweek
    df['start_month'] = df['Start_Time'].dt.month
    df['start_year'] = df['Start_Time'].dt.year
    df['duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    return df

def analyze_temporal_patterns(df):
    """Analyze and visualize temporal patterns in accidents."""
    # Accidents by hour
    plt.figure()
    sns.countplot(data=df, x='start_hour', order=range(24))
    plt.title('Accidents by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.savefig('reports/figures/accidents_by_hour.png')
    plt.close()

    # Accidents by day of week
    plt.figure()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['Start_Time'].dt.day_name()
    sns.countplot(data=df, x='day_name', order=day_order)
    plt.title('Accidents by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/accidents_by_day.png')
    plt.close()

    # Accidents by month
    plt.figure()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    df['month_name'] = df['Start_Time'].dt.month_name()
    sns.countplot(data=df, x='month_name', order=month_order)
    plt.title('Accidents by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/accidents_by_month.png')
    plt.close()

    # Accident duration
    plt.figure()
    sns.histplot(data=df, x='duration_minutes', bins=50, kde=True)
    plt.title('Accident Duration Distribution')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Number of Accidents')
    plt.xlim(0, 1000)
    plt.savefig('reports/figures/accident_duration.png')
    plt.close()

def analyze_geographical_patterns(df):
    """Analyze and visualize geographical patterns in accidents."""
    # Accidents by state
    plt.figure()
    state_counts = df['State'].value_counts().head(20)
    sns.barplot(x=state_counts.index, y=state_counts.values)
    plt.title('Top 20 States by Number of Accidents')
    plt.xlabel('State')
    plt.ylabel('Number of Accidents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/accidents_by_state.png')
    plt.close()

    # Accident density heatmap
    plt.figure()
    sns.kdeplot(data=df, x='Start_Lng', y='Start_Lat', cmap='viridis', fill=True)
    plt.title('Accident Density Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('reports/figures/accident_density.png')
    plt.close()

def analyze_weather_patterns(df):
    """Analyze and visualize weather-related patterns."""
    # Temperature distribution
    plt.figure()
    sns.histplot(data=df, x='Temperature(F)', bins=50, kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (F)')
    plt.ylabel('Number of Accidents')
    plt.savefig('reports/figures/temperature_distribution.png')
    plt.close()

    # Visibility distribution
    plt.figure()
    sns.histplot(data=df, x='Visibility(mi)', bins=50, kde=True)
    plt.title('Visibility Distribution')
    plt.xlabel('Visibility (miles)')
    plt.ylabel('Number of Accidents')
    plt.savefig('reports/figures/visibility_distribution.png')
    plt.close()

    # Weather conditions
    plt.figure()
    weather_counts = df['Weather_Condition'].value_counts().head(20)
    sns.barplot(x=weather_counts.values, y=weather_counts.index)
    plt.title('Top 20 Weather Conditions')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Weather Condition')
    plt.tight_layout()
    plt.savefig('reports/figures/weather_conditions.png')
    plt.close()

def analyze_road_features(df):
    """Analyze and visualize road feature patterns."""
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                    'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                    'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    
    plt.figure()
    feature_counts = df[road_features].sum().sort_values(ascending=False)
    sns.barplot(x=feature_counts.values, y=feature_counts.index)
    plt.title('Road Features Present in Accidents')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Road Feature')
    plt.tight_layout()
    plt.savefig('reports/figures/road_features.png')
    plt.close()

def analyze_severity_patterns(df):
    """Analyze and visualize severity patterns."""
    # Severity distribution
    plt.figure()
    sns.countplot(data=df, x='Severity')
    plt.title('Accident Severity Distribution')
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Accidents')
    plt.savefig('reports/figures/severity_distribution.png')
    plt.close()

    # Severity by hour
    plt.figure()
    sns.boxplot(data=df, x='start_hour', y='Severity')
    plt.title('Severity by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Severity Level')
    plt.savefig('reports/figures/severity_by_hour.png')
    plt.close()

    # Severity by weather
    plt.figure()
    weather_severity = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=weather_severity.values, y=weather_severity.index)
    plt.title('Average Severity by Weather Condition (Top 20)')
    plt.xlabel('Average Severity')
    plt.ylabel('Weather Condition')
    plt.tight_layout()
    plt.savefig('reports/figures/severity_by_weather.png')
    plt.close()

def analyze_correlations(df):
    """Analyze and visualize correlations between features."""
    numerical_cols = ['Severity', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
                     'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
                     'Precipitation(in)', 'duration_minutes']
    
    plt.figure(figsize=(12, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()

def main():
    """Main function to run the EDA."""
    print("Starting EDA...")
    create_output_directories()
    
    print("Loading data...")
    df = load_data()
    
    print("Analyzing temporal patterns...")
    analyze_temporal_patterns(df)
    
    print("Analyzing geographical patterns...")
    analyze_geographical_patterns(df)
    
    print("Analyzing weather patterns...")
    analyze_weather_patterns(df)
    
    print("Analyzing road features...")
    analyze_road_features(df)
    
    print("Analyzing severity patterns...")
    analyze_severity_patterns(df)
    
    print("Analyzing correlations...")
    analyze_correlations(df)
    
    print("EDA complete. Visualizations saved to reports/figures/")

if __name__ == "__main__":
    main() 