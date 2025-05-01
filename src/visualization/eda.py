"""
Exploratory Data Analysis for US Traffic Accidents dataset.

This script performs targeted EDA on the US traffic accidents dataset,
generating only the specified visualizations and saving them to the reports/figures/eda directory.
Also includes urban vs. suburban analysis to compare accident patterns between these area types.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
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

# Define paths
REPORTS_DIR = Path('reports/figures/eda')

def create_output_directories():
    """Create necessary output directories if they don't exist."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {REPORTS_DIR}")

def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv('data/processed/cleaned_accidents.csv')
    
    # Convert timestamp columns
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='ISO8601')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='ISO8601')
    
    # Create temporal features
    df['start_hour'] = df['Start_Time'].dt.hour
    df['start_day_of_week'] = df['Start_Time'].dt.dayofweek
    df['start_month'] = df['Start_Time'].dt.month
    df['start_year'] = df['Start_Time'].dt.year
    df['duration_minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    return df

def load_data_with_features():
    """Load the processed accident data with features for urban-suburban analysis."""
    print("Loading processed data with features...")
    
    # Load data
    df = pd.read_csv('data/processed/accidents_with_features.csv')
    
    # Ensure we have the required columns
    required_columns = ['is_urban', 'start_hour', 'start_month', 'Severity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataset")
    
    # Convert is_urban to string labels for better visualization
    df['Area Type'] = df['is_urban'].map({1: 'Urban', 0: 'Suburban'})
    
    return df

def analyze_temporal_patterns(df):
    """Analyze and visualize temporal patterns in accidents."""
    # Accidents by hour
    plt.figure()
    sns.countplot(data=df, x='start_hour', order=range(24))
    plt.title('Accidents by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.savefig(REPORTS_DIR / 'accidents_by_hour.png')
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
    plt.savefig(REPORTS_DIR / 'accidents_by_day.png')
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
    plt.savefig(REPORTS_DIR / 'accidents_by_month.png')
    plt.close()

    # Accident duration
    plt.figure()
    sns.histplot(data=df, x='duration_minutes', bins=50, kde=True)
    plt.title('Accident Duration Distribution')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Number of Accidents')
    plt.xlim(0, 1000)
    plt.savefig(REPORTS_DIR / 'accident_duration.png')
    plt.close()

def analyze_geographical_patterns(df, sample_size=100000):
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
    plt.savefig(REPORTS_DIR / 'accidents_by_state.png')
    plt.close()

    # Accident density heatmap (using a sample for performance)
    plt.figure()
    print(f"Generating density heatmap using a sample of {sample_size} points...")
    
    # Take a random sample if the dataset is larger than the sample size
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    sns.kdeplot(data=df_sample, x='Start_Lng', y='Start_Lat', cmap='viridis', fill=True)
    plt.title(f'Accident Density Heatmap (Sample Size: {sample_size})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(REPORTS_DIR / 'accident_density.png')
    plt.close()
    print("Density heatmap generated.")

def analyze_weather_patterns(df):
    """Analyze and visualize weather-related patterns."""
    # Temperature distribution
    plt.figure()
    sns.histplot(data=df, x='Temperature(F)', bins=50, kde=True)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (F)')
    plt.ylabel('Number of Accidents')
    plt.savefig(REPORTS_DIR / 'temperature_distribution.png')
    plt.close()

    # Visibility distribution
    plt.figure()
    sns.histplot(data=df, x='Visibility(mi)', bins=50, kde=True)
    plt.title('Visibility Distribution')
    plt.xlabel('Visibility (miles)')
    plt.ylabel('Number of Accidents')
    plt.savefig(REPORTS_DIR / 'visibility_distribution.png')
    plt.close()

    # Weather conditions
    plt.figure()
    weather_counts = df['Weather_Condition'].value_counts().head(20)
    sns.barplot(x=weather_counts.values, y=weather_counts.index)
    plt.title('Top 20 Weather Conditions')
    plt.xlabel('Number of Accidents')
    plt.ylabel('Weather Condition')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'weather_conditions.png')
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
    plt.savefig(REPORTS_DIR / 'road_features.png')
    plt.close()
    
    # Add detailed analysis of intersection features
    plt.figure(figsize=(14, 10))
    intersection_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Roundabout', 'Give_Way']
    
    # Create a dataframe to store the severity breakdown for each feature
    severity_by_feature = pd.DataFrame()
    
    for feature in intersection_features:
        # Calculate average severity and count for accidents with and without this feature
        with_feature = df[df[feature] == True]
        without_feature = df[df[feature] == False]
        
        # Add counts to dataframe
        severity_by_feature.loc[f"{feature} (Yes)", "Count"] = len(with_feature)
        severity_by_feature.loc[f"{feature} (No)", "Count"] = len(without_feature)
        
        # Add average severity to dataframe
        severity_by_feature.loc[f"{feature} (Yes)", "Avg Severity"] = with_feature['Severity'].mean()
        severity_by_feature.loc[f"{feature} (No)", "Avg Severity"] = without_feature['Severity'].mean()
    
    # Sort by count
    severity_by_feature = severity_by_feature.sort_values(by="Count", ascending=False)
    
    # Create a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Bar plot for counts
    bars = sns.barplot(data=severity_by_feature.reset_index(), x="index", y="Count", 
              palette=["#1f77b4", "#aec7e8"]*len(intersection_features), ax=ax1)
    ax1.set_xlabel('Intersection Feature')
    ax1.set_ylabel('Number of Accidents')
    ax1.set_xticklabels(bars.get_xticklabels(), rotation=45, ha='right')
    
    # Secondary axis for severity
    ax2 = ax1.twinx()
    ax2.plot(severity_by_feature.reset_index().index, severity_by_feature["Avg Severity"], 
             'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Average Severity', color='r')
    ax2.tick_params(axis='y', colors='r')
    
    plt.title('Accident Counts and Severity by Intersection Features')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'intersection_features_analysis.png')
    plt.close()

def analyze_intersection_combinations(df):
    """Analyze how combinations of intersection features affect accident counts."""
    print("Analyzing intersection feature combinations...")
    
    # Select key intersection features
    intersection_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Stop']
    
    # Create a new column with the combination of features
    df_copy = df.copy()
    df_copy['feature_combination'] = ''
    
    for feature in intersection_features:
        # Add the feature name to the combination string if the feature exists
        df_copy.loc[df_copy[feature] == True, 'feature_combination'] += feature + ' + '
    
    # Remove trailing ' + ' from combination strings
    df_copy['feature_combination'] = df_copy['feature_combination'].str.rstrip(' + ')
    
    # Replace empty strings with 'No Features'
    df_copy.loc[df_copy['feature_combination'] == '', 'feature_combination'] = 'No Features'
    
    # Get top 10 most common feature combinations
    top_combinations = df_copy['feature_combination'].value_counts().head(10)
    
    # Calculate average severity for each combination
    severity_by_combination = df_copy.groupby('feature_combination')['Severity'].mean()
    
    # Create plot
    plt.figure(figsize=(14, 10))
    
    # Create a dual-axis plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Bar plot for counts
    bars = sns.barplot(x=top_combinations.index, y=top_combinations.values, palette='Blues_d', ax=ax1)
    ax1.set_xlabel('Intersection Feature Combination')
    ax1.set_ylabel('Number of Accidents')
    ax1.set_xticklabels(bars.get_xticklabels(), rotation=45, ha='right')
    
    # Secondary axis for severity
    ax2 = ax1.twinx()
    severity_values = [severity_by_combination.get(combo, 0) for combo in top_combinations.index]
    ax2.plot(range(len(top_combinations)), severity_values, 'ro-', linewidth=2, markersize=6)
    ax2.set_ylabel('Average Severity', color='r')
    ax2.tick_params(axis='y', colors='r')
    
    plt.title('Accident Counts and Severity by Intersection Feature Combinations')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'intersection_combinations_analysis.png')
    plt.close()

def analyze_severity_patterns(df):
    """Analyze and visualize severity patterns."""
    # Severity distribution
    plt.figure()
    sns.countplot(data=df, x='Severity')
    plt.title('Accident Severity Distribution')
    plt.xlabel('Severity Level')
    plt.ylabel('Number of Accidents')
    plt.savefig(REPORTS_DIR / 'severity_distribution.png')
    plt.close()

    # Severity by hour
    plt.figure()
    sns.boxplot(data=df, x='start_hour', y='Severity')
    plt.title('Severity by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Severity Level')
    plt.savefig(REPORTS_DIR / 'severity_by_hour.png')
    plt.close()

    # Severity by weather
    plt.figure()
    weather_severity = df.groupby('Weather_Condition')['Severity'].mean().sort_values(ascending=False).head(20)
    sns.barplot(x=weather_severity.values, y=weather_severity.index)
    plt.title('Average Severity by Weather Condition (Top 20)')
    plt.xlabel('Average Severity')
    plt.ylabel('Weather Condition')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'severity_by_weather.png')
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
    plt.savefig(REPORTS_DIR / 'feature_correlations.png')
    plt.close()
    
    # Plot correlations with target (Severity)
    plt.figure(figsize=(12, 8))
    target_corr = corr_matrix['Severity'].drop('Severity')
    target_corr.sort_values().plot(kind='barh')
    plt.title('Feature Correlations with Severity')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'target_correlations.png')
    plt.close()

# Urban vs Suburban Analysis Functions

def analyze_urban_suburban_distributions(df):
    """Analyze general distributions of urban vs. suburban accidents."""
    print("Analyzing urban vs. suburban distributions...")
    
    # Distribution of accidents by area type
    plt.figure(figsize=(10, 6))
    area_counts = df['Area Type'].value_counts()
    area_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
    
    # Add count labels on bars
    for i, count in enumerate(area_counts):
        plt.text(i, count + 0.1, f"{count:,}", ha='center')
    
    plt.title('Distribution of Accidents by Area Type', fontsize=16)
    plt.ylabel('Number of Accidents', fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_area_type_distribution.png')
    plt.close()

def analyze_timing_by_month(df):
    """Analyze accident timing by month for urban vs. suburban areas using counts."""
    print("Analyzing accident timing by month for urban vs. suburban areas (using counts)...")
    
    # Count accidents by month and area type
    monthly_counts = df.groupby(['start_month', 'Area Type']).size().reset_index(name='count')
    
    # Create line plot for timing by month (using count)
    plt.figure(figsize=(14, 8))
    
    # Create line plot
    sns.lineplot(
        data=monthly_counts, 
        x='start_month', 
        y='count',  # Changed from 'percentage' to 'count'
        hue='Area Type',
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    
    # Add labels and title
    plt.title('Accident Counts by Month: Urban vs. Suburban', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)  # Changed label
    
    # Set x-axis ticks to represent months
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), months, rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_timing_by_month.png')
    plt.close()
    
    # Create count difference plot
    plt.figure(figsize=(14, 8))
    
    # Reshape data to calculate differences in counts
    pivot_data = monthly_counts.pivot(index='start_month', columns='Area Type', values='count').fillna(0)
    pivot_data['Difference'] = pivot_data['Urban'] - pivot_data['Suburban']
    
    # Plot difference
    plt.bar(range(1, 13), pivot_data['Difference'], color=['r' if x < 0 else 'g' for x in pivot_data['Difference']])
    
    # Add labels and title
    plt.title('Difference in Accident Counts (Urban - Suburban) by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Difference in Number of Accidents', fontsize=14)  # Changed label
    
    # Set x-axis ticks
    plt.xticks(range(1, 13), months, rotation=45)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_timing_by_month_difference.png')
    plt.close()

def analyze_timing_by_hour(df):
    """Analyze accident timing by hour for urban vs. suburban areas using counts."""
    print("Analyzing accident timing by hour for urban vs. suburban areas (using counts)...")
    
    # Count accidents by hour and area type
    hourly_counts = df.groupby(['start_hour', 'Area Type']).size().reset_index(name='count')
    
    # Create line plot for timing by hour (using count)
    plt.figure(figsize=(14, 8))
    
    # Create line plot
    sns.lineplot(
        data=hourly_counts, 
        x='start_hour', 
        y='count',  # Changed from 'percentage' to 'count'
        hue='Area Type',
        markers=True, 
        dashes=False,
        linewidth=2.5
    )
    
    # Add labels and title
    plt.title('Accident Counts by Hour: Urban vs. Suburban', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Number of Accidents', fontsize=14)  # Changed label
    
    # Set x-axis ticks for hours
    plt.xticks(range(0, 24))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_timing_by_hour.png')
    plt.close()
    
    # Create count difference plot
    plt.figure(figsize=(14, 8))
    
    # Reshape data to calculate differences in counts
    pivot_data = hourly_counts.pivot(index='start_hour', columns='Area Type', values='count').fillna(0)
    pivot_data['Difference'] = pivot_data['Urban'] - pivot_data['Suburban']
    
    # Plot difference
    plt.bar(range(0, 24), pivot_data['Difference'], color=['r' if x < 0 else 'g' for x in pivot_data['Difference']])
    
    # Add labels and title
    plt.title('Difference in Accident Counts (Urban - Suburban) by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Difference in Number of Accidents', fontsize=14)  # Changed label
    
    # Set x-axis ticks
    plt.xticks(range(0, 24))
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_timing_by_hour_difference.png')
    plt.close()

def analyze_severity_by_month(df):
    """Analyze accident severity by month for urban vs. suburban areas."""
    print("Analyzing accident severity by month for urban vs. suburban areas...")
    
    # Create a pivot table of mean severity by month and area type
    severity_by_month = df.pivot_table(
        values='Severity',
        index='start_month',
        columns='Area Type',
        aggfunc='mean'
    )
    
    # Create line plot for severity by month
    plt.figure(figsize=(14, 8))
    
    # Plot mean severity for each area type
    severity_by_month.plot(
        marker='o', 
        linewidth=2.5,
        ax=plt.gca()
    )
    
    # Add labels and title
    plt.title('Mean Accident Severity by Month: Urban vs. Suburban', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Mean Severity', fontsize=14)
    
    # Set x-axis ticks to represent months
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), months, rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_severity_by_month.png')
    plt.close()
    
    # Create heatmap showing severity by month and area type
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        severity_by_month.T,
        annot=True,
        cmap='YlOrRd',
        fmt='.2f'
    )
    plt.title('Mean Accident Severity by Month: Urban vs. Suburban', fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_severity_by_month_heatmap.png')
    plt.close()

def analyze_severity_by_hour(df):
    """Analyze accident severity by hour for urban vs. suburban areas."""
    print("Analyzing accident severity by hour for urban vs. suburban areas...")
    
    # Create a pivot table of mean severity by hour and area type
    severity_by_hour = df.pivot_table(
        values='Severity',
        index='start_hour',
        columns='Area Type',
        aggfunc='mean'
    )
    
    # Create line plot for severity by hour
    plt.figure(figsize=(14, 8))
    
    # Plot mean severity for each area type
    severity_by_hour.plot(
        marker='o', 
        linewidth=2.5,
        ax=plt.gca()
    )
    
    # Add labels and title
    plt.title('Mean Accident Severity by Hour: Urban vs. Suburban', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Mean Severity', fontsize=14)
    
    # Set x-axis ticks for hours
    plt.xticks(range(0, 24))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_severity_by_hour.png')
    plt.close()
    
    # Create heatmap showing severity by hour and area type
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        severity_by_hour.T,
        annot=True,
        cmap='YlOrRd',
        fmt='.2f'
    )
    plt.title('Mean Accident Severity by Hour: Urban vs. Suburban', fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'urban_suburban_severity_by_hour_heatmap.png')
    plt.close()

def analyze_urban_suburban(df):
    """Run all urban vs. suburban analyses."""
    print("\nStarting Urban vs. Suburban Accident Analysis...")
    
    # Run analyses
    analyze_urban_suburban_distributions(df)
    analyze_timing_by_month(df)
    analyze_timing_by_hour(df)
    analyze_severity_by_month(df)
    analyze_severity_by_hour(df)
    
    print("Urban vs. Suburban analysis complete.")

def main():
    """Main function to run the EDA."""
    print("Starting targeted EDA...")
    create_output_directories()
    
    print("Loading basic data...")
    basic_df = load_data()
    
    print("Analyzing temporal patterns...")
    analyze_temporal_patterns(basic_df)
    
    print("Analyzing geographical patterns...")
    analyze_geographical_patterns(basic_df)
    
    print("Analyzing weather patterns...")
    analyze_weather_patterns(basic_df)
    
    print("Analyzing correlations...")
    analyze_correlations(basic_df)
    
    print("Analyzing road features...")
    analyze_road_features(basic_df)
    
    print("Analyzing intersection feature combinations...")
    analyze_intersection_combinations(basic_df)
    
    # Load data with features for urban-suburban analysis
    feature_df = load_data_with_features()
    
    # Run urban-suburban analysis
    analyze_urban_suburban(feature_df)
    
    print("EDA complete. Visualizations saved to reports/figures/eda/")

if __name__ == "__main__":
    main() 