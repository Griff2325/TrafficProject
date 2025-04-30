"""
Data cleaning module for the US Traffic Accidents dataset.

This module contains functions for cleaning and preprocessing the US Traffic Accidents dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the raw dataset from the specified path.
    
    Args:
        data_path (str): Path to the raw data file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        logger.info(f"Attempting to load data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataset and return a summary.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Summary of missing values
    """
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    missing_summary = missing_summary.sort_values('Percentage', ascending=False)
    
    logger.info("Missing values summary:")
    logger.info(missing_summary)
    return missing_summary

def check_duplicates(df: pd.DataFrame) -> int:
    """
    Check for duplicate rows in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        int: Number of duplicate rows
    """
    duplicates = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {duplicates}")
    return duplicates

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning steps.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    # Convert date/time columns to datetime
    date_columns = ['Start_Time', 'End_Time', 'Weather_Timestamp']
    for col in date_columns:
        if col in df_clean.columns:
            logger.info(f"Converting {col} to datetime")
            df_clean[col] = pd.to_datetime(df_clean[col])
    
    # Remove duplicates if any
    duplicates = check_duplicates(df_clean)
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
    
    return df_clean

def main():
    """
    Main function to run the data cleaning process.
    """
    logger.info("Starting data cleaning process...")
    
    # Define paths
    data_dir = Path('data')
    raw_data_path = data_dir / 'raw' / 'US_Accidents_March23.csv'
    processed_data_path = data_dir / 'processed' / 'cleaned_accidents.csv'
    
    logger.info(f"Raw data path: {raw_data_path}")
    logger.info(f"Processed data path: {processed_data_path}")
    
    # Create processed directory if it doesn't exist
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created processed directory: {processed_data_path.parent}")
    
    try:
        # Load data
        logger.info("Loading raw data...")
        df = load_data(raw_data_path)
        
        # Check missing values
        logger.info("Checking missing values...")
        missing_summary = check_missing_values(df)
        
        # Clean data
        logger.info("Cleaning data...")
        df_clean = clean_data(df)
        
        # Save cleaned data
        logger.info(f"Saving cleaned data to {processed_data_path}")
        df_clean.to_csv(processed_data_path, index=False)
        logger.info(f"Cleaned data saved successfully")
        
    except Exception as e:
        logger.error(f"Error in data cleaning process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 