# US Accidents Dataset Description (US_Accidents_March23.csv)

This file describes the columns available in the `US_Accidents_March23.csv` dataset and their usage in the project.

## Core Columns (Used in Analysis)

### Identification and Source
- **ID**: Unique identifier of the accident record (Used)
- **Source**: Source of raw accident data (Used for data quality assessment)

### Temporal Information
- **Start_Time**: Start time of the accident in local time zone (Used)
- **End_Time**: End time of the accident in local time zone (Used)
- **Weather_Timestamp**: Time when weather conditions were recorded (Used)

### Location Information
- **Start_Lat**: Latitude in GPS coordinate of the start point (Used)
- **Start_Lng**: Longitude in GPS coordinate of the start point (Used)
- **End_Lat**: Latitude in GPS coordinate of the end point (Used)
- **End_Lng**: Longitude in GPS coordinate of the end point (Used)
- **Distance(mi)**: Length of road extent affected by the accident in miles (Used)
- **Street**: Street name (Used)
- **City**: City name (Used)
- **County**: County name (Used)
- **State**: State abbreviation (Used)
- **Zipcode**: ZIP code (Used)
- **Country**: Country code (Used)
- **Timezone**: Timezone of the location (Used)

### Weather Conditions
- **Temperature(F)**: Temperature in Fahrenheit (Used)
- **Wind_Chill(F)**: Wind chill temperature in Fahrenheit (Used)
- **Humidity(%)**: Humidity percentage (Used)
- **Pressure(in)**: Air pressure in inches (Used)
- **Visibility(mi)**: Visibility in miles (Used)
- **Wind_Direction**: Wind direction (Used)
- **Wind_Speed(mph)**: Wind speed in miles per hour (Used)
- **Precipitation(in)**: Precipitation in inches (Used)
- **Weather_Condition**: Description of weather conditions (Used)

### Road Features
- **Amenity**: Whether the accident occurred near an amenity (Used)
- **Bump**: Whether the accident occurred near a bump (Used)
- **Crossing**: Whether the accident occurred at a crossing (Used)
- **Give_Way**: Whether the accident occurred at a give way sign (Used)
- **Junction**: Whether the accident occurred at a junction (Used)
- **No_Exit**: Whether the accident occurred at a no exit (Used)
- **Railway**: Whether the accident occurred at a railway crossing (Used)
- **Roundabout**: Whether the accident occurred at a roundabout (Used)
- **Station**: Whether the accident occurred near a station (Used)
- **Stop**: Whether the accident occurred at a stop sign (Used)
- **Traffic_Calming**: Whether the accident occurred near traffic calming (Used)
- **Traffic_Signal**: Whether the accident occurred at a traffic signal (Used)
- **Turning_Loop**: Whether the accident occurred at a turning loop (Used)

### Light Conditions
- **Sunrise_Sunset**: Whether the accident occurred during day or night (Used)
- **Civil_Twilight**: Whether the accident occurred during civil twilight (Used)
- **Nautical_Twilight**: Whether the accident occurred during nautical twilight (Used)
- **Astronomical_Twilight**: Whether the accident occurred during astronomical twilight (Used)

### Accident Details
- **Severity**: Severity level (1-4) indicating impact on traffic (Used as Target Variable)
- **Description**: Description of the accident (Not used in modeling)
- **Number**: Street number in address (Not used in modeling)
- **Side**: Side of the road where the accident occurred (R/L) (Not explicitly used in modeling, might be indirectly through coordinates)
- **Airport_Code**: Nearby airport code (Not used in modeling)

## Derived Features (Created by `src/features/build_features.py`)

This section lists features engineered from the core columns to potentially improve model performance.

### Temporal Features
- **start_hour**: Hour of the day when the accident started (0-23)
- **start_day_of_week**: Day of the week (0=Monday, 6=Sunday)
- **start_month**: Month of the year (1-12)
- **start_year**: Year of the accident
- **duration_minutes**: Calculated duration of the accident in minutes (End_Time - Start_Time)
- **time_of_day**: Categorized time of day ('Night', 'Morning', 'Afternoon', 'Evening')
- **season**: Categorized season based on month ('Winter', 'Spring', 'Summer', 'Fall')
- **is_weekend**: Binary flag (1 if Saturday/Sunday, 0 otherwise)
- **is_rush_hour**: Binary flag (1 if 7-9 AM or 4-6 PM, 0 otherwise)
- **is_holiday**: Binary flag (1 if the date is a US federal holiday, 0 otherwise)

### Weather Features
- **temperature_category**: Categorized temperature ('Freezing', 'Cold', 'Mild', 'Warm', 'Hot')
- **wind_speed_category**: Categorized wind speed ('Calm', 'Light', 'Moderate', 'Strong', 'Very Strong')
- **humidity_category**: Categorized humidity ('Very Dry', 'Dry', 'Moderate', 'Humid', 'Very Humid')
- **visibility_category**: Categorized visibility ('Zero', 'Very Poor', 'Poor', 'Moderate', 'Good')
- **precipitation_category**: Categorized precipitation ('None', 'Light', 'Moderate', 'Heavy', 'Very Heavy')
- **weather_severity**: Numerical index derived from `Weather_Condition` (higher value indicates more severe weather)
- **[WeatherCol]_missing**: Binary flags indicating if original weather columns had missing values (e.g., `Temperature(F)_missing`)

### Road Features
- **is_intersection**: Binary flag (1 if Junction, Crossing, or Traffic_Signal is True, 0 otherwise)
- **has_traffic_control**: Binary flag (1 if Traffic_Signal, Stop, or Give_Way is True, 0 otherwise)
- **is_complex_intersection**: Binary flag (1 if (Junction OR Crossing) AND Traffic_Signal are True, 0 otherwise)
- **road_feature_count**: Count of how many road features (Amenity, Bump, Crossing, etc.) are True for the accident location

### Location Features
- **region**: US region derived from State ('Northeast', 'Midwest', 'South', 'West')
- **is_urban**: Binary flag indicating if the accident occurred within an urban area (1) or not (0). Derived using a spatial join with the [2020 TIGER/Line Urban Areas shapefile](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=Urban+Areas) via `geopandas`. Falls back to a state-based estimate if the shapefile is missing.

*(Note: The accuracy and presence of derived features depend on the successful execution of the `src/features/build_features.py` script.)*

*(Additional columns exist in the dataset, such as weather conditions, address details, and points of interest. These will be explored during the data loading and cleaning phase.)* 