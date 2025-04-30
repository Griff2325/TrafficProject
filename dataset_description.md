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
- **Severity**: Severity level (1-4) indicating impact on traffic (Used)
- **Description**: Description of the accident (Not used in current analysis)
- **Number**: Number of vehicles involved (Not used in current analysis)
- **Side**: Side of the road where the accident occurred (Used)
- **Airport_Code**: Nearby airport code (Not used in current analysis)

### Derived Features (Created During Processing)

#### Temporal Features
- **start_hour**: Hour of the day when the accident started
- **start_day_of_week**: Day of the week (0-6)
- **start_month**: Month of the year
- **start_year**: Year
- **start_quarter**: Quarter of the year
- **duration_minutes**: Duration of the accident in minutes
- **time_of_day**: Categorized time of day (Night, Morning, Afternoon, Evening)
- **season**: Season of the year (Winter, Spring, Summer, Fall)
- **is_weekend**: Whether the accident occurred on a weekend
- **is_rush_hour**: Whether the accident occurred during rush hour

#### Road Features
- **is_crossing**: Binary flag for crossing
- **is_junction**: Binary flag for junction
- **has_traffic_calming**: Binary flag for traffic calming
- **has_traffic_signal**: Binary flag for traffic signal
- **severity_level**: Categorized severity (Low, Medium, High, Very High)

#### Weather Features
- **temperature_category**: Categorized temperature (Freezing, Cold, Mild, Warm, Hot)
- **wind_speed_category**: Categorized wind speed (Calm, Light, Moderate, Strong, Very Strong)
- **humidity_category**: Categorized humidity (Very Dry, Dry, Moderate, Humid, Very Humid)
- **visibility_category**: Categorized visibility (Zero, Very Poor, Poor, Moderate, Good)
- **wind_direction_category**: Categorized wind direction
- **precipitation_category**: Categorized precipitation (None, Light, Moderate, Heavy, Very Heavy)
- **weather_severity**: Combined weather severity index

#### Light Features
- **is_day**: Binary flag for daylight
- **is_night**: Binary flag for nighttime
- **is_civil_twilight**: Binary flag for civil twilight
- **is_nautical_twilight**: Binary flag for nautical twilight
- **is_astronomical_twilight**: Binary flag for astronomical twilight

#### Location Features
- **is_urban**: Binary flag for urban areas
- **region**: US region (Northeast, Midwest, South, West)

*(Additional columns exist in the dataset, such as weather conditions, address details, and points of interest. These will be explored during the data loading and cleaning phase.)* 