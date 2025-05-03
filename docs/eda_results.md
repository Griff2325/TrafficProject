# Exploratory Data Analysis Results

This document summarizes the findings from the exploratory data analysis of the US traffic accidents dataset.

## 1. Temporal Patterns

### Hourly Distribution
![Accidents by Hour](../reports/figures/eda/accidents_by_hour.png)

- Peak accident hours occur during rush hours:
  - Morning peak: 6-9 AM
  - Evening peak: 3-6 PM
- Lowest accident rates occur during early morning hours (2-4 AM)

### Daily Distribution
![Accidents by Day](../reports/figures/eda/accidents_by_day.png)

- Weekdays show higher accident rates than weekends
- Friday has the highest number of accidents
- Sunday has the lowest number of accidents

### Monthly Distribution
![Accidents by Month](../reports/figures/eda/accidents_by_month.png)

- Higher accident rates in winter months (November-February)
- Lower accident rates in summer months (June-August)
- Gradual increase from summer to winter

### Accident Duration
![Accident Duration](../reports/figures/eda/accident_duration.png)

- Most accidents last between 30-120 minutes
- Long-tail distribution with some accidents lasting several hours
- Median duration: ~60 minutes

## 2. Geographical Patterns

### State Distribution
![Accidents by State](../reports/figures/eda/accidents_by_state.png)

- California has the highest number of accidents
- Texas and Florida follow closely

### Accident Density
![Accident Density](../reports/figures/eda/accident_density.png)

- Clear hotspots in major metropolitan areas
- Higher density along major highways
- Urban areas show significantly higher accident rates
- Rural areas show scattered, lower-density patterns

## 3. Weather Patterns

### Temperature Impact
![Temperature Distribution](../reports/figures/eda/temperature_distribution.png)

- Most accidents occur in moderate temperatures (50-80°F)
- Fewer accidents in extreme temperatures
- Seems to be a normal distribution, temperature doesn't appear to have a huge impact

### Weather Conditions
![Weather Conditions](../reports/figures/eda/weather_conditions.png)

- Most accidents occur in clear weather
- Rain and cloudy conditions show high accident rates
- Snow and fog conditions show lower frequency but higher severity

## 4. Road Features

![Road Features](../reports/figures/eda/road_features.png)

- Traffic signals and junctions are most common accident locations
- Crossings and stop signs show high accident rates
- Roundabouts show lower accident rates than signals

## 5. Severity Analysis

### Severity Distribution
![Severity Distribution](../reports/figures/eda/severity_distribution.png)

- Most accidents are of moderate severity (Level 2)

### Severity by Weather
![Severity by Weather](../reports/figures/eda/severity_by_weather.png)

- Light blowing snow leads to most severe accidents
- All severe weather shown seems to cause more severe accidents

### Mean Severity Heatmap (Hour vs. Day of Week)
![Mean Severity by Hour and Day of Week](../reports/figures/eda/heatmap_hour_dayofweek_severity.png)

- This heatmap shows the average severity level for accidents occurring at specific hours on specific days.
- Saturday and Sunday morning accidents tend to be most severe
- Overall, Sundays have the most severe accidents

## 6. Urban vs. Suburban Analysis

This section compares accident patterns between areas classified as Urban and Suburban using the GeoPandas spatial join.

### Area Type Distribution
![Distribution by Area Type](../reports/figures/eda/urban_suburban_area_type_distribution.png)

- Most accidents occur in urban areas

### Timing Analysis (Monthly)
![Accident Counts by Month](../reports/figures/eda/urban_suburban_timing_by_month.png)

- By month, urban and suburban both follow a similar trend

### Timing Analysis (Hourly)
![Accident Counts by Hour](../reports/figures/eda/urban_suburban_timing_by_hour.png)

- Both urban and suburban follow a similar trend, but urban areas are greatly exaggerated during commute hours as people are getting to and leaving work in the city

### Severity Analysis (Monthly)
![Mean Severity by Month](../reports/figures/eda/urban_suburban_severity_by_month.png)

- Severity in urban and suburban are similar, with urban experiencing a noticible dip in September

### Severity Analysis (Hourly)
![Mean Severity by Hour](../reports/figures/eda/urban_suburban_severity_by_hour.png)

- Urban areas typically have less severe accidents during the workday while suburban severity rates are higher over the same time
- Suburban areas have a spike in severity around noon (propably due to people driving to get lunch)

### Severity Analysis (Day of Week)
![Mean Severity by Day of Week](../reports/figures/eda/urban_suburban_severity_by_day_of_week.png)

- Urban areas tend to have more severe accidents during weekends - likely due to activities, bars, restaurants, etc.

## 7. Correlation Analysis

![Correlation Matrix](../reports/figures/eda/feature_correlations.png)

### Key Correlations
- Strong positive correlation between:
  - Temperature and Wind Chill
  - Visibility and Precipitation
- Moderate negative correlation between:
  - Temperature and Humidity
  - Visibility and Wind Speed
- Weak correlation between:
  - Severity and most weather features
  - Duration and weather conditions

  This doesn't seem to bring any real insights, as these weather features typically occur together.

## 8. Key Insights

1. **Temporal Factors**:
   - Rush hours and weekdays are high-risk periods
   - Winter months show increased accident rates
   - Early morning hours show lower accident rates

2. **Geographical Factors**:
   - Urban areas and major highways are high-risk locations
   - States with high population density show more accidents
   - Clear regional patterns in accident distribution

3. **Weather Impact**:
   - Most accidents occur in clear weather
   - Rain and cloudy conditions show high accident rates
   - Snow and fog conditions show lower frequency but higher severity

4. **Road Features**:
   - Traffic signals and junctions are critical locations
   - Road features designed for safety show lower accident rates
   - Infrastructure plays a significant role in accident patterns

5. **Severity Patterns**:
   * Weather conditions strongly influence accident severity
   * Time of day impacts severity levels
   * Most accidents are of moderate severity

6. **Urban vs. Suburban Differences**:
   * Urban areas account for the majority of accidents.
   * Urban accident counts peak significantly during commute hours compared to suburban areas.
   * Monthly trends for accident counts and severity are broadly similar between urban and suburban areas.
   * Urban accidents tend to be *less* severe during typical workday hours.
   * Urban accidents tend to be *more* severe on weekends.
   * Suburban areas experience higher average severity during the workday, with a notable peak around noon.
