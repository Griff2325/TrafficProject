# Exploratory Data Analysis Results

This document summarizes the findings from the exploratory data analysis of the US traffic accidents dataset.

## 1. Temporal Patterns

### Hourly Distribution
![Accidents by Hour](figures/accidents_by_hour.png)

- Peak accident hours occur during rush hours:
  - Morning peak: 7-9 AM
  - Evening peak: 4-6 PM
- Lowest accident rates occur during early morning hours (2-4 AM)

### Daily Distribution
![Accidents by Day](figures/accidents_by_day.png)

- Weekdays show higher accident rates than weekends
- Friday has the highest number of accidents
- Sunday has the lowest number of accidents

### Monthly Distribution
![Accidents by Month](figures/accidents_by_month.png)

- Higher accident rates in winter months (November-February)
- Lower accident rates in summer months (June-August)
- Gradual increase from summer to winter

### Accident Duration
![Accident Duration](figures/accident_duration.png)

- Most accidents last between 30-120 minutes
- Long-tail distribution with some accidents lasting several hours
- Median duration: ~60 minutes

## 2. Geographical Patterns

### State Distribution
![Accidents by State](figures/accidents_by_state.png)

- California has the highest number of accidents
- Texas and Florida follow closely
- Northeastern states show high accident density
- Urban states generally have higher accident rates

### Accident Density
![Accident Density](figures/accident_density.png)

- Clear hotspots in major metropolitan areas
- Higher density along major highways
- Urban areas show significantly higher accident rates
- Rural areas show scattered, lower-density patterns

## 3. Weather Patterns

### Temperature Impact
![Temperature Distribution](figures/temperature_distribution.png)

- Most accidents occur in moderate temperatures (50-80°F)
- Fewer accidents in extreme temperatures
- Slight increase in accidents during freezing temperatures

### Visibility Impact
![Visibility Distribution](figures/visibility_distribution.png)

- Most accidents occur in good visibility conditions (>5 miles)
- Significant number of accidents in poor visibility (<1 mile)
- Clear correlation between visibility and accident frequency

### Weather Conditions
![Weather Conditions](figures/weather_conditions.png)

- Most accidents occur in clear weather
- Rain and cloudy conditions show high accident rates
- Snow and fog conditions show lower frequency but higher severity

## 4. Road Features

![Road Features](figures/road_features.png)

- Traffic signals and junctions are most common accident locations
- Crossings and stop signs show high accident rates
- Roundabouts and traffic calming show lower accident rates
- Railway crossings show moderate accident rates

## 5. Severity Analysis

### Severity Distribution
![Severity Distribution](figures/severity_distribution.png)

- Most accidents are of moderate severity (Level 2-3)
- Fewer high-severity accidents (Level 4)
- Low-severity accidents (Level 1) are relatively common

### Severity by Hour
![Severity by Hour](figures/severity_by_hour.png)

- Higher severity accidents tend to occur during rush hours
- Nighttime accidents show higher severity
- Early morning hours show lower severity

### Severity by Weather
![Severity by Weather](figures/severity_by_weather.png)

- Snow and ice conditions show highest average severity
- Fog and heavy rain show high severity
- Clear weather shows moderate severity
- Light rain shows lower severity

## 6. Correlation Analysis

![Correlation Matrix](figures/correlation_matrix.png)

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

## Key Insights

1. **Temporal Factors**:
   - Rush hours and weekdays are high-risk periods
   - Winter months show increased accident rates
   - Early morning hours show lower accident rates

2. **Geographical Factors**:
   - Urban areas and major highways are high-risk locations
   - States with high population density show more accidents
   - Clear regional patterns in accident distribution

3. **Weather Impact**:
   - Adverse weather conditions increase accident likelihood
   - Poor visibility significantly impacts accident rates
   - Temperature extremes show complex relationships with accidents

4. **Road Features**:
   - Traffic signals and junctions are critical locations
   - Road features designed for safety show lower accident rates
   - Infrastructure plays a significant role in accident patterns

5. **Severity Patterns**:
   - Weather conditions strongly influence accident severity
   - Time of day impacts severity levels
   - Most accidents are of moderate severity

These insights will guide the feature engineering process and model development in the next steps of the project. 