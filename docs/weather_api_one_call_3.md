# OpenWeatherMap One Call API 3.0 Documentation

## Overview

The OpenWeatherMap One Call API 3.0 provides comprehensive weather data through four main endpoints:

1. Current and Forecasts Weather Data
2. Weather Data for Timestamp
3. Daily Aggregation
4. Weather Overview

## API Endpoints

### 1. Current and Forecasts Weather Data

**Endpoint:**
```
https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={API_KEY}
```

**Required Parameters:**
- `lat`: Latitude (decimal between -90 and 90)
- `lon`: Longitude (decimal between -180 and 180)
- `appid`: Your unique API key

**Optional Parameters:**
- `exclude`: Exclude specific data parts (current, minutely, hourly, daily, alerts)
- `units`: Measurement units (standard, metric, imperial)
- `lang`: Language for response

### 2. Weather Data for Timestamp

**Endpoint:**
```
https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp}&appid={API_KEY}
```

**Additional Parameters:**
- `dt`: Timestamp for historical weather data (Unix time)

### 3. Daily Aggregation

**Endpoint:**
```
https://api.openweathermap.org/data/3.0/onecall/day_summary?lat={lat}&lon={lon}&date={YYYY-MM-DD}&appid={API_KEY}
```

**Additional Parameters:**
- `date`: Date in YYYY-MM-DD format
- `tz`: Optional timezone specification

### 4. Weather Overview

**Endpoint:**
```
https://api.openweathermap.org/data/3.0/onecall/overview?lat={lat}&lon={lon}&appid={API_KEY}
```

## Key Features

- Minute forecast for 1 hour
- Hourly forecast for 48 hours
- Daily forecast for 8 days
- Government weather alerts
- Historical data for timestamps

## Response Format

All API responses are in JSON format with comprehensive weather data including temperature, humidity, wind speed, precipitation, and more.

## Authentication

All requests require a valid API key obtained from OpenWeatherMap's subscription service.