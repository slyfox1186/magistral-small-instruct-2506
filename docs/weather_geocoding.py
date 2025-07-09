# OpenWeatherMap Geocoding API Documentation

## Overview

The Geocoding API is a tool for converting location names to coordinates and vice versa. It supports two primary methods:

1. Direct Geocoding
2. Reverse Geocoding

## Direct Geocoding

### By Location Name

#### API Endpoint
```
http://api.openweathermap.org/geo/1.0/direct?q={city name},{state code},{country code}&limit={limit}&appid={API key}
```

#### Parameters
- `q` (required): City name, state code, country code
- `appid` (required): API key
- `limit` (optional): Number of locations to return (max 5)

#### Example Request
```
http://api.openweathermap.org/geo/1.0/direct?q=London&limit=5&appid={API key}
```

#### Response Fields
- `name`: Location name
- `local_names`: Location names in different languages
- `lat`: Latitude
- `lon`: Longitude
- `country`: Country code
- `state`: State (where available)

### By Zip/Post Code

#### API Endpoint
```
http://api.openweathermap.org/geo/1.0/zip?zip={zip code},{country code}&appid={API key}
```

#### Parameters
- `zip code` (required): Zip/post code and country code
- `appid` (required): API key

#### Response Fields
- `zip`: Specified zip code
- `name`: Area name
- `lat`: Latitude of zip code centroid
- `lon`: Longitude of zip code centroid
- `country`: Country code

## Reverse Geocoding

### API Endpoint
```
http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit={limit}&appid={API key}
```

#### Parameters
- `lat`, `lon` (required): Geographical coordinates
- `appid` (required): API key
- `limit` (optional): Number of location names to return

#### Response Fields
Similar to direct geocoding:
- `name`: Location name
- `local_names`: Location names in different languages
- `lat`: Latitude
- `lon`: Longitude
- `country`: Country code
- `state`: State (where available)

## Authentication

All requests require a valid API key obtained from OpenWeatherMap's subscription service.

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request
- 401: Unauthorized (invalid API key)
- 404: Not found
- 429: Too many requests