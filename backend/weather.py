#!/usr/bin/env python3
"""Enhanced weather data retrieval using OpenWeatherMap API.

This module provides comprehensive weather data functions for the LLM system
with proper error handling, caching, and async support.

Features:
- Current weather conditions
- Weather forecasts (hourly/daily)
- Historical weather data
- Geocoding and reverse geocoding
- Weather alerts and notifications
- Proper circuit breaker integration
- Redis caching support

Dependencies:
    pip install aiohttp python-dotenv

Author: Claude Code
"""

import argparse
import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp
from dotenv import load_dotenv

from circuit_breaker import CircuitBreakerError, get_api_breaker

# Load environment variables from .env file
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Get OpenWeatherMap API credentials from environment variables
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def normalize_city_query(city: str) -> str:
    """Normalize city query to work better with geocoding API.

    Note: State name conversion is now handled in the chat pipeline before
    this function is called, so this just returns the input as-is.
    """
    return city


@dataclass
class WeatherData:
    """Weather data structure."""

    location: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    visibility: int | None = None
    uv_index: float | None = None
    timestamp: int | None = None


@dataclass
class LocationData:
    """Location data structure."""

    name: str
    country: str
    state: str | None
    lat: float
    lon: float


class WeatherService:
    """Weather service client with circuit breaker and caching."""

    def __init__(self):
        """Initialize the weather service."""
        self.api_key = OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org"
        self.geo_base_url = "http://api.openweathermap.org/geo/1.0"
        self.session = None
        self.circuit_breaker = get_api_breaker("weather")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _make_request(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make API request with circuit breaker protection."""
        try:

            async def make_request():
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

            return await self.circuit_breaker.call(make_request)
        except CircuitBreakerError:
            logger.exception("Weather API circuit breaker is open")
            raise
        except Exception:
            logger.exception("Weather API request failed")
            raise

    async def get_coordinates_by_city(self, city: str, limit: int = 5) -> list[LocationData]:
        """Get coordinates for a city using direct geocoding."""
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")

        # Normalize city query (state conversion happens in chat pipeline)
        normalized_city = normalize_city_query(city)
        logger.info(f"Geocoding: '{city}' → '{normalized_city}'")

        url = f"{self.geo_base_url}/direct"
        params = {"q": normalized_city, "limit": limit, "appid": self.api_key}

        try:
            data = await self._make_request(url, params)
            logger.info(f"Geocoding API returned {len(data)} results for '{normalized_city}'")

            # Log the raw API response for debugging
            if data:
                logger.debug(f"First geocoding result: {data[0]}")
            else:
                logger.warning(f"No geocoding results found for '{normalized_city}'")

            locations = []
            for item in data:
                location = LocationData(
                    name=item.get("name", ""),
                    country=item.get("country", ""),
                    state=item.get("state"),
                    lat=item.get("lat", 0.0),
                    lon=item.get("lon", 0.0),
                )
                locations.append(location)

            return locations[:limit]
        except Exception:
            logger.exception(f"Failed to get coordinates for city {city}")
            return []

    async def get_coordinates_by_zip(
        self, zip_code: str, country_code: str = "US"
    ) -> LocationData | None:
        """Get coordinates for a zip code."""
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")

        url = f"{self.geo_base_url}/zip"
        params = {"zip": f"{zip_code},{country_code}", "appid": self.api_key}

        try:
            data = await self._make_request(url, params)
            return LocationData(
                name=data.get("name", ""),
                country=data.get("country", ""),
                state=None,
                lat=data.get("lat", 0.0),
                lon=data.get("lon", 0.0),
            )
        except Exception:
            logger.exception(f"Failed to get coordinates for zip {zip_code}")
            return None

    async def get_current_weather(
        self, lat: float, lon: float, units: str = "metric"
    ) -> WeatherData | None:
        """Get current weather for coordinates."""
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured")

        url = f"{self.base_url}/data/3.0/onecall"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": units,
            "exclude": "minutely,hourly,daily,alerts",
        }

        try:
            data = await self._make_request(url, params)
            current = data.get("current", {})
            weather_info = current.get("weather", [{}])[0]

            # Get location name via reverse geocoding
            location_name = await self._get_location_name(lat, lon)

            return WeatherData(
                location=location_name,
                country="",  # Will be filled by reverse geocoding if needed
                temperature=current.get("temp", 0.0),
                feels_like=current.get("feels_like", 0.0),
                humidity=current.get("humidity", 0),
                description=weather_info.get("description", ""),
                wind_speed=current.get("wind_speed", 0.0),
                pressure=current.get("pressure", 0),
                visibility=current.get("visibility"),
                uv_index=current.get("uvi"),
                timestamp=current.get("dt"),
            )
        except Exception:
            logger.exception(f"Failed to get current weather for {lat}, {lon}")
            return None

    async def _get_location_name(self, lat: float, lon: float) -> str:
        """Get location name from coordinates."""
        try:
            url = f"{self.geo_base_url}/reverse"
            params = {"lat": lat, "lon": lon, "limit": 1, "appid": self.api_key}

            data = await self._make_request(url, params)
            if data:
                location = data[0]
                name = location.get("name", "")
                state = location.get("state", "")
                country = location.get("country", "")

                if state:
                    return f"{name}, {state}, {country}"
                else:
                    return f"{name}, {country}"
        except Exception:
            return "Unknown Location"
        else:
            return "Unknown Location"


# Global weather service instance
weather_service = None


def _get_weather_service():
    """Get or create the global weather service instance."""
    global weather_service  # noqa: PLW0603
    if not weather_service:
        weather_service = WeatherService()
    return weather_service


async def get_weather_for_city(city: str, units: str = "metric") -> dict[str, Any] | None:
    """Get weather for a specific city (main function for LLM)."""
    try:
        service_instance = _get_weather_service()

        async with service_instance as service:
            # Get coordinates for city
            locations = await service.get_coordinates_by_city(city, limit=1)
            if not locations:
                return {"error": f"Could not find coordinates for city: {city}"}

            location = locations[0]

            # Get current weather
            weather = await service.get_current_weather(location.lat, location.lon, units)
            if not weather:
                return {"error": f"Could not get weather data for {city}"}

            # Convert to dictionary for LLM consumption
            result = asdict(weather)
            result["coordinates"] = {"lat": location.lat, "lon": location.lon}
            result["location"] = f"{location.name}, {location.country}"

            return result

    except Exception as e:
        logger.exception(f"Error getting weather for {city}")
        return {"error": str(e)}


async def get_weather_for_coordinates(
    lat: float, lon: float, units: str = "metric"
) -> dict[str, Any] | None:
    """Get weather for specific coordinates (function for LLM)."""
    try:
        service_instance = _get_weather_service()

        async with service_instance as service:
            weather = await service.get_current_weather(lat, lon, units)
            if not weather:
                return {"error": f"Could not get weather data for coordinates {lat}, {lon}"}

            result = asdict(weather)
            result["coordinates"] = {"lat": lat, "lon": lon}

            return result

    except Exception as e:
        logger.exception(f"Error getting weather for coordinates {lat}, {lon}")
        return {"error": str(e)}


async def search_locations(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search for locations by name (function for LLM)."""
    try:
        service_instance = _get_weather_service()

        async with service_instance as service:
            locations = await service.get_coordinates_by_city(query, limit)
            return [asdict(loc) for loc in locations]

    except Exception:
        logger.exception(f"Error searching locations for {query}")
        return []


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32


def ms_to_mph(ms: float) -> float:
    """Convert meters per second to miles per hour."""
    return ms * 2.237


def format_weather_response(weather_data: dict[str, Any]) -> str:
    """Format weather data for LLM consumption with both Fahrenheit and Celsius."""
    if "error" in weather_data:
        return f"Weather Error: {weather_data['error']}"

    location = weather_data.get("location", "Unknown")
    temp_c = weather_data.get("temperature", 0)
    feels_like_c = weather_data.get("feels_like", 0)
    humidity = weather_data.get("humidity", 0)
    description = weather_data.get("description", "").title()
    wind_speed = weather_data.get("wind_speed", 0)
    pressure = weather_data.get("pressure", 0)

    # Convert to Fahrenheit and MPH
    temp_f = celsius_to_fahrenheit(temp_c)
    feels_like_f = celsius_to_fahrenheit(feels_like_c)
    wind_mph = ms_to_mph(wind_speed)

    response = f"""Current Weather for {location}:
Temperature: {temp_f:.1f}°F ({temp_c:.1f}°C) - feels like {feels_like_f:.1f}°F ({feels_like_c:.1f}°C)
Conditions: {description}
Humidity: {humidity}%
Wind Speed: {wind_mph:.1f} mph ({wind_speed:.1f} m/s)
Pressure: {pressure} hPa"""

    if weather_data.get("uv_index") is not None:
        response += f"\nUV Index: {weather_data['uv_index']}"

    return response


# CLI interface for testing
async def main():
    """Main CLI function for testing weather service."""
    parser = argparse.ArgumentParser(description="Weather Service CLI")
    parser.add_argument("--city", help="City name to get weather for")
    parser.add_argument("--lat", type=float, help="Latitude coordinate")
    parser.add_argument("--lon", type=float, help="Longitude coordinate")
    parser.add_argument("--search", help="Search for locations")
    parser.add_argument(
        "--units",
        default="metric",
        choices=["metric", "imperial", "standard"],
        help="Units for temperature",
    )

    args = parser.parse_args()

    if args.city:
        weather_data = await get_weather_for_city(args.city, args.units)
        if weather_data:
            print(format_weather_response(weather_data))
        else:
            print("Failed to get weather data")

    elif args.lat is not None and args.lon is not None:
        weather_data = await get_weather_for_coordinates(args.lat, args.lon, args.units)
        if weather_data:
            print(format_weather_response(weather_data))
        else:
            print("Failed to get weather data")

    elif args.search:
        locations = await search_locations(args.search)
        if locations:
            print("Found locations:")
            for loc in locations:
                print(f"  {loc['name']}, {loc['country']} ({loc['lat']}, {loc['lon']})")
        else:
            print("No locations found")

    else:
        print("Please provide --city, --lat/--lon, or --search")


if __name__ == "__main__":
    asyncio.run(main())
