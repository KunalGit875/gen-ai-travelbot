from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.tools import StructuredTool
from datetime import datetime, timedelta
from meteostat import Point, Monthly
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
import dateparser
import requests
import os

# ---------- Search Tool ----------
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

# ---------- Wikipedia Tool ----------
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# ---------- Weather Tool ----------

def get_weather_or_climate(city: str, date_str: str) -> str:
    """
    city: Name of the city (e.g., "Bangalore")
    date_str: Natural language date (e.g., "tomorrow", "next week", "2025-06-20")
    """
    try:
        parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        if not parsed_date:
            return f"Could not understand the date '{date_str}'. Try something like 'tomorrow' or 'next Friday'."
        parsed_date = parsed_date.date()
        today = datetime.today().date()

        # Geocoding
        geolocator = Nominatim(user_agent="travel-assistant")
        location = geolocator.geocode(city)
        if not location:
            return f"Could not find location for '{city}'."

        latitude, longitude = location.latitude, location.longitude

        # --- Short-term (Forecast) ---
        if parsed_date <= today + timedelta(days=5):
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                return "OpenWeather API key missing."

            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            if data.get("cod") != "200":
                return f"Error: {data.get('message', 'Failed to get forecast')}"

            # Filter forecasts for the target date
            forecast = [f for f in data["list"] if f["dt_txt"].startswith(parsed_date.isoformat())]
            if not forecast:
                return f"No forecast available for {city} on {parsed_date}."

            detailed_forecast = []
            for f in forecast:
                detailed_forecast.append(
                    f"""
                    Time: {f['dt_txt']}
                    - Weather: {f['weather'][0]['description'].capitalize()}
                    - Temperature: {f['main']['temp']}°C (Feels like: {f['main']['feels_like']}°C)
                    - Min/Max: {f['main']['temp_min']}°C / {f['main']['temp_max']}°C
                    - Humidity: {f['main']['humidity']}%
                    - Pressure: {f['main']['pressure']} hPa
                    - Wind: {f['wind']['speed']} m/s, Direction: {f['wind'].get('deg', 'N/A')}°
                    - Cloud Cover: {f['clouds']['all']}%
                    - Visibility: {f.get('visibility', 0) / 1000} km
                    """.strip()
                )

            return f"📅 Detailed weather forecast for {city.title()} on {parsed_date}:\n\n" + "\n\n".join(detailed_forecast)

        # --- Long-term (Climate) ---
        else:
            point = Point(latitude, longitude)
            end = datetime.today()
            start = end.replace(year=end.year - 10)
            data = Monthly(point, start, end).fetch()

            if data.empty:
                return f"No historical climate data found for {city}."

            month = parsed_date.month
            monthly_data = data[data.index.month == month]

            avg_temp = monthly_data["tavg"].mean()
            min_temp = monthly_data["tmin"].mean()
            max_temp = monthly_data["tmax"].mean()
            avg_precip = monthly_data["prcp"].mean()
            snow = monthly_data["snow"].mean() if "snow" in monthly_data else 0

            return (
                f"📊 Typical climate in {city.title()} during {parsed_date.strftime('%B')} (10-year average):\n"
                f"- Avg. Temperature: {avg_temp:.1f}°C\n"
                f"- Min/Max Temperature: {min_temp:.1f}°C / {max_temp:.1f}°C\n"
                f"- Precipitation: {avg_precip:.1f} mm\n"
                f"- Snowfall: {snow:.1f} mm\n"
            )

    except Exception as e:
        return f"Error: {str(e)}"

# Tool schema for LangChain agent
class WeatherInput(BaseModel):
    city: str
    date_str: str

weather_forecast_tool = StructuredTool.from_function(
    name="Weather_forecast",
    description="Get weather or climate info for a city and date. Accepts natural dates like 'tomorrow' or 'next week'.",
    func=get_weather_or_climate,
    args_schema=WeatherInput,
)
