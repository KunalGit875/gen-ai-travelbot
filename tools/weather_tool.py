import os, requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from meteostat import Point, Monthly
from pydantic import BaseModel
from langchain.tools import StructuredTool
import dateparser

def get_weather_or_climate(city: str, date_str: str) -> str:
    parsed_date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
    if not parsed_date: return f"Couldn't understand date: {date_str}"
    parsed_date = parsed_date.date()

    location = Nominatim(user_agent="travel-assistant").geocode(city)
    if not location: return f"Couldn't find city: {city}"

    lat, lon = location.latitude, location.longitude
    today = datetime.today().date()

    if parsed_date <= today + timedelta(days=5):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key: return "OpenWeather API key missing."
        res = requests.get(f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric")
        data = res.json()
        forecast = [f for f in data["list"] if f["dt_txt"].startswith(parsed_date.isoformat())]
        if not forecast: return f"No forecast for {city} on {parsed_date}"
        return f"Forecast for {city} on {parsed_date}:\n" + '\n'.join(f"{f['dt_txt']}: {f['weather'][0]['description'].capitalize()}, {f['main']['temp']}°C" for f in forecast)
    else:
        point = Point(lat, lon)
        end = datetime.today()
        start = end.replace(year=end.year - 5)
        data = Monthly(point, start, end).fetch()
        monthly = data[data.index.month == parsed_date.month]
        return f"Avg. {city} in {parsed_date.strftime('%B')}: {monthly['tavg'].mean():.1f}°C, {monthly['prcp'].mean():.1f}mm rain"

class WeatherInput(BaseModel):
    city: str
    date_str: str

weather_forecast_tool = StructuredTool.from_function(
    name="Weather_forecast",
    description="Get weather/climate info for a city and date. Accepts phrases like 'next week' or '2025-07-01'.",
    func=get_weather_or_climate,
    args_schema=WeatherInput
)
