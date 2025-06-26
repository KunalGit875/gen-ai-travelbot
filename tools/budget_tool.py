import json, re
from pydantic import BaseModel
from langchain.tools import tool
from langchain_core.output_parsers import PydanticOutputParser

@tool
def estimate_budget_tool(destination: str, duration: int, style: str) -> str:
    """
    Estimate budget components based on destination, duration, and travel style.
    Returns JSON with hotel, food, activities, and transport cost.
    """
    city_base_rates = {
        "goa": {"hotel": 1000, "food": 400, "activities": 300},
        "manali": {"hotel": 900, "food": 350, "activities": 250},
        "dubai": {"hotel": 3500, "food": 1200, "activities": 1000},
        "paris": {"hotel": 5000, "food": 2000, "activities": 1200},
    }
    style_multipliers = {"budget": 1.0, "midrange": 1.8, "luxury": 3.5}
    rates = city_base_rates.get(destination.lower(), city_base_rates["goa"])
    multiplier = style_multipliers.get(style.lower(), 1.8)

    return json.dumps({
        "hotel_price_per_night": int(rates["hotel"] * multiplier),
        "food_cost_per_day": int(rates["food"] * multiplier),
        "activity_cost_per_day": int(rates["activities"] * multiplier),
        "transport_cost": 2000
    })
class TravelModel(BaseModel):
    destination: str
    duration: int
    style: str
    hotels: list[str] = []
    restaurants: list[str] = []
    activities: list[str] = []
    places_to_visit: list[str] = []

def format_travel_summary(data: TravelModel, tool_output: dict) -> str:
    hotel_total = tool_output["hotel_price_per_night"] * data.duration
    food_total = tool_output["food_cost_per_day"] * data.duration
    activity_total = tool_output["activity_cost_per_day"] * len(data.activities)
    total = hotel_total + food_total + activity_total + tool_output["transport_cost"]

    summary = f"""
{data.duration}-Day {data.style.title()} Trip to {data.destination.title()}:

Hotels:
""" + ''.join(f"- {h} (₹{tool_output['hotel_price_per_night']:,}/night)\n" for h in data.hotels)

    summary += "\nRestaurants:\n" + ''.join(f"- {r}\n" for r in data.restaurants)
    summary += "\nPlaces:\n" + ''.join(f"- {p}\n" for p in data.places_to_visit)
    summary += "\nActivities:\n" + ''.join(f"- {a}\n" for a in data.activities)

    summary += f"""
\n*Total Est. Cost*: ₹{total:,}
  • Hotel: ₹{hotel_total:,}
  • Food: ₹{food_total:,}
  • Activities: ₹{activity_total:,}
  • Transport: ₹{tool_output['transport_cost']:,}
"""
    return summary.strip()
