import requests
import json

url1 = "https://djsnss-web.onrender.com/events/upcoming-events"
url1_data = requests.get(url1).json()

upcoming_events = []
if url1_data and "events" in url1_data:
    for data in url1_data["events"]:
        event = {
            "name": data.get("name"),
            "description": data.get("description"),
            "longDescription": data.get("longDescription"),
            "location": data.get("location"), 
            "date": data.get("date"),
            "maxVolunteers": data.get("maxVolunteers"),
        }
        upcoming_events.append(event)

url2 = "https://djsnss-web.onrender.com/events/past-events"
url2_data = requests.get(url2).json()

past_events = []
if url2_data and "events" in url2_data:
    for data in url2_data["events"]:
        event = {
            "name": data.get("name"),
            "description": data.get("description"),
            "longDescription": data.get("longDescription"),
            "location": data.get("location"),
            "date": data.get("date"),
            "maxVolunteers": data.get("maxVolunteers"),
        }
        past_events.append(event)

with open("events.json", "w", encoding="utf-8") as f:
    json.dump({"upcomingEvents": upcoming_events, "pastEvents": past_events}, f, ensure_ascii=False, indent=4)