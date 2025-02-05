import requests
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API credentials
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Define the city
CITY = "A518986"  # Rabat 

def fetch_aqi_data():
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_API_TOKEN}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")
    
    return response.json()

def process_data(raw_data):
    data = raw_data['data']
    iaqi = data.get("iaqi", {})
    time_info = data.get("time", {})
    
    features = {
        "timestamp": time_info.get("iso", datetime.now().isoformat()),
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "aqi": data.get("aqi", 0),
        "dominant_pollutant": data.get("dominentpol", 0),
        "latitude": data.get("city", {}).get("geo", [0, 0])[0],
        "longitude": data.get("city", {}).get("geo", [0, 0])[1],
        "city_name": data.get("city", {}).get("name", 0),
        "location": data.get("city", {}).get("location", 0)
    }
    
    # Create DataFrame and enforce data types
    df = pd.DataFrame([features])
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Cast numerical columns to float64
    numerical_cols = ["pm25", "pm10", "aqi", "latitude", "longitude"]
    for col in numerical_cols:
        df[col] = df[col].astype("float64")
    
    # Add city identifier
    df["city"] = CITY
    
    return df

def store_features(df):
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, 
        project="AirQualityIndex"
    )
    fs = project.get_feature_store()
    
    # Get or create feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="AQI and location features",
        event_time="timestamp",
    )
    
    fg.insert(df)

if __name__ == "__main__":
    # Fetch and process data
    raw_data = fetch_aqi_data()
    processed_df = process_data(raw_data)
    
    # Store in Feature Store
    store_features(processed_df)
    print("Data successfully stored in Hopsworks!")