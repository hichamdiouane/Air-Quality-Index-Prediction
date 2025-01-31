import requests
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get variables
AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Define the city
CITY = "A518986"  # Rabat 

# Fetch raw data from AQICN API
def fetch_aqi_data():
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_API_TOKEN}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")
    
    data = response.json()["data"]
    return data

# Process raw data into features and targets
def process_data(raw_data):
    # Extract features with fallback to NaN for missing values
    iaqi = raw_data.get("iaqi", {})
    features = {
        "timestamp": datetime.now().isoformat(),
        "pm25": iaqi.get("pm25", {}).get("v", None),  # Use None for missing values
        "pm10": iaqi.get("pm10", {}).get("v", None),
        "o3": iaqi.get("o3", {}).get("v", None),
        "no2": iaqi.get("no2", {}).get("v", None),
        "temperature": iaqi.get("t", {}).get("v", None),
        "humidity": iaqi.get("h", {}).get("v", None),
        "pressure": iaqi.get("p", {}).get("v", None),
        "aqi": raw_data.get("aqi", None),
    }

    # Create DataFrame and enforce data types
    df = pd.DataFrame([features])
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Cast numerical columns to float64 (NaN-friendly)
    numerical_cols = ["pm25", "pm10", "o3", "no2", "temperature", "humidity", "pressure", "aqi"]
    for col in numerical_cols:
        df[col] = df[col].astype("float64")  # Use Pandas nullable float type
    
    # Add city identifier
    df["city"] = CITY
    
    return df

# Store data in Hopsworks Feature Store
def store_features(df):
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, 
        project="AirQualityIndex"
    )
    fs = project.get_feature_store()

    # Delete old feature group if it exists
    try:
        fs.get_feature_group(name="aqi_features", version=1).delete()
    except:
        pass

    # Create a new feature group with explicit schema
    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp", "city"],
        description="AQI and weather features",
        event_time="timestamp",  # Explicitly define timestamp column
    )
    
    fg.insert(df)

if __name__ == "__main__":
    # Fetch and process data
    raw_data = fetch_aqi_data()
    processed_df = process_data(raw_data)
    
    # Store in Feature Store
    store_features(processed_df)
    print("Data successfully stored in Hopsworks!")