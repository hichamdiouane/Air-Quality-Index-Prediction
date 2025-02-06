import streamlit as st
import hopsworks
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from xgboost import XGBRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests

load_dotenv()

AQICN_API_TOKEN = os.getenv("AQICN_API_TOKEN")

def get_current_aqi():
    url = f"https://api.waqi.info/feed/A518986/?token={AQICN_API_TOKEN}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")
    
    return response.json()['data'].get("aqi", 0)

def load_model():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    mr = project.get_model_registry()
    model = mr.get_model("aqi_xgboost_tuned", version=1)
    model_dir = model.download()
    xgb_model = XGBRegressor()
    xgb_model.load_model(os.path.join(model_dir, "model.json"))
    return xgb_model

def get_latest_features():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(name="aqi_feature_view", version=1)
    df = feature_view.get_batch_data(1)
    return df

def prepare_features(df):
    # Print feature columns for debugging
    print("Available columns:", df.columns.tolist())
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    
    required_columns = ['hour', 'day_of_week', 'month', 'dominant_pollutant_encoded']
    
    # Handle dominant_pollutant encoding
    if 'dominant_pollutant' in df.columns:
        label_encoder = LabelEncoder()
        df['dominant_pollutant_encoded'] = label_encoder.fit_transform(df['dominant_pollutant'])
    
    # Drop unnecessary columns
    drop_columns = ['aqi', 'timestamp', 'city', 'city_name', 'latitude', 'location', 'longitude', 'dominant_pollutant']
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    
    # Ensure all required columns are present
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    print("Feature shape after preprocessing:", X_scaled.shape)
    return X_scaled

def predict_next_3_days(model, latest_features):
    X = prepare_features(latest_features)
    predictions = []
    current_features = X.copy()
    
    for _ in range(3):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        current_features = current_features * 0.95
    
    return predictions

def main():
    st.title("🌍 AQI Prediction Dashboard")
    
    with st.spinner("Loading model and data..."):
        try:
            model = load_model()
            features_df = get_latest_features()
            current_aqi = get_current_aqi()
            
            if features_df.empty:
                st.error("No data available in Feature Store")
                return
                
            latest_aqi = features_df["aqi"].values[0]
            predictions = predict_next_3_days(model, features_df)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

    st.subheader("Current Air Quality")
    st.metric(label="Current AQI", value=current_aqi)

    st.subheader("3-Day Forecast")
    days = ["Tomorrow", "Day 2", "Day 3"]
    cols = st.columns(3)
    for col, day, pred in zip(cols, days, predictions):
        with col:
            st.metric(label=day, value=f"{pred:.1f}")

    fig, ax = plt.subplots()
    ax.plot([0] + list(range(1,4)), 
            [latest_aqi] + predictions, 
            marker="o", 
            linestyle="--")
    ax.set_title("AQI Forecast Trend")
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["Today"] + days)
    ax.set_ylabel("AQI")
    st.pyplot(fig)

if __name__ == "__main__":
    main()