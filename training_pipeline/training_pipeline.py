import hopsworks
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv

load_dotenv()

def train_model():
    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()

    # Retrieve Feature View
    try:
        feature_view = fs.get_feature_view(name="aqi_feature_view", version=1)
    except:
        # If Feature View doesn't exist, create it
        fg = fs.get_feature_group(name="aqi_features", version=1)
        feature_view = fs.create_feature_view(
            name="aqi_feature_view",
            version=1,
            query=fg.select_all()
        )

    # Create training dataset if it doesn't exist
    try:
        df, _ = feature_view.get_training_data(training_dataset_version=1)
    except:
        print("Training dataset not found. Creating a new one...")
        feature_view.create_training_data(
            description="Training dataset for AQI prediction",
            data_format="csv",
            write_options={"wait_for_job": True}
        )
        # Fetch the newly created training dataset
        df, _ = feature_view.get_training_data(training_dataset_version=1)

    # Prepare data
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month

    label_encoder = LabelEncoder()
    df['dominant_pollutant_encoded'] = label_encoder.fit_transform(df['dominant_pollutant'])

    y = df["aqi"]
    X = df.drop(columns=["aqi", "timestamp", "city", "city_name", "latitude", "location", "longitude", "dominant_pollutant"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Predefined best parameters
    best_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'rmse'
    }

    # Train XGBoost model
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Hyperparameters: {best_params}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    # Save to Model Registry
    mr = project.get_model_registry()
    model_dir = "aqi_model"
    os.makedirs(model_dir, exist_ok=True)

    # Save model files
    model.save_model(os.path.join(model_dir, "model.json"))

    # Create model entry with numeric metrics
    model_meta = mr.python.create_model(
        name="aqi_xgboost_tuned",
        metrics={
            "mae": float(mae), 
            "rmse": float(rmse),
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8
        },
        description="XGBoost model for AQI prediction"
    )
    model_meta.save(model_dir)

    print("Model saved to Hopsworks Model Registry!")

if __name__ == "__main__":
    train_model()