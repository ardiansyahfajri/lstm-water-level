import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.components.feature_engineering import apply_feature_engineering

STATS_DIR = "data/stats/"
MODELS_DIR = "models/"
UPLOAD_DIR = "data/uploads/"
TEMP_PROCESSED_DIR = "data/processed/"
TEMP_FEATURES_DIR = "data/features/"

EXPECTED_ORDER = [
    'tavg', 'rh_avg', 'rr', 'tma', 'slr', 'wx', 'wy', 'max_wx', 'max_wy',
    'sin_day', 'cos_day', 'wavelet_ca3', 'wavelet_cd3', 'wavelet_cd2', 'wavelet_cd1'
]

def predict_next_5_days(dam_name: str, raw_file_path: str, n_iter: int = 100):
    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Raw uploaded file not found: {raw_file_path}")

    temp_processed_path = os.path.join(TEMP_PROCESSED_DIR, f"{dam_name}.csv")
    os.makedirs(TEMP_PROCESSED_DIR, exist_ok=True)

    df_raw = pd.read_csv(raw_file_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'], format='%m/%d/%Y', errors='coerce')
    if df_raw['date'].isna().any():
        raise ValueError("Some dates could not be parsed. Please ensure they are in MM/DD/YYYY format.")
    df_raw.to_csv(temp_processed_path, index=False)

    apply_feature_engineering(dam_name)

    temp_feature_path = os.path.join(TEMP_FEATURES_DIR, f"{dam_name}.csv")
    df = pd.read_csv(temp_feature_path)

    # Enforce correct column order for model input
    df = df[[col for col in EXPECTED_ORDER if col in df.columns]]
    print("Final inference feature order:", df.columns.tolist())

    df = df.select_dtypes(include=[np.number]).astype(np.float32)

    if df.shape[0] < 7:
        raise ValueError("Engineered data must contain at least 7 rows")
    X_input = df.tail(7)

    mean_path = os.path.join(STATS_DIR, f"{dam_name}_train_mean.csv")
    std_path = os.path.join(STATS_DIR, f"{dam_name}_train_std.csv")
    train_mean = pd.read_csv(mean_path)
    train_std = pd.read_csv(std_path)
    columns = train_mean.columns
    mean_values = train_mean.iloc[0]
    std_values = train_std.iloc[0]
    X_input[columns] = (X_input[columns] - mean_values) / std_values

    X_input = X_input.values.reshape((1, 7, df.shape[1]))

    model_path = os.path.join(MODELS_DIR, f"{dam_name}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found for dam: " + dam_name)
    model = tf.keras.models.load_model(model_path)

    predictions = []
    for _ in range(n_iter):
        pred = model(X_input, training=True).numpy().flatten()
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0).flatten()
    std_pred = np.std(predictions, axis=0).flatten()

    tma_mean = mean_values['tma']
    tma_std = std_values['tma']
    mean_rescaled = mean_pred * tma_std + tma_mean
    upper = mean_rescaled + 1.96 * std_pred * tma_std
    lower = mean_rescaled - 1.96 * std_pred * tma_std

    last_date = df_raw['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5)

    result = {
        "dam_name": dam_name,
        "forecast": {
            str(date.date()): {
                "mean": round(m, 2),
                "lower": round(l, 2),
                "upper": round(u, 2)
            }
            for date, m, l, u in zip(future_dates, mean_rescaled, lower, upper)
        }
    }

    return result