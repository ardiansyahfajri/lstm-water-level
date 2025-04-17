import os
import math
import numpy as np
import pandas as pd
import pywt

PROCESSED_DIR = "data/processed/"
FEATURES_DIR = "data/features/"

DESIRED_COLUMN_ORDER = [
    'tma', 'tavg', 'rh_avg', 'rr', 'wx', 'wy', 'max_wx', 'max_wy',
    'sin_day', 'cos_day', 'slr',
    'wavelet_ca3', 'wavelet_cd3', 'wavelet_cd2', 'wavelet_cd1'
]

def load_processed_data(dam_name: str):
    file_path = os.path.join(PROCESSED_DIR, f"{dam_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed file not found for dam: {dam_name}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()
    df.index = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df.drop('date', axis=1, inplace=True)
    return df

def calculate_et0(tmean, rh, wind_speed, ss, altitude, latitude, doy):
    Gsc = 0.0820
    sigma = 4.903e-9
    p = 101.3 * ((293 - 0.0065 * altitude) / 293) ** 5.26
    gamma = 0.000665 * p
    es = 0.6108 * math.exp((17.27 * tmean) / (tmean + 237.3))
    ea = es * (rh / 100)
    delta = (4098 * es) / ((tmean + 237.3) ** 2)
    dr = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    delta_s = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    lat_rad = math.radians(latitude)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta_s))
    ra = (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(lat_rad) * math.sin(delta_s) +
        math.cos(lat_rad) * math.cos(delta_s) * math.sin(ws)
    )
    rso = (0.25 + 0.5 * ss) * ra
    albedo = 0.23
    rns = (1 - albedo) * rso
    f = (0.9 * ss) + 0.1
    eps = 0.34 - 0.14 * math.sqrt(ea)
    rnl = sigma * ((tmean + 273.16) ** 4) * f * eps
    rn = rns - rnl
    return round(rn, 2)

def apply_feature_engineering(dam_name: str):
    df = load_processed_data(dam_name)
    print("Raw columns:", df.columns.tolist())
    
    altitude = 791
    latitude = -6.88356
    
    df['doy'] = df.index.dayofyear
    
    df['slr'] = df.apply(lambda row: calculate_et0(
        tmean=row['tavg'],
        rh=row['rh_avg'],
        wind_speed=row['ff_avg'],
        ss=row['ss'] / 12,
        altitude=altitude,
        latitude=latitude,
        doy=row['doy']
    ), axis=1)
    
    df = df.drop(columns=['doy', 'ss'])

    wv = df.pop('ff_avg')
    max_wv = df.pop('ff_x')
    wd_rad = df.pop('ddd_x') * np.pi / 180

    df['wx'] = wv * np.cos(wd_rad)
    df['wy'] = wv * np.sin(wd_rad)
    df['max_wx'] = max_wv * np.cos(wd_rad)
    df['max_wy'] = max_wv * np.sin(wd_rad)


    df['doy'] = df.index.dayofyear
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['sin_day'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    df = df.drop(['doy', 'day', 'month', 'year'], axis=1)

    coeffs = pywt.wavedec(df['tma'], wavelet='db4', level=3)

    cA3_rec = pywt.upcoef('a', coeffs[0], 'db4', level=3, take=len(df['tma'])) 
    cD3_rec = pywt.upcoef('d', coeffs[1], 'db4', level=3, take=len(df['tma'])) 
    cD2_rec = pywt.upcoef('d', coeffs[2], 'db4', level=2, take=len(df['tma']))  
    cD1_rec = pywt.upcoef('d', coeffs[3], 'db4', level=1, take=len(df['tma']))  

    df['wavelet_ca3'] = cA3_rec  
    df['wavelet_cd3'] = cD3_rec  
    df['wavelet_cd2'] = cD2_rec  
    df['wavelet_cd1'] = cD1_rec 

    # Force correct column order
    # df = df[[col for col in DESIRED_COLUMN_ORDER if col in df.columns]]

    os.makedirs(FEATURES_DIR, exist_ok=True)
    feature_path = os.path.join(FEATURES_DIR, f"{dam_name}.csv")
    df.to_csv(feature_path)

    return df.columns.tolist()