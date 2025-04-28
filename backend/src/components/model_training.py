import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

FEATURES_DIR = "data/features/"
MODELS_DIR = "models/"
STATS_DIR = "data/stats/"

def create_sequences(data, input_len=7, output_len=5, target_index=3):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len, :])
        y.append(data[i+input_len:i+input_len+output_len, target_index:target_index+1])
    return np.array(X), np.array(y)

def train_lstm_for_dam(dam_name: str, input_len=7, output_len=5, batch_size=64,
                       lr=0.001, epochs=300, use_val=True):
    file_path = os.path.join(FEATURES_DIR, f"{dam_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature-engineered file not found for dam: {dam_name}")
    
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    if use_val:
        n = len(df)
        train_df = df.iloc[:int(n*0.8)]
        val_df = df.iloc[int(n*0.8):]
        
        columns_to_normalize = ['tavg', 'rh_avg', 'rr', 'tma', 'slr', 'wx', 'wy', 'max_wx', 'max_wy',
                                'sin_day', 'cos_day', 'wavelet_ca3', 'wavelet_cd3', 'wavelet_cd2', 'wavelet_cd1']

        train_mean = train_df[columns_to_normalize].mean()
        train_std = train_df[columns_to_normalize].std()

        train_df.loc[:, columns_to_normalize] = (train_df.loc[:, columns_to_normalize] - train_mean) / train_std
        val_df.loc[:, columns_to_normalize] = (val_df.loc[:, columns_to_normalize] - train_mean) / train_std

        X_train, y_train = create_sequences(train_df.values, input_len, output_len, target_index=3)
        X_val, y_val = create_sequences(val_df.values, input_len, output_len, target_index=3)

    else:
        train_mean = df.mean()
        train_std = df.std()
        df[train_mean.index] = (df[train_mean.index] - train_mean) / train_std
        X_train, y_train = create_sequences(df.values, input_len, output_len, target_index=3)
        X_val, y_val = None, None

    os.makedirs(STATS_DIR, exist_ok=True)
    train_mean.to_frame().T.to_csv(os.path.join(STATS_DIR, f"{dam_name}_train_mean.csv"), index=False)
    train_std.to_frame().T.to_csv(os.path.join(STATS_DIR, f"{dam_name}_train_std.csv"), index=False)

    input_layer = Input(shape=(input_len, df.shape[1]))
    encoder = LSTM(128, activation='relu', kernel_regularizer=regularizers.l2(0.02))(input_layer)
    encoder = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(encoder)
    encoder = Dropout(0.4)(encoder)

    decoder = RepeatVector(output_len)(encoder)
    decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
    decoder = Dropout(0.4)(decoder)
    output = TimeDistributed(Dense(1))(decoder)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=RMSprop(learning_rate=lr), loss='mse')

    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, f"{dam_name}.keras")

    monitor_metric = 'val_loss' if use_val else 'loss'

    callbacks = [
        ReduceLROnPlateau(monitor=monitor_metric, factor=0.3, patience=15, cooldown=30, min_lr=1e-8),
        EarlyStopping(monitor=monitor_metric, patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor=monitor_metric, save_best_only=True, mode='min', verbose=1)
    ]

    if use_val:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    return {"message": f"Model trained with {'train/val split' if use_val else 'all data'} for {dam_name}", "model_path": checkpoint_path}