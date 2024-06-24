import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import os

# Load the CSV data
file_path = 'Initial-Model-Creation/Cash4Life-Florida/data/parsed_c4l_fl-06_29_2024-02-20-2017.csv'
data = pd.read_csv(file_path)

# Drop the Date column as it is not needed for training
data.drop(columns=['Date'], inplace=True)

# Ensure the first 5 numbers are unique and between 1-60, and the 6th number is between 1-4
def validate_data(row):
    unique_numbers = set(row[:5])
    return len(unique_numbers) == 5 and all(1 <= num <= 60 for num in unique_numbers) and 1 <= row[5] <= 4

data = data[data.apply(validate_data, axis=1)]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Experiment with different sequence lengths
sequence_lengths = [5, 10, 15]  # Example sequence lengths to try
epoch_list = [1, 2, 3, 4, 5, 6]

# Define cross-validation parameters
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for seq_length in sequence_lengths:
    # Create sequences
    X, y = create_sequences(data_normalized, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    fold = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_dir = os.path.join("c4l_models",f"models_seq_{seq_length}_fold_{fold}")
        os.makedirs(model_dir, exist_ok=True)

        for epochs in epoch_list:
            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(6))

            model.compile(optimizer='adam', loss='mse')

            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            csv_logger = CSVLogger(os.path.join(model_dir, f"training_log_seq_{seq_length}_epochs_{epochs}.csv"))

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping, csv_logger])

            early_stopped = len(history.epoch) < epochs

            model_path = os.path.join(model_dir, f"model_seq_{seq_length}_epochs_{epochs}.h5")
            model.save(model_path)

            css_content = f"Model sequence length: {seq_length}\n"
            css_content += f"Number of epochs: {epochs}\n"
            css_content += f"Early stopping: {'Yes' if early_stopped else 'No'}\n"
            css_path = os.path.join(model_dir, f"css_seq_{seq_length}_epochs_{epochs}.txt")
            with open(css_path, "w") as f:
                f.write(css_content)
        
        fold += 1

print("Cross-validated models with different sequence lengths and early stopping have been trained and saved.")
