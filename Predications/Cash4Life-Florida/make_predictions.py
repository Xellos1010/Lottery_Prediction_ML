import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to load models and make predictions
def load_and_predict(model_dir, X_test):
    predictions = []
    for model_file in sorted(os.listdir(model_dir)):
        if model_file.endswith(".h5"):
            model_path = os.path.join(model_dir, model_file)
            model = tf.keras.models.load_model(model_path)
            prediction = model.predict(X_test)
            predictions.append(prediction)
    return predictions

# Function to de-normalize predictions
def denormalize_predictions(predictions, scaler):
    predictions_denorm = scaler.inverse_transform(predictions)
    for pred in predictions_denorm:
        pred[:5] = np.round(pred[:5])
        pred[5] = np.round(pred[5])
    return predictions_denorm

# Load the CSV data and preprocess it similarly as before
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

# Use the same sequence lengths and prepare test data
sequence_lengths = [5, 10, 15]  # Example sequence lengths used previously

# Prepare predictions for each sequence length
all_predictions = {}

base_dir = 'c4l_models'

for seq_length in sequence_lengths:
    X, y = create_sequences(data_normalized, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Iterate over each fold directory
    for fold_dir in os.listdir(base_dir):
        if f"models_seq_{seq_length}" in fold_dir:
            model_dir = os.path.join(base_dir, fold_dir)
            predictions = load_and_predict(model_dir, X_test)
            if seq_length not in all_predictions:
                all_predictions[seq_length] = []
            all_predictions[seq_length].extend(predictions)

# Save the predictions to a dated file
today = datetime.today().strftime('%Y-%m-%d')
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f'predictions_{today}.csv')

# Save predictions to CSV
with open(output_file, 'w') as f:
    f.write("Sequence Length, Model Epoch, Prediction\n")
    for seq_length, preds in all_predictions.items():
        for i, pred in enumerate(preds):
            pred_denorm = denormalize_predictions(pred, scaler)
            f.write(f"{seq_length}, Model {i+1}, {','.join(map(str, pred_denorm[0]))}\n")

print(f"Predictions have been saved to {output_file}")
