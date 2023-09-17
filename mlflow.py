import os
import mlflow
print(mlflow.__version__)

import warnings
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mypro
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Load your merged train and test data
train_data_merged = pd.read_csv(r"C:\Users\manil\ROSSMAN_SALES_PREDICTION_BY_MANILA\train_df.csv")
test_data_merged = pd.read_csv(r"C:\Users\manil\ROSSMAN_SALES_PREDICTION_BY_MANILA\test_df.csv")

print(test_data_merged.columns)
train_data_merged.drop('SalesClass', axis=1, inplace=True)

train_data_merged['Date'] = pd.to_datetime(train_data_merged['Date'])
test_data_merged['Date'] = pd.to_datetime(test_data_merged['Date'])

print(train_data_merged.info())
print(test_data_merged.info())

# Define X and y
X = train_data_merged[['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                      'StoreType', 'Assortment', 'CompetitionDistance',
                      'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                      'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'weekday',
                      'is_weekend', 'Season', 'IsBeginningOfMonth',
                      'IsMidOfMonth', 'IsEndOfMonth', 'DaysToHoliday', 'DaysAfterHoliday']]
y = train_data_merged['SalesPerCustomer']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Initialize MLflow
mlflow.set_experiment('Sales_Prediction')

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
# Start an MLflow run
with mlflow.start_run(run_name='LSTM'):
    # Train the LSTM model
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_data=(X_test_lstm, y_test))
    y_pred = lstm_model.predict(X_test_lstm)
    mse = mean_squared_error(y_test, y_pred)

    # Infer and set the model signature
    signature = infer_signature(X_train_lstm, lstm_model.predict(X_train_lstm))
    mlflow.keras.log_model(lstm_model, 'LSTM_model', signature=signature)

    # Log the parameters and metrics
    mlflow.log_params({'epochs': 10, 'batch_size': 64})
    mlflow.log_metric('mse', mse)

# Save the LSTM model with a timestamp
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
model_filename_with_timestamp = f'lstm_model_{timestamp}.pkl'
with open(os.path.join('saved_models', model_filename_with_timestamp), 'wb') as f:
    pickle.dump(lstm_model, f)

print('LSTM model saved successfully.')

# Preprocess the test data
test_data_merged['Date'] = pd.to_datetime(test_data_merged['Date'])

# Extract the features from the test data (similar to what you did for training data)
X_test = test_data_merged[['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
               'StoreType', 'Assortment', 'CompetitionDistance',
               'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
               'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'weekday',
               'is_weekend', 'Season', 'IsBeginningOfMonth',
               'IsMidOfMonth', 'IsEndOfMonth', 'DaysToHoliday', 'DaysAfterHoliday']]

# Normalize the input features using Min-Max scaling
X_test_scaled = scaler.transform(X_test)

# Reshape the data for LSTM input (samples, time steps, features)
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Use the trained LSTM model to make predictions
y_test_pred = lstm_model.predict(X_test_lstm)

# Add the Sales Prediction column to the test data
test_data_merged['SalesPrediction'] = y_test_pred

# Save the test data with the Sales Prediction column
test_data_merged.to_csv("test_data_with_predictions.csv", index=False)
print('Test data with Sales Prediction saved successfully.')

# Specify the experiment's name or ID where you want to log the artifact
experiment_name_or_id = 'Sales_Prediction'

# Set the MLflow tracking URI (replace with your MLflow server URL)
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URL


# Set the artifact URI to store artifacts in S3 (if desired)
mlflow.tracking.artifact_uri = "s3://your-s3-bucket/path/to/artifacts"

# Log the test data with predictions as an artifact
mlflow.log_artifact("test_data_with_predictions.csv")

# Generate plots to visualize the predictions
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(test_data_merged['Date'], test_data_merged['SalesPrediction'], label='Predicted Sales', color='blue')
plt.title('Predicted Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.legend()
plt.grid(True)

# Save the plot as an image file (optional)
plt.savefig("predicted_sales_plot.png")

# Log the image and CSV file as artifacts with the correct paths
mlflow.log_artifact("predicted_sales_plot.png", artifact_path="plots")

# Show the plot
plt.show()
