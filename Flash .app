from flask import Flask, render_template
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)

# Load the LSTM model
lstm_model = load_model("lstm_model_retrained.h5")

@app.route("/")
def index():
    # Define the date from which you want to predict (2015-09-17)
    start_date = datetime(2015, 9, 17)

    # Create a list of dates for the next 6 weeks
    date_list = [start_date + timedelta(weeks=i) for i in range(6)]

    # Initialize X_lstm for predictions
    X_lstm = np.zeros((1, 1, 23))  # Initialize with zeros or actual data

    # Initialize an empty list to store the predicted sales
    predicted_sales = []

    # Predict sales for the next 6 weeks and format them with 6 decimal places
    for week_number in range(1, 7):
        # Predict for the next week
        next_week_sales = lstm_model.predict(X_lstm[-1].reshape(1, 1, -1))[0][0]
        formatted_sales = f'{next_week_sales:.6f}'  # Format with 6 decimal places
        predicted_sales.append(formatted_sales)
        
        # Update the input data for the next prediction
        X_lstm = np.append(X_lstm, X_lstm[-1:], axis=0)
        X_lstm[-1][0][-1] = next_week_sales  # Update the last feature with the predicted sales

    
    predictions_df = pd.DataFrame({'Date': date_list, 'Week': range(1, 7), 'PredictedSales': predicted_sales})

    
    predictions_df_from_csv = pd.read_csv("E:\OneDrive\Desktop\rossamann")
    return render_template("index.html", predictions=predictions_df_from_csv.to_html(index=False))

if __name__ == "__main__":
    app.run(debug=True)


