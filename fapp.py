from flask import Flask, request, render_template, Response
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import joblib

app = Flask(__name__)
def html_response(content):
    return f"<html><body>{content}</body></html>"

# Route for rendering HTML
@app.route('/')
def index():
    return html_response("<h1>Falsk sales prediction!</h1>")

if __name__ == '__main__':
    app.run(debug=True)
# Load your trained machine learning model using joblib
model_path = r"E:\Flask\09-09-2023-14-53-20-148.pkl"
model = joblib.load(model_path)

def adjust_date(input_date, days_to_add):
    return input_date + pd.DateOffset(days=days_to_add)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        store = int(request.form['store'])
        day_of_week = int(request.form['day_of_week'])
        open_status = int(request.form['open'])
        promo = int(request.form['promo'])
        state_holiday = int(request.form['state_holiday'])
        school_holiday = int(request.form['school_holiday'])
        
        # Continue converting other input features to their appropriate data types
        store_type = int(request.form['store_type'])
        assortment = int(request.form['assortment'])
        competition_distance = float(request.form['competition_distance'])
        competition_open_month = int(request.form['competition_open_month'])
        competition_open_year = int(request.form['competition_open_year'])
        promo2 = int(request.form['promo2'])
        promo2_since_week = int(request.form['promo2_since_week'])
        promo2_since_year = int(request.form['promo2_since_year'])
        promo_interval = request.form['promo_interval']
        weekday = int(request.form['weekday'])
        is_weekend = int(request.form['is_weekend'])
        sales_per_customer = float(request.form['sales_per_customer'])
        is_month_start = int(request.form['is_month_start'])
        is_month_middle = int(request.form['is_month_middle'])
        is_month_end = int(request.form['is_month_end'])
        
        # Handle promo_interval separately
        if promo_interval == "None":
            promo_interval = 0  # or any other default value that makes sense
        else:
            promo_interval = float(promo_interval)

        # Create a list of selected feature names that match the features used during training
        selected_feature_names = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                                'StoreType', 'Assortment', 'CompetitionDistance',
                                'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                                'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'weekday',
                                'is_weekend', 'Season', 'IsMonthStart',
                                'IsMonthMiddle', 'IsMonthEnd']

        # Create a dictionary to map feature names to their corresponding values
        input_features_dict = {
            'Store': store,
            'DayOfWeek': day_of_week,
            'Open': open_status,
            'Promo': promo,
            'StateHoliday': state_holiday,
            'SchoolHoliday': school_holiday,
            'StoreType': store_type,
            'Assortment': assortment,
            'CompetitionDistance': competition_distance,
            'CompetitionOpenSinceMonth': competition_open_month,
            'CompetitionOpenSinceYear': competition_open_year,
            'Promo2': promo2,
            'Promo2SinceWeek': promo2_since_week,
            'Promo2SinceYear': promo2_since_year,
            'PromoInterval': promo_interval,
            'weekday': weekday,
            'is_weekend': is_weekend,
            'SalesPerCustomer': sales_per_customer,  # Include your target feature
            'IsMonthStart': is_month_start,
            'IsMonthMiddle': is_month_middle,
            'IsMonthEnd':  is_month_end,
        }

        # Assuming "SalesPerCustomer" is your target value, remove it from input_features_dict
        input_features_dict.pop('SalesPerCustomer')

        # Create a dictionary to map feature names to their corresponding indices
        feature_indices = {feature: idx for idx, feature in enumerate(selected_feature_names)}

        # Create a numpy array to hold the input features
        input_features = np.zeros(len(selected_feature_names))

        # Populate the input_features array with values based on the feature indices
        for feature, value in input_features_dict.items():
            idx = feature_indices.get(feature)
            if idx is not None:
                input_features[idx] = value

        # Reshape the input_features for LSTM input (samples, time steps, features)
        input_features = input_features.reshape(1, 1, -1)

        # Initialize lists to store predictions for the next 6 months
        predictions_next_6_months = []

        # Define the start date for predictions (adjust as needed)
        start_date = pd.to_datetime('2023-09-01')

        # Loop to predict sales for the next 6 months
        for i in range(6):
            # Use your trained LSTM model to make predictions
            predicted_values = model.predict(input_features)
            predicted_value = predicted_values[0][0]
            
            # Add the prediction to the list
            predictions_next_6_months.append(predicted_value)

            # Update the input features for the next prediction (adjust date-related features)
            input_features[0][0][1] += 7  # Increment the day of the week by 7 days
            input_features[0][0][17] += 1  # Increment the month by 1
            input_features[0][0][19] += 1  # Increment the day of the month by 1

        # Create a list of dates for the next 6 months
        plot_dates = [start_date + pd.DateOffset(days=i) for i in range(6)]

        # Plot predictions
        plt.figure(figsize=(6, 4))  # Adjust plot size as needed
        plt.plot(plot_dates, predictions_next_6_days, marker='o', linestyle='-', color='green')  # Change color to green
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('ROSSMAN SALES PREDICTION FOR NEXT 6 Days')
        plt.grid(True)

        # Create a BytesIO buffer to hold the plot image
        plot_buffer = BytesIO()
        FigureCanvas(plt.gcf()).print_png(plot_buffer)
        plot_buffer.seek(0)

        # Encode the image in base64 for embedding in HTML
        plot_base64 = base64.b64encode(plot_buffer.read()).decode()

        
if __name__ == '__main__':
    app.run(debug=True)