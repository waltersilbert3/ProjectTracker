# routes.py
from flask import Flask, request, jsonify
from .data import fetch_stock_data, preprocess_data
from .model import prepare_data, train_model, make_prediction
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
   """Handle requests to predict stock prices."""
   data = request.json
   ticker = data.get('ticker', 'AAPL')  # Default to Apple if no ticker is provided
  
   # Step 1: Fetch and preprocess the stock data
   stock_data = fetch_stock_data(ticker)
   preprocessed_data = preprocess_data(stock_data)
  
   # Step 2: Scale and prepare the data for LSTM
   scaler = MinMaxScaler(feature_range=(0, 1))
   scaled_data = scaler.fit_transform(preprocessed_data['Close'].values.reshape(-1, 1))
   X, y = prepare_data(scaled_data)


   X = X.reshape(X.shape[0], X.shape[1], 1)


   # Step 3: Train the LSTM model
   model = train_model(X, y)
  
   # Step 4: Make a prediction for the next day (using the last 60 data points)
   X_test = np.array([scaled_data[-60:]]).reshape(1, 60, 1)
   predicted_price = make_prediction(model, X_test)


   # Step 5: Invert the scaling to get the actual predicted price
   predicted_price = scaler.inverse_transform(predicted_price)
  
   return jsonify({'prediction': predicted_price[0][0]})






