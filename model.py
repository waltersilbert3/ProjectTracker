# model.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def prepare_data(data, time_step=60):
   """Prepare data for the LSTM model."""
   X, y = [], []
   for i in range(time_step, len(data)):
       X.append(data[i-time_step:i, 0])
       y.append(data[i, 0])
   return np.array(X), np.array(y)


def build_model():
   """Build the LSTM model."""
   model = Sequential()
   model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
   model.add(Dropout(0.2))
   model.add(LSTM(units=50, return_sequences=False))
   model.add(Dropout(0.2))
   model.add(Dense(units=1))  # Output layer for predicting the closing price
   model.compile(optimizer='adam', loss='mean_squared_error')
   return model


def train_model(X_train, y_train, epochs=10, batch_size=32):
   """Train the LSTM model."""
   model = build_model()
   model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
   return model


def make_prediction(model, X_test):
   """Make predictions using the trained model."""
   predictions = model.predict(X_test)
   return predictions
