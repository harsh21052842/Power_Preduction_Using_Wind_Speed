import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


dataset = pd.read_excel('RNN_Training(Wind Speed).xlsx', skiprows=1)
# Replace 'your_dataset.xlsx' with the actual file path

# Drop the date column (assuming it's the first column)
dataset = dataset.iloc[:, 1:]

# Convert all values to numeric
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
dataset.dropna(inplace=True)

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset.values)

# Split data into input (X) and output (y)
X = scaled_data[:, :-1]  # Exclude the last column (target variable)
y = scaled_data[:, -1]   # Last column is the target variable (wind speed)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), shuffle=False)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions for the next day based on the current hour's data
current_hour_data = X_test[-1]  # Get the last hour's data from the test set
current_hour_data = current_hour_data.reshape((1, 1, current_hour_data.shape[1]))  # Reshape for LSTM input
next_day_prediction = model.predict(current_hour_data)

# Inverse transform the prediction to get the original wind speed scale
# Fit a separate scaler for the target variable
scaler_y = MinMaxScaler()
scaler_y.fit(dataset.values[:, -1].reshape(-1, 1))  # Reshape for single feature

# Replace the inverse transformation line with this
next_day_prediction = scaler_y.inverse_transform(next_day_prediction)

print("Predicted wind speed for the next day based on the current hour's data:", next_day_prediction)
