import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()

# Load the dataset
df = pd.read_csv('TCS.csv')

# Preprocess the dataset (you may need to adjust this based on your specific dataset)
data = df[['Open', 'High', 'Low', 'Close']]

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Create sequences and targets
sequence_length = 10  # Adjust this window size as needed
X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    # Convert to binary classification: 1 if next closing price is higher, 0 otherwise
    y.append(1 if data[i+sequence_length, 3] > data[i+sequence_length-1, 3] else 0)

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification, so use sigmoid activation

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1)

# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred)

# Confusion matrix and classification report
confusion = confusion_matrix(y_test, y_pred_binary)
sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])

print("Confusion Matrix:")
print(confusion)
print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
print(f"Specificity (True Negative Rate): {specificity:.2f}")
