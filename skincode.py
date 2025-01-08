import pandas as pd
import numpy as np
import serial
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Collection
# Configure the serial connection to Arduino
arduino_port = 'COM11'  # Update this to the correct port
baud_rate = 9600
file_name = 'pressure_data.csv'

# Open serial connection to Arduino
ser = serial.Serial(arduino_port, baud_rate, timeout=0.01)

# Open file to log data
with open(file_name, 'w') as file:
    file.write('Timestamp,Pressure Value (V)\n')  # Write CSV header
    
    # Collect and log data for 60 seconds
    start_time = time.time()
    duration = 60  # Collect data for 60 seconds
    while time.time() - start_time < duration:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                pressure_value = float(line)
                timestamp = time.strftime('%H:%M:%S')
                file.write(f'{timestamp},{pressure_value}\n')
                print(f'Time: {timestamp}, Pressure Value: {pressure_value}')
            except ValueError:
                continue

# Close the serial connection
ser.close()

# Step 2: Data Preprocessing
# Load the data
data = pd.read_csv(file_name)

# Check for missing values and drop them
data.dropna(inplace=True)

# Convert the 'Timestamp' to a datetime object
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%H:%M:%S')

# Create a target variable for ulcer risk (1 = risk, 0 = no risk)
data['Risk'] = np.where(data['Pressure Value (V)'] > 700, 1, 0)

# Step 3: Feature Engineering
# Add moving averages and other features
data['Pressure_MA'] = data['Pressure Value (V)'].rolling(window=5).mean()
data['Pressure_MA'].fillna(method='bfill', inplace=True)

# Extract hour and minute for time-based features
data['Hour'] = data['Timestamp'].dt.hour
data['Minute'] = data['Timestamp'].dt.minute

# Step 4: Model Training
# Define features and target variable
X = data[['Pressure Value (V)', 'Pressure_MA', 'Hour', 'Minute']]
y = data['Risk']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Making Predictions and Alerts
def predict_risk(pressure_value, model, data):
    # Calculate moving average manually for the incoming value
    if len(data) >= 4:
        pressure_ma = (pressure_value + data['Pressure Value (V)'].iloc[-4:].sum()) / 5
    else:
        pressure_ma = pressure_value  # If not enough data, use incoming value as MA
    
    incoming_data = pd.DataFrame({
        'Pressure Value (V)': [pressure_value],
        'Pressure_MA': [pressure_ma],  # Use calculated moving average
        'Hour': [pd.Timestamp.now().hour],
        'Minute': [pd.Timestamp.now().minute]
    })
    
    risk_prediction = model.predict(incoming_data)
    return risk_prediction[0]  # Return the prediction (1 or 0)

# Example incoming pressure value
incoming_pressure_value = 750  # Simulate incoming pressure value
risk = predict_risk(incoming_pressure_value, model, data)

if risk == 1:
    print("Alert: High pressure detected! Risk of ulcer.")
else:
    print("Pressure is within safe limits.")

# Plotting Pressure Values and Moving Average
plt.figure(figsize=(14, 7))
plt.plot(data['Timestamp'], data['Pressure Value (V)'], label='Pressure Value (V)', color='blue', alpha=0.6)
plt.plot(data['Timestamp'], data['Pressure_MA'], label='Moving Average', color='orange', linestyle='--')
plt.title('Pressure Values Over Time')
plt.xlabel('Time')
plt.ylabel('Pressure Value (V)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Adding predictions to the data for visualization
data['Risk Prediction'] = model.predict(X)

plt.figure(figsize=(14, 7))
plt.plot(data['Timestamp'], data['Pressure Value (V)'], label='Pressure Value (V)', color='blue', alpha=0.6)
plt.scatter(data['Timestamp'], data['Risk Prediction'] * 800, label='Risk Prediction', color='red', marker='o', alpha=0.5)
plt.title('Pressure Values and Risk Predictions Over Time')
plt.xlabel('Time')
plt.ylabel('Pressure Value (V)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()