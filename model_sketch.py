# model_sketch.py
# AI Model to Predict Crop Yield Based on IoT Sensor Data

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Example dummy dataset (normally collected via sensors)
data = pd.DataFrame({
    'soil_moisture': [20, 45, 60, 35, 50],
    'temperature': [30, 25, 28, 27, 29],
    'sunlight': [200, 250, 180, 220, 240],
    'crop_yield': [2.0, 3.5, 4.0, 3.0, 3.8]  # in tons/hectare
})

# Features and target
X = data[['soil_moisture', 'temperature', 'sunlight']]
y = data['crop_yield']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict on new data
new_sensor_input = [[40, 28, 230]]  # [soil_moisture, temperature, sunlight]
prediction = model.predict(new_sensor_input)

print("🌾 Predicted crop yield:", prediction[0], "tons/hectare")
