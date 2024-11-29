# PATIENT_DATASET_ZOBIA-AHMED-330-LAB-06
# HOME TASK ZOBIA AHMED / 2022F-BSE-330 CODE:

#ZOBIA AHMED / 2022F-BSE-330 / LAB 06 / HOME TASK:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
df = pd.read_csv("patient_data_zobia.csv")
# Define features (X) and target (y)
# Selecting numerical columns only for regression
X = df[['Age', 'BMI', 'Blood Pressure', 'Physical Activity Level']]
y = df['Glucose Level']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 06 / HOME TASK:\n")
print("Model Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score (Accuracy): {r2:.2f}")
# Visualize actual vs predicted glucose levels
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted Glucose Levels')
plt.xlabel('Actual Glucose Level')
plt.ylabel('Predicted Glucose Level')
plt.grid(True)
plt.show()
