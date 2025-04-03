import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load your planet data
data = {
    "temperature": [15, -50, 22, -80, 30, 45, -10, 10, 20, -60],
    "distance_from_star": [1.0, 5.2, 0.9, 10.5, 1.2, 0.4, 2.3, 0.8, 1.1, 7.8],
    "oxygen_percent": [21, 0, 19, 0, 23, 10, 0, 20, 22, 0],
    "surface_gravity": [9.8, 22, 8.5, 25, 10, 3, 20, 9.5, 9.0, 30],
    "water_presence": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    "habitability": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# Prepare data
X = df[['temperature']]
y = df['habitability']

# Train Logistic Regression
model = LogisticRegression()
model.fit(X, y)

# Generate data for the sigmoid curve
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

# Plot the sigmoid curve with temperature unit
plt.figure(figsize=(8, 6))
plt.plot(x_range, y_prob, label='Sigmoid Curve', color='blue')
plt.scatter(X, y, color='red', label='Data Points')
plt.xlabel('Temperature (°C)')  # Added unit to the x-axis label
plt.ylabel('Probability of Habitability')
plt.title('Logistic Regression Sigmoid Curve (Temperature vs. Habitability)')
plt.legend()
plt.grid(True)
plt.show()