# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Suppress potential convergence warnings for this simple dataset
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Step 2: Define and Load Simulated Data
# Simulated dataset with 5 features + 1 label
data = {
    "temperature": [15, -50, 22, -80, 30, 45, -10, 10, 20, -60],
    "distance_from_star": [1.0, 5.2, 0.9, 10.5, 1.2, 0.4, 2.3, 0.8, 1.1, 7.8],
    "oxygen_percent": [21, 0, 19, 0, 23, 10, 0, 20, 22, 0],
    "surface_gravity": [9.8, 22, 8.5, 25, 10, 3, 20, 9.5, 9.0, 30],
    "water_presence": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0],
    "habitability": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]  # 1 = Habitable, 0 = Not Habitable
}

# Convert data to DataFrame
df = pd.DataFrame(data)

print("--- Simulated Planet Data ---")
print(df)
print("-" * 30)

# Step 3: Splitting the Data
# Separate features (X) and the target label (y)
X = df.drop("habitability", axis=1)  # Features are all columns EXCEPT 'habitability'
y = df["habitability"]               # Label is the 'habitability' column

# Split data into training set (80%) and test set (20%)
# random_state ensures the split is the same every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Split ---")
print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print("-" * 30)

# Step 4: Building and Training the Model
# Create a Logistic Regression model instance
model = LogisticRegression()

# Train the model using the training data
# This is where the model "learns" the patterns
@ignore_warnings(category=ConvergenceWarning) # Added to hide warnings for this small dataset
def train_model():
    model.fit(X_train, y_train)

train_model()
print("--- Model Training ---")
print("Logistic Regression model trained successfully!")
print("-" * 30)

# Optional: Evaluate the model on the test set (Good practice, but maybe skip for demo brevity)
# accuracy = model.score(X_test, y_test)
# print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
# print("-" * 30)


# Step 5: Creating a Prediction Function
# Define function to predict habitability for new, hypothetical planet data
def predict_habitability(temp, dist, oxy, grav, water):
    # Create a NumPy array from the input features in the correct order
    input_data = np.array([[temp, dist, oxy, grav, water]])
    # Use the trained model to make a prediction
    prediction = model.predict(input_data)
    # Return a user-friendly string based on the prediction (0 or 1)
    return "Habitable 🌍" if prediction[0] == 1 else "Not Habitable ❄️"

# Step 6: Running Predictions (Demo)
print("--- Habitability Predictions ---")

# Demo: Predict for a few hypothetical exoplanets
# Example 1: Based loosely on a 'habitable' entry in the original data
print(f"Planet 1 (Temp: 22, Dist: 0.9, Oxy: 19, Grav: 8.5, Water: 1): {predict_habitability(22, 0.9, 19, 8.5, 1)}")

# Example 2: Based loosely on a 'non-habitable' entry
print(f"Planet 2 (Temp: -50, Dist: 5.2, Oxy: 0, Grav: 22, Water: 0): {predict_habitability(-50, 5.2, 0, 22, 0)}")

# Example 3: Another 'habitable' example
print(f"Planet 3 (Temp: 30, Dist: 1.2, Oxy: 23, Grav: 10, Water: 1): {predict_habitability(30, 1.2, 23, 10, 1)}")

# Example 4: A potentially borderline or different case
print(f"Planet 4 (Temp: 5, Dist: 1.5, Oxy: 5, Grav: 12, Water: 1): {predict_habitability(5, 1.5, 5, 12, 1)}")

print("-" * 30)

# Step 7: Instructions (Already covered - save as .py and run 'python exoplanet_predictor.py')
print("To run again, save this code as exoplanet_predictor.py and run 'python exoplanet_predictor.py' in your terminal.")