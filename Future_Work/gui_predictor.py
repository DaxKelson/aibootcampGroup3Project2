import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("Model_Files/best_model_random_forest_balanced_hyperparameter_tuned.pkl")
scaler = joblib.load("Model_Files/scaler.pkl")  # Load the saved scaler

# Default feature values
default_features = np.array([
    3, 142.5, 0, 0.0, 48, 1052.255952, 0.380327, 0.619539, 0.32194, 0.0,
    1, 8, 6, 11, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
], dtype=np.float64)

# Feature names (ensure it matches your dataset)
feature_names = [
    'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 
    'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'OperatingSystems', 
    'Browser', 'Region', 'TrafficType', 'Weekend', 'VisitorType_New_Visitor', 'VisitorType_Other', 
    'VisitorType_Returning_Visitor', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 
    'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep'
]

# Binary features
binary_features = ['Weekend', 'VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor', 
                   'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 
                   'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep']

# Initialize Tkinter window
root = tk.Tk()
root.title("Ecommerce Purchase Predictor")

# Create a frame for the sliders
slider_frame = tk.Frame(root)
slider_frame.pack(pady=10)

# Create sliders with default values in a grid layout
sliders = []
checkboxes = []  # To store the checkboxes for binary features
checkbox_vars = []  # List to hold the IntVar for checkboxes
num_columns = 5  # Set the number of columns to 5

for i, (feature, default_value) in enumerate(zip(feature_names, default_features)):
    row, col = divmod(i, num_columns)  # Calculate grid position
    
    if feature in binary_features:  # If the feature is binary
        tk.Label(slider_frame, text=feature).grid(row=row * 2, column=col, padx=5, pady=2)  # Feature label
        
        # Create IntVar for checkbox to track value (0 or 1)
        checkbox_var = tk.IntVar(value=int(default_value))
        checkbox = tk.Checkbutton(slider_frame, text="1 (Yes)", variable=checkbox_var)
        checkbox.grid(row=row * 2 + 1, column=col, padx=5, pady=2)  # Place checkbox below label
        
        # Add checkbox and IntVar to respective lists
        checkboxes.append(checkbox)
        checkbox_vars.append(checkbox_var)
    else:  # For other features, use sliders
        tk.Label(slider_frame, text=feature).grid(row=row * 2, column=col, padx=5, pady=2)  # Feature label
        slider = tk.Scale(slider_frame, from_=0, to=1500, orient="horizontal")  # Adjust max value as needed
        slider.set(default_value)  # Set default value
        slider.grid(row=row * 2 + 1, column=col, padx=5, pady=2)  # Place slider below label
        sliders.append(slider)

def predict_purchase():
    try:
        # Collect user input from all sliders and checkboxes (28 feature values)
        user_input = np.array([slider.get() if isinstance(slider, tk.Scale) else checkbox_var.get() 
                               for slider, checkbox_var in zip(sliders, checkbox_vars)], dtype=np.float64)
        
        # For non-binary features, use sliders, for binary features, use checkboxes
        # Combine the input for binary and non-binary features
        user_input = np.concatenate([user_input, np.array([checkbox_var.get() for checkbox_var in checkbox_vars])])

        # Print user input before scaling
        print(f"Original user input: {user_input}")

        # Apply scaling only to the numerical columns (PageValues, BounceRates, ExitRates)
        num_cols_indices = [8, 6, 7]  # Indices for 'PageValues', 'BounceRates', 'ExitRates'
        user_input[num_cols_indices] = scaler.transform(user_input[num_cols_indices].reshape(1, -1))

        # Print scaled user input
        print(f"Scaled user input: {user_input}")

        # Predict purchase using the model
        prediction = model.predict(user_input.reshape(1, -1))[0]

        # Display result
        result_label.config(
            text="Will Purchase" if prediction == 1 else "Won't Purchase",
            fg="green" if prediction == 1 else "red"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict_purchase)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
