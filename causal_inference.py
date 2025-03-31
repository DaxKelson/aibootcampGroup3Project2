from sklearn.linear_model import LogisticRegression
from causalinference import CausalModel
import pandas as pd
from skimage.filters import threshold_otsu
import numpy as np

#Import CSV
online_shopping_df = pd.read_csv('online_shoppers_intention.csv')

#convert strings to integers
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 
          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
online_shopping_df["Month"] = online_shopping_df["Month"].map(months)

visitor = {'New_Visitor': 1, 'Returning_Visitor': 1, 'Other': 2}
online_shopping_df["VisitorType"] = online_shopping_df["VisitorType"].map(visitor)

#In order to use a column that is continuous we need to use a sigmoid to divide it into two classes
# Apply Sigmoid Transformation to PageValues
online_shopping_df['PageValues_Sigmoid'] = 1 / (1 + np.exp(-online_shopping_df['PageValues'].values))

# Find the best threshold using Otsu's method
threshold = threshold_otsu(online_shopping_df['PageValues_Sigmoid'].values)
online_shopping_df['High_PageValues'] = (online_shopping_df['PageValues_Sigmoid'].values > threshold).astype(int)

# Define treatment (High_PageValues)
treatment = online_shopping_df['High_PageValues']

# Define confounders. 
# Possible confounders have high correlations with the treatment as well as the outcome
confounders = online_shopping_df[['TrafficType', 'VisitorType', 'SpecialDay', 'Region']]

# Fit Propensity Score Model
psm = LogisticRegression().fit(confounders.values, treatment.values.ravel())
online_shopping_df['propensity_score'] = psm.predict_proba(confounders.values)[:, 1]

# Define outcome (Y), treatment (D), and covariates (X)
Y = online_shopping_df['Revenue'].astype(int).values.ravel()  # Ensure 1D array
D = online_shopping_df['High_PageValues'].astype(int).values.ravel()  # Ensure 1D array

X = online_shopping_df[['TrafficType', 'VisitorType', 'SpecialDay', 'Region']].values

print(f"Y shape: {Y.shape}, D shape: {D.shape}, X shape: {X.shape}")

# Create Causal Model
causal = CausalModel(Y, D, X)

# Estimate treatment effect using matching
causal.est_via_matching()
print(causal.estimates)

#Since all columns dont correlate very highly with revenue. We need to seek other methods for selecting confounders