import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load and preprocess the dataset
df = pd.read_csv('crop_yield.csv')

# Select features and target variable
X = df[['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop', 'Season', 'State']]
y = df['Yield']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['Crop', 'Season', 'State'])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_encoded, y)

# Save the model and the column names
joblib.dump(model, 'crop_yield_model.pkl')
joblib.dump(X_encoded.columns, 'model_columns.pkl')
