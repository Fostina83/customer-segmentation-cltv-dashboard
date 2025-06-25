# predict_cltv.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("segmented_customers.csv")

# Select features
features = [
    'Recency', 'Frequency', 'Income',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'Education', 'Marital_Status'
]
target = 'Total_Spend'

X = df[features]
y = df[target]

# Preprocess categorical features
categorical = ['Education', 'Marital_Status']
numeric = list(set(features) - set(categorical))

# Pipeline with encoding + model
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical)
    ], remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Optional: Save model using joblib
import joblib
joblib.dump(model, "cltv_model.pkl")
print("Model saved as cltv_model.pkl")
