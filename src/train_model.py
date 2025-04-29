#!/usr/bin/env python3
# train_model.py

import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import bentoml

# Directory for processed data
PROCESSED_DIR = "data/processed"

# Load processed data
print("Loading processed data.")
X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()

# Linear Regression model
print("Training Linear Regression model.")
model = LinearRegression()
model.fit(X_train, y_train)

# model performance
print("Evaluating model.")
predictions = model.predict(X_test)
print(f"R^2 score: {r2_score(y_test, predictions):.4f}")
print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False):.4f}")

# Save model to BentoML
print("Saving model with BentoML.")
saved_model = bentoml.sklearn.save_model(
    "admissions_linear",
    model,
    signatures={"predict": {"batchable": True, "batch_dim": 0}}
)
print("Model saved as:", saved_model)
