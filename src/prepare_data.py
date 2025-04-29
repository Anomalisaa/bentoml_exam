#!/usr/bin/env python3
# prepare_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths
RAW_PATH = "data/raw/admission.csv"
PROCESSED_DIR = "data/processed"

# Create directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load raw data
print("Loading data.")
df = pd.read_csv(RAW_PATH)
# Clean column names
df.columns = df.columns.str.strip()
# remove ID column
if "Serial No." in df.columns:
    df = df.drop(columns=["Serial No."])
# Remove duplicates and missing values
df = df.drop_duplicates().dropna()

# Separate features and target
TARGET = "Chance of Admit"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split into training and test sets
print("Splitting data into train and test sets.")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
print("Scaling features.")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and processed data
print("Saving scaler and processed data.")
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

# Save as CSV
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    os.path.join(PROCESSED_DIR, "X_train.csv"), index=False
)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    os.path.join(PROCESSED_DIR, "X_test.csv"), index=False
)
y_train.to_frame(name=TARGET).to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_frame(name=TARGET).to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

print("All files saved in:", PROCESSED_DIR)