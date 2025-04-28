import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# paths
RAW_DATA_PATH = os.path.join("data", "raw", "admission.csv")
PROCESSED_DIR = os.path.join("data", "processed")

# processed data directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

# Drop ID
df = df.drop(columns=['Serial No.'], errors='ignore')

# Remove duplicates and missing values
df = df.drop_duplicates()
df = df.dropna()

# Separate features and target
target = 'Chance of Admit'
X = df.drop(columns=[target])
y = df[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# save scaler for later use
joblib.dump(scaler, os.path.join(PROCESSED_DIR, 'scaler.pkl'))

# 9. Save processed data
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)

print("Data processed and saved in data/processed:")
print(f"- X_train: {X_train.shape}\n- X_test: {X_test.shape}\n- y_train: {y_train.shape}\n- y_test: {y_test.shape}")