import os
import numpy as np
import joblib
import bentoml
from bentoml.io import JSON
import jwt
from datetime import datetime, timedelta

# Load scaler and model runner
scaler = joblib.load(os.path.join("data", "processed", "scaler.pkl"))
runner = bentoml.tensorflow.get("admissions_dense_nn:latest").to_runner()

# Define BentoML service
service = bentoml.Service("admissions_service", runners=[runner])

# Helper: create JWT token
def create_jwt():
    expire = datetime.utcnow() + timedelta(hours=1)
    token = jwt.encode({"exp": expire}, "secret", algorithm="HS256")
    return token if isinstance(token, str) else token.decode()

# Login endpoint: returns JWT
@service.api(input=JSON(), output=JSON(), route="/login")
def login(body: dict):
    user = body.get("username")
    pw = body.get("password")
    if user == "user123" and pw == "password123":
        return {"token": create_jwt()}
    raise bentoml.exceptions.BadInput("Invalid credentials")

# Function to extract features
def extract_features(body: dict) -> np.ndarray:
    feature_keys = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    try:
        values = [float(body[k]) for k in feature_keys]
    except KeyError as e:
        raise bentoml.exceptions.BadInput(f"Missing feature: {e.args[0]}")
    return np.array(values).reshape(1, -1)

# Predict endpoint: requires Bearer token
@service.api(input=JSON(), output=JSON())
async def predict(input_data):
    arr = np.array([[input_data[col] for col in ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]]])
    arr_scaled = scaler.transform(arr)
    result = await runner.async_run(arr_scaled)
    return {"prediction": float(result[0])}
