#!/usr/bin/env python3

import os
import numpy as np
import joblib
import bentoml
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException, BadInput
from http import HTTPStatus
import jwt
from datetime import datetime, timedelta
from bentoml import Context

# Custom exception for 401 Unauthorized
class Unauthorized(BentoMLException):
    error_code = HTTPStatus.UNAUTHORIZED

# Load scaler
scaler = joblib.load("data/processed/scaler.pkl")

# Load model runner
runner = bentoml.sklearn.get("admissions_linear:latest").to_runner()

# Define the BentoML service
svc = bentoml.Service("admissions_service", runners=[runner])

# JWT secret from env (fallback to "secret" for local dev/tests)
SECRET_KEY = os.getenv("JWT_SECRET", "secret")

def create_jwt():
    exp = datetime.utcnow() + timedelta(hours=1)
    return jwt.encode({"exp": exp}, SECRET_KEY, algorithm="HS256")

def verify_jwt(token: str):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise Unauthorized("Token expired")
    except jwt.InvalidTokenError:
        raise Unauthorized("Invalid token")

@svc.api(input=JSON(), output=JSON(), route="/login")
def login(body: dict):
    if body.get("username") == "user123" and body.get("password") == "password123":
        return {"token": create_jwt()}
    raise Unauthorized("Invalid credentials")

@svc.api(input=JSON(), output=JSON(), route="/predict")
async def predict(input_data: dict, ctx: Context):
    auth = ctx.request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise Unauthorized("Missing Bearer token")

    token = auth.split(" ", 1)[1]
    verify_jwt(token)

    feature_keys = [
        "GRE Score", "TOEFL Score", "University Rating",
        "SOP", "LOR", "CGPA", "Research",
    ]
    try:
        arr = np.array([[input_data[k] for k in feature_keys]])
    except KeyError as e:
        raise BadInput(f"Missing required feature: {e.args[0]}")

    arr_scaled = scaler.transform(arr)
    result = await runner.async_run(arr_scaled)
    return {"prediction": float(result[0])}
