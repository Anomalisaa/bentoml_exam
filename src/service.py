import os
import numpy as np
import joblib
import bentoml
from bentoml.io import JSON
import jwt
from datetime import datetime, timedelta
from bentoml import Context
from starlette.exceptions import HTTPException
from bentoml.exceptions import BadInput

# scaler and runner
scaler = joblib.load(os.path.join("data", "processed", "scaler.pkl"))
runner = bentoml.sklearn.get("admissions_linear:latest").to_runner()

# BentoML service
#service = bentoml.Service("admissions_service", runners=[runner]) <-- old 500 error
svc = bentoml.Service("admissions_service", runners=[runner])

SECRET_KEY = "secret"

# JWT creation
def create_jwt():
    expire = datetime.utcnow() + timedelta(hours=1)
    return jwt.encode({"exp": expire}, SECRET_KEY, algorithm="HS256")

# JWT verification
def verify_jwt(token):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Login endpoint
@svc.api(input=JSON(), output=JSON(), route="/login")
def login(body: dict):
    user = body.get("username")
    pw = body.get("password")
    if user == "user123" and pw == "password123":
        return {"token": create_jwt()}
    raise BadInput("Invalid credentials")

# endpoint (JWT protected)
@svc.api(input=JSON(), output=JSON())
async def predict(input_data, ctx: Context):
    # JWT verification
    auth_header = ctx.request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.split(" ")[1]
    verify_jwt(token)

    # Data extraction
    feature_keys = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    try:
        arr = np.array([[input_data[col] for col in feature_keys]])
    except KeyError as e:
        raise BadInput(f"Missing feature: {e.args[0]}")

    # Scaling
    arr_scaled = scaler.transform(arr)

    # Prediction
    result = await runner.async_run(arr_scaled)
    return {"prediction": float(result[0])}