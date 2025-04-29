# tests/test_api.py

import os
import time
import subprocess
import pytest
import requests
import jwt
from datetime import datetime, timedelta

# === Configuration constants ===
IMAGE_TAG = "admissions_service:latest"
HOST = "127.0.0.1"
PORT = 3000
BASE_URL = f"http://{HOST}:{PORT}"

# Hard-coded credentials (must match service.py)
USERNAME = "user123"
PASSWORD = "password123"

# JWT secret (in service.py read via os.getenv("JWT_SECRET", "secret"))
SECRET_KEY = "secret"

@pytest.fixture(scope="session", autouse=True)
def docker_container():
    """
    Start the Docker container with the JWT_SECRET env var set,
    then tear it down after all tests.
    """
    cmd = [
        "docker", "run", "--rm",
        # inject the JWT secret so service.py and tests agree
        "-e", f"JWT_SECRET={SECRET_KEY}",
        "-p", f"{PORT}:{PORT}",
        "--name", "admissions_test",
        IMAGE_TAG,
    ]
    proc = subprocess.Popen(cmd)
    # give the service time to start
    time.sleep(5)
    yield
    proc.terminate()
    proc.wait()

def test_invalid_login():
    """Wrong credentials should return 401 Unauthorized"""
    resp = requests.post(
        f"{BASE_URL}/login",
        json={"username": "wrong", "password": "wrong"},
    )
    assert resp.status_code == 401

def test_valid_login_returns_token():
    """Correct credentials should return a JWT token"""
    resp = requests.post(
        f"{BASE_URL}/login",
        json={"username": USERNAME, "password": PASSWORD},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "token" in body and isinstance(body["token"], str)
    # verify it's a JWT with an exp claim
    decoded = jwt.decode(body["token"], options={"verify_signature": False})
    assert "exp" in decoded

@pytest.fixture
def valid_token():
    """Fetch a valid JWT from the API for use in prediction tests"""
    resp = requests.post(
        f"{BASE_URL}/login",
        json={"username": USERNAME, "password": PASSWORD},
    )
    return resp.json()["token"]

def test_predict_without_token():
    """Calling /predict without Authorization header should return 401"""
    resp = requests.post(f"{BASE_URL}/predict", json={})
    assert resp.status_code == 401

def test_predict_with_invalid_token():
    """Calling /predict with a malformed token should return 401"""
    headers = {"Authorization": "Bearer not_a_real_token"}
    resp = requests.post(f"{BASE_URL}/predict", headers=headers, json={})
    assert resp.status_code == 401

def test_predict_with_expired_token():
    """Calling /predict with an expired token should return 401"""
    expired_token = jwt.encode(
        {"exp": datetime.utcnow() - timedelta(hours=1)},
        SECRET_KEY,
        algorithm="HS256",
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    resp = requests.post(f"{BASE_URL}/predict", headers=headers, json={})
    assert resp.status_code == 401

def test_predict_with_malformed_input(valid_token):
    """Missing required feature → 400 Bad Request"""
    headers = {"Authorization": f"Bearer {valid_token}"}
    bad_payload = {
        # omit "GRE Score"
        "TOEFL Score": 110,
        "University Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 9.1,
        "Research": 1,
    }
    resp = requests.post(f"{BASE_URL}/predict", headers=headers, json=bad_payload)
    assert resp.status_code == 400

def test_predict_success(valid_token):
    """Valid token & valid payload → returns a float between 0 and 1"""
    headers = {"Authorization": f"Bearer {valid_token}"}
    payload = {
        "GRE Score": 320,
        "TOEFL Score": 110,
        "University Rating": 4,
        "SOP": 4.5,
        "LOR": 4.0,
        "CGPA": 9.1,
        "Research": 1,
    }
    resp = requests.post(f"{BASE_URL}/predict", headers=headers, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert 0.0 <= data["prediction"] <= 1.0
