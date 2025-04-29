# tests/test_api.py
import time
import requests
import pytest
from subprocess import Popen

BASE_URL = "http://127.0.0.1:3000"

@pytest.fixture(scope="session", autouse=True)
def start_container():
    proc = Popen([
        "docker", "run", "--rm", "-p", "3000:3000",
        "--name", "admissions_test", "admissions_service:latest"
    ])
    # Wait
    time.sleep(3)
    yield
    proc.terminate()
    proc.wait()

def test_login_and_predict():
    # Login
    login_resp = requests.post(
        f"{BASE_URL}/login",
        json={"username": "user123", "password": "password123"},
        timeout=5,
    )
    assert login_resp.status_code == 200
    token = login_resp.json().get("token")
    assert token, "no JWT returned"

    # Predict
    pred_resp = requests.post(
        f"{BASE_URL}/predict",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "GRE Score": 320,
            "TOEFL Score": 110,
            "University Rating": 4,
            "SOP": 4.5,
            "LOR": 4.5,
            "CGPA": 9.0,
            "Research": 1,
        },
        timeout=5,
    )
    assert pred_resp.status_code == 200
    assert isinstance(pred_resp.json().get("prediction"), float)
