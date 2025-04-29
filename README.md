# Admissions Prediction API with BentoML

This project provides a containerized BentoML-based API to predict the chance of admission. 

## Prerequisites
You will need the following tools:
- Git
- Python 3.8+
- Docker
- BentoML CLI (pip install bentoml)
- pytest and requests (pip install pytest requests)

# Project Structure
This is the project structure of the repository.

examen_bentoml/
├── bentofile.yaml
├── Dockerfile.template          
├── data/
│   ├── raw/
│   │   └── admission.csv
│   └── processed/
│       └── scaler.plk
│       └── X_test.csv
│       └── X_train.csv
│       └── y_test.csv
│       └── y_train.csv
├── models/
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── service.py
├── tests/
│   └── test_api.py   
│   └── test_prediction.sh 
└── README.md   
└── requirements.txt               

# How to...
1. Download raw data

```bash
cd data/raw
wget https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv
```

2. Create & activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Build Bento & Containerize

4.1 Build the Bento

```bash
bentoml build
```

4.2 Create Docker Image

Create a Docker image: admissions_prediction

```bash
bentoml containerize admissions_prediction:1.0.0 \
  -t <YourName>_admissions_prediction:latest
```


4.3 Run the Container

```bash
docker run --rm -p 3000:3000 admissions_prediction:latest
```

The API will be available at http://localhost:3000.

4. Test the API

4.1 Login to Obtain JWT Token

curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"secret"}'

Response:

{"access_token":"<JWT_TOKEN>"}

4.2 Request a Prediction

curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -d '{
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.5,
    "lor": 4.0,
    "cgpa": 9.1,
    "research": 1
}'

The response will be: 

{"chance_of_admit": 0.82}

# Run Unit Tests

```bash
pytest tests/test_service.py -v
```

You see: All tests should PASS.