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

```bash
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
```

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

4. Run the data preparation script

```bash
python src/prepare_data.py --input data/raw/admission.csv --output-dir data/processed
```

# Train the Model & Register with BentoML
1. Train the Model
```bash
python src/train_model.py --data-dir data/processed --model-name admissions_linear
```
2. Confirm the model is saved in the BentoML Model Store
```bash
bentoml models list
```
# Build Bento & Containerize

1. Build the Bento

```bash
bentoml build
```

2. Create Docker Image

Create a Docker image: admissions_prediction. Use the service name and version defined in bentofile.yaml.

```bash
bentoml containerize admissions_prediction:1.0.0 \
  -t admissions_prediction:latest
```

# Run the Containerized API

Start the container, passing the JWT secret for authentication:

```bash
docker run --rm \
  -e JWT_SECRET="secret" \
  -p 3000:3000 \
  --name admissions_service \
  admissions_service:latest
  ```

The API is now available at http://localhost:3000.

And you will see:
```bash
{"prediction": 0.81127...}
```

# Run Unit Tests

Run the pytest suite:

```bash
pytest tests/test_service.py -v
```
All 7 tests should PASS, covering:
- Invalid/valid login (401 vs. 200 + token)
- Missing/invalid/expired JWT (401)
- Malformed input (400)
- Successful prediction (200 + float)