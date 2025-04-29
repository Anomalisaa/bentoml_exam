#!/bin/bash

# Get Token
TOKEN=$(curl -s -X POST http://127.0.0.1:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user123","password":"password123"}' | jq -r '.token')

echo "Known Token:"
echo $TOKEN
echo ""

# Prediction Request
echo "Starte Prediction Request..."
curl -v -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "GRE Score": 320,
    "TOEFL Score": 110,
    "University Rating": 4,
    "SOP": 4.5,
    "LOR": 4.0,
    "CGPA": 9.0,
    "Research": 1
  }'
