import requests

patient = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 72,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 33
}

response = requests.post("http://localhost:8000/predict", json=patient)
result = response.json()

print(f"Prediction  : {result['label']}")
print(f"Probability : {result['probability']:.1%}")
print(f"Risk band   : {result['risk_band']}")
print(f"Advice      : {result['advice']}")