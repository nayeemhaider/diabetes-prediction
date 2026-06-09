import requests

patients = [
    {"Pregnancies":1,"Glucose":89,"BloodPressure":66,"SkinThickness":23,"Insulin":94,"BMI":28.1,"DiabetesPedigreeFunction":0.17,"Age":21},
    {"Pregnancies":8,"Glucose":183,"BloodPressure":64,"SkinThickness":0,"Insulin":0,"BMI":23.3,"DiabetesPedigreeFunction":0.67,"Age":32},
    {"Pregnancies":3,"Glucose":130,"BloodPressure":78,"SkinThickness":28,"Insulin":110,"BMI":31.2,"DiabetesPedigreeFunction":0.42,"Age":44},
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"patients": patients}
)

for i, pred in enumerate(response.json()["predictions"]):
    print(f"Patient {i+1}: {pred['label']:15} {pred['probability']:.1%}  [{pred['risk_band']}]")