from src.symptom_predictor import predict_symptoms


# Example symptoms from the dataset
selected_symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions",
]


result = predict_symptoms(
    selected_symptoms,
    top_k=3,
)

print("Selected symptoms:")
for symptom in result["selected_symptoms"]:
    print("-", symptom)

print("\nTop prediction:")
print(result["prediction"])

print("\nTop 3 model predictions:")

for item in result["top_predictions"]:
    print(
        f"{item['disease']}: "
        f"{item['probability'] * 100:.2f}%"
    )