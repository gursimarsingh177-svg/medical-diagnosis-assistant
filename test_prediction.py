from src.image_predictor import predict_xray


IMAGE_PATH = (
    "dataset/chest_xray/train/PNEUMONIA/"
    "person1000_virus_1681.jpeg"
)


label, confidence = predict_xray(IMAGE_PATH)


print("Prediction:", label)
print("Confidence:", f"{confidence * 100:.2f}%")