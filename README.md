# 🏥 Medical Diagnosis Assistant

An educational AI/ML project built with Python and Streamlit that demonstrates chest X-ray classification, symptom-based disease prediction, and medical analysis report generation.

## 📌 Project Overview

The Medical Diagnosis Assistant combines deep learning and traditional machine learning to demonstrate how AI techniques can be applied to healthcare-related image and symptom analysis.

The application provides an interactive web interface where users can upload chest X-ray images, select symptoms, review model predictions, and generate a downloadable PDF report.

**Disclaimer:** This is an educational prototype, not a clinically validated diagnostic system. Predictions must not be used as a substitute for professional medical advice.

## ✨ Features

### 1. Chest X-Ray Analysis
- Upload chest X-ray images in JPG, JPEG, or PNG format.
- Use a MobileNetV2-based deep learning model to classify images as NORMAL or PNEUMONIA.
- Use an experimental image-validation model before disease classification.
- Display model prediction scores.

### 2. Symptom-Based Prediction
- Select symptoms through the Streamlit interface.
- Use a trained machine-learning model to rank possible disease labels across 41 classes.
- Display the top three model predictions and their scores.

### 3. PDF Analysis Report
- Generate a downloadable PDF report.
- Include available X-ray and symptom-analysis results.
- Display an educational and medical-safety disclaimer.

## 🛠️ Technologies Used

- **Python** — application development
- **TensorFlow and Keras** — deep learning models
- **MobileNetV2** — transfer learning for chest X-ray classification
- **scikit-learn** — symptom-based machine learning
- **Streamlit** — interactive web application
- **NumPy and Pillow** — image and numerical processing
- **Joblib** — saved symptom model
- **ReportLab** — PDF report generation

## 📂 Project Structure

```text
medical-diagnosis-assistant/
├── app.py
├── src/
│   ├── image_predictor.py
│   └── symptom_predictor.py
├── models/
│   ├── chest_xray_model.keras
│   ├── xray_validator.keras
│   └── symptom_model.pkl
├── dataset/
├── train_image_model.py
├── train_xray_validator.py
├── train_symptom_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

The structure above illustrates the principal project files. Actual filenames may differ. The dataset and large generated files may be excluded from the GitHub repository.

## ⚙️ Installation and Setup

### Prerequisites
- Python
- Git
- Visual Studio Code or another Python editor

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/medical-diagnosis-assistant.git
cd medical-diagnosis-assistant
```

Replace `YOUR-USERNAME` with the repository owner's GitHub username.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the required packages:

```powershell
python -m pip install streamlit tensorflow pillow numpy pandas scikit-learn joblib reportlab
```

### 4. Add the required files

Ensure the trained model files are available in the `models/` directory:

- `chest_xray_model.keras`
- `xray_validator.keras`
- `symptom_model.pkl`

The models may need to be downloaded separately if they are not distributed with the repository. Training datasets are also required if you want to retrain the models.

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## 📊 Model Evaluation

The chest X-ray disease classifier achieved approximately 88.78% test accuracy in the recorded evaluation, with 96.67% pneumonia recall and 75.64% normal recall.

The validator reported 100% test accuracy on its current test dataset. This result does not establish reliable performance on unfamiliar images.

The symptom model's output scores are model estimates, not verified real-world probabilities of disease.

## ⚠️ Limitations and Safety

- This project is for learning and demonstration purposes.
- The chest X-ray dataset is pediatric and may not generalize to other patient populations.
- The image validator is experimental and can reject genuine X-rays or accept unrelated images.
- Dataset quality, preprocessing, class balance, and evaluation methodology affect model performance.
- The application has not been clinically validated.
- Always consult a qualified healthcare professional for medical assessment.

## 🚀 Future Improvements

- Improve X-ray validation using more diverse and challenging image datasets.
- Evaluate the system on independent datasets.
- Improve model interpretability with visual explanations.
- Add stronger automated tests and model monitoring.
- Improve deployment, documentation, and accessibility.

## 👨‍💻 Author

**GitHub:** https://github.com/gursimarsingh177-svg/medical-diganosis-assistant

## 📄 License

Add an appropriate open-source license after confirming that your code and datasets can be distributed under its terms.
