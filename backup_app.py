import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.image_predictor import predict_xray

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title("🏥 Medical Diagnosis Assistant")

st.markdown(
    """
    **AI-powered educational decision-support system**
    
    This project demonstrates Deep Learning-based chest X-ray
    classification using a MobileNetV2 transfer-learning model.
    """
)

st.warning(
    "⚠️ Educational/decision-support prototype only. "
    "This system is not a medical diagnosis and must not replace "
    "a qualified healthcare professional."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("**Image Model:** MobileNetV2")
    st.write("**Task:** Chest X-ray Classification")
    st.write("**Classes:** NORMAL / PNEUMONIA")
    st.write("**Input Size:** 224 × 224")

    st.divider()

    st.subheader("📊 Test Performance")

    st.metric("Test Accuracy", "86.86%")
    st.metric("Pneumonia Recall", "96.92%")
    st.metric("Normal Recall", "70.09%")

    st.divider()

    st.info(
        "Dataset limitation: the underlying X-ray dataset is a "
        "pediatric chest X-ray dataset. Results should therefore "
        "not be generalized to all patient populations."
    )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩻 X-Ray Analysis",
        "🧠 Symptom Analysis",
        "📄 Analysis Report",
    ]
)


# =========================================================
# SESSION STATE
# =========================================================

if "xray_result" not in st.session_state:
    st.session_state.xray_result = None

if "xray_confidence" not in st.session_state:
    st.session_state.xray_confidence = None

if "symptom_result" not in st.session_state:
    st.session_state.symptom_result = None

if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []


# =========================================================
# TAB 1 — X-RAY ANALYSIS
# =========================================================

with tab1:

    st.header("🩻 Chest X-Ray Analysis")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for model analysis.",
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                uploaded_file,
                caption="Uploaded X-Ray",
                width="stretch",
            )

        with col2:

            st.subheader("Image Information")

            st.write(
                f"**File:** {uploaded_file.name}"
            )

            st.write(
                f"**Type:** {uploaded_file.type}"
            )

            st.write(
                f"**Size:** {uploaded_file.size / 1024:.2f} KB"
            )

        st.divider()

        if st.button(
            "🔍 Analyze X-Ray",
            type="primary",
            width="stretch",
        ):

            with st.spinner(
                "Running Deep Learning analysis..."
            ):

                try:

                    label, confidence = predict_xray(
                        uploaded_file
                    )

                    st.session_state.xray_result = label
                    st.session_state.xray_confidence = confidence

                except Exception as e:

                    st.error(
                        f"Unable to analyze the image: {e}"
                    )


    # Display prediction
    if st.session_state.xray_result is not None:

        st.divider()

        st.subheader("📊 Model Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.success(
                f"Model Prediction: "
                f"{st.session_state.xray_result}"
            )

        with result_col2:

            st.metric(
                "Model Confidence",
                f"{st.session_state.xray_confidence * 100:.2f}%"
            )

        st.caption(
            "Model confidence is a prediction probability, "
            "not clinical certainty."
        )


# =========================================================
# TAB 2 — SYMPTOM ANALYSIS
# =========================================================

with tab2:

    st.header("🧠 Symptom Analysis")

    st.write(
        "Select the symptoms reported by the user."
    )

    symptoms = [
        "Fever",
        "Cough",
        "Fatigue",
        "Headache",
        "Sore Throat",
        "Runny Nose",
        "Shortness of Breath",
        "Chest Pain",
    ]

    selected = []

    cols = st.columns(2)

    for i, symptom in enumerate(symptoms):

        with cols[i % 2]:

            if st.checkbox(symptom):

                selected.append(symptom)

    if st.button(
        "🧠 Analyze Symptoms",
        type="primary",
        width="stretch",
    ):

        st.session_state.selected_symptoms = selected

        # Prototype symptom decision-support logic.
        # This is intentionally presented as a prototype,
        # not as a clinically validated diagnosis model.

        symptom_set = set(selected)

        if {
            "Fever",
            "Cough",
            "Fatigue",
        }.issubset(symptom_set):

            result = (
                "Flu-like / respiratory illness pattern"
            )

        elif {
            "Cough",
            "Shortness of Breath",
        }.issubset(symptom_set):

            result = (
                "Respiratory symptom pattern"
            )

        elif {
            "Headache",
            "Fever",
        }.issubset(symptom_set):

            result = (
                "Fever-associated headache pattern"
            )

        elif "Cough" in symptom_set:

            result = "Cough-related respiratory pattern"

        elif "Headache" in symptom_set:

            result = "Headache symptom pattern"

        else:

            result = (
                "Insufficient symptom information "
                "for a useful pattern"
            )

        st.session_state.symptom_result = result

    if st.session_state.symptom_result is not None:

        st.divider()

        st.subheader("Symptom Analysis Result")

        st.info(
            st.session_state.symptom_result
        )

        st.write(
            "**Selected symptoms:**",
            ", ".join(
                st.session_state.selected_symptoms
            )
            if st.session_state.selected_symptoms
            else "None",
        )

        st.warning(
            "This symptom component is a prototype "
            "decision-support feature and is not a "
            "clinically validated disease predictor."
        )


# =========================================================
# TAB 3 — REPORT
# =========================================================

with tab3:

    st.header("📄 Medical Analysis Report")

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.write(f"**Report Date:** {now}")

    st.divider()

    st.subheader("X-Ray Analysis")

    if st.session_state.xray_result:

        st.write(
            f"**Prediction:** "
            f"{st.session_state.xray_result}"
        )

        st.write(
            f"**Confidence:** "
            f"{st.session_state.xray_confidence * 100:.2f}%"
        )

    else:

        st.info(
            "No X-ray analysis has been performed yet."
        )


    st.subheader("Symptom Analysis")

    if st.session_state.symptom_result:

        st.write(
            f"**Result:** "
            f"{st.session_state.symptom_result}"
        )

        st.write(
            f"**Symptoms:** "
            f"{', '.join(st.session_state.selected_symptoms)}"
        )

    else:

        st.info(
            "No symptom analysis has been performed yet."
        )


    st.divider()

    st.subheader("Recommendation")

    st.write(
        "Consult a qualified healthcare professional "
        "for proper clinical evaluation."
    )

    st.caption(
        "This AI system is an educational decision-support "
        "prototype and does not provide a medical diagnosis."
    )


    # =====================================================
    # PDF REPORT
    # =====================================================

    if st.button(
        "📥 Generate PDF Report",
        width="stretch",
    ):

        buffer = io.BytesIO()

        pdf = canvas.Canvas(
            buffer,
            pagesize=A4,
        )

        width, height = A4

        y = height - 50

        pdf.setFont(
            "Helvetica-Bold",
            18,
        )

        pdf.drawString(
            50,
            y,
            "Medical AI Analysis Report",
        )

        y -= 40

        pdf.setFont(
            "Helvetica",
            11,
        )

        pdf.drawString(
            50,
            y,
            f"Date: {now}",
        )

        y -= 30

        pdf.drawString(
            50,
            y,
            "X-Ray Analysis",
        )

        y -= 20

        xray_text = (
            st.session_state.xray_result
            if st.session_state.xray_result
            else "Not analyzed"
        )

        pdf.drawString(
            70,
            y,
            f"Prediction: {xray_text}",
        )

        y -= 20

        if st.session_state.xray_confidence:

            pdf.drawString(
                70,
                y,
                (
                    "Confidence: "
                    f"{st.session_state.xray_confidence * 100:.2f}%"
                ),
            )

        y -= 35

        pdf.drawString(
            50,
            y,
            "Symptom Analysis",
        )

        y -= 20

        symptom_text = (
            st.session_state.symptom_result
            if st.session_state.symptom_result
            else "Not analyzed"
        )

        pdf.drawString(
            70,
            y,
            f"Result: {symptom_text}",
        )

        y -= 35

        pdf.drawString(
            50,
            y,
            "Recommendation",
        )

        y -= 20

        pdf.drawString(
            70,
            y,
            "Consult a qualified healthcare professional.",
        )

        y -= 45

        pdf.setFont(
            "Helvetica-Bold",
            10,
        )

        pdf.drawString(
            50,
            y,
            "IMPORTANT DISCLAIMER",
        )

        y -= 20

        pdf.setFont(
            "Helvetica",
            9,
        )

        pdf.drawString(
            50,
            y,
            "This report is generated by an educational AI prototype.",
        )

        y -= 15

        pdf.drawString(
            50,
            y,
            "It is not a medical diagnosis or substitute for professional care.",
        )

        pdf.save()

        buffer.seek(0)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name="medical_analysis_report.pdf",
            mime="application/pdf",
            width="stretch",
        )