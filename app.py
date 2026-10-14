import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from src.image_predictor import predict_xray
from src.symptom_predictor import predict_symptoms, feature_names

PROJECT_ROOT = Path(__file__).resolve().parent


def display_symptom(symptom):
    """Turn a model feature name into a readable UI label."""
    return " ".join(str(symptom).replace("_", " ").split()).title()


def draw_wrapped_pdf_text(pdf, text, x, y, width=495, font_name="Helvetica", font_size=10, leading=14):
    """Draw text with line wrapping and add a page if the text reaches the bottom."""
    page_height = A4[1]
    pdf.setFont(font_name, font_size)
    lines = simpleSplit(str(text), font_name, font_size, width)

    for line in lines or [""]:
        if y < 45:
            pdf.showPage()
            y = page_height - 50
            pdf.setFont(font_name, font_size)
        pdf.drawString(x, y, line)
        y -= leading

    return y


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

    st.metric("Test Accuracy", "88.78%")
    st.metric("Pneumonia Recall", "96.67%")
    st.metric("Normal Recall", "75.64%")

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

if "symptom_result" not in st.session_state or (
    st.session_state.symptom_result is not None
    and not isinstance(st.session_state.symptom_result, dict)
):
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

        if st.session_state.xray_result == "NOT A CHEST X-RAY":

            st.error("Image rejected: this upload was not accepted as a chest X-ray.")
            st.write(
                "The NORMAL/PNEUMONIA classifier was **not run**. "
                "Upload a genuine chest X-ray image to continue."
            )
            st.caption(
                "The image checks are educational guardrails, not a guarantee. "
                "Grayscale non-X-ray images may still fool the validator."
            )

        else:

            with result_col1:

                st.info(
                    f"Model Prediction: "
                    f"{st.session_state.xray_result}"
                )

            with result_col2:

                st.metric(
                    "Model Score (not a diagnosis)",
                    f"{st.session_state.xray_confidence * 100:.2f}%"
                )

            st.caption(
                "This score is a model probability, not clinical certainty. "
                "The classifier is educational and not clinically validated."
            )


# =========================================================
# TAB 2 — SYMPTOM ANALYSIS
# =========================================================

with tab2:

    st.header("🧠 Symptom Analysis")

    st.write(
        "Search for and select the symptoms to pass to the trained "
        "41-class symptom model."
    )

    st.caption(
        f"The model contains {len(feature_names)} symptom features. "
        "Start typing in the box to search."
    )

    selected = st.multiselect(
        "Select symptoms",
        options=list(feature_names),
        format_func=display_symptom,
        key="symptom_selector",
        help="Choose one or more symptoms from the model's training features.",
    )

    if st.button(
        "🧠 Analyze Symptoms",
        type="primary",
        width="stretch",
    ):

        st.session_state.selected_symptoms = list(selected)

        if not selected:
            st.session_state.symptom_result = None
            st.error("Please select at least one symptom before analyzing.")
        else:
            try:
                with st.spinner("Comparing symptoms with the trained model..."):
                    st.session_state.symptom_result = predict_symptoms(
                        selected,
                        top_k=3,
                    )
            except Exception as e:
                st.session_state.symptom_result = None
                st.error(f"Unable to analyze the selected symptoms: {e}")

    if isinstance(st.session_state.symptom_result, dict):

        result = st.session_state.symptom_result
        st.divider()
        st.subheader("📊 Model Results")
        st.success(f"Top model prediction: {result['prediction']}")

        st.markdown("**Top 3 model predictions**")
        for item in result.get("top_predictions", []):
            probability = float(item["probability"])
            st.write(f"**{item['disease']}** — {probability * 100:.2f}%")
            st.progress(max(0.0, min(1.0, probability)))

        readable_symptoms = [display_symptom(s) for s in result.get("selected_symptoms", [])]
        st.write("**Selected symptoms:** " + ", ".join(readable_symptoms))

        st.warning(
            "These scores come from an educational model trained on a "
            "symptom dataset. They are not a real-world probability of disease, "
            "a confirmed diagnosis, or a substitute for medical advice."
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

        if st.session_state.xray_result == "NOT A CHEST X-RAY":
            st.write("**Image validation:** Rejected. Disease classification was skipped.")
        else:
            st.write(f"**Prediction:** {st.session_state.xray_result}")
            st.write(
                f"**Model score:** {st.session_state.xray_confidence * 100:.2f}%"
            )

    else:

        st.info(
            "No X-ray analysis has been performed yet."
        )


    st.subheader("Symptom Analysis")

    if isinstance(st.session_state.symptom_result, dict):

        symptom_result = st.session_state.symptom_result
        st.write(f"**Top model prediction:** {symptom_result['prediction']}")

        readable_symptoms = [display_symptom(s) for s in st.session_state.selected_symptoms]
        st.write("**Selected symptoms:** " + (", ".join(readable_symptoms) if readable_symptoms else "None"))

        st.write("**Top 3 model scores:**")
        for index, item in enumerate(symptom_result.get("top_predictions", []), start=1):
            st.write(f"{index}. {item['disease']} — {float(item['probability']) * 100:.2f}%")

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

        if xray_text == "NOT A CHEST X-RAY":
            y = draw_wrapped_pdf_text(
                pdf,
                "Image validation: rejected. The NORMAL/PNEUMONIA classifier was skipped.",
                70,
                y,
                width=475,
            )
        else:
            y = draw_wrapped_pdf_text(
                pdf,
                f"Prediction: {xray_text}",
                70,
                y,
                width=475,
            )
            if st.session_state.xray_confidence is not None:
                y = draw_wrapped_pdf_text(
                    pdf,
                    f"Model score (not a diagnosis): {st.session_state.xray_confidence * 100:.2f}%",
                    70,
                    y,
                    width=475,
                )

        y -= 35
        y = draw_wrapped_pdf_text(
            pdf,
            "Symptom Analysis",
            50,
            y,
            font_name="Helvetica-Bold",
            font_size=12,
            leading=16,
        )
        y -= 4

        symptom_result = st.session_state.symptom_result
        if isinstance(symptom_result, dict):
            y = draw_wrapped_pdf_text(
                pdf,
                f"Top model prediction: {symptom_result['prediction']}",
                70,
                y,
            )

            readable_symptoms = [display_symptom(s) for s in st.session_state.selected_symptoms]
            selected_text = ", ".join(readable_symptoms) if readable_symptoms else "None"
            y = draw_wrapped_pdf_text(
                pdf,
                f"Selected symptoms: {selected_text}",
                70,
                y,
                width=475,
            )

            y = draw_wrapped_pdf_text(
                pdf,
                "Top three model scores:",
                70,
                y,
                font_name="Helvetica-Bold",
            )
            for item in symptom_result.get("top_predictions", []):
                y = draw_wrapped_pdf_text(
                    pdf,
                    f"{item['disease']}: {float(item['probability']) * 100:.2f}%",
                    85,
                    y,
                )
        else:
            y = draw_wrapped_pdf_text(
                pdf,
                "Symptom analysis: Not analyzed",
                70,
                y,
            )

        y -= 12

        y = draw_wrapped_pdf_text(
            pdf,
            "Recommendation",
            50,
            y,
            font_name="Helvetica-Bold",
            font_size=12,
            leading=16,
        )
        y = draw_wrapped_pdf_text(
            pdf,
            "Consult a qualified healthcare professional for proper clinical evaluation.",
            70,
            y,
        )
        y -= 20
        y = draw_wrapped_pdf_text(
            pdf,
            "IMPORTANT DISCLAIMER",
            50,
            y,
            font_name="Helvetica-Bold",
            font_size=10,
        )
        y = draw_wrapped_pdf_text(
            pdf,
            "This report is generated by an educational AI prototype.",
            50,
            y,
            font_size=9,
            leading=12,
        )
        y = draw_wrapped_pdf_text(
            pdf,
            "It is not a medical diagnosis or substitute for professional care.",
            50,
            y,
            font_size=9,
            leading=12,
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