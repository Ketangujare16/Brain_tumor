import io
import os
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- USER CONFIG: change these to match your setup ---
MODEL_PATH = "/content/final_mobilenet_modelgliomaboost.keras"  # change if different
# CLASS_NAMES must match your model's output order
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]  # example - EDIT THIS
# End user config
# -----------------------------------------------------

@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path, compile=False)
    return model

def get_target_size(model):
    # Try to infer input size
    try:
        shape = model.input_shape
        # shape could be (None, H, W, C) or (None, None, None, 3)
        if len(shape) == 4:
            _, h, w, _ = shape
            if h is None or w is None:
                return (224, 224)  # fallback
            return (h, w)
    except Exception:
        pass
    return (224, 224)

def preprocess_image(pil_img, target_size):
    img = pil_img.convert("RGB")
    img = img.resize(target_size[::-1], Image.BILINEAR)  # PIL size as (width,height)
    arr = np.asarray(img) / 255.0
    # ensure batch shape
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)

def predict(model, img_array):
    preds = model.predict(img_array)
    # If model outputs logits, apply softmax
    if preds.ndim == 2:
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    else:
        probs = preds[0]
        if probs.sum() > 1.001 or probs.sum() < 0.999:
            # if not normalized, softmax
            probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
    top_idx = int(np.argmax(probs))
    return top_idx, probs[top_idx], probs

def compute_gradcam(model, img_array, class_index, layer_name=None):
    """
    Returns a (H, W) heatmap for the grad-cam and a PIL image overlay.
    """
    # If layer_name is None, pick last conv layer automatically
    if layer_name is None:
        # find last conv2d layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    if layer_name is None:
        raise ValueError("No Conv2D layer found; please provide layer_name")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    # pooled grads
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        heatmap = heatmap
    else:
        heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    return heatmap

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img.convert("RGB"))
    heatmap_color = plt.cm.jet(heatmap)[:, :, :3]  # RGB
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)

def risk_level_from_prediction(pred_class_name, confidence, no_tumor_label="no_tumor"):
    """
    Simple risk heuristic:
      - If predicted label is the no_tumor_label -> 'Low'
      - Else: confidence >= 0.85 -> 'High'; 0.6-0.85 -> 'Medium'; else 'Low'
    Customize as needed.
    """
    if pred_class_name.lower() == no_tumor_label.lower():
        return "Low"
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.6:
        return "Medium"
    return "Low"

def create_pdf_report(patient_info, pred_label, confidence, risk_level, original_pil, gradcam_pil, out_path):
    """
    Generates a PDF report with FPDF2.
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, "Brain MRI Tumor Report", ln=True, align='C')
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(40, 8, f"Name: {patient_info.get('name','')}", ln=False)
    pdf.cell(80, 8, f"Age: {patient_info.get('age','')}", ln=False)
    pdf.cell(60, 8, f"Gender: {patient_info.get('gender','')}", ln=True)
    pdf.cell(0, 8, f"Email: {patient_info.get('email','')}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 8, "Prediction Summary:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, f"Tumor Type: {pred_label}", ln=True)
    pdf.cell(0, 7, f"Confidence: {confidence:.4f}", ln=True)
    pdf.cell(0, 7, f"Risk Level: {risk_level}", ln=True)
    pdf.ln(6)

    # Save temporary images
    tmp_dir = tempfile.mkdtemp()
    orig_path = os.path.join(tmp_dir, "orig.jpg")
    grad_path = os.path.join(tmp_dir, "gradcam.jpg")
    original_pil.save(orig_path, format='JPEG', quality=85)
    gradcam_pil.save(grad_path, format='JPEG', quality=85)

    # Insert images
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 8, "Original MRI:", ln=True)
    pdf.image(orig_path, x=10, y=None, w=90)
    pdf.ln(48)
    pdf.cell(0, 8, "Grad-CAM overlay:", ln=True)
    pdf.image(grad_path, x=10, y=None, w=90)
    pdf.ln(48)

    pdf.output(out_path)
    return out_path

# --- Streamlit UI ---
st.set_page_config(page_title="Brain Tumor Detector - Report", layout="centered")
st.title("Brain Tumor Detector — Live Report (Streamlit)")

st.info("Fill patient details, upload an MRI image, then press Predict → Generate PDF")

with st.sidebar:
    st.header("Model & Settings")
    st.write("Model path:")
    st.text(MODEL_PATH)
    st.write("Classes (edit in app.py if incorrect):")
    st.text(", ".join(CLASS_NAMES))

# Load model
try:
    model = load_model(MODEL_PATH)
    TARGET_SIZE = get_target_size(model)  # (H, W)
    st.success(f"Model loaded. Detected input size: {TARGET_SIZE}")
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}: {e}")
    st.stop()

# Patient inputs
st.subheader("Patient information")
name = st.text_input("Name")
age = st.text_input("Age")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
email = st.text_input("Email (optional)")

uploaded_file = st.file_uploader("Upload MRI image (jpg/png)", type=['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

# Buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict")
with col2:
    genpdf_btn = st.button("Generate PDF (requires prediction)")

# State holders
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

if predict_btn:
    if uploaded_file is None:
        st.error("Please upload an MRI image first.")
    else:
        with st.spinner("Preprocessing & predicting..."):
            arr = preprocess_image(image, TARGET_SIZE)  # shape (1,H,W,3)
            idx, conf, probs = predict(model, arr)
            pred_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
            risk = risk_level_from_prediction(pred_label, float(conf))

            # Grad-CAM
            try:
                heatmap = compute_gradcam(model, arr, idx)
                gradcam_pil = overlay_heatmap_on_image(image, heatmap)
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")
                gradcam_pil = image.copy()

            # Save to state
            st.session_state['last_prediction'] = {
                "patient": {"name": name, "age": age, "gender": gender, "email": email},
                "label": pred_label,
                "confidence": float(conf),
                "risk": risk,
                "original_image": image,
                "gradcam_image": gradcam_pil
            }

            st.success(f"Predicted: {pred_label} (confidence: {conf:.4f})")
            st.image(gradcam_pil, caption="Grad-CAM overlay", use_column_width=True)
            st.write("All class probabilities:")
            probs_display = {CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class_{i}": float(probs[i]) for i in range(len(probs))}
            st.json(probs_display)

if genpdf_btn:
    pred = st.session_state.get('last_prediction')
    if not pred:
        st.error("No prediction found. Run Predict first.")
    else:
        out_pdf = f"tumor_report_{(name or 'patient')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_pdf_path = os.path.join(tempfile.gettempdir(), out_pdf)
        create_pdf_report(pred['patient'], pred['label'], pred['confidence'], pred['risk'],
                          pred['original_image'], pred['gradcam_image'], out_pdf_path)
        with open(out_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.success("PDF report generated.")
        st.download_button("Download PDF report", data=pdf_bytes, file_name=out_pdf, mime="application/pdf")

