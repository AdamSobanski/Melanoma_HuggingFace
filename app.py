"""
Melanoma Detection — Streamlit frontend
Upload one or more skin lesion images. Each one is sent to the existing
Gradio pipeline Space (YOLOv8n -> U-Net -> EfficientNet-B4) hosted at
Ai-Adam-Six-Sigma/Melanoma_project via its API. Results are shown per image
and logged as a row in SeaTable.

No models are loaded here — this app is a thin client. All heavy inference
runs on the pipeline Space's ZeroGPU hardware.
"""

from datetime import datetime
import io
import json
import os
import tempfile
import uuid

import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
from seatable_api import Base

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPACE_ID = "Ai-Adam-Six-Sigma/Melanoma_project"

# Bump this manually every time you swap weights on the pipeline Space, so
# rows in SeaTable can be traced back to the model version that produced them.
PIPELINE_VERSION = "yolo-v1_unet-v1_b4-v1"

SEATABLE_SERVER_URL = st.secrets.get("SEATABLE_SERVER_URL") or "https://cloud.seatable.io"
SEATABLE_API_TOKEN = st.secrets.get("SEATABLE_API_TOKEN")
SEATABLE_TABLE_NAME = st.secrets.get("SEATABLE_TABLE_NAME") or "melanoma_predictions_template"

# Hugging Face token: authenticated calls get a much larger ZeroGPU quota
# than anonymous ones. Add HF_TOKEN to the Streamlit secrets.
HF_TOKEN = st.secrets.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# SeaTable
# ---------------------------------------------------------------------------

@st.cache_resource
def get_seatable_base():
    if not SEATABLE_API_TOKEN:
        return None
    base = Base(SEATABLE_API_TOKEN, SEATABLE_SERVER_URL)
    base.auth()
    return base


def log_to_seatable(image: Image.Image, result: dict):
    base = get_seatable_base()
    if base is None:
        return

    # Get current UTC/local timestamp matching SeaTable's date format
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "Timestamp": now_str,
        "Pipeline Version": PIPELINE_VERSION,
        "Prediction Classification": result["prediction"],  # Column name updated to match SeaTable exactly
        "Malignant Probability": result["malignant_probability"],
        "Detection Confidence Yolo": result["detection"]["confidence"],
    }

    # Crop the lesion using YOLO's bounding box and upload it as an attachment.
    box = result.get("detection", {}).get("box")
    if box:
        x1, y1, x2, y2 = box
        crop = image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG")
        upload_info = base.upload_bytes_file(
            name=f"{uuid.uuid4()}.jpg", content=buf.getvalue(), file_type="image"
        )
        row["Lesion Crop"] = [upload_info["url"]]

    base.append_row(SEATABLE_TABLE_NAME, row)

# ---------------------------------------------------------------------------
# Pipeline Call via Gradio Client (ZeroGPU)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_gradio_client():
    """Initializes and caches the authenticated Gradio client connection to the Space."""
    try:
        return Client(SPACE_ID, token=HF_TOKEN)      # newer gradio_client
    except TypeError:
        return Client(SPACE_ID, hf_token=HF_TOKEN)   # older gradio_client


def call_pipeline(image: Image.Image) -> dict:
    """Saves image to a temporary file and sends it to the pipeline Space via official gradio_client."""
    client = get_gradio_client()

    # Resize image to prevent Out-Of-Memory (OOM) errors on ZeroGPU
    image_copy = image.copy()
    image_copy.thumbnail((1280, 1280))

    # Save to a temporary file as gradio_client requires a valid file path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image_copy.save(tmp.name, format="JPEG")
        tmp_path = tmp.name

    try:
        # Official SDK call handling ZeroGPU sessions and FileData structures automatically
        raw_result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )
    finally:
        # Always remove the temporary file, even if the call fails
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if isinstance(raw_result, str):
        return json.loads(raw_result)

    return raw_result

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Melanoma Detection Pipeline")

st.warning(
    "⚠️ This is an educational prototype, not a diagnostic tool. Results may be "
    "unreliable, especially for non-dermoscopic images. Always consult a dermatologist."
)

st.title("Melanoma Detection Pipeline")
st.write("Upload one or more close-up images of a skin lesion. Each one runs through YOLO detection → U-Net segmentation → EfficientNet-B4 classification.")

if not HF_TOKEN:
    st.info(
        "No Hugging Face token configured — running with the small anonymous "
        "ZeroGPU quota. Add HF_TOKEN to the app secrets for a higher limit."
    )

uploaded_files = st.file_uploader(
    "Choose image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    result = call_pipeline(image)
                except Exception as e:
                    if "ZeroGPU" in str(e) or "quota" in str(e).lower():
                        st.error(
                            "The daily GPU quota has been exhausted. "
                            "Please try again later."
                        )
                    else:
                        st.error(f"Something went wrong: {e}")
                    continue

            if isinstance(result, dict) and result.get("error"):
                st.error(result["error"])
                continue

            st.subheader(f"Prediction: {result['prediction']}")
            st.write(f"Malignant probability: {result['malignant_probability']:.2%}")
            st.write(f"Detection confidence: {result['detection']['confidence']:.2%}")

            if result["prediction"] == "malignant":
                st.warning("This is not a medical diagnosis. Please consult a dermatologist.")

            try:
                log_to_seatable(image, result)
            except Exception as e:
                st.caption(f"(Note: could not log this prediction — {e})")