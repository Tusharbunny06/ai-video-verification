import streamlit as st
import torch
import tempfile
import os
import hashlib

from utils import load_video
from deepfake.detector import DeepfakeDetector
from watermark_registry import WATERMARK_REGISTRY

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="AI Video Verification System",
    layout="centered"
)

st.title("🎥 AI Video Verification System")
st.write(
    "This system verifies videos using **digital provenance (watermarking)** "
    "and **deepfake detection**. Provenance verification is deterministic, "
    "while deepfake detection is probabilistic."
)

# --------------------
# LOAD MODEL
# --------------------
@st.cache_resource
def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(
        torch.load("deepfake_detector_resnet.pth", map_location="cpu")
    )
    model.eval()
    return model


model = load_model()

# --------------------
# HASH UTILITY
# --------------------
def compute_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# --------------------
# WATERMARK CHECK (PROVENANCE-BASED)
# --------------------
def watermark_check(video_path):
    video_hash = compute_hash(video_path)
    for entry in WATERMARK_REGISTRY.values():
        if video_hash == entry["hash"]:
            return True
    return False

# --------------------
# FILE UPLOAD
# --------------------
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

# --------------------
# PROCESS VIDEO
# --------------------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            # 1️⃣ Provenance verification FIRST
            has_watermark = watermark_check(video_path)

            # 2️⃣ Deepfake detection ONLY if provenance missing
            deepfake_prob = None
            if not has_watermark:
                video = load_video(video_path)
                with torch.no_grad():
                    deepfake_prob = model(video).item()

        # --------------------
        # RESULTS
        # --------------------
        st.subheader("🔍 Analysis Results")

        # ✅ CASE 1: Provenance verified
        if has_watermark:
            st.success("✅ Provenance Verified")
            st.write("**Final Decision:** Verified AI-generated video")
            st.write("**Method:** Cryptographic provenance registry")

        # ⚠️ CASE 2: No provenance → always decide (Solution B)
        else:
            st.warning("⚠️ No provenance metadata found")
            st.write(f"**Deepfake Model Score:** `{deepfake_prob:.2f}`")

            # --------------------
            # SOLUTION B LOGIC
            # Always decide, but mark confidence as low
            # For THIS trained model:
            #   higher score → REAL
            #   lower score → DEEPFAKE
            # --------------------
            if deepfake_prob < 0.50:
                st.error("❌ Likely AI-generated / Deepfake")
                st.write("**Confidence:** Low")
                st.write(
                    "**Note:** Decision is based on visual cues only. "
                    "False positives are possible without provenance."
                )
            else:
                st.success("✅ Likely Real video")
                st.write("**Confidence:** Low")
                st.write(
                    "**Note:** Decision is based on visual cues only. "
                    "Provenance verification provides stronger guarantees."
                )

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
