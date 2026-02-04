import streamlit as st
import torch
import tempfile
import os
import hashlib

from utils import load_video
from deepfake.detector import DeepfakeDetector
from watermark_registry import WATERMARK_REGISTRY
from prompt_to_video import generate_video_from_prompt

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="AI Video Verification System",
    layout="centered"
)

st.title("🎥 AI Video Verification System")
st.write(
    "This system demonstrates **end-to-end verification of Generative AI videos**. "
    "Videos generated from text prompts embed provenance at generation time, "
    "while uploaded videos without provenance are analyzed using deepfake detection."
)

# ======================================================
# 🎨 PROMPT → VIDEO GENERATION (GENAI)
# ======================================================
st.header("🎨 Prompt → Video Generation (GenAI)")

st.write(
    "This module generates videos directly from **text prompts** using Generative AI. "
    "Provenance metadata is embedded at **generation time**, demonstrating how "
    "AI-generated videos should support reliable verification."
)

prompt = st.text_input("Enter a prompt for AI video generation")

if st.button("Generate AI Video from Prompt"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating AI video from prompt..."):
            generated_video_path = generate_video_from_prompt(
                prompt=prompt,
                watermark_registry=WATERMARK_REGISTRY
            )

        st.success("AI video generated with embedded provenance")
        st.video(generated_video_path)

        st.info(
            "This video is registered as AI-generated using "
            "generation-time provenance."
        )

st.divider()

# ======================================================
# LOAD DEEPFAKE MODEL
# ======================================================
@st.cache_resource
def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(
        torch.load("deepfake_detector_resnet.pth", map_location="cpu")
    )
    model.eval()
    return model


model = load_model()

# ======================================================
# HASH UTILITY
# ======================================================
def compute_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# ======================================================
# PROVENANCE CHECK
# ======================================================
def watermark_check(video_path):
    video_hash = compute_hash(video_path)
    for entry in WATERMARK_REGISTRY.values():
        if video_hash == entry["hash"]:
            return True
    return False

# ======================================================
# VIDEO UPLOAD & VERIFICATION
# ======================================================
st.header("📤 Upload Video for Verification")

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

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

        # ======================================================
        # RESULTS
        # ======================================================
        st.subheader("🔍 Analysis Results")

        if has_watermark:
            st.success("✅ Provenance Verified")
            st.write("**Final Decision:** Verified AI-generated video")
            st.write("**Method:** Generation-time digital provenance")

        else:
            st.warning("⚠️ No provenance metadata found")
            st.write(f"**Deepfake Model Score:** `{deepfake_prob:.2f}`")

            if deepfake_prob < 0.50:
                st.error("❌ Likely AI-generated / Deepfake")
                st.write("**Confidence:** Low")
                st.write(
                    "Detection is probabilistic and unreliable without provenance. "
                    "False positives are possible."
                )
            else:
                st.success("✅ Likely Real video")
                st.write("**Confidence:** Low")
                st.write(
                    "Decision is based on visual cues only. "
                    "Provenance verification provides stronger guarantees."
                )

    # Cleanup
    if os.path.exists(video_path):
        os.remove(video_path)
