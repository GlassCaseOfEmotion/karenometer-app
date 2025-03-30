import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import tempfile

st.title("🧠 Karenometer: Face Match Checker")
st.write("Upload two face photos to compare and get a similarity score.")

# Upload images
img1_file = st.file_uploader("Upload Reference Photo", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Comparison Photo", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    st.image([img1, img2], caption=["Reference", "Comparison"], width=300)

    # Save images temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp1, tempfile.NamedTemporaryFile(suffix=".jpg") as temp2:
        img1.save(temp1.name)
        img2.save(temp2.name)

        try:
            result = DeepFace.verify(img1_path=temp1.name, img2_path=temp2.name)
            score = (1 - result["distance"]) * 100  # Similarity % approximation

            st.success(f"🎯 Match Score: **{score:.2f}%**")

            if score > 90:
                st.markdown("✅ Likely the same person.")
            elif score > 70:
                st.markdown("🟡 Possibly the same person. Check image quality and angle.")
            else:
                st.markdown("❌ Likely different people.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
