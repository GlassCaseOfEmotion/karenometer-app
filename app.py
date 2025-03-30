
import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.title("🧠 Face Match Checker")
st.write("Upload two face photos and get a similarity match percentage.")

# Upload images
img1_file = st.file_uploader("Upload Reference Photo", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Comparison Photo", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    # Load images
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    st.image([img1, img2], caption=["Reference", "Comparison"], width=300)

    # Convert to numpy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    try:
        # Get face encodings
        enc1 = face_recognition.face_encodings(img1_array)[0]
        enc2 = face_recognition.face_encodings(img2_array)[0]

        # Cosine similarity
        cos_sim = np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))
        match_percent = cos_sim * 100

        st.success(f"🎯 Match Score: **{match_percent:.2f}%**")

        if match_percent > 90:
            st.markdown("✅ Likely the same person.")
        elif match_percent > 70:
            st.markdown("🟡 Possibly the same person. Check image quality and angles.")
        else:
            st.markdown("❌ Likely different people.")
    except IndexError:
        st.error("Couldn't detect a face in one of the images. Try a clearer photo.")
