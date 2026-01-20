import streamlit as st
import numpy as np
import cv2
import torch
import joblib
from PIL import Image 
import io
from pathlib import Path

from skimage.feature import graycomatrix, graycoprops
import torchvision.models as models
import torchvision.transforms as T


# =========================
# CONFIGURATION STREAMLIT
# =========================

st.set_page_config(
    page_title="Détection de la Qualité des Œufs",
    layout="centered"
)

st.title("Détection Automatique de la Qualité des Œufs")
st.write(
    "Cette application utilise un modèle d’intelligence artificielle "
    "pour classifier automatiquement un œuf comme **sain** ou **défectueux** "
    "à partir d’une image."
)


# =========================
# INITIALISATION
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_pipeline():
    return joblib.load("pipeline_qualite_oeuf.joblib")

pipeline = load_pipeline()


@st.cache_resource
def load_cnn():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    return model

cnn_model = load_cnn()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# PRÉTRAITEMENTS
# =========================

def preprocess_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    gray = clahe.apply(gray)
    gray = cv2.resize(gray, (224, 224))
    return gray


# =========================
# EXTRACTION GLCM
# =========================

def extract_glcm_features(gray):
    glcm = graycomatrix(
        gray,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        features.extend(graycoprops(glcm, prop).ravel())

    return np.array(features)


# =========================
# EXTRACTION CNN
# =========================

def extract_cnn_features(image):
    image_pil = Image.fromarray(image)
    tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = cnn_model(tensor)

    return features.cpu().numpy().squeeze()


# =========================
# EXTRACTION HYBRIDE
# =========================

def extract_hybrid_cnn_glcm(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    gray = preprocess_gray(image)
    glcm_feat = extract_glcm_features(gray)
    cnn_feat = extract_cnn_features(image)

    return np.concatenate([cnn_feat, glcm_feat])


# =========================
# INTERFACE UTILISATEUR
# =========================

uploaded_file = st.file_uploader(
    "📤 Téléversez une image d’œuf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(uploaded_file, caption="Image fournie", use_container_width=True)

    if st.button("🔍 Lancer la prédiction"):

        with st.spinner("Analyse en cours..."):

            image_bytes = uploaded_file.read()

            X = extract_hybrid_cnn_glcm(image_bytes)
            X = X.reshape(1, -1)

            prediction = pipeline.predict(X)[0]
            probabilities = pipeline.predict_proba(X)[0]

        st.subheader("📊 Résultat de la classification")

        if prediction == 1:
            st.error("❌ Œuf défectueux")
        else:
            st.success("✅ Œuf sain")

        st.write(
            f"**Probabilité œuf sain :** {probabilities[0]:.2%}"
        )
        st.write(
            f"**Probabilité œuf défectueux :** {probabilities[1]:.2%}"
        )

# =========================
# LANCER LAPPLICATION : streamlit run app.py
# =========================