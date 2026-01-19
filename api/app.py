# ============================================================
# API DE DÉTECTION DE LA QUALITÉ DES ŒUFS
# Projet Deep Learning – Groupe 14
# ============================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import joblib
import tempfile
import os

# ============================================================
# INITIALISATION DE L'API
# ============================================================

app = FastAPI(
    title="API Qualité des Œufs",
    description="Détection automatique des œufs sains et défectueux à partir d'images",
    version="1.0"
)

# ============================================================
# CHARGEMENT DU PIPELINE FINAL
# ============================================================

PIPELINE_PATH = "pipeline_qualite_oeuf.joblib"

pipeline = joblib.load(PIPELINE_PATH)

# ============================================================
# FONCTIONS DE PRÉTRAITEMENT ET EXTRACTION
# ============================================================

def preprocess_image(image_path, size=(224, 224)):
    """
    Prétraitement complet de l'image :
    - Lecture
    - Redimensionnement
    - Conversion en niveaux de gris
    - Filtre médian
    - CLAHE
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def extract_glcm_features(gray):
    """
    Extraction simple GLCM (contraste, homogénéité, énergie)
    """
    from skimage.feature import graycomatrix, graycoprops

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]

    return np.array(features)


def extract_classical_features(image_path):
    """
    Extraction finale utilisée par le pipeline :
    CNN (embeddings déjà intégrés en amont)
    + GLCM (texture)
    """
    gray = preprocess_image(image_path)
    glcm_feat = extract_glcm_features(gray)

    return glcm_feat


# ============================================================
# ROUTES DE L'API
# ============================================================

@app.get("/")
def root():
    return {
        "message": "API Qualité des Œufs opérationnelle",
        "modele": "Hybrid CNN + GLCM",
        "status": "OK"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédiction de la qualité d'un œuf à partir d'une image
    """
    try:
        # Sauvegarde temporaire de l'image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extraction des caractéristiques
        X = extract_classical_features(tmp_path)
        X = X.reshape(1, -1)

        # Prédiction
        prediction = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]

        # Nettoyage
        os.remove(tmp_path)

        # Interprétation
        label = "Œuf sain" if prediction == 0 else "Œuf défectueux"

        return JSONResponse(
            content={
                "prediction": label,
                "classe_numerique": int(prediction),
                "probabilites": {
                    "sain": float(proba[0]),
                    "defectueux": float(proba[1])
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"erreur": str(e)}
        )
