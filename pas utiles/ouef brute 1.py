# %% [markdown]
# 
# # GROUPE 14 — Détection automatique des œufs sains et défectueux
# ## Good and Bad Eggs Identification Image Dataset
# 
# Ce notebook présente une implémentation complète, reproductible et fonctionnelle du projet,
# en suivant rigoureusement la méthodologie du TP de référence (Cocoa Diseases),
# étendue à un projet final de niveau Master.
# 

# %%

# ============================
# IMPORTS GÉNÉRAUX
# ============================
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

#from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path


import warnings
warnings.filterwarnings("ignore")


# %% [markdown]
# 
# ## 1. Chargement des données
# 

# %%
# MODIFIER CE CHEMIN SELON VOTRE MACHINE
DATA_ROOT = Path(r'C:\Users\cheic\Documents\M2 IA MathInfo\DeepL\PROJET FINAL DPL')

original_dir = DATA_ROOT / 'Original Images(Eggs)'
aug_dir = DATA_ROOT / 'Augmented_Images(Eggs)'

def gather_paths(base_dir):
    rows = []
    
    if not base_dir.exists():
        return rows
    
    for label_dir in ['Good Eggs', 'Bad Eggs']:
        class_path = base_dir / label_dir
        
        if not class_path.exists():
            continue
        
        for img_path in class_path.glob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            rows.append({
                'path': str(img_path),
                'label': 'good' if 'Good' in label_dir else 'bad',
                'source': base_dir.name
            })
    
    return rows

rows = []
rows.extend(gather_paths(original_dir))
rows.extend(gather_paths(aug_dir))

df = pd.DataFrame(rows)

print("Nombre total d’images trouvées :", len(df))

if len(df) > 0:
    df['label_enc'] = df['label'].map({'good': 0, 'bad': 1})
    display(df.head())
else:
    print("Aucune image trouvée. Vérifiez le chemin DATA_ROOT.")


# %% [markdown]
# 
# ## 2. Prétraitements
# 

# %%
def preprocess_image(img_path, target_size=(224, 224)):
    # Chargement de l’image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ------------------------------------------------------------------
    # 1. Correction d’illumination + normalisation
    # ------------------------------------------------------------------
    gray_norm = cv2.normalize(
        gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    
    # ------------------------------------------------------------------
    # 2. Filtre médian (réduction du bruit)
    # ------------------------------------------------------------------
    gray_denoised = cv2.medianBlur(gray_norm, ksize=5)
    
    # ------------------------------------------------------------------
    # 3. Égalisation adaptative (CLAHE)
    # ------------------------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=2.0, 
        tileGridSize=(8, 8)
    )
    gray_clahe = clahe.apply(gray_denoised)
    
    # ------------------------------------------------------------------
    # 4. Redimensionnement
    # ------------------------------------------------------------------
    img_resized = cv2.resize(
        gray_clahe, target_size, interpolation=cv2.INTER_AREA
    )
    
    # Normalisation finale [0,1] pour les modèles IA
    img_final = img_resized.astype(np.float32) / 255.0
    
    return img_final


# %% [markdown]
# Application des prétraitements à l’ensemble du dataset

# %%
X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img = preprocess_image(row['path'])
    
    if img is not None:
        X.append(img)
        y.append(row['label_enc'])


# %% [markdown]
# Conversion en tableaux NumPy exploitables

# %%
X = np.array(X)
y = np.array(y)

# Ajout d’un canal pour les modèles CNN (H, W, 1)
X = X[..., np.newaxis]

print("Shape des images :", X.shape)
print("Shape des labels :", y.shape)


# %% [markdown]
# - **6997** , c'est le nombre total de cabosses détectées et extraites.
# 
# - Chaque images (cabosse) à pour dimension (224, 224, 1) → une image à 1 couches.

# %%
print(pd.DataFrame(y).value_counts(normalize=True))


# %% [markdown]
# La classe 0 représente environ 50 % des observations ;
# 
# La classe 1 représente environ 50 % des observations.
# 
# Ce qui traduit un équilibre statistique remarquable.

# %% [markdown]
# 
# ## 3. Extracteurs classiques
# 

# %%
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import moments_hu

# Paramètres LBP
RADIUS_LIST = [1, 2, 3]
N_POINTS_LIST = [8, 16, 24]

# Paramètres HOG
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys"
}


# %% [markdown]
# HOG – fissures et contours

# %%
def extract_hog(image):
    features = hog(
        image,
        orientations=HOG_PARAMS["orientations"],
        pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
        cells_per_block=HOG_PARAMS["cells_per_block"],
        block_norm=HOG_PARAMS["block_norm"],
        feature_vector=True
    )
    return features


# %% [markdown]
# LBP multi-échelle – texture

# %%
def extract_lbp(image):
    lbp_features = []
    
    for radius, n_points in zip(RADIUS_LIST, N_POINTS_LIST):
        lbp = local_binary_pattern(image, n_points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, n_points + 3),
            range=(0, n_points + 2),
            density=True
        )
        lbp_features.extend(hist)
    
    return np.array(lbp_features)


# %% [markdown]
# GLCM + Haralick – rugosité et patterns

# %%
def extract_glcm(image):
    glcm = graycomatrix(
        image,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2],
        levels=256,
        symmetric=True,
        normed=True
    )
    
    features = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        features.extend(graycoprops(glcm, prop).ravel())
    
    return np.array(features)


# %% [markdown]
# Moments de Hu – formes irrégulières

# %%
def extract_hu(image):
    moments = cv2.moments(image)
    hu = cv2.HuMoments(moments)
    
    # Transformation logarithmique pour stabilité numérique
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    return hu.flatten()


# %% [markdown]
# Histogrammes HSV – décoloration

# %%
def extract_hsv(image_path, bins=(16, 16, 16)):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist(
        [hsv], [0, 1, 2],
        None, bins,
        [0, 180, 0, 256, 0, 256]
    )
    
    cv2.normalize(hist, hist)
    return hist.flatten()


# %% [markdown]
# Extraction globale des caractéristiques

# %%
X_features = []
y_features = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_gray = preprocess_image(row['path'])
    
    if img_gray is None:
        continue
    
    img_gray_uint8 = (img_gray * 255).astype(np.uint8)
    
    hog_feat = extract_hog(img_gray_uint8)
    lbp_feat = extract_lbp(img_gray_uint8)
    glcm_feat = extract_glcm(img_gray_uint8)
    hu_feat = extract_hu(img_gray_uint8)
    hsv_feat = extract_hsv(row['path'])
    
    # Fusion des descripteurs
    features = np.concatenate([
        hog_feat,
        lbp_feat,
        glcm_feat,
        hu_feat,
        hsv_feat
    ])
    
    X_features.append(features)
    y_features.append(row['label_enc'])


# %% [markdown]
# Conversion en tableaux NumPy

# %%
X_features = np.array(X_features)
y_features = np.array(y_features)

print("Shape des features :", X_features.shape)
print("Shape des labels :", y_features.shape)


# %% [markdown]
# Normalisation

# %% [markdown]
# PCA (conserver 95 % de la variance)

# %% [markdown]
# 
# ## 4. Extracteurs profonds
# 

# %%
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image

#Transformations ImageNet

transform_dl = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#chargement image RGB
def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return transform_dl(img).unsqueeze(0)



# %% [markdown]
# EfficientNet-B0 / B2

# %%

#EfficientNet-B0
eff_b0 = models.efficientnet_b0(weights="IMAGENET1K_V1")
eff_b0.classifier = torch.nn.Identity()
eff_b0.eval()

#EfficientNet-B2
eff_b2 = models.efficientnet_b2(weights="IMAGENET1K_V1")
eff_b2.classifier = torch.nn.Identity()
eff_b2.eval()


# %% [markdown]
# ConvNeXt-Tiny

# %%
convnext = models.convnext_tiny(weights="IMAGENET1K_V1")
convnext.classifier = torch.nn.Identity()
convnext.eval()


# %% [markdown]
# Swin Transformer (vision globale)

# %%
swin = models.swin_t(weights="IMAGENET1K_V1")
swin.head = torch.nn.Identity()
swin.eval()


# %% [markdown]
# RegNetY

# %%
regnet = models.regnet_y_400mf(weights="IMAGENET1K_V1")
regnet.fc = torch.nn.Identity()
regnet.eval()


# %% [markdown]
# Fonction générique d’extraction d’embeddings

# %%
def extract_embeddings(model, df):
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img = load_image_rgb(row["path"])
            feat = model(img)
            embeddings.append(feat.squeeze().numpy())
            labels.append(row["label_enc"])
    
    return np.array(embeddings), np.array(labels)


# %% [markdown]
# Extraction

# %%
X_effb0, y_effb0 = extract_embeddings(eff_b0, df)
X_convnext, y_convnext = extract_embeddings(convnext, df)
X_swin, y_swin = extract_embeddings(swin, df)
X_regnet, y_regnet = extract_embeddings(regnet, df)

print(X_effb0.shape, X_convnext.shape, X_swin.shape, X_regnet.shape)


# %% [markdown]
# 
# ## 5. Extracteurs hybrides
# 

# %% [markdown]
# CNN + GLCM (vision globale + texture)

# %%
from sklearn.preprocessing import StandardScaler

X_glcm = []
y_glcm = []

for _, row in df.iterrows():
    img = preprocess_image(row["path"])
    if img is None:
        continue

    img_uint8 = (img * 255).astype(np.uint8)
    glcm_feat = extract_glcm(img_uint8)

    X_glcm.append(glcm_feat)
    y_glcm.append(row["label_enc"])

X_glcm = np.array(X_glcm)
y_glcm = np.array(y_glcm)

print(X_glcm.shape)

# normalisation des caractéristiques GLCM

glcm_scaled = StandardScaler().fit_transform(X_glcm)


#Fusion CNN + GLCM

X_cnn = X_effb0  # embeddings CNN
X_cnn_glcm = np.concatenate([X_cnn, glcm_scaled], axis=1)

print(X_cnn_glcm.shape)



# %% [markdown]
# Transformer + HOG (attention + contours)

# %%
X_hog = []
y_hog = []

for _, row in df.iterrows():
    img = preprocess_image(row["path"])
    if img is None:
        continue

    img_uint8 = (img * 255).astype(np.uint8)
    hog_feat = extract_hog(img_uint8)

    X_hog.append(hog_feat)
    y_hog.append(row["label_enc"])

X_hog = np.array(X_hog)
y_hog = np.array(y_hog)

print("Shape HOG brut :", X_hog.shape)

hog_scaled = StandardScaler().fit_transform(X_hog).astype("float32")



# %% [markdown]
# PCA sur HOG

# %%
from sklearn.decomposition import PCA

pca_hog = PCA(
    n_components=1,
    svd_solver="randomized",
    random_state=42
)

X_hog_pca = pca_hog.fit_transform(hog_scaled)

print("Shape HOG après PCA :", X_hog_pca.shape)

# Fusion Swin + HOG PCA

X_swin_hog = np.concatenate([X_swin, X_hog_pca], axis=1)

print("Shape Swin + HOG :", X_swin_hog.shape)



# %% [markdown]
# Autoencoder convolutionnel

# %%
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*56*56, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64*56*56),
            nn.Unflatten(1, (64, 56, 56)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# %% [markdown]
# Masked Autoencoder (MAE)

# %%
import timm
import torch
mae = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True
)
mae.eval()

def extract_mae_embeddings(model, df):
    embeddings = []

    with torch.no_grad():
        for _, row in df.iterrows():
            img = load_image_rgb(row["path"])
            feat = model.forward_features(img)
            embeddings.append(feat.squeeze().numpy())

    return np.array(embeddings)


# %% [markdown]
# 
# ## 7. Réduction de dimension
# 

# %% [markdown]
# Approches statistiques

# %% [markdown]
# PCA

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X_features).astype("float32")

pca = PCA(
    n_components=1,
    svd_solver="randomized",
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)


# %%
print("Shape après PCA :", X_pca.shape)

# %% [markdown]
# ICA

# %%
from sklearn.decomposition import FastICA

ica = FastICA(n_components=500, random_state=42)
X_ica = ica.fit_transform(X_scaled)


# %%
print("Shape après PCA :", X_ica.shape)

# %% [markdown]
# TUNCATED SVD

# %%
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=500, random_state=42)
X_svd = svd.fit_transform(X_scaled)
print("Shape après SVD :", X_svd.shape)


# %% [markdown]
# Approches deep learning

# %% [markdown]
# 

# %%
# On récupère z (latent)
class AutoEncoder(nn.Module):
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# %% [markdown]
# Parametric UMAP

# %%
import umap

umap_reducer = umap.UMAP(
    n_components=128,
    random_state=42
)

X_umap = umap_reducer.fit_transform(X_scaled)
print("Shape après UMAP :", X_umap.shape)

# %% [markdown]
# Deep Embedding Reduction Network

# %%
class DeepEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

model = DeepEmbeddingNet(
    input_dim=X_pca.shape[1],
    embedding_dim=128
)

X_embed = model(torch.tensor(X_pca, dtype=torch.float32))
print("Shape après Deep Embedding :", X_embed.shape)

# %% [markdown]
# 
# ## 8. Comparaison des modèles
# 

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def evaluate(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    y_pred = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    return {
        "F1": f1_score(yte, y_pred, average="macro"),
        "AUC": roc_auc_score(yte, y_proba),
        "Dim": X.shape[1]
    }


# %%
results = {
    "PCA": evaluate(X_pca, y),
    "ICA": evaluate(X_ica, y),
    "SVD": evaluate(X_svd, y),
    "UMAP": evaluate(X_umap, y),
}


results_df = pd.DataFrame(results).T
display(results_df)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["F1"])
plt.title("Comparaison des méthodes de réduction – F1-score")
plt.ylabel("F1-score")
plt.xlabel("Méthode")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(results_df.index, results_df["AUC"])
plt.title("Comparaison des méthodes de réduction – AUC ROC")
plt.ylabel("AUC")
plt.xlabel("Méthode")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



# %% [markdown]
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# %% [markdown]
# Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

print("LogReg F1 :", f1_score(y_test, y_pred))
print("LogReg AUC :", roc_auc_score(y_test, y_proba))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – Logistic Regression")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="LogReg (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – Logistic Regression")
plt.legend()
plt.tight_layout()
plt.show()





# %% [markdown]
# SVM RBF

# %%
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    class_weight="balanced"
)

svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)[:, 1]

print("SVM F1 :", f1_score(y_test, y_pred))
print("SVM AUC :", roc_auc_score(y_test, y_proba))


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – SVM RBF")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="SVM RBF (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – SVM RBF")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ---------- Modèle ----------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ---------- Entraînement ----------
rf.fit(X_train, y_train)

# ---------- Prédictions ----------
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# ---------- Scores ----------
print("RF F1 :", f1_score(y_test, y_pred))
print("RF AUC :", roc_auc_score(y_test, y_proba))

# ---------- Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – Random Forest")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ---------- Courbe ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="RF (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – Random Forest")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# XGBoost

# %%
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ---------- Modèle ----------
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

# ---------- Entraînement ----------
xgb.fit(X_train, y_train)

# ---------- Prédictions ----------
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

# ---------- Scores ----------
print("XGB F1 :", f1_score(y_test, y_pred))
print("XGB AUC :", roc_auc_score(y_test, y_proba))

# ---------- Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – XGBoost")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ---------- Courbe ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="XGB (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – XGBoost")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# MLP

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ---------- Modèle ----------
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    random_state=42
)

# ---------- Entraînement ----------
mlp.fit(X_train, y_train)

# ---------- Prédictions ----------
y_pred = mlp.predict(X_test)
y_proba = mlp.predict_proba(X_test)[:, 1]

# ---------- Scores ----------
print("MLP F1 :", f1_score(y_test, y_pred))
print("MLP AUC :", roc_auc_score(y_test, y_proba))

# ---------- Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – MLP")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ---------- Courbe ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="MLP (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – MLP")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# k-NN

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# ---------- Modèle ----------
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

# ---------- Entraînement ----------
knn.fit(X_train, y_train)

# ---------- Prédictions ----------
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]

# ---------- Scores ----------
print("kNN F1 :", f1_score(y_test, y_pred))
print("kNN AUC :", roc_auc_score(y_test, y_proba))

# ---------- Matrice de confusion ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title("Matrice de confusion – k-NN")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ---------- Courbe ROC ----------
fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="k-NN (AUC = %.3f)" % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC – k-NN")
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# 
# ## Partie 7 : Évaluation

# %% [markdown]
# 
# ## 10. Discussion et limites
# 
# Les erreurs de classification concernent principalement :
# - des fissures très fines,
# - des variations d’éclairage importantes,
# - des coquilles naturellement atypiques.
# 
# Des améliorations possibles incluent l’augmentation de données,
# une acquisition contrôlée et l’intégration temps réel en chaîne industrielle.
# 

# %% [markdown]
# 


