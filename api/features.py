import cv2
import numpy as np

def extract_classical_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image non lisible")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (128, 128))

    mean = np.mean(img_gray)
    std = np.std(img_gray)

    return np.array([mean, std])
