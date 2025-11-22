import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gdown
import os
from keras.models import load_model

# ------------------------------
# DESCARGAR MODELO DESDE DRIVE
# ------------------------------

MODEL_URL = "https://drive.google.com/uc?id=1HL3cChEPT45ozbK-DHFd5DiThsliLHc2"
MODEL_PATH = "model.h5"

# Si el modelo no existe en el servidor, descargarlo
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ------------------------------
# CARGAR MODELO
# ------------------------------
model = load_model(MODEL_PATH)

# Clases (ajusta si usas otras)
CLASSES = ["Abrir", "Mal"]

# ------------------------------
# INTERFAZ STREAMLIT
# ------------------------------

st.title("LSM – MVP (Clasificador sencillo)")
st.write("Sube una imagen para predecir la seña:")

uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir a array
    img = np.array(image)

    # Preprocesar
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)
    idx = np.argmax(pred)
    confidence = pred[0][idx]

    st.subheader("Predicción:")
    st.write(f"👉 **{CLASSES[idx]}**")
    st.write(f"Confianza: {confidence:.2f}")




