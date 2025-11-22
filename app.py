import streamlit as st
import numpy as np
import cv2
import requests
import tempfile
from keras.models import load_model

# -------------------------------
# 1) DESCARGAR TU MODELO DESDE DRIVE
# -------------------------------

MODEL_URL = "https://drive.google.com/uc?export=download&id=1HL3cChEPT45ozbK-DHFd5DiThsliLHc2"

@st.cache_resource
def load_mvp_model():
    response = requests.get(MODEL_URL)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    temp.write(response.content)
    temp.flush()
    model = load_model(temp.name)
    return model

model = load_mvp_model()

# -------------------------------
# 2) CONFIGURACIÓN DE LA APP
# -------------------------------

st.title("🤟 MVP de LSM — Abrir vs Mal")
st.write("Sube una imagen para clasificar la seña.")

uploaded = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

if uploaded:
    # Mostrar imagen
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Imagen subida")

    # Preprocesar
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predecir
    pred = model.predict(img_input)
    clases = ["Abrir", "Mal"]
    pred_index = np.argmax(pred)

    st.subheader("Resultado:")
    st.write(f"**Predicción: {clases[pred_index]}**")
    st.write("Probabilidades: ", pred.tolist())



