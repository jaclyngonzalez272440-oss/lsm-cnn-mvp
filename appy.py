import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# -----------------------------------
# Descargar modelo desde Google Drive
# -----------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1HL3cChEPT45ozbK-DHFd5DiThsliLHc2"
MODEL_FILE = "lsm_model_mvp.h5"

# Descargar si no existe
if not os.path.isfile(MODEL_FILE):
    with st.spinner("Descargando modelo..."):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# Cargar modelo
model = tf.keras.models.load_model(MODEL_FILE)

CLASSES = ["Abrir", "Mal"]

# -----------------------------------
# Configuración de Streamlit
# -----------------------------------
st.set_page_config(page_title="LSM MVP", page_icon="🤟")

st.title("🤟 MVP Traductor de Señas LSM")
st.write("Clasifica señas básicas como parte de un MVP educativo para tu proyecto.")

# -----------------------------------
# Subir imagen
# -----------------------------------
img_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predicción
    pred = model.predict(img)
    pred_class = CLASSES[np.argmax(pred)]
    prob = np.max(pred)

    st.subheader("Resultado:")
    st.write(f"**Seña predicha:** {pred_class}")
    st.write(f"**Probabilidad:** {prob:.2f}")

