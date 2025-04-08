
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("♻️ Nuevo Clasificador de Residuos")
st.write("Sube una imagen y te diré si es aprovechable, no aprovechable u orgánico aprovechable.")

model = tf.keras.models.load_model('modelo_residuos.h5')
clases = ['Aprovechable', 'No aprovechable', 'Orgánico aprovechable']

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagen cargada", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    clase_predicha = clases[np.argmax(prediction)]
    confianza = np.max(prediction) * 100

    st.markdown(f"### 🧠 Predicción: **{clase_predicha}** ({confianza:.2f}% de confianza)")
