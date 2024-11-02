import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import wget
import zipfile
import os

# Descargar y descomprimir el modelo desde Dropbox si no existe
def download_and_extract_model():
    model_url = 'https://dl.dropboxusercontent.com/s/umy60ud0xd594ekl9b2ht/xception_model_final.zip?rlkey=7r2t9wrqb6es7ndftqnxt6ekb&st=n23144fz'
    zip_path = 'xception_model_final.zip'
    extract_folder = 'extracted_files'

    # Descargar el archivo zip si no existe
    if not os.path.exists(zip_path):
        try:
            wget.download(model_url, zip_path)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")
            return None

    # Descomprimir el archivo
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    # Ajusta la ruta para que apunte al archivo correcto
    return os.path.join(extract_folder, 'xception_model_final.keras')

# Descargar y cargar el modelo
modelo_path = download_and_extract_model()

if modelo_path and os.path.exists(modelo_path):
    try:
        model = tf.keras.models.load_model(modelo_path)
        st.success("Modelo cargado correctamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
else:
    st.error("No se encontró el archivo del modelo.")

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen de rayos X...", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file is not None and 'model' in locals():
    # Mostrar la imagen subida
    st.image(uploaded_file, width=300, caption="Imagen cargada")

    # Preprocesamiento de la imagen para hacer la predicción
    img = image.load_img(uploaded_file, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)

    # Mostrar resultados
    if prediction[0][1] > 0.5:
        st.success('La imagen muestra un caso de **neumonía**.')
    else:
        st.success('La imagen es **normal**.')
