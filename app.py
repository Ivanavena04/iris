import streamlit as st
import pickle
import numpy as np
import os  # Importa el módulo os

# Cargar el modelo desde el archivo .sav
@st.cache_resource
def load_model():
    model_path = 'final_model.sav'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El archivo '{model_path}' no se encuentra.")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Cargar el modelo
try:
    loaded_model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Título de la aplicación
st.title("Predicción de Especies de Flores Iris")

# Crear un formulario para la entrada de datos
st.write("Ingrese las características de la flor Iris:")

# Entradas para las características
sepal_length = st.number_input("Longitud del sépalo (cm):", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
sepal_width = st.number_input("Ancho del sépalo (cm):", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.number_input("Longitud del pétalo (cm):", min_value=0.0, max_value=7.0, value=6.0, step=0.1)
petal_width = st.number_input("Ancho del pétalo (cm):", min_value=0.0, max_value=3.0, value=2.0, step=0.1)

# Botón para hacer la predicción
if st.button("Predecir"):
    # Crear un array con las características de entrada
    input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Hacer la predicción
    prediction = loaded_model.predict(input_array)

    # Mostrar la predicción
    species = ["Setosa", "Versicolor", "Virginica"]  # Especies de Iris
    st.write(f"La especie de la flor Iris es: {species[prediction[0]]}")