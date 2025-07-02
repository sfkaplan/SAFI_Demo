#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Cargar modelos
models = {
    "Regresión Logística": joblib.load("logistic_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "LightGBM": joblib.load("lightgbm_model.pkl")
}

st.title("Predicción de Churn en Fondos de Inversión")

# Inputs del usuario
st.sidebar.header("Características del Cliente")

rendimiento = st.sidebar.slider("Rendimiento del fondo (%)", -10.0, 20.0, 5.0) / 100
volatilidad = st.sidebar.slider("Volatilidad", 0.01, 0.5, 0.2)
comisiones = st.sidebar.slider("Comisiones (%)", 0.1, 3.0, 1.0) / 100
benchmark = st.sidebar.slider("Comparación con benchmark (%)", -10.0, 10.0, 0.0) / 100
frecuencia = st.sidebar.slider("Frecuencia de transacciones por mes", 0, 10, 2)
tiempo = st.sidebar.slider("Tiempo de permanencia (años)", 0.0, 10.0, 2.0)
edad = st.sidebar.slider("Edad", 18, 80, 40)
patrimonio = st.sidebar.number_input("Patrimonio invertido", value=100000.0)
ubicacion = st.sidebar.selectbox("Ubicación", ['La Paz', 'Santa Cruz', 'Cochabamba', 'Oruro'])

modelo_elegido = st.selectbox("Elegí un modelo", list(models.keys()))

# Crear DataFrame con los inputs
input_df = pd.DataFrame([{
    "rendimiento": rendimiento,
    "volatilidad": volatilidad,
    "comisiones": comisiones,
    "comparacion_benchmark": benchmark,
    "frecuencia_transacciones": frecuencia,
    "tiempo_permanencia": tiempo,
    "edad": edad,
    "patrimonio": patrimonio,
    "ubicacion": ubicacion
}])

# Predicción
model = models[modelo_elegido]
prob = model.predict_proba(input_df)[0][1]

st.markdown(f"### Probabilidad de churn: **{prob:.2%}**")

