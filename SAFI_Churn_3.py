#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# --- Cargar modelos y preprocesador ---
@st.cache_resource
def cargar_modelos():
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = cloudpickle.load(f)
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = cloudpickle.load(f)
    with open("random_forest_model.pkl", "rb") as f:
        rf_model = cloudpickle.load(f)
    with open("lightgbm_model.pkl", "rb") as f:
        lgb_model = cloudpickle.load(f)
    return {
        "Regresión Logística": logistic_model,
        "Random Forest": rf_model,
        "LightGBM": lgb_model,
    }, preprocessor

models, preprocessor = cargar_modelos()

# --- Título ---
st.title("Predicción de Churn en Fondos de Inversión")

# --- Sidebar: Inputs del usuario ---
st.sidebar.header("Características del Cliente")

rendimiento = st.sidebar.slider("Rendimiento del fondo (%)", -10.0, 20.0, 5.0) / 100
volatilidad = st.sidebar.slider("Volatilidad", 0.01, 0.5, 0.2)
comisiones = st.sidebar.slider("Comisiones (%)", 0.1, 3.0, 1.0) / 100
benchmark = st.sidebar.slider("Comparación con benchmark (%)", -10.0, 10.0, 0.0) / 100
frecuencia = st.sidebar.slider("Frecuencia de transacciones por mes", 0, 10, 2)
tiempo = st.sidebar.slider("Tiempo de permanencia (años)", 0.0, 10.0, 2.0)
edad = st.sidebar.slider("Edad", 18, 80, 40)
patrimonio = st.sidebar.number_input("Patrimonio invertido", value=100000.0)
ubicacion = st.sidebar.selectbox("Ubicación", ["La Paz", "Santa Cruz", "Cochabamba", "Oruro"])

modelo_elegido = st.selectbox("Elegí un modelo", list(models.keys()))

# --- Convertir input a DataFrame ---
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

# --- Transformar y predecir ---
X_input = preprocessor.transform(input_df)
modelo = models[modelo_elegido]
prob = modelo.predict_proba(X_input)[0][1]

# --- Resultado ---
st.markdown(f"### Probabilidad de churn: **{prob:.2%}**")

