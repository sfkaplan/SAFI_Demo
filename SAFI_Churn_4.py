#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import shap
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
st.title("🔮 Predicción y Análisis de Churn en Fondos de Inversión")

# --- Sidebar: Inputs del usuario ---
st.sidebar.header("📊 Características del Cliente")

rendimiento = st.sidebar.slider("Rendimiento del fondo (%)", -10.0, 20.0, 5.0) / 100
volatilidad = st.sidebar.slider("Volatilidad", 0.01, 0.5, 0.2)
comisiones = st.sidebar.slider("Comisiones (%)", 0.1, 3.0, 1.0) / 100
benchmark = st.sidebar.slider("Comparación con benchmark (%)", -10.0, 10.0, 0.0) / 100
frecuencia = st.sidebar.slider("Frecuencia de transacciones por mes", 0, 10, 2)
tiempo = st.sidebar.slider("Tiempo de permanencia (años)", 0.0, 10.0, 2.0)
edad = st.sidebar.slider("Edad", 18, 80, 40)
patrimonio = st.sidebar.number_input("Patrimonio invertido", value=100000.0)
ubicacion = st.sidebar.selectbox("Ubicación", ["Capital", "Interior", "CABA", "Conurbano"])

modelo_elegido = st.selectbox("🧠 Elegí un modelo", list(models.keys()))

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

# --- Transformar input ---
X_input = preprocessor.transform(input_df)
feature_names = preprocessor.get_feature_names_out()

# --- Tabs para Predicción e Interpretabilidad ---
tab_prediccion, tab_interpretabilidad = st.tabs(["🔮 Predicción", "📈 Interpretabilidad"])

with tab_prediccion:
    st.header("Predicción de Churn")
    modelo = models[modelo_elegido]
    prob = modelo.predict_proba(X_input)[0][1]
    st.markdown(f"### ✅ Probabilidad de churn: **{prob:.2%}**")

with tab_interpretabilidad:
    st.header("Interpretabilidad del Modelo")

    if modelo_elegido == "Regresión Logística":
        st.subheader("📑 Coeficientes, errores estándar y p-valores")
        # Obtener datos transformados para reestimación
        df = pd.read_csv("datos_churn.csv")
        X_full = preprocessor.transform(df.drop("churn", axis=1))
        y_full = df["churn"]

        # Ajustar un modelo con statsmodels
        X_design = sm.add_constant(X_full)
        sm_model = sm.Logit(y_full, X_design).fit(disp=0)
        coef_df = pd.DataFrame({
            "Variable": ["Intercepto"] + list(feature_names),
            "Coeficiente": sm_model.params.round(4),
            "Error estándar": sm_model.bse.round(4),
            "p-valor": sm_model.pvalues.round(4)
        })
        st.dataframe(coef_df)

    else:
        st.subheader("🌳 SHAP values: contribución de las variables")
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_input)

        # Mostrar gráfico de contribución
        st.write("#### Gráfico de contribución (para la predicción actual)")
        shap.initjs()
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[1][0],
                                             base_values=explainer.expected_value[1],
                                             data=pd.DataFrame(X_input, columns=feature_names).iloc[0]))
        st.pyplot(fig)


