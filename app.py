import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# =========================
# Función para generar dataset sintético
# =========================
def generar_datos(n_muestras, n_columnas, tipo="mixto"):
    np.random.seed(42)
    
    data = {}

    deportes = ["Fútbol", "Baloncesto", "Tenis", "Natación", "Atletismo", "Ciclismo"]

    # Variables cuantitativas
    if n_columnas >= 1:
        data["Edad"] = np.random.randint(15, 40, size=n_muestras)
    if n_columnas >= 2:
        data["Altura_cm"] = np.random.normal(175, 10, size=n_muestras).astype(int)
    if n_columnas >= 3:
        data["Peso_kg"] = np.random.normal(70, 12, size=n_muestras).astype(int)

    # Variables cualitativas
    if n_columnas >= 4:
        data["Deporte"] = np.random.choice(deportes, size=n_muestras)
    if n_columnas >= 5:
        data["Género"] = np.random.choice(["Masculino", "Femenino"], size=n_muestras)
    if n_columnas >= 6:
        data["Nivel"] = np.random.choice(["Amateur", "Profesional"], size=n_muestras)

    return pd.DataFrame(data)

# =========================
# Configuración Streamlit
# =========================
st.set_page_config(page_title="EDA Deportivo", layout="wide")

st.title("⚽🏀 Exploratory Data Analysis (EDA) de Datos Deportivos")
st.markdown("Este es un análisis exploratorio interactivo de un dataset sintético sobre deportes.")

# =========================
# Sidebar
# =========================
st.sidebar.header("Configuración del Dataset")

n_muestras = st.sidebar.slider("Número de muestras", min_value=50, max_value=500, value=200, step=10)
n_columnas = st.sidebar.slider("Número de columnas", min_value=2, max_value=6, value=4)

df = generar_datos(n_muestras, n_columnas)

# Mostrar dataset
if st.checkbox("Mostrar tabla de datos"):
    st.dataframe(df.head(20))

# =========================
# Selección de columnas y gráficos
# =========================
st.sidebar.header("Opciones de visualización")

graficas = ["Tendencia (línea)", "Barras", "Dispersión", "Pastel", "Histograma"]
grafica_seleccionada = st.sidebar.selectbox("Selecciona tipo de gráfica", graficas)

columnas_disponibles = df.columns.tolist()
columnas_seleccionadas = st.sidebar.multiselect("Selecciona columnas para analizar", columnas_disponibles)

# =========================
# Visualizaciones
# =========================
st.subheader("📊 Visualización de Datos")

if len(columnas_seleccionadas) == 0:
    st.warning("Por favor selecciona al menos una columna para visualizar.")
else:
    fig, ax = plt.subplots(figsize=(8, 5))

    # Gráfico de línea (tendencia)
    if grafica_seleccionada == "Tendencia (línea)" and len(columnas_seleccionadas) == 1:
        ax.plot(df[columnas_seleccionadas[0]], marker="o")
        ax.set_title(f"Tendencia de {columnas_seleccionadas[0]}")
        st.pyplot(fig)

    # Gráfico de barras
    elif grafica_seleccionada == "Barras" and len(columnas_seleccionadas) == 1:
        df[columnas_seleccionadas[0]].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Frecuencia de {columnas_seleccionadas[0]}")
        st.pyplot(fig)

    # Dispersión (requiere dos columnas)
    elif grafica_seleccionada == "Dispersión" and len(columnas_seleccionadas) == 2:
        sns.scatterplot(x=df[columnas_seleccionadas[0]], y=df[columnas_seleccionadas[1]], ax=ax)
        ax.set_title(f"Dispersión entre {columnas_seleccionadas[0]} y {columnas_seleccionadas[1]}")
        st.pyplot(fig)

    # Pastel (requiere una columna categórica)
    elif grafica_seleccionada == "Pastel" and len(columnas_seleccionadas) == 1:
        df[columnas_seleccionadas[0]].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Distribución de {columnas_seleccionadas[0]}")
        st.pyplot(fig)

    # Histograma (numérica)
    elif grafica_seleccionada == "Histograma" and len(columnas_seleccionadas) == 1:
        df[columnas_seleccionadas[0]].plot(kind="hist", bins=15, ax=ax)
        ax.set_title(f"Histograma de {columnas_seleccionadas[0]}")
        st.pyplot(fig)

    else:
        st.warning("Selecciona columnas adecuadas para este tipo de gráfico.")

# =========================
# Análisis Descriptivo
# =========================
st.subheader("📈 Análisis Estadístico")

if st.checkbox("Mostrar estadísticas descriptivas"):
    st.write(df.describe(include="all"))

