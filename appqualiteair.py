import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from matplotlib.cm import get_cmap
from sklearn.linear_model import LinearRegression

# ==========================
# Configuration de la page
# ==========================
st.set_page_config(
    page_title="Analyse de la qualité de l'air – Saint-Germain-des-Prés",
    layout="wide"
)

# ----------------------------
# Chargement des données
# ----------------------------
df = pd.read_csv("qualiteair.csv", sep=';')

# Correction des décimales (virgule → point)
for col in ['TEMP', 'HUMI', 'PM10']:
    df[col] = df[col].astype(str).str.replace(",", ".")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convertir DATE/HEURE en datetime
df["DATE/HEURE"] = pd.to_datetime(df["DATE/HEURE"], errors="coerce")

# ----------------------------
# Interface Streamlit
# ----------------------------
st.title("🌫️ Analyse de la qualité de l’air – Saint-Germain-des-Prés")
st.write("Application générée automatiquement à partir du fichier **qualiteair.csv**.")

# Sidebar
st.sidebar.header("Options d'affichage")
option = st.sidebar.selectbox(
    "Choisissez une variable à visualiser",
    ["PM10", "TEMP", "HUMI", "Corrélations"]
)

# ==========================
# Fonctions graphiques
# ==========================
def plot_time_series(column_name, ylabel):
    fig, ax = plt.subplots()
    ax.plot(df["DATE/HEURE"], df[column_name])
    ax.set_title(f"Évolution de {ylabel} dans le temps")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_boxplot(column_name, ylabel):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column_name], ax=ax)
    ax.set_title(f"Distribution de {ylabel}")
    ax.set_xlabel(ylabel)
    st.pyplot(fig)

# ==========================
# AFFICHAGE SELON L’OPTION
# ==========================

# ----------- PM10 -----------
if option == "PM10":
    st.subheader("🟦 Évolution des particules PM10")

    # --- KPI ---
    pm10_moy = df["PM10"].mean()
    pm10_max = df["PM10"].max()
    pm10_min = df["PM10"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("😷 PM10 moyenne", f"{pm10_moy:.2f} µg/m³")
    col2.metric("😷 PM10 max", f"{pm10_max:.2f} µg/m³")
    col3.metric("😷 PM10 min", f"{pm10_min:.2f} µg/m³")

    # Time series
    plot_time_series("PM10", "PM10 (µg/m³)")
    corr_text = (
        "Dans ce graphique montrant l'évolution PM10 au cours des mois en 2025, on peut constater une évolution constante avec certains pics."
    )
    st.write(corr_text)

    # Boxplot
    plot_boxplot("PM10", "PM10 (µg/m³)")

    st.write(df["PM10"].describe())

# ----------- TEMPÉRATURE -----------
elif option == "TEMP":
    st.subheader("🌡️ Évolution de la température")

    temp_moy = df["TEMP"].mean()
    temp_max = df["TEMP"].max()
    temp_min = df["TEMP"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("🌡 Température moyenne", f"{temp_moy:.2f} °C")
    col2.metric("🌡 Température max", f"{temp_max:.2f} °C")
    col3.metric("🌡 Température min", f"{temp_min:.2f} °C")

    plot_time_series("TEMP", "Température (°C)")
    corr_text = (
        "Dans ce graphique montrant l'évolution de la température au cours des mois en 2025, on peut constater une forte évolution de février à juillet puis une baisse progréssive."
    )
    st.write(corr_text)
    plot_boxplot("TEMP", "Température (°C)")

    st.write(df["TEMP"].describe())

# ----------- HUMIDITÉ -----------
elif option == "HUMI":
    st.subheader("💧 Évolution de l'humidité")

    humi_moy = df["HUMI"].mean()
    humi_max = df["HUMI"].max()
    humi_min = df["HUMI"].min()

    col1, col2, col3 = st.columns(3)
    col1.metric("💧 Humidité moyenne", f"{humi_moy:.2f} %")
    col2.metric("💧 Humidité max", f"{humi_max:.2f} %")
    col3.metric("💧 Humidité min", f"{humi_min:.2f} %")

    plot_time_series("HUMI", "Humidité (%)")
    corr_text = (
        "Dans ce graphique montrant l'évolution de l'humidité au cours des mois en 2025, on peut constater une évolution constante avec un pic aux alentours de mars 2025."
    )
    st.write(corr_text)
    plot_boxplot("HUMI", "Humidité (%)")

    st.write(df["HUMI"].describe())

# ----------- CORRÉLATIONS -----------
elif option == "Corrélations":
    st.subheader("📊 Corrélations entre PM10, Température et Humidité")

    # Matrice de corrélation
    fig, ax = plt.subplots()
    corr = df[['PM10', 'TEMP', 'HUMI']].corr()
    im = ax.imshow(corr, cmap="coolwarm")
    plt.colorbar(im)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

    corr_text = (
        "Les corrélations entre PM10, TEMP et HUMI sont faibles à modérées : "
        "PM10 et TEMP présentent une légère corrélation positive (\\~0,1), ce qui signifie que lorsque la température augmente, la concentration de PM10 tend à augmenter légèrement ; "
        "PM10 et HUMI ont une faible corrélation négative (\\~-0,1), indiquant un léger recul de PM10 quand l’humidité augmente ; "
        "TEMP et HUMI montrent une corrélation négative modérée (\\~-0,3), ce qui traduit que l’humidité diminue lorsque la température augmente. "
        "En résumé, ces variables ne sont pas fortement liées entre elles, la relation la plus notable étant l’inverse entre TEMP et HUMI."
    )
    st.write(corr_text)


    # --- Graphique comparatif PM10 / TEMP / HUMI ---
    stats = {
        'PM10 (µg/m³)': [df['PM10'].mean(), df['PM10'].max(), df['PM10'].min()],
        'Température (°C)': [df['TEMP'].mean(), df['TEMP'].max(), df['TEMP'].min()],
        'Humidité (%)': [df['HUMI'].mean(), df['HUMI'].max(), df['HUMI'].min()]
    }
    stats_df = pd.DataFrame(stats, index=['Moyenne', 'Max', 'Min'])
    
    fig2, ax2 = plt.subplots(figsize=(8,5))
    stats_df.plot(kind='bar', ax=ax2)
    ax2.set_title("Comparaison PM10, Température et Humidité")
    ax2.set_ylabel("Valeurs")
    plt.xticks(rotation=0)
    st.pyplot(fig2)

# ----------------------------
# Données brutes
# ----------------------------
st.subheader("📄 Aperçu des données brutes")
st.dataframe(df)

