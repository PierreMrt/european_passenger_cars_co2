import streamlit as st
from pages.predict import run_predict_page
from pages.compare import run_compare_page
from pages.exploration import run_exploration_page
from pages.results import run_results_page

st.title("Analyse des émissions de CO₂ des voitures européennes")

tab = st.sidebar.radio(
    "Sélectionnez l'onglet",
    ["Exploration des données", "Analyse des modèles", "Prédiction du modèle", "Comparaison Marques/Pays"]
)

if tab == "Exploration des données":
    run_exploration_page()
elif tab == "Analyse des modèles":
    run_results_page()
elif tab == "Prédiction du modèle":
    run_predict_page()
elif tab == "Comparaison Marques/Pays":
    run_compare_page()
