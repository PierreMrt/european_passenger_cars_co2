import streamlit as st
from utils.data_loaders import load_processed_data
from utils.viz_tools import brand_country_comparison_plot

def run_compare_page():
    """
    Affiche la page de comparaison des émissions et caractéristiques par marque ou pays.
    Permet à l'utilisateur de sélectionner une métrique et un groupe, puis affiche un graphique correspondant.
    """
    st.header("Comparaison par Marque et Pays")
    data = load_processed_data()

    metric_options = {
        "Émissions de CO₂": "Ewltp (g/km)",
        "Taille du moteur": "ec (cm3)",
        "Puissance": "ep (KW)",
        "Âge": "age_months"
    }

    group_by_options = {
        "Marque": "Mk",
        "Pays": "Country"
    }

    selected_metric_fr = st.selectbox("Métrique", list(metric_options.keys()))
    group_by_fr = st.radio("Grouper par", list(group_by_options.keys()))

    selected_metric = metric_options[selected_metric_fr]
    group_by = group_by_options[group_by_fr]

    if st.button("Afficher la comparaison"):
        fig = brand_country_comparison_plot(data, group_by, group_by_fr, selected_metric_fr)
        st.plotly_chart(fig, width='stretch')


