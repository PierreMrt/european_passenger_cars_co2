import streamlit as st
import plotly.io as pio
from pathlib import Path

def run_exploration_page():
    """
    Affiche la page d'exploration des données avec les graphiques générés.
    Inclut le taux de complétion des colonnes et la répartition des types de carburant.
    """
    st.header("Exploration des Données")
    
    st.markdown("""
    Cette page présente une analyse exploratoire du jeu de données brut des émissions de CO₂ 
    des voitures européennes avant traitement.
    """)
    
    # Section 1: Taux de complétion des colonnes
    st.subheader("📊 Taux de complétion des colonnes")
    st.markdown("""
    Ce graphique montre le pourcentage de valeurs renseignées pour chaque colonne du dataset.
    Les colonnes avec un faible taux de complétion ont été identifiées pour un nettoyage ultérieur.
    """)
    
    # Charger et afficher le graphique de complétion
    completion_path = Path("fig/completion_columns.html")
    if completion_path.exists():
        with open(completion_path, 'r', encoding='utf-8') as f:
            completion_html = f.read()
        st.components.v1.html(completion_html, height=800, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'completion_columns.html' n'a pas été trouvé dans le dossier 'fig/'.")
    
    # Séparateur
    st.divider()
    
    # Section 2: Répartition des types de carburant
    st.subheader("⛽ Répartition des types de carburant")
    st.markdown("""
    Ce graphique présente la distribution des véhicules selon leur type de carburant.
    Cette analyse aide à identifier les carburants dominants et les types rares en vue de garder seulement ceux pertinents.
    """)
    
    # Charger et afficher le graphique des types de carburant
    fuel_path = Path("fig/fuel_type_distribution.html")
    if fuel_path.exists():
        with open(fuel_path, 'r', encoding='utf-8') as f:
            fuel_html = f.read()
        st.components.v1.html(fuel_html, height=600, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'fuel_type_distribution.html' n'a pas été trouvé dans le dossier 'fig/'.")

    # Séparateur
    st.divider()
    
    # Section 3: Puissance par carburant
    st.subheader("🔋 Distribution de la puissance par type de carburant")
    st.markdown("""
    Ce graphique la puissance des véhicules par type de carburant, montrant l'existence d'outliers, probablement des véhicules de sport qui pourraient biaiser le modèle.
    """)
    
    # Charger et afficher le graphique puissance vs émissions
    power_path = Path("fig/power_boxplot_by_fuel.html")
    if power_path.exists():
        with open(power_path, 'r', encoding='utf-8') as f:
            power_html = f.read()
        st.components.v1.html(power_html, height=750, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'power_boxplot_by_fuel.html' n'a pas été trouvé dans le dossier 'fig/'.")

    # Séparateur
    st.divider()

    # Section 4: Cylindrée vs Émissions
    st.subheader("🔧 Relation entre Cylindrée et Émissions de CO₂")
    st.markdown("""
    Ce graphique illustre la corrélation entre la cylindrée du moteur et les émissions de CO₂.
    Les lignes de régression par type de carburant montrent comment cette relation varie selon
    le combustible utilisé.
    """)
    
    cylindree_path = Path("fig/cylindree_vs_emissions.html")
    if cylindree_path.exists():
        with open(cylindree_path, 'r', encoding='utf-8') as f:
            cylindree_html = f.read()
        st.components.v1.html(cylindree_html, height=750, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'cylindree_vs_emissions.html' n'a pas été trouvé dans le dossier 'fig/'.")
    
    st.divider()
    
    # Section 5: Matrice de corrélation
    st.subheader("📈 Matrice de Corrélation des Caractéristiques")
    st.markdown("""
    Cette matrice de corrélation révèle les relations linéaires entre les différentes caractéristiques
    techniques des véhicules. Les valeurs proches de 1 (rouges) indiquent une forte corrélation positive,
    tandis que les valeurs proches de -1 (bleues) indiquent une corrélation négative.
    """)
    
    corr_path = Path("fig/correlation_matrix.html")
    if corr_path.exists():
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_html = f.read()
        st.components.v1.html(corr_html, height=650, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'correlation_matrix.html' n'a pas été trouvé dans le dossier 'fig/'.")

if __name__ == "__main__":
    run_exploration_page()
