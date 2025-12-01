import streamlit as st
import pandas as pd
from pathlib import Path


def run_results_page():
    """
    Affiche la page de résultats avec la comparaison des modèles et l'importance des variables.
    """
    st.header("Analyse des modèles")
    
    st.markdown("""
    Cette page présente les résultats de la prédiction des émissions de CO₂ en comparant plusieurs approches.
    """)

    st.divider()

    st.markdown("""
    ## Régression linéaire vs Random Forest
    #### Consommation de carburant inclue comme variable prédictive.
    """)
    
    # Chargement des données
    metrics_path = Path('fig/model_metrics.csv')
    comparison_path = Path('fig/predictions_comparison.html')
    importance_path = Path('fig/feature_importance.html')

    # Métriques
    metrics = pd.read_csv(metrics_path).iloc[0]
    r2_lin = metrics['r2_lin']
    rmse_lin = metrics['rmse_lin']
    r2_rf = metrics['r2_rf']
    rmse_rf = metrics['rmse_rf']


    
    # Affichage des métriques
    st.markdown("### Métriques de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Régression Linéaire")
        st.text(f"Score R²: {r2_lin:.4f}")
        st.text(f"RMSE: {rmse_lin:.3f}")
    
    with col2:
        st.markdown("#### Random Forest")
        st.text(f"Score R²: {r2_rf:.4f}")
        st.text(f"RMSE: {rmse_rf:.3f}")
    
    # Graphique de comparaison
    st.subheader("Comparaison des Prédictions")
    
    if comparison_path.exists():
        with open(comparison_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'predictions_comparison.html' n'a pas été trouvé dans le dossier 'fig/'.")
    
    st.markdown("""
    **Interprétation :** Les points plus proches de la ligne rouge (idéale) indiquent de meilleures prédictions. 
    Le Random Forest montre généralement une meilleure performance avec un R² plus élevé.
    """)
    
    # Importance des variables
    st.subheader("Importance des Variables")
    
    if importance_path.exists():
        with open(importance_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=650, scrolling=True)
    else:
        st.warning("⚠️ Le fichier 'feature_importance.html' n'a pas été trouvé dans le dossier 'fig/'.")
    
    st.markdown("""
    **Interprétation :** Ce graphique montre les variables qui ont le plus d'influence sur les prédictions 
    du modèle Random Forest. La consommation de carburant est très fortement corrélée et empêche l'analyse de l'influence des variables techniques.
    """)


# Point d'entrée pour Streamlit
if __name__ == "__main__":
    run_results_page()
