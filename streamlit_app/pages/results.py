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

    st.divider()
    
    st.markdown("""
    ## Random Forest SANS Consommation de Carburant
    #### Comparaison : Feature Engineering vs Sans Feature Engineering
    """)
    
    st.markdown("""
    Pour mieux comprendre l'influence des variables techniques (cylindrée, puissance, masse, âge) 
    sur les émissions de CO₂, nous excluons la consommation de carburant qui était trop corrélée.
    """)
    
    # Chargement des métriques RF sans fuel
    metrics_rf_no_fuel_path = Path('fig/rf_no_fuel_metrics.csv')
    comparison_rf_no_fuel_path = Path('fig/rf_no_fuel_comparison.html')
    importance_no_fuel_path = Path('fig/feature_importance_no_fuel.html')
    
    if metrics_rf_no_fuel_path.exists():
        metrics_rf = pd.read_csv(metrics_rf_no_fuel_path).iloc[0]
        r2_no_fe = metrics_rf['r2_no_fe']
        rmse_no_fe = metrics_rf['rmse_no_fe']
        r2_fe = metrics_rf['r2_fe']
        rmse_fe = metrics_rf['rmse_fe']
        
        # Affichage des métriques
        st.markdown("### Métriques de Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sans Feature Engineering")
            st.text(f"Score R²: {r2_no_fe:.4f}")
            st.text(f"RMSE: {rmse_no_fe:.3f}")
        
        with col2:
            st.markdown("#### Avec Feature Engineering")
            st.text(f"Score R²: {r2_fe:.4f}")
            st.text(f"RMSE: {rmse_fe:.3f}")
            
            # Calcul de l'amélioration
            improvement = ((r2_fe - r2_no_fe) / r2_no_fe) * 100
            if improvement > 0:
                st.success(f"📈 Amélioration: +{improvement:.2f}%")
            elif improvement < 0:
                st.error(f"📉 Dégradation: {improvement:.2f}%")
        
        # Graphique de comparaison
        st.subheader("Comparaison des Prédictions")
        
        if comparison_rf_no_fuel_path.exists():
            with open(comparison_rf_no_fuel_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=650, scrolling=True)
        else:
            st.warning("⚠️ Le fichier 'rf_no_fuel_comparison.html' n'a pas été trouvé.")
        
        st.markdown("""
        **Interprétation :** Bien que le feature engineering ne montre pas d'amélioration significative 
        du R² pour ce modèle Random Forest (les arbres de décision sont naturellement invariants à l'échelle), 
        il reste une pratique recommandée pour plusieurs raisons :

        - **Robustesse** : Le modèle sera plus stable face à de nouvelles données avec des échelles différentes et les transformations réduisent l'impact des outliers.
        - **Généralisation** : Utilisation facilitée pour d'autres modèles qui pourraient bénéficier de données standardisées (ex. régression linéaire, SVM)

        Même si l'impact immédiat sur les performances est marginal pour ce modèle spécifique, la standardisation et la normalisation restent importantes si nous voulons pousser plus loin notre cas d'usage.
        """)
        
        # Importance des variables
        st.subheader("Importance des Variables (avec Feature Engineering)")
        
        if importance_no_fuel_path.exists():
            with open(importance_no_fuel_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=650, scrolling=True)
        else:
            st.warning("⚠️ Le fichier 'feature_importance_no_fuel.html' n'a pas été trouvé.")
        
        st.markdown("""
        **Interprétation :** Sans la consommation de carburant, nous pouvons observer l'influence réelle 
        des caractéristiques techniques du véhicule. Après le type de carburant (hybride ou non), La masse, la cylindrée et la puissance sont 
        généralement les variables les plus importantes pour prédire les émissions de CO₂.
        """)
    
    else:
        st.warning("⚠️ Les métriques pour le Random Forest sans fuel consumption n'ont pas été générées. Exécutez make_graphs.py.")

# Point d'entrée pour Streamlit
if __name__ == "__main__":
    run_results_page()
