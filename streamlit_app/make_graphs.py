"""
Script pour générer et sauvegarder les graphiques de résultats des modèles.
À exécuter avant de lancer l'application Streamlit pour éviter les temps de chargement.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm
import sys
import os

# Ajouter le chemin parent pour importer les modules utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


pio.templates.default = "plotly_white"

def columns_completion_graph(df):
    """
    Crée et sauvegarde le graphique de complétion des colonnes.
    """

    # Calculer le pourcentage de complétion pour chaque colonne
    completion_stats = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        total_count = len(df)
        completion_pct = (non_null_count / total_count) * 100
        
        completion_stats.append({
            'Colonne': col,
            'Pourcentage de complétion': completion_pct,
            'Valeurs remplies': non_null_count,
            'Valeurs totales': total_count
        })

    # Créer un DataFrame avec les statistiques
    df_stats = pd.DataFrame(completion_stats)

    # Trier par pourcentage de complétion (du plus faible au plus élevé)
    df_stats = df_stats.sort_values('Pourcentage de complétion')

    # Créer le graphique avec Plotly Express
    fig = px.bar(
        df_stats,
        x='Pourcentage de complétion',
        y='Colonne',
        orientation='h',
        title='Taux de remplissage des colonnes (%)',
        labels={
            'Pourcentage de complétion': 'Taux (%)',
            'Colonne': 'Nom de la colonne'
        },
        color='Pourcentage de complétion',
        color_continuous_scale=['#c5d5e8', '#a8c2e0', '#8aaed8', '#6d9bcf', '#4f87c7', '#3174bf'],
        range_color=[0, 100],
        hover_data={
            'Valeurs remplies': True,
            'Valeurs totales': True,
            'Pourcentage de complétion': ':.2f'
        },
        template="plotly_white",
    )

    # Personnaliser l'apparence
    fig.update_layout(
        height=max(600, len(df.columns) * 20),  # Ajuster la hauteur selon le nombre de colonnes
        width=700,
        xaxis_range=[0, 100],
        font=dict(size=10),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        )
    )

    # Personnaliser les axes
    fig.update_xaxes(
        gridcolor='white',
        gridwidth=1,
    )

    fig.update_yaxes(
        gridcolor='white',
        gridwidth=1,
    )

    # Sauvegarder le graphique
    fig.write_html('fig/completion_columns.html')
    print("✓ Graphique sauvegardé: fig/completion_columns.html")


def fuel_type_distribution_graph(df):
    """
    Crée et sauvegarde le graphique de répartition des types de carburant.
    """
    # Calculer les statistiques pour chaque type de carburant
    fuel_stats = df['Ft'].value_counts().reset_index()
    fuel_stats.columns = ['Type de carburant', 'Nombre de véhicules']

    # Calculer les pourcentages
    total_vehicles = fuel_stats['Nombre de véhicules'].sum()
    fuel_stats['Pourcentage'] = (fuel_stats['Nombre de véhicules'] / total_vehicles * 100).round(2)

    # Trier par nombre de véhicules décroissant
    fuel_stats = fuel_stats.sort_values('Nombre de véhicules', ascending=True)

    # Créer le graphique en barres horizontales
    fig = px.bar(
        fuel_stats,
        x='Pourcentage',
        y='Type de carburant',
        orientation='h',
        title='Répartition des véhicules par type de carburant',
        labels={
            'Pourcentage': 'Pourcentage du total (%)',
            'Type de carburant': 'Type de carburant'
        },
        color='Pourcentage',
        color_continuous_scale=['#c5d5e8', '#a8c2e0', '#8aaed8', '#6d9bcf', '#4f87c7', '#3174bf'],
        hover_data={
            'Nombre de véhicules': ':,',
            'Pourcentage': ':.2f'
        },
        text='Pourcentage',
        template="plotly_white"
    )
    
    
    fig.write_html('fig/fuel_type_distribution.html')
    print("✓ Graphique sauvegardé: fig/fuel_type_distribution.html")


def boxplot_by_fuel_type_graph(df):
    """
    Crée et sauvegarde le graphique boxplot de la puissance par type de carburant.
    """
    fig = px.box(
        df,
        x='Ft',
        y='ep (KW)',
        title='Distribution de la puissance par type de carburant',
        labels={
            'Ft': 'Type de carburant',
            'ep (KW)': 'Puissance (KW)'
        },
        color='Ft',
        template="plotly_white"
    )
    
    fig.update_layout(showlegend=False)
    
    fig.write_html('fig/power_boxplot_by_fuel.html')
    print("✓ Graphique sauvegardé: fig/power_boxplot_by_fuel.html")


def cylindree_vs_emissions_graph(df):
    """
    Crée et sauvegarde le graphique de la relation entre cylindrée et émissions de CO₂.
    """
    fig = px.scatter(
        df,
        x='ec (cm3)',
        y='Ewltp (g/km)',
        color='Ft',
        trendline='ols',
        title='Relation entre Cylindrée et Émissions de CO₂',
        labels={
            'ec (cm3)': 'Cylindrée (cm3)',
            'Ewltp (g/km)': 'Émissions de CO₂ (g/km)',
            'Ft': 'Type de carburant'
        },
        template="plotly_white"
    )
    
    fig.write_html('fig/cylindree_vs_emissions.html')
    print("✓ Graphique sauvegardé: fig/cylindree_vs_emissions.html")


def correlation_matrix_graph(df):
    """
    Crée et sauvegarde le graphique de la matrice de corrélation.
    """
    # Sélectionner uniquement les colonnes numériques pertinentes
    numeric_cols = ['Ewltp (g/km)', 'ep (KW)', 'ec (cm3)', 'm (kg)', 'Fuel consumption', 'age_months']
    df_numeric = df[numeric_cols]

    # Calculer la matrice de corrélation
    corr_matrix = df_numeric.corr()

    # Renommer les colonnes pour des labels en français
    label_mapping = {
        'Ewltp (g/km)': 'Émissions CO₂',
        'ep (KW)': 'Puissance',
        'ec (cm3)': 'Cylindrée',
        'm (kg)': 'Masse',
        'Fuel consumption': 'Consommation carburant',
        'age_months': 'Âge (mois)'
    }

    # Appliquer le renommage
    corr_matrix_renamed = corr_matrix.rename(columns=label_mapping, index=label_mapping)
    
    fig = px.imshow(
        corr_matrix_renamed,
        text_auto='.2f',
        title='Matrice de Corrélation des Variables Numériques',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        template="plotly_white"
    )
    
    fig.update_layout(
        xaxis_title='Variables',
        yaxis_title='Variables',
        width=700,
        height=700
    )
    
    fig.write_html('fig/correlation_matrix.html')
    print("✓ Graphique sauvegardé: fig/correlation_matrix.html")


def create_predictions_comparison_graph(y_test, y_pred_lin, y_pred_rf, r2_lin, r2_rf):
    """
    Crée et sauvegarde le graphique de comparaison des prédictions.
    """
    # Création d'un DataFrame pour combiner les données
    df = pd.DataFrame({
        'Valeurs réelles': list(y_test) + list(y_test),
        'Valeurs prédites': list(y_pred_lin) + list(y_pred_rf),
        'Modèle': ['Linéaire'] * len(y_test) + ['Random Forest'] * len(y_test)
    })
    
    # Création du graphique scatter
    fig = px.scatter(
        df,
        x='Valeurs réelles',
        y='Valeurs prédites',
        color='Modèle',
        title='Comparaison : Régression Linéaire vs Random Forest',
        labels={
            'Valeurs réelles': 'Valeurs réelles (Ewltp g/km)',
            'Valeurs prédites': 'Valeurs prédites (Ewltp g/km)'
        },
        color_discrete_map={
            'Linéaire': 'blue',
            'Random Forest': 'orange'
        },
        opacity=0.4,
        template="plotly_white"
    )
    
    # Ajout de la ligne idéale (y=x)
    y_min, y_max = df['Valeurs réelles'].min(), df['Valeurs réelles'].max()
    fig.add_scatter(
        x=[y_min, y_max],
        y=[y_min, y_max],
        mode='lines',
        name='Idéal',
        line=dict(color='red', dash='dash', width=2)
    )
    
    # Mise à jour des légendes avec les scores R²
    fig.for_each_trace(
        lambda trace: trace.update(name=f"Linéaire (R²={r2_lin:.3f})") 
        if trace.name == "Linéaire" 
        else trace.update(name=f"Random Forest (R²={r2_rf:.3f})") 
        if trace.name == "Random Forest" 
        else trace
    )
    
    fig.update_layout(
        xaxis_title='Valeurs réelles (Ewltp g/km)',
        yaxis_title='Valeurs prédites (Ewltp g/km)',
        legend_title='Modèle',
        hovermode='closest',
        width=700
    )
    
    return fig


def create_feature_importance_graph(feature_importances, feature_names, top_n=15):
    """
    Crée et sauvegarde le graphique d'importance des variables.
    """
    # Création d'un DataFrame et tri par importance
    feat_imp_df = pd.DataFrame({
        'Caractéristique': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    # Inversion de l'ordre pour afficher les plus importantes en haut
    feat_imp_df = feat_imp_df.iloc[::-1]
    
    # Création du graphique à barres horizontales
    fig = px.bar(
        feat_imp_df,
        x='Importance',
        y='Caractéristique',
        orientation='h',
        title=f'Importances des variables (Random Forest - Top {top_n})',
        labels={
            'Importance': 'Importance',
            'Caractéristique': ''
        },
        color='Importance',
        color_continuous_scale='Blues',
        template="plotly_white" 
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='',
        showlegend=False,
        coloraxis_showscale=False,
        width=700
    )
    
    return fig


def first_models_graphs(df):
    """
    Fonction principale pour générer tous les graphiques de résultats.
    """
    
    # Préparation des données
    target = "Ewltp (g/km)"
    
    num_features = [
        "Fuel consumption",
        "ec (cm3)",
        "ep (KW)",
        "m (kg)",
        "age_months",
    ]
    
    cat_features = ["Ft"]
    
    num_features = [c for c in num_features if c in df.columns]
    cat_features = [c for c in cat_features if c in df.columns]
    
    data = df[num_features + cat_features + [target]].copy()
    
    for c in num_features + [target]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    
    data = data.dropna(subset=num_features + cat_features + [target])
    
    print("Encodage des données...")
    # Encodage catégoriel
    data_encoded = pd.get_dummies(data, columns=cat_features, drop_first=True)
    
    X = data_encoded.drop(columns=[target])
    y = data_encoded[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Régression linéaire
    print("Entraînement de la régression linéaire...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    
    r2_lin = r2_score(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    
    print(f"Régression linéaire - R²: {r2_lin:.4f}, RMSE: {rmse_lin:.3f}")
    
    # Random Forest
    print("Entraînement du Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    print(f"Random Forest - R²: {r2_rf:.4f}, RMSE: {rmse_rf:.3f}")
    
    # Sauvegarde des métriques
    print("Sauvegarde des métriques...")
    metrics = {
        'r2_lin': r2_lin,
        'rmse_lin': rmse_lin,
        'r2_rf': r2_rf,
        'rmse_rf': rmse_rf
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('fig/model_metrics.csv', index=False)
    
    # Génération et sauvegarde des graphiques
    print("Génération du graphique de comparaison des prédictions...")
    fig_comparison = create_predictions_comparison_graph(
        y_test, y_pred_lin, y_pred_rf, r2_lin, r2_rf
    )
    fig_comparison.write_html('fig/predictions_comparison.html')
    print("✓ Graphique sauvegardé: fig/predictions_comparison.html")
    
    print("Génération du graphique d'importance des variables...")
    fig_importance = create_feature_importance_graph(
        rf_model.feature_importances_,
        X.columns.tolist(),
        top_n=15
    )
    fig_importance.write_html('fig/feature_importance.html')
    print("✓ Graphique sauvegardé: fig/feature_importance.html")
    

if __name__ == "__main__":
    raw_data = pd.read_csv('data/data.csv', low_memory=False)
    reduced_data = pd.read_csv('data/data_reduced.csv', low_memory=False)
    processed_data = pd.read_csv("data/data_processed.csv", low_memory=False)

    columns_completion_graph(raw_data)
    """     fuel_type_distribution_graph(raw_data)
    boxplot_by_fuel_type_graph(reduced_data)
    cylindree_vs_emissions_graph(processed_data)
    correlation_matrix_graph(processed_data)
    first_models_graphs(processed_data) """

    print("\n✅ Tous les graphiques ont été générés avec succès!")