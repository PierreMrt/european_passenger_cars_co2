from .model_utils import prettify_feature_name

import pandas as pd
import plotly.express as px


def plot_shap_values(shap_values, feature_names, feature_labels):
    """
    Trace un graphique à barres horizontales des valeurs SHAP avec Plotly Express.

    Args:
        shap_values (list or array): Valeurs SHAP d'une prédiction.
        feature_names (list): Noms transformés des caractéristiques correspondantes.
        feature_labels (dict): Dictionnaire de noms pour les caractéristiques.

    Returns:
        plotly.graph_objs._figure.Figure: Figure du graphique SHAP.
    """
    # 1. Préparation des données
    labels = [prettify_feature_name(name, feature_labels) for name in feature_names]
    
    # Création d'un DataFrame pour manipuler facilement les tris et couleurs
    df = pd.DataFrame({
        'Feature': labels,
        'SHAP Value': shap_values
    })
    
    # Calcul de la valeur absolue pour le tri
    df['Abs Value'] = df['SHAP Value'].abs()
    
    # Tri par valeur absolue décroissante et sélection du top 5
    df_top = df.sort_values(by='Abs Value', ascending=False).head(5)
    
    # Définition de la couleur selon le signe (> 0 = Crimson, sinon Green)
    df_top['Color'] = df_top['SHAP Value'].apply(lambda x: 'crimson' if x > 0 else 'green')

    # 2. Création du graphique
    # On inverse l'ordre du DF pour que Plotly affiche le plus grand en haut (l'axe Y se dessine de bas en haut)
    df_top = df_top.iloc[::-1] 

    fig = px.bar(
        df_top,
        x='SHAP Value',
        y='Feature',
        orientation='h',
        title="Contribution des variables (SHAP)",
        text='SHAP Value', 
    )

    # 3. Personnalisation du style
    fig.update_traces(
        marker_color=df_top['Color'],  
        texttemplate='%{text:.3f}',   
        textposition='outside'        
    )

    fig.update_layout(
        xaxis_title="Valeur SHAP",
        yaxis_title=None,              
        showlegend=False,             
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Ajout de la ligne verticale à 0
    fig.add_vline(x=0, line_width=1, line_color="gray")

    return fig