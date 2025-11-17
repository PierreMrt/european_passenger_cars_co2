import matplotlib.pyplot as plt
from .model_utils import prettify_feature_name

def brand_country_comparison_plot(data, group_by, metric):
    """
    Génère un graphique comparant une métrique moyenne par marque ou par pays.

    Args:
        data (pandas.DataFrame): Données contenant les émissions et caractéristiques des véhicules.
        group_by (str): Colonne pour grouper les données, soit 'Mk' (Marque) ou 'Country' (Pays).
        metric (str): Nom de la métrique à analyser (Émissions de CO₂, Taille du moteur, Puissance, Âge).

    Returns:
        matplotlib.figure.Figure: Figure du graphique à barres.
    """
    metric_map = {
        "Émissions de CO₂": "Ewltp (g/km)",
        "Taille du moteur": "ec (cm3)",
        "Puissance": "ep (KW)",
        "Âge": "age_months"
    }
    group_col = "Mk" if group_by == "Marque" else "Country"
    metric_col = metric_map.get(metric, metric)

    grouped = data.groupby(group_col)[metric_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} par {group_by}")
    plt.tight_layout()
    return fig


def plot_shap_values(shap_values, feature_names, feature_labels):
    """
    Trace un graphique à barres horizontales des valeurs SHAP des caractéristiques.

    Args:
        shap_values (list or array): Valeurs SHAP d'une prédiction.
        feature_names (list): Noms transformés des caractéristiques correspondantes.
        feature_labels (dict): Dictionnaire de noms pour les caractéristiques.

    Returns:
        matplotlib.figure.Figure: Figure du graphique SHAP.
    """
    labels = [prettify_feature_name(name, feature_labels) for name in feature_names]
    shap_pair = sorted(zip(labels, shap_values), key=lambda x: abs(x[1]), reverse=True)[:5]
    feat, vals = zip(*shap_pair)
    colors = ['crimson' if v > 0 else 'green' for v in vals]
    
    fig, ax = plt.subplots()
    bars = ax.barh(feat, vals, color=colors)
    ax.set_xlabel("Valeur SHAP")
    ax.set_title("Contribution des variables (SHAP)")
    ax.axvline(0, color='gray', linewidth=1)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    return fig