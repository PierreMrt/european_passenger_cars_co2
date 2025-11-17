import shap


def calculate_emission_percentile(predicted_emission, data):
    """
    Calcule le percentile d'une émission prédite par rapport à un jeu de données.
    
    Args:
        predicted_emission (float): Valeur d'émission de CO₂ prédite.
        data (pandas.DataFrame): Jeu de données contenant la colonne 'Ew (g/km)'.
    
    Returns:
        float: Percentile de l'émission (0-100).
    """
    emissions = data['Ewltp (g/km)'].dropna()
    percentile = (emissions < predicted_emission).sum() / len(emissions) * 100
    return percentile

def predict_emission(model, df):
    """
    Effectue une prédiction d'émission de CO₂ à partir d'un modèle entraîné et d'une entrée de données.

    Args:
        model: Pipeline ou modèle scikit-learn entraîné.
        df (pandas.DataFrame): Données prétraitées à utiliser pour la prédiction.

    Returns:
        float: Valeur prédite de l'émission de CO₂.
    """
    return model.predict(df)[0]


def explain_prediction(model, df):
    """
    Génère une explication locale des prédictions à l'aide de SHAP.

    Args:
        model: Pipeline scikit-learn contenant un préprocesseur et un régresseur.
        df (pandas.DataFrame): Données d'entrée non transformées à expliquer.

    Returns:
        dict ou str: Dictionnaire contenant les noms des caractéristiques, les valeurs SHAP associées 
        et un sous-dictionnaire pour affichage, ou un message d'erreur si SHAP n'est pas disponible.
    """
    try:
        preprocessor = model.named_steps.get('features')
        regressor = None
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_importances_"):
                regressor = step
                break
        X_transformed = preprocessor.transform(df)
        
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_transformed)

        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = df.columns

        top_indices = sorted(range(len(shap_values[0])), key=lambda i: abs(shap_values[0][i]), reverse=True)[:5]
        return {
            'dict': {feature_names[i]: shap_values[0][i] for i in top_indices},
            'shap_values': [shap_values[0][i] for i in top_indices],
            'feature_names': [feature_names[i] for i in top_indices]
        }
    except Exception as e:
        return f"SHAP non disponible ({e})"


def prettify_feature_name(name, feature_labels):
    """
    Donne un nom convivial aux caractéristiques pour une meilleure lisibilité.

    Args:
        name (str): Nom de la caractéristique transformée.
        feature_labels (dict): Dictionnaire de correspondance entre noms techniques et conviviaux.

    Returns:
        str: Nom convivial ou nom original si aucune correspondance.
    """
    if '__' in name:
        orig, level = name.split('__', 1)
        base = feature_labels.get(orig, orig)
        level = level.replace("Ft_", "")
        return f"{base} : {level}"
    return feature_labels.get(name, name)
