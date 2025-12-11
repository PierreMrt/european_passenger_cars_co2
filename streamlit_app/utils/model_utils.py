import shap
import pandas as pd


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


def find_similar_less_polluting_cars(user_inputs, data, prediction, top_n=3, tolerance=0.15):
    """
    Trouve les véhicules similaires moins polluants avec le même type de carburant
    et des caractéristiques dans une tolérance de ±10%.
    
    Args:
        user_inputs (dict): Les caractéristiques du véhicule sélectionné par l'utilisateur
        data (pd.DataFrame): Le jeu de données complet (data_processed.csv)
        prediction (float): L'émission de CO₂ prédite pour le véhicule de l'utilisateur
        top_n (int): Nombre de véhicules à retourner
        tolerance (float): Tolérance en pourcentage (0.10 = 10%)
    
    Returns:
        pd.DataFrame: Les véhicules similaires moins polluants
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Colonnes numériques pour calculer la similarité (sans age_months)
    numeric_features = ["ec (cm3)", "ep (KW)", "m (kg)"]
    
    # Filtrer les véhicules avec Cn non nul
    data_clean = data[data['Cn'].notna()].copy()
    
    # Filtrer par le même type de carburant
    if 'Ft' in user_inputs:
        data_clean = data_clean[data_clean['Ft'] == user_inputs['Ft']]
    
    # Filtrer les véhicules moins polluants
    less_polluting = data_clean[data_clean['Ewltp (g/km)'] < prediction].copy()
    
    if len(less_polluting) == 0:
        return pd.DataFrame()
    
    # Appliquer les filtres de tolérance ±10% pour chaque caractéristique
    for feat in numeric_features:
        user_value = user_inputs[feat]
        min_value = user_value * (1 - tolerance)
        max_value = user_value * (1 + tolerance)
        less_polluting = less_polluting[
            (less_polluting[feat] >= min_value) & 
            (less_polluting[feat] <= max_value)
        ]
    
    if len(less_polluting) == 0:
        return pd.DataFrame()
    
    # Préparer les données pour le calcul de similarité
    user_values = np.array([[user_inputs[feat] for feat in numeric_features]])
    car_values = less_polluting[numeric_features].values
    
    # Normaliser les valeurs
    scaler = StandardScaler()
    all_values = np.vstack([user_values, car_values])
    scaled_values = scaler.fit_transform(all_values)
    
    user_scaled = scaled_values[0:1]
    cars_scaled = scaled_values[1:]
    
    # Calculer la similarité cosinus
    similarities = cosine_similarity(user_scaled, cars_scaled)[0]
    
    # Ajouter les scores de similarité
    less_polluting['similarity_score'] = similarities
    
    # Trier par émissions (croissant) puis par similarité (décroissant)
    similar_cars = less_polluting.sort_values(
        by=['Ewltp (g/km)', 'similarity_score'], 
        ascending=[True, False]
    )
    
    # Supprimer les doublons basés sur Cn (garder le premier = le moins polluant)
    similar_cars = similar_cars.drop_duplicates(subset=['Cn'], keep='first')
    
    return similar_cars.head(top_n)
