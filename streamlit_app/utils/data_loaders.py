import pandas as pd
import joblib

def load_processed_data():
    """
    Charge les données prétraitées à partir d'un fichier CSV.

    Returns:
        pandas.DataFrame: Jeu de données prétraité.
    """
    return pd.read_csv("data/data_processed.csv")

def load_model():
    """
    Charge le modèle entraîné sauvegardé avec joblib.

    Returns:
        Pipeline ou modèle scikit-learn chargé.
    """
    return joblib.load("models/random_forest_model.jbl.lzma")

def get_input_features():
    """
    Fournit les valeurs par défaut des caractéristiques d'entrée pour la prédiction.

    Returns:
        dict: Dictionnaire des caractéristiques avec leurs valeurs par défaut.
    """
    return {
        "ec (cm3)": 1199.0,
        "ep (KW)": 130.0,
        "m (kg)": 1500.0,
        "age_months": 52.0,
        "Ft": "petrol"
    }

