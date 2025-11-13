import pandas as pd
import logging
import joblib
import argparse

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.ensemble import RandomForestRegressor

from feature_engineering import get_feature_transformer
from data_reduction import reduction as reduce_data
from preprocessing import processing as preprocess

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'), 
        logging.StreamHandler()   
    ]
)
logger = logging.getLogger(__name__)

# Chemins des fichiers
INPUT_DATA = 'data/data_processed.csv'
MODEL_OUTPUT = 'models/random_forest_model.jbl.lzma'

def load_data(input_path):
    """
    Charge les données depuis un fichier CSV.
    Args:
        input_path (str): Chemin vers le fichier CSV à charger.
    Returns:
        pd.DataFrame: DataFrame contenant les données chargées.
    """
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Données chargées depuis {input_path}")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")
        raise

def prepare_data(df, target, features):
    """
    Prépare les données pour l'entraînement et le test.
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        target (str): Nom de la colonne cible.
        features (list): Liste des colonnes à utiliser comme features.
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Données préparées et divisées en ensembles d'entraînement/test.")
    return X_train, X_test, y_train, y_test

def get_pipeline():
    """
    Crée et retourne un Pipeline scikit-learn.
    Returns:
        Pipeline: Pipeline configuré.
    """
    feature_transformer = get_feature_transformer()
    pipeline = Pipeline([
        ('features', feature_transformer),
        ('model', RandomForestRegressor(
            bootstrap=True,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=100
        ))
    ])
    logger.info("Pipeline créé avec succès.")
    return pipeline

def train_and_evaluate_model(X_train, y_train):
    """
    Entraîne et évalue le modèle avec validation croisée.
    Args:
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Cible d'entraînement.
    Returns:
        tuple: pipeline (Pipeline entraîné), cv_results (résultats de la validation croisée).
    """
    pipeline = get_pipeline()
    scoring = {
        "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
        "r2": "r2",
    }
    logger.info("Début de la validation croisée...")
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=5,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    logger.info("Validation croisée terminée.")

    # Fit le pipeline sur l'ensemble d'entraînement complet
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline entraîné sur l'ensemble d'entraînement complet.")
    return pipeline, cv_results

def evaluate_test_set(pipeline, X_test, y_test):
    """
    Évalue le Pipeline sur l'ensemble de test.
    Args:
        pipeline (Pipeline): Pipeline entraîné.
        X_test (pd.DataFrame): Features de test.
        y_test (pd.Series): Cible de test.
    Returns:
        tuple: test_mse, test_r2.
    """
    y_pred = pipeline.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    logger.info(f"Évaluation sur l'ensemble de test : MSE = {test_mse:.4f}, R² = {test_r2:.4f}")
    return test_mse, test_r2

def save_model(model, output_path):
    """
    Sauvegarde le modèle ou le Pipeline entraîné.
    Args:
        model: Modèle ou Pipeline à sauvegarder.
        output_path (str): Chemin où sauvegarder le modèle.
    """
    try:
        joblib.dump(model, output_path, compress=9)
        logger.info(f"Modèle sauvegardé avec succès dans {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")
        raise

def run_pipeline(df=None):
    """
    Exécute la pipeline de traitement et de modélisation.
    Args:
        df (pd.DataFrame, optionnel): DataFrame à utiliser. Si None, charge les données depuis INPUT_DATA.
    """
    # Paramètres
    target = "Ewltp (g/km)"
    features = ["m (kg)", "Ft", "ec (cm3)", "ep (KW)", "age_months"]

    # Chargement ou utilisation du DataFrame fourni
    if df is None:
        df = load_data(INPUT_DATA)
        logger.info("Données chargées depuis le fichier par défaut.")

    # Sauvegarde des données traitées (sauf si on commence à partir de 'preprocessed')
    if not isinstance(df, str):  # Si df n'est pas un chemin (cas où on commence à 'preprocessed')
        df.to_csv(INPUT_DATA, index=False)
        logger.info(f"Données traitées sauvegardées dans {INPUT_DATA}.")

    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(df, target, features)

    # Entraînement et évaluation du modèle
    pipeline, cv_results = train_and_evaluate_model(X_train, y_train)

    # Affichage des résultats de la validation croisée
    logger.info("Résultats de la validation croisée :")
    logger.info(f"  MSE d'entraînement : {-cv_results['train_neg_mse'].mean():.4f}")
    logger.info(f"  R² d'entraînement : {cv_results['train_r2'].mean():.4f}")
    logger.info(f"  MSE de validation : {-cv_results['test_neg_mse'].mean():.4f}")
    logger.info(f"  R² de validation : {cv_results['test_r2'].mean():.4f}")

    # Évaluation sur l'ensemble de test
    test_mse, test_r2 = evaluate_test_set(pipeline, X_test, y_test)

    # Sauvegarde du Pipeline
    save_model(pipeline, MODEL_OUTPUT)

def main(start_from="raw"):
    """
    Fonction principale pour exécuter la pipeline en fonction du point de départ.
    Args:
        start_from (str): Point de départ de la pipeline ('raw', 'reduced', 'preprocessed').
    """
    logger.info(f"Début de la pipeline avec le point de départ : {start_from}")

    if start_from == "raw":
        # Exécuter toutes les étapes
        logger.info(f"Début de la réduction des données...")
        df = reduce_data(csv=False)
        logger.info(f"Données réduites avec succès.")
        logger.info(f"Début du prétraitement des données...")
        df = preprocess(df, csv=False)
        logger.info(f"Données prétraitées avec succès.")
        run_pipeline(df)
    elif start_from == "reduced":
        # Commencer après la réduction des données
        logger.info(f"Début du prétraitement des données...")
        df = preprocess(csv=False)
        logger.info(f"Données prétraitées avec succès.")
        run_pipeline(df)
    elif start_from == "preprocessed":
        # Commencer après le prétraitement (utiliser les données déjà traitées)
        run_pipeline()
    else:
        raise ValueError("Valeur invalide pour 'start_from'. Utilisez 'raw', 'reduced' ou 'preprocessed'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de modélisation pour les émissions de CO2 des voitures.")
    parser.add_argument(
        "--start-from",
        choices=["raw", "reduced", "preprocessed"],
        default="raw",
        help="Point de départ pour la pipeline : 'raw' (données brutes), 'reduced' (après réduction), ou 'preprocessed' (après prétraitement)."
    )
    args = parser.parse_args()
    main(args.start_from)
