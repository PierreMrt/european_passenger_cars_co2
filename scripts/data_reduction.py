# Imports
import pandas as pd

# Constants
INPUT_PATH = "../data/data.csv"
OUTPUT_PATH = "../data/data_reduced.csv"

# Functions
def del_columns(df):
    """
    Supprime les colonnes spécifiées du DataFrame si elles existent.

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame contenant les données.

    Retourne :
    ---------
    pandas.DataFrame
        Un DataFrame avec les colonnes spécifiées supprimées si elles existaient.
    """
    # Liste de toutes les colonnes à supprimer
    col_to_del = [
        # Colonnes à supprimer car vides
        'MMS', 'Enedc (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ernedc (g/km)', 'De', 'Vf',
        # Colonnes à supprimer car pas assez remplies
        'RLFI', 'z (Wh/km)', 'Erwltp (g/km)',
        # Colonnes à supprimer car doublons d'information
        'Mp', 'Mh', 'Man', 'Cr', 'm (kg)', 'Fm', 'VFN',
        # Colonnes à supprimer car non pertinentes
        'ID', 'Status', 'r', 'year', 'Tan', 'Va', 'Ve', 'Ct', 'Cr', 'T',
        # Colonnes concernant uniquement les véhicules électriques
        'Electric range (km)'
    ]

    # Filtrer les colonnes qui existent dans le DataFrame
    col_to_del = [col for col in col_to_del if col in df.columns]

    # Suppression des colonnes sélectionnées
    df = df.drop(col_to_del, axis=1)
   
    # suppresion des doublons
    df = df.drop_duplicates()

    return df


def del_rows(df):
    """
    Supprime les lignes du DataFrame correspondant à des véhicules utilisant des types de carburant peu communs.

    Cette fonction filtre le DataFrame pour exclure les véhicules dont le type de carburant
    est considéré comme peu commun ou non conventionnel.

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame contenant les données des véhicules, incluant une colonne 'Ft'
        qui spécifie le type de carburant.

    Retourne :
    ---------
    pandas.DataFrame
        Un DataFrame filtré, ne contenant pas les véhicules utilisant des carburants
        spécifiés comme peu communs.
    """

    exclusions = ['lpg', 'e85', 'ng', 'hydrogen', 'unknown', 'ng-biomethane', 'electric']
    return df[~df['Ft'].str.lower().isin(exclusions)]



if __name__ == '__main__':
    df = pd.read_csv(INPUT_PATH)
    df = del_columns(df)
    df = del_rows(df)
    df.to_csv(OUTPUT_PATH, index=False)