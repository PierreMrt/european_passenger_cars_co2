# Imports
import pandas as pd

# Constants
INPUT_PATH = "data/data_reduced.csv"
OUTPUT_PATH = "data/data_processed.csv"

# Functions
def clean_columns(df):
    """
    Nettoie les noms des colonnes du DataFrame en supprimant les espaces superflus au début et à la fin.

    Cette fonction applique la méthode `str.strip()` à chaque nom de colonne du DataFrame
    pour éliminer les espaces en trop, ce qui peut être utile pour uniformiser les noms de colonnes.

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame dont les noms de colonnes doivent être nettoyés.

    Retourne :
    ---------
    pandas.Index
        Un objet Index contenant les noms des colonnes nettoyés.
    """
    return df.columns.str.strip()



def correct_data_type(df):
    df["Date of registration"] = pd.to_datetime(df["Date of registration"])
    return df



def remove_outliers(df, value_col):
    """
    Supprime les lignes du DataFrame correspondant à des valeurs aberrantes élevées
    dans la colonne spécifiée, basée sur la méthode de l'intervalle interquartile (IQR).

    Cette fonction est utilisée pour éliminer les voitures ayant une puissance trop élevée,
    souvent associées aux voitures de sport, en se basant sur la colonne "ep (KW)".

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame contenant les données des voitures.
    value_col : str
        Le nom de la colonne sur laquelle détecter et supprimer les valeurs aberrantes.
        Dans ce contexte, il s'agit de "ep (KW)", représentant la puissance des voitures.

    Retourne :
    ---------
    pandas.DataFrame
        Un DataFrame filtré sans les valeurs aberrantes élevées dans la colonne spécifiée.
    """
    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)

    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    return df[df[value_col] < upper]



def preprocessing(df):
    df = clean_columns(df)
    df = correct_data_type(df)
    df = remove_outliers(df, 'ep (KW)')

    return df






