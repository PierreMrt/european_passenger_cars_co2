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
    """
    Corrige le type de données de la colonne "Date of registration" dans le DataFrame
    en le convertissant au format datetime.

    Cette fonction est utilisée pour s'assurer que la colonne contenant les dates
    d'enregistrement est au bon format de données, ce qui facilite les opérations
    et analyses ultérieures sur les dates.

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame contenant les données, incluant une colonne nommée
        "Date of registration" qui doit être convertie en type datetime.

    Retourne :
    ---------
    pandas.DataFrame
        Le DataFrame avec la colonne "Date of registration" convertie en type datetime.
    """
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



def clean_fuel_type(df):
    """
    Nettoie les valeurs de la colonne 'Ft' du DataFrame en les convertissant en minuscules.

    Cette fonction standardise les valeurs de la colonne représentant le type de carburant
    en les mettant toutes en minuscules. Cela facilite les comparaisons et les analyses
    ultérieures en évitant les problèmes liés à la casse.

    Paramètres :
    ------------
    df : pandas.DataFrame
        Le DataFrame contenant les données, incluant une colonne nommée 'Ft'
        qui représente le type de carburant.

    Retourne :
    ---------
    pandas.DataFrame
        Le DataFrame avec les valeurs de la colonne 'Ft' converties en minuscules.
    """
    df['Ft'] = df['Ft'].str.lower()
    return df



def clean_maker_names(df):
    # complexe
    pass


def preprocessing(df):
    df = clean_columns(df)
    df = correct_data_type(df)
    df = clean_fuel_type(df)
    df = remove_outliers(df, 'ep (KW)')

    return df






