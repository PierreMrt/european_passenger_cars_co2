# Imports
import pandas as pd
import string
from datetime import datetime

# Constants
INPUT_PATH = "../data/data_reduced.csv"
OUTPUT_PATH = "../data/data_processed.csv"

# Functions
def clean_columns(df):
    """
    Nettoie les noms des colonnes d'un DataFrame en supprimant les espaces superflus au début et à la fin.

    Paramètres :
    df (pandas.DataFrame) : Le DataFrame dont les noms de colonnes doivent être nettoyés.

    Retourne :
    pandas.DataFrame : Le DataFrame avec les noms de colonnes nettoyés.
    """
    df.columns = df.columns.str.strip()
    return df



def registration_date_as_age(df):
    """
    Convertit la colonne 'Date of registration' en âge du véhicule exprimé en mois,
    puis retire la colonne de date initiale.

    Cette fonction facilite l'exploitation de l'ancienneté des véhicules en la traduisant
    sous forme quantitative (âge en mois), adaptée à la modélisation prédictive. 
    Elle supprime la colonne de date d'origine une fois convertie.

    Paramètres :
    ------------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'Date of registration' avec les dates d'enregistrement sous forme 
        de chaîne de caractères ou datetime.

    Retourne :
    ---------
    pandas.DataFrame
        Le DataFrame avec une nouvelle colonne 'age_months' indiquant l'âge du véhicule en mois,
        la colonne 'Date of registration' supprimée.
    """
    df["Date of registration"] = pd.to_datetime(df["Date of registration"])
    today = pd.Timestamp(datetime.now().date())

    df['age_months'] = (today.year - df['Date of registration'].dt.year) * 12 + (today.month - df['Date of registration'].dt.month)
    df['age_months'] = df['age_months'].where(df['Date of registration'].notna())

    df = df.drop('Date of registration', axis=1)

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
    """
    Nettoie et homogénéise les noms de fabricants dans un DataFrame :
    - Supprime la ponctuation, les espaces superflus et met en majuscules.
    - Pour chaque nom de fabricant, si plusieurs noms uniques partagent le même premier mot,
      remplace le nom par ce premier mot dans la colonne 'Mk'.
    - Cela permet de regrouper automatiquement les variantes comme 'MITSUBISHI MOTORS CORPORATION' et 'MITSUBISHI' sous 'MITSUBISHI'.


    Paramètres :
    df (pandas.DataFrame) : DataFrame contenant une colonne 'Mk' avec les noms des fabricants.


    Retour :
    pandas.DataFrame : Le DataFrame avec des noms de fabricants harmonisés dans la colonne 'Mk'.
    """
    translator = str.maketrans('', '', string.punctuation)
    df['Mk'] = df['Mk'].apply(lambda x: x.translate(translator) if isinstance(x, str) else x)
    df['Mk'] = df['Mk'].str.strip()
    df['Mk'] = df['Mk'].str.upper()

    # Récupère les noms uniques et leur premier mot
    names = df['Mk'].dropna().unique()
    first_words = [str(name).split()[0] for name in names]
    first_word_counts = pd.Series(first_words).value_counts()
    duplicated_firsts = set(first_word_counts[first_word_counts > 1].index)

    # Crée un mapping : si le premier mot est dupliqué, on ne garde que lui
    name_map = {}
    for name in names:
        first = str(name).split()[0]
        if first in duplicated_firsts:
            name_map[name] = first
        else:
            name_map[name] = name

    df['Mk'] = df['Mk'].map(name_map)

    return df



def pareto_major_brands(df, threshold=80):
    """
    Garde uniquement les marques qui constituent les X% les plus fréquents de la colonne,
    le reste est regroupé sous 'OTHERS' (règle de Pareto).
    
    Paramètres :
        df : pandas.DataFrame
        threshold : float, seuil de pourcentage cumulé

    Retour :
        df (pandas.DataFrame) : Colonne modifiée où seules les marques majeures sont conservées.
    """
    counts = df['Mk'].value_counts()
    cum_pct = counts.cumsum() / counts.sum() * 100

    major_brands = cum_pct[cum_pct <= threshold].index

    df['Mk'] = df['Mk'].apply(lambda x: x if x in major_brands else 'OTHERS')
    return df


def manage_missing_values(df):
    """
    Traite les valeurs manquantes en préparation d'une modélisation prédictive.

    - Supprime les lignes où la variable cible 'Ewltp (g/km)' est manquante, car elles ne peuvent pas être utilisées pour l'entraînement ou la prédiction.
    - Remplit les valeurs manquantes de la colonne 'ec (cm3)' par la valeur la plus fréquente (mode), 
      cette colonne présentant une distribution multimodale correspondant typiquement à des tailles de moteur standards.
    - Remplit les valeurs manquantes de la colonne 'm (kg)' (masse du véhicule) par la médiane, 
      car la distribution de cette colonne est unimodale mais asymétrique vers la droite (présence de véhicules plus lourds).
    - Remplit les valeurs manquantes de la colonne 'age_months' (âge du véhicule en mois) par la médiane, 
      correspondant à une variable numérique continue représentant l'ancienneté du véhicule.
      
    Paramètres :
        df (pandas.DataFrame): Le DataFrame contenant les données à nettoyer.

    Retour :
        pandas.DataFrame: Le DataFrame nettoyé, prêt pour la modélisation.
    """
    df = df[df['Ewltp (g/km)'].notna()].copy()

    mode_value = df['ec (cm3)'].mode()[0]
    df.loc[:, 'ec (cm3)'] = df['ec (cm3)'].fillna(mode_value)

    median_value = df['m (kg)'].median()
    df.loc[:, 'm (kg)'] = df['m (kg)'].fillna(median_value)

    median_age_months = df['age_months'].median()
    df.loc[:, 'age_months'] = df['age_months'].fillna(median_age_months)

    return df



def processing(csv=True):
    df = pd.read_csv(INPUT_PATH)

    df = clean_columns(df)
    df = registration_date_as_age(df)
    df = clean_fuel_type(df)
    df = clean_maker_names(df)
    df = pareto_major_brands(df, threshold=90)
    df = remove_outliers(df, 'ep (KW)')
    df = manage_missing_values(df)
    df = df.drop_duplicates()

    if csv:
        df.to_csv(OUTPUT_PATH, index=False)
    else:
        return df


if __name__ == '__main__':
    processing(csv=True)