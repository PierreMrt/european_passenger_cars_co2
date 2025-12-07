# Analyse des Émissions de CO₂ des Voitures Européennes

Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).

Ce projet vise donc à analyser et modéliser les émissions de CO₂ des voitures européennes en utilisant des techniques de **prétraitement des données**, **d'ingénierie des caractéristiques**, et de **modélisation machine learning**.

---

## Table des Matières

- [Jeu de données](#données-utilisées)
- [Structure du Projet](#structure-du-projet)
- [Description des Fichiers](#description-des-fichiers)
  - [`data_reduction.py`](#data_reductionpy)
  - [`preprocessing.py`](#preprocessingpy)
  - [`feature_engineering.py`](#feature_engineeringpy)
  - [`pipepline.py`](#pipeplinepy)
- [Prérequis](#prérequis)
- [Utilisation](#utilisation)
- [Streamlit](#application-streamlit)
- [Résultats](#résultats-du-modèle)

---

## Données utilisées

Le jeu de données utilisé est le suivant, en prenant les véhicules belges, français et allemand, immatriculés entre 2022 et 2024:

https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b


## Structure du Projet

```
european_passenger_cars_co2/
│
├── data/
│   └── data_processed.csv              # Données prétraitées
│
├── exploration/
│   └── *.ipynb                         # Notebooks de travail et d'exploration des données
│
├── fig/
│   ├── *.csv                           # Métriques des différents modèles
│   └── *.html                          # Figures et graphiques interactifs (Plotly)
│
├── logs/
│   └── training.log                    # Logs des entraînements de modèles
│
├── models/
│   └── random_forest_model.jbl.lzma    # Modèle entraîné sauvegardé (compressé)
│
├── scripts/
│   ├── data_reduction.py               # Réduction et nettoyage des données brutes
│   ├── feature_engineering.py          # Ingénierie des caractéristiques
│   ├── pipeline.py                     # Pipeline de modélisation complète
│   └── preprocessing.py                # Prétraitement des données
│
├── streamlit_app/
│   ├── pages/                          
│   │   ├── exploration.py              # Page d'exploration des données
│   │   ├── predict.py                  # Page de prédiction interactive
│   │   └── results.py                  # Page d'analyse des résultats des modèles
│   ├── utils/                          
│   │   ├── data_loaders.py             # Chargement des données
│   │   ├── model_utils.py              # Utilitaires pour les modèles
│   │   └── viz_tools.py                # Outils de visualisation
│   ├── app.py                          # Application principale Streamlit
│   └── make_graphs.py                  # Génération des graphiques pour Streamlit (stockés dans fig/)
│
├── .gitattributes                      # Configuration pour Git LFS
├── .gitignore                          # Fichiers à ignorer par Git
├── LICENSE                             # Licence du projet (MIT)
├── README.md                           # Documentation du projet
├── requirements.txt                    # Dépendances Python du projet
└── Table-definition.xlsx               # Définition des colonnes du dataset
```

---

## Description des Fichiers

### `data_reduction.py`
Ce script est responsable de la **réduction et du nettoyage des données brutes**. Il inclut les fonctionnalités suivantes :
- Suppression des doublons.
- Gestion des valeurs manquantes.
- Sélection des colonnes pertinentes pour l'analyse.

---

### `preprocessing.py`
Ce script est dédié au **nettoyage et prétraitement des données** avant leur utilisation dans un modèle de machine learning. Il inclut plusieurs étapes essentielles :
- Nettoyage des noms de colonnes et des valeurs des données.
- Conversion des dates en caractéristiques numériques (comme l'âge des véhicules).
- Suppression des valeurs aberrantes et gestion des valeurs manquantes.
- Normalisation des noms de fabricants et des types de carburant.

Ce script prépare les données brutes pour qu'elles soient prêtes à être utilisées dans les étapes suivantes d'ingénierie des caractéristiques et de modélisation.


---

### `feature_engineering.py`
Ce script est responsable de **l'ingénierie des caractéristiques** en utilisant `ColumnTransformer` de scikit-learn. Il définit les transformations à appliquer aux différentes colonnes des données :
- Normalisation et mise à l'échelle des caractéristiques numériques (comme la cylindrée, la puissance, la masse, et l'âge des véhicules).
- Encodage des caractéristiques catégorielles (comme le type de carburant).

Ce transformateur est conçu pour être intégré dans un pipeline scikit-learn afin de préparer les données pour l'entraînement du modèle.

---

### `pipepline.py`
Ce script contient le **pipeline complet** pour entraîner et évaluer un modèle de machine learning. Il utilise :
- argparse pour la modularité
- Un pipeline scikit-learn pour enchaîner les étapes de feature engineering et de modélisation.
- Validation croisée pour évaluer les performances du modèle.
- Sauvegarde du modèle entraîné.

---

## Prérequis

Pour exécuter ce projet, vous aurez besoin des bibliothèques Python suivantes :
- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `joblib`
- `plotly`
- `streamlit`


Créez un environnement virtuel :
```bash
python -m venv .venv
```

Activez le :
- Sur windows : `source .venv/Script/activate`
- Sur linux/Mac : `source .venv/bin/activate`

Installez-les avec la commande suivante :
```bash
pip install -r requirements.txt
```

---

## Utilisation

### Lancer la pipeline complète

La pipeline peut être exécutée en une seule commande avec différents points de départ :

- **À partir des données brutes** (exécute toutes les étapes : réduction, prétraitement, entraînement) :
  ```bash
  python scripts/pipeline.py --start-from raw
  ```

- **À partir des données réduites** (exécute le prétraitement et l'entraînement) :
  ```bash
  python scripts/pipeline.py --start-from reduced
  ```

- **À partir des données prétraitées** (exécute uniquement l'entraînement) :
  ```bash
  python scripts/pipeline.py --start-from preprocessed
  ```

### Lancer les étapes individuelles

Les étapes de la pipeline peuvent également être exécutées individuellement:

1. **Prétraitement des données** :
   Exécutez le script `data_reduction.py` pour nettoyer et réduire les données brutes :
   ```bash
   python scripts/data_reduction.py
   ```

2. **Ingénierie des caractéristiques** :
   Exécutez le script `preprocessing.py` pour nettoyer et gérer les valeurs manquantes :
   ```bash
   python scripts/preprocessing.py
   ```

3. **Entraînement du modèle** :
   Exécutez le script `pipeline.py` pour entraîner le modèle et sauvegarder les résultats (équivalent à `--start-from preprocessed`) :
   ```bash
   python scripts/pipeline.py
   ```

### Remarques

- Assurez-vous que les fichiers de données nécessaires (`data.csv`, `data_processed.csv`, etc.) sont présents dans le répertoire `data/` avant de lancer les scripts.
- Les résultats intermédiaires et finaux seront sauvegardés dans les répertoires `data/` et `models/`.


---

## Application Streamlit

L'application Streamlit offre une interface interactive permettant d'explorer et d'analyser les émissions de CO₂ des voitures européennes. Elle se compose de trois pages principales :

#### 📊 Exploration
Analyse exploratoire du jeu de données brut avant traitement, comprenant :
- **Taux de complétion des colonnes** : Visualisation du pourcentage de valeurs renseignées pour identifier les colonnes nécessitant un nettoyage
- **Répartition des types de carburant** : Distribution des véhicules selon leur carburant pour identifier les types dominants
- **Distribution de la puissance par carburant** : Détection des outliers (véhicules de sport) qui pourraient biaiser le modèle
- **Relation cylindrée vs émissions** : Corrélation entre cylindrée et CO₂ avec lignes de régression par type de carburant
- **Matrice de corrélation** : Relations linéaires entre caractéristiques techniques

#### 📈 Résultats
Analyse comparative des modèles de prédiction avec trois sections :
- **Classification K-means** : Regroupement des véhicules par caractéristiques similaires (analyse exploratoire)
- **Régression linéaire vs Random Forest (avec consommation)** : Comparaison des performances et identification de la forte corrélation entre consommation et émissions
- **Random Forest sans consommation** : Comparaison avec/sans feature engineering pour analyser l'influence réelle des variables techniques (cylindrée, puissance, masse, âge)

#### 🔮 Prédiction
Outil interactif permettant de saisir les caractéristiques d'un véhicule et d'obtenir une prédiction des émissions de CO₂ via le modèle Random Forest entraîné

Pour lancer l'application :

``` bash
streamlit run streamlit_app/app.py
```

---


## Résultats du Modèle

Le modèle Random Forest final **sans consommation de carburant** mais **avec feature engineering** a obtenu d'excellentes performances :

### Métriques de Performance

| Ensemble | MSE (g/km) | R² | RMSE (g/km) |
|----------|------------|-----|-------------|
| **Entraînement** | 24,05 | 0,9866 | 4,90 |
| **Validation croisée** | 34,98 | 0,9805 | 5,91 |
| **Test** | 32,70 | 0,9816 | 5,72 |

### Interprétation

Ces résultats démontrent la capacité du modèle à prédire les émissions de CO₂ avec une **précision de 98,16%** sur des données non vues et avec une **erreur moyenne de prédiction d'environ 5,7 g/km**. L'alignement entre les métriques d'entraînement et de validation indique que le modèle **généralise bien sans surapprentissage** significatif.

L'exclusion de la consommation de carburant permet d'analyser l'influence réelle des caractéristiques techniques (masse, cylindrée, puissance, type de carburant, âge) sur les émissions, rendant le modèle plus utile pour des analyses prédictives sur de nouveaux véhicules dont la consommation n'est pas encore connue.

### Variables les plus influentes

D'après l'analyse SHAP disponible dans l'application Streamlit :
1. Type de carburant (hybride ou non)
2. Masse du véhicule
3. Cylindrée du moteur
4. Puissance

### Chargement du Modèle

Le modèle entraîné peut être chargé pour faire des prédictions :

``` python
import joblib

model = joblib.load('models/random_forest_model.jbl.lzma')
```

---

## Contribuer

Les contributions sont les bienvenues ! Pour contribuer :
1. Fork ce dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-nouvelle-fonctionnalite`).
3. Commitez vos modifications (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`).
4. Poussez la branche (`git push origin feature/ma-nouvelle-fonctionnalite`).
5. Ouvrez une Pull Request.

---

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
