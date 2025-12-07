# Analyse des Émissions de CO₂ des Voitures Européennes

Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).

Ce projet vise donc à analyser et modéliser les émissions de CO₂ des voitures européennes en utilisant des techniques de **prétraitement des données**, **d'ingénierie des caractéristiques**, et de **modélisation machine learning**.

---

## Table des Matières

- [Données Utilisées](#données-utilisées)
- [Structure du Projet](#structure-du-projet)
- [Description des Fichiers](#description-des-fichiers)
  - [data_reduction.py](#data_reductionpy)
  - [preprocessing.py](#preprocessingpy)
  - [feature_engineering.py](#feature_engineeringpy)
  - [pipeline.py](#pipelinepy)
- [Installation](#installation)
  - [Prérequis](#prérequis)
  - [Configuration de l'Environnement](#configuration-de-lenvironnement)
- [Utilisation](#utilisation)
  - [Pipeline Complète](#lancer-la-pipeline-complète)
  - [Étapes Individuelles](#lancer-les-étapes-individuelles)
- [Application Streamlit](#application-streamlit)
  - [Exploration](#-exploration)
  - [Résultats](#-résultats)
  - [Prédiction](#-prédiction)
- [Résultats et Méthodologie](#résultats-et-méthodologie-du-modèle)
  - [Stratégie de Modélisation](#stratégie-de-modélisation)
  - [Hyperparamètres](#hyperparamètres-du-modèle)
  - [Feature Engineering](#feature-engineering)
  - [Validation Croisée](#stratégie-de-validation-croisée)
  - [Métriques de Performance](#métriques-de-performance)
  - [Variables Influentes](#variables-les-plus-influentes)
  - [Utilisation du Modèle](#chargement-du-modèle)

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
├── notebooks/
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


## Résultats et Méthodologie du Modèle

### Stratégie de Modélisation

Le modèle Random Forest a été sélectionné après une analyse comparative de plusieurs approches de régression (régression linéaire, Random Forest avec/sans consommation de carburant). Le choix final s'est porté sur Random Forest **sans consommation de carburant** mais **avec feature engineering** pour les raisons suivantes :

- **Objectif prédictif** : Permet d'estimer les émissions avant la connaissance de la consommation réelle
- **Interprétabilité** : Identifie l'influence réelle des caractéristiques techniques du véhicule
- **Robustesse** : Gère efficacement les relations non-linéaires entre variables
- **Performance** : Atteint une précision élevée tout en évitant le surapprentissage

### Hyperparamètres du Modèle

Le modèle Random Forest final utilise les hyperparamètres suivants :

``` python
RandomForestRegressor(
  n_estimators=100,     # Nombre d'arbres dans la forêt
  max_depth=20,         # Profondeur maximale des arbres
  min_samples_split=5,  # Échantillons minimum pour diviser un nœud
  min_samples_leaf=1,   # Échantillons minimum dans une feuille
  bootstrap=True,       # Bootstrap des échantillons
  random_state=42       # Reproductibilité
)
```

Ces hyperparamètres ont été sélectionnés à l'aide d'une GridSearch (cf `notebooks/model_creation.ipynb`).

### Feature Engineering

Les transformations appliquées aux variables sont définies dans un `ColumnTransformer` intégré au pipeline :

| Variable | Transformation | Justification |
|----------|---------------|---------------|
| **ec (cm3)** | StandardScaler | Normalisation de la cylindrée (distribution gaussienne) |
| **ep (KW)** | StandardScaler | Normalisation de la puissance (distribution gaussienne) |
| **m (kg)** | StandardScaler | Normalisation de la masse (large plage de valeurs) |
| **age_months** | MinMaxScaler | Mise à l'échelle [0,1] de l'âge (croissance monotone) |
| **Ft (type carburant)** | OneHotEncoder | Encodage catégoriel avec gestion des valeurs inconnues |

Ces transformations permettent :
- D'uniformiser les échelles pour éviter la dominance de certaines variables
- De gérer correctement les variables catégorielles
- De maintenir l'interprétabilité du modèle via SHAP

### Stratégie de Validation Croisée

La méthodologie d'évaluation suit un protocole en trois étapes :

#### 1. Division Train/Test
- **Ratio** : 80% entraînement / 20% test
- **Stratification** : Aléatoire avec `random_state=42` pour reproductibilité
- **Objectif** : Conserver un ensemble de test intact pour évaluation finale

#### 2. Validation Croisée (5-Fold)

``` python
cross_validate(
pipeline, X_train, y_train,
cv=5, # 5 plis
scoring={
"neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
"r2": "r2"
},
return_train_score=True, # Calcul des scores d'entraînement
n_jobs=-1 # Parallélisation
)
```

- **5 folds** : Chaque observation sert à l'entraînement et à la validation
- **Métriques doubles** : MSE (erreur absolue) et R² (coefficient de détermination)
- **Détection du surapprentissage** : Comparaison train/validation scores


#### 3. Évaluation Finale sur Test Set
- **Données jamais vues** : Le modèle est évalué sur les 20% de données test
- **Métriques finales** : MSE et R² calculés sur des prédictions hors échantillon

### Métriques de Performance

| Ensemble | MSE (g/km) | R² | RMSE (g/km) |
|----------|------------|-----|-------------|
| **Entraînement (CV)** | 24,05 | 0,9866 | 4,90 |
| **Validation croisée** | 34,98 | 0,9805 | 5,91 |
| **Test** | 32,70 | 0,9816 | 5,72 |

**Interprétation des résultats** :
- **R² = 0,9816** : Le modèle explique 98,16% de la variance des émissions de CO₂
- **RMSE = 5,72 g/km** : Erreur moyenne de prédiction faible
- **Écart train/test minimal** : Pas de surapprentissage significatif (différence de 1,2% sur R²)
- **Généralisation robuste** : Performances cohérentes entre validation croisée et test

### Variables les Plus Influentes

L'analyse SHAP (SHapley Additive exPlanations) révèle l'importance relative des features :

1. **Type de carburant (Ft)** : Impact majeur, particulièrement pour les hybrides
2. **Masse du véhicule (m)** : Corrélation positive forte avec les émissions
3. **Cylindrée (ec)** : Indicateur clé de la consommation potentielle
4. **Puissance (ep)** : Influence modérée
5. **Âge (age_months)** : Impact mineur

Les graphiques SHAP détaillés sont disponibles dans l'application Streamlit (page **Résultats**).

### Pipeline Complet

Le pipeline scikit-learn intègre toutes les transformations et le modèle :

``` python
Pipeline([
  ('features', ColumnTransformer([...])), # Feature engineering
  ('model', RandomForestRegressor(...))   # Modèle
])
```

**Avantages** :
- Pas de fuite de données (data leakage) entre train et test
- Transformations appliquées automatiquement lors de `.predict()`
- Reproductibilité garantie
- Facilité de déploiement

### Chargement du Modèle

Le modèle entraîné peut être chargé pour faire des prédictions :

``` python
import joblib
import pandas as pd

# Chargement du pipeline complet
model = joblib.load('models/random_forest_model.jbl.lzma')

# Prédiction (les transformations sont appliquées automatiquement)
new_vehicle = pd.DataFrame({
    'm (kg)': [1500],
    'Ft': ['Petrol'],
    'ec (cm3)': [1600],
    'ep (KW)': [110],
    'age_months': [12]
})

predicted_co2 = model.predict(new_vehicle)
print(f"Émissions prédites : {predicted_co2[0]:.2f} g/km")
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
