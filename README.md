# Analyse des Émissions de CO₂ des Voitures Européennes

Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).

Ce projet vise donc à analyser et modéliser les émissions de CO₂ des voitures européennes en utilisant des techniques de **prétraitement des données**, **d'ingénierie des caractéristiques**, et de **modélisation machine learning**.

---
## TO DO

- Mettre data_reduction preprocessing dans la pipeline ? Se pose la question de la taille des jeux de données
- Nettoyage du repo (est ce qu'on garde toutes les explorations?)
- Streamlit

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
- [Résultats](#résultats)

---

## Données utilisées

Le jeu de données utilisé est le suivant:

https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b


## Structure du Projet

```
european_passenger_cars_co2/
│
├── data/
│   └── data_processed.csv              # Données prétraitées
│
├── exploration/
│   └── *.ipynb                         # Explorations des données
│
├── fig/
│   └── *.png                           # Figures sauvegardées    
│
├── logs/
│   └── training.log                    # Logs des entrainements de modèles
│
├── models/
│   └── random_forest_model.jbl.lzma    # Modèle entraîné sauvegardé
│
├── scripts/
│   ├── data_reduction.py               # Réduction et nettoyage des données
│   ├── feature_engineering.py          # Ingénierie des caractéristiques
│   ├── pipeline.py                     # Pipeline de modélisation
│   └── preprocessing.py                # Prétraitement des données
│
├── streamlit_app/
│   ├── pages/                          # Pages du streamlit
│   │   ├── compare.py                  
│   │   └── predict.py
│   ├── utils/                          # Fichiers de config du streamlit
│   │   ├── data_loaders.py
│   │   ├── model_utils.py
│   │   └── viz_tools.py
│   └── app.py                          # Application streamlit
│
├── .gitattributes                      # Configuration pour LFS
├── .gitignore                          # Fichiers à ignorer
├── LICENSE                             # Licence du projet
├── README.md                           # Documentation du projet
└── requirements.txt                    # Prérequis à installer

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
- `matplotlib`
- `streamlit`
- `seaborn`

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

## Résultats

A COMPLETER

Les résultats du modèle (métriques d'évaluation, modèle entraîné) sont sauvegardés dans le dossier `models/`. Vous pouvez charger le modèle sauvegardé avec `joblib` pour faire des prédictions :

```python
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
