### But du projet

Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).

### Données utilisées

Le jeu de données suivant:

https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b


### Création du fichier data_processed.csv

Pour créer le fichier data_processed.csv, il faut:

1. `cd scripts`

2. `python -m data_reduction`

### TO DO

--- **reduction des données** ---

(fait via puissance) supression outliers via puissance (ep) ou rapport puissance / poids (ep / Mt)

(fait) supprimer plus de colonnes (VFN, z, Erwltp)

(clairement possible de suprimmer ech) après suppression outliers, regarder si IT, ech vides ou non



--- **exploration des données** ---

Si IT pas trop vide après suppression, voir le nombre de valeurs unique en vue de discrétiser et voir si certaines marques consomment moins grâce à ça

Créer qq graph avant/après preprocessing



--- **preprocessing** ---

(commencé) Uniformiser le nom des marques 

Vérifier les types des colonnes

(fait) faire un script .py pour preprocessing et reduction des données

sur les valeurs manquantes, voir si on les supprime ou on les remplace. Regarder c'est vide seulement pour un certain type de véhicules ou non (une marque, une particularité, etc )

faire une foncion générique pour gérer les doublons, outliers, valeurs manquantes


--- **feature engineering**---

voir la relation entre les variables, si corrélation importante entre 2 variables 

créer de nouveaux indicateurs (ex: power to weight)



--- **faire un dataset commun** ---

(fait) prendre FR, DE, sur plusieurs années

Nettoyer le repo
