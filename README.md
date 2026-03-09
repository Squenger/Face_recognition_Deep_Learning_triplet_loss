# Reconnaissance Faciale par PCA (Eigenfaces)

## Description

Ce projet implémente un système de **reconnaissance faciale basé sur l'analyse en composantes principales (PCA)**, également connue dans ce contexte sous le nom de **méthode des Eigenfaces**.

L'objectif est de réduire la dimension des images faciales tout en conservant les caractéristiques principales permettant d'identifier une personne.

Le projet comprend :

- le calcul des **eigenfaces**
- la **projection des visages dans l'espace PCA**
- la **reconstruction d'images**
- la **reconnaissance faciale**
- l'**évaluation des performances** selon différents paramètres.

---

# Méthode

## PCA (Principal Component Analysis)

La PCA permet de projeter les images dans un espace de dimension réduite tout en conservant la variance maximale des données.

Dans le contexte de la reconnaissance faciale :

1. On calcule le **visage moyen** du dataset.
2. On centre les images.
3. On calcule les **vecteurs propres** de la matrice de covariance.
4. Ces vecteurs propres correspondent aux **Eigenfaces**.

Chaque visage peut alors être représenté comme une **combinaison linéaire d'eigenfaces**.

---

# Pipeline

Le pipeline de reconnaissance comprend les étapes suivantes :

1. **Chargement du dataset**
2. **Prétraitement**
   - conversion en niveaux de gris
   - redimensionnement
   - normalisation
3. **Calcul des eigenfaces**
4. **Projection des images dans l'espace PCA**
5. **Reconnaissance faciale**
6. **Évaluation des performances**

---

# Méthodes de reconnaissance

Deux approches sont implémentées :

## 1. Nearest Neighbor (NN)

Le visage test est projeté dans l'espace PCA puis comparé à **tous les visages de la base**.

La personne reconnue est celle qui minimise la **distance euclidienne**.

## 2. Centre de classe

Chaque personne est représentée par le **centre de ses projections PCA**.

Le visage test est comparé aux centres.

Avantages :
- plus rapide
- moins sensible au bruit

---

# Reconstruction des visages

Il est possible de reconstruire une image à partir de ses coordonnées PCA.

La reconstruction est donnée par :

econstruction = mean_face + eigenfaces^T * weights

La qualité dépend du **nombre de composantes principales K** utilisées.

---

# Dataset utilisé

Le projet utilise le dataset :

**Cropped Yale Face Database**

Caractéristiques :

- 38 personnes
- ~64 images par personne
- variations :
  - illumination
  - expression
  - pose légère

Les images sont stockées en format **PGM**.

---

# Paramètres expérimentaux

Les performances sont évaluées selon :

- nombre d'images d'entraînement par personne `n_train`
- nombre d'eigenfaces `K`

Exemple de valeurs testées :

n_train = [5]
K = [20,40,60,80,100,120,140,160,180,190]



Les résultats sont affichés sous forme de :

- courbes de performance
- heatmaps

---

# Résultats

Les performances sont mesurées avec le **taux de reconnaissance** :

accuracy = nombre de bonnes prédictions / nombre total de tests


Deux méthodes sont comparées :

- comparaison directe des images
- comparaison avec les centres de classes

---

# Librairies utilisées

- numpy
- matplotlib
- Pillow
- os

---

# Lancer le projet

Placer le dataset `CroppedYale` dans le même dossier que le script.

Puis exécuter :

```bash
python PCA.py