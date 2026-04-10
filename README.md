# Projet Siamese — Reconnaissance Faciale par Triplet Loss

Réseau siamois basé sur ResNet18 entraîné avec la Triplet Loss sur le dataset **LFW (Labeled Faces in the Wild)**. Le projet permet d'entraîner, évaluer, comparer et utiliser un modèle d'embedding facial.

---

## Prérequis

### Dataset LFW (obligatoire mais il figure déja sur le repository github)

Télécharger depuis [Kaggle – LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) et placer les fichiers dans `data/archive/` :

```
data/
└── archive/
    ├── lfw-deepfunneled/
    │   └── lfw-deepfunneled/
    │       ├── Aaron_Eckhart/
    │       │   └── Aaron_Eckhart_0001.jpg
    │       └── ...         (un dossier par personne)
    └── lfw_allnames.csv    (liste des noms + nombre d'images)
```


---

## Structure du projet

```
Projet Siamese/
├── siamese_trainer.py              # Module principal (classes + entraînement)
├── Triplet_loss.ipynb              # Notebook d'exploration original
├── data/archive/                   # Dataset LFW
├── model_resnet_triplet.pth       #Poids obenu après un première entrainement (10 epoches)
├── model_resnet_triplet13epocheshardmining.pth
├── model_resnet_triplet20.pth
├── model_resnet_triplet40.pth
├── model_resnet_triplet50.pth      # Modèles entraînés le plus avancé
└── comparison_curves.png           # Courbes de comparaison générées
```

---

## Fonctionnalités

### 1. Entraînement

```python
from siamese_trainer import SiameseTrainer

trainer = SiameseTrainer(
    lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
    lfw_allnames_path='data/archive/lfw_allnames.csv',
    batch_size=32,
    margin=1.0,
    lr=0.0001,
)

trainer.train(n_epochs=20)
trainer.save_weights('mon_model.pth')
```
#### Explication de l'architechture
- Architecture : **ResNet18 pré-entraîné**, tête FC remplacée (512→256→128)
- Stratégie : **Online Hard Negative Mining** pour des triplets difficiles
- Triplets : (ancre, positif = même personne, négatif = personne différente)

### 2. Chargement d'un modèle existant

```python
trainer.load_weights('model_resnet_triplet50.pth')
```

### 3. Évaluation

```python
results = trainer.evaluate(n_batches=100, save_prefix='eval')
```

Génère automatiquement :
- `eval_histogram.png` — distributions des distances intra/inter-classe avec seuil optimal
- `eval_curves.png` — courbe Accuracy vs FAR + courbe F1-Score vs Seuil

### 4. Comparaison de plusieurs modèles

```python
SiameseTrainer.compare_models(
    models_list=[
        ('ResNet 13 epochs', 'model_resnet_triplet13epocheshardmining.pth'),
        ('ResNet 20 epochs', 'model_resnet_triplet20.pth'),
        ('ResNet 50 epochs', 'model_resnet_triplet50.pth'),
    ],
    lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
    lfw_allnames_path='data/archive/lfw_allnames.csv',
    save_prefix='comparison',
)
```

Génère `comparison_results.png` (histogrammes) et `comparison_curves.png` (courbes comparatives).

### 5. Inférence sur une paire d'images

```python
result = trainer.evaluer_paire(img1, img2, threshold=1.0)
# result = {'distance': 0.73, 'same_person': True, 'verdict': 'même personne'}
```

Affiche les deux images avec la distance euclidienne et le verdict (vert = même personne, rouge = différentes).

---

## Métriques utilisées

| Métrique | Description |
|---|---|
| **Accuracy** | Taux de classification correcte au seuil optimal |
| **FAR** | False Alert Rate — taux de fausse acceptation |
| **F1-Score** | Harmonie précision/rappel selon le seuil |
| **Seuil optimal** | Seuil de distance minimisant les erreurs |

---


Exécute automatiquement : entraînement (3 époques) → sauvegarde → évaluation → comparaison des modèles pré-entraînés.
