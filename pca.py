# -*- coding: utf-8 -*-
"""
Projet M2 - Reconnaissance faciale avec PCA (Eigenfaces)

Ce script implémente un système complet de reconnaissance faciale basé sur
l'algorithme PCA (Principal Component Analysis), également connu dans ce
contexte sous le nom de méthode des Eigenfaces.

Fonctionnalités principales :
- Chargement et prétraitement d'une base de données d'images
- Calcul des Eigenfaces par PCA
- Projection des visages dans l'espace PCA
- Reconstruction de visages
- Reconnaissance faciale par :
    - comparaison directe (Nearest Neighbor)
    - comparaison avec les centres de classe
- Évaluation des performances selon différents paramètres

Auteur : beddo
Date : 2026
"""

# ================= IMPORTS =================

import matplotlib.pyplot as plt  # pour affichage des images et graphiques
import os                        # gestion des dossiers/fichiers
import numpy as np               # calcul numérique (vecteurs, matrices)
from PIL import Image            # lecture des images .pgm


# ================= CHARGEMENT DU DATASET =================

def load_dataset(root, target_size=(168, 192)):
    """
    Charge une base d'images faciales organisée par dossiers.

    Chaque dossier correspond à une personne et contient plusieurs images
    au format .pgm.

    Paramètres
    ----------
    root : str
        Chemin vers le dossier racine du dataset.
    target_size : tuple
        Taille à laquelle les images sont redimensionnées.

    Retour
    ------
    images : ndarray
        Matrice (nb_images, nb_pixels) contenant les images aplaties.
    labels : ndarray
        Labels correspondant à chaque image.
    """
    
    images = []  # contiendra les images aplaties
    labels = []  # contiendra les labels associés

        # Parcours de chaque dossier correspondant à une personne
    for person in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person)

        # Ignore si ce n'est pas un dossier
        if not os.path.isdir(person_dir):
            continue

        # Extraction du label à partir du nom du dossier
        # Exemple : s1 -> label 0, yaleB01 -> label 0
        digits = ''.join(filter(str.isdigit, person))
        if not digits: continue
        label = int(digits) - 1  # labels commencent à 0

        # Parcours des images de la personne
        for img_name in sorted(os.listdir(person_dir)):
            # On ne garde que les fichiers .pgm
            if not img_name.lower().endswith(".pgm"):
                continue

            img_path = os.path.join(person_dir, img_name)
            try:
                # Lecture en niveau de gris
                img = Image.open(img_path).convert("L")

                # Redimensionnement si demandé
                if target_size:
                    img = img.resize(target_size)

                # Conversion de l'image en tableau numpy
                img_array = np.array(img).astype(np.float32)

                # Normalisation standard (centrage + réduction)
                # Cela améliore la stabilité de la PCA
                img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
                
                # Transformation de l'image 2D en vecteur 1D
                images.append(np.array(img_array).flatten())
                labels.append(label)
            except Exception as e:
                print(f"Erreur chargement {img_path}: {e}")

    return np.array(images), np.array(labels)


# ================= CALCUL DES EIGENFACES =================

def compute_eigenfaces(X, K):
    """
    Calcule les Eigenfaces à partir des images d'entraînement.

    La PCA est réalisée via une décomposition SVD.

    Paramètres
    ----------
    X : ndarray
        Matrice (nb_images, nb_pixels) contenant les images vectorisées.
    K : int
        Nombre de composantes principales à conserver.

    Retour
    ------
    mean_face : ndarray
        Visage moyen du dataset.
    eigenfaces : ndarray
        Matrice contenant les K eigenfaces.
    """
    
   
    # Calcul du visage moyen de la base de données
    mean_face = np.mean(X, axis=0)

    # Centrage des données (soustraction du visage moyen)
    X_centered = X - mean_face

    # Décomposition en valeurs singulières (SVD)
    # équivalente à la PCA mais numériquement plus stable
    U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)

    # Les vecteurs propres correspondent aux eigenfaces
    # On garde les K premières eigenfaces
    eigenfaces = U[:, :K].T
    return mean_face, eigenfaces


# ================= VISUALISATION =================

def plot_eigenfaces_and_mean(mean_face, eigenfaces, h, w, n):
    
    
    
    plt.figure(figsize=(15, 5))
    
    # Affichage du visage moyen
    plt.subplot(1, n + 1, 1)
    plt.imshow(mean_face.reshape(h, w), cmap="gray")
    plt.title("Mean Face")
    plt.axis("off")

    # Affichage des n premières eigenfaces
    for i in range(n):
        plt.subplot(1, n + 1, i + 2)
        plt.imshow(eigenfaces[i].reshape(h, w), cmap="gray")
        plt.title(f"EF {i+1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


# ================= PROJECTION PCA =================

def project(face, mean_face, eigenfaces):
    """
    Projette un visage dans l'espace PCA.

    Cette projection correspond aux coordonnées du visage
    dans la base des eigenfaces.

    Paramètres
    ----------
    face : ndarray
        Image vectorisée.
    mean_face : ndarray
        Visage moyen.
    eigenfaces : ndarray
        Base PCA.

    Retour
    ------
    z : ndarray
        Coordonnées PCA du visage.
    """
    
    # Projection dans l'espace des eigenfaces
    z = eigenfaces @ (face - mean_face)
    
    # Normalisation du vecteur de projection
    z = z / (np.linalg.norm(z) + 1e-8)
    return z



# ================= RECONNAISSANCE AVEC SEUIL =================

def recognize_face(test_face, projections, labels, mean_face, eigenfaces, threshold=None):
    """
    Reconnaissance faciale par comparaison directe (Nearest Neighbor).

    Le visage test est projeté dans l'espace PCA puis comparé
    à toutes les projections du dataset.

    Paramètres
    ----------
    test_face : ndarray
        Visage à reconnaître.
    projections : ndarray
        Projections PCA des visages de la base.
    labels : ndarray
        Labels correspondants.
    threshold : float
        Seuil optionnel pour détecter un visage inconnu.

    Retour
    ------
    label : int ou None
        Identité prédite.
    distance : float
        Distance minimale trouvée.
    """
    test_proj = project(test_face, mean_face, eigenfaces)
    
    # Calcul de la distance euclidienne entre le visage test
    # et tous les visages de la base
    distances = np.linalg.norm(projections - test_proj, axis=1)
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    # Si distance trop grande → inconnu
    if threshold is not None and min_dist > threshold:
        return None, min_dist

    return labels[min_idx], min_dist



# ================= RECONNAISSANCE PAR CENTRE DE CLASSE =================

def recognize_class(test_face, class_centers, mean_face, eigenfaces, threshold=None):
    """
    Reconnaissance faciale par comparaison avec les centres de classe.

    Chaque personne est représentée par la moyenne de ses projections
    dans l'espace PCA.

    Cette méthode est plus rapide que la comparaison avec toutes les images.

    Paramètres
    ----------
    test_face : ndarray
    class_centers : ndarray
    mean_face : ndarray
    eigenfaces : ndarray
    threshold : float

    Retour
    ------
    label : int ou None
    distance : float
    """
    
    # Projection du visage test
    test_proj = project(test_face, mean_face, eigenfaces)

    # Distance aux centres de classe
    distances = np.linalg.norm(class_centers - test_proj, axis=1)
    
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]

    # Si distance trop grande → inconnu
    if threshold is not None and min_dist > threshold:
        return None, min_dist
    # Classe la plus proche
    return min_idx, min_dist


# ================= RECONSTRUCTION =================

def reconstruct_face(weights, mean_face, eigenfaces):
    """
    Reconstruit une image à partir de sa projection PCA.

    Paramètres
    ----------
    weights : ndarray
        Coordonnées PCA du visage.
    mean_face : ndarray
        Visage moyen.
    eigenfaces : ndarray
        Base PCA.

    Retour
    ------
    reconstructed_face : ndarray
        Image reconstruite.
    """
    # Reconstruction à partir des poids PCA
    return mean_face + eigenfaces.T @ weights



# ================= TRAIN / TEST SPLIT =================

def train_test_split(X, y, n_train):
    """
    Sépare le dataset en données d'entraînement et de test.

    Pour chaque personne, on sélectionne n_train images pour
    l'entraînement et le reste pour le test.

    Paramètres
    ----------
    X : ndarray
        Images.
    y : ndarray
        Labels.
    n_train : int
        Nombre d'images d'entraînement par personne.

    Retour
    ------
    X_train, y_train, X_test, y_test
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Pour chaque personne
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)

        # Séparation train/test
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_train.extend(X[train_idx])
        y_train.extend(y[train_idx])
        X_test.extend(X[test_idx])
        y_test.extend(y[test_idx])

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test)
    )

#%%

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
# Cette section exécute ces étapes du système :
# 1) Chargement des données
# 2) Calcul des eigenfaces et du visages moyen
# 3) Affiche un certain nombre d'eigenfaces
# 4) Projection des visages
# 5) Calcul des centres de classe : moyenne des coefficients de la projection 
# pour une personne
# ============================================================

# 1) Charger les données

X, y = load_dataset("CroppedYale")


# 2) Calcul des eigenfaces et du visages moyen

# Nombre de composantes principales
K = len(X) # typiquement 40–100
mean_face, eigenfaces = compute_eigenfaces(X, K)


# 3) Visualisation

# Nombre de eigenfaces que l'on veut affichées
n=15
h, w = 192, 168
plot_eigenfaces_and_mean(mean_face, eigenfaces, h, w, n)


# 4) Projection de toutes les images

projections = np.array([
    project(x, mean_face, eigenfaces)
    for x in X
])


# 5) Calcul des centres de classe

# Liste qui contiendra les coefficients moyens
class_centers = []
for label in np.unique(y):
    class_centers.append(np.mean(projections[y == label], axis=0))
class_centers = np.array(class_centers)

#%%

# ============================================================
# Comparaison visage original et visage reconstruit avec nombre de eigenfaces variable
# ============================================================
 # Cette section exécute ces étapes du système :
 # 1) Sélection d'un visage test
 # 2) Normalisation et projection en vecteur du visage test
 # 3) Projection du visage test
 # 4) Reconstruction du visage test
 # 5) Affichage du visage test original et du visage test reconstruit pour comparer
 # 6) De même que 5) avec un nombre de eigenfaces variable
 # ============================================================

# 1) Sélection d'un visage test
A=Image.open('aimine1.png').convert("L")
A = A.resize((168, 192))
A = np.array(A).astype(np.float32)

# 2) Normalisation et projection en vecteur du visage test
A = (A - A.mean()) / (A.std() + 1e-8)
A=A.flatten()

# 3) Projection du visage test
A_proj = project(A, mean_face, eigenfaces)


# 4) Reconstruction du visage test
A_recon = reconstruct_face(A_proj, mean_face, eigenfaces)
A_recon_img = A_recon.reshape(192, 168)


# 5) Affichage des du visage test original et du visage test reconstruit
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(A.reshape(192,168), cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Reconstruit PCA (K=2452)")
plt.imshow(A_recon_img, cmap='gray')
plt.axis('off')
plt.show()

# 6) Reconstruction des visages avec un nombre d'eigenfaces variable
for k in [100,500,1000,1500]:
    eigenfaces_k = eigenfaces[:k]
    A_proj_k = project(A, mean_face, eigenfaces_k)
    A_recon = reconstruct_face(A_proj_k, mean_face, eigenfaces_k)

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(A.reshape(192,168), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title(f"Reconstruit PCA (K = {k})")
    plt.imshow(A_recon.reshape(h, w), cmap="gray")
    plt.axis("off")
    plt.show()
    
    
    
#%%

# ============================================================
# Comparaison reconnaissance avec centre de classe ou pas
# ============================================================
 # Cette section exécute ces étapes du système :
 # 1) Sélection d'un visage test
 # 2) Reconnaissance en comparant toutes les images 
 # 3) Reconnaissance en comparant toutes les centres de classe
 # ============================================================


# 1) Sélection d'un visage test
idx = 64  # index du visage que l'on selectionne dans la base de données
test_face = X[idx]



# 2) TEST DE RECONNAISSANCE 

pred_label, dist =recognize_face(test_face, projections, y, mean_face, eigenfaces, threshold=None)
print("Personne reconnue : Sujet", pred_label + 1)

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Visage test")
plt.imshow(test_face.reshape(h, w), cmap="gray")
plt.axis("off")

# image représentative de la classe reconnue
recognized_face = X[y == pred_label][0]

plt.subplot(1, 2, 2)
plt.title(f"Reconnu comparaison image : Sujet{pred_label + 1}")
plt.imshow(recognized_face.reshape(h, w), cmap="gray")
plt.axis("off")
plt.show()



# 3)cTEST DE RECONNAISSANCE AVEC CENTRE

pred_label, dist = recognize_class(test_face,class_centers,mean_face,eigenfaces)
print("Personne reconnue : Sujet", pred_label + 1)

plt.figure(figsize=(6, 3))

plt.imshow(test_face.reshape(h, w), cmap="gray")
plt.axis("off")

plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.title("Visage test")
plt.imshow(test_face.reshape(h, w), cmap="gray")
plt.axis("off")

# image représentative de la classe reconnue
recognized_face = X[y == pred_label][0]

plt.subplot(1, 2, 2)
plt.title(f"Reconnu comparaison centre : Sujet{pred_label + 1}")
plt.imshow(recognized_face.reshape(h, w), cmap="gray")
plt.axis("off")

plt.show()


#%% 

# ============================================================
# Evaluation de notre code en séparant la base de données en une partie apprentissage
# et une partie test. Puis on compte le nombre de visages reconnus correctement en 
# utilisant soit la méthode de centre de classe ou en utilisant la méthode où on compare 
# tous les visages
# ============================================================


# nombre de photos que l'on apprend par personne (maximum = 64)
n_train=32 

# Nombre de eigenfaces que l'on utilise
K_train=70

# permet la séparation des données
X_train, y_train, X_test, y_test = train_test_split(X, y,n_train) 

# calcul le visage moyen et les eigenfaces sur les données d'entrainements
mean_face_train, eigenfaces_train = compute_eigenfaces(X_train, K_train) 


# calcul la projection 
train_proj = np.array([
    project(x, mean_face_train, eigenfaces_train)
    for x in X_train
])

# calcul centre d'une personne dans l'espace de prjection
class_centers_train = []
for label in np.unique(y_train):
    class_centers_train.append(np.mean(train_proj[y_train == label], axis=0))
class_centers_train = np.array(class_centers_train)


# nombre de personnes correctement reconnues en comparant toutes les images
correct = 0
# nombre de personnes correctement reconnues en comparant aux centres de class
correct_center = 0

# parcour l'ensemble des données de test pour reconnaitre la personne puis comptabilise 
# si c'est une bonne reconnaissance
for face, true_label in zip(X_test, y_test):
    pred_label, dist = recognize_face(
        face, 
        train_proj,
        y_train,
        mean_face_train, 
        eigenfaces_train, 
        threshold=None
        )
    if pred_label == true_label:
        correct += 1
    
    pred_label_center, dist_center = recognize_class(
        face,
        class_centers_train,
        mean_face_train,
        eigenfaces_train
    )
    if pred_label_center == true_label:
        correct_center += 1
      
    
accuracy = correct / len(y_test)
print(f"Taux de reconnaissance : {accuracy * 100:.2f} %")

accuracy_center = correct_center / len(y_test)
print(f"Taux de reconnaissance pour center : {accuracy_center * 100:.2f} %")



#%% 

# ============================================================
# Similaire à la partie précèdente mais avec des listes afin de voir la dépendance 
# de nos résultats en fonction du nombre de photo prise par personne et du nombre 
# d'eigenfaces
# ============================================================



# listes de paramètres à tester
n_train_list = [5]
K_train_list = [20,40,60,80,100,120,140,160,180,190]


# matrices pour stocker les résultats
results_nn = np.zeros((len(n_train_list), len(K_train_list)))
results_center = np.zeros((len(n_train_list), len(K_train_list)))


# BOUCLE EXPÉRIMENTALE 
for i, n_train in enumerate(n_train_list):
    print(f"\n===== n_train = {n_train} =====")

    # split dépend seulement de n_train
    X_train, y_train, X_test, y_test = train_test_split(X, y, n_train)

    for j, K_train in enumerate(K_train_list):
        print(f"  -> K = {K_train}")

        # ===== PCA sur train =====
        mean_face_train, eigenfaces_train = compute_eigenfaces(X_train, K_train)

        # ===== projection train =====
        train_proj = np.array([
            project(x, mean_face_train, eigenfaces_train)
            for x in X_train
        ])

        # ===== centres de classe =====
        class_centers_train = []
        for label in np.unique(y_train):
            class_centers_train.append(
                np.mean(train_proj[y_train == label], axis=0)
            )
        class_centers_train = np.array(class_centers_train)

        # ===== évaluation =====
        correct = 0
        correct_center = 0

        for face, true_label in zip(X_test, y_test):

            # --- NN ---
            pred_label, _ = recognize_face(
                face,
                train_proj,
                y_train,
                mean_face_train,
                eigenfaces_train,
                threshold=None
            )
            if pred_label == true_label:
                correct += 1

            # --- centre ---
            pred_label_center, _ = recognize_class(
                face,
                class_centers_train,
                mean_face_train,
                eigenfaces_train
            )
            if pred_label_center == true_label:
                correct_center += 1

        # ===== stockage résultats =====
        results_nn[i, j] = correct / len(y_test)
        results_center[i, j] = correct_center / len(y_test)



# AFFICHAGE FINAL

print("\n=== MATRICE NN ===")
print(results_nn)

print("\n=== MATRICE CENTER ===")
print(results_center)

plt.figure(figsize=(8,5))

plt.plot(K_train_list, results_nn[0,:], marker='o', label="Comparaison images")
plt.plot(K_train_list, results_center[0,:], marker='s', label="Comparaison centres")

plt.xlabel("Nombre d'eigenfaces (k)")
plt.ylabel("Taux de reconnaissance")
plt.title("Performance en fonction de k")
plt.grid(True)
plt.legend()

plt.show()


# HEATMAP
plt.figure(figsize=(8,6))
plt.imshow(results_nn, aspect='auto')
plt.colorbar(label="Accuracy")
plt.xticks(range(len(K_train_list)), K_train_list)
plt.yticks(range(len(n_train_list)), n_train_list)
plt.xlabel("Nombre d'eigenfaces (K)")
plt.ylabel("Images d'entraînement par personne (n_train)")
plt.title("Performance NN")
plt.show()

# HEATMAP CENTERS
plt.figure(figsize=(8,6))
plt.imshow(results_center, aspect='auto')
plt.colorbar(label="Accuracy")
plt.xticks(range(len(K_train_list)), K_train_list)
plt.yticks(range(len(n_train_list)), n_train_list)
plt.xlabel("Nombre d'eigenfaces (K)")
plt.ylabel("Images d'entraînement par personne (n_train)")
plt.title("Performance centre de classe")
plt.show()

