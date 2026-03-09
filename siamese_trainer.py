"""
siamese_trainer.py
==================
Reprise orientée-classe du notebook Triplet_loss.ipynb.

Contient :
  - FaceNetResNet  : réseau d'embedding basé sur ResNet18 (transfert learning)
  - Siamois        : réseau siamois qui passe 3 images dans le même embedding
  - TripletLoss    : fonction de perte triplet avec marge
  - TripletLFW     : dataset générant des triplets (ancre / positif / négatif)
                     à partir du dataset LFW (Labeled Faces in the Wild)
  - SiameseTrainer : classe principale qui orchestre l'entraînement, la
                     sauvegarde / le chargement des poids et l'évaluation
                     des performances (histogrammes + courbes FAR/Accuracy/F1)

Utilisation rapide
------------------
    from siamese_trainer import SiameseTrainer

    trainer = SiameseTrainer(
        lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path='data/archive/lfw_allnames.csv',
        batch_size=32,
        margin=1.0,
        lr=0.0001,
    )

    # Entraîner le modèle
    trainer.train(n_epochs=20)

    # Sauvegarder les poids
    trainer.save_weights('mon_model.pth')

    # Charger des poids existants
    trainer.load_weights('model_resnet_triplet50.pth')

    # Évaluer et générer les courbes
    trainer.evaluate(n_batches=100, save_prefix='results')

    # Comparer plusieurs modèles sauvegardés
    SiameseTrainer.compare_models(
        models=[
            ('ResNet 13 epochs', 'model_resnet_triplet13epocheshardmining.pth'),
            ('ResNet 50 epochs', 'model_resnet_triplet50.pth'),
        ],
        lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path='data/archive/lfw_allnames.csv',
        save_prefix='comparison',
    )

    # Utiliser le modèle pour comparer deux images
    trainer.evaluer_paire(img1, img2, threshold=1.0)
"""

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt



# On prioritise l'utilisation du GPU Apple Silicon (mps) si disponible,
# sinon le GPU CUDA, sinon le CPU.
def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# 1. Réseau d'embedding : FaceNetResNet
class FaceNetResNet(nn.Module):
    """
    Réseau d'embedding pour la reconnaissance faciale basé sur ResNet18
    pré-entraîné (transfert learning).

    Architecture :
      - ResNet18 pré-entraîné dont tous les poids sont gelés.
      - La couche fully-connected finale est remplacée par :
          Linear(512 → 256) → PReLU → Linear(256 → 128)
        afin de produire un vecteur d'embedding de dimension 128.
      - Seule cette nouvelle tête est entraînable.
    """

    def __init__(self):
        super(FaceNetResNet, self).__init__()

        # Utilisation d'un ResNet18 pré-entraîné
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # On gèle les poids du ResNet pré-entraîné
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Remplacement de la couche finale pour obtenir un embedding de taille 128
        num_features = self.resnet.fc.in_features  # Nombre de caractéristiques en entrée de la couche finale

        # Nouvelle couche finale dont la sortie est de taille 128
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.PReLU(),
            nn.Linear(256, 128)
        )

        # Seule la nouvelle tête est entraînable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return x


# 2. Réseau siamois : Siamois
class Siamois(nn.Module):
    """
    Réseau siamois qui partage le même embedding pour les trois branches
    (ancre, positif, négatif).

    Le même réseau d'embedding est appliqué successivement aux trois images
    du triplet.
    """

    def __init__(self, embedding: nn.Module = None):
        super(Siamois, self).__init__()

        # Utilisation de l'embedding fourni en paramètre
        # (utile pour le transfert d'apprentissage)
        self.embedding = embedding

    def forward(self, x1, x2, x3):
        # On passe les trois images (ancre, positif, négatif) dans le même
        # réseau d'embedding
        output1 = self.embedding(x1)  # obtenir l'embedding de l'ancre
        output2 = self.embedding(x2)  # obtenir l'embedding du positif
        output3 = self.embedding(x3)  # obtenir l'embedding du négatif
        return output1, output2, output3


# 3. Fonction de perte : TripletLoss
class TripletLoss(nn.Module):
    """
    Triplet Loss avec marge.

    Formule mathématique :
        L = max(d(A,P) - d(A,N) + marge, 0)

    On veut que la distance d(ancre, positif) soit la plus petite possible
    tout en assurant que d(ancre, négatif) soit grande. En ajoutant une marge,
    cette condition se traduit par :
        d(A,P) - d(A,N) + marge < 0

    Si cette condition est vérifiée, la perte vaut 0 (comme pour les SVM,
    on est en dehors de la marge). Sinon, la perte est positive.
    """

    def __init__(self, marge: float = 1.0):
        super(TripletLoss, self).__init__()
        self.marge = marge

    def forward(self, anchor, positive, negative):
        # Calcul des distances euclidiennes (au carré pour la stabilité numérique)
        distance_positive = (anchor - positive).pow(2).sum(1)  # Distance entre ancre et positif
        distance_negative = (anchor - negative).pow(2).sum(1)  # Distance entre ancre et négatif

        # Calcul de la perte triplet
        # relu vaut 0 si l'argument est négatif et l'argument sinon
        losses = torch.relu(distance_positive - distance_negative + self.marge)
        return losses.mean()


# 4. Dataset : TripletLFW
class TripletLFW(Dataset):
    """
    Dataset générant des triplets (ancre, positif, négatif) à partir du
    dataset LFW (Labeled Faces in the Wild).

    Le fichier lfw_allnames.csv est utilisé pour identifier correctement
    les personnes ayant au moins 2 images afin de pouvoir construire des
    paires positives (même personne, images différentes).
    """

    def __init__(self, dataset: ImageFolder,
                 lfw_allnames_path: str = 'data/archive/lfw_allnames.csv'):
        """
        Initialise le dataset TripletLFW en utilisant le fichier
        lfw_allnames.csv pour identifier correctement les personnes
        avec au moins 2 images.

        Parameters
        ----------
        dataset : ImageFolder
            Dataset brut chargé avec torchvision.datasets.ImageFolder.
        lfw_allnames_path : str
            Chemin vers le fichier CSV contenant les noms et le nombre
            d'images de chaque personne du dataset LFW.
        """
        self.dataset = dataset

        # Lire le fichier lfw_allnames.csv
        lfw_info = pd.read_csv(lfw_allnames_path)

        # Filtrer les personnes avec au moins 2 images
        valid_people = lfw_info[lfw_info['images'] >= 2]['name'].tolist()
        print(f"Nombre de personnes avec >= 2 images: {len(valid_people)}")

        # Récupérer les chemins des images depuis ImageFolder
        self.img_paths = [item[0] for item in self.dataset.imgs]

        # Créer un mapping : nom_personne -> [indices des images]
        self.person_to_indices = {}
        for idx, img_path in enumerate(self.img_paths):
            # Extraire le nom de la personne du chemin : .../NomPersonne/image.jpg
            person_name = img_path.split('/')[-2]  # Obtenir le dossier parent

            if person_name in valid_people:
                if person_name not in self.person_to_indices:
                    self.person_to_indices[person_name] = []
                self.person_to_indices[person_name].append(idx)

        # Filtrer pour garder uniquement les personnes avec au moins 2 images
        self.valid_people = [p for p in self.person_to_indices.keys()
                             if len(self.person_to_indices[p]) >= 2]

        self.valid_indices = []
        for person in self.valid_people:
            self.valid_indices.extend(self.person_to_indices[person])

        print(f"Nombre de personnes valides dans le dataset: {len(self.valid_people)}")
        print(f"Nombre d'images utilisables: {len(self.valid_indices)}")

    def __getitem__(self, index):
        # ---- Ancre (référence) ----
        real_index = self.valid_indices[index]
        img1, _ = self.dataset[real_index]

        # Obtenir le nom de la personne depuis le chemin
        person_name = self.img_paths[real_index].split('/')[-2]

        # ---- Positif : une autre image de la MÊME personne ----
        positive_indices = self.person_to_indices[person_name].copy()
        positive_indices.remove(real_index)  # Retirer l'image ancre

        if len(positive_indices) > 0:
            positive_index = int(np.random.choice(positive_indices))
        else:
            # Fallback (ne devrait pas arriver ici)
            positive_index = int(np.random.choice(self.person_to_indices[person_name]))
        img2, _ = self.dataset[positive_index]

        # ---- Négatif : une image d'une PERSONNE DIFFÉRENTE ----
        negative_person = np.random.choice(self.valid_people)
        tentative = 0
        while negative_person == person_name:
            negative_person = np.random.choice(self.valid_people)
            tentative += 1
            if tentative > 100:
                break

        negative_index = int(np.random.choice(self.person_to_indices[negative_person]))
        img3, _ = self.dataset[negative_index]

        return (img1, img2, img3), []  # le label n'est pas utilisé dans le triplet loss

    def __len__(self):
        return len(self.valid_indices)  # nombre d'images valides dans le dataset


# 5. Online Hard Negative Mining
def hard_negative_mining(anchor_embeddings, positive_embeddings, negative_embeddings_list):
    """
    Implémente le Online Hard Negative Mining (OHNM).

    Au lieu de sélectionner un négatif aléatoire, OHNM choisit le négatif
    **le plus difficile** (celui avec la plus petite distance à l'ancre parmi
    les candidats). Cela améliore l'entraînement en forçant le modèle à
    apprendre à distinguer les cas difficiles.

    Formule :
        negative = argmin_{n ∈ N} d(anchor, n)

    Parameters
    ----------
    anchor_embeddings : Tensor, shape (batch_size, embedding_dim)
    positive_embeddings : Tensor, shape (batch_size, embedding_dim)
    negative_embeddings_list : list[Tensor], chacun de shape (batch_size, embedding_dim)
        Liste de candidats négatifs.

    Returns
    -------
    hard_negatives : Tensor, shape (batch_size, embedding_dim)
        Le négatif le plus difficile pour chaque ancre.
    distances_negatives : Tensor, shape (batch_size, num_candidates)
        Distances entre l'ancre et chaque candidat négatif.
    """
    batch_size = anchor_embeddings.shape[0]

    # Calculer les distances avec tous les candidats négatifs
    distances_negatives = []
    for neg_emb in negative_embeddings_list:
        dist = (anchor_embeddings - neg_emb).pow(2).sum(1)  # (batch_size,)
        distances_negatives.append(dist)

    # distances_negatives est une liste de tenseurs (batch_size,)
    # On empile et on trouve le minimum pour chaque exemple du batch
    distances_negatives = torch.stack(distances_negatives, dim=1)  # (batch_size, num_candidates)

    # Trouver l'indice du négatif le plus difficile pour chaque exemple
    hard_indices = torch.argmin(distances_negatives, dim=1)  # (batch_size,)

    # Récupérer les embeddings du négatif le plus difficile
    hard_negatives = torch.stack(negative_embeddings_list, dim=1)  # (batch_size, num_candidates, emb_dim)
    hard_negatives = hard_negatives[torch.arange(batch_size), hard_indices]  # (batch_size, emb_dim)

    return hard_negatives, distances_negatives


# 6. Classe principale : SiameseTrainer
class SiameseTrainer:
    """
    Classe principale orchestrant l'entraînement du réseau siamois avec la
    triplet loss sur le dataset LFW.

    Fonctionnalités
    ---------------
    - Chargement et prétraitement du dataset LFW (triplets).
    - Entraînement avec Online Hard Negative Mining sur un nombre d'époques
      configurable.
    - Sauvegarde et chargement des poids du modèle.
    - Évaluation : histogrammes des distances et courbes FAR/Accuracy/F1.
    - Comparaison de plusieurs modèles pré-entraînés (méthode de classe).
    - Inférence : comparaison de deux images quelconques.
    """

    def __init__(
        self,
        lfw_root: str = 'data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path: str = 'data/archive/lfw_allnames.csv',
        batch_size: int = 32,
        margin: float = 1.0,
        lr: float = 0.0001,
        device: torch.device = None,
    ):
        """
        Parameters
        ----------
        lfw_root : str
            Chemin vers le dossier racine des images LFW
            (structure : lfw_root/NomPersonne/image.jpg).
        lfw_allnames_path : str
            Chemin vers le fichier CSV lfw_allnames.
        batch_size : int
            Taille des mini-lots pour le DataLoader.
        margin : float
            Marge de la TripletLoss (voir formule).
        lr : float
            Taux d'apprentissage de l'optimiseur Adam.
        device : torch.device, optionnel
            Device sur lequel effectuer les calculs. Si None, la détection
            automatique (mps > cuda > cpu) est utilisée.
        """
        self.device = device if device is not None else _get_device()
        print(f"Using device: {self.device}")

        # ---- Transformations appliquées aux images ----
        # Redimensionnement à 224×224 (entrée standard de ResNet),
        # conversion en tenseur et normalisation (moyenne 0.5, écart-type 0.5
        # pour chaque canal RGB).
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalisation
        ])

        # ---- Chargement du dataset ----
        self.lfw_root = lfw_root
        self.lfw_allnames_path = lfw_allnames_path
        self.batch_size = batch_size

        self._loader = None  # chargé à la demande

        # ---- Modèle ----
        # Transfert learning : ResNet18 pré-entraîné dont seule la tête
        # fully-connected est remplacée et entraînable (voir FaceNetResNet).
        embedding_net = FaceNetResNet()
        self.model = Siamois(embedding_net).to(self.device)

        # ---- Affichage du nombre de paramètres entraînables ----
        params_a_modifier = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Nombre de paramètres à entraîner : {len(params_a_modifier)}")

        # ---- Critère de perte et optimiseur ----
        self.criterion = TripletLoss(marge=margin).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # Propriété : DataLoader (chargé une seule fois)
    @property
    def loader(self) -> DataLoader:
        """Charge le dataset LFW et crée le DataLoader si nécessaire."""
        if self._loader is None:
            if not os.path.isdir(self.lfw_root):
                raise FileNotFoundError(
                    f"Dossier LFW introuvable : {self.lfw_root}"
                )
            # Chargement brut du dataset via ImageFolder
            lfw_raw = ImageFolder(root=self.lfw_root, transform=self.transform)
            # Génération des triplets (ancre / positif / négatif)
            triplet_dataset = TripletLFW(lfw_raw, self.lfw_allnames_path)
            # Création du DataLoader avec batch de 32 triplets
            self._loader = DataLoader(
                triplet_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # num_workers=0 pour la compatibilité macOS/MPS
            )
        return self._loader

    #   Entraînement
    def train(self, n_epochs: int = 10, log_every: int = 100):
        """
        Entraîne le modèle sur n_epochs époques avec Online Hard Negative Mining.

        À chaque mini-lot :
          1. Passer l'ancre et le positif dans l'embedding.
          2. Passer le négatif (unique dans notre dataset simplifié) dans
             l'embedding — OHNM non utilisé ici car TripletLFW renvoie un
             seul négatif ; pour activer le vrai OHNM, le dataset devrait
             renvoyer plusieurs candidats négatifs.
          3. Calculer la TripletLoss et rétropropager.

        Parameters
        ----------
        n_epochs : int
            Nombre d'époques d'entraînement.
        log_every : int
            Afficher la perte toutes les ``log_every`` itérations.
        """
        self.model.train()

        for epoch in range(n_epochs):
            running_loss = 0.0  # perte cumulée pour l'affichage

            for batch_index, (data, _) in enumerate(self.loader):
                (anchor, positive, negative) = data
                anchor   = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Remise à zéro des gradients
                self.optimizer.zero_grad()

                # Obtenir les embeddings pour l'ancre et le positif
                anchor_out   = self.model.embedding(anchor)
                positive_out = self.model.embedding(positive)

                # --- Online Hard Negative Mining ---
                # Dans cet exemple, un seul négatif est fourni par le dataset.
                # Pour un vrai OHNM, il faudrait plusieurs candidats négatifs.
                negative_embeddings = [self.model.embedding(negative)]

                # Appliquer le Hard Negative Mining pour sélectionner
                # le meilleur négatif parmi les candidats
                negative_out, _ = hard_negative_mining(
                    anchor_out, positive_out, negative_embeddings
                )

                # Calcul de la perte avec le négatif sélectionné
                loss = self.criterion(anchor_out, positive_out, negative_out)

                # Rétropropagation (calcul des gradients via la fonction forward modifiée)
                loss.backward()

                # Mise à jour des poids du modèle
                self.optimizer.step()

                # ---- Affichage des statistiques ----
                l = loss.item()
                running_loss += l
                if batch_index % log_every == 0:  # Afficher toutes les log_every mini-batches
                    print(f'[Epoch {epoch + 1}, Batch {batch_index}] loss: {l:.3f}')

            avg_loss = running_loss / len(self.loader)
            print(f'Epoch {epoch + 1}/{n_epochs}  —  avg loss: {avg_loss:.3f}')

    # Sauvegarde des poids
    def save_weights(self, path: str = 'model_resnet_triplet.pth'):
        """
        Sauvegarde l'état du modèle (state_dict) dans un fichier .pth.

        Parameters
        ----------
        path : str
            Chemin de destination du fichier de poids.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Poids sauvegardés → {path}")

    # Chargement des poids
    def load_weights(self, path: str):
        """
        Charge les poids d'un fichier .pth dans le modèle courant.

        Parameters
        ----------
        path : str
            Chemin du fichier de poids à charger.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Poids chargés depuis {path}")

    #   Évaluation : histogrammes + métriques
    def evaluate(
        self,
        n_batches: int = 100,
        save_prefix: str = 'evaluation',
        n_thresholds: int = 200,
    ):
        """
        Évalue le modèle courant sur le dataset LFW.

        Génère :
          1. Un histogramme des distances intra-classe (positive) et
             inter-classe (négative) avec le seuil optimal marqué.
          2. Une courbe Accuracy en fonction du FAR (False Alert Rate).
          3. Une courbe du F1-Score en fonction du seuil de distance.

        Parameters
        ----------
        n_batches : int
            Nombre de mini-lots utilisés pour l'évaluation
            (limite la durée si le dataset est grand).
        save_prefix : str
            Préfixe pour les fichiers images générés
            (``<save_prefix>_histogram.png``, ``<save_prefix>_curves.png``).
        n_thresholds : int
            Nombre de seuils testés pour le calcul des métriques.

        Returns
        -------
        dict avec les clés 'best_acc', 'best_threshold', 'best_far'.
        """
        self.model.eval()
        pos_distances = []
        neg_distances = []

        print(f"Évaluation sur {n_batches} mini-lots…")

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.loader):
                if batch_idx >= n_batches:
                    break

                anchor, positive, negative = data
                anchor   = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # Calcul des embeddings
                emb_a = self.model.embedding(anchor)
                emb_p = self.model.embedding(positive)
                emb_n = self.model.embedding(negative)

                # Distance euclidienne entre ancre-positif et ancre-négatif
                dist_p = (emb_a - emb_p).pow(2).sum(1).sqrt()
                dist_n = (emb_a - emb_n).pow(2).sum(1).sqrt()

                pos_distances.extend(dist_p.cpu().numpy())
                neg_distances.extend(dist_n.cpu().numpy())

        pos_distances = np.array(pos_distances)
        neg_distances = np.array(neg_distances)

        # ---- Calcul des métriques pour chaque seuil ----
        min_d = min(pos_distances.min(), neg_distances.min())
        max_d = max(pos_distances.max(), neg_distances.max())
        thresholds = np.linspace(min_d, max_d, n_thresholds)

        best_acc   = 0.0
        best_thresh = 0.0
        best_far   = 0.0

        len_pos = len(pos_distances)
        len_neg = len(neg_distances)

        model_fars = []
        model_accs = []
        model_f1s  = []

        for th in thresholds:
            tp = np.sum(pos_distances <  th)  # Vrais positifs (même personne, distance < seuil)
            tn = np.sum(neg_distances >= th)  # Vrais négatifs (personnes diff., distance ≥ seuil)
            fp = np.sum(neg_distances <  th)  # Fausse acceptation (False Acceptance)
            fn = np.sum(pos_distances >= th)  # Faux rejet (False Rejection)

            acc = (tp + tn) / (len_pos + len_neg) if (len_pos + len_neg) > 0 else 0
            far = fp / len_neg if len_neg > 0 else 0  # Taux de Fausse Alerte

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            model_fars.append(far)
            model_accs.append(acc)
            model_f1s.append(f1)

            if acc > best_acc:
                best_acc    = acc
                best_thresh = th
                best_far    = far

        print(f"Meilleure accuracy : {best_acc:.2%}")
        print(f"Seuil optimal      : {best_thresh:.4f}")
        print(f"FAR (Taux de Fausse Alerte) au seuil optimal : {best_far:.2%}")

        # ---- Graphique 1 : Histogramme des distances ----
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pos_distances, bins=50, alpha=0.6, color='green',
                label='Même personne (positif)', density=True)
        ax.hist(neg_distances, bins=50, alpha=0.6, color='red',
                label='Personnes différentes (négatif)', density=True)
        ax.axvline(best_thresh, color='blue', linestyle='--', linewidth=2,
                   label=f'Seuil : {best_thresh:.2f}')
        ax.set_title(
            f"Acc: {best_acc:.1%} | FAR: {best_far:.1%} | Seuil: {best_thresh:.2f}"
        )
        ax.set_xlabel("Distance euclidienne")
        ax.set_ylabel("Densité")
        ax.legend(loc='upper right')
        plt.tight_layout()
        hist_path = f'{save_prefix}_histogram.png'
        plt.savefig(hist_path)
        plt.close(fig)
        print(f"Histogramme sauvegardé → {hist_path}")

        # ---- Graphique 2 : Courbes FAR/Accuracy et F1/Seuil ----
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Courbe Accuracy en fonction du FAR
        # Le tri par FAR permet d'avoir une courbe continue
        sort_idx = np.argsort(model_fars)
        ax1.plot(np.array(model_fars)[sort_idx], np.array(model_accs)[sort_idx], lw=2)
        ax1.set_title("Accuracy en fonction du Taux de Fausse Alerte (FAR)")
        ax1.set_xlabel("Taux de Fausse Alerte (FAR)")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Courbe F1-Score en fonction du seuil
        ax2.plot(thresholds, model_f1s, lw=2)
        ax2.set_title("F1 Score en fonction du Seuil (Threshold)")
        ax2.set_xlabel("Seuil de Distance euclidienne")
        ax2.set_ylabel("F1 Score")
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        curves_path = f'{save_prefix}_curves.png'
        plt.savefig(curves_path)
        plt.close(fig2)
        print(f"Courbes sauvegardées → {curves_path}")

        return {
            'best_acc':       best_acc,
            'best_threshold': best_thresh,
            'best_far':       best_far,
            'thresholds':     thresholds,
            'fars':           model_fars,
            'accs':           model_accs,
            'f1s':            model_f1s,
        }

    #   Dénormalisation d'un tenseur image (pour affichage matplotlib)
    @staticmethod
    def _denormaliser(tensor: torch.Tensor) -> np.ndarray:
        """
        Inverse la normalisation appliquée lors du pré-traitement afin de
        pouvoir afficher l'image avec matplotlib.

        Parameters
        ----------
        tensor : Tensor, shape (C, H, W)
            Image normalisée (moyenne 0.5, écart-type 0.5 par canal).

        Returns
        -------
        np.ndarray, shape (H, W, C) avec valeurs dans [0, 1].
        """
        tensor = tensor.clone()  # Crée une copie pour éviter de modifier l'original

        # Inversion de la normalisation appliquée lors du pré-traitement
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        tensor = tensor * std + mean

        # S'assure que les valeurs sont entre 0 et 1
        tensor = torch.clamp(tensor, 0, 1)

        # Convertit le tenseur en format numpy pour affichage avec matplotlib
        return tensor.permute(1, 2, 0).cpu().numpy()

    #   Inférence : comparaison de deux images
    def evaluer_paire(
        self,
        img1,
        img2,
        threshold: float = 1.0,
        normalized: bool = True,
        show: bool = True,
    ) -> dict:
        """
        Évalue si deux images représentent la même personne.

        Parameters
        ----------
        img1, img2 :
            Images à comparer. Deux formats possibles :
            - Si ``normalized=True`` : tenseurs PyTorch normalisés shape (C, H, W).
            - Si ``normalized=False`` : tableaux numpy shape (H, W, C),
              valeurs dans [0, 255].
        threshold : float
            Seuil de distance en-dessous duquel les deux images sont
            considérées comme représentant la même personne.
        normalized : bool
            Indique si les images sont déjà normalisées (True) ou non (False).
        show : bool
            Si True, affiche la figure matplotlib avec les deux images et
            le résultat.

        Returns
        -------
        dict avec les clés 'distance', 'same_person', 'verdict'.
        """
        self.model.eval()

        # ---- Préparation des images ----
        if normalized:
            # Images déjà normalisées : ajout de la dimension batch
            img1_batch = img1.unsqueeze(0).to(self.device)
            img2_batch = img2.unsqueeze(0).to(self.device)
        else:
            # Images numpy non normalisées : conversion en tenseur
            img1_batch = (torch.tensor(img1).permute(2, 0, 1)
                          .unsqueeze(0).float().to(self.device))
            img2_batch = (torch.tensor(img2).permute(2, 0, 1)
                          .unsqueeze(0).float().to(self.device))

        with torch.no_grad():
            emb1 = self.model.embedding(img1_batch)
            emb2 = self.model.embedding(img2_batch)

            # Distance euclidienne entre les deux embeddings
            distance = (emb1 - emb2).pow(2).sum(1).sqrt().item()

        # ---- Décision ----
        same_person = distance < threshold
        color   = "green" if same_person else "red"
        verdict = "même personne" if same_person else "personnes différentes"

        # ---- Affichage ----
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            img1_disp = self._denormaliser(img1) if normalized else img1
            img2_disp = self._denormaliser(img2) if normalized else img2

            axes[0].imshow(img1_disp)
            axes[0].set_title("Image 1")
            axes[0].axis('off')

            axes[1].imshow(img2_disp)
            axes[1].set_title(
                f"Distance : {distance:.4f}\n"
                f"Seuil : {threshold} → Verdict : {verdict}",
                color=color,
            )
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

        return {'distance': distance, 'same_person': same_person, 'verdict': verdict}

    #   Méthode de classe : comparaison de plusieurs modèles (selon le niveau d'entrainement)
    @classmethod
    def compare_models(
        cls,
        models_list: list,
        lfw_root: str = 'data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path: str = 'data/archive/lfw_allnames.csv',
        batch_size: int = 32,
        n_batches: int = 100,
        save_prefix: str = 'comparison',
        device: torch.device = None,
    ):
        """
        Compare plusieurs modèles pré-entraînés et génère :
          1. Une figure d'histogrammes (un subplot par modèle).
          2. Une figure de courbes comparatives (FAR/Accuracy + F1/Seuil).

        Reproduit exactement les graphiques de ``compare_models.py``.

        Parameters
        ----------
        models_list : list[tuple[str, str]]
            Liste de paires (nom_affichage, chemin_fichier_poids).
            Exemple :
                [('ResNet 13 epochs', 'model_resnet_triplet13epocheshardmining.pth'),
                 ('ResNet 50 epochs', 'model_resnet_triplet50.pth')]
        lfw_root : str
            Chemin vers le dossier racine des images LFW.
        lfw_allnames_path : str
            Chemin vers le fichier CSV lfw_allnames.
        batch_size : int
            Taille des mini-lots.
        n_batches : int
            Nombre de mini-lots évalués par modèle (~3 200 triplets à 32).
        save_prefix : str
            Préfixe des fichiers images générés
            (``<save_prefix>_results.png``, ``<save_prefix>_curves.png``).
        device : torch.device, optionnel
            Device de calcul. Détection automatique si None.
        """
        if device is None:
            device = _get_device()

        # ---- Chargement unique du DataLoader ----
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        lfw_raw        = ImageFolder(root=lfw_root, transform=transform)
        triplet_dataset = TripletLFW(lfw_raw, lfw_allnames_path)
        loader          = DataLoader(triplet_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=0)

        # ---- Grille pour les histogrammes ----
        n_models = len(models_list)
        ncols    = min(n_models, 2)
        nrows    = (n_models + ncols - 1) // ncols
        fig_hist, axes = plt.subplots(nrows, ncols,
                                      figsize=(8 * ncols, 6 * nrows))
        # Toujours avoir un tableau 1-D d'axes
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Dictionnaire pour stocker les métriques (courbes comparatives)
        metrics_per_model = {}

        for i, (name, path) in enumerate(models_list):
            ax = axes[i]
            print(f"\nTraitement de « {name} » depuis {path}…")

            if not os.path.exists(path):
                print(f"  Fichier {path} introuvable. Ignoré.")
                ax.text(0.5, 0.5, f"Fichier introuvable :\n{path}",
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # ---- Initialisation du modèle ----
            embedding_net = FaceNetResNet()
            model         = Siamois(embedding_net).to(device)

            try:
                state_dict = torch.load(path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
            except Exception as e:
                print(f"  Erreur lors du chargement : {e}")
                ax.text(0.5, 0.5, f"Erreur de chargement :\n{e}",
                        ha='center', va='center', transform=ax.transAxes)
                continue

            # ---- Boucle d'évaluation ----
            pos_distances = []
            neg_distances = []

            print(f"  Évaluation de « {name} » sur {n_batches} mini-lots…")

            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(loader):
                    if batch_idx >= n_batches:  # Évaluer sur ~3 200 triplets
                        break

                    anchor, positive, negative = data
                    anchor   = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)

                    emb_a = model.embedding(anchor)
                    emb_p = model.embedding(positive)
                    emb_n = model.embedding(negative)

                    # Distance euclidienne
                    dist_p = (emb_a - emb_p).pow(2).sum(1).sqrt()
                    dist_n = (emb_a - emb_n).pow(2).sum(1).sqrt()

                    pos_distances.extend(dist_p.cpu().numpy())
                    neg_distances.extend(dist_n.cpu().numpy())

            pos_distances = np.array(pos_distances)
            neg_distances = np.array(neg_distances)

            # ---- Calcul des métriques ----
            min_d = min(pos_distances.min(), neg_distances.min())
            max_d = max(pos_distances.max(), neg_distances.max())
            thresholds = np.linspace(min_d, max_d, 200)

            best_acc    = 0.0
            best_thresh = 0.0
            best_far    = 0.0

            len_pos = len(pos_distances)
            len_neg = len(neg_distances)

            model_fars = []
            model_accs = []
            model_f1s  = []

            for th in thresholds:
                tp = np.sum(pos_distances <  th)
                tn = np.sum(neg_distances >= th)
                fp = np.sum(neg_distances <  th)  # Fausse Acceptation
                fn = np.sum(pos_distances >= th)  # Faux Rejet

                acc = (tp + tn) / (len_pos + len_neg) if (len_pos + len_neg) > 0 else 0
                far = fp / len_neg if len_neg > 0 else 0  # Taux de Fausse Alerte

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1        = (2 * precision * recall / (precision + recall)
                             if (precision + recall) > 0 else 0)

                model_fars.append(far)
                model_accs.append(acc)
                model_f1s.append(f1)

                if acc > best_acc:
                    best_acc    = acc
                    best_thresh = th
                    best_far    = far

            metrics_per_model[name] = {
                'thresholds': thresholds,
                'fars':       model_fars,
                'accs':       model_accs,
                'f1s':        model_f1s,
            }

            print(f"  Résultats pour « {name} » :")
            print(f"    Meilleure accuracy : {best_acc:.2%}")
            print(f"    Seuil optimal      : {best_thresh:.4f}")
            print(f"    FAR (Taux de Fausse Alerte) : {best_far:.2%}")

            # ---- Histogramme pour ce modèle ----
            ax.hist(pos_distances, bins=50, alpha=0.6, color='green',
                    label='Même visage (positif)', density=True)
            ax.hist(neg_distances, bins=50, alpha=0.6, color='red',
                    label='Visage différent (négatif)', density=True)
            ax.axvline(best_thresh, color='blue', linestyle='--', linewidth=2,
                       label=f'Seuil : {best_thresh:.2f}')
            ax.set_title(
                f"{name}\nAcc: {best_acc:.1%} | FAR: {best_far:.1%} | "
                f"Seuil: {best_thresh:.2f}"
            )
            ax.set_xlabel("Distance euclidienne")
            ax.set_ylabel("Densité")
            ax.legend(loc='upper right', fontsize='small')

        # Masquer les axes vides si le nombre de modèles est impair
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        results_path = f'{save_prefix}_results.png'
        plt.savefig(results_path)
        plt.close(fig_hist)
        print(f"\nHistogrammes sauvegardés → {results_path}")

        # ---- Figure des courbes comparatives ----
        fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for name, data in metrics_per_model.items():
            fars = np.array(data['fars'])
            accs = np.array(data['accs'])

            # Le tri par FAR permet d'avoir une courbe continue
            sort_idx = np.argsort(fars)
            ax1.plot(fars[sort_idx], accs[sort_idx], label=name, lw=2)

            ax2.plot(data['thresholds'], data['f1s'], label=name, lw=2)

        # Courbe Accuracy en fonction du FAR
        ax1.set_title("Accuracy en fonction du Taux de Fausse Alerte (FAR)")
        ax1.set_xlabel("Taux de Fausse Alerte (FAR)")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        # Courbe F1-Score en fonction du seuil
        ax2.set_title("F1 Score en fonction du Seuil (Threshold)")
        ax2.set_xlabel("Seuil de Distance euclidienne")
        ax2.set_ylabel("F1 Score")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        plt.tight_layout()
        curves_path = f'{save_prefix}_curves.png'
        plt.savefig(curves_path)
        plt.close(fig_curves)
        print(f"Courbes comparatives sauvegardées → {curves_path}")


# Usage rapide pour tester le code (il est préférable de tout configurer sois même surtout pour les chemins d'accès)
if __name__ == '__main__':
    """
    Exemple d'utilisation complet :
      1. Entraîner le modèle 3 époques sur LFW.
      2. Sauvegarder les poids.
      3. Évaluer les performances.
    """
    trainer = SiameseTrainer(
        lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path='data/archive/lfw_allnames.csv',
        batch_size=32,
        margin=1.0,
        lr=0.0001,
    )

    # Entraîner
    trainer.train(n_epochs=3, log_every=100)

    # Sauvegarder
    trainer.save_weights('model_resnet_triplet.pth')

    # Évaluer
    trainer.evaluate(n_batches=100, save_prefix='eval_results')

    # Comparer plusieurs modèles existants
    SiameseTrainer.compare_models(
        models_list=[
            ('ResNet (13 epochs)', 'model_resnet_triplet13epocheshardmining.pth'),
            ('ResNet (20 epochs)', 'model_resnet_triplet20.pth'),
            ('ResNet (40 epochs)', 'model_resnet_triplet40.pth'),
            ('ResNet (50 epochs)', 'model_resnet_triplet50.pth'),
        ],
        lfw_root='data/archive/lfw-deepfunneled/lfw-deepfunneled',
        lfw_allnames_path='data/archive/lfw_allnames.csv',
        save_prefix='comparison',
    )
