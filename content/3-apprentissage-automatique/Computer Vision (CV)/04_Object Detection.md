---
title: Object Detection (Deep Learning)
description: Localisation, sliding window, YOLO, anchor boxes, R-CNN family, landmark detection
weight: 4
---

# Object Detection (Deep Learning)

> [!info] Contenu migré depuis 03_Detection_algorithms.md
> Ce fichier couvre le pan "deep learning" de l'ancien fichier Detection Algorithms (sections IV et V). Le pan "CV classique géométrique" (boundary fitting, SIFT) est dans [[02_Detection_algorithms]], et les briques de base (edges, coins) dans [[01_Filtres classiques#Annexe (à trier) : Edge & Corner Detection]].

## I - Basis of Neural Nets for Object Detection

Classification (une seule catégorie) → Classification with localization (+ bounding box) → Detection (plusieurs objets, plusieurs classes).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im1 (4).png|400]]

### 1. Classification with localization

ConvNet classique + 4 sorties supplémentaires $b_x, b_y, b_h, b_w$ (coordonnées de la bounding box). Convention : coin haut-gauche (0,0), bas-droit (1,1), $(b_x, b_y)$ = centre du rectangle, $b_h, b_w$ = hauteur/largeur.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im2 (3).png|400]]

Vecteur cible :

$$
y = \begin{pmatrix} p_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ c_3 \end{pmatrix}
$$

$p_c$ = probabilité qu'il y ait un objet des classes ciblées ; si $p_c=1$, on sort aussi la bounding box et la classe $c_1,...,c_3$.

**Exemple 1** (objet présent, classe 2) :

$$
y = \begin{pmatrix} 1 & b_x & b_y & b_h & b_w & 0 & 1 & 0 \end{pmatrix}^T
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im3 (4).png|350]]

**Exemple 2** (pas d'objet, "?" = don't care) :

$$
y = \begin{pmatrix} 0 & ? & ? & ? & ? & ? & ? & ? \end{pmatrix}^T
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im4 (1).png|350]]

**Loss function :**

$$
\mathcal{L}(\hat{y}, y) = \begin{cases}
\sum_{k=1}^{8}(\hat{y}_k - y_k)^2 & \text{si } y_1=1 \; (p_c=1) \\
(\hat{y}_1 - y_1)^2 & \text{si } y_1 = 0
\end{cases}
$$

En pratique : log-vraisemblance pour les classes $c_i$, erreur quadratique pour la bounding box, régression logistique pour $p_c$.

### 2. Sliding Window Detection (approche naïve)

Dataset d'images de voitures "closely cropped" (juste l'objet, pas le fond). ConvNet binaire (voiture ou non).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im9 (3).png|350]]

Sliding window : on fait glisser une fenêtre de taille fixe sur l'image, on passe chaque crop au ConvNet, puis on répète avec des fenêtres de tailles croissantes.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im8 (2).png|400]]

Inconvénient majeur : coût de calcul énorme (crops indépendants), stride large → perte de qualité, stride petit → très lent.

### 3. Convolutional Implementation of Sliding Window

**FC → Conv.** On transforme les couches fully-connected en couches convolutionnelles pour partager le calcul entre les positions de la fenêtre glissante.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im10 (3).png|400]]

Implémentation basée sur OverFeat (Sermanet et al. 2014) : calcule le score de chaque fenêtre en un seul passage au lieu de séquentiellement, en partageant les calculs entre régions communes.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im12.png|400]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im13.png|400]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im14.png|400]]

Reste un problème : la précision de la position des bounding box.

### 4. Bounding Box Predictions — YOLO

Aucune fenêtre de la sliding window ne correspond parfaitement à l'objet (rouge = vrai, bleu = prédit) :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im17.png|300]]

**YOLO (You Only Look Once, Redmon et al. 2015).** Grille $3\times3$ sur l'image, on applique la classification+localisation à chaque cellule. Vecteur cible identique par cellule. YOLO assigne l'objet à la cellule contenant le **midpoint** de l'objet.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im15.png|400]]

Sortie finale : $3\times3\times8$ (ou $19\times19\times8$ en pratique, pour réduire le risque de plusieurs objets dans une même cellule). Un seul ConvNet, implémentation convolutionnelle → utilisable en temps réel.

**Encodage des coordonnées.** Dans la cellule concernée, $(0,0)$ à $(1,1)$ localement, $(b_x, b_y)$ = position relative du centre, $b_h, b_w$ = taille relative à la cellule (peut dépasser 1 si la box déborde de la cellule).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im16.png|400]]

### 5. Intersection Over Union (IoU)

$$
IoU = \frac{\text{aire de l'intersection}}{\text{aire de l'union}}
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im18.png|300]]

Détection jugée correcte si $IoU \ge 0.5$ (parfait si $=1$). Mesure la similarité entre deux bounding boxes.

### 6. Non-max Suppression (NMS)

Un même objet peut être détecté plusieurs fois (plusieurs fenêtres qui se chevauchent).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im19.png|400]]
*Détections multiples de la même voiture à cause de la sliding window.*

**Algorithme :**
1. Éliminer toutes les boxes avec $p_c \le 0.6$
2. Choisir la box avec le plus grand $p_c$ comme prédiction
3. Éliminer toute box restante avec $IoU \ge 0.5$ vis-à-vis de la box choisie
4. Répéter tant qu'il reste des boxes

### 7. Anchor Boxes

Problème : une cellule de grille ne peut détecter qu'un seul objet. Si deux objets ont un midpoint proche (ex piéton + voiture), on utilise des **anchor boxes** prédéfinies (souvent 5+, ici 2 pour l'exemple) — chaque anchor box a sa propre prédiction associée.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im20.png|350]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im21.png|400]]

On associe chaque objet ground truth à la (cellule, anchor box) de meilleur IoU. Sortie : $3\times3\times16$ (2 anchors × 8).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im22.png|207]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im23.png|400]]

En pratique deux objets avec le même midpoint sont rares ; le vrai intérêt des anchor boxes est de spécialiser l'apprentissage (une anchor pour les petits objets type piéton, une pour les grands type voiture). Les anchor boxes peuvent être choisies à la main ou via k-means sur les formes d'objets du dataset.

### 8. YOLO — training et prédiction

**Training :**

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im25.png|400]]

Sortie du ConvNet : $3\times3\times16$.

**Prédiction :**

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im26.png|350]]

Pour chaque classe (piéton, voiture, moto...), on applique le non-max suppression pour générer les prédictions finales.

### 9. Region Proposals (R-CNN family)

Inconvénient de sliding window : classifie plein de régions sans intérêt. **R-CNN** (Girshick) : segmentation préalable pour proposer ~2000 régions ("blobs") pertinentes, classifieur appliqué seulement sur ces régions.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im24.png|400]]

- **R-CNN** : propose des régions, classifie une par une, sort label + bounding box. Lent.
- **Fast R-CNN** (Girshick 2015) : même algorithme mais implémentation convolutionnelle de la sliding window. Les propositions de régions restent lentes.
- **Faster R-CNN** (Ren et al. 2016) : un réseau convolutionnel remplace l'algorithme de segmentation traditionnel pour proposer les régions.

## II - Landmark Detection

Cas général : demander à un réseau de sortir les $(x,y)$ de points importants ("landmarks") — ex détection des coins des yeux pour la reconnaissance faciale.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im6 (2).png|350]]

On peut sortir plusieurs points ($l_{1x}, l_{1y}, l_{2x}, l_{2y}, ...$), jusqu'à 64 points (129 sorties avec la classe) — utile pour les filtres AR (couronne sur la tête, etc). Nécessite un dataset $(x,y)$ annoté à la main.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im5.png|171]]

**Pose detection** : positions du corps (poignet, coude gauche...), même principe avec des landmarks définis à l'avance.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/im7 (1).png|272]]

Important : la cohérence des landmarks entre images (le premier point est toujours le coude gauche, etc).
