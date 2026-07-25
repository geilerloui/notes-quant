---
title: Detection Algorithms — Boundary & SIFT (brouillon)
description: Boundary detection (fitting de lignes/courbes, snakes, Hough), SIFT (blobs, scale space, descripteur) — CV classique géométrique
weight: 2
---

# Detection Algorithms — Boundary & SIFT

> [!warning] Brouillon
> Nettoyé depuis les notes Overleaf/LaTeX d'origine, pas encore condensé. Ce fichier couvre le pan "CV classique géométrique" (boundary fitting, SIFT). L'object detection deep learning (YOLO, R-CNN, landmarks) est dans [[04_Object Detection]]. Sections Hough Transform et Generalized Hough Transform étaient vides à l'époque, jamais rédigées.

> [!info] Section migrée
> La section "Edge Detection" (edges, gradients, Laplacien, Canny, coins/Harris) a été déplacée dans [[01_Filtres classiques#Annexe (à trier) : Edge & Corner Detection]] — elle prolonge directement Sobel/Laplacien, pas encore triée/dédupliquée avec le contenu existant.

## I - Boundary Detection

### 1. Fitting Lines and Curves

Objectif : passer des pixels d'edge aux frontières (boundaries) d'objets — trouver une ligne ou une courbe à partir d'un ensemble de points.

Pipeline typique : edge detection → seuillage (image binaire) → shrink & expand → thinning → fitting final.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im1-1.png|300]]

**Moindres carrés naïfs (distance verticale).** Problème : minimiser la distance verticale donne un résultat biaisé (la ligne rouge) sur certains nuages de points.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im1-2.png|300]]

**Distance perpendiculaire.** Paramétrisation avec $\rho$ = distance la plus courte de la ligne à l'origine.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im1-3.png|300]]

Fonction d'énergie :

$$
E=\frac{1}{N} \sum_{i} \frac{\left(x_{i} \sin \theta-y_{i} \cos \theta+\rho\right)^{2}}{\text{Distance Perpendiculaire}}
$$

→ ligne bleue, plus correcte. Très similaire à l'axe de moment second minimal vu pour les images binaires (corner detection).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im1-4.png|300]]

On retrouve les trois moments $a, b, c$ et une formule simple donnant $\theta$ et $\rho$.

**Curves.** Généralisation aux courbes :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im1-5.png|300]]

Minimiser :

$$
E=\frac{1}{N} \sum_{i}\left(y_{i}-a x_{i}^{3}-b x_{i}^{2}-c x_{i}-d\right)^{2}
$$

Résolution par moindres carrés : $\frac{\partial E}{\partial a}=0, \frac{\partial E}{\partial b}=0, \frac{\partial E}{\partial c}=0, \frac{\partial E}{\partial d}=0$. Solution close-form lourde si beaucoup d'inconnues → système linéaire :

$$
y_{i}=a x_{i}^{3}+b x_{i}^{2}+c x_{i}+d \quad \text{pour chaque } (x_i, y_i)
$$

Système sur-déterminé à 4 inconnues $(a,b,c,d)$, solution moindres carrés :

$$
X^{T} X \mathbf{a}=X^{T} \mathbf{y} \Rightarrow \mathbf{a}=\left(X^{T} X\right)^{-1} X^{T} \mathbf{y}
$$

Pseudo-inverse $X^{+}=\left(X^{T} X\right)^{-1} X^{T}$, d'où $\mathbf{a}=X^{+} \mathbf{y}$.

### 2. Active Contours (Snakes)

Outil pour trouver les frontières d'un objet.
- **Donné** : frontière approximative (contour) autour de l'objet
- **Tâche** : faire évoluer le contour pour épouser la vraie frontière

On déforme itérativement le contour initial pour qu'il soit (1) proche des pixels à fort gradient (edges), (2) lisse.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-1 (2).png|300]]

**Exemple (boundary tracking).** Utile pour tracker un objet dans une vidéo : le contour trouvé sur la frame 1 sert de contour initial pour la frame 2, etc. (lèvres, voiture...).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-2.png|300]]

**Représentation.** $n$ points de contrôle, distance fixe entre points consécutifs :

$$
\mathbf{v}=\left\{v_{i}=\left(x_{i}, y_{i}\right) \mid i=0,1,2, \ldots, n-1\right\}
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-3.png|300]]

**Forces.** Basées sur le gradient de l'image (norme du $\nabla$). On floute le gradient pour créer un champ de force/potentiel qui attire le contour même à distance. On veut maximiser la somme des carrés de la magnitude du gradient, ce qui équivaut à minimiser son opposé :

$$
\text{Maximiser } \sum \|\nabla I\|^2 \equiv \text{Minimiser } E_{\text{image}} = -\sum_{i=0}^{n-1}\left\|\nabla n_{\sigma} * I\left(v_{i}\right)\right\|^{2}
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-4.png|300]]

**Algorithme greedy** (sous-optimal, lent pour grands contours) :
1. Pour chaque point $v_i$, le déplacer dans une fenêtre $W$ où $E_{image}$ est minimum. Répéter pour tous les points
2. Si la somme des déplacements < seuil, stop. Sinon retour à 1

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-5.png|300]]

Problème : certains points de contrôle "accrochent" et créent des torsions dans le contour → il faut ajouter des contraintes d'élasticité et de lissage.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-6.png|300]]

**Énergie interne du contour** (élastique comme un élastique, lisse comme une bande métallique) :

$$
E_{\text{contour}}=\alpha E_{\text{elastic}}+\beta E_{\text{smooth}}
$$

$(\alpha, \beta)$ contrôlent l'influence de l'élasticité et du lissage. Pour un point continu $\mathbf{v}(s)=(x(s), y(s))$ :

$$
E_{\text{elastic}}=\left\|\frac{d \mathbf{v}}{d s}\right\|^{2} \qquad E_{\text{smooth}}=\left\|\frac{d^{2} \mathbf{v}}{d s^{2}}\right\|^{2}
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-7.png|300]]

Approximations discrètes au point de contrôle $\mathbf{v}_i$ :

$$
E_{\text{elastic}}(\mathbf{v}_i) \approx \left\|\mathbf{v}_{i+1}-\mathbf{v}_i\right\|^2 = (x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2
$$

$$
E_{\text{smooth}}(\mathbf{v}_i) \approx \left\|(\mathbf{v}_{i+1}-\mathbf{v}_i) - (\mathbf{v}_i - \mathbf{v}_{i-1})\right\|^2 = (x_{i+1}-2x_i+x_{i-1})^2 + (y_{i+1}-2y_i+y_{i-1})^2
$$

Le long du contour entier :

$$
E_{\text{elastic}} = \sum_{i=0}^{n-1}\left[(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2\right]
$$

$$
E_{\text{smooth}} = \sum_{i=0}^{n-1}\left[(x_{i+1}-2x_i+x_{i-1})^2 + (y_{i+1}-2y_i+y_{i-1})^2\right]
$$

**Énergie totale** à minimiser :

$$
E_{\text{total}} = E_{\text{image}} + E_{\text{contour}}
$$

**Nouvel algorithme greedy :**
1. Échantillonner uniformément le contour ($n$ points)
2. Pour chaque point, le déplacer dans une fenêtre $W$ où $E_{\text{total}}$ (contour entier) est minimum
3. Si somme des déplacements < seuil, stop. Sinon retour à 1

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-8.png|300]]

Avec les termes élastique et de lissage, convergence vers un contour propre. Effet de $\alpha$ sur le résultat :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/2.Boundary/im2-9.png|300]]

### 3. Hough Transform

*(Section vide dans les notes d'origine — à rédiger.)*

### 4. Generalized Hough Transform

*(Section vide dans les notes d'origine — à rédiger.)*

## II - SIFT (Scale Invariant Feature Transform)

### 1. What is an Interest Point?

Même scène vue sous deux angles différents : une même zone change de taille, éclairage, intensité. Un bon détecteur de points d'intérêt doit compenser ces variations pour permettre le matching entre images.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-1.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-2.png|300]]

Beaucoup de patches dans une image ne sont pas assez intéressants pour le matching.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-3.png|300]]

**Propriétés désirables d'un point/feature :**
- Contenu riche (variation de luminosité, couleur...) dans la fenêtre locale
- Signature bien définie pour le matching
- Position bien définie dans l'image
- Invariant à la rotation et à l'échelle
- Insensible aux changements d'éclairage

Les **edges/lignes** ne sont pas assez descriptifs (beaucoup de structures similaires le long d'un edge).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-4.png|300]]

Les **coins** ont été utilisés historiquement mais restent limités. Les **blobs** sont plus intéressants : après normalisation d'échelle, un blob a une apparence locale (variation de luminosité) qui définit à la fois son apparence et sa position.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-5.png|300]]

Attributs nécessaires pour un blob : position, taille (cercle), orientation (flèche), puis une description pour le matching.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im1-6.png|300]]

### 2. Detecting Blobs

**1D blob.** Trois blobs $f(x)$, B et C sont juste A à plus grande échelle. On applique la dérivée seconde d'un gaussien $n_\sigma$. Pour un blob large (C), deux zero-crossings bien séparés. Pour des blobs petits (A, B), les zero-crossings se chevauchent. Le pic diminue quand $\sigma$ augmente → on multiplie par $\sigma^2$ pour normaliser.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-1 (2).png|300]]

En faisant varier $\sigma$, on obtient un maximum net exactement au centre du blob.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-2.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-3.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-4.png|300]]

On applique la dérivée seconde du gaussien $\sigma$-normalisée à différentes échelles → maximum à la position du blob, quelle que soit sa taille. Relation entre largeur du gaussien et taille du blob :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-5.png|300]]

**Échelle caractéristique** : le $\sigma$ auquel la dérivée seconde $\sigma$-normalisée atteint son extremum.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-6.png|300]]

$$
\text{Échelle caractéristique} \propto \text{Taille du blob}
$$

$$
\frac{\text{Taille blob A}}{\text{Taille blob B}}=\frac{\sigma_A^*}{\sigma_B^*} \qquad \frac{\text{Taille blob B}}{\text{Taille blob C}}=\frac{\sigma_B^*}{\sigma_C^*}
$$

**Résumé détection de blob 1D :**
- Donné : signal 1D $f(x)$
- Calculer : $\sigma^{2} \frac{\partial^{2} n_{\sigma}}{\partial x^{2}} * f(x)$ à plusieurs échelles $(\sigma_0, ..., \sigma_k)$
- Trouver $(x^*, \sigma^*) = \arg\max_{(x,\sigma)} \left|\sigma^{2} \frac{\partial^{2} n_{\sigma}}{\partial x^{2}} * f(x)\right|$ — $x^*$ = position du blob, $\sigma^*$ = échelle caractéristique (taille)

**2D.** Le **Laplacien de Gaussien normalisé (NLoG)** est l'équivalent 2D. Position des blobs = extrema locaux du NLoG appliqué à plusieurs échelles.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-7.png|300]]

**Scale space.** Augmenter $\sigma$ = flouter davantage = réduire la résolution effective. On crée une pile d'images :

$$
S(x, y, \sigma)=n(x, y, \sigma) * I(x, y)
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-8.png|300]]

Choix des $\sigma$ : $\sigma_k = \sigma_0 s^k$, $k = 0,1,2,...$ ($s$ = multiplicateur constant, $\sigma_0$ = échelle initiale).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im2-9.png|300]]
![[im2-10.png|300]]

### 3. SIFT Detector

Proposé par David Lowe, approximation efficace du NLoG via **Difference of Gaussians (DoG)** :

$$
\text{DoG} \approx (s-1)\,\text{NLoG}
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-1 (2).png|300]]

**Implémentation.** Pile d'images lissées jusqu'à $s^k$, puis différences successives d'images (au lieu de calculer le Laplacien directement).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-2.png|300]]

Recherche des extrema dans cette pile via une fenêtre $3\times3\times3$ (lien avec la non-max suppression vue en corner detection) → features SIFT, seuillage pour ne garder que les extrema forts.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-3.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-4.png|300]]

**Invariance d'échelle.** Le NLoG atteint son pic à un $\sigma$ différent selon l'échelle de l'objet.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-5 (1).png|300]]

$$
\frac{\sigma_1^*}{\sigma_2^*} : \text{ratio des tailles de blobs}
$$

**Orientation.** Grille sur la zone du blob, gradient (magnitude + orientation) à chaque pixel de la grille pour déterminer une orientation dominante et la neutraliser.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-6.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-7.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im3-8.png|300]]

### 4. SIFT Descriptor

Signature pour le matching. Une fois orientation et échelle normalisées, on découpe la zone en une grille et on calcule un **histogramme d'orientation de gradient** par quadrant (magnitude ignorée car sensible à l'éclairage), puis on concatène les histogrammes des 4 quadrants.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-1 (1).png|300]]

**Métriques de comparaison** entre deux histogrammes $H_1(k)$, $H_2(k)$ de longueur $N$ :

Distance L2 :

$$
d(H_1, H_2)=\sqrt{\sum_{k}(H_1(k)-H_2(k))^2}
$$

Corrélation normalisée ($\bar{H}_i = \frac{1}{N}\sum_k H_i(k)$) :

$$
d(H_1, H_2)=\frac{\sum_{k}\left[(H_1(k)-\bar{H}_1)(H_2(k)-\bar{H}_2)\right]}{\sqrt{\sum_{k}(H_1(k)-\bar{H}_1)^2}\sqrt{\sum_{k}(H_2(k)-\bar{H}_2)^2}}
$$

Plus la distance est grande, meilleur le match (match parfait si $d=1$ pour la corrélation).

Intersection :

$$
d(H_1, H_2)=\sum_k \min(H_1(k), H_2(k))
$$

**Invariance à l'échelle** :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-2.png|300]]

**Invariance à la rotation** :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-3.png|300]]

**Robustesse au clutter/occlusion** (gros problème historique en reconnaissance d'objets) :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-4.png|300]]

**Applications** : stitching pour créer un panorama, warp/combinaison d'images, auto-collage (Nomura 2007).

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-5.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-6.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-7.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-8.png|300]]

**SIFT pour objets 3D.** Beaucoup de matches à faible changement de point de vue ; avec 30° de rotation ça se dégrade, à 90° presque plus aucun match. **SIFT n'est fiable que pour de petits changements de viewpoint.**

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/3.SIFT/im4-9.png|300]]

