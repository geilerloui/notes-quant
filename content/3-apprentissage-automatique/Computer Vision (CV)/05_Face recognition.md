---
title: Face Recognition
description: Verification vs recognition, one-shot learning, Siamese networks, triplet loss
weight: 5
---

# Face Recognition

## 1. What is face recognition?

Vocabulaire :
- **Face verification problem** : on donne une image et un nom/ID. Le système doit vérifier si c'est bien la personne revendiquée (comparaison 1-vs-1).
- **Recognition problem** : on a une base de $K$ personnes. Étant donné une image en entrée, le système doit sortir l'ID si l'image correspond à l'une des $K$ personnes (ou "not recognized").

On se concentre ici sur le problème de **verification**.

## 2. One-shot learning

Un des défis de la face recognition est de résoudre le problème du **one-shot learning** : pour la plupart des applications, il faut reconnaître une personne à partir d'une seule image (un seul exemple de son visage). Historiquement, les algorithmes de deep learning ne fonctionnent pas bien avec un seul exemple d'entraînement.

Exemple : une base de 4 photos d'employés. Quelqu'un se présente à l'accueil, le système doit le reconnaître avec seulement quelques images en base.

**(i) Approche "deep nets classique".** On donne l'image d'une personne, on la passe dans un ConvNet, softmax à 5 sorties (le 5e = "personne inconnue"). Problème : pas assez d'exemples pour entraîner un deep net, et si un nouveau collègue arrive il faut tout réentraîner. Mauvaise approche.

**(ii) Approche par fonction de similarité.** On apprend plutôt une fonction $d()$ qui prend deux images en entrée et sort leur degré de différence :

$$
d(img_1, img_2) = \text{degré de différence entre les images}
$$

$$
\begin{cases}
d(img_1, img_2) \le \tau \Rightarrow \text{"même personne"}\\
d(img_1, img_2) > \tau \Rightarrow \text{"personnes différentes"}\\
\end{cases}
$$

$\tau$ est un hyperparamètre (seuil). C'est ce qu'on utilise pour la verification.

Pour la **recognition** : on calcule la distance entre la nouvelle image et chaque personne de la base avec $d()$, et on prend la meilleure correspondance.

![[images/3-Apprentissage automatique/04_Computer vision/04_Face recognition/im1.png|400]]

Calculer cette fonction de distance permet de résoudre le problème du one-shot learning.

## 3. Siamese Network

Le rôle de $d()$ est de prendre deux visages en entrée et de dire à quel point ils sont proches. Une bonne façon de faire ça est le **Siamese network** : on passe une image à travers une séquence de couches et on obtient un vecteur de features $f(x^{(1)})$ (parfois suivi d'un softmax, qu'on n'utilise pas ici). On prend $f(x^{(1)})$ de taille $128$ — on peut le voir comme un **encodage** de l'image $x^{(1)}$.

![[images/3-Apprentissage automatique/04_Computer vision/04_Face recognition/im2.png|400]]

Pour comparer deux images, on passe les deux images dans le **même réseau** (poids partagés), on obtient deux vecteurs de taille 128 : $f(x^{(1)})$ et $f(x^{(2)})$.

Distance :

$$
d(x^{(1)}, x^{(2)}) = ||f(x^{(1)}) - f(x^{(2)})||_2^2
$$

Ce processus s'appelle une architecture **Siamese Neural Network**. Beaucoup de ces idées viennent du papier [DeepFace (Taigman et al. 2014)](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html).

**Training.** On veut que l'encodage retourne une distance correcte :

$$
\text{Si } x^{(i)}, x^{(j)} \text{ sont la même personne, } \left\|f(x^{(i)})-f(x^{(j)})\right\|^{2} \text{ est petit.}
$$

$$
\text{Si } x^{(i)}, x^{(j)} \text{ sont des personnes différentes, } \left\|f(x^{(i)})-f(x^{(j)})\right\|^{2} \text{ est grand.}
$$

Puis on met à jour les paramètres par backpropagation.

## 4. Triplet Loss

Une façon d'apprendre un bon encodage est la fonction **triplet loss**. On compare des paires d'images : on veut que "anchor" et "positive" (même personne) soient proches, et que "anchor" et "negative" (personne différente) soient très éloignés.

![[images/3-Apprentissage automatique/04_Computer vision/04_Face recognition/im3.png|400]]

Triplet loss = on regarde un anchor, un positive et un negative :

$$
\begin{aligned}
&||f(A) - f(P) ||^2 \le ||f(A) - f(N) ||^2 \\
&||f(A) - f(P) ||^2 - ||f(A) - f(N) ||^2 \le 0 \\
\end{aligned}
$$

Problème : l'encodage pourrait trivialement satisfaire ça avec $f(\text{image})=0$ partout. Pour empêcher ça, on modifie l'objectif avec une **marge** $\alpha$ :

$$
\begin{aligned}
&||f(A) - f(P) ||^2 - ||f(A) - f(N) ||^2 \le 0 - \alpha\\
&||f(A) - f(P) ||^2 - ||f(A) - f(N) ||^2 + \alpha \le 0 
\end{aligned}
$$

Avec $||f(A) - f(P) ||^2 = d(A,P)$... $\alpha$ est la "marge" (terminologie héritée des SVM). On peut aussi réécrire :

$$
||f(A) - f(P) ||^2 + \alpha \le ||f(A) - f(N) ||^2 
$$

Exemple : $\alpha=0.2$ et $d(A,P)=0.5$ → il faut $d(A,N) \ge 0.7$.

**Dérivation de la fonction de triplet loss.** Pour trois images A, P, N :

$$
L(A, P, N) = \max\left( ||f(A) - f(P) ||^2 - ||f(A) - f(N) ||^2 + \alpha , 0\right)
$$

Le $\max$ avec 0 fait que si l'expression est déjà $\le 0$, la loss est nulle, sinon elle vaut l'expression. Généralisation à tous les exemples :

$$
J = \sum_{i=1}^{m} L(A^{i}, P^{i}, N^{i})
$$

Pour construire ce dataset de triplets, il faut des paires A, P de la même personne — donc plusieurs photos par personne (ex 10k photos pour 1k personnes). Avec une seule photo par personne, impossible d'entraîner ce système.

**Comment choisir les triplets ?** Si A, P, N sont choisis aléatoirement, la contrainte $d(A,P) + \alpha \le d(A,N)$ est trop facile à satisfaire et le réseau n'apprend pas grand-chose. Il faut choisir des triplets **difficiles**, où $d(A,P) \approx d(A,N)$.

Détails dans le papier [FaceNet: A unified embedding for face recognition and clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf).

## 5. Face Verification and Binary Classification

La triplet loss est une bonne façon d'apprendre les paramètres, mais il y a une autre approche : voir la face verification comme un problème de **classification binaire**.

![[images/3-Apprentissage automatique/04_Computer vision/04_Face recognition/im4.png|400]]

Deux Siamese networks dont les sorties sont passées dans une unité de régression logistique, avec target = 1 si même personne, 0 sinon.

$$
\hat{y} = \sigma\left( \sum_{k=1}^{128} w_k | f(x^{(i)})_k - f(x^{(j)})_k | + b\right)
$$

$f(x^{(i)})_k$ = $k$-ème composante de l'encodage de $x^{(i)}$. On apprend les poids $w_k$ et $b$ de la régression logistique sur ces 128 features.

Autre fonction possible, la similarité du Khi² :

$$
\hat{y} = \sigma\left( \sum_{k=1}^{128} w_k \frac{( f(x^{(i)})_k - f(x^{(j)})_k )^2}{ f(x^{(i)})_k + f(x^{(j)})_k } + b\right)
$$

$\hat{y}$ vaut 1 ou 0. Ces formulations viennent du papier DeepFace.

**Astuce de déploiement** : pour une nouvelle image à comparer à une base connue, on précalcule et stocke les embeddings des images déjà connues, et on ne calcule que l'embedding de la nouvelle image à chaque comparaison.

## 6. Quadruplet Loss

*(Section à compléter — voir [cette vidéo](https://www.youtube.com/watch?v=0Zd0iu6ZEzc).)*
