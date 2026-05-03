---
title: Convolutional Neural Networks (CNN)
description: Convolution 1D et 2D, filtres classiques, intuition pour les CNN
weight: 1
---

# Convolutional Neural Networks

Les CNN sont des réseaux de neurones spécialisés pour les données structurées spatialement (images surtout, mais aussi signaux 1D, séries temporelles, audio). Leur brique de base est la **convolution**, qui n'est rien d'autre qu'un filtre qui glisse sur le signal.

L'objectif de cette note est de comprendre :
1. Ce qu'est une convolution discrète, en 1D puis 2D
2. Quels motifs détectent les filtres classiques (dérivée 1ère, 2nde, lissage)
3. Comment un CNN apprend ces filtres tout seul

## I - Convolution 1D

### 1. Principe : glisser, multiplier, sommer

On a un signal $f$ (liste de nombres) et un **noyau** $g$ (un petit motif, le "filtre"). La convolution $f * g$ produit un nouveau signal qui mesure, en chaque point, à quel point $g$ ressemble à ce qui se passe dans $f$ autour de ce point.

L'opération a trois étapes :
1. On **pose** $g$ sur un morceau de $f$ de même taille
2. On **multiplie** case par case et on **somme** → un seul nombre, une case du résultat
3. On **décale** $g$ d'un cran à droite, on recommence

> [!note]- Convention ML vs convention mathématique
> En analyse, la convolution est définie avec un retournement du noyau :
> $$(f * g)[n] = \sum_k f[k] \cdot g[n-k]$$
> Ce retournement rend l'opération commutative et associative (utile en proba : densité d'une somme de variables indépendantes).
> 
> **En ML, les frameworks (PyTorch, TF) calculent la corrélation croisée, sans retournement** :
> $$(f \star g)[n] = \sum_k f[n+k] \cdot g[k]$$
> 
> C'est sans importance pour l'apprentissage : si le bon noyau est retourné, le réseau l'apprend retourné. Convention pure.

### 2. Exemple chiffré

Prenons $f = [1, 2, 3, 4, 5]$ et $g = [-1, 0, 1]$.

**Position 1** : on aligne $g$ sur $f[0:3] = [1, 2, 3]$
$$1 \times (-1) + 2 \times 0 + 3 \times 1 = 2$$

**Position 2** : on aligne sur $f[1:4] = [2, 3, 4]$
$$2 \times (-1) + 3 \times 0 + 4 \times 1 = 2$$

**Position 3** : on aligne sur $f[2:5] = [3, 4, 5]$
$$3 \times (-1) + 4 \times 0 + 5 \times 1 = 2$$

Résultat : $[2, 2, 2]$.

### 3. Interprétation : un filtre = un détecteur de motif

Le filtre $[-1, 0, 1]$ calcule essentiellement *« case de droite − case de gauche »*. Il **mesure de combien le signal varie localement**. Sur un signal qui monte régulièrement (pente constante), le résultat est constant. Sur un signal plat, il vaut 0 partout. Sur une transition (saut), il s'allume.

> [!note]- "Pente" ≠ fraction
> Le filtre $[-1, 0, 1]$ calcule $f[i+1] - f[i-1]$, c'est-à-dire $2h \cdot f'(x_i)$ approximativement. Ce n'est **pas** la pente au sens dérivée ($\Delta f / \Delta x$), mais une **variation locale** proportionnelle à la dérivée.
> 
> En ML on s'en fiche : le réseau apprend les coefficients, donc le facteur d'échelle se cale automatiquement avec les couches suivantes.

### 4. Tableau récapitulatif des filtres 1D classiques

| Filtre | Calcule | Approxime | Détecte |
|---|---|---|---|
| $[-1, 1]$ | $f[i+1] - f[i]$ | $f'$ (ordre 1) | variation locale (asymétrique) |
| $[-1, 0, 1]$ | $f[i+1] - f[i-1]$ | $f'$ (ordre 2) | variation locale (symétrique) |
| $[1, -2, 1]$ | $f[i-1] - 2f[i] + f[i+1]$ | $f''$ | concavité, points d'inflexion |
| $[1, 1, 1]/3$ | moyenne des 3 voisins | lissage | atténuation du bruit |

### 5. Différence simple vs centrée

Les deux approximent la dérivée première, mais avec un compromis différent :

- $[-1, 1]$ : **très local** (2 cases), **asymétrique**, erreur d'ordre $h$
- $[-1, 0, 1]$ : **moins local** (3 cases), **symétrique**, erreur d'ordre $h^2$

> [!note]- Pourquoi l'ordre 2 par symétrie (Taylor)
> Développement autour de $x_i$ avec un pas $h$ :
> $$f(x_{i+1}) = f(x_i) + h f'(x_i) + \tfrac{h^2}{2} f''(x_i) + \tfrac{h^3}{6} f'''(x_i) + O(h^4)$$
> $$f(x_{i-1}) = f(x_i) - h f'(x_i) + \tfrac{h^2}{2} f''(x_i) - \tfrac{h^3}{6} f'''(x_i) + O(h^4)$$
> 
> **Différence simple** : $f(x_{i+1}) - f(x_i) = h f'(x_i) + \tfrac{h^2}{2} f''(x_i) + \dots$
> En divisant par $h$ : on retrouve $f'(x_i)$ + erreur en $h$.
> 
> **Différence centrée** : $f(x_{i+1}) - f(x_{i-1}) = 2h f'(x_i) + \tfrac{h^3}{3} f'''(x_i) + \dots$
> Les termes en $h^2$ s'annulent par symétrie. Erreur en $h^2$ après division par $2h$.
> 
> Pour $h = 0{,}1$ : erreur simple $\sim 0{,}1$, erreur centrée $\sim 0{,}01$. Dix fois plus précis.

### 6. Dérivée seconde

Le filtre $[1, -2, 1]$ détecte la **concavité** :
- Négatif sur les zones en cloche (concaves, $f'' < 0$)
- Positif sur les zones en creux (convexes, $f'' > 0$)
- S'annule aux **points d'inflexion** (changement de concavité)

Sur $\sin(x)$ par exemple :
- $f''(x) = -\sin(x)$
- Négatif sur $[0, \pi]$ (cloche), positif sur $[\pi, 2\pi]$ (creux)
- Le filtre détecte exactement ce comportement

### 7. Lissage et bruit

Le filtre $[1, 1, 1]/3$ moyenne 3 voisins. Sur un signal lisse comme $\sin$, il ne fait quasiment rien. **Son utilité apparaît sur les signaux bruités**.

**Pourquoi c'est crucial** : la dérivation **amplifie le bruit**. Si $\tilde{f}[i] = f[i] + \varepsilon[i]$ (bruit gaussien d'écart-type $\sigma$) :

$$\tilde{f}[i+1] - \tilde{f}[i] = \underbrace{(f[i+1] - f[i])}_{\sim h \cdot f' \text{ (petit)}} + \underbrace{(\varepsilon[i+1] - \varepsilon[i])}_{\sim \sigma \text{ (reste)}}$$

Le signal utile est petit (proportionnel au pas $h$), le bruit reste de l'ordre de $\sigma$. **Le rapport signal/bruit s'effondre.** Pour la dérivée seconde, le filtre $[1, -2, 1]$ amplifie même la variance par 6 et divise par $h^2$ — résultat : sur un signal légèrement bruité, la dérivée seconde brute est totalement noyée.

D'où la stratégie systématique en pratique : **lisser avant de dériver**. Ou mieux : utiliser un filtre qui combine les deux directement (Sobel, dérivée de gaussienne, LoG).

> [!note]- Lien avec les séries temporelles financières
> Ces filtres ont des noms familiers en quant :
> - **Lissage** ↔ moyenne mobile (SMA, EMA)
> - **Dérivée 1ère** ↔ rendements, momentum
> - **Dérivée 2nde** ↔ accélération du momentum, "convexité" du prix
> 
> Un CNN-1D entraîné sur des séries de prix **redécouvre** spontanément ces opérateurs (parmi d'autres) dans ses premières couches.

## II - Convolution 2D

### 1. Image = champ scalaire 2D, deux représentations



![[im1-1 (1) 1.png|225]]

Une image en niveaux de gris est mathématiquement une **fonction de 2 variables** :
$$I : (i, j) \in \mathbb{R}^2 \longmapsto \text{intensité} \in \mathbb{R}$$

C'est ce qu'on appelle un **champ scalaire 2D** : à chaque point du plan, on associe **un scalaire** (l'intensité du pixel).

Cette même fonction admet **deux visualisations équivalentes**, qu'on retrouve dans la figure plus bas :

- **Vue "image"** (plot 1) : on encode l'intensité par une **couleur** (du noir au blanc). C'est la représentation naturelle d'une photo.
- **Vue "surface 3D"** (plot 2) : on encode l'intensité par une **altitude** $z = I(x, y)$. La fonction devient une nappe / un relief dans $\mathbb{R}^3$.

Les deux montrent **exactement le même objet**. La vue 3D rend juste explicite l'intuition que les bords sont des "falaises" (variations brutales d'altitude).

> [!note]- Vocabulaire : champ scalaire vs champ vectoriel
> | Objet | Domaine | Sortie en chaque point | Type |
> |---|---|---|---|
> | Image grayscale | $\mathbb{R}^2$ | scalaire (intensité) | **champ scalaire 2D** |
> | Image RGB | $\mathbb{R}^2$ | vecteur 3D $(R, G, B)$ | **champ vectoriel 2D** (3 canaux) |
> | Gradient d'une image grayscale $\nabla I$ | $\mathbb{R}^2$ | vecteur 2D $(\partial_x I, \partial_y I)$ | **champ de gradient** (= champ vectoriel issu d'une dérivation) |
> | Scan IRM / volume | $\mathbb{R}^3$ | scalaire | **champ scalaire 3D** |
>
> Ne pas confondre **canal** (composante de la sortie vectorielle, ex. R, G, B) avec **dimension spatiale** (variable d'entrée). Une image RGB reste définie sur un domaine 2D ; les 3 canaux sont juste 3 fonctions scalaires empilées.

| Type de donnée | Variables d'entrée | Sortie par point | Conv utilisée |
|---|---|---|---|
| Signal 1D (audio, série temp.) | 1 | 1 | Conv1D |
| Image grayscale | 2 | 1 | Conv2D |
| Image RGB | 2 | 3 | Conv2D (3 canaux d'entrée) |
| Vidéo (RGB + temps) | 3 | 3 | Conv3D ou Conv2D+1D |
| Scan médical (volume) | 3 | 1 | Conv3D |

### 2. Détecter un bord = détecter une falaise

Un **bord** dans une image, c'est un endroit où l'intensité change brutalement (transition noir → blanc par exemple). Sur la vue 3D, un bord apparaît comme une **falaise** : un endroit où l'altitude $I(x, y)$ chute fortement sur une courte distance.

Détecter les bords revient donc à détecter les **fortes variations locales** de l'intensité. C'est exactement ce que font les filtres dérivée première et seconde, généralisés en 2D.

### 3. Filtres 2D classiques

**Sobel** (dérivée première, séparée par direction) :

$$K_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \qquad K_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

$K_x$ détecte les variations selon $x$ (bords verticaux), $K_y$ selon $y$ (bords horizontaux). On combine les deux pour obtenir la **magnitude du gradient** :

$$|\nabla I| = \sqrt{(I * K_x)^2 + (I * K_y)^2}$$

**Laplacien** (dérivée seconde, isotrope) :

$$K_L = \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

C'est la version 2D du filtre $[1, -2, 1]$ : il calcule $\Delta I = \partial_{xx} I + \partial_{yy} I$.

> [!note]- Pourquoi les coefficients $1, 2, 1$ de Sobel ?
> Sobel n'est **pas** une dérivée pure. Le pattern $1, 2, 1$ est une approximation de **lissage gaussien** (poids plus fort au centre).
> 
> $K_x$ se factorise comme :
> $$K_x = \underbrace{\begin{pmatrix} 1 \\ 2 \\ 1 \end{pmatrix}}_{\text{lissage vertical}} \otimes \underbrace{\begin{pmatrix} -1 & 0 & 1 \end{pmatrix}}_{\text{dérivée horizontale}}$$
> 
> Donc Sobel = **dérivée dans une direction + lissage dans la perpendiculaire**, en un seul filtre. C'est exactement le compromis "lisser avant de dériver" vu en 1D, intégré dans le noyau lui-même.

### 4. Sobel vs Laplacien : deux façons de détecter un bord

Les deux détectent les bords, mais leur réponse est très différente :

| Filtre | Comportement sur un bord | Forme dans la sortie |
|---|---|---|
| **Sobel** ($f'$) | s'allume **fort** sur le bord | un **pic** large à la position du bord |
| **Laplacien** ($f''$) | passe **par zéro** au milieu du bord | un **lobe positif puis négatif** (zero-crossing) |

En pratique en computer vision classique, on préfère souvent **Sobel** pour deux raisons :
1. Plus **robuste au bruit** (la dérivée seconde amplifie le bruit comme on l'a vu en 1D)
2. Donne directement la **direction** du gradient en plus de la magnitude (utile pour orienter les bords)

Le Laplacien reste utile pour la localisation **sub-pixel** (le zero-crossing est très précis) et pour la détection de blobs (filtre LoG = Laplacien de gaussienne).

### 5. Visualisation : disque sur fond sombre

![[Pasted image 20260502220100.png]]

*Figure : Application de Sobel et du Laplacien sur une image grayscale (disque clair sur fond sombre). En haut : (1) le champ scalaire visualisé en image — intensité encodée par couleur, (2) le **même** champ scalaire visualisé comme surface 3D — intensité encodée par altitude, (3) coupe horizontale au niveau du centre montrant le profil "falaise" du bord. En bas : (4) magnitude du gradient $|\nabla I|$ via Sobel, (5) Laplacien $\Delta I$, et (6) coupe 1D comparant les deux réponses.*

**Lecture des plots :**

- **(1) et (2) sont la même fonction** : un champ scalaire 2D, vu d'abord comme image (couleur = intensité) puis comme surface (altitude = intensité). C'est cette dualité qui rend l'analogie "bord = falaise" littérale plutôt que métaphorique.
- **(2) Surface 3D** : on voit littéralement la "falaise" — un plateau à intensité 1 au centre, qui tombe vers 0 sur les bords. C'est cette pente raide qui définit géométriquement le bord.
- **(4) Sobel** : un **anneau lumineux** parfait sur tout le pourtour du disque. Le filtre s'allume **partout où il y a une falaise**, peu importe son orientation, parce que la magnitude $\sqrt{G_x^2 + G_y^2}$ est isotrope.
- **(5) Laplacien** : un **double anneau** — un lobe rouge (positif) côté extérieur du bord, un lobe bleu (négatif) côté intérieur. Le bord exact se situe **au zero-crossing entre les deux**. On voit aussi nettement la sensibilité au bruit (pixels rouges/bleus parasites en dehors du disque), absente sur Sobel.
- **(6) Coupe 1D** : Sobel fait deux **pics** (entrée et sortie du disque), Laplacien fait deux **changements de signe** au même endroit. C'est l'analogue exact de ce qu'on a vu sur $\sin(x)$ avec les filtres $[-1, 0, 1]$ et $[1, -2, 1]$.

### 6. Ce qu'apprend un CNN

Quand on visualise les filtres appris en première couche d'un CNN entraîné sur ImageNet, on retrouve :

- des filtres **Sobel-like** (dérivée orientée + lissage perpendiculaire)
- des filtres **Gabor** (oscillation × gaussienne, détecteurs de texture orientée)
- des filtres **blob** (LoG-like, détecteurs de taches)

Le réseau **redécouvre** automatiquement le zoo classique du traitement du signal — sans qu'on lui dise quoi chercher. Les couches plus profondes combinent ensuite ces réponses bas niveau en motifs de plus en plus abstraits (textures → parties d'objets → objets entiers).

**À compléter** : taille de filtre, padding/stride/dilation, pooling, partage de poids, équivariance par translation, architecture en blocs, champ réceptif.

## III - À retenir

- Une convolution discrète, c'est **un filtre qui glisse, multiplie, somme**. Rien de plus.
- Chaque filtre est un **détecteur de motif local** (variation, concavité, moyenne, bord, texture, etc.).
- Sur des signaux réels (bruités), on **combine toujours lissage + dérivation**, soit en deux passes, soit dans un filtre intelligent (Sobel, LoG).
- Un CNN, c'est un empilement de telles convolutions où **les coefficients des filtres sont appris par descente de gradient** au lieu d'être fixés à la main.
- Les premières couches d'un CNN entraîné sur des images réelles redécouvrent spontanément les filtres classiques du traitement du signal (Sobel, Gabor, LoG-like).
