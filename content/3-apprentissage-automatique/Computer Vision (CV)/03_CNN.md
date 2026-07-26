---
title: Convolutional Neural Networks (CNN)
description: Pourquoi la convolution, volume/canaux, hyperparamètres, architectures classiques, symétries (Mallat)
weight: 3
---

# Convolutional Neural Networks

Les CNN sont des réseaux de neurones spécialisés pour les données structurées spatialement (images surtout, mais aussi signaux 1D, séries temporelles, audio). Leur brique de base est la **convolution**, qui n'est rien d'autre qu'un filtre qui glisse sur le signal.

L'objectif de cette note est de comprendre :
1. Pourquoi utiliser la convolution plutôt qu'un réseau fully-connected sur une image
2. Ce que la convolution 2D ajoute par rapport aux filtres classiques ([[01_Filtres classiques]]) : canaux/volume, et des filtres **appris** plutôt que choisis à la main
3. L'architecture d'un CNN (hyperparamètres, couches, réseaux classiques)
4. Pourquoi ça marche en profondeur (symétries, hiérarchie — section Mallat)

*(Convolution 1D/2D de base, filtres classiques (Sobel, Laplacien...) : déplacés dans [[01_Filtres classiques]].)*

## I - Pourquoi la convolution

Avant les CNN, une couche de réseau de neurones était **fully-connected** : chaque neurone de sortie est connecté à *tous* les pixels d'entrée.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im15.png]]

**Le problème : ça explose en nombre de paramètres.** Pour une grande image $1000\times1000\times3$ (RGB), on a 3 millions de features en entrée. Avec 1000 neurones en première couche cachée, la matrice de poids $W^{(1)}$ ferait $1000 \times 3M$, soit 3 milliards de paramètres — bien trop pour éviter l'overfitting, et trop coûteux à stocker.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im1 (4).png]]

Autre exemple : image $32\times32\times3$, filtre $5\times5$, 6 filtres → sortie $28\times28\times6$. En fully-connected, ça donnerait $3072 \times 4704 \approx 14M$ paramètres.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im16.png]]

**La solution : la convolution.** Un filtre $5\times5$ a 25 paramètres $+1$ biais $=26$. Avec 6 filtres : $6\times26=156$ paramètres, au lieu de 16 millions. Ce gain vient de trois propriétés :

**1. Partage de poids (parameter sharing).** Un filtre $3\times3$ appris peut s'appliquer n'importe où dans l'image — un détecteur de bord vertical utile dans une partie de l'image l'est probablement aussi ailleurs.

**2. Connexions creuses (sparsity of connections).** Dans chaque couche, chaque valeur de sortie ne dépend que d'un petit nombre d'entrées : le pixel de sortie en haut à gauche n'est connecté qu'à 9 pixels d'entrée, tous les autres n'ont aucun effet dessus.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im17.png]]

**3. Invariance par translation.** Une image d'un chat décalée de quelques pixels reste une image d'un chat.

*(Note technique corrélation vs convolution déjà traitée dans [[01_Filtres classiques#III - Template Matching (corrélation croisée)]].)*

## II - Volume, canaux, filtres appris

Tout ce qu'on a vu dans [[01_Filtres classiques]] reste en **1 canal** (image grayscale). Ce qui change vraiment avec les CNN : le passage à un **volume multi-canal**, et des filtres **appris** au lieu d'être choisis à la main.

### Convolutions sur un volume

Passons à une image 3D (RGB), le 3 étant le nombre de canaux. L'image de sortie, elle, a un seul canal :

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im11 (1).png]]

**Exemple — bords verticaux.** Pour détecter des bords verticaux uniquement dans le canal rouge, on met toutes les valeurs du filtre à 0 sauf sur le canal rouge. Pour détecter des bords peu importe la couleur, on met le vecteur central à 0 :

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im12.png]]

**Couche de convolution dans un réseau de neurones.** On ajoute un biais au filtre et on passe le résultat dans une ReLU (ou une autre activation). On empile ensuite tous les filtres pour créer un nouveau tenseur :

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im14.png]]

### Filtres appris plutôt que choisis à la main

En traitement d'image classique, on choisissait le filtre à la main (Sobel, Scharr...) :

$$
\begin{bmatrix} 1 & 0 & -1 \\ 2 & 0 & -2 \\ 1 & 0 & -1 \end{bmatrix}
\quad
\begin{bmatrix} 3 & 0 & -3 \\ 10 & 0 & -10 \\ 3 & 0 & -3 \end{bmatrix}
$$

Avec le deep learning, on calcule ces filtres automatiquement par rétropropagation — le réseau peut apprendre n'importe quelle orientation de détecteur de bord (40°, 70°, 73°...), pas seulement les quelques designs historiques :

$$
Img \star \begin{bmatrix} w_1 & w_2 & w_3 \\ w_4 & w_5 & w_6 \\ w_7 & w_8 & w_9 \end{bmatrix}
$$

### Le lien avec $y = Wx + b$ (le MLP qu'on connaît déjà)

Dans un MLP, la formule est simple : $y = Wx + b$, une multiplication de matrices. Dans un CNN, l'opération ressemble à autre chose — une fenêtre qui glisse sur $X$, un calcul répété à chaque position. Pourtant, c'est **exactement la même formule**, juste maquillée. Deux niveaux pour le voir.

**Niveau 1 — le code naïf (la fenêtre qui glisse) :**

```
pour chaque filtre f (parmi C_out filtres) :
    pour chaque position de sortie (i, j) :
        patch = X[i:i+K, j:j+K, :]              # sous-fenêtre KxK, tous les canaux d'entrée
        Y[i, j, f] = somme(patch * W[f]) + b[f]  # produit terme à terme, somme
    Y[:, :, f] = activation(Y[:, :, f])
```

C'est le "glisser, multiplier, sommer" de [[01_Filtres classiques]], juste répété $C_{out}$ fois (une fois par filtre) et sommé sur 3 dimensions (pas 2, puisqu'on inclut les canaux).

**Niveau 2 — pourquoi c'est $y=Wx+b$ en réalité.** L'astuce s'appelle **im2col** (image-to-column), et c'est ce que PyTorch/TF font en interne :

1. On prend **chaque patch** visité par la fenêtre, on l'**aplatit** en un vecteur (un patch $3\times3$ → un vecteur de 9 nombres). On empile tous ces vecteurs-patchs en lignes → une matrice $X_{unfold}$ de taille (nombre de positions de sortie) × (taille du patch aplati).
2. On aplatit aussi chaque filtre en une colonne → une matrice $W$ de taille (taille du patch aplati) × $C_{out}$.
3. Alors : $Y_{unfold} = X_{unfold} \cdot W + b$ — **exactement** la formule du MLP, une seule multiplication de matrices.

**Preuve numérique.** Input $4\times4$, filtre $3\times3$ $w=\begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$, sortie $2\times2$ (4 positions) :

$$
X = \begin{pmatrix} 1&2&3&4\\5&6&7&8\\9&10&11&12\\13&14&15&16 \end{pmatrix}
$$

Chaque patch $3\times3$ aplati devient une ligne de $X_{unfold}$ — aucune valeur n'est inventée ou modifiée, on recopie juste les pixels de $X$ dans un ordre différent (ligne par ligne au lieu de carré) :

> [!note]- Détail patch par patch (les 4 lignes de $X_{unfold}$)
> - **Patch 1**, fenêtre en haut-gauche $\begin{pmatrix}1&2&3\\5&6&7\\9&10&11\end{pmatrix}$ → ligne $[1,2,3,5,6,7,9,10,11]$
> - **Patch 2**, fenêtre décalée d'une colonne $\begin{pmatrix}2&3&4\\6&7&8\\10&11&12\end{pmatrix}$ → ligne $[2,3,4,6,7,8,10,11,12]$
> - **Patch 3**, fenêtre décalée d'une ligne $\begin{pmatrix}5&6&7\\9&10&11\\13&14&15\end{pmatrix}$ → ligne $[5,6,7,9,10,11,13,14,15]$
> - **Patch 4**, fenêtre décalée d'une ligne et d'une colonne $\begin{pmatrix}6&7&8\\10&11&12\\14&15&16\end{pmatrix}$ → ligne $[6,7,8,10,11,12,14,15,16]$

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im2col_patch.png|550]]
*Le patch 1 (bordure rouge, à gauche) devient la ligne 1 de $X_{unfold}$ (bordure rouge, à droite) — même 9 valeurs, juste réarrangées.*

$$
X_{unfold} = \begin{pmatrix}
1&2&3&5&6&7&9&10&11\\
2&3&4&6&7&8&10&11&12\\
5&6&7&9&10&11&13&14&15\\
6&7&8&10&11&12&14&15&16
\end{pmatrix}
$$

Remarque : les valeurs $2,3,6,7,10,11$ apparaissent à la fois dans le Patch 1 et le Patch 2 (chevauchement des fenêtres) — chaque pixel proche du centre contribue à plusieurs patchs, donc à plusieurs lignes de $X_{unfold}$.

$$
Y = X_{unfold} \cdot w = \begin{pmatrix}-6\\-6\\-6\\-6\end{pmatrix} \;\Rightarrow\; \begin{pmatrix}-6&-6\\-6&-6\end{pmatrix}
$$

(Résultat constant partout : $X$ est un dégradé linéaire parfait, donc le filtre dérivée donne une réponse constante — même phénomène que Sobel sur une rampe, vu dans [[01_Filtres classiques#3. Sobel (dérivée première)]].)

**Ce que ça révèle.** La seule vraie différence avec le MLP n'est pas la formule ($y=Wx+b$ reste valable) — c'est **comment on construit $X$**. Au lieu d'aplatir toute l'image en un seul vecteur (fully-connected), on construit une ligne par position de sortie en piochant un petit patch, avec chevauchement entre les lignes (regarde les colonnes communes entre la 1ère et la 2ème ligne de $X_{unfold}$ ci-dessus). Ce chevauchement, combiné au fait que le **même** $w$ multiplie toutes les lignes, c'est exactement le partage de poids et la sparsité de la section I — encodés directement dans la construction de $X_{unfold}$, pas dans une formule différente.

### Rétropropagation : le gradient d'une convolution est une convolution transposée

Le mécanisme (chain rule, gradient qui remonte couche par couche) est le même que pour n'importe quel réseau — rien à redériver ici. Seule question intéressante : concrètement, à quoi ressemble le gradient une fois qu'il traverse une couche de convolution ?

$$
\frac{\partial \mathcal L}{\partial X_{unfold}} = \frac{\partial \mathcal L}{\partial Y_{unfold}} \cdot W^T
$$

Ce gradient ressort "déplié" (une ligne par patch). Pour revenir à la forme image, on replie chaque ligne à la position de son patch d'origine — et comme les patchs se chevauchent, un pixel touché par plusieurs patchs reçoit la **somme** de leurs contributions.

**Exemple minimal.** Reprenons $w = \begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$ et un gradient entrant $\frac{\partial \mathcal L}{\partial Y} = \begin{pmatrix}1&1\\1&1\end{pmatrix}$ (les 4 positions de sortie). Chaque position renvoie une copie de $w$, repliée à sa place dans la grille $4\times4$, puis on additionne les chevauchements :

- Pixel $(0,0)$ — coin, touché par un seul patch (Patch 1) : $\frac{\partial \mathcal L}{\partial X}[0,0] = w[0,0] = 1$
- Pixel $(1,1)$ — centre, touché par les 4 patchs, chacun avec une position locale différente dans $w$ : $0 + 1 + 0 + 1 = 2$

> [!note]- Détail : pourquoi le pixel $(1,1)$ reçoit $0+1+0+1=2$ (chain rule à plusieurs variables)
> Un seul filtre $w$ est réutilisé à 4 positions différentes — ce n'est pas "plusieurs filtres", c'est le même $w$ qui recouvre le pixel $X[1,1]=6$ à un endroit différent à chaque fois. Comme ce pixel influence 4 sorties ($Y[0,0], Y[0,1], Y[1,0], Y[1,1]$), la règle de la chaîne dit que son effet total sur la loss est la **somme** de son effet à travers chacune :
>
> $$
> \frac{\partial \mathcal L}{\partial X[1,1]} = \sum_{(a,b)} \frac{\partial \mathcal L}{\partial Y[a,b]} \times \frac{\partial Y[a,b]}{\partial X[1,1]}
> $$
>
> où $\frac{\partial Y[a,b]}{\partial X[1,1]}$ est simplement le poids de $w$ qui multipliait $X[1,1]$ dans ce patch précis :
>
> - **Patch 1** (fenêtre haut-gauche) : $X[1,1]$ tombe au centre du patch → poids $w[1,1] = 0$
> - **Patch 2** (fenêtre décalée à droite) : $X[1,1]$ tombe à gauche-centre → poids $w[1,0] = 1$
> - **Patch 3** (fenêtre décalée en bas) : $X[1,1]$ tombe en haut-centre → poids $w[0,1] = 0$
> - **Patch 4** (fenêtre décalée en bas-droite) : $X[1,1]$ tombe en haut-gauche → poids $w[0,0] = 1$
>
> Avec $\frac{\partial \mathcal L}{\partial Y}=1$ partout : $\frac{\partial \mathcal L}{\partial X[1,1]} = 1\times0 + 1\times1 + 1\times0 + 1\times1 = 2$. Même résultat que par le "repliement + somme" ci-dessus — la convolution transposée n'est qu'une façon efficace de calculer cette somme pour tous les pixels d'un coup, plutôt que pixel par pixel à la main.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/backprop_miroir.png|500]]

C'est exactement la définition d'une **convolution transposée** avec le même filtre $W$ (section IV) : étaler chaque valeur de sortie sur une zone de l'entrée via $W$, en sommant les zones qui se chevauchent. Donc à chaque couche de convolution, il y a une convolution transposée cachée dans le backward pass — même dans un simple classifieur, sans segmentation ni autoencodeur.

## III - Hyperparamètres des filtres

### A. Remplissage (Padding)

Le padding se justifie pour deux raisons :
- la sortie rétrécit à chaque convolution
- on perd trop d'information sur les bords de l'image

**Solution** : ajouter une bordure de zéros autour de l'image, de largeur $p$ :

$$
\boxed{p = \text{padding}}
$$

Exemple : avec padding, une image d'entrée $8\times8$ donne une sortie $6\times6$.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im8 (2).png]]

Dimension de sortie avec padding :

$$
(n+2p-f+1) \times (n+2p-f+1)
$$

**Deux types de padding :**
- **"Valid"** : pas de padding
$$
\begin{aligned}
&n\times n \;\star\; f\times f \;\longrightarrow\; (n-f+1)\times(n-f+1) \\
&6\times6 \;\star\; 3\times3 \;\longrightarrow\; 4\times4
\end{aligned}
$$
- **"Same"** : on choisit $p$ pour que la sortie ait la même taille que l'entrée. On veut $n+2p-f+1=n \iff p=\frac{f-1}{2}$. Par exemple, pour un filtre $3\times3$, il faut $p=\frac{3-1}{2}=1$.

### B. Le pas (Stride)

Nouveau paramètre, le **stride** $s$ : le pas de déplacement du filtre (ici, 2 cases à la fois plutôt qu'une) :

$$
\boxed{\text{stride}=s}
$$

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im9 (3).png]]

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im10 (3).png]]

**Cas général** : image $n\times n$, filtre $f\times f$, padding $p$, stride $s$ → dimension de sortie :

$$
\left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor \times \left\lfloor \frac{n+2p-f}{s}+1 \right\rfloor
$$

### C. Nombre de filtres (Number of filters)

Combien de détecteurs différents on apprend à cette couche. Chaque filtre produit son propre canal de sortie — avec $C_{out}$ filtres, la sortie a $C_{out}$ canaux, chacun détectant un motif différent (un bord vertical, un bord horizontal, une texture...). C'est ce paramètre qui contrôle la **profondeur** du volume de sortie (cf. [[01_Filtres classiques#Décor : une image, c'est une fonction]] pour la distinction canal / dimension spatiale).

![[Pasted image 20260725200315.png]]
*Input $6\times6\times3$ (RGB). On choisit ici 2 filtres (Filter 1, Filter 2), chacun de taille $3\times3\times3$ : chaque filtre couvre **toute la profondeur** de l'entrée (les 3 canaux), pas juste une tranche. Chaque filtre produit une seule tranche 2D de sortie ($4\times4$), en sommant sa convolution sur les 3 canaux. Avec 2 filtres, on empile les 2 tranches obtenues → sortie $4\times4\times2$. Le nombre de filtres est un choix libre (1, 2, 5, 64...) : chaque filtre en plus ajoute un détecteur de motif différent (et un canal de sortie de plus), au prix de plus de paramètres à apprendre.*

### D. Dilatation (Dilation)

Au lieu d'un patch compact $K\times K$ (tous les pixels voisins collés), on espace les éléments du noyau d'un pas $d$ (le "taux de dilatation") — le noyau couvre alors une zone plus large de l'image **sans ajouter de poids** (toujours $K\times K$ paramètres), juste en sautant des pixels entre chaque élément du filtre. Utile pour agrandir le champ réceptif rapidement, sans augmenter le coût de calcul ni empiler plus de couches.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/dilatation_X_pattern.png|500]]
*Filtre normal $3\times3$ (9 poids, empreinte compacte $3\times3$, tous les pixels marqués X comptent) vs filtre dilaté $3\times3$, $d=2$ (mêmes 9 poids, mais étalés sur une empreinte $5\times5$ — seuls les pixels X participent au calcul, les positions "." sont totalement absentes, pas juste multipliées par zéro).*

**Exemple numérique**, sur un patch $5\times5$ avec un filtre $w=\begin{pmatrix}1&0&-1\\0&1&0\\-1&0&1\end{pmatrix}$ :

$$
I = \begin{pmatrix}
10&11&12&13&14\\
15&16&17&18&19\\
20&21&22&23&24\\
25&26&27&28&29\\
30&31&32&33&34
\end{pmatrix}
$$

Filtre normal (patch compact $3\times3$, coin haut-gauche) :
$$
1{\times}10 + 0{\times}11 + (-1){\times}12 + 0{\times}15 + 1{\times}16 + 0{\times}17 + (-1){\times}20 + 0{\times}21 + 1{\times}22 = 16
$$

Filtre dilaté ($d=2$, seulement les positions X : lignes/colonnes $0,2,4$) :
$$
1{\times}10 + 0{\times}12 + (-1){\times}14 + 0{\times}20 + 1{\times}22 + 0{\times}24 + (-1){\times}30 + 0{\times}32 + 1{\times}34 = 22
$$

Mêmes poids, mais le filtre dilaté lit des valeurs prises sur une empreinte $5\times5$ au lieu de $3\times3$.

> [!note]- D'où vient cette idée ?
> Le terme "dilated convolution" est popularisé par **Yu & Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions" (ICLR 2016)**, pour la segmentation sémantique — agréger du contexte à plusieurs échelles sans perdre en résolution. L'idée elle-même est bien plus ancienne : elle vient du traitement du signal, l'**"algorithme à trous"** pour le calcul efficace de la transformée en ondelettes non-décimée (Holschneider et al., 1989-1990) — près de 40 ans avant sa popularisation en deep learning.

**![[Pasted image 20260725191343.png|408]]**

## IV - Pooling et convolutions transposées

Trois briques, pas interchangeables — elles ne vont pas toutes dans le même sens :

| | Convolution (section II) | Pooling | Convolution transposée |
|---|---|---|---|
| **Rôle** | extraire des features | sous-échantillonner (compresser) | suréchantillonner (reconstruire) |
| **Poids appris ?** | oui | non (opération fixe : max/moyenne) | oui |
| **Effet sur la taille spatiale** | réduit ou garde (selon padding) | réduit toujours | **agrandit** toujours |
| **Utilisée pour** | toute architecture | classification (LeNet, AlexNet, VGG...) | segmentation, génératif (GAN), décodeur d'autoencodeur |

Conv et pooling vont dans le même sens (rétrécir l'image au fil des couches — image → petit vecteur de classe). La convolution transposée fait l'inverse : repartir d'une représentation compressée pour reconstruire une sortie de pleine taille. On ne la voit donc jamais dans un réseau de classification pure, seulement dans les architectures qui doivent reconstruire une sortie spatiale.

### Pooling

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im13.png]]

### Convolutions transposées

Exemple : on veut suréchantillonner une entrée $2\times2$ en une sortie $3\times3$. La différence clé : quand deux fenêtres se chevauchent, on **additionne** leurs produits.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im18.png]]

Problème : le pixel central est visité quatre fois par toutes les fenêtres, contrairement aux autres — c'est l'effet "damier" (checkerboard), qui apparaît parce que certains pixels sont influencés bien plus fortement que d'autres. Alternative proposée par certains : une convolution suivie d'une couche d'upsampling, plutôt qu'une convolution transposée.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im19.png|339]]

## V - Architectures classiques

### LeNet-5

Objectif : reconnaissance de chiffres manuscrits, entraîné sur des images en niveaux de gris.

![[im1 (5).png]]

Six filtres, stride 1, pas de padding. Puis pooling (average pooling), puis 16 filtres, puis des couches fully-connected, jusqu'à une prédiction $\hat y$ sur 10 classes (chiffres) — sans softmax à l'époque.

Le nombre de canaux augmente progressivement pendant que les dimensions spatiales diminuent : CONV → pool → conv → pool → fc → fc → output.

> [!note]- Détails historiques
> Une non-linéarité était appliquée après l'average pooling (plus utilisé aujourd'hui) ; sigmoid/tanh étaient utilisés plutôt que ReLU.

### AlexNet (2012)

Papier écrit par Alex Krizhevsky. Entrée à 3 canaux (le papier mentionne 224×224, techniquement incorrect). Stride large donc les dimensions rétrécissent rapidement, max pooling, couche fully-connected finale à 4096, puis softmax.

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im2 (4).png]]

- Beaucoup de similarités avec LeNet, mais bien plus grand : 160M de paramètres (LeNet en a beaucoup moins)
- Utilise ReLU
- À l'époque les GPU étaient plus lents : le calcul était réparti sur plusieurs GPU, les couches étaient scindées
- Utilisait aussi une Local Response Normalization, peu utilisée aujourd'hui

### VGG-16

Le "16" fait référence à ses 16 couches.

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im3 (5).png]]

Réseau plus petit, avec des hyperparamètres identiques pour toutes les convolutions et max poolings — deux convolutions d'affilée avant chaque pooling. 138M de paramètres, ce qui reste large même aujourd'hui.

### Highway Network (précurseur des ResNets)

Un Highway Network ressemble beaucoup à un réseau feed-forward classique. Rappel : pour un FFNN standard, une entrée $y$ donne :

$$
z = g(Wy + b)
$$

Dans un Highway Network, seule une **fraction** de l'entrée passe par cette transformation. Le reste est autorisé à traverser le réseau **sans être transformé**. Le ratio entre ces deux fractions est contrôlé par $t$ (la *transform gate*, porte de transformation) et $(1-t)$ (la *carry gate*, porte de transport). $t$ est calculé via une sigmoïde :

$$
z = t \odot g(W_H y + b_H) + (1-t) \odot y
$$

- $W_H, b_H$ sont les paramètres de la transformation affine standard.
- $t = \sigma(W_T y + b_T)$ est la porte de transformation.
- $1-t$ est la porte de transport (ce qui n'est pas transformé, transporté tel quel).

![[highway1.png|309]]

En sortie, la fraction transformée de l'entrée est sommée à sa fraction non-transformée.

> [!note]- Exemple d'origine (NLP)
> Le Highway Network sert par exemple à pondérer la contribution relative d'un embedding de mot (GloVe) et d'un embedding de caractère (1D-CNN). Sur un mot hors-vocabulaire (OOV) comme "misunderestimate", on veut augmenter le poids de la représentation par caractères (le GloVe risque d'être du bruit aléatoire). Sur un mot courant comme "table", une contribution plus équilibrée entre les deux est préférable.

**Lien avec ResNet.** C'est le précurseur direct des réseaux résiduels ci-dessous (Srivastava et al., mai 2015, quelques mois avant ResNet). L'idée de "laisser passer une partie de l'information sans la transformer" est la même que la skip connection d'un ResNet — sauf qu'ici le ratio est **appris et dépend de l'entrée** (porte $t$), alors qu'un ResNet simplifie à l'extrême en fixant ce ratio à 1 partout (pas de porte, l'identité est simplement additionnée).

### Réseaux résiduels (ResNet)

**Vue d'ensemble.** Les réseaux très profonds sont difficiles à entraîner (vanishing/exploding gradient). Les ResNets permettent d'entraîner des réseaux très profonds, construits à partir de **blocs résiduels**. Exemple à deux couches :

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im4 (2).png]]

Où :

$$
z_{l+1} = W_{l+1} a_l + b_{l+1} \quad a_{l+1} = g(z_{l+1}) \quad z_{l+2} = W_{l+2}a_{l+1} + b_{l+2} \quad a_{l+2} = g(z_{l+2})
$$

Deux chemins :
(i) **Chemin principal** : la façon standard de faire circuler l'information.
(ii) **Raccourci / skip connection** : on saute des connexions, juste avant la ReLU finale. Mathématiquement, il suffit de réécrire une équation :
$$
a_{l+2} = g(z_{l+2} + a_l)
$$

**ResNet vu comme un graphe** :

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im5 (1).png]]

**Pourquoi les ResNets sont intéressants ?** Dans un réseau "plain" (sans skip connections), l'erreur augmente en pratique après trop de couches (même si en théorie ça ne devrait pas). À l'inverse, l'erreur d'entraînement d'un ResNet continue de diminuer :

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im7 (2).png]]

**Pourquoi les ResNets fonctionnent.** Comparons deux réseaux :

![[im8 (3).png]]

Équations de la skip connection :

$$
\begin{aligned}
a_{l+2} &= g(z_{l+2} + a_l) \\
&= g(W_{l+2} a_{l+1} + b_{l+2} + a_l)
\end{aligned}
$$

Avec une régularisation $L2$ sur $W,b$, les valeurs se contractent ; si $W_{l+2}=0, b_{l+2}=0$ :

$$
\begin{aligned}
a_{l+2} &= g(z_{l+2} + a_l) \\
&= g(0 + a_l) = g(a_l) = a_l
\end{aligned}
$$

(en supposant $g$ = ReLU et $a\ge0$, donc $g(a_l)=a_l$).

Ça montre que la fonction identité est facile à apprendre pour un bloc résiduel : ajouter la skip connection peut ne rien changer ($a_{l+2}=a_l$), donc dans le pire cas ça ne nuit pas à la performance ; si le bloc apprend quelque chose d'utile, tant mieux.

> [!note]- Détail important
> Ce modèle utilise une convolution "same", car on veut parfois $a_l = a_{l+2}$, donc les deux activations doivent avoir la même dimension. Sinon, on écrit $a_{l+2} = W_s a_l$.

**ResNet sur une vraie image** : exemple de "Plain Network" avec beaucoup de convolutions $3\times3$ "same" à padding identique.

![[im6 (3).png]]

### Inception Network

**Network in Network et convolutions $1\times1$.** Basé sur le papier [Network in Network](https://arxiv.org/abs/1312.4400) (Lin et al., déc. 2013). Deux noms pour la même technique : convolutions $1\times1$ ou "Network in Network".

![[im10 (4).png]]

Exemple :
- pour réduire le nombre de canaux, on peut utiliser 32 filtres $1\times1\times192$
- on peut aussi appliquer 192 filtres $1\times1\times192$, ce qui garde la même taille d'image mais ajoute une non-linéarité

![[im9 (4).png|418]]

**Motivation de l'Inception.** Plutôt que de choisir une taille de filtre, un pooling ou une couche convolutive, on applique un $1\times1$, puis un $3\times3$, et on empile les résultats — on répète. L'idée : on propose plein de combinaisons différentes et le modèle apprend lui-même quel module est le plus intéressant :

![[im11 (2).png]]

**Inconvénient : le coût de calcul.** Concentrons-nous sur le module $5\times5$ :

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im12 (1).png|420]]

Pour produire une sortie $28\times28\times32$ avec 32 filtres $5\times5\times192$ chacun : $28\times28\times32\times(5\times5\times192) = 120$ millions d'opérations.

**Solution** : appliquer d'abord une convolution $1\times1$, puis une convolution $5\times5$ :

![[im13 (1).png]]

Coût : 16 filtres $1\times1$ → $28\times28\times16\times192=2{,}4M$. Puis la deuxième couche : $28\times28\times32\times(5\times5\times16)=10{,}0M$. Total : $12{,}4M$ multiplications — une réduction massive par rapport aux 120M initiaux.

**Le réseau Inception complet**, avec des convolutions $1\times1$ pour réduire la dimension :

![[im14 (1).png]]

## VI - Transfer Learning

**(i) Peu de données d'entraînement.** Exemple : construire un détecteur de chat entre Tigre, Mitsy ou aucun des deux. On télécharge une implémentation de réseau (avec ses poids), on retire le softmax existant et on le remplace par le sien (3 classes). On gèle les paramètres du réseau et on n'entraîne que les poids du nouveau softmax.

![[im15 (2).png]]

**(ii) Quantité de données moyenne.** On peut geler les premières couches et entraîner les dernières, ou même remplacer complètement les dernières couches.

![[im16 (1).png]]

**(iii) Beaucoup de données.** On peut utiliser le réseau et les poids open source comme simple initialisation, puis entraîner tout le réseau (remplace l'initialisation aléatoire).

![[im17 (1).png]]


## VII — Pourquoi les CNN fonctionnent : symétries et hiérarchie (Mallat)

> Cette section développe le point 3 du programme de Mallat esquissé dans [[00_Perceptron Multi-Couches#II.7 — Pourquoi les réseaux marchent : géométrie et symétries (Mallat)]]. L'argument central : les CNN échappent à la malédiction de la dimension parce qu'ils **encodent explicitement les symétries** des données visuelles dans leur architecture.

**Symétries et réduction de dimension.** Une symétrie de $f$ c'est une transformation $g$ telle que $f(g \cdot x) = f(x)$ même si $g \cdot x$ est loin de $x$ — une régularité **globale**, contrairement à Lipschitz qui est locale. Si $f$ possède un groupe de symétrie $G$, l'espace d'étude se réduit de $\Omega$ à $\Omega/G$. Plus $G$ est grand, plus la réduction est massive — et donc plus on s'affranchit de la borne $n \geq \epsilon^{-d}$.

**Symétries des images.** Une image $x$ est un champ de pixels $x(u)$ où $u$ indexe la position 2D. Deux symétries naturelles :

- **Translation globale** : $x(u) \to x(u-g)$ — décaler l'image ne change pas la nature de l'objet. C'est exactement ce qu'encodent les convolutions via le partage de poids : un même filtre appliqué à toutes les positions.
- **Déformation locale** : $x(u) \to x(u - g(u))$ — déformer l'image (rotation locale, étirement) ne change pas non plus la nature de l'objet. $g$ est maintenant une *fonction*, pas un scalaire — le groupe $G$ devient de très grande dimension, d'où une réduction massive de la dimension effective.

![[déformation.png|300]]
**Figure.** *Déformations du chiffre 3 — chaque version est une transformation $x(u) \to x(u - g(u))$ avec un $g$ différent. La fonction $f$ (« c'est un 3 ») reste invariante.*

![[déformation_2.png|500]]
**Figure.** *Types de déformations : translation, rotation, distorsion locale (difféomorphisme). L'invariance par translation est encodée par les convolutions ; les déformations locales sont gérées par la data augmentation et le pooling.*

> [!note]- Invariance vs équivariance
> Les convolutions sont **équivariantes** par translation : si l'image est décalée, la feature map l'est aussi de la même façon. L'**invariance** (la prédiction finale ne dépend pas de la position) n'arrive qu'après le pooling global. La distinction est importante pour comprendre ce que chaque couche apprend.

**Ce que le réseau apprend.** Un CNN opère via une représentation $\Phi_w$ apprise, puis un classifieur linéaire :

$x \to \Phi_w(x) \to f_w(x) = \sigma(\langle \omega, \Phi_w(x) \rangle)$

Si le problème possède la symétrie $g$ et que le réseau l'a apprise, alors $\Phi_w(g \cdot x) = \Phi_w(x)$ et donc $f_w(g \cdot x) = f_w(x)$. Expérimentalement : deux CNN entraînés avec des initialisations différentes ont des poids individuels très différents, mais des représentations $\Phi_w$ fonctionnellement équivalentes. **Ce ne sont pas les poids qui comptent, ce sont les invariants qu'ils encodent.**

**Hiérarchie multi-échelles — de $d$ à $O(\log d)$.** Les pixels d'une image interagissent principalement avec leurs voisins proches, mais les interactions à grande distance comptent aussi (en somme, elles sont du même ordre que les interactions locales à cause du grand nombre). La solution : **moyenner progressivement** à mesure qu'on s'éloigne. On passe d'un problème de dimension $d$ (tous les pixels) à $O(\log d)$ par hiérarchisation — ce qui **casse la malédiction de la dimension**.

C'est exactement ce que fait l'architecture CNN couche par couche :

| Couche | Ce qu'elle voit | Dimension de représentation |
|---|---|---|
| Couche 1 | Pixels voisins (3×3) | Bords, orientations |
| Couche 2 | Régions moyennes | Textures, motifs |
| Couche 3+ | Grandes plages | Parties d'objets |
| Couche finale | Image entière | Objet |

Chaque couche agrège une zone spatiale de plus en plus large, tout en réduisant la dimension de la représentation. L'outil mathématique qui formalise cette hiérarchie c'est la **théorie des ondelettes** (Mallat, 1980-90) — précurseur direct des CNN modernes.

> [!note]- Bilan : pourquoi les CNN échappent à la borne $\epsilon^{-d}$
> La borne de Mallat suppose Lipschitz seule. Les CNN échappent à cette borne parce qu'ils exploitent trois propriétés structurelles que Lipschitz n'encode pas :
> 1. **Symétries globales** (translation via partage de poids) — réduisent la dimension via $\Omega/G$
> 2. **Compositionnalité** (pixels → bords → formes → objets) — chaque niveau simple, complexité par composition
> 3. **Hiérarchie multi-échelles** (pooling progressif) — réduit $d$ à $O(\log d)$
> Un MLP générique sur des images n'encode aucune de ces trois propriétés — même sur des données qui les ont, il reste dans le pire cas de la malédiction.

---

## Annexe : Ce qu'apprend un CNN

*(Volontairement à la fin — on voit d'abord l'architecture, l'analyse de ce qu'un CNN apprend vient logiquement après.)*

On entend souvent dire qu'un réseau de neurones est une **boîte noire** : on lui donne une image, il donne une prédiction, et on ne comprendrait rien à ce qui se passe entre les deux. C'est faux, et cette annexe le montre concrètement sur un vrai réseau entraîné (AlexNet, pré-entraîné sur ImageNet) — pas une simulation.

### 1. Couche 1 : des filtres qu'on reconnaît

La première couche de convolution d'AlexNet contient 64 filtres appris, chacun de taille $11\times11\times3$ (3 canaux car l'image d'entrée est en couleur RGB). Chaque filtre est un **template** au sens de la partie III de [[01_Filtres classiques]] : on le fait glisser sur toute l'image, et à chaque position il calcule un score de corrélation croisée avec le patch local. L'ensemble de ces scores forme la **carte d'activation** — exactement la même logique que la carte $N_{tf}[i,j]$ construite avec le Roi de Pique.

![[filtres_cnn_annexe_input.png|207]]

Voici deux filtres **achromatiques** (les 3 canaux ont presque le même motif — le filtre réagit à la luminosité, pas à la couleur) : ce sont des détecteurs de bord orienté, cousins directs du Sobel construit à la main.

![[filtres_cnn_annexe_filtre60.png|411]]
![[filtres_cnn_annexe_filtre5.png|404]]

Sur leurs cartes d'activation, on voit nettement les contours du visage, du casque et de la navette ressortir — le réseau a **retrouvé tout seul**, par descente de gradient sur des millions d'images, un filtre qui fait à peu près la même chose que le Sobel qu'on a conçu à la main.

Certains filtres, en revanche, sont fortement **chromatiques** : leurs 3 canaux sont très différents entre eux, ils encodent une opposition de couleur (rouge-vert ou bleu-jaune) plutôt qu'un contraste de luminosité.

![[filtres_cnn_annexe_filtre2.png|431]]

> [!note]- Le lien avec la biologie
> Ce n'est pas un hasard si le réseau réapprend deux familles distinctes — détecteurs de bord (luminosité) et détecteurs d'opposition de couleur. Ce sont exactement les deux grandes familles de champs récepteurs trouvées dans la rétine et le cortex visuel primaire (V1) chez l'humain (travaux historiques de Hubel & Wiesel sur les cellules à orientation, et sur les cellules à opposition de couleur dans la voie rétino-géniculée). Un réseau entraîné uniquement à classer des images redécouvre, sans qu'on le lui impose, une organisation proche de celle du système visuel biologique.

### 2. Couches suivantes : de moins en moins une image

À partir de la 2ᵉ couche de convolution, un filtre n'a plus la forme $(3, k, k)$ mais $(C_{\text{in}}, k, k)$ où $C_{\text{in}}$ est le nombre de canaux produits par la couche précédente (64, puis 192, 384...). Un filtre de conv2 ne regarde donc plus des pixels bruts : il fait une **combinaison pondérée** de 64 cartes de détecteurs déjà passées par une non-linéarité (ReLU), sur une fenêtre spatiale. On ne peut plus l'afficher comme une petite image reconnaissable — un tenseur $(64, 5, 5)$ n'a pas de représentation visuelle directe. On ne peut plus voir "ce qu'il détecte" que via sa carte d'activation.

![[filtres_cnn_profondeur.png|493]]

Deux choses se passent en même temps à mesure qu'on avance dans les couches :

- la **résolution spatiale** de la carte diminue (pooling, stride) ;
- le **champ réceptif** de chaque case augmente — chaque case "voit" une zone de plus en plus grande de l'image d'origine, donc peut encoder un motif de plus en plus composite.

Les cartes deviennent visuellement de moins en moins interprétables comme des "bords" et de plus en plus comme des taches/formes abstraites, localisées sur des zones précises de l'image (visage, casque...). C'est exactement la hiérarchie décrite dans le tableau de la partie VII (Mallat) — Couche 1 : bords → Couche 2 : textures/motifs → Couches suivantes : parties d'objets — sauf qu'ici on le voit sur un vrai réseau entraîné plutôt que sur un argument théorique.

### 3. Vers la prédiction : de l'image à un vecteur

À la fin des couches de convolution/pooling, on obtient un dernier empilement de cartes (256 canaux × 6×6 pour AlexNet), qu'on **aplatit** (flatten) en un seul vecteur d'environ 9216 nombres. Ce vecteur devient l'entrée $x$ d'un MLP classique — exactement la formule $y = Wx + b$ retrouvée en partie II. Ce vecteur n'est plus une image au sens visuel : c'est une **représentation compressée** de ce que le réseau a détecté à toutes les échelles (souvent appelée *embedding* ou *représentation apprise* — le terme *espace latent* est plus spécifiquement réservé aux modèles génératifs, mais l'idée de compression abstraite est la même).

### 4. Pour aller plus loin

Ce qu'on vient de montrer ici tient en quelques images, mais c'est devenu un sous-domaine de recherche à part entière, l'**interprétabilité** (*interpretability*) :

- Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks* (2014) — technique du "deconvnet" pour remonter d'une activation vers les pixels d'origine.
- Bau et al., *Network Dissection: Quantifying Interpretability of Deep Visual Representations* (CVPR 2017) — associe automatiquement un concept humain à chaque canal et mesure combien de canaux sont réellement interprétables.
- Olah, Mordvintsev, Schubert, *Feature Visualization* (Distill, 2017) et *The Building Blocks of Interpretability* (Distill, 2018) — génèrent par optimisation l'image qui active le plus fort un neurone donné, plutôt que de chercher une image existante qui l'active.

*(Illustrations générées avec `torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)` — vrais poids entraînés, pas une simulation.)*
