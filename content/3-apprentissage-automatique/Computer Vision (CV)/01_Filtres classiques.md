---
title: Filtres classiques (Image Processing "old school")
description: Filtres linéaires (box, gaussien, Sobel, Laplacien) et non-linéaires (médian, bilatéral), avant les CNN
weight: 0
---

# Filtres classiques

Avant que les CNN n'apprennent les filtres automatiquement, le traitement d'image "à la main" reposait sur une poignée de filtres qu'on choisissait pour la tâche. Cette note couvre le zoo classique — c'est ce que [[03_CNN|les CNN redécouvrent tout seuls]] en première couche.

## Décor : une image, c'est une fonction

![[im1-1 (1) 1.png|225]]

Avant de parler de filtres, il faut poser ce que $f[i,j]$ (utilisé dans toute la note) représente vraiment. Une image en niveaux de gris est mathématiquement une **fonction de deux variables** :

$$
I : (i,j) \longmapsto \text{intensité}
$$

C'est un **champ scalaire 2D** : à chaque position $(i,j)$ du plan, on associe **un seul nombre** (l'intensité du pixel). Cette fonction admet deux visualisations équivalentes, du même objet (c'est ce que montre l'image ci-dessus) :

- **Vue "image"** : on encode l'intensité par une **couleur** (noir → blanc). C'est la représentation naturelle d'une photo — celle qu'on utilise partout dans cette note.
- **Vue "surface 3D"** : on encode l'intensité par une **altitude** $z=I(i,j)$. La fonction devient un relief.

Les deux montrent exactement le même objet. La vue 3D rend juste explicite une intuition utile pour la suite : un **bord** (transition brutale d'intensité) est une **falaise** — un endroit où l'altitude chute fortement sur une courte distance. Détecter un bord (Sobel, Laplacien, section I) revient à détecter ces falaises ; lisser (box, gaussien) revient à aplanir le relief.

> [!note]- Vocabulaire : champ scalaire vs champ vectoriel
> | Objet | Domaine | Sortie en chaque point | Type |
> |---|---|---|---|
> | Image grayscale | $\mathbb{R}^2$ | scalaire (intensité) | **champ scalaire 2D** |
> | Image RGB | $\mathbb{R}^2$ | vecteur 3D $(R,G,B)$ | **champ vectoriel 2D** (3 canaux) |
> | Gradient d'une image grayscale $\nabla I$ | $\mathbb{R}^2$ | vecteur 2D $(\partial_x I, \partial_y I)$ | **champ de gradient** (issu d'une dérivation) |
> | Scan IRM / volume | $\mathbb{R}^3$ | scalaire | **champ scalaire 3D** |
>
> Ne pas confondre **canal** (composante de la sortie vectorielle, ex. R,G,B) avec **dimension spatiale** (variable d'entrée). Une image RGB reste définie sur un domaine 2D ; les 3 canaux sont juste 3 fonctions scalaires empilées.

| Type de donnée | Variables d'entrée | Sortie par point | Conv utilisée |
|---|---|---|---|
| Signal 1D (audio, série temp.) | 1 | 1 | Conv1D |
| Image grayscale | 2 | 1 | Conv2D |
| Image RGB | 2 | 3 | Conv2D (3 canaux d'entrée) |
| Vidéo (RGB + temps) | 3 | 3 | Conv3D ou Conv2D+1D |
| Scan médical (volume) | 3 | 1 | Conv3D |

*Cette table anticipe sur [[03_CNN]] : tous nos filtres ici restent sur la ligne "image grayscale" (1 canal). Le passage à plusieurs canaux (RGB, volumes...) est justement l'un des points qui manque pour comprendre les CNN — cf. [[03_CNN#II - Volume, canaux, filtres appris]].*

## 0. Le point de départ : la convolution

Tout part d'une seule opération, la **convolution discrète 2D**, qui transforme une image d'entrée $f$ en une image de sortie $g$ à l'aide d'un noyau $h$ :

$$
\underbrace{g[i, j]}_{\text{image de sortie}}=\sum_{m} \sum_{n} \underbrace{f[m, n]}_{\text{image d'entrée}} \; \underbrace{h[i-m, j-n]}_{\text{noyau / filtre}}
$$

($h$ subit un double flip par construction de la formule — cf. [[#Annexe (à trier) : Convolution 1D]] pour le détail.)

![[Pasted image 20260725143136.png]]

Concrètement, pour un noyau $3\times3$, calculer un seul pixel de sortie $O(x,y)$ revient à superposer le noyau $K$ sur le voisinage $3\times3$ du pixel $(x,y)$ dans l'image d'entrée $I$, multiplier terme à terme, et sommer :

$$
O(x,y) = \sum_{i=0}^{2}\sum_{j=0}^{2} K(i,j) \times I(x-1+j,\, y-1+i)
$$

On répète ce calcul en glissant le noyau sur **tous** les pixels de l'image pour obtenir l'image de sortie complète — c'est le "glisser, multiplier, sommer" évoqué plus haut.

Cette formule ne dit rien sur *quel* $h$ utiliser — elle définit juste le mécanisme ("glisser, multiplier, sommer"). Tout le travail de conception d'un filtre "à la main" consiste à **choisir $h$** en fonction de ce qu'on veut obtenir en sortie : lisser l'image, détecter des bords, retrouver un motif...

C'est là que se sépare la note en trois parties, chacune répondant à une question différente sur la même mécanique "glisser, calculer, répéter" :

| Partie | Question | Réponse |
|---|---|---|
| **I - Filtres linéaires** | Comment *transformer* l'image ? | $h$ fixe, branché directement dans la formule (box, gaussien, Sobel, Laplacien) — une somme pondérée = une convolution |
| **II - Filtres non-linéaires** | Comment *nettoyer* l'image ? | La convolution ne suffit plus, il faut une opération algorithmique (trier, comparer) — médian, bilatéral |
| **III - Template Matching** | *Où* se trouve ce motif précis dans l'image ? | Corrélation croisée : pas de transformation, on cherche une correspondance |

## I - Filtres linéaires

> 💡 Pour comparer les filtres entre eux sans se perdre, on suit une **seule image** (le Cameraman, un classique du domaine) sur toute cette partie : box, gaussien, Sobel, Laplacien lui sont appliqués tour à tour.

Tous les filtres de cette partie sont de simples noyaux $h$ figés à la main. C'est exactement ce qu'un CNN apprend tout seul par descente de gradient : si on regarde ce qu'il y a réellement dans un kernel de conv une fois entraîné (surtout en première couche), on y retrouve typiquement des versions apprises de ces mêmes motifs — détecteurs de bords proches de Sobel, lissages proches du gaussien... (cf. [[03_CNN#Annexe : Ce qu'apprend un CNN]]). La différence n'est pas dans la nature de l'opération (toujours une somme pondérée $\sum f \cdot h$), juste dans **qui choisit les poids** : ici c'est nous à la main, dans un CNN c'est l'entraînement.

### 1. Box filter

Premier choix de $h$, le plus simple possible : un noyau rempli d'une **valeur constante** partout. Voyons ce que ça donne concrètement, étape par étape — sur l'image de référence suivante, reprise dans toute la suite de cette partie :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/00_original.png|301]]
*Image de référence (non filtrée).*

**Étape 1 — noyau brut, non normalisé.** Prenons un noyau $5\times5$ rempli de 1 partout :

$$
h_{5\times5} = \begin{pmatrix} 1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1 \end{pmatrix}
$$

Si $h$ vaut 1 partout, la convolution transforme chaque pixel de sortie en la *somme* de tous les pixels du voisinage — donc une valeur bien plus grande que l'intensité d'origine (pour ce noyau $5\times5$, une intensité de 255 peut monter jusqu'à $255 \times 25 = 6375$). Le résultat sature (l'image devient blanche/cramée) :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/01_box_non_normalise.png|301]]
*Box filter $5\times5$ non normalisé, appliqué à l'image de référence ci-dessus : la somme des pixels voisins fait saturer l'image.*

**Étape 2 — on normalise.** Correction simple : diviser $h$ par l'aire du noyau ($K^2$, donc $25$ ici), pour transformer la somme en moyenne :

$$
h_{5\times5}^{norm} = \frac{1}{25}\begin{pmatrix} 1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1\\1&1&1&1&1 \end{pmatrix}
$$

**Exemple numérique** (noyau $3\times3$ pour rester lisible) : un petit patch pris dans l'image, autour d'un pixel dont la valeur d'origine est $50$ :

$$
\underbrace{\begin{pmatrix} 40&42&46\\46&50&55\\52&58&60 \end{pmatrix}}_{\text{patch }3\times3\text{ de l'image}} \; * \; \underbrace{\frac{1}{9}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}}_{\text{noyau box normalisé}} \;\Rightarrow\; O(x,y) = \frac{40+42+46+46+50+55+52+58+60}{9} \approx 49{,}9
$$

Le pixel de sortie devient simplement **la moyenne des 9 valeurs du patch** — proche de la valeur d'origine ($50$) si le voisinage est déjà homogène, mais tiré vers la moyenne locale sinon : c'est exactement ce lissage, répété sur toute l'image, qui produit le résultat flouté ci-dessous.

> [!note]- La taille du noyau est un hyperparamètre
> $K$ (ici $3$ ou $5$) — la taille du carré qu'on fait glisser sur l'image — n'est pas fixée par la théorie : c'est un choix, un **hyperparamètre**, comme $\sigma$ pour le gaussien (section suivante). Plus $K$ est grand, plus le lissage est fort, mais plus le calcul coûte cher (cf. la discussion de coût dans la section gaussienne). Le terme est le même qu'en CNN (cf. [[03_CNN]]), où la taille du noyau est aussi un hyperparamètre à choisir.

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/02_box_normalise.png|301]]
*Box filter $5\times5$ normalisé, appliqué à l'image de référence : lissage propre, l'image est floutée sans saturer.*

**Étape 3 — le défaut caché.** Le noyau box est un **carré plein** (valeur constante à l'intérieur, zéro dehors) — et cette forme géométrique se retrouve telle quelle dans le résultat. Sur des points isolés, ça saute aux yeux :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/07_box_vs_gaussien_points.png|500]]
*Input : 5 points isolés sur fond noir. Box filter (31×31) : chaque point devient un carré à bords nets. Gaussien (σ=8) : chaque point devient un halo rond qui s'estompe en douceur.*

Le halo box est carré parce que le noyau lui-même est carré (une simple moyenne, uniforme dans un carré, zéro dehors). Le halo gaussien est rond parce que la formule du gaussien ne dépend que de la *distance* au centre ($i^2+j^2$), jamais de la direction — elle est **isotrope** par construction.

**Pourquoi c'est un vrai défaut, pas juste esthétique :** c'est exactement le problème derrière l'effet "bokeh carré" en photo/rendu 3D — un flou appliqué sur un point lumineux hors-focus (lampadaire flou en arrière-plan, reflet...) donne un halo carré avec un filtre box, artificiel et facilement reconnaissable comme un défaut numérique, alors qu'un flou gaussien donne un halo rond, perçu comme naturel par l'œil (c'est pour ça que "Gaussian Blur" est l'option par défaut dans les logiciels de retouche, et que "Box Blur" est signalé comme donnant un rendu "carré"/dur). C'est exactement le filtre **gaussien**, détaillé ci-dessous.

### 2. Filtre gaussien

Au lieu du plateau à coupure nette du box filter, on prend le profil le plus "naturel" mathématiquement pour décroître en douceur : la gaussienne — valeur max au centre, qui s'atténue progressivement, sans jamais couper brutalement.

$$
n_{\sigma}[i, j]=\frac{1}{2 \pi \sigma^{2}} e^{-\frac{1}{2}\left(\frac{i^{2}+j^{2}}{\sigma^{2}}\right)}
$$

Toujours normalisé à 1 (intégrale = 1), quelle que soit la taille $\sigma$. Le seul paramètre à régler est $\sigma$ : plus il est grand, plus le lissage est fort — $\sigma$ contrôle littéralement *combien* on floute :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/09_gaussien_sigma4_vs_sigma16.png|500]]
*Gaussien à $\sigma=4$ vs $\sigma=16$ : plus $\sigma$ augmente, plus le lissage est fort (comparer à l'image de référence de la section précédente).*

**Exemple numérique** (noyau $3\times3$ approché, $\sigma\approx1$) : contrairement au box filter, les poids ne sont **plus uniformes** — ils décroissent avec la distance au centre :

$$
h_{3\times3}^{gauss} = \frac{1}{16}\begin{pmatrix}1&2&1\\2&4&2\\1&2&1\end{pmatrix}
$$

Sur le même patch que pour le box filter :

$$
\underbrace{\begin{pmatrix} 40&42&46\\46&50&55\\52&58&60 \end{pmatrix}}_{\text{patch }3\times3\text{ de l'image}} \; * \; \underbrace{\frac{1}{16}\begin{pmatrix}1&2&1\\2&4&2\\1&2&1\end{pmatrix}}_{\text{noyau gaussien}} \;\Rightarrow\; O(x,y) = \frac{40+2\times42+46+ ...+2\times58+60}{16} = \frac{800}{16} = 50
$$

À comparer avec le box filter sur le même patch : $\approx 49{,}9$ (moyenne uniforme). Ici le résultat tombe exactement sur $50$, la valeur d'origine du pixel central — parce que ce pixel central pèse $4/16 = 25\%$ du total, contre seulement $1/16 \approx 6\%$ pour chaque coin. C'est cette pondération décroissante avec la distance qui distingue le gaussien d'une simple moyenne : les voisins proches comptent beaucoup, les lointains comptent peu, jamais de cutoff brutal comme le box filter (section précédente).

**Et avec un $\sigma$ plus grand, sur le même noyau $3\times3$ ?** Les poids s'aplatissent :

$$
h_{3\times3}^{gauss,\, \sigma=4} \approx \begin{pmatrix}0{,}109&0{,}112&0{,}109\\0{,}112&0{,}116&0{,}112\\0{,}109&0{,}112&0{,}109\end{pmatrix} \Rightarrow O(x,y) \approx 49{,}89
$$

Quasiment identique au box filter ($49{,}9$) ! Un $\sigma$ grand sur un noyau resté petit ne "voit" que le sommet plat de la cloche, pas sa décroissance — d'où la règle $K \approx 6\sigma$ vue plus haut : il faut agrandir le noyau avec $\sigma$, sinon le gaussien perd son avantage et revient à un simple box filter (en plus cher à calculer).

> [!note]- Règle de taille de noyau
> Pour un noyau $K \times K$, on veut $K \approx 2\pi\sigma \approx 6\sigma$ (les deux formulations coïncident numériquement, $2\pi \approx 6{,}28$). En dessous, on tronque trop la gaussienne. Concrètement : plus $\sigma$ est grand, plus le noyau doit être large pour capturer toute la cloche gaussienne sans la couper (sinon on retombe dans le problème de coupure nette qu'on vient de résoudre).

![[im3-9.png|437]]
*Le noyau s'élargit avec $\sigma$ (et donc $K$) : la cloche s'étale mais garde toujours une décroissance douce, jamais de coupure brutale.*

**Propriété clé : séparabilité.** Une gaussienne 2D se factorise en deux gaussiennes 1D (une horizontale, une verticale) :

$$
g[i, j]=\underbrace{\frac{1}{\sqrt{2\pi}\sigma}\sum_{m} e^{-\frac{m^2}{2\sigma^2}} f[i-m, \cdot]}_{\text{passe 1D verticale}} \; * \; \underbrace{\frac{1}{\sqrt{2\pi}\sigma}\sum_{n} e^{-\frac{n^2}{2\sigma^2}}}_{\text{passe 1D horizontale}}
$$

Coût : $O(K^2)$ par pixel en 2D direct → $O(2K)$ en deux passes 1D. Gain massif pour les gros noyaux.

### 3. Sobel (dérivée première)

Changement d'objectif. Le gaussien n'est pas "insuffisant" — il fait très bien ce pour quoi il est fait : **lisser**, réduire le bruit. Mais lisser et **détecter un bord** sont deux tâches différentes, qui demandent de choisir un $h$ différent (cf. section 0 : tout est affaire de choisir le bon $h$ selon l'objectif).

Un bord, c'est un endroit où l'intensité change *brusquement*. Pour le détecter, on ne veut pas moyenner les pixels voisins (ça, c'est le boulot du gaussien) — on veut au contraire calculer une **différence** entre pixels voisins : grande différence = bord, différence quasi nulle = zone plate. C'est exactement ce que fait une dérivée.

**D'où sort concrètement le noyau ?** La dérivée d'une fonction $f$ en $x$, c'est juste la **pente** : de combien $f$ change quand $x$ augmente d'une petite quantité, $\frac{\text{variation de } f}{\text{variation de } x}$. Sur une image, $x$ est la position d'un pixel — un entier — donc la plus petite variation possible est $1$ pixel (le voisin immédiat). La version la plus simple : $f'(x) \approx f(x{+}1) - f(x)$. Une version plus stable, qui regarde des deux côtés au lieu d'un seul :

$$
f'(x) \approx f(x{+}1) - f(x{-}1)
$$

C'est **littéralement** le noyau $[-1,\ 0,\ 1]$ : pixel de gauche $\times(-1)$, centre $\times 0$ (ignoré), pixel de droite $\times(+1)$, on additionne.

**Et en 2D ?** Une image est une fonction de deux variables $f(x,y)$, donc on ne parle plus de "la" dérivée mais de **dérivées partielles** : $\frac{\partial f}{\partial x}$ (variation selon $x$, $y$ gelé) et $\frac{\partial f}{\partial y}$ (variation selon $y$, $x$ gelé). $\frac{\partial f}{\partial x}$ se calcule avec **exactement la même formule qu'au-dessus**, en trimballant $y$ sans y toucher :

$$
\frac{\partial f}{\partial x}(x,y) \approx f(x{+}1,y) - f(x{-}1,y)
$$

Ça donnerait un noyau d'**une seule ligne** $[-1,\ 0,\ 1]$ appliqué à la ligne $y$ uniquement. Le souci : une dérivée sur une seule ligne est fragile (un pixel de bruit fausse toute l'estimation). Sobel calcule donc cette même estimation sur **3 lignes voisines** ($y{-}1$, $y$, $y{+}1$ — trois estimations légèrement différentes de $\frac{\partial f}{\partial x}$ dans la zone), puis les combine par une **moyenne pondérée** ($1,2,1$ : la ligne centrale compte double) au lieu d'une simple moyenne. Le tout tient en une seule convolution $3\times3$ :

$$K_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix} \qquad K_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

$K_x$ estime $\frac{\partial f}{\partial x}$ (détecte les bords verticaux), $K_y$ estime $\frac{\partial f}{\partial y}$ (détecte les bords horizontaux) — ce sont deux filtres séparés, donc deux **images de sortie** différentes après convolution : $I * K_x$ et $I * K_y$ ont chacune une valeur à *chaque pixel* $(i,j)$, exactement comme $g[i,j]$ dans la formule de convolution de la section 0. Ce ne sont pas des scalaires uniques pour toute l'image, ce sont deux images complètes.


**Vérification numérique.** Sur un patch avec un dégradé horizontal (les 3 lignes identiques, donc aucune variation verticale) :

$$
\begin{pmatrix} 40&50&60\\40&50&60\\40&50&60 \end{pmatrix} \; * \; K_x \;=\; 80
$$

Comparé à la différence centrée seule, $f(x{+}1)-f(x{-}1) = 60-40=20$ : le résultat de Sobel ($80$) est exactement $4\times$ plus grand — $4$ étant la somme des poids de lissage vertical ($1+2+1$). Comme les 3 lignes sont identiques, le lissage vertical n'ajoute aucune information, juste ce facteur d'échelle : ce qui reste est bien la dérivée centrée qu'on vient de dériver plus haut, à un facteur constant près.

Problème concret : un bord diagonal (ou ni parfaitement vertical ni parfaitement horizontal) donne une réponse *partielle* dans les deux images à la fois, sans qu'aucune des deux ne dise clairement "il y a un bord ici, fort ou faible". Pour avoir une seule information exploitable par pixel, on combine les deux valeurs **à chaque position $(i,j)$ séparément** :

$$|\nabla I|[i,j] = \sqrt{(I * K_x)[i,j]^2 + (I * K_y)[i,j]^2}$$

Pour un pixel donné, le résultat est bien un scalaire (un seul nombre = "force du bord à cet endroit précis"). Mais comme on répète ce calcul à **chaque** pixel de l'image, l'ensemble des résultats forme à nouveau une image complète — une **carte** où chaque pixel encode sa propre force de bord, indépendamment de l'orientation. C'est cette carte (et non $K_x$ ou $K_y$ pris isolément) qu'on seuille ensuite pour décider "pixel de bord ou pas" — c'est elle qui sert de brique de base à Canny, à la détection de coins, etc.

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/10_sobel_kx_ky_vers_carte.png|600]]
*De gauche à droite : $I*K_x$ (bords verticaux — trépied, bâtiment), $I*K_y$ (bords horizontaux — épaules, sol), combinés pixel par pixel en la carte $|\nabla I|$.*

> [!note]- $|\nabla I|$ = norme du gradient, rien de plus
> $\nabla f = (f_x, f_y)$ est le vecteur gradient, et $|\nabla f| = \sqrt{f_x^2+f_y^2}$ en est simplement la norme — c'est aussi la valeur maximale que peut prendre la **dérivée directionnelle** $D_u f = \nabla f \cdot u$ (obtenue quand $u$ pointe dans la direction du gradient). Rien de spécifique à Sobel là-dedans : c'est du calcul multivarié standard. Ce qui est spécifique à Sobel, c'est uniquement **comment on estime $f_x$ et $f_y$** — pas la vraie dérivée continue (impossible sur des pixels discrets), mais l'approximation par différence centrée + lissage $1,2,1$ vue plus haut.

> [!note]- Pourquoi "Sobel" et pas juste "filtre dérivée première" ?
> Parce que "dérivée première approximée sur une grille discrète" est un problème générique qui a plusieurs solutions concurrentes, chacune avec un nom propre selon les poids de lissage choisis :
>
> | Nom | Dérivée | Lissage perpendiculaire |
> |---|---|---|
> | **Sobel** (1968) | $[-1,0,1]$ | $[1,2,1]$ |
> | **Prewitt** | $[-1,0,1]$ | $[1,1,1]$ (uniforme) |
> | **Scharr** | $[-1,0,1]$ | $[3,10,3]$ (meilleure symétrie rotationnelle) |
> | **Roberts Cross** | noyau $2\times2$ diagonal | aucun |
>
> Tous répondent à la même question, avec des poids de lissage différents — donc des compromis bruit/précision différents. "Sobel" désigne cette recette numérique précise, pas le concept générique.

> [!note]- Pourquoi 1, 2, 1 ?
> Sobel n'est pas une dérivée pure : le motif $1,2,1$ est un lissage gaussien grossier perpendiculaire à la direction dérivée. $K_x$ = lissage vertical $\otimes$ dérivée horizontale. C'est le même compromis "lisser avant de dériver" vu en 1D, intégré dans le noyau.

### 4. Laplacien (dérivée seconde)

Motivation : Sobel dit "il y a un bord dans le coin, et il est fort" — mais sa réponse est **étalée sur plusieurs pixels autour du vrai bord** (parce que la pente d'une transition d'intensité monte progressivement, atteint un maximum, puis redescend). Du coup, savoir exactement *quel pixel* est le bord reste flou : on a un pic large, pas un point précis.

Rappel de calcul : la pente (dérivée première) est maximale exactement là où **sa propre dérivée s'annule** — c'est la définition d'un maximum. Autrement dit, chercher le passage par zéro de la dérivée *seconde* localise le point exact où la pente est la plus forte, beaucoup plus précisément que chercher le sommet (large et flou) de la dérivée première. C'est tout l'intérêt du Laplacien : il ne détecte pas "où c'est fort", il détecte "où exactement" via ce zero-crossing.

**D'où sort le noyau ?** La dérivée seconde, c'est simplement "la dérivée de la dérivée". En discret, on calcule d'abord deux pentes décalées d'un demi-pixel de part et d'autre de $x$ :

$$
f'(x{+}\tfrac12) \approx f(x{+}1)-f(x) \qquad\qquad f'(x{-}\tfrac12) \approx f(x)-f(x{-}1)
$$

Puis on dérive à nouveau, c'est-à-dire qu'on prend la différence de ces deux pentes (même recette qu'avant, appliquée une deuxième fois) :

$$
f''(x) \approx f'(x{+}\tfrac12) - f'(x{-}\tfrac12) = \big[f(x{+}1)-f(x)\big] - \big[f(x)-f(x{-}1)\big] = f(x{+}1) - 2f(x) + f(x{-}1)
$$

Le noyau 1D est donc $[1,\ -2,\ 1]$. En 2D, le Laplacien additionne les deux dérivées secondes partielles, $\Delta f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$ (la courbure totale, selon $x$ **et** selon $y$) :

$$
\Delta f(x,y) \approx \underbrace{\big[f(x{+}1,y) - 2f(x,y) + f(x{-}1,y)\big]}_{\partial^2 f/\partial x^2} + \underbrace{\big[f(x,y{+}1) - 2f(x,y) + f(x,y{-}1)\big]}_{\partial^2 f/\partial y^2}
$$

En regroupant les termes, il ne reste que 5 voisins (les 4 axiaux, pas les coins) et $-4f(x,y)$ (deux fois $-2f(x,y)$, une fois par axe) — exactement le noyau :

$$K_L = \begin{pmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{pmatrix}$$

Ce noyau capte d'un coup la courbure selon $x$ **et** selon $y$ — contrairement à Sobel qui calcule $K_x$ et $K_y$ séparément puis les combine par une norme. C'est en ce sens que le Laplacien est **isotrope** : un seul masque suffit.

**Vérification numérique du zero-crossing.** Prenons un profil 1D qui traverse un bord (transition progressive $40\to50\to60$) : $f = [40,\ 40,\ 50,\ 60,\ 60]$, indices $x=0..4$. On calcule $f''(x) = f(x{+}1)-2f(x)+f(x{-}1)$ en chaque point intérieur :

$$
f''(1) = 40 - 2(40) + 50 = 10 \qquad f''(2) = 40-2(50)+60 = 0 \qquad f''(3) = 50-2(60)+60 = -10
$$

$f''$ passe de $+10$ à $0$ à $-10$ : il change de signe **exactement** au point $x=2$, le centre de la transition — c'est le zero-crossing qui localise précisément le bord, là où Sobel ($f'$) donnerait juste un pic large sans dire "c'est précisément ici".

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/11_zero_crossing.png|500]]
*De haut en bas : le profil d'intensité $f(x)$ à travers un bord, sa dérivée $f'(x)$ (Sobel — pic large, position floue), et sa dérivée seconde $f''(x)$ (Laplacien) qui passe par zéro exactement à la position du bord.*

| Filtre | Comportement sur un bord | Forme dans la sortie |
|---|---|---|
| **Sobel** ($f'$) | s'allume fort sur le bord | pic large à la position du bord |
| **Laplacien** ($f''$) | passe par zéro au milieu du bord | lobe positif puis négatif (zero-crossing) |

En pratique Sobel est préféré (plus robuste au bruit, donne la direction du gradient — chose que le Laplacien ne donne pas, cf. plus haut). Le Laplacien reste utile pour la localisation sub-pixel précise et la détection de blobs (LoG = Laplacien de gaussienne, cf. [[02_Detection_algorithms#2. Detecting Blobs]]).

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/06_laplacien.png|301]]
*Laplacien sur l'image de référence : contours fins en double liseré (positif/négatif de part et d'autre du zero-crossing), plus sensible au bruit que la carte Sobel ci-dessus.*

> [!note]- Autre exemple : Sobel vs Laplacien sur une forme simple
> Sur un disque clair, la différence est encore plus lisible : Sobel donne un anneau lumineux unique (isotrope), le Laplacien un double anneau (zero-crossing).
> ![[Pasted image 20260502220100.png]]

## II - Filtres non-linéaires

Un filtre non-linéaire ne peut pas s'écrire comme une convolution — c'est une opération algorithmique (tri, seuil conditionnel...). Un kernel de convolution reste condamné à n'être qu'une somme pondérée (cf. section 0), donc un CNN ne peut pas apprendre ce genre d'opération *dans une couche de conv elle-même*. Le non-linéaire y apparaît ailleurs : les **fonctions d'activation** (ReLU...) et surtout le **pooling** (max-pooling notamment, qui fait glisser une fenêtre et prend le max — même logique algorithmique que le médian ci-dessous, sans poids appris). Le bilatéral, lui, préfigure l'idée derrière les mécanismes d'**attention** modernes (pondérer selon la ressemblance du contenu, pas juste la position).

### 1. Filtre médian

**Deux familles de bruit.** Jusqu'ici, tous les filtres linéaires supposaient implicitement un bruit **additif gaussien** : chaque pixel reçoit une petite variation aléatoire (bruit de capteur), sans valeur extrême isolée — une moyenne locale marche bien contre ça, elle lisse la variance. Mais il existe une autre famille, le bruit **impulsionnel** ("poivre-et-sel") : quelques pixels au hasard sont remplacés par une valeur extrême (blanc ou noir pur), le reste de l'image reste intact.

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/12_bruit_gaussien_vs_impulsionnel.png|500]]
*En haut : bruit gaussien, petite variation partout. En bas : bruit impulsionnel, quelques valeurs extrêmes isolées (points noirs).*

**Pourquoi le gaussien échoue sur l'impulsionnel.** Une moyenne (pondérée ou non) est une opération **linéaire** : elle *dilue* un outlier proportionnellement à son poids, mais ne l'élimine jamais. Exemple numérique sur un patch dont un seul pixel est corrompu (valeur $255$ au milieu de pixels $\approx 43$) :

$$
\text{patch} = [42,\ 45,\ 255,\ 44,\ 43]
$$

$$
\text{moyenne (box)} = \frac{42+45+255+44+43}{5} = 85{,}8 \qquad \text{— tirée vers le haut, encore visiblement faussée}
$$

**Le médian : trier, garder la valeur du milieu.** Sur le même patch :

$$
\text{trié} = [42,\ 43,\ 44,\ 45,\ 255] \quad\Rightarrow\quad \text{médiane} = 44
$$

L'outlier finit tout seul en bout de liste triée — il n'est **jamais choisi**, aucune influence sur le résultat. C'est une opération de tri, pas une somme pondérée : c'est pour ça qu'elle ne peut pas s'écrire comme une convolution (cf. intro de la partie II).

**La formule, sans noyau $K$.** Contrairement au box/gaussien/Sobel/Laplacien, il n'y a plus de $K(i,j)$ à multiplier : on applique directement l'opérateur "médiane" à l'**ensemble** des valeurs du patch, sur une fenêtre $W$ (ex : $W=\{-1,0,1\}\times\{-1,0,1\}$ pour un $3\times3$) :

$$
O(x,y) = \underset{(i,j)\, \in\, W}{\text{median}} \; I(x+i,\, y+j)
$$

Sur le patch de la section box filter :

$$
I = \begin{pmatrix} 40&42&46\\46&50&55\\52&58&60 \end{pmatrix} \;\Rightarrow\; O(x,y) = \text{median}\{40,42,46,46,50,55,52,58,60\}
$$

$$
\text{trié} = [40,42,46,46,\mathbf{50},52,55,58,60] \quad\Rightarrow\quad O(x,y) = 50 \quad(\text{la } 5^e \text{ valeur sur } 9)
$$

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im4-1 (1).png|500]]
*Le lissage gaussien échoue sur du bruit impulsionnel : il dilue les points de bruit (ils deviennent flous) sans les supprimer.*

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im4-2.png|500]]
*Médian $K=3$ sur le même bruit : suppression propre, contrairement au gaussien ci-dessus.*

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im4-3.png|500]]
*Médian $K=7$ sur un bruit plus dense : échec cette fois. La médiane ne fonctionne que si **moins de 50% des pixels de la fenêtre sont corrompus** — sur un bruit assez dense, même une fenêtre $7\times7$ (49 pixels) peut dépasser ce seuil, et la "médiane" devient elle-même une valeur corrompue. Grossir $K$ n'est donc pas une solution universelle : ça dépend de la densité du bruit, pas seulement de sa présence.*

**Limite générale** (au-delà du bruit) : le médian reste aveugle au contenu de l'image — même comportement partout, peu importe le contexte du pixel (contrairement au bilatéral ci-dessous). Et comme tout filtre à fenêtre fixe, un $K$ trop grand finit par emporter des détails fins, pas seulement du bruit.

### 2. Filtre bilatéral

**Le problème du gaussien pur.** Il pondère uniquement par la distance spatiale, jamais par le contenu — donc s'il y a un bord juste à côté du pixel traité, le gaussien mélange quand même les deux côtés du bord entre eux, peu importe qu'ils soient très différents. Résultat : les bords se retrouvent flous, comme n'importe quelle autre zone.

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/13_gaussien_vs_bilateral.png|600]]
*Original / gaussien ($\sigma=4$, bords flous — le trépied "bave" dans le fond) / bilatéral (bords nets, zones plates lissées).*

**Idée du bilatéral :** garder le lissage gaussien spatial, mais **pondérer aussi par la ressemblance d'intensité** — un pixel de l'autre côté d'un bord (intensité très différente) compte peu, même s'il est spatialement proche.

$$
g[i, j]=\frac{1}{W_{sb}} \sum_{m} \sum_{n} f[m, n] \; \underbrace{n_{\sigma_{s}}[i-m, j-n]}_{\text{noyau spatial}} \; \underbrace{n_{\sigma_{b}}(f[m, n]-f[i, j])}_{\text{noyau d'intensité}}
$$

$1/W_{sb}$ normalise pour que la somme des poids reste égale à 1.

**La pipeline, en 4 étapes :**

1. On prend un patch autour du pixel $(x,y)$
2. On calcule un **poids spatial** $n_{\sigma_s}$ — exactement comme le gaussien classique, dépend seulement de la position du voisin (proche = poids fort)
3. On calcule, **séparément et indépendamment**, un **poids d'intensité** $n_{\sigma_b}$ — pour chaque voisin, à quel point sa valeur diffère de celle du pixel central : ressemblant → poids fort, très différent → poids quasi nul
4. On **multiplie** les deux poids terme à terme (le poids spatial n'est ni "faux" ni "corrigé" — c'est juste un critère parmi deux, combiné au second), on normalise, et cette combinaison sert de poids pour la moyenne pondérée

**Exemple sur un vrai point de l'image**, à la frontière du trépied (le point rouge ci-dessous) :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/14_bilateral_point_zoom.png|500]]
*Point étudié et patch $3\times3$ réel extrait à cet endroit.*

$$
I = \begin{pmatrix} 213&210&121\\213&\mathbf{150}&45\\188&48&44 \end{pmatrix}
$$

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/15_bilateral_pipeline.png|850]]
*Patch × poids spatial × poids d'intensité = poids combiné → moyenne pondérée. Les valeurs sombres ($45,48,44$, très différentes du centre $150$) ont un poids d'intensité $\approx 0$ malgré un poids spatial non-nul : elles sont éliminées du résultat.*

> [!note]- En une phrase
> On calcule deux questions séparées — "proche en **position** ?" (poids spatial) et "proche en **valeur** ?" (poids d'intensité) — on multiplie les deux réponses, et seuls les voisins qui répondent "oui" aux **deux** gardent un poids élevé dans la moyenne finale. Le poids d'intensité ne suit aucune règle géométrique (pas de préférence pour une diagonale ou une direction) : il dépend uniquement du contenu réel de l'image à cet endroit.

**Comparaison directe :** gaussien seul $\approx 137{,}4$ (tiré vers le bas par les valeurs sombres voisines) vs bilatéral $\approx 149{,}7$ (quasi inchangé, très proche de la vraie valeur locale $150$) — c'est exactement ça, "préserver les contours".

**Mais attention à ce que montre vraiment cet exemple.** Le centre ($150$, poids combiné $4$) pèse à lui seul $4/4{,}56 \approx 88\%$ du poids total — les autres voisins comptent très peu ($\approx 12\%$ cumulés). Autrement dit, à ce point précis (presque un coin isolé, la plupart des voisins étant de l'autre côté du bord), le bilatéral fait *très peu* de lissage — il se contente presque de garder la valeur du pixel telle quelle, ce qui est cohérent : tout près d'un bord fort, "presque ne rien lisser" est exactement le comportement voulu.

**Deuxième exemple, sur un dégradé doux (pas un coin dur)**, pour voir le cas où le bilatéral lisse vraiment :

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/16_bilateral_point2_zoom.png|500]]
*Point étudié sur un dégradé progressif (pas de bord franc).*

$$
I = \begin{pmatrix} 162&164&163\\206&\mathbf{207}&209\\220&222&222 \end{pmatrix}
$$

![[images/3-Apprentissage automatique/04_Computer vision/00_Filtres classiques/17_bilateral_pipeline_point2.png|850]]
*Ici le centre ne pèse plus que $35\%$ du poids total — plusieurs voisins (ligne du milieu et du bas, valeurs proches de $207$) contribuent vraiment, seule la ligne du haut (plus sombre, $\approx 163$) est fortement atténuée.*

Résultat : bilatéral $\approx 209{,}7$ vs gaussien $\approx 199{,}8$ — un écart net, cette fois dû à un vrai lissage partiel (plusieurs voisins comptent), pas juste "je garde ma valeur" comme dans le premier exemple. Les deux comportements sont corrects : près d'un bord dur, le bilatéral se fige (premier exemple) ; sur un dégradé doux, il lisse en tenant compte de qui ressemble à qui (second exemple).

## III - Template Matching (corrélation croisée)

Objectif : retrouver où un petit **template** $t$ apparaît exactement dans une image $f$ plus grande.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im4-9.png|311]]

*Exemple : template = juste la tête sur la carte de jeu, image = la carte entière. On cherche où ce template apparaît dans l'image.*

**Approche : minimiser la différence.** On fait glisser $t$ sur $f$ et on calcule la somme des différences au carré (SSD) à chaque position $(i,j)$ :

$$
\min ~~ E[i, j]=\sum_{m} \sum_{n}(f[m, n]-t[m-i, n-j])^{2}
$$

En développant le carré : $E = \sum f^2 - 2\sum ft + \sum t^2$. Minimiser $E$ revient à **maximiser** $\sum f \cdot t$ — c'est la **corrélation croisée** :

$$
\boxed{R_{tf}[i, j]=\sum_{m} \sum_{n} f[m, n] \, t[m-i, n-j] = t \otimes f}
$$

Répété à **chaque position** $(i,j)$ où on fait glisser le template, $R_{tf}$ forme une image complète — une **carte de corrélation**, exactement comme la carte $|\nabla I|$ de Sobel (section I) : une valeur par pixel, ici "à quel point le template ressemble à l'image ici", pas un scalaire isolé.

> [!note]- Dérivation complète (pourquoi minimiser $E$ = maximiser $R_{tf}$, et sa limite)
> En développant $E[i,j] = \sum(f-t)^2$ terme à terme :
> $$
> E[i,j] = \underbrace{\sum f[m,n]^2}_{\text{énergie du patch d'image}} \;-\; 2\underbrace{\sum f[m,n]\,t[m-i,n-j]}_{=\, R_{tf}[i,j]} \;+\; \underbrace{\sum t[m-i,n-j]^2}_{\text{énergie du template}}
> $$
> Le terme $\sum t^2$ est **vraiment constant** : le template ne change jamais, peu importe où on le fait glisser. Mais le terme $\sum f^2$ (énergie du patch d'image à cette position précise) **n'est pas constant** — il change selon où on regarde dans l'image. "Minimiser $E$ = maximiser $R$" n'est donc rigoureusement vrai **que si on suppose** $\sum f^2$ à peu près constant d'une position à l'autre — une approximation, pas un fait garanti.
>
> Exemple numérique : template $t=[3,5,2]$, patch $f_A=[3,5,2]$ (match parfait), patch $f_B=[6,10,4]$ (même forme, 2× plus fort en énergie) :
>
> | | $E$ (direct) | $R_{tf}$ | $\sum f^2$ |
> |---|---|---|---|
> | $f_A$ | $0$ | $38$ | $38$ |
> | $f_B$ | $38$ | $76$ | $152$ |
>
> $A$ est le vrai match ($E_A=0 < E_B=38$), mais $R_B > R_A$ ! La corrélation brute favorise $B$ à tort, précisément parce que $\sum f^2$ diffère entre les deux patchs — c'est exactement le problème repris juste en dessous (candidats A/B/C), et la raison d'être de la NCC plus bas.

> [!note]- Corrélation vs convolution
> Même structure que la convolution $g[i,j] = \sum f[m,n]\,t[i-m,j-n] = t*f$ (cf. [[#Convention ML vs convention mathématique]]), à un détail près : **pas de flip** du noyau dans la corrélation ($t[m-i,n-j]$ au lieu de $t[i-m,j-n]$). Idem que la distinction "convolution vraie" vs "corrélation croisée" utilisée en pratique par PyTorch/TF.

**Problème : la corrélation brute favorise les zones de forte énergie**, pas forcément la vraie réponse. Exemple 1D avec un signal $f$ et un template $t$ : trois candidats A (vraie réponse), B, C avec :

$$
R_{tf}(C) > R_{tf}(B) > R_{tf}(A)
$$

alors que A est la bonne réponse — B et C ont juste des valeurs plus grandes en amplitude, donc une corrélation brute plus élevée, sans ressembler davantage au template.

![[im4-10.png|455]]

**Solution : corrélation croisée normalisée (NCC).** On divise par l'énergie locale :

$$
N_{tf}[i, j]=\frac{\sum_{m} \sum_{n} f[m, n] \, t[m-i, n-j]}{\sqrt{\sum_{m} \sum_{n} f^{2}[m, n]} \; \sqrt{\sum_{m} \sum_{n} t^{2}[m-i, n-j]}}
$$

> [!note]- Attention à la fenêtre du dénominateur
> Les deux sommes du dénominateur doivent être **locales**, à la même position/fenêtre que la corrélation : $\sqrt{\sum f^2[m,n]}$ = énergie du **patch de l'image sous le template** à la position $(i,j)$ (pas l'image entière !), et $\sqrt{\sum t^2[m-i,n-j]}$ = énergie du **template entier** (constante). Si on normalisait par l'énergie de l'image entière, le résultat ne dépendrait plus de $(i,j)$ et la normalisation ne corrigerait rien.

Avec la version normalisée, l'ordre s'inverse et devient correct :

$$
N_{tf}(A) > N_{tf}(B) > N_{tf}(C)
$$

Sur une vraie image, le point le plus brillant de la carte de corrélation $N_{tf}$ correspond bien à la position du template :

![[im4-11.png|410]]
*$f$ (image, à gauche) passé dans l'opérateur de corrélation croisée $\otimes$ avec $t$ (le template) $=$ $N_{tf}[i,j]$, la carte des scores (à droite).*

**Pour relier tout ça, en mots simples — c'est exactement ce que montre l'image ci-dessus :**

1. On a $f$ (l'image de gauche) et $t$ (le template). On veut savoir *si* $t$ existe dans $f$, et *où*.
2. On fait glisser $t$ sur $f$, position par position ($\otimes$ sur le schéma). À **chaque** position, on calcule un seul nombre : $N_{tf}[i,j]$, le score de ressemblance à cet endroit.
3. Répété à toutes les positions, ça donne la carte de droite — une image à part entière, où chaque pixel encode "à quel point ça ressemble à $t$ ici" (clair = forte ressemblance, sombre = faible).
4. Le carré (le point le plus clair de la carte) marque le score maximum : sa position *est* la réponse à "où est le template".

## IV - Tableau récapitulatif

| Filtre | Type | Idée | Cas d'usage / limite |
|---|---|---|---|
| Box (normalisé) | linéaire | moyenne locale uniforme | simple mais artefacts axiaux |
| Gaussien | linéaire, séparable | moyenne pondérée isotrope | lisse bien, floute les bords |
| Sobel | linéaire | dérivée 1ère + lissage perpendiculaire | détection de bords, robuste au bruit |
| Laplacien | linéaire | dérivée 2nde isotrope | zero-crossing précis, sensible au bruit |
| Médian | non-linéaire | tri + valeur centrale | bruit poivre-et-sel, perd les détails fins |
| Bilatéral | non-linéaire | gaussien spatial × gaussien d'intensité | préserve les contours, plus coûteux |

## V - À retenir

- **Linéaire** = exprimable en convolution (box, gaussien, Sobel, Laplacien). **Non-linéaire** = algorithmique, pas de convolution possible (médian, bilatéral).
- Le gaussien bat le box filter car isotrope ; sa séparabilité le rend cheap à calculer.
- Sobel/Laplacien = dérivée 1ère/2nde généralisées en 2D pour détecter les bords.
- Le médian et le bilatéral existent pour corriger les défauts du lissage gaussien : perte des détails et non prise en compte du contexte local (contours).
- Template matching = corrélation croisée normalisée (NCC) ; la version brute est biaisée par l'énergie locale de l'image, d'où la nécessité de normaliser.
- Ces filtres ne sont pas obsolètes : ce sont exactement ceux que les [[03_CNN#Annexe : Ce qu'apprend un CNN|CNN redécouvrent automatiquement]] en première couche.

---

> [!warning] À ranger — hors-sujet CV, déplacé depuis 02_CNN.md
> Ce qui suit (convolution 1D) est générique au traitement du signal (audio, séries temporelles...), pas spécifique à la vision. Probablement à recaser dans un futur dossier "Signal 1D / Séries temporelles" (ex : ML Fundamentals). Laissé ici temporairement, non trié.

## Annexe (à trier) : Convolution 1D

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

| Filtre        | Calcule                   | Approxime      | Détecte                        |
| ------------- | ------------------------- | -------------- | ------------------------------ |
| $[-1, 1]$     | $f[i+1] - f[i]$           | $f'$ (ordre 1) | variation locale (asymétrique) |
| $[-1, 0, 1]$  | $f[i+1] - f[i-1]$         | $f'$ (ordre 2) | variation locale (symétrique)  |
| $[1, -2, 1]$  | $f[i-1] - 2f[i] + f[i+1]$ | $f''$          | concavité, points d'inflexion  |
| $[1, 1, 1]/3$ | moyenne des 3 voisins     | lissage        | atténuation du bruit           |

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

---

> [!warning] À trier — migré depuis 03_Detection_algorithms.md (section I)
> Ce qui suit prolonge directement Sobel/Laplacien (bords) et étend l'idée aux coins. Recouvrement probable avec les sections Sobel et Laplacien plus haut dans cette note — à dédupliquer plus tard. Canny et la détection de coins (Harris) sont eux du contenu réellement nouveau, pas encore couvert ailleurs. Notation d'origine non harmonisée avec le reste de la note ($I_x, I_y$ au lieu de $G_x, G_y$, etc.) — à uniformiser lors du tri.

## Annexe (à trier) : Edge & Corner Detection

### 1. What is an Edge?

**Définition.** Un edge (bord/contour) est un changement rapide d'intensité dans une petite région.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im1-1.png|300]]
*Photo d'une sculpture vs croquis fait par un artiste — l'artiste ne dessine que les traits marquants (highlights, ombrage).*

**Causes des edges dans le monde réel :**
- **Surface Normal Discontinuity** : deux surfaces du même matériau mais orientation différente → luminosité différente
- **Depth Discontinuity** : un objet devant un autre, matériaux différents → différence de luminosité
- **Surface Reflectance Discontinuity** : marquage/texte, différence de réflectance entre les lettres et le fond
- **Illumination Discontinuity** : ombre portée → luminosité différente

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im1-2.png|300]]

**Types d'edges.** Step function, gradient linéaire, gradient non-linéaire (roof edge), une ligne = un edge à deux côtés. On simplifie en général au cas step function.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im1-3.png|300]]

En pratique les images sont bruitées et discrètes, elles ne ressemblent pas à une step function propre.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im1-4.png|300]]

Ce qu'on veut d'un opérateur de détection d'edge :
- **Position** : où exactement se situe l'edge (précision sub-pixel)
- **Magnitude (Strength)** : pour pouvoir seuiller
- **Orientation (Direction)** : orientation du edge dans l'image

Avec un taux de détection élevé (peu de FP/FN), bonne localisation, résilience au bruit.

### 2. Edge detection using gradients

**1D.** La dérivée première monte, atteint un maximum au milieu de l'edge (montant), et l'inverse pour un edge descendant. Les **extrema locaux de la dérivée première** indiquent où sont les edges.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-1 (2).png|300]]

On ne regarde que la valeur absolue ; la hauteur du pic donne la force du edge.

**2D.** Généralisation à l'opérateur gradient :

$$
\nabla I=\left[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\right]
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-2.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-3.png|300]]

Magnitude (force) :

$$
S=\|\nabla I\|=\sqrt{\left(\frac{\partial I}{\partial x}\right)^{2}+\left(\frac{\partial I}{\partial y}\right)^{2}}
$$

Orientation :

$$
\theta=\tan ^{-1}\left(\frac{\partial I}{\partial y} / \frac{\partial I}{\partial x}\right)
$$

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-4.png|300]]

**Implémentation discrète.** Approximation par différences finies (au moins 2 pixels selon x et y), implémentable via convolution.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-5.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-6.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-7.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im2-8.png|300]]

### 3. Edge detection using Laplacian

En 1D : la dérivée seconde du signal vaut zéro à l'edge mais avec un **zero-crossing** très marqué — pas un pic comme la dérivée première.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-1 (2).png|300]]

Opérateur Laplacien :

$$
\nabla^{2} I=\frac{\partial^{2} I}{\partial x^{2}}+\frac{\partial^{2} I}{\partial y^{2}}
$$

- Les edges sont des **zero-crossings** du Laplacien de l'image
- Le Laplacien ne donne **pas** la direction du edge

Implémentation discrète (masque 3×3, coefficient $-4$ car on somme les deux dérivées secondes au carré) :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-2.png|300]]

Problème : les edges peuvent apparaître à n'importe quelle orientation (ex 45°), le masque de base ne capture pas la diagonale → version enrichie du filtre. Comme une image ne peut pas être négative, on décale (128 = zero-crossing) pour obtenir la carte finale des edges.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-3.png|300]]

**Effet du bruit.** Le bruit sur un edge fait perdre complètement l'edge à la dérivée première brute — il faut lisser avant d'appliquer le Laplacien.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-4.png|300]]

Solution : lissage gaussien avant dérivation, pic bien localisé à l'edge.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-5 (1).png|300]]

**Derivative of Gaussian (DoG).** La dérivée est linéaire, le lissage gaussien est linéaire → on peut précalculer la dérivée du noyau gaussien directement comme un seul masque de convolution.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-6.png|300]]

**Laplacian of Gaussian (LoG)** : même principe, zero-crossing net à la position de l'edge.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-7.png|300]]
![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-8.png|300]]

Gradient vs Laplacien :

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im3-9.png|300]]

Propriétés :

![[im3-10.png|300]]

### 4. Canny Edge Detector

Le détecteur d'edge le plus utilisé, combine les meilleures propriétés du gradient et du Laplacien.

1. Lisser l'image avec un gaussien 2D : $n_{\sigma} * I$
2. Calculer le gradient avec l'opérateur de Sobel : $\nabla n_{\sigma} * I$
3. Trouver la magnitude du gradient à chaque pixel : $\left\|\nabla n_{\sigma} * I\right\|$
4. Trouver l'orientation du gradient à chaque pixel : $\widehat{\boldsymbol{n}}=\frac{\nabla n_{\sigma} * I}{\left\|\nabla n_{\sigma} * I\right\|}$
5. Calculer le Laplacien le long de la direction du gradient $\widehat{n}$ à chaque pixel : $\frac{\partial^{2}\left(n_{\sigma} * I\right)}{\partial \widehat{n}^{2}}$
6. Trouver les zero-crossings du Laplacien directionnel → localisation des edges

> [!warning]- Étape manquante à compléter
> La description ci-dessus (Laplacien directionnel + zero-crossing) est une formulation valide de Canny, mais la version la plus classiquement enseignée utilise plutôt une **suppression non-maximale le long de la direction du gradient** suivie d'un **seuillage par hystérésis** (deux seuils, haut et bas, pour chaîner les edges faibles connectés à des edges forts). Ce point n'est pas dans les notes d'origine — à rajouter si besoin de la version "canonique".

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im4-1 (1).png|300]]

Rôle de $\sigma$ : plus on lisse, moins on détecte d'edges (edges plus épars) — on explore l'espace d'échelle (scale space) de l'image. Canny ne change qu'un seul paramètre ($\sigma$) pour sélectionner les edges pertinents à l'échelle voulue.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im4-2.png|300]]

### 5. Corner Detection

**Définition.** Un coin (corner) est un point où deux edges se rencontrent, i.e. changement rapide d'intensité dans **deux** directions dans une petite région.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im5-1.png|300]]

**Image gradients.** Normalisés : blanc = valeur positive forte, noir = valeur négative forte, gris = proche de zéro. Sur un edge on a un fort gradient le long du contour. Sur un coin on a des valeurs fortes (positives et négatives) pour $I_x$ **et** $I_y$.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im5-2.png|300]]

Trois cas dans l'espace des gradients :
- **Région plate** : tous les gradients sont petits
- **Région d'edge** : cluster compact au centre (régions plates de part et d'autre) + un cluster linéaire allongé proche du edge
- **Région de coin** : cluster compact + deux clusters additionnels (un pour chaque edge, fort $I_x$ pour l'un, fort $I_y$ pour l'autre)

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im5-3.png|300]]

On ajuste une ellipse centrée à l'origine sur la distribution des gradients : demi-grand axe $\lambda_1$, demi-petit axe $\lambda_2$ → classification de la région selon ces deux valeurs.

![[images/3-Apprentissage automatique/04_Computer vision/03_Detection_algo/1.Edge/im5-4.png|300]]

**Ajustement de l'ellipse.** On traite l'observation comme un objet binaire, on trouve l'axe de moment second minimal ($E_{min}$) et l'axe perpendiculaire de moment second maximal ($E_{max}$) :

$$
\text{Longueur demi-grand axe} = \lambda_1 = E_{max} \qquad \text{Longueur demi-petit axe} = \lambda_2 = E_{min}
$$

![[im5-5.png|300]]

On calcule les moments seconds $a, b, c$ (moment produit) à partir des points :

![[im5-6.png|300]]
![[im5-7.png|300]]

**Fonction de Harris.** Un point proche de l'origine = région plate. Harris a trouvé une fonction empirique $R$ : la luminosité est proportionnelle à $R$, au-dessus d'un seuil → coin probable.

![[im5-8.png|300]]

Exemple (image "BBC") : noir = faible $R$, blanc = fort $R$, valeurs très élevées aux coins.

![[im5-9.png|300]]

**Peak detection : non-maximal suppression.**
1. Faire glisser une fenêtre de taille $k$ sur l'image
2. À chaque position, si le pixel central est le maximum de la fenêtre → on le garde (positif). Sinon → on le supprime (négatif)

![[im5-10.png|300]]

Résultat sur une pièce de monnaie : bonne détection des coins.

![[im5-11.png|300]]

Cas plus difficile (circuit imprimé) : quelques coins manqués mais globalement bonne détection.

![[im5-12.png|300]]
