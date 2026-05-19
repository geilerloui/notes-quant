---
title: Fondamentaux de la Random Matrix Theory
date: 2026-05-12
tags: [random-matrix-theory, probabilités, spectre, wigner]
---
## I. La question

Une matrice aléatoire est une matrice dont les entrées sont des variables aléatoires. Question naïve : si on tire une matrice symétrique $M \in \mathbb{R}^{N \times N}$ avec des entrées i.i.d. gaussiennes, à quoi ressemble son spectre ?

Intuition naïve : *"ça doit être aléatoire, donc imprévisible"*. C'est faux.

Quand $N$ grandit, l'histogramme des valeurs propres se range selon une loi limite **déterministe**. Plus $N$ est grand, plus la convergence est forte — c'est un phénomène de concentration. La matrice est aléatoire, mais sa "forme spectrale" ne l'est pas.

Cette observation est née dans deux mondes très éloignés.

> [!example] Deux origines historiques
> **Wigner (1955)** étudie les noyaux atomiques lourds (uranium). Les niveaux d'énergie sont les valeurs propres d'un opérateur hamiltonien trop complexe pour être calculé. Wigner postule : *remplaçons cet hamiltonien par une matrice aléatoire*, et voyons quelles propriétés statistiques émergent. La loi du demi-cercle naît de cette idée.
>
> **Marchenko-Pastur (1967)** étudient un problème de statistique : si on observe $T$ réalisations i.i.d. d'un vecteur aléatoire $X \in \mathbb{R}^N$ dont les $N$ coordonnées sont indépendantes et de variance $1$ — autrement dit, la "vraie" matrice de covariance vaut l'identité $\mathbb{I}_N$ — alors la matrice de covariance *empirique* $\hat{\Sigma}$ qu'on calcule depuis les données **n'est pas** l'identité. Ses valeurs propres sont étalées sur un intervalle, selon une loi déterministe qui porte leur nom.

> [!note]- Rappel de notation : $\mathbb{I}_N$ et $\mathcal{N}(0, \Sigma)$
> $\mathbb{I}_N$ désigne la **matrice identité** de taille $N \times N$ : des $1$ sur la diagonale, des $0$ partout ailleurs.
>
> $\mathcal{N}(0, \Sigma)$ désigne un **vecteur aléatoire gaussien** de $\mathbb{R}^N$, centré (espérance nulle), dont la matrice de covariance vaut $\Sigma$. Le cas particulier $\mathcal{N}(0, \mathbb{I}_N)$ correspond donc à un vecteur dont les $N$ coordonnées sont **indépendantes**, chacune $\mathcal{N}(0, 1)$ — *aucune corrélation entre les coordonnées, variance $1$ partout*. C'est le scénario "bruit pur" : on suppose qu'il n'y a aucune structure cachée.

C'est cette deuxième origine qui nous intéresse comme quant. La matrice de covariance empirique est *l'objet* central de la gestion de portefeuille, et RMT en révèle les pathologies.

## II. Ensembles classiques

On distingue les matrices aléatoires selon deux critères : la **symétrie** imposée à la matrice (réelle symétrique, hermitienne, rectangulaire…) et la **distribution** des entrées (gaussienne ou non, indépendantes ou non). Trois grandes familles dominent la théorie. On les présente ici brièvement — la section III approfondit GOE, et la section V revient en détail sur Wishart.

### (i) GOE — Gaussian Orthogonal Ensemble

> [!warning] Définition — Matrice GOE
> Une **matrice du Gaussian Orthogonal Ensemble** (GOE) est une matrice $M \in \mathbb{R}^{N \times N}$ qui vérifie les trois conditions suivantes :
>
> 1. **Symétrie réelle** : $M_{ij} = M_{ji}$ pour tous $i, j$.
> 2. **Entrées gaussiennes** : chaque coefficient $M_{ij}$ suit une loi normale centrée.
> 3. **Indépendance** : les entrées au-dessus de la diagonale (incluse) sont mutuellement indépendantes ; celles en-dessous sont déterminées par la contrainte de symétrie.
>
> Sous forme schématique :
> $$M = \begin{pmatrix} M_{11} & M_{12} & \cdots & M_{1N} \\ M_{12} & M_{22} & \cdots & M_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ M_{1N} & M_{2N} & \cdots & M_{NN} \end{pmatrix}, \qquad M_{ij} \sim \mathcal{N}(0, \sigma_{ij}^2)$$
> où les variances $\sigma_{ij}^2$ sont calibrées selon une convention discutée plus bas.

**Pourquoi le mot *orthogonal* ?** Parce que la distribution de $M$ est invariante par changement de repère orthogonal $M \mapsto OMO^\top$ (cf. [[Algèbre/Changements de base et invariance|rappels d'algèbre linéaire]]). En clair : si tu tires une GOE et qu'un collègue en tire une autre dans un repère tourné, vous ne pouvez pas distinguer les deux statistiquement. C'est la matrice "la plus aléatoire possible" parmi les matrices symétriques — aucune direction de l'espace n'est privilégiée.

**Pourquoi cet ensemble est central.** C'est le plus simple à analyser et celui sur lequel toute la théorie classique a été construite. Il servira de cadre pédagogique pour développer la machinerie dans la section III (normalisation, loi limite, bulk/edge).

> [!note]- Détail technique : pourquoi $\text{Var}(M_{ii}) = 2$ et $\text{Var}(M_{ij}) = 1$ ?
> Pour rendre l'invariance orthogonale exacte, il faut calibrer les variances. La convention standard est $M_{ii} \sim \mathcal{N}(0, 2)$ sur la diagonale et $M_{ij} \sim \mathcal{N}(0, 1)$ hors diagonale (pour $i \neq j$).
>
> Le ratio 2:1 sort d'un calcul algébrique : sous une rotation $O$, les éléments diagonaux et hors-diagonaux se mélangent, et seul ce ratio précis fait que la distribution reste inchangée. Les valeurs absolues (1 et 2) sont conventionnelles — on aurait pu prendre (1/2, 1/4) avec une normalisation différente. **Ce qu'il faut retenir : entrées gaussiennes centrées indépendantes, avec une variance calibrée pour l'invariance.**

### (ii) GUE — Gaussian Unitary Ensemble

> [!warning] Définition — Matrice GUE
> Une **matrice du Gaussian Unitary Ensemble** (GUE) est une matrice $M \in \mathbb{C}^{N \times N}$ qui vérifie :
>
> 1. **Hermicité** : $M_{ji} = \overline{M_{ij}}$ (les coefficients sont complexes conjugués par transposition).
> 2. **Entrées gaussiennes complexes** : chaque $M_{ij}$ ($i \neq j$) a partie réelle et partie imaginaire i.i.d. $\mathcal{N}(0, 1/2)$ ; les entrées diagonales sont réelles, $M_{ii} \sim \mathcal{N}(0, 1)$.
> 3. **Indépendance** des entrées au-dessus de la diagonale.

C'est la version complexe de GOE. Le nom *unitaire* vient de l'invariance par transformations unitaires $M \mapsto UMU^*$ (l'équivalent complexe des transformations orthogonales). En physique, GUE modélise les hamiltoniens quantiques sans symétrie de renversement du temps. Mathématiquement, il est plus simple à analyser que GOE — la loi jointe des valeurs propres a une forme plus compacte. **On ne l'utilisera pas directement** : c'est juste utile à savoir qu'il existe et que les résultats type "demi-cercle" s'y appliquent aussi.

### (iii) Wishart — covariance empirique

> [!warning] Définition — Matrice de Wishart
> Soit $X \in \mathbb{R}^{N \times T}$ une matrice de **données** dont les $T$ colonnes sont des vecteurs i.i.d. tirés selon $\mathcal{N}(0, \mathbb{I}_N)$. La **matrice de Wishart** associée est
> $$W = \frac{1}{T} X X^\top \in \mathbb{R}^{N \times N}.$$
> Elle est :
>
> 1. **Symétrique réelle** : $W^\top = W$ par construction ($X X^\top$ est toujours symétrique).
> 2. **Semi-définie positive** : ses valeurs propres sont toutes $\geq 0$, car pour tout vecteur $v \in \mathbb{R}^N$, $v^\top W v = \frac{1}{T} \|X^\top v\|^2 \geq 0$.
>
> Concrètement, $W$ est *la* matrice de covariance empirique d'un échantillon gaussien centré de taille $T$ en dimension $N$.

**Pourquoi cet ensemble est central pour nous.** En finance, $X$ représente $N$ actifs observés sur $T$ jours (chaque ligne = un actif, chaque colonne = un jour de rendements), et $W$ est la matrice de covariance estimée à partir de ces données. C'est l'objet sur lequel reposent tous les modèles de portefeuille (Markowitz, Barra, etc.). RMT va nous dire ce qui se passe quand $N$ et $T$ sont **comparables** — et la réponse est dérangeante.

On laisse Wishart de côté pour l'instant et on développe la théorie dans le cadre plus simple de GOE. On y reviendra **en détail à la section V**, une fois les outils en main, pour exhiber concrètement la pathologie qu'on vient d'annoncer.

## III. Étude détaillée du GOE

On va maintenant développer toute la machinerie sur l'ensemble le plus simple : le GOE. Trois étapes : (1) comprendre pourquoi il faut normaliser, (2) énoncer la loi du demi-cercle, (3) distinguer bulk et edge.

### A. La normalisation $1/\sqrt{N}$ — pourquoi elle est obligatoire

C'est l'idée la plus importante de la section. Sans elle, rien ne marche.

**Le problème.** Tire une matrice GOE $M$ de taille $N \times N$ (entrées gaussiennes standard, symétrique). Calcule sa plus grande valeur propre $\lambda_{\max}$. Que se passe-t-il quand $N$ grandit ?

**Réponse :** $\lambda_{\max}$ explose. Plus précisément, $\lambda_{\max}(M) \approx 2\sqrt{N}$ :

| $N$ | $\lambda_{\max}$ typique |
|---|---|
| $100$ | $\sim 20$ |
| $10\,000$ | $\sim 200$ |
| $10^6$ | $\sim 2000$ |

Le spectre ne converge vers rien — il part à l'infini avec $N$.

**Intuition de pourquoi.** Une matrice $N \times N$ a $\sim N^2$ entrées. Quand tu calcules le produit $Mv$ pour un vecteur unitaire $v$, chaque coordonnée du résultat est une somme de $N$ termes aléatoires indépendants. Par CLT, cette somme a une amplitude de l'ordre de $\sqrt{N}$. Les valeurs propres, qui sont des "facteurs d'étirement" sous l'action de $M$, héritent de cette échelle.

**La solution.** On travaille avec la matrice renormalisée

$$\widetilde{M} = \frac{M}{\sqrt{N}}$$

Ses valeurs propres sont divisées par $\sqrt{N}$, donc $\lambda_{\max}(\widetilde{M}) \approx 2$ **indépendamment de $N$**. Le spectre se range sur l'intervalle $[-2, 2]$ quelle que soit la taille. C'est sur cet intervalle stable que vit la loi du demi-cercle.

> [!example] Exemple à la main — l'effet de la normalisation
> Reprenons une matrice 2×2 :
> $$M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}, \quad \text{spectre} : \{0.4, 3.0\}$$
> On divise par $\sqrt{N} = \sqrt{2} \approx 1.414$. Les valeurs propres sont divisées par ce facteur :
> $$\widetilde{M} = M / \sqrt{2} \quad \Longrightarrow \quad \text{spectre} : \{0.28, 2.12\}$$
> Toutes les valeurs propres tombent dans l'intervalle $[-2, 2]$. Si on refaisait l'exercice à $N = 1000$ sans normaliser, $\lambda_{\max}$ serait $\sim 63$ et le spectre s'étalerait sur $[-63, 63]$. Avec normalisation, il reste cantonné dans $[-2, 2]$, peu importe $N$.

> [!example] Analogie : le théorème central limite
> Tu connais ce phénomène ailleurs. Si $X_1, \dots, X_n$ sont i.i.d. centrées de variance 1, alors $S_n = X_1 + \dots + X_n$ a une variance de $n$ et "explose". Pour avoir une limite non-triviale, on regarde $S_n / \sqrt{n}$, qui converge vers $\mathcal{N}(0,1)$.
>
> La normalisation $1/\sqrt{N}$ en RMT joue *exactement* le même rôle : c'est le bon facteur d'échelle pour qu'une matrice aléatoire ait une limite spectrale non-triviale.

> [!warning] À retenir
> RMT ne dit pas que "tout converge tout seul". Elle dit qu'**à la bonne échelle**, des objets aléatoires révèlent une structure déterministe. Le choix de la normalisation est ce qui définit le régime d'étude. Pour Wishart, on verra en section V que la normalisation correcte est $1/T$ (et non $1/\sqrt{T}$), parce qu'on a un produit $XX^\top$ — c'est-à-dire un *carré*, qui demande une normalisation au carré.

![[fig1_normalisation.png]]
*Figure 1. À gauche : la plus grande valeur propre d'une GOE non-normalisée croît comme $2\sqrt{N}$ — le spectre n'a pas de limite. À droite : après normalisation par $\sqrt{N}$, $\lambda_{\max}$ se stabilise autour de $2$ quelle que soit la taille.*

### B. La loi du demi-cercle de Wigner

C'est *le* résultat fondateur. On considère une matrice GOE normalisée $W_N = M / \sqrt{N}$. Soit $\lambda_1, \dots, \lambda_N$ ses valeurs propres.

Avant d'énoncer le théorème, on doit poser proprement la notation $\mu_N$ et $\delta_{\lambda_i}$ — c'est l'objet mathématique qui décrit "l'histogramme des valeurs propres".

> [!note]- Rappel de notation : qu'est-ce que $\mu_N$ et $\delta_{\lambda_i}$ ?
> **Masse de Dirac $\delta_a$.** C'est une mesure de probabilité concentrée entièrement sur un seul point $a \in \mathbb{R}$ : probabilité $1$ d'être en $a$, masse nulle partout ailleurs. Graphiquement, on la représente par un "pic" (une tige verticale) au point $a$. Ce n'est *pas* un vecteur, c'est une mesure.
>
> **Mesure spectrale empirique $\mu_N$.** Ce n'est pas un vecteur non plus — c'est une mesure de probabilité sur $\mathbb{R}$. La formule
> $$\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{\lambda_i}$$
> dit exactement : *"on plante une tige de hauteur $1/N$ à chacune des $N$ valeurs propres"*. Le facteur $1/N$ est là pour que les hauteurs des tiges somment à $1$ (probabilité totale = $1$).
>
> **Attention — $\mu_N$ n'est PAS uniforme.** Toutes les tiges ont la *même hauteur* $1/N$, mais elles sont placées **aux positions des valeurs propres** — pas à intervalles réguliers. Pour une GOE, les valeurs propres sont concentrées près de $0$ et clairsemées près de $\pm 2$ : donc beaucoup de tiges au centre, peu sur les bords. C'est *cette densité de tiges qui varie* qui fait apparaître la forme en demi-disque quand on regroupe par bins. Voir la figure ci-dessous pour visualiser exactement l'opération.
>
> **Pourquoi cette formulation ?** Les valeurs propres sont des points isolés (objets discrets), pas une fonction continue. Les masses de Dirac permettent de représenter des objets discrets *et* continus dans le même cadre : la mesure discrète $\mu_N$ (faite de Dirac) peut converger, quand $N \to \infty$, vers une mesure continue $\mu$ ayant une densité $\rho(\lambda)$. C'est exactement ce que dit le théorème.

![[fig_dirac_vs_histogramme.png]]
*Figure 2. **Gauche** : la mesure spectrale empirique $\mu_N$ comme somme de Dirac. Chacune des $N=200$ tiges a exactement la même hauteur $1/N = 0.005$, mais leurs positions (les valeurs propres) sont concentrées vers $0$ et rares vers $\pm 2$. **Droite** : quand on regroupe ces Dirac par intervalles (bins), les hauteurs d'histogramme reflètent la densité de tiges et reconstituent la loi du demi-cercle $\rho_{\rm sc}(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}$ (courbe rouge).*

> [!warning] Théorème (Wigner, 1955)
> Quand $N \to \infty$, la mesure spectrale empirique
> $$\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{\lambda_i}$$
> converge faiblement vers la **loi du demi-cercle**, qui est la mesure de probabilité de densité
> $$\rho_{\text{sc}}(\lambda) = \frac{1}{2\pi} \sqrt{4 - \lambda^2} \, \mathbb{1}_{[-2, 2]}(\lambda)$$
>
> En clair : à $N$ grand, l'histogramme des valeurs propres ressemble à la courbe $\rho_{\rm sc}$ — un demi-disque de rayon $2$ centré à l'origine.

Concrètement : le spectre est étalé sur $[-2, 2]$, avec une densité en forme de demi-disque (d'où le nom). Aucune valeur propre ne dépasse $2$ asymptotiquement, et la majorité s'accumule au centre.

![[fig2_semicircle.png]]
*Figure 3. Histogramme des valeurs propres d'une matrice GOE normalisée pour $N=50$, $N=200$, $N=1000$. La courbe rouge est la densité limite $\rho_{\text{sc}}(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}$. La convergence est visible dès $N=200$, et excellente à $N=1000$.*

#### Intuition par les moments

Pourquoi un demi-cercle, et pas une gaussienne ? L'idée vient du calcul des moments. La $k$-ième moment de $\mu_N$ est

$$\int \lambda^k \, d\mu_N(\lambda) = \frac{1}{N} \sum_i \lambda_i^k = \frac{1}{N} \, \text{tr}(W_N^k)$$

Or $\text{tr}(W_N^k) = \sum_{i_1, \dots, i_k} W_{i_1 i_2} W_{i_2 i_3} \cdots W_{i_k i_1}$. Chaque terme est un **chemin fermé** dans le graphe complet à $N$ sommets, de longueur $k$. En espérance, seuls les chemins où chaque arête est parcourue un nombre **pair** de fois survivent (car les entrées sont centrées). Une combinatoire fine montre que les chemins dominants, à $N$ grand, correspondent aux **arbres**, dont le nombre est compté par les **nombres de Catalan** :

$$\lim_{N \to \infty} \mathbb{E}\!\left[ \frac{1}{N} \text{tr}(W_N^{2k}) \right] = C_k = \frac{1}{k+1}\binom{2k}{k}$$

et les moments impairs tendent vers $0$. Ces moments-là sont exactement les moments de la loi du demi-cercle. La loi est entièrement caractérisée par ses moments, donc CQFD (modulo la rigueur sur la convergence faible).

> [!note]- Pourquoi les Catalan ?
> Les nombres de Catalan comptent un grand nombre d'objets combinatoires : chemins de Dyck, parenthésages, arbres binaires, triangulations… Tous ces objets vérifient la même récurrence $C_{k+1} = \sum_{i=0}^k C_i C_{k-i}$. Cette récurrence se traduit, côté densité, en une équation algébrique pour la transformée de Stieltjes — équation dont la solution donne précisément $\rho_{\text{sc}}$. C'est l'idée centrale qu'on retrouvera pour Marchenko-Pastur.

### C. Bulk vs edge

La loi du demi-cercle décrit la **forme globale** du spectre — combien de valeurs propres tombent dans tel ou tel intervalle. Mais à $N$ fini, il y a deux phénomènes plus fins qu'elle ne capture pas, et qui sont gouvernés par d'autres lois limites. Cette section les pose proprement.

#### Trois échelles, trois lois limites

Pour bien voir, il faut comprendre qu'on étudie le spectre à **trois échelles d'observation différentes**, comme on zoomerait sur une carte.

> [!warning] Les trois échelles
> **Échelle 1 — densité globale (macro).** *Question :* combien de $\lambda_i$ dans un intervalle de largeur fixée comme $[0.5, 0.7]$ ? *Réponse :* la **loi du demi-cercle**. C'est ce qu'on vient de voir en section B.
>
> **Échelle 2 — espacement local entre valeurs propres voisines (micro, dans le bulk).** *Question :* à quelle distance typique se trouvent $\lambda_i$ et $\lambda_{i+1}$ ? *Réponse :* la **loi du sine kernel**.
>
> **Échelle 3 — fluctuations de $\lambda_{\max}$ (bord du spectre, edge).** *Question :* à quelle distance de $2$ se trouve $\lambda_{\max}$ pour un tirage donné ? *Réponse :* la **loi de Tracy-Widom**.

Le mot **bulk** désigne tout simplement l'intérieur du support spectral, ici $(-2, 2)$ — là où la densité limite est non nulle. Le mot **edge** désigne les bords, $\pm 2$. Donc Tracy-Widom vit à l'edge, sine kernel vit dans le bulk.

| Régime | Question posée | Échelle de fluctuation | Loi limite |
|---|---|---|---|
| Densité macro | Combien de $\lambda_i$ dans $[a,b]$ ? | — (échelle $N$) | Demi-cercle |
| Bulk (micro) | Distance entre $\lambda_i$ voisins ? | $1/N$ | Sine kernel |
| Edge | Distance de $\lambda_{\max}$ à $2$ ? | $N^{-2/3}$ | Tracy-Widom |

On détaille maintenant les deux dernières lignes.

#### Tracy-Widom : la loi des fluctuations de $\lambda_{\max}$

**Le constat empirique.** On sait que $\lambda_{\max} \to 2$ asymptotiquement. Mais à $N$ fini, $\lambda_{\max}$ flotte autour de $2$. Si on tire 4000 GOE de taille $N$ et qu'on regarde la distribution des $\lambda_{\max}$ obtenus, on voit que plus $N$ est grand, plus la distribution est *étroite* (concentrée près de $2$). Sur le panneau (a) de la figure 4 : à $N=20$ la distribution est large, à $N=500$ elle est très resserrée autour de 2.

**Le problème.** Si on superpose ces histogrammes tels quels, on ne peut pas les comparer — ils ont des largeurs très différentes. Pour pouvoir parler de la *forme* de la distribution de $\lambda_{\max}$, il faut **rescaler**.

**Le rescaling.** On définit la variable centrée-réduite

$$\xi = N^{2/3}\,(\lambda_{\max} - 2)$$

On retire la valeur asymptotique ($2$), puis on multiplie par $N^{2/3}$ pour "agrandir l'image". C'est l'exact analogue du théorème central limite, où on regarde $(S_n - n\mu)/\sqrt{n}$ pour révéler la gaussienne limite. Ici, le facteur n'est pas $\sqrt{N}$ mais $N^{2/3}$ — c'est *ça* qui est non trivial.

> [!warning] Théorème (Tracy-Widom, 1994)
> Pour une matrice GOE normalisée, la variable rescalée
> $$\xi_N = N^{2/3}\,(\lambda_{\max} - 2)$$
> converge en loi, quand $N \to \infty$, vers une distribution déterministe appelée **loi de Tracy-Widom** $TW_1$, indépendante de $N$.
>
> Cette loi a une densité asymétrique (queue plus longue à gauche qu'à droite), de moyenne $\approx -1.21$ et d'écart-type $\approx 1.27$.

Pourquoi $N^{2/3}$ et pas $\sqrt{N}$ ? Parce qu'au bord du spectre, la densité du demi-cercle $\rho_{\rm sc}(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}$ s'annule comme $\sqrt{2-\lambda}$ près de $\lambda = 2$. Cette annulation en racine carrée modifie les exposants : un calcul (qu'on ne fait pas ici) montre que l'échelle naturelle des fluctuations devient $N^{-2/3}$ au lieu de $N^{-1/2}$.

![[fig_tracy_widom.png]]
*Figure 4. **Gauche** : sans rescaling, la distribution de $\lambda_{\max}$ se contracte sur $\lambda = 2$ quand $N$ augmente — impossible de comparer les formes. **Droite** : après rescaling $\xi = N^{2/3}(\lambda_{\max} - 2)$, les histogrammes pour $N=20$ et $N=100$ se superposent à la courbe Tracy-Widom (noire). L'asymétrie est claire : queue plus longue à gauche.*

> [!example] À quoi sert Tracy-Widom ?
> C'est un **test statistique de détection de signal**. Concrètement : tu calcules une covariance empirique sur $N$ actifs et tu trouves une grosse valeur propre $\lambda_1$ qui dépasse l'edge théorique. Question : est-ce un vrai facteur de marché, ou juste la fluctuation aléatoire du plus grand $\lambda$ du bulk sous hypothèse de bruit pur ?
>
> Avec Tracy-Widom, tu peux quantifier :
> $$P\!\left( \lambda_{\max}^{\text{bruit}} > \lambda_1 \right) = P\!\left( \xi > N^{2/3}(\lambda_1 - 2) \right)$$
> et la queue droite de TW te donne directement cette probabilité.
>
> Si elle vaut $10^{-6}$, $\lambda_1$ est très probablement un vrai signal. Si elle vaut $0.3$, c'est compatible avec du bruit pur. C'est l'**outil fondamental** pour décider combien de facteurs garder dans une PCA, et l'objet de la transition BBP (voir note dédiée).

#### Sine kernel : la répulsion entre valeurs propres

**Le constat empirique.** Prends une grosse GOE et regarde de près deux valeurs propres consécutives $\lambda_i$ et $\lambda_{i+1}$ au centre du spectre. À quelle distance sont-elles ? Si les valeurs propres étaient indépendantes (tirées au hasard sur $[-2, 2]$ comme des points de Poisson), leur espacement suivrait une loi exponentielle $p(s) = e^{-s}$, qui a sa **valeur maximale en $s = 0$** : deux points peuvent s'agglutiner arbitrairement près l'un de l'autre.

Pour une GOE, ce n'est pas du tout ce qu'on observe. Les valeurs propres se **repoussent** : il est très rare de voir $\lambda_i \approx \lambda_{i+1}$. La densité d'espacement vaut **zéro** en $s = 0$, et atteint son maximum vers $s \approx 0.8$.

> [!warning] Théorème (Sine kernel, Dyson 1962)
> Soit $\lambda_i$ et $\lambda_{i+1}$ deux valeurs propres consécutives d'une GOE normalisée, au centre du spectre. On normalise l'espacement par l'espacement moyen local :
> $$s = \frac{\lambda_{i+1} - \lambda_i}{\bar{s}}, \qquad \bar{s} = \frac{\pi}{N}$$
> Quand $N \to \infty$, la loi de $s$ converge vers une distribution déterministe appelée **loi du sine kernel**. Une excellente approximation (la *conjecture de Wigner*) est :
> $$p(s) \approx \frac{\pi}{2}\,s\,e^{-\pi s^2/4}$$
>
> Propriétés clés : $p(0) = 0$ (répulsion totale), maximum vers $s \approx 0.8$, queue gaussienne — *pas* exponentielle.

L'espacement moyen $\bar{s} = \pi/N$ se calcule directement à partir du demi-cercle : à $\lambda = 0$, la densité limite vaut $\rho_{\rm sc}(0) = 1/\pi$, donc il y a $N \cdot \rho_{\rm sc}(0) = N/\pi$ valeurs propres par unité de longueur autour de $0$, et l'espacement moyen est l'inverse de cette densité, soit $\pi/N$.

![[fig_sine_kernel.png]]
*Figure 5. Distribution des espacements normalisés $s = (\lambda_{i+1} - \lambda_i)/\bar{s}$ pour les valeurs propres au centre du spectre de 200 GOE de taille $N=500$. L'histogramme empirique (bleu) suit la courbe sine kernel (rouge). En pointillés verts : la densité $p(s) = e^{-s}$ qu'on observerait si les valeurs propres étaient des points indépendants. La différence en $s = 0$ est le **phénomène de répulsion** : pour RMT, $p(0) = 0$ ; pour Poisson, $p(0) = 1$ est maximal.*

> [!note]- Universalité du sine kernel
> Le sine kernel apparaît bien au-delà de RMT : il gouverne l'espacement des zéros de la fonction zêta de Riemann sur la ligne critique, et il a été observé sur des données *complètement non mathématiques* — comme les écarts entre les heures d'arrivée des bus dans certaines villes (Cuernavaca, Mexique). Cette universalité est l'une des grandes énigmes de RMT moderne : pourquoi *cette* loi en particulier émerge-t-elle dans des contextes aussi disparates ?
>
> Pour un quant, en pratique : le sine kernel sert essentiellement de **test de cohérence**. Si tu calcules les espacements des valeurs propres d'une matrice de covariance et qu'ils suivent le sine kernel, c'est cohérent avec l'hypothèse "structure RMT pure". Si les espacements s'en écartent, c'est qu'il y a une structure non triviale (facteurs cachés, corrélations résiduelles) à investiguer.

#### Pourquoi cette distinction est cruciale en finance

Le **bulk** correspond au bruit d'estimation : ces valeurs propres parasites créées par l'estimation empirique, qu'on a vues sur l'exemple Wishart $N=3, T=5$ en section V. Elles n'encodent aucun signal financier — ce sont juste des artefacts statistiques.

L'**edge**, lui, est l'endroit où vivent les vrais facteurs. Un facteur de marché réel se traduit par une valeur propre qui **sort du bulk** vers la droite, dépassant le bord supérieur. Toute la question est : *à partir de quel seuil une valeur propre est-elle significativement au-dessus du bruit ?* Tracy-Widom fournit la réponse rigoureuse. C'est l'objet de la **transition BBP** (Baik-Ben Arous-Péché), qu'on traitera dans une note dédiée.

## IV. Outils — densité spectrale et transformée de Stieltjes

Deux objets reviennent partout en RMT. Autant les poser proprement maintenant, avant d'attaquer Wishart.

### Densité spectrale empirique

Pour une matrice $M$ de valeurs propres $\lambda_1, \dots, \lambda_N$ :

$$\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{\lambda_i}$$

C'est une mesure de probabilité aléatoire sur $\mathbb{R}$ (chaque tirage de $M$ donne une mesure différente). Toute la RMT consiste essentiellement à étudier la limite de $\mu_N$ quand $N \to \infty$. Pour les ensembles classiques, cette limite est une mesure déterministe (Wigner, Marchenko-Pastur, etc.).

### Transformée de Stieltjes

Soit $\mu$ une mesure de probabilité sur $\mathbb{R}$. Sa **transformée de Stieltjes** est la fonction définie pour $z \in \mathbb{C} \setminus \mathbb{R}$ par

$$g(z) = \int_{\mathbb{R}} \frac{d\mu(\lambda)}{\lambda - z}$$

Pour une mesure empirique $\mu_N$, on a directement

$$g_N(z) = \frac{1}{N} \sum_{i=1}^N \frac{1}{\lambda_i - z} = \frac{1}{N} \, \text{tr}\!\left( (M - z\mathbb{I})^{-1} \right)$$

C'est la trace normalisée de la **résolvante** de $M$. Cette identité est tout l'intérêt de l'outil : $g_N$ s'exprime via une opération matricielle (l'inverse), ce qui rend les calculs traitables.

> [!warning] Pourquoi Stieltjes est central
> Trois propriétés clés.
>
> **(1) Inversion.** À partir de $g$, on récupère la densité $\rho$ de $\mu$ par
> $$\rho(\lambda) = \lim_{\varepsilon \to 0^+} \frac{1}{\pi} \, \text{Im} \, g(\lambda + i\varepsilon)$$
> Donc connaître $g$, c'est connaître $\mu$.
>
> **(2) Convergence faible $\Leftrightarrow$ convergence ponctuelle de $g$.** Pour montrer $\mu_N \to \mu$, il suffit de montrer $g_N(z) \to g(z)$ pour tout $z \in \mathbb{C} \setminus \mathbb{R}$.
>
> **(3) Équations auto-cohérentes.** Pour les ensembles classiques (Wigner, Wishart…), la limite $g(z)$ vérifie une équation algébrique simple qu'on peut résoudre explicitement. C'est *la* technique de preuve standard.

À titre d'exemple anticipé, pour la loi du demi-cercle, $g$ vérifie

$$g(z) = \frac{1}{-z - g(z)} \quad \Longleftrightarrow \quad g(z)^2 + z\,g(z) + 1 = 0$$

et la résolution donne $g(z) = \frac{-z + \sqrt{z^2 - 4}}{2}$ (avec la bonne branche). On retrouve $\rho_{\text{sc}}$ par inversion. La note sur Marchenko-Pastur reprendra exactement ce schéma.

![[fig4_stieltjes.png]]
*Figure 6. Partie imaginaire de la transformée de Stieltjes empirique $g_N(\lambda + i\varepsilon)$ (avec $\varepsilon = 0.01$) pour $N=500$. La courbe $\frac{1}{\pi}\text{Im}\,g_N(\lambda + i\varepsilon)$ reconstitue exactement la densité du demi-cercle — illustration concrète de la formule d'inversion.*

## V. Retour sur Wishart — l'estimation "ment"

On a maintenant tous les outils. Revenons à la matrice qui nous intéresse vraiment :

$$W = \frac{1}{T} X X^\top, \qquad X \in \mathbb{R}^{N \times T}$$

dont les colonnes sont i.i.d. selon $\mathcal{N}(0, \mathbb{I}_N)$. Avant d'énoncer Marchenko-Pastur (objet de la prochaine note), on va exhiber concrètement la pathologie qu'on a annoncée en section II.

> [!example] Exemple à la main — l'estimation "ment" même dans le cas le plus simple
> Mettons-nous dans un scénario où on **connaît la vérité** : on simule $N = 3$ actifs strictement indépendants, chacun de variance $1$. Autrement dit, la **vraie** covariance de la population est
> $$\Sigma_{\text{vrai}} = \mathbb{I}_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \quad \text{spectre : } \{1, 1, 1\}$$
> Les trois valeurs propres valent exactement $1$ — aucune dispersion, aucune corrélation.
>
> Maintenant, on observe $T = 5$ jours de rendements simulés selon cette vérité :
> $$X = \begin{pmatrix}
> 0.4 & -0.2 & 0.1 & 0.8 & -0.5 \\
> -0.1 & 0.3 & -0.4 & 0.2 & 0.6 \\
> 0.7 & -0.5 & 0.2 & -0.3 & 0.1
> \end{pmatrix}$$
> (chaque ligne = un actif, chaque colonne = un jour). À partir de ces données — et **uniquement** de ces données, comme dans la vraie vie — on calcule la covariance **empirique** :
> $$\hat{\Sigma} = W = \frac{1}{T} X X^\top \approx \begin{pmatrix}
> 0.22 & -0.10 & -0.03 \\
> -0.10 & 0.13 & -0.16 \\
> -0.03 & -0.16 & 0.18
> \end{pmatrix} \quad \text{spectre : } \{0.02, \; 0.18, \; 0.33\}$$
>
> **Compare les deux spectres :**
>
> | Spectre vrai (population) | Spectre empirique (5 jours) |
> |---|---|
> | $\{1, 1, 1\}$ — tout égal | $\{0.02, 0.18, 0.33\}$ — fortement dispersé |
>
> La vérité est plate. L'estimation, elle, fabrique des valeurs propres qui s'écartent énormément les unes des autres. **Aucune des deux n'est l'identité, alors qu'on a tiré les données *depuis* l'identité.** L'estimation crée artificiellement de la dispersion spectrale, juste parce qu'on a peu de données ($T = 5$) par rapport à la dimension ($N = 3$).

> [!warning] Le ratio $q = N/T$
> Toute la pathologie est gouvernée par le **ratio dimension / longueur d'historique** :
> $$q = \frac{N}{T}$$
>
> - $q \to 0$ (énormément de données par actif) : la covariance empirique converge vers la vraie covariance, le spectre se concentre autour de $\{1, 1, \dots, 1\}$.
> - $q$ comparable à $1$ : la dispersion empirique est sévère, comme dans notre exemple ($q = 3/5$).
> - $q > 1$ (plus d'actifs que de jours) : la matrice empirique devient singulière, certaines valeurs propres sont exactement $0$.
>
> En finance, $q$ est typiquement **non négligeable** : un univers de $N = 500$ actifs (S&P 500) sur $T = 250$ jours (une année boursière) donne $q = 2$. Le bruit d'estimation domine.

**Pourquoi $1/T$ et pas $1/\sqrt{T}$ ?** Section III.1, on avait normalisé GOE par $1/\sqrt{N}$ pour stabiliser le spectre. Ici, on travaille avec un *produit* $X X^\top$. Si $X$ a des entrées d'ordre $1$, alors $XX^\top$ a des entrées qui sont des sommes de $T$ produits — donc d'ordre $T$ par la loi des grands nombres. Pour stabiliser ces entrées (et donc le spectre) en $T \to \infty$, il faut diviser par $T$, pas par $\sqrt{T}$. La normalisation au carré reflète la structure produit.

## Ce qui suit

On a vu **empiriquement** que le spectre de Wishart est étalé là où la vérité est ponctuelle, et que cet étalement dépend du ratio $q = N/T$. La question naturelle : *comment décrire précisément cet étalement ?* La réponse est la **loi de Marchenko-Pastur**, qui donne la densité limite explicite du spectre de $W$ quand $N, T \to \infty$ à $q$ fixé. On l'obtiendra exactement par la même méthode qu'on a esquissée pour le demi-cercle (section IV) : équation auto-cohérente sur la transformée de Stieltjes. C'est l'objet de la prochaine note.
