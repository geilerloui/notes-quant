---
title: Fondamentaux de la Random Matrix Theory
date: 2026-05-12
tags: [random-matrix-theory, probabilités, spectre, wigner]
---

## 0. Rappels — spectre et invariance orthogonale

Avant d'attaquer, deux objets d'algèbre linéaire qu'on va utiliser sans arrêt.

### (i) Spectre d'une matrice

Le **spectre** d'une matrice carrée $M$, c'est juste l'**ensemble de ses valeurs propres**. Rien de plus.

Rappel : $\lambda \in \mathbb{R}$ est valeur propre de $M$ s'il existe un vecteur non nul $v$ tel que $Mv = \lambda v$. Géométriquement : il existe une direction $v$ dans laquelle $M$ agit comme une simple multiplication par le scalaire $\lambda$ — pas de rotation, juste un étirement (ou compression) d'un facteur $\lambda$. Une matrice $N \times N$ a $N$ valeurs propres (comptées avec multiplicité).

Quand on parlera de "loi du spectre" en RMT, on parlera de **l'histogramme de ces $N$ valeurs propres**. On tire une matrice aléatoire $N \times N$, on calcule ses $N$ valeurs propres, on les met dans un histogramme. La question centrale : à quoi ressemble cet histogramme quand $N$ devient grand ?

> [!example] Exemple à la main — spectre d'une matrice 2×2
> Soit
> $$M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}$$
> Cette matrice est symétrique, avec deux valeurs propres bien contrastées :
> $$\lambda_1 = 3.0, \quad \lambda_2 = 0.4$$
> associées aux vecteurs propres
> $$v_1 = \begin{pmatrix} 0.89 \\ 0.45 \end{pmatrix}, \quad v_2 = \begin{pmatrix} -0.45 \\ 0.89 \end{pmatrix}$$
> Ces deux vecteurs sont orthogonaux ($v_1^\top v_2 = 0$) et de norme 1 — ils forment une **base orthonormale** de $\mathbb{R}^2$.
>
> Cela signifie que n'importe quel vecteur $w \in \mathbb{R}^2$ s'écrit comme combinaison linéaire de $v_1$ et $v_2$ :
> $$w = \alpha_1 v_1 + \alpha_2 v_2, \quad \text{avec} \quad \alpha_i = v_i^\top w$$
> Par exemple, pour $w = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ : $\alpha_1 = 0.89$, $\alpha_2 = -0.45$. Le vecteur $w$ se décompose en $0.89$ parts de direction $v_1$ et $-0.45$ parts de direction $v_2$. Ce sont exactement les coordonnées de $w$ dans le repère des vecteurs propres — c'est-à-dire $Q^\top w$ avec $Q = [v_1 | v_2]$.

> [!warning] Théorème spectral
> Toute matrice **symétrique réelle** $M \in \mathbb{R}^{N \times N}$ admet $N$ valeurs propres réelles et $N$ vecteurs propres **orthogonaux entre eux**. On peut donc écrire
> $$M = Q \Lambda Q^\top$$
> avec $Q = [v_1 | \cdots | v_N]$ orthogonale ($Q^\top Q = I$) et $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$. Les vecteurs propres forment une base orthonormale de $\mathbb{R}^N$ — tout vecteur $w$ se décompose uniquement comme $w = \sum_i (v_i^\top w)\, v_i$.

![[images/1-Mathématiques/Random Matrix Theory/figA_vecteurs_propres.png]]
*Figure A. Chaque vecteur bleu (sur le cercle unité) est transformé en un vecteur rouge par $M$ — en général il change de direction. Seuls les deux vecteurs propres (vert foncé) restent sur leur droite : $\lambda_1 = 3.0$ étire, $\lambda_2 = 0.4$ compresse.*

### (ii) Invariance orthogonale, sans algèbre

Une matrice orthogonale $O$ représente une **rotation** de l'espace (éventuellement combinée à un miroir). C'est un changement de repère qui préserve les longueurs et les angles. L'opération $M \mapsto O M O^\top$ revient à **regarder $M$ dans un autre repère** — c'est la même matrice, exprimée dans d'autres coordonnées.

> [!warning] L'invariance orthogonale en une phrase
> Dire que "la loi de $M$ est invariante par $M \mapsto OMO^\top$" signifie : **la distribution de $M$ est la même dans tous les repères**. Aucune direction de l'espace n'est privilégiée.

Conséquence concrète qu'on utilisera : les **vecteurs propres** d'une telle matrice pointent dans des directions complètement uniformes sur la sphère. La matrice est "isotrope" — elle ne sait rien des axes de coordonnées.

> [!example] Exemple à la main — rotation préserve le spectre
> Reprenons la matrice $M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}$. Appliquons une rotation d'angle $\theta = \pi/4$ : $M' = O M O^\top$ avec $O = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$. Les entrées de $M'$ sont différentes — par exemple $M'_{11} \approx 1.78$ au lieu de $2.48$.
>
> Mais ses **valeurs propres sont exactement les mêmes** : $\{0.4, 3.0\}$. Le spectre est invariant par changement de repère. C'est une propriété générale, pas un hasard.
>
> Pour une matrice GOE, ce n'est pas seulement le spectre mais aussi **toute la distribution** de la matrice qui est invariante par $M \mapsto OMO^\top$ : si on rejoue le tirage de $M$ après rotation, on retombe sur la même loi.

![[images/1-Mathématiques/Random Matrix Theory/figB_ellipse.png]]
*Figure B. Le cercle unité est transformé en ellipse par $M$. Les axes de l'ellipse coïncident exactement avec les vecteurs propres, et leur longueur est proportionnelle à la valeur propre correspondante : la valeur propre mesure l'étirement dans la direction propre.*

### (iii) Diagonalisation et changement de base

Pour une matrice **symétrique** $\Sigma$, les vecteurs propres sont toujours orthogonaux entre eux — c'est le théorème spectral. On peut donc former la matrice $P = [v_1 | v_2 | \cdots | v_N]$ dont les colonnes sont les vecteurs propres normalisés : $P$ est une matrice **orthogonale** ($P^{-1} = P^\top$). On obtient alors la décomposition

$$\Sigma = P \Lambda P^\top$$

avec $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$ la matrice diagonale des valeurs propres. C'est la **diagonalisation** de $\Sigma$.

L'intérêt concret : $P^\top x$ est un **changement de base** qui exprime le vecteur $x$ dans le repère des vecteurs propres. Dans ce nouveau repère, $\Sigma$ devient $\Lambda$ — une matrice diagonale. Cela signifie que les composantes de $P^\top x$ sont **décorrélées** : chaque axe capture une source de variance indépendante, de magnitude $\lambda_i$.

> [!warning] Pourquoi c'est central en RMT
> Les matrices de covariance $\Sigma$ sont toujours symétriques. Leur diagonalisation $\Sigma = P\Lambda P^\top$ décompose le risque d'un portefeuille en directions indépendantes — les vecteurs propres sont les **facteurs de risque**, les valeurs propres leur **variance**. La question de RMT est précisément : quand $\Sigma$ est estimée sur des données finies, quels $\lambda_i$ et $v_i$ sont du signal, et lesquels sont du bruit statistique ?

![[images/1-Mathématiques/Random Matrix Theory/figC_diagonalisation.png]]
*Figure C. À gauche : l'ellipse de $\Sigma$ dans le repère original — les axes propres sont obliques. À droite : après le changement de base $P^\top x$, l'ellipse est alignée sur les axes de coordonnées. Les directions sont décorrélées, chaque axe porte une variance $\lambda_i$.*

![[images/1-Mathématiques/Random Matrix Theory/figD_pca_nuage.png]]
*Figure D. À gauche : nuage de données corrélées dans le repère original, avec les vecteurs propres en rouge. À droite : le même nuage après projection $P^\top x$ — les données sont décorrélées et les axes de variance maximale sont alignés sur les axes de coordonnées. C'est exactement ce que fait la PCA.*

## I. La question

Une matrice aléatoire est une matrice dont les entrées sont des variables aléatoires. Question naïve : si on tire une matrice symétrique $M \in \mathbb{R}^{N \times N}$ avec des entrées i.i.d. gaussiennes, à quoi ressemble son spectre ?

Intuition naïve : *"ça doit être aléatoire, donc imprévisible"*. C'est faux.

Quand $N$ grandit, l'histogramme des valeurs propres se range selon une loi limite **déterministe**. Plus $N$ est grand, plus la convergence est forte — c'est un phénomène de concentration. La matrice est aléatoire, mais sa "forme spectrale" ne l'est pas.

Cette observation est née dans deux mondes très éloignés.

> [!example] Deux origines historiques
> **Wigner (1955)** étudie les noyaux atomiques lourds (uranium). Les niveaux d'énergie sont les valeurs propres d'un opérateur hamiltonien trop complexe pour être calculé. Wigner postule : *remplaçons cet hamiltonien par une matrice aléatoire*, et voyons quelles propriétés statistiques émergent. La loi du demi-cercle naît de cette idée.
>
> **Marchenko-Pastur (1967)** étudient un problème de statistique : si on observe $T$ réalisations i.i.d. d'un vecteur $X \in \mathbb{R}^N$ de covariance $\mathbb{I}_N$, la matrice de covariance empirique $\hat{\Sigma}$ n'est pas l'identité — ses valeurs propres sont étalées sur un intervalle. Comment ? Selon une loi déterministe, qui porte leur nom.

C'est cette deuxième origine qui nous intéresse comme quant. La matrice de covariance empirique est *l'objet* central de la gestion de portefeuille, et RMT en révèle les pathologies.

## II. Ensembles classiques

On distingue les matrices aléatoires selon leur **symétrie** et la **distribution** des entrées. Trois ensembles dominent.

### (i) GOE — Gaussian Orthogonal Ensemble

Matrices **symétriques réelles** $M \in \mathbb{R}^{N \times N}$ avec entrées gaussiennes centrées indépendantes (au-dessus de la diagonale), et la contrainte de symétrie $M_{ij} = M_{ji}$.

Pourquoi le mot *orthogonal* ? Parce que la distribution de $M$ est invariante par changement de repère orthogonal $M \mapsto OMO^\top$ (cf. rappel 0.ii). En clair : si tu tires une GOE et qu'un collègue en tire une autre dans un repère tourné, vous ne pouvez pas distinguer les deux statistiquement. C'est la matrice "la plus aléatoire possible" parmi les matrices symétriques.

> [!note]- Détail technique : pourquoi $\text{Var}(M_{ii}) = 2$ et $\text{Var}(M_{ij}) = 1$ ?
> Pour rendre l'invariance orthogonale exacte, il faut calibrer les variances. La convention standard est $M_{ii} \sim \mathcal{N}(0, 2)$ sur la diagonale et $M_{ij} \sim \mathcal{N}(0, 1)$ hors diagonale (pour $i \neq j$).
>
> Le ratio 2:1 sort d'un calcul algébrique : sous une rotation $O$, les éléments diagonaux et hors-diagonaux se mélangent, et seul ce ratio précis fait que la distribution reste inchangée. Les valeurs absolues (1 et 2) sont conventionnelles — on aurait pu prendre (1/2, 1/4) avec une normalisation différente. **Ce qu'il faut retenir : entrées gaussiennes centrées indépendantes, avec une variance calibrée pour l'invariance.**

### (ii) GUE — Gaussian Unitary Ensemble

Variante complexe du GOE : matrices **hermitiennes** $M \in \mathbb{C}^{N \times N}$. C'est l'ensemble le plus étudié en physique (mécanique quantique sans symétrie de renversement du temps). Plus simple à analyser que GOE car la loi jointe des valeurs propres a une forme plus compacte. On ne l'utilisera pas directement — juste à savoir qu'il existe.

### (iii) Wishart — covariance empirique

C'est *notre* ensemble principal. Soit $X \in \mathbb{R}^{N \times T}$ une matrice dont les colonnes sont $T$ vecteurs i.i.d. $\mathcal{N}(0, \mathbb{I}_N)$. La matrice

$$W = \frac{1}{T} X X^\top \in \mathbb{R}^{N \times N}$$

est appelée **matrice de Wishart**. C'est exactement la matrice de covariance empirique d'un échantillon gaussien centré de taille $T$ en dimension $N$.

En finance, $X$ représente $N$ actifs observés sur $T$ jours, et $W = \frac{1}{T} X X^\top$ est la matrice de covariance des rendements. C'est l'objet sur lequel reposent les modèles de portefeuille (Markowitz, Barra, etc.). RMT va nous dire ce qui se passe quand $N$ et $T$ sont comparables — et la réponse est dérangeante.

> [!example] Exemple à la main — Wishart avec $N=3$ actifs, $T=5$ jours
> Imaginons 3 actifs ($N=3$) observés sur 5 jours ($T=5$). On a une matrice de rendements
> $$X = \begin{pmatrix}
> 0.4 & -0.2 & 0.1 & 0.8 & -0.5 \\
> -0.1 & 0.3 & -0.4 & 0.2 & 0.6 \\
> 0.7 & -0.5 & 0.2 & -0.3 & 0.1
> \end{pmatrix}$$
> (chaque ligne = un actif, chaque colonne = un jour). La covariance empirique vaut
> $$W = \frac{1}{T} X X^\top \approx \begin{pmatrix}
> 0.22 & -0.10 & -0.03 \\
> -0.10 & 0.13 & -0.16 \\
> -0.03 & -0.16 & 0.18
> \end{pmatrix}$$
> Ses valeurs propres : $\lambda_1 \approx 0.02$, $\lambda_2 \approx 0.18$, $\lambda_3 \approx 0.33$. **Toutes positives** — c'est forcément le cas pour une matrice $XX^\top$ (on dit qu'elle est *semi-définie positive*).
>
> Ce qui est troublant : la "vraie" covariance des rendements ici est inconnue, mais si en réalité les 3 actifs étaient i.i.d. avec covariance $\mathbb{I}_3$ (donc spectre $\{1, 1, 1\}$), on aurait quand même obtenu trois valeurs propres dispersées comme ci-dessus. **L'estimation crée artificiellement de la dispersion spectrale.** C'est *la* pathologie que Marchenko-Pastur va quantifier.

## III. La normalisation $1/\sqrt{N}$ — pourquoi elle est obligatoire

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
> Reprenons notre matrice 2×2 :
> $$M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}, \quad \text{spectre} : \{0.4, 3.0\}$$
> On divise par $\sqrt{N} = \sqrt{2} \approx 1.414$. Les valeurs propres sont divisées par ce facteur :
> $$\widetilde{M} = M / \sqrt{2} \quad \Longrightarrow \quad \text{spectre} : \{0.28, 2.12\}$$
> Toutes les valeurs propres tombent dans l'intervalle $[-2, 2]$. Si on refaisait l'exercice à $N = 1000$ sans normaliser, $\lambda_{\max}$ serait $\sim 63$ et le spectre s'étalerait sur $[-63, 63]$. Avec normalisation, il reste cantonné dans $[-2, 2]$, peu importe $N$.

> [!example] Analogie : le théorème central limite
> Tu connais ce phénomène ailleurs. Si $X_1, \dots, X_n$ sont i.i.d. centrées de variance 1, alors $S_n = X_1 + \dots + X_n$ a une variance de $n$ et "explose". Pour avoir une limite non-triviale, on regarde $S_n / \sqrt{n}$, qui converge vers $\mathcal{N}(0,1)$.
>
> La normalisation $1/\sqrt{N}$ en RMT joue *exactement* le même rôle : c'est le bon facteur d'échelle pour qu'une matrice aléatoire ait une limite spectrale non-triviale.

Pour Wishart, c'est le facteur $1/T$ (et non $1/\sqrt{T}$) qui joue ce rôle, parce qu'on a un produit $X X^\top$ — c'est-à-dire un *carré*, qui demande une normalisation au carré.

> [!warning] À retenir
> RMT ne dit pas que "tout converge tout seul". Elle dit qu'**à la bonne échelle**, des objets aléatoires révèlent une structure déterministe. Le choix de la normalisation est ce qui définit le régime d'étude.

![[fig1_normalisation.png]]
*Figure 1. À gauche : la plus grande valeur propre d'une GOE non-normalisée croît comme $2\sqrt{N}$ — le spectre n'a pas de limite. À droite : après normalisation par $\sqrt{N}$, $\lambda_{\max}$ se stabilise autour de $2$ quelle que soit la taille.*

## IV. La loi du demi-cercle de Wigner

C'est *le* résultat fondateur. On considère une matrice GOE normalisée $W_N = M / \sqrt{N}$. Soit $\lambda_1, \dots, \lambda_N$ ses valeurs propres.

> [!warning] Théorème (Wigner, 1955)
> Quand $N \to \infty$, la distribution empirique des valeurs propres
> $$\mu_N = \frac{1}{N} \sum_{i=1}^N \delta_{\lambda_i}$$
> converge faiblement vers la **loi du demi-cercle** de densité
> $$\rho_{\text{sc}}(\lambda) = \frac{1}{2\pi} \sqrt{4 - \lambda^2} \, \mathbb{1}_{[-2, 2]}(\lambda)$$

Concrètement : le spectre est étalé sur $[-2, 2]$, avec une densité en forme de demi-disque (d'où le nom). Aucune valeur propre ne dépasse $2$ asymptotiquement, et la majorité s'accumule au centre.

![[fig2_semicircle.png]]
*Figure 2. Histogramme des valeurs propres d'une matrice GOE normalisée pour $N=50$, $N=200$, $N=1000$. La courbe rouge est la densité limite $\rho_{\text{sc}}(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}$. La convergence est visible dès $N=200$, et excellente à $N=1000$.*

### Intuition par les moments

Pourquoi un demi-cercle, et pas une gaussienne ? L'idée vient du calcul des moments. La $k$-ième moment de $\mu_N$ est

$$\int \lambda^k \, d\mu_N(\lambda) = \frac{1}{N} \sum_i \lambda_i^k = \frac{1}{N} \, \text{tr}(W_N^k)$$

Or $\text{tr}(W_N^k) = \sum_{i_1, \dots, i_k} W_{i_1 i_2} W_{i_2 i_3} \cdots W_{i_k i_1}$. Chaque terme est un **chemin fermé** dans le graphe complet à $N$ sommets, de longueur $k$. En espérance, seuls les chemins où chaque arête est parcourue un nombre **pair** de fois survivent (car les entrées sont centrées). Une combinatoire fine montre que les chemins dominants, à $N$ grand, correspondent aux **arbres**, dont le nombre est compté par les **nombres de Catalan** :

$$\lim_{N \to \infty} \mathbb{E}\!\left[ \frac{1}{N} \text{tr}(W_N^{2k}) \right] = C_k = \frac{1}{k+1}\binom{2k}{k}$$

et les moments impairs tendent vers $0$. Ces moments-là sont exactement les moments de la loi du demi-cercle. La loi est entièrement caractérisée par ses moments, donc CQFD (modulo la rigueur sur la convergence faible).

> [!note]- Pourquoi les Catalan ?
> Les nombres de Catalan comptent un grand nombre d'objets combinatoires : chemins de Dyck, parenthésages, arbres binaires, triangulations… Tous ces objets vérifient la même récurrence $C_{k+1} = \sum_{i=0}^k C_i C_{k-i}$. Cette récurrence se traduit, côté densité, en une équation algébrique pour la transformée de Stieltjes — équation dont la solution donne précisément $\rho_{\text{sc}}$. C'est l'idée centrale qu'on retrouvera pour Marchenko-Pastur.

## V. Bulk vs edge

Le spectre d'une matrice aléatoire se décompose en deux régimes qu'il faut distinguer rigoureusement.

> [!warning] Bulk et edge
> Le **bulk** désigne la zone où la densité spectrale limite est non nulle — pour GOE, l'intervalle $(-2, 2)$. C'est là que vit la majorité des valeurs propres, dont la distribution est gouvernée par le semicircle.
>
> L'**edge** désigne les valeurs propres extrêmes — typiquement $\lambda_{\max}$ et $\lambda_{\min}$. Asymptotiquement, $\lambda_{\max} \to 2$ et $\lambda_{\min} \to -2$. Mais à $N$ fini, $\lambda_{\max}$ fluctue autour de $2$ avec une amplitude de l'ordre de $N^{-2/3}$, et la distribution de ces fluctuations est la **loi de Tracy-Widom**.

Cette distinction est *cruciale* pour la finance : le bulk correspond au "bruit" de la matrice de covariance empirique (les valeurs propres parasites créées par l'estimation), et l'edge correspond aux "vrais" facteurs (les valeurs propres qui sortent du bulk car associées à un signal réel). La séparation entre les deux est un test statistique — c'est l'objet de la transition BBP (note 3).

| Régime | Échelle de fluctuation | Loi limite | Application |
|---|---|---|---|
| Bulk | $1/N$ | Sine kernel (universel) | Densité globale, MP |
| Edge | $N^{-2/3}$ | Tracy-Widom | Détection de signal |

![[fig3_edge.png]]
*Figure 3. Distribution de $\lambda_{\max}$ pour $N=200$ et $N=1000$ (sur 5000 tirages), centrée et rescalée par $N^{-2/3}$. Les deux histogrammes se superposent sur la même courbe limite : la **loi de Tracy-Widom** (note 3).*

## VI. Outils — densité spectrale et transformée de Stieltjes

Deux objets reviennent partout en RMT. Autant les poser proprement maintenant.

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
*Figure 4. Partie imaginaire de la transformée de Stieltjes empirique $g_N(\lambda + i\varepsilon)$ (avec $\varepsilon = 0.01$) pour $N=500$. La courbe $\frac{1}{\pi}\text{Im}\,g_N(\lambda + i\varepsilon)$ reconstitue exactement la densité du demi-cercle — illustration concrète de la formule d'inversion.*

## Ce qui suit

La loi du demi-cercle est satisfaisante mathématiquement, mais ses applications finance sont indirectes. Le résultat *réellement* utile en quant est son cousin pour les matrices de Wishart — la **loi de Marchenko-Pastur** — qui décrit le spectre des matrices de covariance empiriques. C'est l'objet de la prochaine note.
