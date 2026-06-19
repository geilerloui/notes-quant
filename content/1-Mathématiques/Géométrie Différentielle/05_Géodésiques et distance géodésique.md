---
title: Géodésiques et distance géodésique
date: 2026-05-11
tags:
  - mathématiques
  - géométrie-différentielle
  - géodésique
  - distance
---

## L'idée fondatrice

Dans les notes précédentes, on a défini implicitement la **longueur d'une courbe** sur une variété :

$$\ell(\gamma) = \int_a^b \|\gamma'(t)\|_{\gamma(t)} \, dt$$

et la **distance entre deux points** comme la longueur de la plus courte courbe les reliant. Cette plus courte courbe, c'est la **géodésique** — l'analogue de la "ligne droite" sur une variété.

Cette note finalise ce qu'on avait laissé en suspens : *qu'est-ce qu'une géodésique en pratique, comment la calcule-t-on, et à quoi sert la distance géodésique* ?

Ce sont des concepts essentiels pour :
- **L'information geometry** (distance entre distributions via la métrique de Fisher)
- **Le manifold learning** (Isomap, k-means géodésique)
- **La statistique sur variétés** (moyenne de Fréchet, ACP géodésique)
- **La finance** (distance entre matrices de covariance SPD)

## I. Géodésique : la "ligne droite" sur une variété

### Intuition

Sur $\mathbb{R}^n$, la plus courte courbe entre deux points est la **ligne droite**. Mais sur une variété courbe, on ne peut pas tracer de ligne droite — il faut rester sur la variété. La géodésique est ce qu'on obtient quand on essaie de tracer une "ligne droite" tout en restant sur $M$.

**Exemple parlant — la sphère.** Si tu veux aller de Paris à New York sur la Terre, tu ne creuses pas un tunnel droit (ce serait la "ligne droite" euclidienne dans $\mathbb{R}^3$). Tu suis un **grand cercle** : c'est la géodésique sur la sphère.

![[Pasted image 20260511172219.png|263]]
*Figure 1. Entre deux points $x$ et $y$ sur la sphère, la **corde euclidienne** (rouge, droite dans $\mathbb{R}^3$) est plus courte mais sort de la sphère. La **géodésique** (vert) est un arc de grand cercle — la plus courte courbe qui reste sur la sphère.*

### Définition

Une **géodésique** sur une variété riemannienne $M$ est une courbe $\gamma : I \to M$ qui :

1. **Minimise localement la longueur** entre deux points proches
2. A une **vitesse de norme constante** : $\|\gamma'(t)\|_{\gamma(t)} = \text{cst}$

> [!note]- Pourquoi "vitesse constante" ?
> Sur $\mathbb{R}^n$, une ligne droite parcourue à vitesse constante est une géodésique ; la même ligne mais parcourue à vitesse variable a la même image mais n'est plus géodésique au sens "paramétrée par la longueur d'arc". La condition $\|\gamma'\|$ constante normalise la paramétrisation. C'est l'analogue d'un mouvement rectiligne **uniforme**.

**Équation des géodésiques.** Une géodésique satisfait une équation différentielle (l'**équation géodésique**) qui dit en substance : "à chaque instant, la composante tangente de l'accélération est nulle". Pour une variété plongée, ça s'écrit :

$$\gamma''(t) \perp T_{\gamma(t)} M$$

C'est-à-dire que l'accélération $\gamma''(t)$ n'a que des composantes **normales** à la variété. Intuitivement : on ne "tourne pas" *dans* la variété, seule la contrainte de rester sur $M$ nous fait dévier.

## II. L'exponentielle riemannienne

Pour faire le lien avec la note 4 sur la rétraction, on introduit un objet central : l'**exponentielle**.

**Définition.** L'**exponentielle riemannienne** en $x$ est l'application

$$\exp_x : T_x M \to M$$

définie par $\exp_x(v) = \gamma_v(1)$, où $\gamma_v$ est l'unique géodésique vérifiant $\gamma_v(0) = x$ et $\gamma_v'(0) = v$.

**Interprétation.** Tu pars de $x$, tu lances une fourmi en ligne droite (sur la variété) avec vitesse initiale $v$, et tu regardes où elle se trouve au bout d'**une unité de temps**. C'est $\exp_x(v)$.

**Lien avec la rétraction.** L'exponentielle est *la* rétraction "idéale" — celle qui suit la vraie géodésique. Mais comme on l'a vu dans la note 4, on lui préfère souvent une rétraction plus simple (normalisation sur la sphère, QR sur Stiefel) parce que l'exponentielle peut coûter cher à calculer.

| Variété | Exponentielle $\exp_x(v)$ | Rétraction (alternative pratique) |
| :--- | :--- | :--- |
| Sphère $S^{n-1}$ | $\cos(\|v\|) \cdot x + \sin(\|v\|) \cdot \tfrac{v}{\|v\|}$ | $\tfrac{x+v}{\|x+v\|}$ |
| Stiefel $V_F(\mathbb{R}^D)$ | exponentielle matricielle (chère) | $\text{qr}(A+V).Q$ |
| SPD$(n)$ (aff.-inv.) | $X^{1/2} \exp(X^{-1/2} V X^{-1/2}) X^{1/2}$ | $X+V$ |

**Sphère en détail.** La formule $\exp_x(v) = \cos(\|v\|) x + \sin(\|v\|) \tfrac{v}{\|v\|}$ se lit ainsi : tu pars de $x$, tu suis le grand cercle dans la direction $v$, et après une distance $\|v\|$ tu arrives sur ce point. C'est *exactement* le déplacement d'un avion qui suit un grand cercle.

## III. La distance géodésique

### Définition

La **distance géodésique** entre $x, y \in M$ est la longueur de la plus courte géodésique les reliant :

$$d_M(x, y) = \inf_{\gamma : x \to y} \ell(\gamma)$$

C'est l'analogue intrinsèque de la distance euclidienne, et elle vérifie les axiomes d'une distance : symétrie, séparation, inégalité triangulaire.

### Formules pour nos variétés

**Sphère $S^{n-1}$.** La distance géodésique est l'**angle** entre les deux vecteurs :

$$\boxed{d_{S^{n-1}}(x, y) = \arccos(\langle x, y \rangle)}$$

Concrètement, pour deux points sur la Terre, c'est l'angle au centre — il suffit de le multiplier par le rayon terrestre pour avoir la distance en kilomètres.

> [!example] Exemple chiffré : Paris ↔ New York
> Coordonnées approximatives sur une sphère unité (latitude/longitude → vecteur 3D) :
> - Paris : $x \approx (0.65, 0.05, 0.76)$
> - New York : $y \approx (-0.18, -0.74, 0.65)$
>
> Produit scalaire : $\langle x, y\rangle \approx 0.39$
>
> Distance angulaire : $\arccos(0.39) \approx 1.17$ rad
>
> Multiplié par le rayon terrestre (6371 km) : ≈ **7460 km**. Conforme à la réalité.
>
> En distance euclidienne (la "corde") : $\|x - y\| \approx 1.10$, ce qui correspond à 7010 km (le tunnel imaginaire à travers la Terre). Plus court, mais inutilisable.

**SPD$(n)$ avec métrique affine-invariante.** Pour deux matrices SPD $X, Y$ :

$$\boxed{d_{\text{SPD}}(X, Y) = \|\log(X^{-1/2} Y X^{-1/2})\|_F}$$

où $\log$ est le **logarithme matriciel** et $\|\cdot\|_F$ est la norme de Frobenius. Cette distance est *invariante* par congruence : $d(GXG^\top, GYG^\top) = d(X, Y)$ pour toute matrice inversible $G$. Très utile pour comparer des matrices de covariance.

**Stiefel $V_F(\mathbb{R}^D)$.** La distance géodésique n'a pas de formule fermée simple (sauf cas particuliers comme $F=1$ qui redonne la sphère). En pratique, on l'estime numériquement ou on utilise des approximations comme la distance dans la métrique de Frobenius après alignement.

![[Pasted image 20260511172201.png]]
*Figure 2. Sur la sphère, la distance géodésique $d_{S^2}(x,y) = \arccos(\langle x,y\rangle)$ est la longueur de l'arc de grand cercle (vert). La distance euclidienne $\|x-y\|$ est la longueur de la corde (rouge). Pour deux points proches, les deux distances sont presque égales ; pour des points antipodaux ($\langle x,y\rangle = -1$), la distance géodésique vaut $\pi$ alors que la distance euclidienne ne vaut que 2.*

### Distance géodésique vs distance euclidienne

| Type | Mesure | Reste sur la variété ? | Quand l'utiliser ? |
| :--- | :--- | :--- | :--- |
| Distance euclidienne (corde) | $\|x - y\|$ dans $\mathbb{R}^n$ | Non | Petits déplacements, approximation rapide |
| Distance géodésique | longueur de la plus courte courbe sur $M$ | Oui | Vraie distance respectant la géométrie de $M$ |

**Pour des points proches**, les deux distances sont quasi égales (la variété est "presque plate" localement). **Pour des points éloignés**, elles divergent — la distance euclidienne peut être très trompeuse.

## IV. Applications

### Statistique sur variétés — moyenne de Fréchet

Sur $\mathbb{R}^n$, la moyenne arithmétique $\bar{x} = \tfrac{1}{N}\sum_i x_i$ minimise $\sum_i \|x - x_i\|^2$. C'est la définition variationnelle de la moyenne.

Sur une variété, on ne peut pas additionner des points directement. On généralise par :

$$\bar{x}_{\text{Fréchet}} = \arg\min_{x \in M} \sum_{i=1}^N d_M(x, x_i)^2$$

C'est la **moyenne de Fréchet**. Sur la sphère, c'est le point qui minimise la somme des carrés des angles aux points donnés. Sur SPD, c'est la "moyenne géométrique" — beaucoup plus pertinente qu'une moyenne arithmétique des matrices de covariance.

### Manifold learning — Isomap

Le **manifold learning** suppose qu'un nuage de points $\{x_1, \dots, x_N\} \subset \mathbb{R}^D$ vit sur une variété inconnue $M$ de dimension $d \ll D$. Le but : trouver une représentation bas-dimensionnelle qui **préserve les distances géodésiques**.

**Isomap** procède en deux étapes :
1. **Estimer la distance géodésique** $d_M(x_i, x_j)$ via un graphe de voisinages : on connecte chaque point à ses $k$ plus proches voisins, et on calcule des plus courts chemins dans ce graphe.
2. **MDS** (Multidimensional Scaling) pour trouver un plongement dans $\mathbb{R}^d$ qui préserve ces distances.

Le succès d'Isomap repose sur l'estimation de la **vraie** distance géodésique — pas la distance euclidienne — qui capture la géométrie *intrinsèque* du nuage.

### Information geometry — distance entre distributions

L'espace des distributions de probabilité $\{p_\theta\}$ est une variété riemannienne (la **métrique de Fisher** est le produit scalaire). La distance géodésique correspondante mesure la **dissimilarité** entre deux distributions de manière intrinsèque.

Pour la famille gaussienne $\mathcal{N}(\mu, \sigma^2)$ par exemple, la distance de Fisher entre $(\mu_1, \sigma_1)$ et $(\mu_2, \sigma_2)$ ressemble à la distance hyperbolique sur le demi-plan de Poincaré — ce qui révèle une géométrie très différente de la distance euclidienne sur les paramètres.

### Finance — distance entre covariances

En modélisation du risque, on travaille avec des matrices de covariance $\Sigma_t$ qui évoluent dans le temps. Pour comparer la **structure** des covariances entre dates, deux options :

- **Distance de Frobenius** $\|\Sigma_1 - \Sigma_2\|_F$ : extrinsèque, n'utilise pas la structure SPD. Sensible aux unités, pas invariante.
- **Distance géodésique sur SPD** : intrinsèque, invariante par changement de base, capture mieux la similarité structurelle.

Utile pour : clustering de régimes de marché, détection de breakpoints, interpolation de covariances.

## V. Trois idées à retenir

1. **Une géodésique, c'est la "plus courte courbe" reliant deux points en restant sur la variété**. C'est l'analogue d'une ligne droite. L'**exponentielle** $\exp_x(v)$ donne l'endroit où on arrive en suivant la géodésique de vitesse initiale $v$ pendant une unité de temps.

2. **La distance géodésique** $d_M(x, y)$ est la longueur de la plus courte géodésique. Pour la sphère : $\arccos(\langle x, y\rangle)$. Pour SPD : norme du log matriciel de $X^{-1/2} Y X^{-1/2}$.

3. **C'est l'outil de base pour faire de la statistique, du ML ou de la finance sur variété** : moyenne de Fréchet, Isomap, comparaison de covariances. La distance euclidienne ambient ne suffit pas dès que la variété est courbée.

## VI. Vers la suite

Avec cette note, le corpus de géométrie différentielle pour ML/finance est complet :

1. Variétés plongées (qu'est-ce qu'une variété)
2. Vecteurs tangents (directions admissibles)
3. Métrique et gradient riemannien (descente)
4. Rétraction et descente sur variété (algorithme complet)
5. Géodésiques et distance géodésique (mesure intrinsèque) ← *cette note*

Prochaines directions naturelles :
- **Note pratique** : implémentation du challenge QRT avec pymanopt
- **Information geometry** : métrique de Fisher, natural gradient, familles exponentielles, divergences
- **Géométrie de SPD** : métrique affine-invariante, applications en estimation de covariance
- **Manifold learning** : Isomap, LLE, Laplacian eigenmaps (mélange géométrie + théorie spectrale des graphes)
