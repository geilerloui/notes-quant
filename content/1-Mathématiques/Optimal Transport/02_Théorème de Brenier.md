---
title: Théorème de Brenier
date: 2026-05-11
tags: [mathématiques, transport-optimal, brenier, cyclical-monotonicity]
---

## L'idée fondatrice

La note précédente a montré que **Kantorovich résout les problèmes de Monge** (existence, convexité, mass splitting). Mais Kantorovich nous donne un **couplage** $\gamma$, pas une **application** $T$ — on a perdu en chemin la simplicité géométrique de Monge.

Question : dans quels cas l'optimum de Kantorovich est-il en fait **concentré sur le graphe d'une application** $T$ ? Autrement dit, quand la solution généralisée de Kantorovich est-elle aussi une vraie solution de Monge ?

Le résultat fondateur de **Brenier (1991)** répond : pour le **coût quadratique** $c(x, y) = \|x - y\|^2$ et des distributions à densité (continues), l'optimum est unique et s'écrit

$$T = \nabla \varphi$$

avec $\varphi$ une **fonction convexe**.

Ce résultat est central parce qu'il relie le transport optimal à la théorie convexe (qu'on connaît bien : gradients, sous-différentiel, transformée de Legendre, etc.). Cette note construit le théorème en plusieurs étapes :

1. Le cas 1D où on montre que la map optimale est **monotone** (preuve par permutation)
2. Une construction explicite en 1D via les **CDFs** : $T = G^{-1} \circ F$
3. La généralisation à $\mathbb{R}^n$ : la **cyclical monotonicity**
4. Le théorème de **Rockafellar** : cycliquement monotone $\Leftrightarrow$ gradient d'une fonction convexe
5. Le théorème de **Brenier** comme synthèse

## I. Le cas 1D : la map optimale "T" est monotone

On se place en dimension 1, avec $\mu, \nu$ deux distributions de densités $f, g$ "assez régulières". On veut résoudre le problème de Monge pour le coût quadratique :

$$\inf_{T_\# \mu = \nu} \int_\mathbb{R} (x - T(x))^2 \, f(x) \, dx$$

**Résultat** : la map optimale est **monotone croissante**. C'est-à-dire que si $x_1 < x_2$ alors $T(x_1) < T(x_2)$.

Intuitivement : si la map optimale "croisait" deux points (envoyer un point de gauche très à droite et un point de droite à gauche), on pourrait **échanger** les deux assignations pour diminuer le coût. C'est exactement l'argument formel.

> [!note]- Preuve par permutation
> Soient $x_1 < x_2$ et supposons par l'absurde que $y_1 = T(x_1)$ et $y_2 = T(x_2)$ vérifient $y_1 > y_2$ (la map "croise"). On va échanger les destinations sur de petits voisinages et montrer que le coût diminue strictement.
> 
> **Construction.** On choisit deux petits intervalles ouverts $I_1 \ni x_1$ et $I_2 \ni x_2$ tels que
> 
> $$\int_{I_1} f(x) \, dx = \varepsilon = \int_{I_2} f(x) \, dx$$
> 
> et on définit une nouvelle application $\tilde{T}$ qui **échange** les images sur ces intervalles :
> 
> $$\tilde{T}(x) = \begin{cases} T(x) + (y_2 - y_1) & \text{si } x \in I_1 \\ T(x) + (y_1 - y_2) & \text{si } x \in I_2 \\ T(x) & \text{sinon} \end{cases}$$
> 
> Par construction, $\tilde{T}$ est encore measure-preserving ($\tilde{T}_\# \mu = \nu$) — on a juste permuté qui va où.
> 
> **Comparaison des coûts.** Comme $T$ est supposée optimale :
> 
> $$\int_\mathbb{R} (x - T(x))^2 f(x) \, dx \leq \int_\mathbb{R} (x - \tilde{T}(x))^2 f(x) \, dx$$
> 
> En développant les carrés et en remarquant que les intégrales de $x^2 f$ et $T(x)^2 f$ (resp. $\tilde{T}(x)^2 f$) coïncident (puisque $T_\# \mu = \tilde{T}_\# \mu = \nu$), il ne reste que les termes croisés sur $I_1 \cup I_2$ :
> 
> $$-\int_{I_1} x T(x) f \, dx - \int_{I_2} x T(x) f \, dx \leq -\int_{I_1} x \tilde{T}(x) f \, dx - \int_{I_2} x \tilde{T}(x) f \, dx$$
> 
> En divisant par $\varepsilon$ et en faisant $\varepsilon \to 0$ (les intervalles se contractent autour de $x_1$ et $x_2$, et $T(x) \to y_i$ sur $I_i$), on obtient :
> 
> $$x_1 (y_2 - y_1) + x_2 (y_1 - y_2) \leq 0$$
> 
> $$\Rightarrow (x_2 - x_1)(y_2 - y_1) \geq 0$$
> 
> Donc $x_1 < x_2 \Rightarrow y_1 \leq y_2$. La map est croissante.
> 
> ![[images/1-Mathématiques/Optimal transport/im6.png|326]]
> *Illustration de la preuve par permutation : si la map "croise" (envoie $x_1$ à droite et $x_2$ à gauche alors que $x_1 < x_2$), on peut **échanger** les destinations pour diminuer le coût total. La map optimale ne croise donc jamais — elle est monotone croissante.*

**Interprétation géométrique.** Sur la droite réelle, "transporter optimalement" $\mu$ vers $\nu$ revient à les **trier ensemble** : le $k$-ième percentile de $\mu$ va sur le $k$-ième percentile de $\nu$.

> [!warning] Attention au piège : $T$ agit point par point, pas par "morceaux"
> Tentation naïve : "je prends un bout du tas de 10 d'épaisseur et je le mets dans le trou de capacité 20, donc il reste 10 à remplir". Mais $T$ envoie **chaque point** $x$ vers **un seul point** $T(x)$. Ce qui s'ajuste, c'est la **densité locale** : un intervalle infinitésimal $[x, x + dx]$ envoyé sur $[T(x), T(x) + T'(x)\, dx]$ a sa densité multipliée par $1/T'(x)$. Si $T'(x) > 1$ (la map "étire"), la densité diminue. Si $T'(x) < 1$ (la map "comprime"), la densité augmente. C'est cette élasticité locale qui permet à $T$ de transformer $\mu$ en $\nu$ même si les densités sont différentes.

> [!example] Exemple 1 : translation pure (densités identiques décalées)
> Tas uniforme sur $[0, 10]$ (densité 10) et trou uniforme sur $[5, 15]$ (densité 10 aussi). Total = 100 dans les deux cas.
> 
> Les quantiles :
> 
> | Quantile | $\mu$ : point dans le tas | $\nu$ : point dans le trou | $T$ envoie où |
> | :--- | :---: | :---: | :---: |
> | 0% | $x = 0$ | $y = 5$ | $T(0) = 5$ |
> | 30% | $x = 3$ | $y = 8$ | $T(3) = 8$ |
> | 50% | $x = 5$ | $y = 10$ | $T(5) = 10$ |
> | 100% | $x = 10$ | $y = 15$ | $T(10) = 15$ |
> 
> Donc $T(x) = x + 5$ : **translation pure**, $T'(x) = 1$ partout. La densité ne change pas localement (1/$T'$ = 1), et tout le trou est rempli uniformément.

> [!example] Exemple 2 : dilatation (densités différentes)
> Tas uniforme sur $[0, 10]$ (densité 10), trou uniforme sur $[0, 20]$ (densité 5). Total = 100 dans les deux cas.
> 
> Les quantiles :
> 
> | Quantile | Tas | Trou | $T$ envoie |
> | :--- | :---: | :---: | :---: |
> | 0% | $x = 0$ | $y = 0$ | $T(0) = 0$ |
> | 25% | $x = 2.5$ | $y = 5$ | $T(2.5) = 5$ |
> | 50% | $x = 5$ | $y = 10$ | $T(5) = 10$ |
> | 100% | $x = 10$ | $y = 20$ | $T(10) = 20$ |
> 
> Donc $T(x) = 2x$ : **dilatation par 2**, $T'(x) = 2$ partout. La densité est divisée par 2 : la matière étalée sur une largeur double a une densité moitié moindre. Vérification : densité du tas 10, divisée par $T'(x) = 2$, donne densité du trou 5. ✓
> 
> **C'est exactement ça qui répond à l'intuition "le trou est plus large que ce que j'apporte"** : $T$ peut **étirer** localement la matière, ce qui ajuste automatiquement la densité.

## II. Construction explicite du "T" via les CDFs

En 1D, on peut **construire explicitement** la map optimale via les fonctions de répartition (CDFs).

Notons :

$$F(x) = \int_{-\infty}^x f(t) \, dt, \qquad G(y) = \int_{-\infty}^y g(t) \, dt$$

les CDFs de $\mu$ et $\nu$. Ce sont des fonctions croissantes de $\mathbb{R}$ vers $[0, 1]$.

**Idée.** Puisque la map optimale est monotone, elle doit préserver les **percentiles** : la masse de $\mu$ avant $x$ (c'est-à-dire $F(x)$) doit être égale à la masse de $\nu$ avant $T(x)$ (c'est-à-dire $G(T(x))$) :

$$F(x) = G(T(x))$$

D'où l'expression explicite :

$$\boxed{T(x) = G^{-1}(F(x))}$$

![[brenier_1d_cdf.png|566]]
*Construction de la map optimale en 1D via les CDFs. On lit l'altitude $u = F(x)$ sur la courbe rouge ($F$ = CDF de $\mu$), puis on redescend sur la courbe bleue ($G$ = CDF de $\nu$) pour trouver le point $T(x) = G^{-1}(u)$.*

> [!example] Exemple : deux gaussiennes
> Si $\mu = \mathcal{N}(0, 1)$ et $\nu = \mathcal{N}(m, \sigma^2)$, on a $G^{-1}(u) = m + \sigma F^{-1}(u)$ (où $F$ est la CDF de la gaussienne centrée réduite). En combinant :
> 
> $$T(x) = m + \sigma x$$
> 
> Pour transporter une gaussienne sur une autre gaussienne (en 1D), il suffit de **translater et dilater**. C'est intuitif : on ne déforme pas la "forme" gaussienne, on la rescale.

**Une astuce qui marche en pratique** : ce résultat 1D est utilisé en géophysique pour comparer des données à des simulations. On découpe l'espace en tranches 1D, on applique $G^{-1} \circ F$ sur chaque tranche, et on agrège. C'est rapide et donne une bonne approximation du transport optimal en dimension supérieure (cas du **sliced Wasserstein**, mentionné en note 03).

## III. La généralisation : cyclical monotonicity

En dimension $n \geq 2$, on ne peut plus utiliser la formule $T = G^{-1} \circ F$ : il n'y a pas d'ordre naturel sur $\mathbb{R}^n$. Mais on peut **généraliser l'argument de permutation** à $N$ points au lieu de 2.

### L'argument de permutation à N points

Soient $x_1, \dots, x_N \in \Omega_s$ et leurs images $y_i = T(x_i)$ par la map optimale. On peut tenter de **permuter cycliquement** les destinations : envoyer $x_i$ vers $y_{i+1}$ (indice modulo $N$). Comme dans le cas 1D, cette permutation reste measure-preserving si on l'applique sur de petits voisinages.

> [!note]- Preuve de l'inégalité cyclique
> On considère $N$ points $x_1, \dots, x_N$ et de petites boules $E_i \ni x_i$ telles que $\int_{E_i} f \, dx = \varepsilon$ pour tout $i$. La permutation circulaire envoie $E_i$ sur $F_{i+1} = T(E_{i+1})$ (au lieu de $F_i$).
> 
> L'optimalité de $T$ pour le coût quadratique donne :
> 
> $$\sum_{i=1}^N \int_{E_i} (x - T(x))^2 f \, dx \leq \sum_{i=1}^N \int_{E_i} (x - \tilde{T}(x))^2 f \, dx$$
> 
> En développant et en simplifiant les termes invariants (mêmes arguments qu'en 1D), il reste :
> 
> $$\sum_{i=1}^N \int_{E_i} x \cdot (\tilde{T}(x) - T(x)) f \, dx \leq 0$$
> 
> En passant à la limite $\varepsilon \to 0$ (les boules se contractent autour de $x_i$, $T \to y_i$ sur $E_i$, et $\tilde{T} \to y_{i+1}$) :
> 
> $$\sum_{i=1}^N x_i \cdot (y_{i+1} - y_i) \leq 0$$

C'est la condition de **cyclical monotonicity** :

$$\boxed{\sum_{i=1}^N \langle x_i, y_{i+1} - y_i \rangle \leq 0 \qquad \forall (x_1, y_1), \dots, (x_N, y_N) \in \text{graph}(T)}$$

### Cas $N = 2$ : l'angle aigu

Pour $N = 2$, l'inégalité se réécrit (cf. preuve en callout) :

$$\langle x_2 - x_1, \, y_2 - y_1 \rangle \geq 0$$

C'est-à-dire que les vecteurs $\vec{x_1 x_2}$ et $\vec{y_1 y_2}$ font un **angle aigu** (produit scalaire positif).

![[Pasted image 20260512150021.png]]
*Cyclical monotonicity en dimension supérieure, cas $N=2$. Les vecteurs $\vec{x_1 x_2}$ (rouge, dans le domaine source) et $\vec{y_1 y_2}$ (bleu, dans le domaine cible) doivent faire un angle aigu. Géométriquement : la map ne peut pas "tordre" la masse, elle peut seulement la déplacer dans des directions cohérentes.*

C'est la généralisation directe de la monotonie 1D ($x_1 < x_2 \Rightarrow y_1 \leq y_2$, qui est équivalent à $(x_2 - x_1)(y_2 - y_1) \geq 0$).

> [!note]- Pourquoi "cyclical" ?
> Pour $N = 2$, l'inégalité s'écrit $\langle x_2 - x_1, y_2 - y_1 \rangle \geq 0$. Pour $N = 3, 4, \dots$, l'inégalité est plus subtile : on **doit la vérifier pour toute permutation cyclique** des indices. Contrairement à la "monotonie" qui ne regarde que les paires, la cyclical monotonicity contrôle **toutes** les permutations possibles — c'est strictement plus fort.
> 
> En dimension 1, monotonie et cyclical monotonicity coïncident. Mais en dimension $\geq 2$, il existe des maps "monotones par paires" qui ne sont pas cycliquement monotones — elles violent l'inégalité dès $N = 3$.

### Lecture algorithmique : élimination de sommets du polytope

La cyclical monotonicity a une **interprétation algorithmique très éclairante** quand on la relie au [polytope de Birkhoff vu en note 01](01_Problème%20de%20Monge%20et%20Kantorovich.md).

Dans l'exemple visuel 3×3, on avait **6 sommets** $\sigma_1, \ldots, \sigma_6$ correspondant aux 6 permutations. Chacun est un candidat à l'optimum, mais on doit calculer le coût pour chacun pour trouver le meilleur.

**La cyclical monotonicity est un filtre** qui élimine directement des sommets sans avoir à calculer leur coût :

| Candidat $\sigma_i$ | Test cyclical monotone | Statut |
| :--- | :---: | :--- |
| $\sigma_i$ ne crée pas de croisement | ✓ | Candidat valide à l'optimum |
| $\sigma_i$ crée un croisement | ✗ | **Éliminé** : on peut faire mieux en échangeant |

Concrètement : si une permutation $\sigma_i$ envoie $x_1 \to y_a$ et $x_2 \to y_b$ avec $\langle x_2 - x_1, y_b - y_a \rangle < 0$ (angle obtus), alors on **sait sans calcul** que ce n'est pas l'optimum — il suffirait d'échanger les destinations ($x_1 \to y_b$, $x_2 \to y_a$) pour diminuer le coût.

**Conséquence forte** : au lieu d'explorer les $n!$ sommets du polytope, on ne regarde que les **sommets cycliquement monotones**, ce qui réduit drastiquement l'espace de recherche. Le théorème de Brenier (section V) renforce encore ça : pour le coût quadratique, **l'unique** sommet cycliquement monotone est l'optimum.

> [!example]- Lien avec les conditions de KKT
> Cette interprétation correspond exactement aux **conditions d'optimalité** que tu connais en programmation linéaire :
> 
> | Concept LP | Équivalent OT |
> | :--- | :--- |
> | Région faisable | Polytope $\Pi(\mu, \nu)$ |
> | Conditions de KKT (nécessaire d'optimalité) | Cyclical monotonicity |
> | Sommets candidats à l'optimum | Permutations cycliquement monotones |
> | Minimisation de la fonction objectif | Choisir parmi les candidats celui qui minimise $\langle C, P \rangle$ |
> 
> En LP classique, KKT te dit "si l'optimum existe, il vérifie ces équations". En OT, la cyclical monotonicity te dit "si $P$ est optimal, alors aucun cycle ne crée de croisement". Même logique : on **caractérise** l'optimum par une condition nécessaire qui filtre l'espace.

## IV. Théorème de Rockafellar

### La transition : on a une condition nécessaire, mais pas exploitable

Faisons le point sur ce qu'on a obtenu :

- **En 1D** (section I-II) : la map optimale est monotone, et on a la formule explicite $T = G^{-1} \circ F$. Concret, calculable.
- **En 2D+** (section III) : la map optimale est cycliquement monotone. C'est une **condition nécessaire** d'optimalité, mais **abstraite** : elle dit "pour tout $N$ et toute permutation cyclique, l'inégalité $\sum \langle x_i, y_{i+1} - y_i \rangle \leq 0$ doit tenir". Difficile à tester en pratique, et surtout difficile de **construire** une telle map.

Ce qu'on cherche : une **forme exploitable** de la cyclical monotonicity. Quelque chose qui nous permette de paramétrer les maps optimales par un objet mathématique bien connu.

C'est exactement ce que fait le théorème de Rockafellar : il transforme une condition combinatoire (toutes les permutations cycliques) en une condition analytique (convexité d'une fonction).

### Première observation : les gradients convexes sont cycliquement monotones

Commençons par le sens "facile" : si on prend une fonction convexe $\varphi : \mathbb{R}^n \to \mathbb{R}$ différentiable, alors son gradient $\nabla \varphi$ est automatiquement cycliquement monotone.

> [!note]- Preuve (sens facile)
> Rappel : pour une fonction convexe différentiable, on a l'inégalité de **tangence sous le graphe** :
> 
> $$\varphi(y) \geq \varphi(x) + \langle \nabla \varphi(x), y - x \rangle \quad \forall x, y$$
> 
> Appliquons cette inégalité aux points $x_i, x_{i+1}$ de notre cycle :
> 
> $$\varphi(x_{i+1}) \geq \varphi(x_i) + \langle \nabla \varphi(x_i), x_{i+1} - x_i \rangle$$
> 
> En sommant sur le cycle ($i = 1, \ldots, N$ avec indices modulo $N$), le membre de gauche **télescope** :
> 
> $$\sum_{i=1}^N \varphi(x_{i+1}) = \sum_{i=1}^N \varphi(x_i)$$
> 
> Donc :
> 
> $$0 \geq \sum_{i=1}^N \langle \nabla \varphi(x_i), x_{i+1} - x_i \rangle$$
> 
> En posant $y_i = \nabla \varphi(x_i)$ et en réarrangeant, c'est exactement la condition cyclical monotone.

**Conclusion** : tout gradient d'une fonction convexe est cycliquement monotone. C'est gratuit, c'est juste la convexité.

### Le théorème de Rockafellar : la réciproque

Rockafellar (1966) a montré le résultat **inverse**, qui est beaucoup plus profond :

> [!quote] Théorème (Rockafellar, 1966)
> Un ensemble $\Gamma \subset \mathbb{R}^n \times \mathbb{R}^n$ est cycliquement monotone **si et seulement si** il est contenu dans le **sous-différentiel** d'une fonction **convexe** $\varphi : \mathbb{R}^n \to \mathbb{R}$ :
> 
> $$\Gamma \subset \{(x, y) : y \in \partial \varphi(x)\}$$
> 
> En particulier, si la map $T$ est cycliquement monotone, alors $T(x) \in \partial \varphi(x)$ pour une certaine fonction convexe $\varphi$. Si $\varphi$ est différentiable, cela donne $T(x) = \nabla \varphi(x)$.

C'est un théorème de **caractérisation** : il dit que les deux notions (cyclical monotone et gradient convexe) sont **équivalentes**. On peut passer librement de l'une à l'autre.

> [!note]- Rappel : sous-différentiel
> Pour une fonction convexe $\varphi$, le sous-différentiel en $x$ est :
> 
> $$\partial \varphi(x) = \{p \in \mathbb{R}^n : \varphi(y) \geq \varphi(x) + \langle p, y - x \rangle \, \forall y\}$$
> 
> Si $\varphi$ est différentiable en $x$, alors $\partial \varphi(x) = \{\nabla \varphi(x)\}$ (singleton). Sinon (par exemple en un coin de $\varphi$, comme $|x|$ en $x = 0$), $\partial \varphi(x)$ contient tous les "gradients admissibles". C'est la généralisation naturelle du gradient aux fonctions convexes non-différentiables.

### Pourquoi c'est crucial : on a maintenant une paramétrisation utilisable

| Sans Rockafellar | Avec Rockafellar |
| :--- | :--- |
| Chercher $T$ cycliquement monotone | Chercher $\varphi$ convexe et calculer $\nabla \varphi$ |
| Condition combinatoire (toutes les permutations cycliques de $N$ points pour tout $N$) | Condition analytique (convexité d'une seule fonction) |
| Pas de formule constructive | Théorie convexe classique, EDP, optimisation |

C'est exactement le **même type de pont** qu'en optimisation convexe : on ne teste pas "$f$ est convexe" en regardant toutes les paires de points, on regarde si la Hessienne est SDP. Rockafellar joue le rôle de "passe d'une caractérisation combinatoire à une caractérisation analytique".

> [!example] Vérification en 1D
> En 1D, on avait montré que la map optimale est monotone croissante avec la formule $T(x) = G^{-1}(F(x))$. Rockafellar dit qu'elle doit aussi être le gradient d'une convexe.
> 
> En 1D, une fonction $T$ est croissante **si et seulement si** elle est la dérivée d'une fonction convexe. En effet :
> 
> $$\varphi(x) = \int_0^x T(t) \, dt \quad \Rightarrow \quad \varphi'(x) = T(x), \quad \varphi''(x) = T'(x) \geq 0$$
> 
> donc $\varphi$ est bien convexe. Tout se boucle : monotone $\Leftrightarrow$ cycliquement monotone (en 1D) $\Leftrightarrow$ gradient de convexe.

## V. Théorème de Brenier

On a maintenant tous les ingrédients :

1. (Section III) Pour le coût quadratique, la map optimale $T$ est **cycliquement monotone**
2. (Section IV) Toute map cycliquement monotone est $T(x) = \nabla \varphi(x)$ avec $\varphi$ convexe

D'où le théorème central :

> [!quote] Théorème (Brenier, 1991)
> Soient $\mu, \nu$ deux distributions de probabilité sur $\mathbb{R}^n$, avec $\mu$ absolument continue (densité $f$). Pour le coût quadratique $c(x, y) = \|x - y\|^2$, le problème de Monge admet une **unique** solution, qui s'écrit
> 
> $$\boxed{T = \nabla \varphi}$$
> 
> où $\varphi : \mathbb{R}^n \to \mathbb{R}$ est une fonction **convexe** (unique à constante additive près).

**Conséquences immédiates :**

- **Pour le coût quadratique, Monge et Kantorovich coïncident** : l'optimum du problème généralisé de Kantorovich est en fait un transport déterministe.
- **La map optimale est la dérivée d'un potentiel** : on peut représenter le transport par une seule fonction $\varphi$, qui est convexe (donc bien comprise par la théorie convexe).
- **Le potentiel $\varphi$ se calcule par dualité** : c'est l'objet central de la note 04.

## VI. (Bonus) L'équation de Monge-Ampère

Le théorème de Brenier nous dit que $T = \nabla \varphi$. Combinons cela avec la **conservation de la masse** (changement de variable dans l'intégrale) pour obtenir une équation aux dérivées partielles vérifiée par $\varphi$.

### Dérivation

Si $\mu$ a densité $f$ et $\nu$ a densité $g$, la condition $T_\# \mu = \nu$ s'écrit, par changement de variable :

$$g(T(x)) \cdot |\det(\nabla T(x))| = f(x)$$

Comme $T = \nabla \varphi$, on a $\nabla T = D^2 \varphi$ (la matrice hessienne de $\varphi$). Et comme $\varphi$ est convexe, $D^2 \varphi$ est semi-définie positive, donc son déterminant est positif. On obtient :

$$\boxed{\det(D^2 \varphi(x)) = \frac{f(x)}{g(\nabla \varphi(x))}}$$

C'est l'**équation de Monge-Ampère**.

### Pourquoi c'est intéressant (et pourquoi c'est dur)

**L'intéressant** : trouver le transport optimal $T$ est ramené à résoudre une EDP elliptique en $\varphi$. Pour le coût quadratique, **l'optimisation devient une analyse d'EDP**. C'est cette connexion qui a motivé une grande partie du développement mathématique du transport optimal au 20e siècle.

**Le dur** :
- L'équation est **fortement non-linéaire** (le déterminant de la hessienne, pas un simple Laplacien)
- Elle est **non-divergente** : on ne peut pas faire les intégrations par parties habituelles
- Les solutions ne sont **pas toujours régulières** : il faut une notion de "solutions faibles" (viscosité, solutions d'Alexandrov, etc.) qui est très technique

**Pour data science** : on ne résout **jamais** Monge-Ampère directement. En pratique on utilise plutôt :
- La **formulation duale** (note 04) qui est plus simple numériquement
- La **régularisation entropique** (note 05) qui transforme le problème en itérations matricielles

L'équation de Monge-Ampère reste une **belle culture** à connaître, mais ce n'est pas l'outil pratique.

## VII. Trois idées à retenir

1. **En 1D, la map optimale est monotone.** Elle se construit explicitement par $T(x) = G^{-1}(F(x))$ via les CDFs. Concrètement : on trie les deux distributions et on apparie les percentiles correspondants.

2. **En dimension supérieure, la map optimale est cycliquement monotone.** C'est la généralisation directe de la monotonie 1D : pour $N = 2$, $\langle x_2 - x_1, y_2 - y_1 \rangle \geq 0$ (angle aigu). La condition pour $N \geq 3$ est strictement plus forte et caractérise les gradients de fonctions convexes (Rockafellar).

3. **Brenier (1991)** : pour le coût quadratique, la solution du transport optimal est unique et s'écrit $T = \nabla \varphi$ avec $\varphi$ convexe. C'est le pont entre transport optimal et théorie convexe. L'EDP de Monge-Ampère $\det(D^2 \varphi) = f / g(\nabla \varphi)$ apparaît comme conséquence de la conservation de la masse.

## VIII. Vers la suite

- **Note 03 — Distance de Wasserstein** : la valeur optimale du coût définit une vraie distance entre distributions, avec des propriétés bien plus riches que les distances euclidiennes ou KL. C'est l'objet qu'on **utilise en pratique** en ML.
- **Note 04 — Dualité de Kantorovich** : reformulation duale du problème via la transformée de Legendre, qui transforme la fonction convexe $\varphi$ en un objet plus pratique numériquement.
- **Note 05 — Sinkhorn** : algorithme rapide qui exploite la régularisation entropique pour calculer Wasserstein en $O(n^2)$ par itération.
