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

## I. Le cas 1D : la map optimale est monotone

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

**Interprétation géométrique.** Sur la droite réelle, "transporter optimalement" $\mu$ vers $\nu$ revient à les **trier ensemble** : le $k$-ième percentile de $\mu$ va sur le $k$-ième percentile de $\nu$.

## II. Construction explicite via les CDFs

En 1D, on peut **construire explicitement** la map optimale via les fonctions de répartition (CDFs).

Notons :

$$F(x) = \int_{-\infty}^x f(t) \, dt, \qquad G(y) = \int_{-\infty}^y g(t) \, dt$$

les CDFs de $\mu$ et $\nu$. Ce sont des fonctions croissantes de $\mathbb{R}$ vers $[0, 1]$.

**Idée.** Puisque la map optimale est monotone, elle doit préserver les **percentiles** : la masse de $\mu$ avant $x$ (c'est-à-dire $F(x)$) doit être égale à la masse de $\nu$ avant $T(x)$ (c'est-à-dire $G(T(x))$) :

$$F(x) = G(T(x))$$

D'où l'expression explicite :

$$\boxed{T(x) = G^{-1}(F(x))}$$

![[brenier_1d_cdf.png|402]]
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

![[brenier_cyclical_monotone.png]]
*Cyclical monotonicity en dimension supérieure, cas $N=2$. Les vecteurs $\vec{x_1 x_2}$ (rouge, dans le domaine source) et $\vec{y_1 y_2}$ (bleu, dans le domaine cible) doivent faire un angle aigu. Géométriquement : la map ne peut pas "tordre" la masse, elle peut seulement la déplacer dans des directions cohérentes.*

C'est la généralisation directe de la monotonie 1D ($x_1 < x_2 \Rightarrow y_1 \leq y_2$, qui est équivalent à $(x_2 - x_1)(y_2 - y_1) \geq 0$).

> [!note]- Pourquoi "cyclical" ?
> Pour $N = 2$, l'inégalité s'écrit $\langle x_2 - x_1, y_2 - y_1 \rangle \geq 0$. Pour $N = 3, 4, \dots$, l'inégalité est plus subtile : on **doit la vérifier pour toute permutation cyclique** des indices. Contrairement à la "monotonie" qui ne regarde que les paires, la cyclical monotonicity contrôle **toutes** les permutations possibles — c'est strictement plus fort.
> 
> En dimension 1, monotonie et cyclical monotonicity coïncident. Mais en dimension $\geq 2$, il existe des maps "monotones par paires" qui ne sont pas cycliquement monotones — elles violent l'inégalité dès $N = 3$.

## IV. Théorème de Rockafellar

Le résultat clé qui transforme la cyclical monotonicity en quelque chose d'exploitable :

> [!quote] Théorème (Rockafellar, 1966)
> Un ensemble $\Gamma \subset \mathbb{R}^n \times \mathbb{R}^n$ est cycliquement monotone si et seulement si il est contenu dans le **sous-différentiel** d'une fonction **convexe** $\varphi : \mathbb{R}^n \to \mathbb{R}$ :
> 
> $$\Gamma \subset \{(x, y) : y \in \partial \varphi(x)\}$$
> 
> En particulier, si la map $T$ est cycliquement monotone, alors $T(x) \in \partial \varphi(x)$ pour une certaine fonction convexe $\varphi$. Si $\varphi$ est différentiable, cela donne $T(x) = \nabla \varphi(x)$.

**Intuition.** Pour une fonction convexe $\varphi$ et son gradient $\nabla \varphi$ :

$$\sum_{i=1}^N \langle x_i, \nabla\varphi(x_{i+1}) - \nabla\varphi(x_i) \rangle \leq 0$$

découle de l'inégalité de convexité $\varphi(x_{i+1}) \geq \varphi(x_i) + \langle \nabla\varphi(x_i), x_{i+1} - x_i \rangle$ sommée sur le cycle. Rockafellar montre la réciproque : toute map cycliquement monotone est de cette forme.

> [!note]- Rappel : sous-différentiel
> Pour une fonction convexe $\varphi$, le sous-différentiel en $x$ est :
> 
> $$\partial \varphi(x) = \{p \in \mathbb{R}^n : \varphi(y) \geq \varphi(x) + \langle p, y - x \rangle \, \forall y\}$$
> 
> Si $\varphi$ est différentiable en $x$, alors $\partial \varphi(x) = \{\nabla \varphi(x)\}$. Sinon (par exemple en un coin de $\varphi$), $\partial \varphi(x)$ contient tous les "gradients admissibles".

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
