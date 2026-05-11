---
title: Variétés plongées
date: 2026-05-11
tags: [mathématiques, géométrie-différentielle, variétés]
---

## L'idée fondatrice

On veut faire du calcul différentiel — gradients, descente, optimisation — sur des objets **qui ne sont pas des espaces vectoriels**. Trois exemples typiques en ML et en finance :

| Variété | Définition | Où ça apparaît |
| :--- | :--- | :--- |
| Sphère $S^{n-1}$ | vecteurs unitaires de $\mathbb{R}^n$ | embeddings normalisés, directions |
| Stiefel $V_F(\mathbb{R}^D)$ | matrices $D \times F$ à colonnes orthonormales | factor models orthogonaux, ACP |
| SPD$(n)$ | matrices symétriques définies positives | covariances, géométrie de Fisher |

Toutes ces variétés ont un point commun : elles vivent **à l'intérieur** d'un espace vectoriel familier, et elles sont définies par une **contrainte algébrique**. C'est cette structure qu'on va exploiter. Plutôt que de construire la géométrie de manière abstraite (par cartes et atlas, comme dans les cours de maths classiques), on profite de l'espace dans lequel la variété est posée. C'est plus concret, plus calculatoire, et c'est exactement ce qui sert en pratique.

## I. L'espace ambient — vocabulaire de base

Avant de définir une variété, il faut un mot pour désigner l'espace **dans lequel** elle vit. On l'appelle l'**espace ambient** (ou *ambient space* en anglais).

C'est l'espace vectoriel "gros" qui contient la variété "petite" :

| Variété | Espace ambient | Dimension ambient | Dimension variété |
| :--- | :--- | :--- | :--- |
| Cercle $S^1$ | $\mathbb{R}^2$ | 2 | 1 |
| Sphère $S^2$ | $\mathbb{R}^3$ | 3 | 2 |
| Sphère $S^{n-1}$ | $\mathbb{R}^n$ | $n$ | $n-1$ |
| Stiefel $V_F(\mathbb{R}^D)$ | $\mathbb{R}^{D \times F}$ | $DF$ | $DF - \tfrac{F(F+1)}{2}$ |
| SPD$(n)$ | matrices symétriques $n \times n$ | $\tfrac{n(n+1)}{2}$ | $\tfrac{n(n+1)}{2}$ |

Exemple concret : la sphère $S^2$, c'est un objet de dimension 2 (une surface), mais on a besoin de 3 coordonnées $(x,y,z)$ pour la décrire. L'espace ambient $\mathbb{R}^3$ est plus gros que ce qu'il faut — c'est juste le décor dans lequel on pose la sphère.

> [!note]- Lien avec les "embeddings" en ML
> Le terme **plongement** (*embedding*) est le même qu'en machine learning : un word embedding, c'est un mot représenté comme vecteur dans $\mathbb{R}^d$ ; un plongement de Poincaré, c'est un point placé dans la boule hyperbolique. L'idée est toujours la même : représenter un objet "intéressant" dans un espace vectoriel familier où on sait faire des calculs.

## II. Une variété, vue de l'extérieur

L'idée géométrique : une **variété plongée de dimension $k$ dans $\mathbb{R}^n$** est un sous-ensemble $M \subset \mathbb{R}^n$ qui, **localement, ressemble à $\mathbb{R}^k$**. Pas de creux, pas de coins, pas de croisements — juste une surface lisse.

![[Pasted image 20260511143506.png]]
*Figure 1. Trois variétés plongées dans leur espace ambient : la sphère $S^2 \subset \mathbb{R}^3$ (dim 2), un tore $T^2 \subset \mathbb{R}^3$ (dim 2), et une courbe lisse $\subset \mathbb{R}^2$ (dim 1). Chacune est définie par une contrainte algébrique.*

**Intuition centrale.** Si je suis à un point $p \in M$ et que je zoome, je vois quelque chose qui ressemble à un morceau de plan (dimension 2) ou de droite (dimension 1) — pas à un coin, pas à un croisement. Une variété, c'est exactement ça : un objet **lisse** vu de près.

**Ce qui n'est *pas* une variété.** Prenons le contre-exemple typique : deux droites qui se croisent dans le plan, comme un signe `+`. Globalement, c'est un objet de dimension 1 (de la "courbe"), mais **au point de croisement**, ce n'est pas lisse : si je zoome sur le centre, je ne vois pas un morceau de droite, je vois toujours une croix. C'est exactement le genre de défaut qu'exclut la définition d'une variété.

> [!note]- Pourquoi la croix échoue formellement
> Au centre de la croix, il n'existe aucun voisinage qui ressemble à $\mathbb{R}^1$. On peut le montrer rigoureusement : si on enlève le point central d'un petit morceau de droite, on obtient deux composantes connexes ; si on enlève le centre de la croix, on obtient **quatre** composantes connexes. Comme un homéomorphisme préserve le nombre de composantes, il ne peut pas exister.

## III. La définition pratique — par les contraintes

La façon de définir une variété dans la vraie vie (ML, finance), c'est par une **équation de contrainte**. On prend une fonction $g : \mathbb{R}^n \to \mathbb{R}^m$ et on regarde :

$$M = \{x \in \mathbb{R}^n \mid g(x) = 0\}$$

Si $g$ est suffisamment lisse et "régulière", cet ensemble est une variété de dimension $n - m$.

**Intuition de la dimension.** Tu pars de l'espace ambient $\mathbb{R}^n$ (dimension $n$). Tu ajoutes $m$ contraintes. Chaque contrainte indépendante "mange" une dimension. Il te reste $n - m$ dimensions libres. C'est tout.

> [!note]- La condition technique : régularité
> "Régulière" signifie que la jacobienne $Dg(x) \in \mathbb{R}^{m \times n}$ est **de rang plein $m$** en tout point $x \in M$. C'est le **théorème de la submersion**. Intuition : les $m$ contraintes $g_1(x)=0, \dots, g_m(x)=0$ doivent être "indépendantes" — aucune n'est redondante avec les autres — sinon on perd une dimension de structure.
>
> En pratique, sur les variétés qu'on rencontre (sphère, Stiefel, SPD), cette condition est facile à vérifier et toujours satisfaite.

**Exemple 1 — la sphère.** On prend $g : \mathbb{R}^n \to \mathbb{R}$ définie par $g(x) = \|x\|^2 - 1$. Alors :

$$S^{n-1} = \{x \in \mathbb{R}^n \mid \|x\|^2 - 1 = 0\}$$

Une seule contrainte ($m = 1$), donc la sphère est de dimension $n - 1$. Pour $n = 3$ : dimension 2. ✓

**Exemple 2 — SPD.** L'ensemble SPD$(n)$ des matrices symétriques **strictement** définies positives est une variété **ouverte** dans l'espace des matrices symétriques (pas de contrainte d'égalité, juste une condition d'inégalité). Dimension : $n(n+1)/2$ — le nombre de coefficients indépendants d'une matrice symétrique.

## IV. La variété de Stiefel en détail

Stiefel mérite sa propre section parce que c'est *la* variété qui apparaît dans le challenge QRT et plus généralement dans tous les factor models avec orthogonalité.

**Définition.** Pour $D \ge F$ :

$$V_F(\mathbb{R}^D) = \left\{ A \in \mathbb{R}^{D \times F} \,\middle|\, A^\top A = I_F \right\}$$

C'est l'ensemble des matrices $D \times F$ dont les **$F$ colonnes sont des vecteurs unitaires et orthogonaux entre eux** dans $\mathbb{R}^D$.

**Visualisation mentale.** Tu prends $F$ vecteurs dans $\mathbb{R}^D$, chacun de norme 1, tous orthogonaux deux à deux. C'est un **repère orthonormé partiel** dans $\mathbb{R}^D$ — partiel parce que tu n'utilises que $F$ vecteurs au lieu des $D$ qu'il faudrait pour un repère complet. L'ensemble de tous les repères orthonormés partiels possibles, c'est Stiefel.

**Cas limites — la clé pour s'y retrouver :**

| Cas | Interprétation | Variété connue |
| :--- | :--- | :--- |
| $V_1(\mathbb{R}^n)$ | un vecteur unitaire dans $\mathbb{R}^n$ | $S^{n-1}$ (la sphère) |
| $V_n(\mathbb{R}^n)$ | $n$ vecteurs unitaires orthogonaux dans $\mathbb{R}^n$ | $O(n)$ (groupe orthogonal) |
| $V_F(\mathbb{R}^D)$, $1 < F < D$ | $F$ vecteurs ortho. partiels dans $\mathbb{R}^D$ | "entre la sphère et le groupe orthogonal" |

Donc **Stiefel est juste une généralisation de la sphère**. C'est pas plus abstrait que ça : à la place d'un seul vecteur unitaire, tu en prends $F$ qui sont en plus orthogonaux entre eux.

**Calcul de la dimension.** L'espace ambient est $\mathbb{R}^{D \times F}$, de dimension $DF$. La contrainte est $A^\top A = I_F$, à valeurs dans les matrices **symétriques** $F \times F$ (car $A^\top A$ est toujours symétrique — on ne peut pas imposer de contrainte sur la partie anti-symétrique qui n'existe pas). L'espace des matrices symétriques $F \times F$ a dimension $F(F+1)/2$ — c'est le nombre de contraintes scalaires effectives. Donc :

$$\dim V_F(\mathbb{R}^D) = DF - \frac{F(F+1)}{2}$$

**Pour le challenge QRT** : $D = 250$, $F = 10$, donc :

$$\dim V_{10}(\mathbb{R}^{250}) = 2500 - 55 = 2445$$

C'est gros, mais bien plus petit que $\mathbb{R}^{D \times F}$ tout entier (dimension $2500$). C'est cette structure qu'on va exploiter pour faire de l'optimisation efficace, au lieu de tirer des matrices au hasard et de les projeter (méthode QRT du benchmark).

## V. Pourquoi cette définition est suffisante

À ce stade tu peux te demander : "pourquoi pas la définition standard par cartes et atlas, comme dans tous les livres ?". Réponse honnête : parce que pour ce qu'on va faire, on n'en a **jamais besoin**.

| Concept | Approche par cartes | Approche extrinsèque |
| :--- | :--- | :--- |
| Définir un point | $p \in M$ avec coordonnées $\varphi_i(p) \in \mathbb{R}^k$ | $x \in \mathbb{R}^n$ avec $g(x) = 0$ |
| Espace tangent | espace des dérivations | $\{v \in \mathbb{R}^n \mid Dg(x) \, v = 0\}$ |
| Gradient | calcul via cartes | projection orthogonale dans $\mathbb{R}^n$ |

Note : les **géodésiques** et l'**exponentielle** (pour avancer sur la variété) ne sont *pas* spécifiques à l'approche par cartes — elles existent dans les deux. On y reviendra dans la note sur la métrique riemannienne. En pratique on leur préfère souvent une **rétraction**, une approximation moins coûteuse à calculer (par exemple la décomposition QR pour Stiefel).

L'approche extrinsèque donne **directement des formules calculables** : projeter dans l'espace tangent, c'est de l'algèbre linéaire dans $\mathbb{R}^n$. C'est exactement ce que font les librairies (`pymanopt`, `geomstats`) et les papiers de ML géométrique.

> [!note]- Quand l'approche par cartes est-elle vraiment utile ?
> Pour des variétés **abstraites** qui n'ont pas de plongement naturel dans un espace euclidien — par exemple en géométrie algébrique, en topologie différentielle pure, ou pour étudier la courbure intrinsèque (Lott-Villani-Sturm, courbure de Ricci synthétique). En ML/finance, toutes les variétés utiles sont plongées et l'approche extrinsèque suffit largement.

## VI. Ce qu'on a besoin de retenir

Trois idées à garder en tête pour la suite :

1. **Une variété, c'est un sous-ensemble lisse de $\mathbb{R}^n$**, défini par une contrainte $g(x) = 0$ (ou un système de contraintes). L'espace $\mathbb{R}^n$ qui la contient s'appelle l'**espace ambient**.
2. **La dimension de la variété**, c'est la dimension de l'espace ambient moins le nombre de contraintes indépendantes.
3. **Tout calcul se fera dans l'espace ambient**, en respectant la contrainte — pas dans un espace abstrait.

## VII. Vers les vecteurs tangents

La prochaine étape : si je suis au point $x \in M$ et que je veux "bouger" tout en restant sur la variété, dans quelles directions ai-je le droit d'aller ? L'ensemble de ces directions admissibles, c'est l'**espace tangent** $T_x M$ — l'objet central pour faire de l'optimisation.

Sur la sphère $S^2$, l'intuition est claire : les directions admissibles forment le **plan tangent** à la sphère en $x$ — un plan orthogonal au vecteur $x$. On verra que ça se généralise très naturellement à toutes les variétés définies par contraintes : $T_x M$ est le **noyau de la jacobienne** $Dg(x)$.

→ Suite : [[02_Vecteurs tangents]]
