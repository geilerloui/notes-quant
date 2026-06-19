---
title: 03_Métrique et gradient riemannien
date: 2026-05-11
tags:
  - mathématiques
  - géométrie-différentielle
  - optimisation
  - gradient
---

## L'idée fondatrice

On a maintenant l'espace tangent $T_x M$ — les directions admissibles en un point $x$ de la variété. Mais pour faire de l'**optimisation**, ce n'est pas suffisant. Il faut aussi pouvoir :

- **Mesurer** la longueur d'un vecteur tangent (pour parler de pas de descente)
- **Comparer** deux directions (angle entre elles, orthogonalité)
- **Calculer un gradient** : la direction de plus grande variation d'une fonction $f : M \to \mathbb{R}$

Tout ça repose sur un seul outil : un **produit scalaire** sur chaque $T_x M$. Quand on en met un, on obtient une **variété riemannienne**, et on peut faire toute la géométrie qu'on connaît dans $\mathbb{R}^n$ (longueurs, angles, gradient, descente) — mais sur la variété.

## I. La métrique riemannienne

### Définition

Une **métrique riemannienne** sur $M$ est la donnée, en chaque point $x \in M$, d'un **produit scalaire** $\langle \cdot, \cdot \rangle_x$ sur l'espace tangent $T_x M$.

Concrètement, c'est une fonction $g_x : T_x M \times T_x M \to \mathbb{R}$ qui pour tous $u, v, w \in T_x M$ et $\lambda \in \mathbb{R}$ vérifie :

- **Bilinéarité** : $g_x(\lambda u + w, v) = \lambda g_x(u, v) + g_x(w, v)$ (et idem à droite)
- **Symétrie** : $g_x(u, v) = g_x(v, u)$
- **Définie positive** : $g_x(v, v) \ge 0$, avec égalité ssi $v = 0$

C'est exactement le produit scalaire usuel, sauf qu'il **dépend du point $x$** où on se trouve. Tu peux le voir comme une "règle locale de mesure" : à chaque endroit de la variété, on a un instrument différent pour mesurer les vecteurs tangents.

> [!note]- Pourquoi la métrique peut dépendre du point
> Dans $\mathbb{R}^n$, on a un seul produit scalaire (le produit scalaire euclidien) valable partout. Mais sur une variété courbe, deux points peuvent avoir des "espaces tangents différents" — par exemple sur la sphère, le plan tangent au pôle nord et le plan tangent à l'équateur ne sont pas le même plan dans $\mathbb{R}^3$. Donc on a besoin d'un produit scalaire **par point**.

### Ce qu'on peut faire avec une métrique

Une fois qu'on a $\langle \cdot, \cdot \rangle_x$, on peut définir :

- **Norme** : $\|v\|_x = \sqrt{\langle v, v \rangle_x}$
- **Angle** entre deux vecteurs tangents : $\cos\theta = \frac{\langle u, v \rangle_x}{\|u\|_x \|v\|_x}$
- **Orthogonalité** : $u \perp v$ si $\langle u, v \rangle_x = 0$
- **Longueur d'une courbe** $\gamma : [a, b] \to M$ : $\ell(\gamma) = \int_a^b \|\gamma'(t)\|_{\gamma(t)} \, dt$
- **Distance** entre deux points : longueur de la plus courte courbe qui les relie

Tout ça avec une seule structure : un produit scalaire par point.

## II. La métrique induite — le cas le plus simple

Pour une **variété plongée** $M \subset \mathbb{R}^n$, il y a un choix de métrique évident : utiliser le produit scalaire usuel de $\mathbb{R}^n$, **restreint** aux espaces tangents.

**Définition.** La **métrique induite** (ou *métrique héritée*) sur $M$ est :

$$\langle u, v \rangle_x := u^\top v \quad \text{pour } u, v \in T_x M$$

C'est-à-dire : on prend deux vecteurs tangents $u$ et $v$ (qui vivent dans $\mathbb{R}^n$ rappelons-le), et on calcule leur produit scalaire **comme s'ils étaient deux vecteurs de $\mathbb{R}^n$ ordinaires**.

**Remarque importante.** Avec cette métrique, le produit scalaire ne dépend pas vraiment du point $x$ — la formule est la même partout. Ce qui dépend de $x$, c'est l'**espace** $T_x M$ lui-même (le plan tangent change quand on bouge sur la sphère), mais le produit scalaire reste $u^\top v$.

**Sphère, Stiefel, SPD : c'est tout ce dont on a besoin.** Pour la plupart des applications en ML/finance, la métrique induite suffit. C'est ce qu'on utilise dans `pymanopt` par défaut, et c'est ce qu'on utilise dans tout ce qui suit.

> [!note]- D'autres métriques existent
> Sur SPD$(n)$ par exemple, on utilise souvent la **métrique affine-invariante** $\langle U, V \rangle_X = \text{tr}(X^{-1} U X^{-1} V)$ plutôt que la métrique induite — elle a de meilleures propriétés (invariance par GL, géodésiques en forme fermée). Sur l'espace des distributions de probabilité, on utilise la **métrique de Fisher**. Le choix de métrique change la géométrie et l'algorithme, mais l'idée reste la même : c'est un produit scalaire sur l'espace tangent.

## III. Le problème du gradient

On veut optimiser une fonction $f : M \to \mathbb{R}$ sur la variété. En analyse classique, le **gradient** $\nabla f$ donne la direction de plus grande pente — on s'en sert pour faire de la descente. Sur une variété, c'est plus compliqué.

### Le gradient euclidien sort de la variété

Si $f$ est définie sur tout $\mathbb{R}^n$ (ou peut s'étendre), on peut calculer son gradient euclidien usuel $\nabla f(x) \in \mathbb{R}^n$. Problème : **ce gradient n'est pas tangent à $M$ en général**.

![[Pasted image 20260511164208.png|545]]
*Figure 1. À gauche : le gradient euclidien $\nabla f(x)$ pointe dans une direction quelconque de $\mathbb{R}^3$, en général **hors du plan tangent** $T_x S^2$. À droite : pour rester sur la variété, on doit projeter ce gradient sur $T_x S^2$ — c'est le **gradient riemannien** $\text{grad}_M f(x)$.*

Si on essaie de faire un pas $x_{k+1} = x_k - \alpha \nabla f(x_k)$, on **sort de la variété**. Sur la sphère, on quitte la surface. C'est exactement le problème qu'on veut résoudre.

![[Pasted image 20260511165105.png|259]]
*Figure 2. Après un pas tangent $\alpha v$ depuis $x$, on arrive en $x + \alpha v$ qui est dans le plan tangent mais **hors de la sphère**. Pour finir l'étape de descente, on ramène ce point sur la sphère via une **rétraction** $R_x$ — c'est le sujet de la note suivante. Cette figure anticipe la chaîne complète : gradient → projection → rétraction.*

### L'idée : projeter sur l'espace tangent

La solution est naturelle : on prend la **composante tangente** du gradient euclidien, c'est-à-dire sa projection orthogonale sur $T_x M$. On obtient un vecteur qui :

1. Est dans $T_x M$ — donc compatible avec la variété
2. Pointe dans la direction de plus grande pente *parmi les directions admissibles*

C'est exactement le **gradient riemannien**.

## IV. Le gradient riemannien

### Définition

Soit $f : M \to \mathbb{R}$ et soit $\bar{f} : \mathbb{R}^n \to \mathbb{R}$ une extension de $f$ à l'espace ambient (c'est-à-dire que $\bar{f}|_M = f$). Le **gradient riemannien** de $f$ en $x \in M$ est :

$$\boxed{\text{grad}_M f(x) = \text{Proj}_{T_x M}\big(\nabla \bar{f}(x)\big)}$$

C'est la **projection orthogonale** du gradient euclidien sur l'espace tangent. C'est *la* formule centrale qu'on va utiliser pour faire de la descente sur variété.

> [!note]- Indépendance par rapport à l'extension
> On peut montrer que $\text{grad}_M f(x)$ ne dépend pas de l'extension $\bar{f}$ choisie. C'est rassurant : seule la variété et la fonction sur la variété importent.
>
> Intuition : si deux extensions $\bar{f}_1$ et $\bar{f}_2$ coïncident sur $M$, leur différence $\bar{f}_1 - \bar{f}_2$ est nulle sur $M$, donc son gradient en $x$ est orthogonal à $T_x M$ (intuitivement : varie uniquement dans les directions "hors variété"). La projection sur $T_x M$ donne donc le même résultat.

### Propriété de descente

Le gradient riemannien a la propriété qu'on attend : **c'est la direction de plus grande pente parmi les directions tangentes**. Plus précisément :

$$\text{grad}_M f(x) = \arg\max_{v \in T_x M, \, \|v\|_x = 1} \, \langle v, \nabla \bar{f}(x) \rangle$$

Donc si on fait un pas dans la direction $-\text{grad}_M f(x)$, $f$ va décroître localement (à condition de rester sur $M$, ce qu'on assurera dans la note suivante avec les rétractions).

## V. Formules explicites pour nos exemples

### Sphère $S^{n-1}$

Rappel : $T_x S^{n-1} = x^\perp = \{v : \langle x, v \rangle = 0\}$. La projection orthogonale sur cet hyperplan est :

$$\text{Proj}_{T_x S^{n-1}}(w) = w - \langle x, w \rangle x = (I - x x^\top) w$$

**Interprétation.** On enlève à $w$ sa composante radiale (selon $x$) pour ne garder que la composante tangente.

**Gradient riemannien.** Pour $f : S^{n-1} \to \mathbb{R}$ d'extension $\bar{f}$ :

$$\text{grad}_{S^{n-1}} f(x) = (I - x x^\top) \nabla \bar{f}(x) = \nabla \bar{f}(x) - \langle x, \nabla \bar{f}(x) \rangle x$$

> [!note]- Exemple numérique : descente sur la sphère
> Prenons $f : S^2 \to \mathbb{R}$ définie par $f(x) = x_3$ (la "hauteur" sur la sphère). On veut minimiser $f$ — l'optimum est le pôle sud $(0, 0, -1)$.
>
> Extension : $\bar{f}(x) = x_3$, donc $\nabla \bar{f}(x) = (0, 0, 1)$ partout.
>
> Au point $x = (1, 0, 0)$ (sur l'équateur) :
> - Gradient euclidien : $\nabla \bar{f}(x) = (0, 0, 1)$ — pointe vers le haut, hors de la sphère.
> - Composante radiale : $\langle x, \nabla \bar{f}(x) \rangle = 0$ (le gradient est déjà tangent !)
> - Gradient riemannien : $\text{grad} f(x) = (0, 0, 1) - 0 \cdot x = (0, 0, 1)$. Tangent à la sphère, pointe vers le pôle nord — donc $-\text{grad} f$ pointe vers le pôle sud. ✓
>
> Au point $x = (0, 0, 1)$ (pôle nord) :
> - Gradient euclidien : $(0, 0, 1)$
> - Composante radiale : $\langle x, \nabla \bar{f}(x) \rangle = 1$
> - Gradient riemannien : $(0, 0, 1) - 1 \cdot (0, 0, 1) = (0, 0, 0)$. Le pôle nord est un **point critique** (maximum global de $f$). ✓

### Stiefel $V_F(\mathbb{R}^D)$

Pour Stiefel, la projection est un peu plus subtile. Rappel : $T_A V_F = \{V : A^\top V \text{ anti-symétrique}\}$.

**Formule de projection.** Pour $W \in \mathbb{R}^{D \times F}$ :

$$\text{Proj}_{T_A V_F}(W) = W - A \cdot \text{sym}(A^\top W)$$

où $\text{sym}(M) = \tfrac{1}{2}(M + M^\top)$ est la partie symétrique de $M$.

> [!note]- D'où vient cette formule ?
> On veut décomposer $W = V + N$ avec $V \in T_A V_F$ (tangent) et $N \perp T_A V_F$ (normal). On peut montrer que les vecteurs normaux à Stiefel en $A$ sont de la forme $N = A S$ avec $S \in \mathbb{R}^{F \times F}$ **symétrique**.
>
> Donc on cherche $S$ symétrique telle que $W - A S$ soit tangent, i.e. $A^\top(W - AS)$ anti-symétrique. Comme $A^\top A = I_F$, ça donne $A^\top W - S$ anti-symétrique, soit $S$ = partie symétrique de $A^\top W$.
>
> D'où $\text{Proj}_{T_A V_F}(W) = W - A \cdot \text{sym}(A^\top W)$.

**Gradient riemannien sur Stiefel.** Pour $f : V_F(\mathbb{R}^D) \to \mathbb{R}$ :

$$\text{grad}_{V_F} f(A) = \nabla \bar{f}(A) - A \cdot \text{sym}\big(A^\top \nabla \bar{f}(A)\big)$$

C'est *la* formule qu'on va utiliser pour le challenge QRT. Concrètement : on calcule le gradient euclidien (facile, c'est une fonction de matrice dans $\mathbb{R}^{D \times F}$), puis on le projette sur l'espace tangent par cette formule. Algèbre linéaire pure, $O(D F^2)$ — bien moins cher qu'une décomposition spectrale.

### SPD$(n)$

SPD est un ouvert, donc $T_X \text{SPD}(n) = \text{Sym}(n)$ (toutes les matrices symétriques). La projection sur Sym$(n)$ depuis l'espace des matrices $n \times n$ est juste la **symétrisation** :

$$\text{Proj}_{\text{Sym}(n)}(W) = \tfrac{1}{2}(W + W^\top)$$

**Gradient riemannien (métrique induite) :**

$$\text{grad}_{\text{SPD}} f(X) = \tfrac{1}{2}\big(\nabla \bar{f}(X) + \nabla \bar{f}(X)^\top\big)$$

## VI. Une étape de descente — l'idée

On a maintenant tout pour décrire l'**algorithme de base** de l'optimisation riemannienne :

1. Au point $x_k \in M$, calculer le gradient euclidien $\nabla \bar{f}(x_k)$
2. Projeter sur $T_{x_k} M$ pour obtenir $\text{grad}_M f(x_k)$
3. Choisir un pas $\alpha_k > 0$ et faire un pas dans la direction $-\text{grad}_M f(x_k)$ : $x_k + \alpha_k v_k$ avec $v_k = -\text{grad}_M f(x_k)$
4. **Problème** : $x_k + \alpha_k v_k$ n'est plus sur $M$ ! Il faut le ramener sur la variété — c'est le rôle de la **rétraction**, qu'on verra dans la note suivante

Pour la sphère, l'intuition est claire : on fait un pas dans le plan tangent, puis on **renormalise** pour revenir sur la sphère. C'est une rétraction simple. Pour Stiefel, on fera une décomposition QR. Pour SPD, on utilisera la projection sur le cône positif ou une exponentielle de matrice.

## VII. Trois idées à retenir

1. **Une métrique riemannienne, c'est un produit scalaire sur chaque espace tangent** $T_x M$. Pour les variétés plongées (sphère, Stiefel, SPD), on utilise la métrique induite : $\langle u, v \rangle_x = u^\top v$.

2. **Le gradient euclidien sort en général de la variété**. Le **gradient riemannien** est sa **projection orthogonale** sur l'espace tangent :

$$\text{grad}_M f(x) = \text{Proj}_{T_x M}\big(\nabla \bar{f}(x)\big)$$

3. **Formules pratiques** :
   - Sphère : $\text{Proj}(w) = w - \langle x, w \rangle x$
   - Stiefel : $\text{Proj}(W) = W - A \cdot \text{sym}(A^\top W)$
   - SPD : $\text{Proj}(W) = \tfrac{1}{2}(W + W^\top)$

## VIII. Vers la rétraction

Avec le gradient riemannien, on sait dans quelle direction aller pour décroître $f$. Mais quand on fait un pas $x + \alpha v$ avec $v$ tangent, on tombe **à côté** de la variété. La dernière brique qui manque c'est la **rétraction** : une manière de "ramener" le point sur la variété après le pas. C'est ça qu'on regarde dans la note suivante.

→ Suite : [[04_Rétraction et descente sur variété]]
