---
title: Rétraction et descente sur variété
date: 2026-05-11
tags: [mathématiques, géométrie-différentielle, optimisation, rétraction]
---

## L'idée fondatrice

On a tout pour calculer un **gradient riemannien** $\text{grad}_M f(x)$ — un vecteur dans $T_x M$ qui pointe dans la direction de plus grande pente sur la variété. Mais quand on fait un pas

$$y = x - \alpha \cdot \text{grad}_M f(x)$$

on tombe **à côté** de la variété : $y \in \mathbb{R}^n$ mais $y \notin M$ en général. Sur la sphère, on a fait un pas dans le plan tangent, donc on se retrouve dans ce plan, qui ne touche la sphère qu'au point $x$.

Il faut une dernière brique : un moyen de **ramener** $y$ sur la variété. C'est ce qu'on appelle une **rétraction**. Avec cette brique, on a l'algorithme complet de **descente de gradient riemannienne**, prêt à coder pour le challenge QRT.

## I. Définition d'une rétraction

Une **rétraction** au point $x \in M$ est une application

$$R_x : T_x M \to M$$

qui prend un vecteur tangent $v \in T_x M$ et renvoie un point de la variété. Pour être une "bonne" rétraction, $R_x$ doit vérifier deux propriétés :

1. **Centrage** : $R_x(0) = x$ — si on ne bouge pas, on reste en $x$.
2. **Différentielle = identité** : $DR_x(0) = \text{Id}_{T_x M}$ — au voisinage de 0, $R_x$ se comporte comme l'identité.

> [!note]- Pourquoi la condition sur la différentielle ?
> La deuxième condition garantit que **pour des petits pas $\alpha$**, on a $R_x(\alpha v) \approx x + \alpha v + O(\alpha^2)$. C'est-à-dire : à l'ordre 1 en $\alpha$, faire une rétraction de $\alpha v$ est équivalent à faire un pas euclidien $\alpha v$. Cette propriété est exactement ce qu'il faut pour que la descente riemannienne *converge* aussi bien que la descente euclidienne dans le cas où on optimise sur $\mathbb{R}^n$.

**Intuition.** $R_x$ est une "façon de bouger sur la variété en partant de $x$ dans la direction $v$". Il existe en général plusieurs rétractions valides pour une même variété — on choisit celle qui est la plus simple à calculer.

> [!note]- Lien avec l'exponentielle et les géodésiques
> L'application **exponentielle** $\exp_x : T_x M \to M$ est une rétraction particulière : elle suit la **géodésique** (la "ligne droite" sur la variété) partant de $x$ avec vitesse initiale $v$. C'est la rétraction théoriquement la plus "naturelle", mais elle est souvent **chère à calculer** (exponentielles de matrices, équations différentielles).
>
> En pratique on préfère une rétraction quelconque (plus simple à coder), parce que la convergence de l'algorithme ne dépend pas vraiment du choix de $R_x$ tant que les conditions ci-dessus sont vérifiées.

## II. Rétraction sur la sphère

Sur la sphère $S^{n-1}$, la rétraction la plus naturelle est la **normalisation** :

$$\boxed{R_x(v) = \frac{x + v}{\|x + v\|}}$$

**Intuition géométrique.** Tu pars de $x$, tu fais un pas $v$ dans le plan tangent, tu arrives au point $x + v$ (hors de la sphère). Pour ramener ce point sur la sphère, tu le **divise par sa norme** — ce qui le projette sur la sphère le long du rayon issu de l'origine.

![[Pasted image 20260511170613.png|357]]
*Figure 1. Rétraction sur la sphère par normalisation. Le pas tangent $\alpha v$ amène à $x + \alpha v$ (hors de la sphère). La rétraction $R_x$ ramène ce point sur la sphère en le divisant par sa norme.*

**Vérification des deux propriétés.**

- $R_x(0) = x / \|x\| = x$ (car $\|x\| = 1$). ✓
- Pour la dérivée en 0 : en développant pour $v$ petit, $\|x + v\|^2 = 1 + 2\langle x, v\rangle + \|v\|^2 = 1 + O(\|v\|^2)$ (car $\langle x, v\rangle = 0$ puisque $v$ est tangent). Donc $\|x + v\| = 1 + O(\|v\|^2)$, et $R_x(v) = (x + v)(1 + O(\|v\|^2)) = x + v + O(\|v\|^2)$. La dérivée en 0 est bien l'identité. ✓

**Coût computationnel** : un produit scalaire + une racine carrée + une division. $O(n)$ flops. Trivial.

## III. Rétraction sur Stiefel — la décomposition QR

C'est la rétraction qu'on utilise pour le challenge QRT. Pour $A \in V_F(\mathbb{R}^D)$ et $V \in T_A V_F$, on définit :

$$\boxed{R_A(V) = Q \text{ où } A + V = QR \text{ (décomposition QR)}}$$

**Intuition.** Tu pars de $A$ (une matrice à colonnes orthonormales), tu fais un pas $V$ qui te donne $A + V$ — une matrice dont les colonnes ne sont plus orthonormales en général. Pour la "remettre sur Stiefel", tu fais une **décomposition QR** et tu gardes le facteur $Q$ : c'est la matrice orthonormale la plus proche de $A + V$.

> [!note]- Rappel sur la décomposition QR
> Toute matrice $M \in \mathbb{R}^{D \times F}$ de rang plein s'écrit de manière unique $M = QR$ avec :
> - $Q \in \mathbb{R}^{D \times F}$ à colonnes orthonormales ($Q^\top Q = I_F$, donc $Q \in V_F$)
> - $R \in \mathbb{R}^{F \times F}$ triangulaire supérieure à diagonale positive
>
> En NumPy : `Q, R = np.linalg.qr(M)`. Coût : $O(D F^2)$.

**Pourquoi ça marche.** Quand on développe $A + V$ avec $V$ petit, $A^\top(A+V) = I_F + A^\top V$. Comme $V$ est tangent, $A^\top V$ est anti-symétrique, donc proche de 0 dans la partie symétrique. La décomposition QR "corrige" la matrice pour qu'elle redevienne dans Stiefel.

**Coût computationnel** : $O(DF^2)$. Pour le challenge QRT ($D=250, F=10$) : environ $25\,000$ flops par rétraction. Trivial même répété des milliers de fois.

> [!note]- Alternative : la rétraction polaire
> Une autre rétraction populaire sur Stiefel est la **rétraction polaire** :
> $$R_A(V) = (A + V) \big((A+V)^\top (A+V)\big)^{-1/2}$$
> Elle correspond au facteur $U$ de la décomposition polaire $A+V = UP$. Elle a de meilleures propriétés théoriques mais coûte légèrement plus cher (SVD ou racine matricielle). En pratique, la rétraction QR suffit largement pour la convergence.

## IV. Rétraction sur SPD

Pour $X \in \text{SPD}(n)$ et $V \in \text{Sym}(n)$, la rétraction la plus simple est :

$$R_X(V) = X + V \quad \text{(si } X + V \succ 0\text{)}$$

C'est l'**identité** : on ne fait rien, sauf qu'on doit vérifier que $X + V$ reste défini positif. Pour un pas assez petit, c'est garanti.

Si on veut une rétraction qui **garantit** toujours d'être dans SPD (peu importe la taille du pas), on utilise l'**exponentielle matricielle** :

$$R_X(V) = X^{1/2} \exp(X^{-1/2} V X^{-1/2}) X^{1/2}$$

C'est l'exponentielle riemannienne de la métrique affine-invariante. Plus chère mais robuste.

## V. L'algorithme de descente de gradient riemannienne

On a maintenant **toutes les briques**. L'algorithme complet :

> [!example] Descente de gradient riemannienne
> **Entrée** : fonction $f : M \to \mathbb{R}$ à minimiser, point initial $x_0 \in M$, pas $\alpha > 0$, nombre d'itérations $K$.
>
> **Pour $k = 0, 1, 2, \dots, K-1$ :**
>
> 1. **Gradient euclidien** : calculer $\nabla \bar{f}(x_k)$
> 2. **Projection sur l'espace tangent** : $g_k = \text{Proj}_{T_{x_k} M}\big(\nabla \bar{f}(x_k)\big) = \text{grad}_M f(x_k)$
> 3. **Pas tangent** : $v_k = -\alpha g_k \in T_{x_k} M$
> 4. **Rétraction** : $x_{k+1} = R_{x_k}(v_k) \in M$
>
> **Sortie** : $x_K$ (approximation du minimum).

Le pas $\alpha$ peut être fixe ou choisi par line search. Pour de meilleures performances on peut aussi utiliser des variantes : **gradient conjugué riemannien** (souvent largement plus rapide), **L-BFGS riemannien**, **Adam riemannien**. Toutes utilisent les mêmes briques (projection + rétraction).

![[Pasted image 20260511170921.png|367]]
*Figure 2. L'algorithme en action sur la sphère. On minimise $f(x) = x_3$ (la hauteur), donc le minimum est au pôle sud. Le gradient euclidien $\nabla \bar{f} = (0,0,1)$ est **constant** et pointe vers le haut — dans le mauvais sens pour notre problème ! Pourtant la trajectoire (verte) **descend** correctement vers le pôle sud. C'est précisément parce qu'à chaque étape, la **projection** sur le plan tangent change l'orientation effective du gradient, et la **rétraction** maintient le point sur la sphère.*

## VI. Récapitulatif : sphère, Stiefel, SPD

| Variété | Espace tangent $T_x M$ | Projection $\text{Proj}_{T_x M}(W)$ | Rétraction $R_x(v)$ |
| :--- | :--- | :--- | :--- |
| Sphère $S^{n-1}$ | $\{v : \langle x, v\rangle = 0\}$ | $W - \langle x, W\rangle x$ | $(x+v) / \|x+v\|$ |
| Stiefel $V_F(\mathbb{R}^D)$ | $\{V : A^\top V \text{ anti-sym.}\}$ | $W - A \cdot \text{sym}(A^\top W)$ | $\text{qr}(A + V).Q$ |
| SPD$(n)$ | $\text{Sym}(n)$ | $\tfrac{1}{2}(W + W^\top)$ | $X + V$ (petit pas) |

C'est tout. Avec ces trois lignes, tu peux faire de l'optimisation sur n'importe laquelle de ces variétés.

## VII. Application au challenge QRT

Rappel du problème : maximiser

$$\text{Métrique}(A, \beta) = \frac{1}{504} \sum_{t=250}^{753} \frac{\langle \tilde{S}_t(A, \beta), \tilde{R}_t\rangle}{\|\tilde{S}_t\| \|\tilde{R}_t\|}$$

avec $A \in V_{10}(\mathbb{R}^{250})$ (contrainte Stiefel) et $\beta \in \mathbb{R}^{10}$ (sans contrainte).

**Stratégie d'optimisation alternée :**

> [!example] Algorithme QRT (esquisse)
> **Initialisation** : $A_0$ tiré uniformément sur Stiefel (via QR d'une gaussienne), $\beta_0 = 0$.
>
> **Pour $k = 0, 1, \dots$ :**
>
> 1. **Fixer $A_k$, optimiser $\beta$** : à $A$ fixé, le score $\text{Métrique}(A_k, \beta)$ est différentiable en $\beta$. On peut soit faire de la descente euclidienne classique, soit (mieux) résoudre approximativement en utilisant le fait que la métrique est une moyenne de cosinus similaires (problème de régression non-linéaire).
> 2. **Fixer $\beta_{k+1}$, optimiser $A$ par descente riemannienne sur Stiefel** :
>    - Calculer le gradient ambient $G = \nabla_A \text{Métrique}(A_k, \beta_{k+1}) \in \mathbb{R}^{250 \times 10}$
>    - Projeter sur $T_{A_k}$ : $\tilde{G} = G - A_k \cdot \text{sym}(A_k^\top G)$
>    - Faire un pas tangent : $V = \alpha \tilde{G}$ (avec $\alpha$ bien choisi)
>    - Rétracter : $A_{k+1} = \text{qr}(A_k + V).Q$
>
> **Critère d'arrêt** : score qui stagne, ou nombre d'itérations max.

**Comparaison avec le benchmark QRT.** Le benchmark fait $N_{\text{iter}} = 1000$ tirages uniformes sur Stiefel + fit de $\beta$. C'est une recherche **aléatoire**, $O(N_{\text{iter}} \cdot \text{coût\_fit})$, sans aucune exploitation du gradient.

L'approche par descente riemannienne :
- **Calcule le gradient** (information de premier ordre, exploite la structure du problème)
- **Coût par itération** : un appel au gradient + QR (≈ $DF^2 = 25\,000$ flops)
- **Convergence** : typiquement quelques dizaines à quelques centaines d'itérations pour un optimum local

Multi-restart pour échapper aux minima locaux : on peut faire 10-50 descentes depuis des initialisations différentes (toujours moins cher que 1000 tirages aveugles). En pratique, ça **bat le benchmark** facilement.

**Code prêt à l'emploi.** La librairie `pymanopt` implémente exactement cet algorithme (avec gradient conjugué et line search). Le code ressemble à :

```python
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient

manifold = Stiefel(250, 10)

@pymanopt.function.autograd(manifold)
def cost(A):
    # implémenter ici -Métrique(A, β) pour minimiser
    ...

problem = pymanopt.Problem(manifold, cost)
optimizer = ConjugateGradient()
A_opt = optimizer.run(problem).point
```

## VIII. Récap final — la boîte à outils complète

Trois notes (variétés plongées, vecteurs tangents, métrique et gradient) plus celle-ci suffisent pour faire de l'optimisation rigoureuse sur Stiefel/SPD/sphère :

1. **Une variété**, c'est un sous-ensemble lisse de $\mathbb{R}^n$ défini par contrainte.
2. **L'espace tangent** $T_x M$ donne les directions admissibles. Formule : $T_x M = \ker Dg(x)$.
3. **Le gradient riemannien** est la projection du gradient euclidien sur $T_x M$.
4. **La rétraction** ramène un pas tangent sur la variété.

Avec ces quatre briques, l'algorithme de descente s'écrit en 4 lignes — et il **bat** les approches naïves de tirage uniforme comme celle du benchmark QRT.

→ Prochaine étape suggérée : créer une note pratique d'**implémentation du challenge QRT** avec `pymanopt`, en mettant en pratique toute cette théorie.
