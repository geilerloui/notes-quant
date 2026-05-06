---
title: Géométrie dans L²
date: 2026-03-21
tags: [probabilités, géométrie, L2, hilbert]
---

## L'idée fondatrice

Un mathématicien s'est posé une question simple : *et si les variables aléatoires se comportaient comme des vecteurs ?*

En algèbre linéaire, un vecteur $v = (v_1, \dots, v_n) \in \mathbb{R}^n$ a une norme naturelle :

$$\|v\| = \sqrt{\sum_i v_i^2}$$

Pour une variable aléatoire $X$, on cherche l'analogue. $X$ n'est pas un nombre fixe — c'est un résultat qui change à chaque tirage. Mais on peut quand même lui donner une "taille" :

$$\|X\|_{L^2} = \sqrt{E(X^2)} = \sqrt{\sum_i p_i x_i^2}$$

Et de la même façon, le produit scalaire $\langle v, w \rangle = \sum_i v_i w_i$ devient :

$$\langle X, Y \rangle_{L^2} = E(XY) = \sum_i p_i \, x_i \, y_i$$

<div style="display:flex; gap:2rem; align-items:center; margin: 1.5rem 0;">
<div style="flex:1">

| Géométrie dans $\mathbb{R}^n$ | Probabilités dans $L^2$ |
|---|---|
| Vecteur $v$ | Variable aléatoire $X$ |
| Composante $v_i$ | Valeur $x_i$ |
| Poids $1/n$ (uniforme) | Probabilité $p_i$ |
| $\langle v, w \rangle = \sum_i v_i w_i$ | $\langle X, Y \rangle = E(XY)$ |
| $\|v\|^2 = \sum_i v_i^2$ | $\|X\|^2 = E(X^2)$ |
| $\langle u, v \rangle = \|u\|\|v\|\cos\theta$ | $\text{Cov}(X,Y) = \langle \tilde{X}, \tilde{Y} \rangle$ |
| $\|u\|^2$ | $\text{Var}(X) = \|\tilde{X}\|^2$ |
| $\cos\theta = \frac{\langle u,v \rangle}{\|u\|\|v\|}$ | $\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$ |


</div>
<div style="flex:1">
<img src="images/geometrie-L2/im1.png" style="width:100%; max-width:100%"/>
<p style="text-align:center; font-size:0.85em; color:#8a8784; margin-top:0.5rem; font-style:italic;">Figure 1. Le produit scalaire et son analogue dans L². Chaque concept géométrique de Rⁿ admet une traduction naturelle en probabilités.</p>
</div>
</div>

## L'espace $L^2$

On ne peut pas travailler avec *toutes* les variables aléatoires. Il faut imposer $E(X^2) < \infty$. Pourquoi ? Parce que si $E(X^2)$ est infini, le produit scalaire $E(XY)$ peut exploser — et toute la géométrie s'effondre.

$L^2$ est donc l'espace des variables aléatoires *raisonnables* : celles pour lesquelles la géométrie tient.

Cet espace est un **espace de Hilbert** : il se comporte exactement comme $\mathbb{R}^3$. On peut projeter, décomposer, appliquer Pythagore. Deux sous-espaces jouent un rôle central :

- $\Delta$ : la droite des **variables constantes**
- $L^2_X$ : le sous-espace des **fonctions de $X$**, i.e. $\{\varphi(X) \mid \varphi : \mathbb{R} \to \mathbb{R}\} \cap L^2$

On a l'inclusion $\Delta \subset L^2_X$ — toute constante est une fonction de $X$.

## Trois applications immédiates

### (i) L'espérance comme projection

$E(X)$ est la **projection orthogonale de $X$ sur $\Delta$**. C'est la constante qui approche le mieux $X$ au sens $L^2$ :

$$E(X) = \underset{a \in \Delta}{\arg\min} \; E\left[(X - a)^2\right]$$

*Preuve.* On développe $E[(X-a)^2]$ :

$$E[(X-a)^2] = E[X^2] - 2a\,E[X] + a^2$$

En dérivant par rapport à $a$ et en annulant : $-2E[X] + 2a = 0$, donc $a = E(X)$. $\square$

La formule de König-Huyghens s'interprète comme le **théorème de Pythagore** appliqué au triangle rectangle $X$, $E(X)$, $a$ :

$$E\left[(X - a)^2\right] = \underbrace{V(X)}_{\|X - E(X)\|^2} + \underbrace{(E(X) - a)^2}_{\text{distance}^2}$$

![Projection de X sur Δ](images/1-Mathématiques/Probabilité/geometrie-L2/im2.png)
*Figure 2. Projection orthogonale de $X$ sur $\Delta$. Le pied de la perpendiculaire est $E(X)$, la constante qui minimise $E[(X-a)^2]$. La formule de König-Huyghens est le théorème de Pythagore appliqué à ce triangle rectangle.*

### (ii) La corrélation comme cosinus

On pose $\tilde{X} = X - E(X)$ et $\tilde{Y} = Y - E(Y)$ — les variables **centrées**.

**(a) Pourquoi $E(\tilde{X}) = 0$ ?**

C'est purement arithmétique. Si $X$ prend les valeurs 10, 12, 14 avec probabilités égales, alors $E(X) = 12$. En centrant :

$$\tilde{X} : \quad 10 - 12 = -2, \quad 12 - 12 = 0, \quad 14 - 12 = +2$$

$$E(\tilde{X}) = \frac{-2 + 0 + 2}{3} = 0$$

On a simplement translaté $X$ pour que sa moyenne soit en $0$. De même $E(\tilde{Y}) = 0$.

**(b) Identification avec le cosinus**

Posons les deux formules côte à côte :

$$\cos\theta = \frac{\langle u, v \rangle}{\|u\|\|v\|} \qquad \rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)}\sqrt{\text{Var}(Y)}}$$

Par identification, en utilisant $\tilde{X}$ et $\tilde{Y}$ :

$$\text{Cov}(X,Y) = E(\tilde{X}\tilde{Y}) = \langle \tilde{X}, \tilde{Y} \rangle_{L^2}$$

$$\text{Var}(X) = E(\tilde{X}^2) = \|\tilde{X}\|^2_{L^2}$$

On obtient donc :

$$\rho(X,Y) = \frac{\langle \tilde{X}, \tilde{Y} \rangle}{\|\tilde{X}\| \cdot \|\tilde{Y}\|} = \cos\theta$$

Le coefficient de corrélation est le **cosinus de l'angle** entre $\tilde{X}$ et $\tilde{Y}$ dans $L^2$.

**(c) Conséquences**

- $X$ et $Y$ non corrélés $\Longleftrightarrow$ $\tilde{X} \perp \tilde{Y}$ dans $L^2$
- $|\rho| = 1$ $\Longleftrightarrow$ $\tilde{X}$ et $\tilde{Y}$ colinéaires $\Longleftrightarrow$ relation linéaire parfaite

![Corrélation comme cosinus](images/1-Mathématiques/Probabilité/geometrie-L2/im3.png)
*Figure 3. Angle $\theta$ entre les variables centrées $\tilde{X}$ et $\tilde{Y}$ dans $L^2$. Note : le dessin original utilise $X$ et $Y$ non centrés — leurs projections sur $\Delta$ devraient tomber en $0$ pour être cohérentes avec $E(\tilde{X}) = E(\tilde{Y}) = 0$.*

### (iii) L'espérance conditionnelle comme projection

$E(Y \mid X)$ est la **projection orthogonale de $Y$ sur $L^2_X$**. C'est la meilleure approximation de $Y$ par une fonction de $X$, au sens $L^2$ :

$$E(Y \mid X) = \underset{\varphi(X) \in L^2_X}{\arg\min} \; E\left[(Y - \varphi(X))^2\right]$$

Le résidu $Y - E(Y \mid X)$ est orthogonal à $L^2_X$ — il est non corrélé avec toute fonction de $X$.

Le **théorème de la variance totale** est le théorème de Pythagore appliqué au triangle $Y$, $E(Y)$, $E(Y \mid X)$ :

$$\underbrace{V(Y)}_{\|Y - E(Y)\|^2} = \underbrace{V(E(Y \mid X))}_{\|E(Y|X) - E(Y)\|^2} + \underbrace{E[V(Y \mid X)]}_{\|Y - E(Y|X)\|^2}$$

Le théorème de l'espérance totale $E(Y) = E(E(Y \mid X))$ est un cas particulier du **théorème des trois perpendiculaires** : la projection de $Y$ sur $\Delta$ passe par la projection de $Y$ sur $L^2_X$.

![Espérance conditionnelle — projection sur L²_X](images/1-Mathématiques/Probabilité/geometrie-L2/im4.png)
*Figure 4. Projection orthogonale de $Y$ sur $L^2_X$. Le résidu $Y - E(Y \mid X)$ est perpendiculaire à tout le sous-espace $L^2_X$. Le théorème de la variance totale est le théorème de Pythagore appliqué au triangle $Y$, $E(Y)$, $E(Y \mid X)$.*