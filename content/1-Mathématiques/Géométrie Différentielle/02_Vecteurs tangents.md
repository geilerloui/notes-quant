---
title: Vecteurs tangents
date: 2026-05-11
tags: [mathématiques, géométrie-différentielle, espace-tangent]
---

## L'idée fondatrice

On est au point $x$ d'une variété $M \subset \mathbb{R}^n$. On veut faire de l'optimisation — donc bouger, faire un pas. **Mais on ne peut pas bouger dans n'importe quelle direction** : il faut rester sur $M$.

Sur la sphère $S^2$, si je suis au pôle nord et que je veux avancer, j'ai le droit d'aller vers l'est, vers l'ouest, vers n'importe quel point du cercle de l'équateur — mais **pas vers le haut** (je sortirais de la sphère) ni vers le bas (pareil). Les directions admissibles forment un **plan** : le plan tangent à la sphère au pôle nord.

C'est exactement la notion d'**espace tangent** $T_x M$ qu'on va définir. C'est l'ensemble des "directions admissibles" en $x$, et c'est l'objet central de tout ce qui suit (gradient, descente, rétraction).

## I. Approche par les courbes

Comment formaliser "direction admissible" ? L'idée la plus naturelle, c'est de regarder ce qui se passe quand une **fourmi se balade sur la variété** : sa vitesse instantanée en chaque point est, par construction, une direction admissible.

### Une courbe sur la variété

Une **courbe sur $M$** est une application $\gamma : (-\varepsilon, \varepsilon) \to M$ qui est lisse. C'est juste une fonction qui prend un nombre $t$ proche de 0 et te donne un point de la variété.

L'interprétation : **$t$ est le temps**, et $\gamma(t)$ est la position de la fourmi à l'instant $t$. Le fait que $\gamma$ reste sur $M$ veut dire que la fourmi ne quitte jamais la variété.

![[Pasted image 20260511151001.png|265]]
*Figure 1. Une courbe $\gamma : \mathbb{R} \to S^2$ qui se promène sur la sphère. À l'instant $t = 0$, la fourmi est au point $x = \gamma(0)$, et son **vecteur vitesse** à cet instant est $\gamma'(0)$ — une flèche tangente à la sphère.*

### Le vecteur vitesse $\gamma'(0)$

Si $\gamma(t) = (\gamma_1(t), \dots, \gamma_n(t)) \in \mathbb{R}^n$, alors on dérive **composante par composante** en se posant à l'instant $t=0$ : 

$$\gamma'(0) = (\gamma_1'(0), \dots, \gamma_n'(0)) \in \mathbb{R}^n$$

C'est juste la dérivée d'une fonction $\mathbb{R} \to \mathbb{R}^n$, rien d'exotique. Le résultat est un **vecteur** de l'espace ambient.

> [!note]- Vocabulaire : vitesse, vecteur vitesse, vitesse scalaire
> Le mot "vitesse" en français est ambigu, il faut bien distinguer :
> - Le **vecteur vitesse** (*velocity* en anglais) : c'est $\gamma'(0) \in \mathbb{R}^n$, un vecteur qui contient la direction du mouvement et la rapidité.
> - La **vitesse scalaire** (*speed* en anglais) : c'est $\|\gamma'(0)\|$, un nombre positif, la rapidité sans la direction.
> - La **direction pure** : c'est $\gamma'(0) / \|\gamma'(0)\|$, un vecteur unitaire qui ne donne que l'orientation.
>
> Dans toute la suite, quand on dit "vitesse" sans préciser, on entend **vecteur vitesse**.

### Exemple concret : l'équateur de $S^2$

Prenons la courbe la plus simple possible sur la sphère : l'**équateur**, parcouru à vitesse 1 :
$$
\gamma(t) = (\cos t, \, \sin t, \, 0)
$$
Cette courbe reste bien sur la sphère : $\|\gamma(t)\|^2 = \cos^2 t + \sin^2 t + 0 = 1$. ✓

**Calcul de la position et de la vitesse à $t = 0$ :**

- $\gamma(0) = (\cos 0, \sin 0, 0) = (1, 0, 0)$. C'est notre point de départ $x$.
- $\gamma'(t) = (-\sin t, \cos t, 0)$ (dérivée composante par composante)
- $\gamma'(0) = (-\sin 0, \cos 0, 0) = (0, 1, 0)$. C'est le vecteur vitesse en $x$.

![[Pasted image 20260511151246.png|248]]
*Figure 2. L'équateur $\gamma(t) = (\cos t, \sin t, 0)$ trace un grand cercle sur la sphère. À l'instant $t = 0$, on est en $x = (1, 0, 0)$ avec vitesse $\gamma'(0) = (0, 1, 0)$ pointant dans la direction $y$.*

**Vérification visuelle.** Le vecteur $\gamma'(0) = (0, 1, 0)$ pointe dans la direction $y$, ce qui est bien la direction tangente à l'équateur au point $(1, 0, 0)$. Et on peut vérifier algébriquement qu'il est orthogonal à $x$ :

$$\langle x, \gamma'(0) \rangle = \langle (1,0,0), (0,1,0) \rangle = 0 \quad ✓$$

C'est notre premier vecteur tangent à la sphère.

### Pourquoi un point peut avoir plusieurs vecteurs tangents

L'équateur n'est qu'**une** courbe possible passant par $(1, 0, 0)$. Si je considère une autre courbe qui passe aussi par ce point mais part dans une direction différente, j'obtiens un autre vecteur tangent.

![[Pasted image 20260511151512.png|247]]
*Figure 3. Trois courbes différentes passant par $x$ à l'instant $t = 0$ donnent trois vecteurs vitesse différents — trois vecteurs tangents distincts. L'**ensemble de toutes les vitesses possibles** (pour toutes les courbes imaginables passant par $x$) forme l'espace tangent $T_x S^2$.*

### Définition formelle

Un **vecteur tangent à $M$ en $x$** est un vecteur $v \in \mathbb{R}^n$ qui s'écrit

$$v = \gamma'(0)$$

pour une certaine courbe $\gamma : (-\varepsilon, \varepsilon) \to M$ vérifiant $\gamma(0) = x$.

L'**espace tangent** $T_x M$ est l'ensemble de tous ces vecteurs : *toutes les vitesses possibles que peut avoir une fourmi qui passe par $x$*.

## II. La caractérisation pratique — par la contrainte

L'approche par courbes est intuitive mais pas calculatoire : on ne va pas exhiber une courbe à chaque fois qu'on veut un vecteur tangent. Il y a une caractérisation bien plus efficace, qui exploite la définition de $M$ par contrainte.

**Rappel.** Notre variété est définie par $M = \{x \in \mathbb{R}^n \mid g(x) = 0\}$, avec $g : \mathbb{R}^n \to \mathbb{R}^m$.

**Théorème.** Pour tout $x \in M$,

$$\boxed{T_x M = \ker Dg(x) = \{v \in \mathbb{R}^n \mid Dg(x) \cdot v = 0\}}$$

C'est **la** formule à retenir. L'espace tangent en $x$, c'est le **noyau de la jacobienne** de la contrainte en $x$.

> [!note]- Démonstration (esquisse)
> **Sens direct.** Soit $v = \gamma'(0)$ avec $\gamma(t) \in M$, donc $g(\gamma(t)) = 0$ pour tout $t$. En dérivant cette identité par rapport à $t$ en $t = 0$, par la règle de la chaîne :
> $$\frac{d}{dt}\bigg|_{t=0} g(\gamma(t)) = Dg(\gamma(0)) \cdot \gamma'(0) = Dg(x) \cdot v = 0$$
> Donc $v \in \ker Dg(x)$.
>
> **Sens réciproque.** Si $v \in \ker Dg(x)$, on construit une courbe $\gamma$ avec $\gamma(0) = x$ et $\gamma'(0) = v$ qui reste sur $M$, en utilisant le théorème des fonctions implicites (c'est ici qu'on utilise la régularité, i.e. que $Dg(x)$ est de rang plein).

**Pourquoi c'est génial.** Calculer $\ker Dg(x)$, c'est de l'**algèbre linéaire pure** : on calcule la jacobienne (un objet concret de $\mathbb{R}^{m \times n}$) et on en prend le noyau. Plus besoin d'imaginer des courbes — on a une formule directe et calculable.

**Conséquence importante.** $T_x M$ est un **sous-espace vectoriel** de $\mathbb{R}^n$, de dimension $n - m$ — c'est-à-dire la même dimension que la variété $M$. Cohérent : il faut bien le même nombre de degrés de liberté pour décrire les directions admissibles que pour décrire la variété elle-même.

## III. Exemple central — la sphère

Reprenons la sphère $S^{n-1} = \{x \in \mathbb{R}^n \mid \|x\|^2 - 1 = 0\}$ avec $g(x) = \|x\|^2 - 1$.

**Calcul de la jacobienne.** $g$ est une fonction scalaire ($m = 1$), sa "jacobienne" est juste son gradient (en ligne) :

$$Dg(x) = \nabla g(x)^\top = 2 x^\top$$

**Espace tangent.** Un vecteur $v$ est tangent à $S^{n-1}$ en $x$ si et seulement si $2 x^\top v = 0$, soit :

$$T_x S^{n-1} = \{v \in \mathbb{R}^n \mid \langle x, v \rangle = 0\} = x^\perp$$

**L'espace tangent à la sphère en $x$, c'est l'hyperplan orthogonal à $x$.** Exactement ce qu'on s'attend à voir géométriquement.

![[Pasted image 20260511151836.png|226]]
*Figure 4. Espace tangent à la sphère $S^2$ en un point $x$. Le plan tangent $T_x S^2$ est l'ensemble des vecteurs orthogonaux à $x$ — c'est l'hyperplan tangent à la sphère en ce point.*

Pour $S^2 \subset \mathbb{R}^3$, $T_x S^2$ est un **plan** de dimension 2, comme prévu ($n - m = 3 - 1 = 2$).

## IV. Exemple central — Stiefel

C'est le calcul important pour le challenge QRT. La contrainte est $g(A) = A^\top A - I_F$, à valeurs dans les matrices symétriques $F \times F$.

**Calcul de la jacobienne.** On dérive $A \mapsto A^\top A$ dans une direction $V \in \mathbb{R}^{D \times F}$ :

$$\frac{d}{dt}\bigg|_{t=0} (A + tV)^\top (A + tV) = V^\top A + A^\top V$$

Donc $Dg(A) \cdot V = V^\top A + A^\top V$, et $V$ est tangent ssi cette quantité est **nulle** :

$$\boxed{T_A V_F(\mathbb{R}^D) = \{V \in \mathbb{R}^{D \times F} \mid A^\top V + V^\top A = 0\}}$$

Autrement dit, $V$ est tangent à Stiefel en $A$ si et seulement si $A^\top V$ est **anti-symétrique**.

> [!note]- Décomposition explicite des vecteurs tangents de Stiefel
> Pour mieux visualiser, on peut décomposer un $V \in T_A V_F(\mathbb{R}^D)$ comme :
> $$V = A \Omega + A_\perp K$$
> où $\Omega \in \mathbb{R}^{F \times F}$ est **anti-symétrique** ($\Omega^\top = -\Omega$), $A_\perp \in \mathbb{R}^{D \times (D-F)}$ complète $A$ en une base orthonormale de $\mathbb{R}^D$, et $K \in \mathbb{R}^{(D-F) \times F}$ est quelconque.
>
> **Vérification de la dimension** : $\Omega$ a $F(F-1)/2$ degrés de liberté (matrice anti-symétrique $F \times F$), et $K$ en a $(D-F) \cdot F$. Au total :
> $$\frac{F(F-1)}{2} + (D-F) F = DF - \frac{F(F+1)}{2}$$
> Ce qui correspond bien à la dimension de Stiefel calculée dans la note précédente. ✓

**Pour le challenge QRT** : $A \in V_{10}(\mathbb{R}^{250})$, et $T_A$ est un espace vectoriel de dimension $2445$ dans $\mathbb{R}^{250 \times 10}$ (qui est de dimension $2500$). Quand on voudra faire un pas de descente, on projettera le gradient ambient sur ce sous-espace — calcul concret qu'on verra dans la note suivante.

## V. Exemple — SPD

L'ensemble SPD$(n)$ est un **ouvert** dans l'espace des matrices symétriques (pas de contrainte d'égalité). Conséquence directe :

$$T_X \text{SPD}(n) = \text{Sym}(n) = \{V \in \mathbb{R}^{n \times n} \mid V^\top = V\}$$

L'espace tangent en *tout point* est l'espace entier des matrices symétriques. C'est un cas plus simple que sphère ou Stiefel : on peut prendre n'importe quelle perturbation symétrique, tant qu'on reste assez près pour conserver la positivité (mais ça, ça concerne la rétraction, pas l'espace tangent).

## VI. Vue d'ensemble

| Variété | Contrainte | $Dg(x)$ | $T_x M$ |
| :--- | :--- | :--- | :--- |
| Sphère $S^{n-1}$ | $\|x\|^2 = 1$ | $2 x^\top$ | $\{v : \langle x, v\rangle = 0\}$ |
| Stiefel $V_F(\mathbb{R}^D)$ | $A^\top A = I_F$ | $V \mapsto V^\top A + A^\top V$ | $\{V : A^\top V \text{ anti-sym.}\}$ |
| SPD$(n)$ | (ouvert) | — | Sym$(n)$ |

**Trois idées à retenir :**

1. **L'espace tangent $T_x M$ est l'ensemble des directions admissibles** en $x$ — celles dans lesquelles on peut bouger sans quitter $M$. Concrètement, c'est l'ensemble des vecteurs vitesse $\gamma'(0)$ pour toutes les courbes $\gamma$ qui passent par $x$ en $t=0$.
2. **C'est un sous-espace vectoriel** de l'espace ambient $\mathbb{R}^n$, de même dimension que $M$.
3. **Formule pratique** : $T_x M = \ker Dg(x)$. Pour la sphère c'est l'orthogonal à $x$ ; pour Stiefel c'est la condition "$A^\top V$ anti-symétrique".

## VII. Vers la métrique riemannienne

Maintenant qu'on sait *quelles directions* sont admissibles, on a besoin de savoir comment **mesurer** ces directions : leur longueur, l'angle entre deux d'entre elles. C'est ce qu'apporte la **métrique riemannienne** — un produit scalaire sur chaque $T_x M$. Et c'est avec cette métrique qu'on définit le **gradient riemannien**, qui est l'objet final dont on a besoin pour faire de l'optimisation.

→ Suite : [[03_Métrique et gradient riemannien]]
