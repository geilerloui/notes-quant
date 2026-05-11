---
title: Le problème de Monge et Kantorovich
date: 2026-05-11
tags: [mathématiques, transport-optimal, monge, kantorovich]
---

## L'idée fondatrice

Comment **transporter** une distribution de masse vers une autre, avec un **effort minimal** ? C'est la question que se pose Gaspard Monge en 1781, à propos du problème des déblais et remblais sur les chantiers militaires : on a un tas de terre quelque part, on veut le déplacer pour combler un trou ailleurs, et on cherche à minimiser le travail total.

Cette question simple est devenue centrale en data science :
- Comparer deux distributions (distance de Wasserstein) — base de WGAN, generative models
- Adapter un dataset à un autre (domain adaptation)
- Calculer une "moyenne" de distributions (Wasserstein barycenters)
- Calibrer des modèles robustement (DRO)

Cette note pose les bases : les deux formulations fondamentales du problème, due à **Monge** (1781) puis **Kantorovich** (1942).

## I. Le problème de Monge

### L'image de la pelle et du sol

On a un tas de terre rouge représenté par une distribution $\mu$. À chaque point $x$ du sol, $\mu(x)$ donne la **hauteur de terre** en ce point. On veut déplacer cette terre dans un trou bleu, représenté par $\nu$, qui décrit la hauteur de terre attendue à chaque point d'arrivée.

Pour faire ce travail, on prend une pelle au point $x$, on charge une quantité de terre proportionnelle à $\mu(x)$, et on l'envoie au point $T(x)$. On répète jusqu'à ce que tout soit transporté.

![[im2 1.png|385]]


L'effort qu'on fait pour déplacer la pelle du point $x$ au point $T(x)$, on le note $D(x, T(x))$ — c'est la **distance de déplacement** (ou plus tard, le coût).

### Définir le travail

Le **travail élémentaire** pour un coup de pelle au point $x$ est le produit "quantité de terre déplacée × distance parcourue" :

$$\boxed{\text{travail}\ : \ \mu(x) \cdot D(x, T(x))}$$

Le **travail total** est l'intégrale de cet effort sur toute la distribution source :

$$\int D(x, T(x)) \, \mu(dx)$$

C'est ce qu'on cherche à minimiser.

### Que doit vérifier l'application T ?

L'application $T$ n'est pas n'importe quelle fonction. Elle doit **transporter effectivement** tout le tas de terre dans le trou.

Comme la terre se conserve, le tas et le trou ont la même quantité totale de terre :

$$\mu(\Omega_s) = \nu(\Omega_t)$$

(où $\Omega_s$ et $\Omega_t$ sont les domaines où vivent $\mu$ et $\nu$ : par exemple deux intervalles sur la droite, ou deux régions du plan)

Maintenant, supposons que je regarde un petit intervalle $B$ dans le trou. Combien de terre y arrive ? Toute la terre qui vient des points $x$ tels que $T(x) \in B$.

On note $T^{-1}(B) = \{x : T(x) \in B\}$ l'ensemble des points sources qui sont envoyés dans $B$. Cet ensemble peut très bien être l'union de plusieurs morceaux disjoints :

$$T^{-1}(B) = A_1 \cup A_2 \cup A_3$$

(des points pris à des endroits différents du tas peuvent atterrir dans le même intervalle $B$)

![[im3 (1) 1.png|448]]
*Trois morceaux $A_1, A_2, A_3$ du tas (rouge) sont envoyés par $T$ dans le même intervalle $B$ du trou (bleu). La terre qui arrive dans $B$ vient de ces trois morceaux.*

La **condition de conservation** s'écrit alors : la quantité totale de terre qui arrive dans $B$ doit être égale à la hauteur prescrite par $\nu$ sur $B$ :

$$\boxed{\mu(A_1) + \mu(A_2) + \mu(A_3) = \nu(B)}$$

ou, en notation compacte :

$$\boxed{\mu(T^{-1}(B)) = \nu(B) \quad \forall B}$$

Quand cette condition est vérifiée, on dit que $T$ **pousse $\mu$ sur $\nu$**, et on écrit :

$$T_\# \mu = \nu$$

(le symbole $\#$ se lit "push-forward")

### La formulation de Monge

On a maintenant tous les ingrédients. Le problème de Monge (1781) s'écrit :

$$\boxed{\inf_{T_\# \mu = \nu} \int D(x, T(x)) \, \mu(dx)}$$

Trouver l'application $T$ qui transporte $\mu$ sur $\nu$ avec le travail total minimal.

> [!note]- Vocabulaire : "coût" plutôt que "distance"
> En pratique, $D(x, y)$ ne représente pas forcément une distance au sens géométrique. On parle plutôt d'un **coût** $c(x, y)$ qui peut être n'importe quelle fonction positive. Le choix le plus courant est $c(x, y) = \|x - y\|^2$ (coût quadratique), parce qu'il a des propriétés mathématiques très utiles (voir note 02 sur Brenier). Mais on peut aussi prendre $c(x, y) = \|x - y\|$ (lié à WGAN, note 04) ou des coûts non géométriques.

## II. Les limitations de Monge

Le problème de Monge a l'air bien posé, mais il a **deux problèmes sérieux**.

### Non-existence : impossible de séparer une masse

Monge force chaque grain de terre à aller à **une seule** destination — l'application $T$ est déterministe : $T(x)$ est un point unique. Cela rend impossible de **séparer** une masse concentrée.

![[Pasted image 20260511180158.png]]
*À gauche : un Dirac source $\delta_{x_0}$ (toute la masse concentrée en un seul point) doit être transporté vers deux cibles $\tfrac{1}{2}\delta_{y_1} + \tfrac{1}{2}\delta_{y_2}$. Aucune application $T$ ne peut le faire : à un point on ne peut associer qu'une seule destination. À droite : fusionner deux masses vers une seule est possible avec $T(x_1) = T(x_2) = y_0$.*

Conséquence : si $\mu$ contient un atome (toute la masse concentrée en un point), Monge ne peut pas la diffuser. **Le problème n'a pas toujours de solution.**

### Non-unicité

Même quand une solution existe, elle n'est **pas toujours unique**. On peut avoir deux applications $f$ et $f'$ différentes qui transportent $\mu$ vers $\nu$ avec exactement le même coût total.

Cela rend le problème difficile à résoudre numériquement : pas de garantie d'unicité, pas d'algorithme convexe.

> [!note]- Un cas où Monge marche très bien : le théorème de Brenier
> Pour le coût quadratique $c(x, y) = \|x - y\|^2$ et des distributions à densité (continues), Brenier (1991) montre qu'il **existe et est unique** une application optimale $T$, et qu'elle s'écrit $T = \nabla \varphi$ avec $\varphi$ convexe. Ce résultat sera l'objet de la note 02.

## III. Le problème de Kantorovich

Pour contourner les problèmes de Monge, Leonid Kantorovich propose en 1942 une formulation plus souple. On va y arriver en trois étapes.

### Étape 1 — La scène : la logistique en temps de guerre

Kantorovich travaille sur l'optimisation de la logistique militaire pendant la Seconde Guerre mondiale. La situation : on a des **casernes** à l'arrière avec des soldats en réserve, et des **positions au front** où il faut envoyer ces soldats.

| ![[im2-1 (2).png\|307]] | ![[im2-2.png\|310]] |
| ----------------------- | ------------------- |
*Trois casernes (rouge, en réserve) avec respectivement 60, 90 et 150 soldats. Trois positions au front (bleu) qui demandent 120, 90 et 90 soldats. Total : 300 = 300, la conservation est OK. **Question** : comment répartir les soldats pour minimiser la distance totale parcourue ?*

Une approche naïve serait de partager proportionnellement (chaque caserne envoie ses soldats selon les proportions demandées au front). Mais ce n'est **pas optimal** : on ferait voyager des soldats loin alors qu'ils pourraient aller dans une position plus proche.

### Étape 2 — La modélisation : matrice de transport et matrice de distance

Kantorovich introduit deux objets pour formaliser le problème.

**La matrice de transport** $P = (p_{ij})$ : c'est ce qu'on cherche. $p_{ij}$ représente **le nombre de soldats envoyés de la caserne $i$ vers la position $j$**.

**La matrice de distance** $D = (d_{ij})$ : c'est donné. $d_{ij}$ est la distance entre la caserne $i$ et la position $j$.

| ![[im2-3.png\|271]] | ![[im2-5.png\|396]] |
| ------------------- | ------------------- |
*À gauche : la matrice de transport $P$ qu'on cherche. À droite : la matrice de distance $D$ qui est donnée (les distances géographiques entre casernes et positions).*

**Les contraintes** sur $P$ portent sur les marginales :

$$\sum_j p_{ij} = a_i \qquad \text{(la caserne $i$ envoie tout son effectif $a_i$)}$$

$$\sum_i p_{ij} = b_j \qquad \text{(la position $j$ reçoit ce qu'elle demande, $b_j$)}$$

$$p_{ij} \geq 0 \qquad \text{(pas d'effectifs négatifs)}$$

La première équation est une **somme sur la ligne $i$** de $P$ ; la seconde une **somme sur la colonne $j$**.

**Le coût total** à minimiser : la distance parcourue par chaque soldat, sommée sur tous les soldats :

$$C(P) = \sum_{i,j} p_{ij} \cdot d_{ij}$$

### Étape 3 — La solution optimale

Voici une solution possible (pas unique) du problème :

![[im2-6.png|347]]
*La matrice de transport $P$ optimale : la caserne 3 (150 soldats) en envoie 90 vers la position A et 60 vers la position C ; la caserne 2 (90 soldats) en envoie 30 vers A et 60 vers B ; la caserne 1 (60 soldats) en envoie 30 vers B. Vérification : sommes des lignes = (60, 90, 150) ✓, sommes des colonnes = (120, 90, 90) ✓.*

Ce qui est crucial ici : **plusieurs casernes peuvent envoyer des soldats à la même position**, et **une caserne peut diviser son effectif entre plusieurs positions**. C'est exactement le **mass splitting** que Monge interdisait.

### Formulation continue : couplages

On peut généraliser cette idée de matrice de transport à des distributions continues. La matrice $P$ devient une **loi jointe** $\gamma$ sur le produit $\Omega_s \times \Omega_t$.

L'interprétation est la même que pour la matrice discrète : $\gamma(x, y)$ représente "la quantité de masse qui voyage de $x$ vers $y$". Les contraintes de marginales deviennent :

$$\int_{\Omega_t} \gamma(x, y) \, dy = \mu(x), \qquad \int_{\Omega_s} \gamma(x, y) \, dx = \nu(y)$$

(la première dit "toute la masse au point $x$ est répartie quelque part dans $\gamma(x, \cdot)$" ; la seconde dit "toute la masse qui arrive en $y$ vient de quelque part dans $\gamma(\cdot, y)$")

On note $\Pi(\mu, \nu)$ l'ensemble des $\gamma$ qui ont ces bonnes marginales : c'est l'ensemble des **couplages** entre $\mu$ et $\nu$.

Le **problème de Kantorovich** s'écrit alors :

$$\boxed{\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\Omega_s \times \Omega_t} c(x, y) \, \gamma(dx, dy)}$$

### Pourquoi Kantorovich résout les problèmes de Monge

| Problème de Monge | Solution Kantorovich |
| :--- | :--- |
| Pas de mass splitting (un Dirac ne peut pas se diviser) | $\gamma(x_0, \cdot)$ peut être une vraie distribution diffuse |
| Non-convexité (ensemble des $T$ non convexe) | $\Pi(\mu, \nu)$ est convexe + fonctionnelle linéaire en $\gamma$ |
| Pas d'existence garantie | Existence garantie sous hypothèses très faibles |

C'est un **problème d'optimisation linéaire** sous contraintes linéaires : numériquement très bien étudié, toujours soluble.

### Visualisation en 2D

Voici à quoi ressemble un couplage $\gamma$ entre deux nuages de points 2D :

![[im2-7.png]]
*Vision lagrangienne : chaque point source $x_i$ (bleu) est relié à un ou plusieurs points cibles $y_j$ (rouge) par des arêtes. Une arête entre $x_i$ et $y_j$ représente la masse $\gamma(x_i, y_j)$ transportée. L'épaisseur de l'arête est proportionnelle à cette masse.*

## IV. Trois cas particuliers

Selon la nature de $\mu$ et $\nu$, on distingue trois régimes :

| Cas | $\mu$ | $\nu$ | Outils typiques |
| :--- | :--- | :--- | :--- |
| **Discret** | Diracs (somme finie) | Diracs (somme finie) | Linear programming, Sinkhorn |
| **Continu** | Densité $f(x)$ | Densité $g(y)$ | EDP (Monge-Ampère) |
| **Semi-discret** | Densité | Diracs | Géométrie computationnelle |

![[im2-8.png|437]]
Figure Visualisation de ce que c'est les trois cas de optimal transport

En **data science** et **machine learning**, on travaille presque toujours dans le **cas discret** (échantillons finis tirés de distributions empiriques). Les notes suivantes se concentrent sur ce cas.

> [!note]- À propos du "semi" dans semi-discret
> Le "semi-discret" en OT n'a **rien à voir** avec le "semi-supervised" en ML. Ici, "semi" signifie qu'**une seule** des deux mesures est discrète (l'autre est continue). En ML, "semi-supervised" signifie que **certaines données sont labelisées** et d'autres pas.

## V. Trois idées à retenir

1. **Monge cherche une application $T$** qui envoie chaque point source vers une **unique** destination. Le coût est $\int D(x, T(x)) \, d\mu(x)$. Simple mais limité : pas de mass splitting, problème non-convexe, pas toujours de solution.

2. **Kantorovich cherche un couplage $\gamma$** — une loi jointe sur le produit des deux espaces — qui distribue probabilistement la masse. Le coût est $\int c(x, y) \, d\gamma(x, y)$. C'est un **problème linéaire**, convexe, et toujours soluble.

3. **Kantorovich généralise Monge**. Tout transport $T$ correspond à un couplage déterministe (concentré sur le graphe de $T$), mais Kantorovich autorise des couplages plus généraux. En particulier, $\inf_K \leq \inf_M$, et l'inégalité est stricte quand Monge n'a pas de solution.

## VI. Vers la suite

- **Note 02 — Théorème de Brenier** : pour le coût quadratique, l'optimum de Kantorovich est en fait un transport déterministe $T = \nabla \varphi$ avec $\varphi$ convexe.
- **Note 03 — Distance de Wasserstein** : la valeur optimale du coût définit une distance entre distributions, avec des propriétés bien plus riches que les distances euclidiennes ou KL.
- **Note 04 — Dualité de Kantorovich** : reformulation duale du problème, plus efficace numériquement et menant directement à WGAN.
- **Note 05 — Sinkhorn** : régularisation entropique et algorithme rapide pour calculer Wasserstein en pratique.
