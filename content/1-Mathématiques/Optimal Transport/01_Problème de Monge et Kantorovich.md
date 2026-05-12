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

> [!warning] Hypothèse fondamentale : les deux distributions sont fixées
> Dans le problème de Monge, on **suppose connues à l'avance** les deux distributions $\mu$ (le tas) et $\nu$ (le trou). Dans la vraie vie, on creuserait le trou en fonction du tas — mais ici la **forme du trou est imposée**. La seule liberté qu'on a, c'est de choisir **comment** transporter la terre : trouver la fonction $T$ qui déplace chaque grain du tas vers une position du trou, en minimisant le coût total. Toute la difficulté du problème est dans le choix de $T$, pas dans le choix de $\nu$.

Pour faire ce travail, on prend une pelle au point $x$, on charge une quantité de terre proportionnelle à $\mu(x)$, et on l'envoie au point $T(x)$. On répète jusqu'à ce que tout soit transporté.

![[im2 1.png|385]]


L'effort qu'on fait pour déplacer la pelle du point $x$ au point $T(x)$, on le note $D(x, T(x))$ — c'est la **distance de déplacement** (ou plus tard, le coût).

### Définir le travail

Le **travail élémentaire** pour un coup de pelle au point $x$ est le produit "quantité de terre déplacée × distance parcourue" :

$$\boxed{\text{travail}\ : \ \mu(x) \cdot D(x, T(x))}$$

Le **travail total** est l'intégrale de cet effort sur toute la distribution source :

$$\int D(x, T(x)) \, \mu(dx)$$

C'est ce qu'on cherche à minimiser.

> [!note]- Notation $\mu(dx)$
> $\mu(dx)$ signifie "la masse contenue dans un voisinage infinitésimal autour de $x$". C'est une notation qui **unifie deux cas** :
> 
> - **Cas continu** (tas de terre avec densité $f$, notre exemple actuel) :
>   $$\mu(dx) = f(x) \, dx \quad \Rightarrow \quad \int D(x, T(x)) \, \mu(dx) = \int D(x, T(x)) \, f(x) \, dx$$
>   C'est l'intégrale classique : densité × largeur infinitésimale.
> 
> - **Cas discret** (Diracs $\mu = \sum_i a_i \delta_{x_i}$, plus tard dans la note avec Kantorovich) :
>   $$\int D(x, T(x)) \, \mu(dx) = \sum_i a_i \cdot D(x_i, T(x_i))$$
>   L'intégrale **devient une somme**.
> 
> Pour le problème du tas de terre, **$\mu(dx) = f(x) \, dx$** — c'est l'intégrale classique avec densité. La notation $\mu(dx)$ permet juste d'écrire une seule formule qui marchera aussi pour le cas discret plus loin.

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

Le problème de Monge a l'air bien posé, mais il a **deux problèmes sérieux**. Ces deux problèmes viennent d'une même propriété fondamentale : **$T$ n'est pas bijective en général**. Commençons par bien comprendre ce point.

### T n'est pas bijective

Avant de regarder les limitations, il faut clarifier ce qu'est $T$ et ce qu'elle n'est **pas**.

**$T$ est une fonction point-à-point** : pour chaque point $x$ du tas, $T(x)$ donne **l'unique** destination de ce grain de terre. C'est juste une fonction comme $f(x) = x^2$ — un seul $y$ pour chaque $x$.

Mais $T$ peut très bien envoyer **plusieurs points sources différents vers le même point cible**. C'est exactement ce qu'on a vu dans la section précédente avec $T^{-1}(B) = A_1 \cup A_2 \cup A_3$ : trois morceaux disjoints du tas se retrouvent dans la même zone $B$. Si $T$ était injective, on n'aurait qu'**un seul morceau** dans la préimage.

Formellement :

| Propriété | Définition | Vérifiée par $T$ ? |
| :--- | :--- | :---: |
| **Surjective au sens des mesures** | $T_\# \mu = \nu$ (toute la masse cible est servie) | ✓ obligatoire |
| **Injective** | $x_1 \neq x_2 \Rightarrow T(x_1) \neq T(x_2)$ | ✗ pas obligatoire |
| **Bijective** | Injective ET surjective | ✗ pas obligatoire |

**Conséquence importante** : $T$ peut **fusionner** des points (plusieurs $x$ vers un même $y$), mais elle ne peut pas **séparer** un point (un $x$ vers plusieurs $y$ — ce n'est pas une fonction). Cette asymétrie est à l'origine des deux problèmes ci-dessous.

> [!note]- Un cas où $T$ est bijective : Brenier
> Quand $\mu$ et $\nu$ ont des densités lisses (pas de Diracs) et qu'on utilise le coût quadratique, le théorème de Brenier (note 02) garantit que $T = \nabla \varphi$ avec $\varphi$ **strictement** convexe. Or le gradient d'une fonction strictement convexe est injectif. Combiné à la surjectivité (push-forward), cela donne une $T$ bijective. C'est exactement ce qu'on a vu en 1D avec $T(x) = G^{-1}(F(x))$ strictement croissante.

### Conséquence 1 : Mass splitting impossible (non-existence)

Puisque $T(x)$ est un **point unique**, il est impossible de **séparer** une masse concentrée : si toute la masse est en $x_0$, $T(x_0)$ ne peut pas l'envoyer en deux endroits différents simultanément.

![[Pasted image 20260511180158.png]]
*À gauche : un Dirac source $\delta_{x_0}$ (toute la masse concentrée en un seul point) doit être transporté vers deux cibles $\tfrac{1}{2}\delta_{y_1} + \tfrac{1}{2}\delta_{y_2}$. Aucune application $T$ ne peut le faire : à un point on ne peut associer qu'une seule destination. À droite : fusionner deux masses vers une seule est possible avec $T(x_1) = T(x_2) = y_0$ — la **fusion** est autorisée parce que $T$ n'a pas besoin d'être injective.*

Conséquence : si $\mu$ contient un atome (toute la masse concentrée en un point) et $\nu$ est diffuse, Monge ne peut pas transporter $\mu$ vers $\nu$. **Le problème n'a pas toujours de solution.**

### Conséquence 2 : Non-unicité

Même quand une solution existe, elle n'est **pas toujours unique**. On peut avoir deux applications $f$ et $f'$ différentes qui transportent $\mu$ vers $\nu$ avec exactement le même coût total.

Cela rend le problème difficile à résoudre numériquement : pas de garantie d'unicité, pas d'algorithme convexe.

## III. Le problème de Kantorovich

Pour contourner les problèmes de Monge, Leonid Kantorovich propose en 1942 une formulation plus souple. On va y arriver en trois étapes.

### Étape 1 — La scène : la logistique en temps de guerre

Kantorovich travaille sur l'optimisation de la logistique militaire pendant la Seconde Guerre mondiale. La situation : on a des **casernes** à l'arrière avec des soldats en réserve, et des **positions au front** où il faut envoyer ces soldats.

| ![[images/1-Mathématiques/Optimal transport/im2-1 (2).png\|307]] | ![[images/1-Mathématiques/Optimal transport/im2-2.png\|310]] |
| ----------------------- | ------------------- |
*Trois casernes (rouge, en réserve) avec respectivement 60, 90 et 150 soldats. Trois positions au front (bleu) qui demandent 120, 90 et 90 soldats. Total : 300 = 300, la conservation est OK. **Question** : comment répartir les soldats pour minimiser la distance totale parcourue ?*

Une approche naïve serait de partager proportionnellement (chaque caserne envoie ses soldats selon les proportions demandées au front). Mais ce n'est **pas optimal** : on ferait voyager des soldats loin alors qu'ils pourraient aller dans une position plus proche.

### Étape 2 — La modélisation : matrice de transport et matrice de distance

Kantorovich introduit deux objets pour formaliser le problème.

**La matrice de transport** $P = (p_{ij})$ : c'est ce qu'on cherche. $p_{ij}$ représente **le nombre de soldats envoyés de la caserne $i$ vers la position $j$**.

**La matrice de distance** $D = (d_{ij})$ : c'est donné. $d_{ij}$ est la distance entre la caserne $i$ et la position $j$.

| ![[images/1-Mathématiques/Optimal transport/im2-3.png\|271]] | ![[im2-5.png\|396]] |
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

### Exemple visuel : Kantorovich comme programme linéaire

Un point important à réaliser : la fonction de coût $C(P) = \sum_{ij} p_{ij} \cdot d_{ij}$ est **linéaire en $P$**, et les contraintes (sommes lignes/colonnes, positivité) sont elles aussi **linéaires**. Kantorovich est donc, littéralement, un **programme linéaire** au sens de tes notes d'optimisation (`7-optimisation\`). Tous les outils des LP s'appliquent directement : simplexe, méthode du point intérieur, théorème fondamental du LP.

![[Pasted image 20260512103429.png|548]]
*Vue 3D du problème de Kantorovich pour le cas 3 sources × 3 cibles avec masses uniformes. **Au sol** (plan $(x, y)$) : le polytope $\Pi(\mu, \nu)$ qui contient toutes les matrices de transport $P$ admissibles. **Au-dessus** : la fonction de coût $C(P)$, qui est un plan incliné puisque linéaire en $P$. **Chaque trait vertical** donne la valeur du coût à un sommet du polytope. **Le sommet vert** est l'optimum, **le rouge** est le pire. Tous les autres sommets sont des solutions admissibles mais sous-optimales.*

> [!warning] La figure est une représentation symbolique
> Pour le cas 3×3 avec masses uniformes, le vrai polytope s'appelle le **polytope de Birkhoff** $B_3$ : l'ensemble des matrices doublement stochastiques 3×3 (entrées positives, sommes lignes et colonnes = 1). Ce polytope a effectivement **6 sommets** = $3! = 6$ permutations (théorème de Birkhoff-von Neumann), mais il vit en **dimension 4** : 9 entrées dans la matrice − 5 contraintes indépendantes (3 lignes + 3 colonnes − 1 redondance car les sommes totales coïncident) = 4 degrés de liberté.
> 
> La figure ci-dessus est donc une **projection / représentation schématique** : on dessine l'hexagone régulier dans le plan $(x, y)$ pour pouvoir visualiser, mais en réalité ce n'est pas le "vrai" $B_3$ qu'on voit. La topologie est correcte (6 sommets, 12 arêtes, polytope convexe), seules les coordonnées $(x, y)$ sont arbitraires. L'important n'est pas la géométrie exacte mais l'**idée** : minimiser une fonction linéaire (le plan rose) sur un polytope (au sol) → optimum à un sommet.

C'est l'espace des matrices de transport $P$, avec $\mu$ et $\nu$ fixées.
Reprends l'exemple des casernes :
- $\mu=(60,90,150)$ (effectifs casernes) - fixé
- $\nu=(120,90,90)$ (besoins positions) - fixé

Avec ces deux marginales fixées, tu peux choisir plein de matrices $P$ différentes qui respectent les contraintes. Par exemple :

$$
P_1=\left(\begin{array}{ccc}
60 & 0 & 0 \\
60 & 30 & 0 \\
0 & 60 & 90
\end{array}\right) \quad \text { ou } \quad P_2=\left(\begin{array}{ccc}
30 & 30 & 0 \\
30 & 30 & 30 \\
60 & 30 & 60
\end{array}\right) \quad \text { etc. }
$$

Chaque matrice $P$ admissible = un point du polytope. Chaque "point" de mon hexagone, c'est une matrice de transport possible, pas une distribution de probabilité différente.

Pourquoi 6 sommets ? Pour $n=3$ casernes et 3 positions avec masses uniformes $(1 / 3,1 / 3,1 / 3)$ chacune, les sommets correspondent aux $3!=6$ permutations :
- $\sigma_1=(1,2,3)$ : caserne $1 \rightarrow$ position 1 , caserne $2 \rightarrow$ position 2 , caserne $3 \rightarrow$ position 3
- $\sigma_2=(1,3,2)$ : caserne $1 \rightarrow$ position 1 , caserne $2 \rightarrow$ position 3, caserne $3 \rightarrow$ position 2
- $\sigma_3=(2,1,3)$ : caserne $1 \rightarrow$ position 2, caserne $2 \rightarrow$ position 1, caserne $3 \rightarrow$ position 3
- $\sigma_4=(2,3,1)$ : caserne $1 \rightarrow$ position 2 , caserne $2 \rightarrow$ position 3 , caserne $3 \rightarrow$ position 1
- $\sigma_5=(3,1,2)$ : caserne $1 \rightarrow$ position 3, caserne $2 \rightarrow$ position 1, caserne 3 → position 2
- $\sigma_6=(3,2,1)$ : caserne $1 \rightarrow$ position 3 , caserne $2 \rightarrow$ position 2 , caserne $3 \rightarrow$ position 1

**Conséquence remarquable** : même si Kantorovich autorise le mass splitting (points intérieurs du polytope), **l'optimum tombe toujours sur un sommet du polytope**, c'est-à-dire sur une **permutation pure**. Autrement dit, sous ces hypothèses (masses uniformes égales sur source et cible), **l'optimum de Kantorovich = un transport Monge déterministe**. Le mass splitting n'apporte rien de mieux !

> [!note]- Lien avec le théorème de Birkhoff-von Neumann
> L'ensemble des matrices doublement stochastiques (lignes et colonnes somment à 1) est exactement l'**enveloppe convexe** des matrices de permutation. Pour $n = 3$, ça donne **6 sommets** = $3!$ permutations. C'est le **polytope de Birkhoff** $B_3$, qui vit en réalité en dimension 4 (mais on projette en hexagone pour visualiser). 
> 
> Donc minimiser une fonction linéaire sur $B_3$ → optimum à un sommet → une permutation. C'est précisément pour ça que la **relaxation continue** (Kantorovich) du problème combinatoire de Monge donne **la même solution**, sans perdre d'optimum mais en gagnant la convexité.

**Et Sinkhorn dans tout ça ?** Quand on régularise par l'entropie (note 05), on pénalise les solutions "piquées" (les sommets). L'optimum régularisé glisse alors **vers l'intérieur du polytope**. Plus précisément :
- Avec $\varepsilon \to 0$ : on retombe sur le sommet (LP exact)
- Avec $\varepsilon$ modéré : l'optimum est dans la **bande** entre le sommet et le centre
- Avec $\varepsilon \to \infty$ : l'optimum est au **centre** = produit tensoriel $\mu \otimes \nu$

C'est exactement ce qu'on a visualisé dans la figure `sinkhorn_epsilon_tradeoff.png` de la note 05.

### Formulation continue : couplages

On peut généraliser cette idée de matrice de transport à des distributions continues. La matrice $P$ devient une **loi jointe** $\gamma$ sur le produit $\Omega_s \times \Omega_t$.

L'interprétation est la même que pour la matrice discrète : $\gamma(x, y)$ représente "la quantité de masse qui voyage de $x$ vers $y$". Les contraintes de marginales deviennent :

$$\int_{\Omega_t} \gamma(x, y) \, dy = \mu(x), \qquad \int_{\Omega_s} \gamma(x, y) \, dx = \nu(y)$$

(la première dit "toute la masse au point $x$ est répartie quelque part dans $\gamma(x, \cdot)$" ; la seconde dit "toute la masse qui arrive en $y$ vient de quelque part dans $\gamma(\cdot, y)$")

On note $\Pi(\mu, \nu)$ l'ensemble des $\gamma$ qui ont ces bonnes marginales : c'est l'ensemble des **couplages** entre $\mu$ et $\nu$.

> [!warning] Un couplage $\gamma$ n'est pas une loi jointe statistique
> En stats classique, deux variables aléatoires $X, Y$ ont une **unique** loi jointe déterminée par la nature du phénomène (covariance, copule, etc.). En OT, c'est différent : étant donné $\mu$ et $\nu$, il existe une **infinité de couplages** $\gamma \in \Pi(\mu, \nu)$ qui respectent les marginales. 
> 
> Concrètement : dans l'exemple des casernes étape 2, plusieurs matrices $P$ différentes ont les mêmes marginales $(60, 90, 150)$ et $(120, 90, 90)$. C'est précisément pour ça que Kantorovich est un problème d'**optimisation** : on cherche le **meilleur** parmi tous ces couplages possibles, celui qui minimise $\int c(x,y) \, d\gamma$.
> 
> Un cas particulier intéressant : le **produit tensoriel** $\gamma_{\otimes}(x,y) = \mu(x) \, \nu(y)$ correspond à l'indépendance statistique (ce que tu obtiendrais avec covariance nulle). C'est rarement le couplage optimal — c'est en fait la limite obtenue quand on **ignore** complètement le coût (voir Sinkhorn avec $\varepsilon \to \infty$ dans la note 05).
> 
> En résumé : connaitre $\mu$ et $\nu$ ne détermine **pas** $\gamma$. C'est nous qui choisissons $\gamma$ pour minimiser le coût.

Le **problème de Kantorovich** s'écrit alors :

$$\boxed{\inf_{\gamma \in \Pi(\mu, \nu)} \int_{\Omega_s \times \Omega_t} c(x, y) \, \gamma(dx, dy)}$$

> [!note]- Notation $\gamma(dx, dy)$
> Même logique que pour $\mu(dx)$ vu plus haut, mais ici $\gamma$ est une **mesure jointe en 2D** sur le produit $\Omega_s \times \Omega_t$. La notation $\gamma(dx, dy)$ signifie "la masse contenue dans un petit pavé $dx \times dy$ autour du point $(x, y)$".
> 
> - **Cas continu** (mesure jointe avec densité $\gamma(x, y)$) :
>   $$\gamma(dx, dy) = \gamma(x, y) \, dx \, dy \quad \Rightarrow \quad \int c(x, y) \, \gamma(dx, dy) = \iint c(x, y) \, \gamma(x, y) \, dx \, dy$$
>   C'est l'intégrale double classique sur le produit cartésien.
> 
> - **Cas discret** (Diracs $\gamma = \sum_{i,j} p_{ij} \, \delta_{(x_i, y_j)}$, l'équivalent continu de la matrice $P$) :
>   $$\int c(x, y) \, \gamma(dx, dy) = \sum_{i,j} p_{ij} \cdot c(x_i, y_j)$$
>   On retombe exactement sur la formule $C(P) = \sum_{i,j} p_{ij} \cdot d_{ij}$ de l'étape 2 ! La notation $\gamma(dx, dy)$ **unifie** la version discrète (matrice $P$) et la version continue (densité jointe).

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
