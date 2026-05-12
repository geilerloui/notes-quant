---
title: Dualité de Kantorovich
date: 2026-05-11
tags: [mathématiques, transport-optimal, dualité, kantorovich-rubinstein, wgan]
---

## L'idée fondatrice

Jusqu'ici, le problème de Kantorovich a été formulé comme une optimisation sur les **couplages** $\gamma \in \Pi(\mu, \nu)$ :

$$\inf_{\gamma \in \Pi(\mu, \nu)} \int c(x, y) \, d\gamma(x, y) \qquad \text{(primal)}$$

C'est un problème d'optimisation linéaire sous contraintes linéaires. Tout problème de cette forme admet un **dual** — une reformulation équivalente, mais sur des objets différents.

Pour le transport optimal, le dual prend une forme particulièrement utile : au lieu de chercher un couplage $\gamma$, on cherche **deux fonctions** $\varphi, \psi$ qui vivent sur l'espace source et l'espace cible. C'est la **dualité de Kantorovich**.

Pourquoi c'est intéressant ?

1. **Compréhension** : le dual donne une interprétation économique élégante (prix d'achat / prix de vente) qui éclaire le sens du problème
2. **Calcul** : dans beaucoup de cas, le dual est **plus facile à résoudre numériquement** que le primal
3. **Lien avec WGAN** : pour $W_1$, le dual se simplifie en la **formule de Kantorovich-Rubinstein**, qui est **exactement** le critère utilisé par WGAN

Cette note construit la dualité, montre le cas particulier $W_1$, et fait le lien avec WGAN.

## I. Rappel : dualité en optimisation linéaire

Avant d'attaquer la dualité de Kantorovich, un mini-rappel sur la dualité en LP.

Considérons un problème d'optimisation linéaire :

$$\min_x \langle c, x \rangle \quad \text{s.c.} \quad Ax = b, \, x \geq 0 \qquad \text{(primal)}$$

Son **dual** est :

$$\max_y \langle b, y \rangle \quad \text{s.c.} \quad A^\top y \leq c \qquad \text{(dual)}$$

Le **théorème de dualité forte** (en LP) dit : si le primal admet une solution optimale finie, alors :

1. Le dual aussi
2. Les **valeurs optimales coïncident** : $\min_x \langle c, x \rangle = \max_y \langle b, y \rangle$

**Intuition** : le dual encode les "prix marchands" associés aux contraintes du primal. Une contrainte serrée (active) correspond à un prix non nul ; une contrainte non saturée correspond à un prix nul (conditions de **complementary slackness**).

Le problème de Kantorovich est exactement de cette forme (LP) : la matrice de transport $P$ joue le rôle de $x$, la matrice des distances joue le rôle de $c$, et les contraintes de marginales jouent le rôle de $Ax = b$.

## II. La dualité de Kantorovich

### Énoncé

> [!quote] Théorème (Dualité de Kantorovich)
> Pour $\mu, \nu$ deux mesures de probabilité et $c$ un coût continu, on a :
> 
> $$\inf_{\gamma \in \Pi(\mu, \nu)} \int c(x, y) \, d\gamma(x, y) = \sup_{(\varphi, \psi)} \left\{ \int \varphi \, d\mu + \int \psi \, d\nu \right\}$$
> 
> où le sup est pris sur tous les couples $(\varphi, \psi)$ de fonctions continues bornées vérifiant la **contrainte de compatibilité** :
> 
> $$\varphi(x) + \psi(y) \leq c(x, y) \qquad \forall (x, y)$$

Deux fonctions $\varphi : \Omega_s \to \mathbb{R}$ et $\psi : \Omega_t \to \mathbb{R}$, vivant chacune sur un seul espace, dont la **somme** est dominée par le coût. On les appelle **potentiels de Kantorovich**.

### Visualisation : LP primal et LP dual côte à côte

La façon la plus claire de saisir la dualité, c'est de **visualiser concrètement** comment se construisent les matrices et vecteurs du LP primal, et de voir ce que devient cette structure dans le LP dual.

#### Setup du primal

On part de deux distributions discrètes $P_r$ et $P_\theta$ (notation équivalente à $\mu$ et $\nu$ — c'est le vocabulaire qu'utilise WGAN) reliées par un plan de transport $\gamma$ :

![[images/1-Mathématiques/Optimal transport/im1.png|273]]
*Plan de transport $\gamma$ entre deux distributions discrètes. Chaque élément $\gamma(x_i, y_j)$ donne la masse transportée de $x_i$ vers $y_j$.*

Pour mettre ça sous forme LP standard, on **vectorise** la matrice $\gamma \in \mathbb{R}^{n \times n}$ : on empile ses colonnes les unes sur les autres pour obtenir un vecteur de taille $n^2$. Même chose pour la matrice de coût. On obtient :

- $x \in \mathbb{R}^{n^2}$ : le plan de transport vectorisé (les variables à trouver)
- $c \in \mathbb{R}^{n^2}$ : les coûts vectorisés (donnés)
- $A \in \mathbb{R}^{2n \times n^2}$ : la matrice des contraintes de marginales
- $b \in \mathbb{R}^{2n}$ : les marginales empilées

Le LP primal s'écrit alors :

$$\min_x \, c^\top x \quad \text{s.c.} \quad Ax = b, \quad x \geq 0$$

**En mots** : on minimise le coût total $c^\top x = \sum_{ij} c_{ij} \gamma_{ij}$, sous les contraintes que le plan de transport ait les bonnes marginales $P_r$ et $P_\theta$.

![[images/1-Mathématiques/Optimal transport/im2 (1).png]]
*Schéma matriciel du LP primal. À gauche : le vecteur $x$ des variables (chaque entrée est un $\gamma(x_i, y_j)$). On multiplie la matrice $A^\top$ par $x$ (le schéma utilise $A^\top$ par souci de lisibilité) pour obtenir le vecteur de contraintes $b$. **Lecture clé** : la première colonne de $A^\top$ multipliée par $x$ donne $\sum_i \gamma(x_1, y_i)$, c'est-à-dire la masse totale qui part du point $x_1$ — ce qui doit égaler $P_r(x_1)$. Les colonnes de $A^\top$ encodent donc directement les contraintes de marginales.*

#### Limites du primal

Trois problèmes apparaissent en pratique avec cette formulation primale :

1. **Explosion de la dimension** : si les supports de $r$ et $\theta$ ont taille $n$, le LP a $n^2$ variables. Pour $n = 1000$ (taille modeste), on a déjà $10^6$ variables.
2. **On veut souvent juste $W_p$, pas $\gamma^*$** : comme discuté en note 03, dans beaucoup d'applis ML on s'intéresse au **scalaire** $W_p$, pas au plan de transport. Le primal calcule les deux mais c'est "inutilement riche".
3. **Le gradient par rapport à $\theta$** : dans WGAN, on veut dériver $W_1$ par rapport aux paramètres $\theta$ d'un modèle génératif. Or dans le primal, $\theta$ apparaît dans les **contraintes** (via $b$ et le quadrant gauche de $A^\top$), ce qui rend la différentiation peu naturelle.

La **solution** à ces trois problèmes : passer au LP dual.

#### Setup du dual

Dans le dual, on n'a plus besoin de regarder l'espace produit $\Omega_s \times \Omega_t$ (de dimension $n^2$). On travaille uniquement sur l'espace de base (dimension $2n$, une variable par contrainte du primal) :

![[images/1-Mathématiques/Optimal transport/im3 (2).png]]
*Schéma matriciel du LP dual. Le vecteur $y$ de variables duales contient deux fonctions : $f$ (sur le support de $P_r$) et $g$ (sur le support de $P_\theta$). La contrainte $A^\top y \leq c$ donne, ligne par ligne, $f(x_i) + g(x_j) \leq d(x_i, x_j)$ pour toute paire $(i, j)$.*

**Lecture de la contrainte.** Si on regarde la première colonne de $A$, on obtient :

$$f(x_1) + g(x_1) \leq D_{1, 1}$$

Et plus généralement, pour toute paire $(i, j)$ :

$$f(x_i) + g(x_j) \leq D_{i, j} = d(x_i, x_j)$$

C'est exactement la **contrainte de compatibilité** $\varphi(x) + \psi(y) \leq c(x, y)$ vue plus haut, avec $f = \varphi$ et $g = \psi$.

**Lecture de l'objectif.** Le vecteur $b$ contient les probabilités $P_r$ et $P_\theta$, donc $b^\top y$ donne :

$$b^\top y = \sum_i f(x_i) P_r(x_i) + \sum_j g(x_j) P_\theta(x_j) = \int f \, dr + \int g \, d\theta$$

On retrouve l'objectif dual $\int \varphi \, d\mu + \int \psi \, d\nu$.

#### Ce qu'on a gagné

| | Primal | Dual |
| :--- | :--- | :--- |
| Nombre de variables | $n^2$ (plan de transport vectorisé) | $2n$ (deux fonctions $f, g$) |
| Espace de travail | Produit $\Omega_s \times \Omega_t$ | Espaces $\Omega_s$ et $\Omega_t$ séparément |
| Que cherche-t-on ? | Une matrice $\gamma$ | Deux fonctions $f, g$ |
| Compatible avec backprop ? | Difficile ($\theta$ dans les contraintes) | Oui ($\theta$ dans l'objectif via $b$) |

Passer du primal au dual, c'est passer de **"trouver une matrice $n \times n$"** à **"trouver deux vecteurs de taille $n$"**. C'est ce gain qui rend l'approche dualisée cruciale en grande dimension.

#### Re-lecture de l'interprétation économique

Maintenant qu'on a vu la structure LP, on peut relire l'interprétation économique avec un exemple concret. Imagine des **boulangeries** qui produisent des croissants (distribution $r$, marginale source) et des **hôtels** qui les consomment (distribution $\theta$, marginale cible). Le transport d'une boulangerie vers un hôtel a un coût fixe $d(x, y)$.

- $f(x)$ = prix de vente du croissant à la boulangerie $x$ → la boulangerie veut **maximiser** $f$
- $g(y)$ = prix d'achat du croissant à l'hôtel $y$ → l'hôtel veut **minimiser** son coût total
- Contrainte : $f(x) + g(y) \leq d(x, y)$ — "l'écart de prix entre achat et vente ne peut pas dépasser le coût de transport", sinon les acteurs n'auraient aucun intérêt à commercer

**L'apport conceptuel clé** : avant on avait une optimisation sur une **loi jointe** (objet 2D complexe), maintenant on a une optimisation sur **deux fonctions sur l'espace de base**. C'est là toute la valeur du passage au dual.

### Intuition économique

L'interprétation classique est en termes de **transporteur** versus **commerçant** :

- **Vous** êtes le transporteur. Pour chaque trajet de $x$ vers $y$, vous payez un coût $c(x, y)$.
- **Un commerçant** vous propose une alternative : il vous achète la marchandise au point $x$ pour un prix $\varphi(x)$, et la revend au point $y$ pour un prix $\psi(y)$. Vous ne payez que la différence (les frais).

Le commerçant doit garantir que sa proposition est **acceptable pour vous** : son tarif net $\varphi(x) + \psi(y)$ ne doit pas dépasser le coût de transport direct $c(x, y)$. C'est exactement la contrainte $\varphi(x) + \psi(y) \leq c(x, y)$.

Le **profit total du commerçant** est :

$$\underbrace{\int \varphi \, d\mu}_{\text{ce qu'il paie aux sources}} + \underbrace{\int \psi \, d\nu}_{\text{ce qu'il reçoit aux cibles}}$$

Le commerçant cherche à **maximiser** son profit (côté dual). Le théorème de dualité forte dit que ce profit maximal est **exactement égal** au coût minimal de transport.

![[kantorovich_duality_economic.png|450]]
*Interprétation économique de la dualité. Pour transporter une unité de masse de $x$ vers $y$, le transporteur a deux options : **A** payer directement $c(x, y)$, ou **B** passer par un commerçant qui achète en $x$ à $\varphi(x)$ et revend en $y$ à $\psi(y)$ (coût net $\varphi(x) + \psi(y)$). Pour que l'option B soit envisageable, on doit avoir $\varphi(x) + \psi(y) \leq c(x, y)$. Le commerçant maximise son profit sous cette contrainte, et la **dualité forte** assure que ce profit maximal = coût minimal de transport.*

### Preuve heuristique

> [!note]- Esquisse de preuve
> On part du primal et on introduit la contrainte de marginale via des multiplicateurs de Lagrange $\varphi(x), \psi(y)$ :
> 
> $$L(\gamma, \varphi, \psi) = \int c(x, y) \, d\gamma - \int \varphi(x) [d\gamma_X(x) - d\mu(x)] - \int \psi(y) [d\gamma_Y(y) - d\nu(y)]$$
> 
> (où $\gamma_X, \gamma_Y$ sont les marginales de $\gamma$). En réarrangeant :
> 
> $$L = \int [c(x, y) - \varphi(x) - \psi(y)] \, d\gamma + \int \varphi \, d\mu + \int \psi \, d\nu$$
> 
> Pour le primal, on minimise sur $\gamma \geq 0$. Si $c(x, y) - \varphi(x) - \psi(y) < 0$ quelque part, on peut mettre une masse infinie de $\gamma$ là et $L \to -\infty$. Pour éviter ça côté dual, on impose :
> 
> $$\varphi(x) + \psi(y) \leq c(x, y) \quad \forall (x, y)$$
> 
> Sous cette contrainte, le terme $[c - \varphi - \psi] \geq 0$ est minimisé en $\gamma = 0$ là où l'inégalité est stricte, et atteint $0$ sur le support optimal. Le reste donne :
> 
> $$\sup_{(\varphi, \psi)} \int \varphi \, d\mu + \int \psi \, d\nu$$

### Conditions d'optimalité

À l'optimum, la contrainte $\varphi(x) + \psi(y) \leq c(x, y)$ est **saturée sur le support du couplage optimal** $\gamma^*$ :

$$\varphi(x) + \psi(y) = c(x, y) \quad \forall (x, y) \in \text{supp}(\gamma^*)$$

C'est la condition de **complementary slackness** : là où on transporte effectivement de la masse ($\gamma^* > 0$), l'option "passer par le commerçant" n'apporte ni gain ni perte par rapport au transport direct.

## III. Le cas $W_1$ : formule de Kantorovich-Rubinstein

Pour le cas particulier $c(x, y) = \|x - y\|$ (donc $W_1$, l'Earth Mover's Distance), la dualité se simplifie de manière spectaculaire.

### Réduction à une seule fonction

Pour $c(x, y) = \|x - y\|$, on peut montrer qu'on peut **se restreindre** à des couples de la forme $(\varphi, -\varphi)$, c'est-à-dire $\psi = -\varphi$. La contrainte de compatibilité devient :

$$\varphi(x) - \varphi(y) \leq \|x - y\| \quad \forall (x, y)$$

C'est exactement la condition que $\varphi$ soit **1-Lipschitz** :

$$|\varphi(x) - \varphi(y)| \leq \|x - y\|$$

> [!note]- Pourquoi on peut prendre $\psi = -\varphi$
> Pour le coût $c(x, y) = \|x - y\|$, qui est une **distance**, la dualité de Kantorovich a une structure particulière. Étant donné un couple optimal $(\varphi, \psi)$, on peut définir le **$c$-transformé** :
> 
> $$\varphi^c(y) = \inf_x \{ c(x, y) - \varphi(x) \}$$
> 
> Pour $c(x, y) = \|x - y\|$ et $\varphi$ 1-Lipschitz, on montre que $\varphi^c = -\varphi$. Cela fait que le problème dual se réduit à optimiser sur une seule fonction 1-Lipschitz.

### La formule de Kantorovich-Rubinstein

On obtient :

$$\boxed{W_1(\mu, \nu) = \sup_{\|\varphi\|_{\text{Lip}} \leq 1} \left\{ \int \varphi \, d\mu - \int \varphi \, d\nu \right\}}$$

**En français** : $W_1$ s'écrit comme **le sup de la différence des espérances** d'une fonction 1-Lipschitz $\varphi$, prises sous $\mu$ et $\nu$.

C'est une formule **incroyablement utile** :
- Elle ne fait intervenir **qu'une seule fonction** $\varphi$ (au lieu de deux dans le dual général, ou d'un couplage 2D dans le primal)
- Elle s'exprime comme une **espérance** sous chaque distribution, ce qui se parallélise très bien
- Les estimateurs **convergent** quand on remplace les vraies distributions par des échantillons (loi des grands nombres)

![[kantorovich_rubinstein_1lip.png]]
*Illustration de Kantorovich-Rubinstein. La fonction 1-Lipschitz optimale $\varphi$ (verte) est **positive** sur le support de $\mu$ (rouge) et **négative** sur le support de $\nu$ (bleu), avec **pente $|\varphi'| = 1$** entre les deux (contrainte saturée). Pour deux gaussiennes centrées en $\pm 3$, on trouve $\mathbb{E}_\mu[\varphi] \approx +3$, $\mathbb{E}_\nu[\varphi] \approx -3$, donc $W_1 \approx 6$ — exactement la distance entre les centres, comme attendu.*

### Calcul effectif

En pratique, calculer $W_1$ par K-R revient à :

$$W_1(\mu, \nu) \approx \max_{\varphi \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n \varphi(x_i) - \frac{1}{m} \sum_{j=1}^m \varphi(y_j)$$

où $\mathcal{F}$ est une famille de fonctions 1-Lipschitz, et $\{x_i\}, \{y_j\}$ sont des échantillons de $\mu$ et $\nu$.

Le défi : **paramétrer une famille suffisamment riche** $\mathcal{F}$ tout en garantissant la contrainte de Lipschitz. C'est exactement le rôle des réseaux de neurones dans WGAN.

## IV. Lien avec WGAN

(Cette section est un teaser — le développement complet de WGAN n'est pas l'objet de cette série de notes.)

**Le problème des GAN classiques.** Un GAN classique (Goodfellow 2014) entraîne un **générateur** $G_\theta$ à produire des données ressemblant à une distribution réelle. Le critère originel est la **divergence de Jensen-Shannon** entre la distribution réelle et la distribution générée. Mais JSD a un défaut majeur : si les deux distributions ont des **supports disjoints** (ce qui est presque toujours le cas en haute dimension), JSD = $\log 2$ est **constante**, et son gradient est nul. Le générateur n'apprend pas.

**L'idée des WGAN.** [Arjovsky et al., 2017] proposent de remplacer JSD par $W_1$. Grâce à K-R, on peut écrire :

$$W_1(\mathbb{P}_{\text{réelle}}, \mathbb{P}_{G_\theta}) = \sup_{\|\varphi\|_\text{Lip} \leq 1} \mathbb{E}_{\mathbb{P}_\text{réelle}}[\varphi(x)] - \mathbb{E}_{\mathbb{P}_{G_\theta}}[\varphi(G_\theta(z))]$$

Le **critique** (terme préféré au "discriminateur" des GAN classiques) est un réseau de neurones $\varphi_w$ qui calcule le sup. La contrainte 1-Lipschitz est imposée approximativement (weight clipping dans la version originale, gradient penalty dans WGAN-GP).

**Pourquoi ça marche** : $W_1$ étant continue et non-saturante même pour des supports disjoints (cf. note 03), le gradient est toujours informatif, et le générateur converge mieux.

C'est ce qui fait de la dualité de Kantorovich un objet **central en ML moderne**, et pas seulement une curiosité mathématique.

## V. Lien avec Brenier

Pour le coût quadratique $c(x, y) = \tfrac{1}{2}\|x - y\|^2$ (le facteur $\tfrac{1}{2}$ simplifie les formules), on peut faire un changement de variable astucieux qui relie la dualité de Kantorovich au théorème de Brenier.

### Changement de variable

Posons $\Phi(x) = \tfrac{1}{2}\|x\|^2 - \varphi(x)$ et $\Psi(y) = \tfrac{1}{2}\|y\|^2 - \psi(y)$. La contrainte de compatibilité $\varphi(x) + \psi(y) \leq \tfrac{1}{2}\|x-y\|^2$ se réécrit (après développement) :

$$\Phi(x) + \Psi(y) \geq \langle x, y \rangle$$

C'est la **caractérisation de la transformée de Legendre** : $\Psi$ est la transformée de Legendre de $\Phi$, c'est-à-dire $\Psi = \Phi^*$ avec :

$$\Phi^*(y) = \sup_x \{ \langle x, y \rangle - \Phi(x) \}$$

> [!note]- Rappel : transformée de Legendre
> Pour une fonction convexe $\Phi$, sa transformée de Legendre est :
> 
> $$\Phi^*(y) = \sup_x \{ \langle x, y \rangle - \Phi(x) \}$$
> 
> C'est une autre fonction convexe (toujours convexe par construction). Les propriétés clés :
> - $\Phi^{**} = \Phi$ (si $\Phi$ est convexe semi-continue)
> - $y \in \partial \Phi(x) \Leftrightarrow x \in \partial \Phi^*(y) \Leftrightarrow \Phi(x) + \Phi^*(y) = \langle x, y \rangle$
> 
> En économie, c'est le lien entre fonction d'utilité et fonction de prix duale.

### Le potentiel de Brenier réapparaît

À l'optimum, la fonction $\Phi$ qui maximise le dual est exactement la **fonction convexe $\varphi$** du théorème de Brenier. Plus précisément :

$$T(x) = \nabla \Phi(x)$$

est la map optimale de Brenier.

**Le pont entre théorèmes** :

| Note | Objet | Caractérisation |
| :--- | :--- | :--- |
| Note 02 (Brenier) | Map $T$ | $T = \nabla \varphi$ avec $\varphi$ convexe |
| Note 04 (Dual) | Potentiels $\varphi, \psi$ | $\varphi(x) + \psi(y) \leq c(x, y)$, saturé sur supp$(\gamma^*)$ |

Pour le coût quadratique, ces deux caractérisations **coïncident** : le potentiel dual = le potentiel de Brenier, et la map optimale est son gradient.

## VI. Trois idées à retenir

1. **Le problème de Kantorovich admet un dual** qui s'écrit comme une maximisation sur deux fonctions $\varphi, \psi$ vérifiant $\varphi(x) + \psi(y) \leq c(x, y)$. La dualité forte assure que primal et dual ont la **même valeur optimale**. Interprétation économique : le coût minimal du transport = le profit maximal d'un commerçant intermédiaire.

2. **Pour $W_1$, le dual se simplifie en Kantorovich-Rubinstein** :
   $$W_1(\mu, \nu) = \sup_{\|\varphi\|_\text{Lip} \leq 1} \int \varphi \, d\mu - \int \varphi \, d\nu$$
   C'est **cette formule** qui rend $W_1$ calculable en pratique par approximation neuronale (WGAN).

3. **Pour le coût quadratique, le potentiel dual = le potentiel de Brenier**. Les deux théorèmes (Brenier et dualité) sont deux facettes du même phénomène, via la transformée de Legendre.

## VII. Vers la suite

- **Note 05 — Sinkhorn** : la dualité prend une forme particulièrement élégante avec régularisation entropique, donnant un algorithme itératif extrêmement rapide.
- **Note 06 — Applications ML** : Wasserstein barycenters, domain adaptation, où la formulation duale est l'outil de base.
- **Note 07 — Géométrie de Wasserstein** : la fonction convexe $\varphi$ du dual a une interprétation géométrique comme **potentiel** d'un flot géodésique.
