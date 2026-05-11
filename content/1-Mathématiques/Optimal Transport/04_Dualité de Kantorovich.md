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

### Intuition économique

L'interprétation classique est en termes de **transporteur** versus **commerçant** :

- **Vous** êtes le transporteur. Pour chaque trajet de $x$ vers $y$, vous payez un coût $c(x, y)$.
- **Un commerçant** vous propose une alternative : il vous achète la marchandise au point $x$ pour un prix $\varphi(x)$, et la revend au point $y$ pour un prix $\psi(y)$. Vous ne payez que la différence (les frais).

Le commerçant doit garantir que sa proposition est **acceptable pour vous** : son tarif net $\varphi(x) + \psi(y)$ ne doit pas dépasser le coût de transport direct $c(x, y)$. C'est exactement la contrainte $\varphi(x) + \psi(y) \leq c(x, y)$.

Le **profit total du commerçant** est :

$$\underbrace{\int \varphi \, d\mu}_{\text{ce qu'il paie aux sources}} + \underbrace{\int \psi \, d\nu}_{\text{ce qu'il reçoit aux cibles}}$$

Le commerçant cherche à **maximiser** son profit (côté dual). Le théorème de dualité forte dit que ce profit maximal est **exactement égal** au coût minimal de transport.

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
