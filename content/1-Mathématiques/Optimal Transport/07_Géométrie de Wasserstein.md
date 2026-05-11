---
title: Géométrie de Wasserstein
date: 2026-05-11
tags: [mathématiques, transport-optimal, géométrie-différentielle, wasserstein, otto, géodésique]
---

## L'idée fondatrice

Les notes précédentes ont traité $W_2$ comme une **distance** entre distributions. Cette note pousse plus loin : on va voir que **l'espace des distributions $\mathcal{P}_2(\mathbb{R}^n)$ muni de $W_2$ est une variété riemannienne** — de dimension infinie certes, mais avec toute la machinerie associée :

- Un **espace tangent** en chaque point
- Une **métrique riemannienne**
- Des **géodésiques** (l'interpolation de McCann, déjà entrevue en note 03)
- Une **exponentielle** (le push-forward par un gradient)
- Un **gradient riemannien** (qui donne le **théorème d'Otto** et le **JKO scheme**)

Cette structure est **exactement** la généralisation de la géométrie différentielle classique qu'on a vue cet après-midi, mais pour des "points" qui sont des distributions de probabilité au lieu de points dans $\mathbb{R}^n$ ou sur une surface.

Pourquoi c'est important :

1. **Théorique** : ça unifie OT et géométrie. La fonction convexe $\varphi$ de Brenier (note 02) joue le rôle de l'exponentielle riemannienne.
2. **Pratique** : c'est la structure géométrique qui sous-tend les **diffusion models** et le **flow matching** modernes. Les meilleurs generative models 2023-2025 sont construits sur la géométrie de Wasserstein.
3. **Pédagogique** : c'est le **bouclage** entre les notes de géométrie différentielle et celles de transport optimal.

Cette note est plus dense que les précédentes : on combine deux ingrédients (OT + géo diff) pour en construire un troisième. Les preuves restent en callouts dépliables.

## I. L'espace tangent en $\mu$

### Intuition : comment "bouger" une distribution

En géométrie différentielle classique, l'espace tangent $T_x M$ en un point $x$ d'une variété est l'ensemble des **vitesses possibles** quand on quitte $x$. C'est un espace vectoriel — la "linéarisation" de la variété en $x$.

Pour $\mathcal{P}_2$, l'analogue est : **comment peut-on faire bouger une distribution $\mu$ ?**

Une réponse naturelle : par un **champ de vecteurs** $v : \mathbb{R}^n \to \mathbb{R}^n$. Si on applique le déplacement infinitésimal $x \mapsto x + \tau v(x)$, on obtient une nouvelle distribution $\mu_\tau$. La dérivée $\partial_\tau \mu_\tau|_{\tau=0}$ représente la "vitesse" associée à $v$.

### Équation de continuité

La relation entre la distribution $\mu_\tau$ et le champ $v$ obéit à l'**équation de continuité** (équation de conservation de la masse en mécanique des fluides) :

$$\partial_\tau \mu_\tau + \nabla \cdot (\mu_\tau \, v) = 0$$

> [!note]- D'où vient cette équation
> Si chaque particule de masse au point $x$ se déplace à la vitesse $v(x)$, la masse contenue dans un volume $V$ change selon le **flux** sortant de la frontière $\partial V$. C'est exactement le théorème de la divergence appliqué au flux $\mu_\tau v$. Cette équation apparaît en mécanique des fluides, en transport de masse, en équations cinétiques.

### Définition formelle

L'**espace tangent** en $\mu$ est l'ensemble des champs de vecteurs qui font bouger $\mu$ de manière compatible avec l'équation de continuité. Mais beaucoup de champs donnent la **même** dérivée $\partial_\tau \mu_\tau$ : si on ajoute à $v$ un champ qui ne "transporte rien" (par exemple un rotationnel dans une distribution radialement symétrique), $\mu_\tau$ ne change pas.

Pour avoir un **espace canonique**, on prend les champs de **gradient** :

$$\boxed{T_\mu \mathcal{P}_2 = \overline{\{ \nabla \varphi : \varphi \in C_c^\infty(\mathbb{R}^n) \}}^{L^2(\mu)}}$$

(la barre indique la fermeture dans $L^2(\mu)$)

**Interprétation** : les "directions tangentes" à $\mathcal{P}_2$ en $\mu$ sont les champs de gradient — ce sont les champs "sans rotation" qui transportent effectivement la masse.

![[wasserstein_tangent.png|372]]
*Une distribution $\mu$ (gaussienne 2D rouge) et un élément $v = \nabla\varphi$ de son espace tangent. Les flèches bleues montrent où chaque parcelle de masse va bouger sous l'effet du champ de gradient $v$. Le produit scalaire $\langle v_1, v_2 \rangle_\mu$ pondère la collision des deux champs par la densité $\mu$.*

> [!note]- Pourquoi les gradients ?
> Par la **décomposition de Helmholtz** (Hodge), tout champ de vecteurs se décompose en une partie gradient + une partie sans divergence. La partie sans divergence ne change pas $\mu_\tau$ (elle "tourne la masse en place"). Donc seule la partie gradient est pertinente pour identifier l'espace tangent. C'est le même principe que la projection orthogonale.

## II. La métrique riemannienne

### Produit scalaire sur l'espace tangent

Pour deux champs tangents $v_1 = \nabla \varphi_1$ et $v_2 = \nabla \varphi_2$, on définit le produit scalaire :

$$\boxed{\langle v_1, v_2 \rangle_\mu = \int_{\mathbb{R}^n} \langle \nabla \varphi_1(x), \nabla \varphi_2(x) \rangle \, d\mu(x)}$$

C'est simplement le produit scalaire $L^2(\mu)$ des champs.

### Norme et longueur de courbes

La norme associée :

$$\|v\|_\mu^2 = \int \|\nabla \varphi\|^2 \, d\mu$$

est la **norme $L^2$ pondérée par $\mu$** du champ.

La longueur d'une courbe $t \mapsto \mu_t$ (avec $\mu_t$ associé au champ $v_t$ via l'équation de continuité) est :

$$\text{long}(\mu) = \int_0^1 \|v_t\|_{\mu_t} \, dt$$

### Connexion à $W_2$ : Benamou-Brenier

Le théorème central qui justifie tout cet édifice :

> [!quote] Théorème (Benamou-Brenier, 2000)
> Pour $\mu, \nu \in \mathcal{P}_2(\mathbb{R}^n)$,
> 
> $$W_2^2(\mu, \nu) = \inf_{(\mu_t, v_t)} \int_0^1 \int \|v_t\|^2 \, d\mu_t \, dt$$
> 
> où l'infimum est pris sur tous les couples $(\mu_t, v_t)$ satisfaisant l'équation de continuité $\partial_t \mu_t + \nabla \cdot (\mu_t v_t) = 0$ avec $\mu_0 = \mu$ et $\mu_1 = \nu$.

**En français** : la distance $W_2$ entre $\mu$ et $\nu$ est la **longueur de la courbe la plus courte** dans $\mathcal{P}_2$ qui les relie, mesurée avec la métrique ci-dessus.

C'est **exactement** la définition d'une distance riemannienne : le minimum de la longueur des chemins. Donc $\mathcal{P}_2$ est bien une variété riemannienne, et $W_2$ est sa distance géodésique.

> [!note]- Vérification heuristique en dimension finie
> Sur $\mathbb{R}^n$ avec la métrique euclidienne standard, la distance entre deux points $x, y$ vaut $\|x - y\|$. La formule de Benamou-Brenier dit l'analogue : $W_2^2(\mu, \nu) = \inf \int_0^1 \int \|v_t\|^2 d\mu_t \, dt$. Le min est atteint sur la trajectoire à vitesse constante (vitesse uniforme), exactement comme une géodésique euclidienne. La formule de Benamou-Brenier généralise donc la formule $\text{dist}^2 = \int_0^1 \|v\|^2 dt$ avec $v = x - y$ constant.

## III. Géodésiques de Wasserstein

### Interpolation de McCann

On a déjà mentionné en note 03 l'**interpolation de McCann** :

$$\mu_t = ((1-t) \, \text{Id} + t \, T)_\# \mu_0$$

où $T = \nabla \varphi$ est la map optimale de Brenier (note 02).

**Théorème** : cette courbe $t \mapsto \mu_t$ est la **géodésique** de Wasserstein entre $\mu_0$ et $\mu_1 = T_\# \mu_0$.

**Interprétation** : on déplace **simultanément** chaque point $x$ de $\mu_0$ vers sa destination $T(x)$ à vitesse constante. Le champ de vitesse correspondant est $v_t(x) = T(x) - x$ (constante en $t$ le long de la trajectoire de chaque particule).

### Pont avec la géométrie différentielle classique

| Géo diff classique | $\mathcal{P}_2$ Wasserstein |
| :--- | :--- |
| Variété $M$, points $x \in M$ | Espace $\mathcal{P}_2$, points = distributions $\mu$ |
| Espace tangent $T_x M$ | $T_\mu \mathcal{P}_2 = \overline{\{\nabla \varphi\}}^{L^2(\mu)}$ |
| Métrique riemannienne $g_x$ | $\langle \nabla \varphi_1, \nabla \varphi_2 \rangle_\mu = \int \langle \nabla \varphi_1, \nabla \varphi_2 \rangle d\mu$ |
| Distance géodésique | $W_2$ |
| Géodésique $\gamma(t)$ | Interpolation de McCann $\mu_t$ |
| Exponentielle $\exp_x(v)$ | Push-forward $(\text{Id} + v)_\# \mu$ pour $v = \nabla \varphi$ |
| Gradient $\nabla_M f$ | Gradient Wasserstein $\text{grad}_W F$ (voir section IV) |

**Conclusion structurelle** : tout ce qu'on a appris en géométrie différentielle se généralise à l'espace des distributions, avec le transport optimal comme dictionnaire.

### Exemple : géodésique entre deux gaussiennes

Pour $\mu_0 = \mathcal{N}(m_0, \Sigma_0)$ et $\mu_1 = \mathcal{N}(m_1, \Sigma_1)$ (avec $\Sigma_0, \Sigma_1$ symétriques définies positives), la géodésique de Wasserstein est aussi composée de gaussiennes :

$$\mu_t = \mathcal{N}(m_t, \Sigma_t)$$

avec $m_t = (1-t) m_0 + t m_1$ (interpolation linéaire des moyennes) et $\Sigma_t$ donné par une formule plus complexe (impliquant la racine carrée matricielle).

**Comparaison avec l'interpolation linéaire**. L'interpolation classique $\tilde{\mu}_t = (1-t) \mu_0 + t \mu_1$ est une **mixture** bimodale dès que $m_0 \neq m_1$ — radicalement différente de l'interpolation Wasserstein qui reste gaussienne tout du long.

![[wasserstein_geodesic.png]]
*Deux interpolations entre $\mu_0 = \mathcal{N}(-3, 0.6^2)$ et $\mu_1 = \mathcal{N}(3, 0.9^2)$. **À gauche** : la géodésique de McCann (Wasserstein). La gaussienne **glisse** continûment de la position $-3$ à la position $+3$ en restant gaussienne tout le long. **À droite** : l'interpolation linéaire $(1-t)\mu_0 + t\mu_1$. À l'instant intermédiaire $t = 0.5$, on obtient une mixture bimodale — un fade-out de $\mu_0$ et un fade-in de $\mu_1$ — qui n'a aucune interprétation comme "transformation continue".*

![[wasserstein_bridge.png]]
*Pont entre géométrie classique et géométrie de Wasserstein. **À gauche** : sur la sphère $S^2$, un point $x$ a un espace tangent $T_x M$, un vecteur $v$ y vit, et la géodésique $\gamma(t) = \exp_x(tv)$ déplace $x$ le long de la surface. **À droite** : sur $\mathcal{P}_2$, une distribution $\mu$ a un espace tangent $T_\mu \mathcal{P}_2$ qui contient le champ $v = \nabla\varphi$ ; la géodésique $\mu_t = ((1-t)\,\text{Id} + tT)_\# \mu$ déforme continûment $\mu$ en $\nu$ via la map de Brenier $T$. Mêmes objets, deux échelles différentes.*

## IV. Théorème d'Otto et JKO scheme

C'est le résultat le plus profond de la géométrie de Wasserstein, dû à **Felix Otto (1998)**.

### Le résultat

Considérons une équation aux dérivées partielles **d'évolution** sur une densité $\rho_t$ — par exemple l'**équation de la chaleur** $\partial_t \rho = \Delta \rho$.

Cette EDP, traditionnellement étudiée comme une EDP parabolique, admet une **interprétation comme un flow de gradient riemannien sur $\mathcal{P}_2$**.

> [!quote] Théorème (Otto, 1998)
> L'équation $\partial_t \rho_t = \nabla \cdot (\rho_t \nabla \delta F / \delta \rho)$ (où $\delta F / \delta \rho$ est la dérivée fonctionnelle) est le **flow de gradient** de la fonctionnelle $F$ pour la métrique de Wasserstein-2.

**Cas particuliers importants** :

- $F(\rho) = \int \rho \log \rho \, dx$ (entropie de Boltzmann) → flow = **équation de la chaleur**
- $F(\rho) = \int V(x) \rho(x) \, dx + \int \rho \log \rho$ → flow = **équation de Fokker-Planck** (équation forward de la diffusion de Langevin)
- $F$ = énergie potentielle → divers flows en mécanique des fluides

**Implication majeure** : des équations EDP qui semblaient sans rapport (chaleur, Fokker-Planck, équations de fluides) sont en fait toutes des **descentes de gradient** dans le même espace géométrique $\mathcal{P}_2$. C'est une unification spectaculaire.

### JKO scheme

[Jordan-Kinderlehrer-Otto, 1998] proposent une **discrétisation en temps** du gradient flow d'Otto, le **JKO scheme** :

$$\mu^{(k+1)} = \arg\min_\nu \left\{ F(\nu) + \frac{1}{2\tau} W_2^2(\nu, \mu^{(k)}) \right\}$$

C'est une **descente de gradient implicite** : à chaque pas $\tau$, on minimise $F$ avec une pénalité Wasserstein qui force à rester près de l'itéré précédent. C'est l'analogue exact de la **descente de gradient proximale** en optimisation classique, mais sur $\mathcal{P}_2$.

**Quand $\tau \to 0$**, JKO converge vers le flow d'Otto continu.

### Pourquoi c'est important pour data science

JKO est aujourd'hui un **outil pratique** :

- **Échantillonnage** : pour échantillonner une distribution $\pi \propto e^{-V}$, on lance un JKO sur $F(\rho) = \int V \rho + \int \rho \log \rho$. C'est l'analogue rigoureux de Langevin MCMC, avec des taux de convergence prouvés.
- **Generative modeling** : les **diffusion models** [Song-Ermon, 2019 ; Ho et al., 2020] sont essentiellement des JKO schémas inversés — on apprend à inverser une trajectoire JKO bruitée.
- **Flow matching** [Lipman et al., 2023] : on entraîne un réseau à apprendre directement le champ de vitesse géodésique de Wasserstein. Plus simple et plus performant que les diffusion models en 2024.

C'est la **frontière actuelle** de la recherche en generative models.

## V. Applications data science

Tour rapide des applications qui exploitent **explicitement** la structure géométrique :

### Flow matching et conditional flow matching

[Lipman et al., 2023] proposent une nouvelle classe de generative models qui apprend directement le **champ de vitesse géodésique** $v_t = T - \text{Id}$ associé à l'interpolation de McCann.

**L'idée** :
- On choisit une distribution simple $\mu_0$ (par ex. gaussienne) et une cible $\mu_1$ (les données)
- On définit le champ $v_t(x)$ qui transporte $\mu_0$ vers $\mu_1$ le long de la géodésique
- On entraîne un réseau $u_\theta(x, t)$ à approximer $v_t$
- À l'inférence, on échantillonne $x_0 \sim \mu_0$ et on intègre $\dot{x} = u_\theta(x, t)$

**Pourquoi ça marche mieux que diffusion** : on cible directement la géodésique (la trajectoire la plus courte), donc le réseau apprend une fonction plus simple, et l'inférence demande moins de pas.

### Diffusion models et Schrödinger bridge

Les diffusion models classiques [Song et al., 2021] approximent une "trajectoire géodésique bruitée" entre une gaussienne et la distribution des données. La connexion rigoureuse passe par le **Schrödinger bridge** — un problème d'OT régularisé par entropie qui généralise les géodésiques.

### Interpolation de datasets

Quand on veut **interpoler entre deux datasets** $D_1, D_2$ (par exemple : domain adaptation progressive), l'interpolation de McCann donne une famille continue de datasets $\{D_t\}_{t \in [0,1]}$ qui **préservent la structure** des deux.

Application : visualization de la "trajectoire" d'une distribution au cours d'un entraînement, comparaison de modèles.

### Wasserstein gradient descent

Pour optimiser une fonctionnelle $F : \mathcal{P}_2 \to \mathbb{R}$ (par ex. la divergence à une distribution cible), on peut faire **du gradient descent dans $\mathcal{P}_2$** via JKO ou des approximations. C'est utilisé en :

- **Variational inference** moderne (alternative à ELBO)
- **Particle methods** pour Bayesian inference (SVGD = Stein Variational Gradient Descent, qui est essentiellement un gradient flow Wasserstein régularisé par RKHS)

## VI. Trois idées à retenir

1. **$(\mathcal{P}_2, W_2)$ est une variété riemannienne** de dimension infinie, avec espace tangent $T_\mu \mathcal{P}_2 = \overline{\{\nabla \varphi\}}^{L^2(\mu)}$ et métrique $\langle \nabla \varphi_1, \nabla \varphi_2 \rangle_\mu = \int \langle \nabla \varphi_1, \nabla \varphi_2 \rangle d\mu$. La distance géodésique est exactement $W_2$ (théorème de Benamou-Brenier).

2. **Les géodésiques sont les interpolations de McCann** $\mu_t = ((1-t) \text{Id} + t T)_\# \mu_0$, où $T = \nabla \varphi$ est la map de Brenier. C'est l'analogue exact de la géodésique $\gamma(t) = \exp_x(tv)$ en géométrie classique.

3. **Beaucoup d'EDP d'évolution sont des gradient flows pour la métrique de Wasserstein** (théorème d'Otto). Le **JKO scheme** discrétise ce gradient flow et est à la base des **diffusion models** et du **flow matching** modernes — les meilleurs generative models actuels.

## VII. Conclusion de la série

Cette série de 7 notes a couvert :

| Note | Sujet | Idée centrale |
| :---: | :--- | :--- |
| 01 | Monge & Kantorovich | Deux formulations du transport, mass splitting |
| 02 | Brenier | Map optimale = gradient d'une fonction convexe |
| 03 | Wasserstein | Distance entre distributions, meilleure que $L^p$ et KL |
| 04 | Dualité | Reformulation duale, lien avec WGAN (Kantorovich-Rubinstein) |
| 05 | Sinkhorn | Régularisation entropique, algorithme rapide |
| 06 | Applications | ML, finance (DRO), vision, sciences |
| 07 | Géométrie | $\mathcal{P}_2$ comme variété riemannienne, pont avec géo diff |

Le **fil rouge** : OT a évolué d'un problème historique (Monge 1781) à un outil mathématique central (Kantorovich 1942, Brenier 1991, Otto 1998), puis à un outil **pratique** en data science (Cuturi 2013) et au cœur du ML moderne (WGAN 2017, flow matching 2023).

### Pour aller plus loin

- **Théorie** : Villani, *Optimal Transport: Old and New* (2008) ; Ambrosio-Gigli-Savaré, *Gradient Flows in Metric Spaces and in the Space of Probability Measures* (2008)
- **Algorithmique** : Peyré-Cuturi, *Computational Optimal Transport* (2019) — la référence
- **Géométrie de Wasserstein** : Santambrogio, *Optimal Transport for Applied Mathematicians* (2015)
- **Applications finance** : Galichon, *Optimal Transport Methods in Economics* (2018) ; revues sur Wasserstein-DRO (Mohajerin Esfahani-Kuhn, Blanchet et al.)
- **Generative models modernes** : papier flow matching [Lipman et al., 2023], papiers diffusion models [Song et al., 2021]
- **Code** : librairie [POT](https://pythonot.github.io/) (Python Optimal Transport), `geomloss` (PyTorch, Sinkhorn différentiable sur GPU)
