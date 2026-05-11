---
title: Distance de Wasserstein
date: 2026-05-11
tags: [mathématiques, transport-optimal, wasserstein, métrique]
---

## L'idée fondatrice

Jusqu'ici, on a vu deux formulations du transport optimal (Monge et Kantorovich) et un théorème central (Brenier). Mais à chaque fois, on s'intéressait à **comment** transporter — la map $T$, le couplage $\gamma$.

La note d'aujourd'hui change de perspective : on s'intéresse à la **valeur optimale du coût**, vue comme une **distance entre distributions**. C'est la **distance de Wasserstein**.

Pourquoi c'est important ? Parce qu'en data science, on compare constamment des distributions :
- **Comparer une vraie distribution à une approximation** : sortie d'un GAN vs distribution réelle, prédiction d'un modèle vs ground truth
- **Mesurer la similarité entre deux datasets** : domain adaptation, transfer learning
- **Calibrer un modèle** : ajuster les paramètres pour que la distribution prédite "ressemble" à la cible

Toutes ces tâches demandent une notion de **distance entre distributions**. Wasserstein est, dans beaucoup de contextes, la **meilleure** : elle a des propriétés que les distances classiques ($L^p$, KL divergence) n'ont pas.

Cette note construit la distance de Wasserstein, montre que c'est bien une **métrique** au sens mathématique, et compare ses propriétés à $L^2$ pour comprendre pourquoi elle est si utile en pratique.

## I. Définition

### La distance de Wasserstein-$p$

À partir du problème de Kantorovich avec coût $c(x, y) = \|x - y\|^p$ (pour $p \geq 1$), on définit :

$$\boxed{W_p(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu, \nu)} \int_{\Omega_s \times \Omega_t} \|x - y\|^p \, \gamma(dx, dy) \right)^{1/p}}$$

En mots : **la racine $p$-ième du coût optimal de transport** pour le coût $\|x-y\|^p$.

> [!note]- Pourquoi élever à la puissance $1/p$ ?
> Sans la racine, on aurait juste "$W_p^p$" (la valeur du coût optimal). En prenant la racine $1/p$, on obtient une grandeur qui se comporte comme une **distance** : elle est en mètres si $x, y$ sont en mètres. C'est l'analogue de la norme $L^p$ d'un vecteur : $\|x\|_p = (\sum |x_i|^p)^{1/p}$.

### Deux cas particuliers très utilisés

**$W_1$ — Earth Mover's Distance.** Avec $p = 1$ :

$$W_1(\mu, \nu) = \inf_\gamma \int \|x - y\| \, d\gamma(x, y)$$

C'est ce qu'on appelle aussi **Earth Mover's Distance (EMD)** [Rubner et al., 2000]. Très utilisée en vision (comparaison d'histogrammes de couleurs), en NLP (Word Mover's Distance pour comparer des documents), et **directement liée aux WGANs** via la dualité (note 04).

**$W_2$ — coût quadratique.** Avec $p = 2$ :

$$W_2(\mu, \nu) = \left( \inf_\gamma \int \|x - y\|^2 \, d\gamma(x, y) \right)^{1/2}$$

C'est le cas où le **théorème de Brenier** s'applique : l'optimum est atteint par $T = \nabla \varphi$ avec $\varphi$ convexe. C'est aussi le cas où l'espace des distributions devient une **variété riemannienne** (voir note 07).

### Propriétés générales

Wasserstein fonctionne dans des contextes que les distances classiques peinent à gérer :

| Propriété | Détail |
| :--- | :--- |
| Pas besoin de support commun | $W_p(\mu, \nu)$ est bien défini même si $\mu$ et $\nu$ sont supportées sur des ensembles disjoints |
| Marche pour des distributions arbitraires | Continues, discrètes (histogrammes, mesures empiriques), mixtes |
| Dépend de la **géométrie** sous-jacente | Contrairement à KL, $W_p$ tient compte des distances entre points du support |

## II. Wasserstein est une métrique

On veut maintenant montrer que $W_p$ satisfait les **axiomes d'une distance** :

1. **Symétrie** : $W_p(\mu, \nu) = W_p(\nu, \mu)$
2. **Positivité** : $W_p(\mu, \nu) \geq 0$ avec égalité si et seulement si $\mu = \nu$
3. **Inégalité triangulaire** : $W_p(\mu, \nu) \leq W_p(\mu, \sigma) + W_p(\sigma, \nu)$

Les deux premiers axiomes sont faciles. Le troisième est plus subtil et demande un outil technique : le **lemme de recollement** (gluing lemma).

### Symétrie

Immédiate : si $\gamma \in \Pi(\mu, \nu)$ alors $\gamma^\top \in \Pi(\nu, \mu)$ (on échange les rôles source/cible). Le coût est inchangé puisque $\|x - y\| = \|y - x\|$. Donc $W_p(\mu, \nu) = W_p(\nu, \mu)$.

### Positivité

Positivité automatique : $\|x-y\|^p \geq 0$ et $\gamma \geq 0$, donc l'intégrale est positive.

> [!note]- Preuve : $W_p(\mu, \nu) = 0 \Leftrightarrow \mu = \nu$
> Si $\mu = \nu$, prendre $\gamma$ concentré sur la diagonale (le couplage "identité") donne $W_p = 0$.
> 
> Réciproquement, supposons $W_p(\mu, \nu) = 0$. Alors il existe $\gamma \in \Pi(\mu, \nu)$ tel que
> 
> $$\int \|x - y\|^p \, d\gamma(x, y) = 0$$
> 
> Cela signifie que $\gamma$ est supporté sur la diagonale $\{(x, y) : x = y\}$. Pour n'importe quel $A \subset \Omega$ :
> 
> $$\mu(A) = \int_{A \times \Omega} d\gamma(x, y) = \int_{A \times A} d\gamma(x, y) = \int_{\Omega \times A} d\gamma(x, y) = \nu(A)$$
> 
> (la deuxième égalité utilise le fait que $\gamma$ est sur la diagonale : si $x \in A$ alors $y = x \in A$ aussi)
> 
> Donc $\mu = \nu$.

### Inégalité triangulaire

C'est la partie difficile. L'idée est intuitivement claire : pour transporter $\mu$ vers $\nu$, on peut faire un détour par une distribution intermédiaire $\sigma$. Le coût total du détour est au moins celui du chemin direct.

**Cas simple (Monge).** Si on a des maps optimales $T_1 : \mu \to \sigma$ et $T_2 : \sigma \to \nu$, on peut composer : $T = T_2 \circ T_1$ envoie bien $\mu$ vers $\nu$ (mais n'est pas forcément optimal). Le coût de $T$ est borné par $T_1$ + $T_2$ via l'inégalité triangulaire de la norme $L^p$.

**Cas général (Kantorovich).** On ne travaille plus avec des maps mais avec des couplages $\gamma$. On ne peut pas "composer" deux couplages directement — il faut un outil supplémentaire.

#### Le lemme de recollement (gluing lemma)

L'outil clé est le **lemme de recollement** : étant donnés deux couplages avec une marginale commune, on peut les "coller" en un seul couplage sur un espace produit triple.

> [!quote] Lemme (Gluing)
> Soient $\pi_{XY} \in \Pi(\mu_X, \mu_Y)$ et $\pi_{YZ} \in \Pi(\mu_Y, \mu_Z)$ (ils partagent la marginale $\mu_Y$). Alors il existe une mesure $\pi \in \mathcal{P}(X \times Y \times Z)$ telle que :
> - $\pi_{XY}$ est la marginale de $\pi$ sur $X \times Y$
> - $\pi_{YZ}$ est la marginale de $\pi$ sur $Y \times Z$

L'idée : $\pi$ "recolle" les deux couplages en utilisant la marginale commune $\mu_Y$ comme charnière.

> [!note]- Preuve de l'inégalité triangulaire (esquisse)
> Soient $\pi_{XY} \in \Pi(\mu, \sigma)$ et $\pi_{YZ} \in \Pi(\sigma, \nu)$ des couplages optimaux pour $W_p(\mu, \sigma)$ et $W_p(\sigma, \nu)$ respectivement. Par le lemme de recollement, il existe $\pi \in \mathcal{P}(X \times Y \times Z)$ ayant $\pi_{XY}$ et $\pi_{YZ}$ comme marginales.
> 
> Définissons $\pi_{XZ}$ comme la marginale de $\pi$ sur $X \times Z$. On vérifie que $\pi_{XZ} \in \Pi(\mu, \nu)$ (les marginales sur $X$ et $Z$ sont bien $\mu$ et $\nu$, par projection). Donc $\pi_{XZ}$ est un couplage admissible (pas forcément optimal) :
> 
> $$W_p(\mu, \nu) \leq \left( \int_{X \times Z} \|x - z\|^p \, d\pi_{XZ} \right)^{1/p} = \|x - z\|_{L^p(\pi)}$$
> 
> Par l'inégalité triangulaire dans $L^p$ :
> 
> $$\|x - z\|_{L^p(\pi)} \leq \|x - y\|_{L^p(\pi)} + \|y - z\|_{L^p(\pi)}$$
> 
> Et chaque terme se calcule via la marginale correspondante :
> 
> $$\|x - y\|_{L^p(\pi)} = \left( \int_{X \times Y} \|x - y\|^p \, d\pi_{XY} \right)^{1/p} = W_p(\mu, \sigma)$$
> 
> (de même pour le second terme). On obtient bien :
> 
> $$W_p(\mu, \nu) \leq W_p(\mu, \sigma) + W_p(\sigma, \nu)$$

**Conclusion** : $W_p$ est bien une **métrique** sur l'espace des distributions de probabilité (avec moment d'ordre $p$ fini).

## III. Pourquoi Wasserstein est meilleure que $L^p$ en pratique

Voici **3 propriétés cruciales** qui font de Wasserstein un meilleur choix que les distances classiques en data science.

### Propriété 1 : Profils proches → comportement similaire à $L^2$

Soient $f$ et $g$ deux densités où $g$ est $f$ translatée de $s$ : $g(y) = f(y - s)$.

Pour $s$ petit, on a :

$$\|f - g\|_{L^2} = O(s) \quad \text{et} \quad W_2(f, g) = O(s)$$

Les deux distances **convergent au même rythme**. Pas de différence notable pour les profils proches.

### Propriété 2 : Profils éloignés → Wasserstein voit, $L^p$ ne voit plus rien

Maintenant prenons $f$ et $g$ deux densités à **supports disjoints** (par exemple deux gaussiennes très éloignées).

$$\|f - g\|_{L^2} = \sqrt{2 \int f^2} = O(1)$$

(c'est juste $\sqrt{\int f^2 + \int g^2}$, indépendant de la distance entre les supports)

En revanche :

$$W_2(f, g) = O(s)$$

(toujours proportionnel à la distance entre les supports)

**Conséquence majeure pour le ML** : si on entraîne un modèle par descente de gradient sur $\|f_\theta - g\|_{L^2}$, et que $f_\theta$ et $g$ sont initialement très loin, le gradient est **nul ou non-informatif**. Avec Wasserstein, le gradient pointe toujours dans la bonne direction, peu importe la distance.

C'est précisément la motivation des **WGAN** : remplacer le critère de Jensen-Shannon (qui sature quand les distributions sont disjointes) par $W_1$ (qui ne sature jamais).

### Propriété 3 : Insensibilité au bruit haute fréquence

Soit $g(x) = f(x) + \sin(2\pi k x)$ : on rajoute à $f$ une oscillation haute fréquence (grand $k$).

$$\|f - g\|_{L^2} = O(1) \quad \text{(le bruit ne s'efface pas)}$$

$$W_2(f, g) = O(1/k) \quad \text{(le bruit devient invisible quand $k \to \infty$)}$$

Plus le bruit est haute fréquence, plus Wasserstein l'ignore. C'est crucial pour des applications réelles où les données sont bruitées.

### Propriété 4 : Convexité

$W_p^p$ est **convexe par rapport à ses arguments** :

$$W_p^p(\alpha \mu_1 + (1-\alpha)\mu_2, \, \alpha \nu_1 + (1-\alpha)\nu_2) \leq \alpha W_p^p(\mu_1, \nu_1) + (1-\alpha) W_p^p(\mu_2, \nu_2)$$

> [!note]- Preuve de la convexité
> Soient $\pi_1, \pi_2$ optimaux pour $W_p(\mu_1, \nu_1)$ et $W_p(\mu_2, \nu_2)$ respectivement.
> 
> Posons $\pi = \alpha \pi_1 + (1-\alpha) \pi_2$. Ce $\pi$ est un couplage entre $\alpha \mu_1 + (1-\alpha) \mu_2$ et $\alpha \nu_1 + (1-\alpha) \nu_2$ (vérification immédiate des marginales).
> 
> Donc :
> 
> $$W_p^p(\mu, \nu) \leq \int \|x-y\|^p \, d\pi = \alpha \int \|x-y\|^p d\pi_1 + (1-\alpha) \int \|x-y\|^p d\pi_2$$
> 
> $$= \alpha W_p^p(\mu_1, \nu_1) + (1-\alpha) W_p^p(\mu_2, \nu_2)$$

**Conséquence** : si on cherche à minimiser une fonctionnelle de la forme $\mu \mapsto W_p^p(\mu, \nu_0)$ sous des contraintes convexes, le problème reste convexe.

### Récap : Wasserstein vs $L^p$ vs KL

| Critère | $L^p$ | KL divergence | Wasserstein |
| :--- | :---: | :---: | :---: |
| Vraie distance (symétrique, triangulaire) | ✓ | ✗ | ✓ |
| Bien définie pour supports disjoints | ✓ | ✗ ($\infty$) | ✓ |
| Tient compte de la géométrie | ✗ | ✗ | ✓ |
| Sensible au bruit haute fréquence | ✗ | partiellement | ✓ |
| Gradients informatifs sur supports disjoints | ✗ | ✗ | ✓ |
| Calcul rapide en haute dimension | ✓ | ✓ | difficile |

Le seul vrai inconvénient de Wasserstein : **le calcul est plus coûteux**. C'est ce que la note 05 (Sinkhorn) résout.

## IV. L'espace de Wasserstein

L'espace des distributions de probabilité (avec moment $p$ fini) muni de la distance $W_p$ forme un **espace métrique**, noté $\mathcal{P}_p(\mathbb{R}^n)$.

Pour $p = 2$ et sous certaines conditions de régularité, cet espace a une structure **riemannienne** :

- Les points sont des distributions de probabilité
- On peut définir un **espace tangent** en chaque point (champ de vecteurs gradient)
- On peut définir une **géodésique** entre deux distributions (l'interpolation de McCann)

Cette structure géométrique est l'objet de la **note 07** (Géométrie de Wasserstein), qui fait le pont avec tes notes de géométrie différentielle.

**Une intuition à retenir dès maintenant** : la géodésique de Wasserstein entre $\mu$ et $\nu$ (avec $T = \nabla \varphi$ la map de Brenier) est l'**interpolation** :

$$\mu_t = ((1-t)\, \text{Id} + t T)_\# \mu, \quad t \in [0, 1]$$

C'est-à-dire qu'on "déplace progressivement" chaque point de $\mu$ vers sa destination, à vitesse constante. Cette interpolation **préserve la forme** des distributions :

- Si $\mu$ est une gaussienne et $\nu$ est une gaussienne, alors chaque $\mu_t$ est une gaussienne
- Si $\mu$ et $\nu$ ont la même variance, l'interpolation est une simple translation

C'est radicalement différent de l'interpolation linéaire $\mu_t = (1-t)\mu + t \nu$ : celle-ci donne en général une "boule de neige" (mixture des deux), pas une vraie transformation continue.

## V. Trois idées à retenir

1. **Wasserstein-$p$** est la racine $p$-ième de la valeur optimale du transport pour coût $\|x-y\|^p$. C'est une **vraie distance** entre distributions de probabilité (symétrique, séparante, triangulaire). $W_1$ s'appelle Earth Mover's Distance et est centrale pour WGAN ; $W_2$ a une riche structure géométrique (Brenier, géodésiques).

2. **Wasserstein voit la géométrie**, contrairement à $L^p$ et KL. Pour deux distributions à supports disjoints : $W$ croît avec la distance entre les supports, tandis que $L^p$ et KL sont constantes (ou infinies). C'est ce qui rend Wasserstein utilisable comme **loss différentiable** quand les distributions ne se recouvrent pas (motivation de WGAN).

3. **Wasserstein est insensible au bruit haute fréquence et est convexe** : deux propriétés très utiles en optimisation. Le seul vrai prix à payer : le coût de calcul, qu'on résout par la régularisation entropique et Sinkhorn (note 05).

## VI. Vers la suite

- **Note 04 — Dualité de Kantorovich** : on reformule le problème primal (en termes de couplage $\gamma$) en un problème dual (en termes de fonctions $\varphi, \psi$). C'est cette reformulation qui donne $W_1 = \sup_{\|f\|_{\text{Lip}} \leq 1} \mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]$ — la base du critère WGAN.
- **Note 05 — Sinkhorn** : régularisation entropique pour rendre le calcul de Wasserstein rapide et différentiable.
- **Note 06 — Applications ML** : Wasserstein barycenters, domain adaptation, image morphing.
- **Note 07 — Géométrie de Wasserstein** : la structure riemannienne de $\mathcal{P}_2$, le théorème d'Otto, géodésiques.
