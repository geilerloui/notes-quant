---
title: Expectation-Maximization (EM)
---
# Expectation-Maximization (EM)

> Cette note construit l'algorithme **EM** depuis zéro, en suivant un fil pédagogique précis : on part de la **difficulté concrète** (maximiser une log-vraisemblance avec variables latentes), on visite deux exemples pratiques (**K-means** puis **GMM**) pour avoir une intuition, et on remonte ensuite vers la **vue abstraite** (EM = coordinate ascent sur l'ELBO) qui unifie tout. EM est l'ancêtre direct du VAE (cf. `[[02_VAE]]`) : si tu comprends EM, le saut vers VI puis VAE devient mécanique. C'est aussi la machinerie derrière les HMM, le clustering bayésien, l'imputation de données manquantes, et plein d'autres modèles à variable latente.

## Conventions de notation

> [!warning] Notation tenue partout dans cette note
> | Symbole | Sens |
> |---|---|
> | $X = \{x^{(1)}, \ldots, x^{(N)}\}$ | Observations (dataset) |
> | $Z = \{z^{(1)}, \ldots, z^{(N)}\}$ | Variables latentes (une par observation) |
> | $\theta$ | Paramètres du modèle |
> | $p_\theta(x, z)$ | Jointe du modèle, factorisée comme $p(z) \, p_\theta(x \mid z)$ |
> | $p_\theta(z \mid x)$ | Posterior — calculable en EM (par Bayes), intractable en VI |
> | $q(z)$ | Distribution variationnelle sur $z$ (objet auxiliaire) |
> | $\gamma_{nk} = p_\theta(z_n = k \mid x_n)$ | Responsabilité (cas mixture, $z$ catégorielle) |
> | $\mathcal{L}(q, \theta)$ | ELBO (lower bound) |
> 
> Si tu vois ailleurs (Andrew Ng, etc.) la notation $Q_i(z^{(i)})$, c'est juste un $q$ avec un indice $i$ par exemple. Dans EM exact c'est inutile (l'optimum est le même pour tous les $i$) — on s'en passe pour rester simple.

## I. Le problème : MLE avec variables latentes

### A. Pourquoi la log-vraisemblance est dure

On veut estimer $\theta$ par maximum de vraisemblance pour un modèle avec variables latentes :

$$\theta^* = \arg\max_\theta \; \log p_\theta(X) = \arg\max_\theta \; \sum_{n=1}^{N} \log p_\theta(x^{(n)}).$$

Chaque terme $\log p_\theta(x^{(n)})$ se développe en marginalisant sur $z^{(n)}$ :

$$\theta^* = \arg\max_\theta \; \log p_\theta(X) = \arg\max_\theta \; \log \sum_{n=1}^{N}\sum_{z^{(n)}}   p_\theta(x^{(n)}, z^{(n)}).$$


> [!warning] Le log est bloqué par la somme
> Le drame est dans cette ligne : **on a un $\log$ d'une somme**. Si la somme s'écrivait directement $\log p_\theta(x, z)$ sans le $\sum_z$, on pourrait dériver facilement (somme de log = produit de termes). Mais $\log \sum$ ne se factorise pas — chaque pas de gradient enchaîne tous les termes de la somme et produit des expressions compliquées.
> 
> Pour un mélange de gaussiennes, par exemple, $\log p_\theta(x) = \log \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ — pas de forme fermée pour l'optimum.

### B. Si on avait $z$, ce serait facile

Supposons un instant qu'on connaisse aussi $Z$ — on parle alors de **données complètes** $(X, Z)$. La log-vraisemblance des données complètes s'écrit :

$$\log p_\theta(X, Z) = \sum_{n=1}^{N} \log p_\theta(x^{(n)}, z^{(n)}) = \sum_{n=1}^{N} \log\!\big[p(z^{(n)}) \cdot p_\theta(x^{(n)} \mid z^{(n)})\big].$$

**Plus de log d'une somme.** Chaque terme est tractable, le gradient se calcule facilement, et l'optimum a souvent une forme fermée (cas conjugué) ou s'obtient par une simple descente de gradient.

> [!note] Le constat clé
> Le problème "MLE avec latents" est dur ; le problème "MLE avec données complètes" est facile. L'asymétrie est énorme.

### C. L'idée d'EM : alterner "deviner $z$" et "optimiser $\theta$"

Puisque connaître $Z$ rendrait le problème facile, l'idée d'EM est de **construire les deux quantités tour à tour** :

1. **E-step (Expectation)** — à $\theta$ fixé, calculer la distribution **du $z$ probable** sachant les observations, c'est-à-dire $p_\theta(z \mid x)$. On ne devine pas une *valeur* de $z$, mais une *distribution* sur $z$.
2. **M-step (Maximization)** — à cette distribution fixée, maximiser une fonction qui ressemble à la log-vraisemblance des données complètes, mais où chaque terme est *pondéré* par la distribution du E-step.

Et on **alterne**. À chaque itération, la log-vraisemblance des observations *augmente* (on le prouvera en §IV). En pratique, ça converge en quelques dizaines d'itérations.

**Toute la suite de la note construit rigoureusement ces deux étapes.** On commence par deux exemples concrets pour avoir l'intuition (K-means, GMM), puis on remonte vers la vue abstraite qui unifie le tout.

## II. K-means : EM dans sa version la plus simple

K-means est l'**ancêtre intuitif** d'EM, antérieur historiquement. On le présente d'abord parce qu'il rend toute la mécanique transparente — quand on aura compris K-means, on comprendra GMM comme sa généralisation probabiliste.

### A. Le modèle

Soit $X = \{x^{(1)}, \ldots, x^{(N)}\}$ un nuage de points en $\mathbb{R}^D$. On veut les partitionner en **$K$ clusters**, chacun caractérisé par un **centroïde** $\mu_k \in \mathbb{R}^D$.

Variables :
- **$\mu_1, \ldots, \mu_K$** : centroïdes (paramètres).
- **$r_{nk} \in \{0, 1\}$** : assignment "dur" — $r_{nk} = 1$ si le point $n$ appartient au cluster $k$, $0$ sinon. Contrainte : $\sum_k r_{nk} = 1$ (chaque point appartient à exactement un cluster).

Les $r_{nk}$ jouent le rôle de **variables latentes**.

### B. L'objectif (distortion)

K-means minimise la **somme des distances au carré** de chaque point à son centroïde :

$$J(\mu, r) = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \, \|x^{(n)} - \mu_k\|^2.$$

C'est une fonction de **deux ensembles de variables** : les centroïdes $\mu$ et les assignments $r$. Comme dans EM, on va alterner sur les deux.

### C. L'algorithme

**Initialisation.** Tirer $K$ points du dataset comme centroïdes initiaux.

**E-step (assignment).** À $\mu$ fixé, minimiser $J$ en $r$ :

$$r_{nk} = \begin{cases} 1 & \text{si } k = \arg\min_j \|x^{(n)} - \mu_j\|^2, \\ 0 & \text{sinon.} \end{cases}$$

Chaque point est affecté au cluster du centroïde le plus proche. Simple.

**M-step (update centroïdes).** À $r$ fixé, minimiser $J$ en $\mu_k$. La dérivée donne :

$$\frac{\partial J}{\partial \mu_k} = -2 \sum_{n=1}^{N} r_{nk}\,(x^{(n)} - \mu_k) = 0 \quad\Longrightarrow\quad \mu_k = \frac{\sum_n r_{nk} \, x^{(n)}}{\sum_n r_{nk}}.$$

Le nouveau centroïde est la **moyenne des points assignés** au cluster $k$.

**Convergence.** On itère E-step et M-step jusqu'à ce que les assignments ne changent plus. À chaque étape, $J$ **décroît strictement** (ou reste constant à convergence) — donc l'algorithme converge en un nombre fini d'étapes vers un *minimum local* de $J$.

![[kmeans_iterations.png]]
*Figure. K-means en action sur un dataset 2D synthétique avec 3 clusters. Itérations 0 (initialisation aléatoire), 1, 2, 5. Les centroïdes (croix) bougent et les points se réorganisent. À droite : la distortion $J$ qui décroît monotonement.*

### D. Limites de K-means

K-means marche bien mais a deux faiblesses structurelles :

- **Assignments durs.** Un point est soit dans un cluster, soit dans un autre. Aucune notion d'incertitude. Pour un point à mi-chemin entre deux clusters, ça force un choix arbitraire.
- **Clusters sphériques.** La distance euclidienne $\|x - \mu_k\|^2$ traite toutes les directions également. Un cluster allongé ou orienté oblique sera mal capturé.

**GMM** résout ces deux limites en passant à un cadre probabiliste : assignments *probabilistes* (responsabilités), et clusters *gaussiens* (forme libre).

## III. Mixture of Gaussians (GMM) : EM "complet"

### A. Le modèle

Un GMM modélise la densité des données comme un **mélange linéaire de $K$ gaussiennes** :

$$p_\theta(x)=\sum_k p(z=k) \cdot p_\theta(x \mid z=k)=\sum_k \pi_k \mathcal{N}\left(x \mid \mu_k, \Sigma_k\right) .$$

avec :
- **$\pi_k$** : poids de mélange (mixing coefficient), $\sum_k \pi_k = 1$, $\pi_k \geq 0$.
- **$\mathcal{N}(x \mid \mu_k, \Sigma_k)$** : densité gaussienne de moyenne $\mu_k$ et covariance $\Sigma_k$.
- **$\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^{K}$** : paramètres à apprendre.

![[gmm_mixture_intro.png]]
*Figure. Un GMM à 2 composantes en 1D. **Tirets bleus** et **tirets oranges** : les deux composantes pondérées $\pi_k \cdot \mathcal{N}(x \mid \mu_k, \sigma_k)$. **Trait plein violet** : le mélange total $p_\theta(x) = \pi_1 \mathcal{N}_1 + \pi_2 \mathcal{N}_2$. Les flèches verticales mesurent la densité du mélange aux 6 points observés — exactement comme pour le MLE classique (`[[04_Estimations]]`), sauf que la distribution est maintenant un mélange. Pour les points du cluster gauche, c'est surtout la composante bleue qui contribue à $p_\theta(x_i)$ ; pour ceux du cluster droit, c'est l'orange. Cette décomposition naturelle est ce que la variable latente $z$ va matérialiser dans la section suivante.*

### B. La variable latente $z$

Pour appliquer EM, on **introduit explicitement** une variable latente $z$ qui code le cluster d'origine. On utilise une représentation 1-of-K : $z = (z_1, \ldots, z_K)$ avec un seul $z_k = 1$, les autres $z_j = 0$.

Le modèle se factorise alors comme :

$$p(z_k = 1) = \pi_k, \qquad p(x \mid z_k = 1) = \mathcal{N}(x \mid \mu_k, \Sigma_k).$$

Ou plus compactement :

$$p(z) = \prod_k \pi_k^{z_k}, \qquad p_\theta(x \mid z) = \prod_k \mathcal{N}(x \mid \mu_k, \Sigma_k)^{z_k}.$$

> [!warning] Comprendre la notation 1-of-K
> Cette notation peut surprendre la première fois. Décodons.
> 
> **La façon naïve d'écrire $z$.** On pourrait simplement dire que $z \in \{1, 2, \ldots, K\}$ est un **entier** qui désigne le cluster d'origine. Par exemple $z = 2$ signifie "le point vient du cluster 2". Avec cette écriture :
> 
> $$p(z = k) = \pi_k, \qquad p(x \mid z = k) = \mathcal{N}(x \mid \mu_k, \Sigma_k).$$
> 
> Une définition par cas, parfaitement claire mais lourde à manipuler dans des formules.
> 
> **La façon 1-of-K.** Au lieu d'un entier, on stocke un **vecteur binaire** $z = (z_1, \ldots, z_K)$ avec exactement une composante égale à $1$ (celle du cluster choisi), toutes les autres à $0$.
> 
> | Version entier | Version 1-of-K (cas $K = 3$) |
> |---|---|
> | $z = 1$ | $z = (1, 0, 0)$ |
> | $z = 2$ | $z = (0, 1, 0)$ |
> | $z = 3$ | $z = (0, 0, 1)$ |
> 
> **Les deux écritures portent strictement la même information.**
> 
> **L'astuce de la formule compacte.** Avec la version 1-of-K, l'écriture $\prod_k \pi_k^{z_k}$ **sélectionne automatiquement** la bonne valeur sans cas par cas. Vérifions sur $z = (0, 1, 0)$ :
> 
> $$p(z) = \pi_1^{0} \cdot \pi_2^{1} \cdot \pi_3^{0} = 1 \cdot \pi_2 \cdot 1 = \pi_2.$$
> 
> Magie : tous les facteurs sauf un valent $1$ (car $\pi^0 = 1$), et le seul facteur non trivial est celui du cluster sélectionné. Pareil pour $p_\theta(x \mid z) = \prod_k \mathcal{N}_k^{z_k}$ : seule la gaussienne $\mathcal{N}_k$ correspondant au $k$ tel que $z_k = 1$ "survit".
> 
> **Pourquoi se compliquer la vie ?** Parce qu'au moment de calculer la jointe $p_\theta(x, z) = p(z) \cdot p_\theta(x \mid z)$ et de prendre son log, le $z_k$ en exposant **descend devant le $\log$** :
> 
> $$\log p_\theta(x, z) = \sum_{k=1}^{K} z_k \, \log\!\big[\pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)\big].$$
> 
> Une seule formule sans cas par cas, et c'est exactement la forme qui permet de prendre $\mathbb{E}_z[\log p_\theta(x, z)]$ proprement dans le M-step — l'espérance de $z_k$ deviendra la responsabilité $\gamma_{nk}$ (cf. §III.C et les preuves en §III.G). Sans cette astuce, on devrait jongler avec des "si $z = 1$ alors..., si $z = 2$ alors..." partout.
> 
> **Pour ta lecture quotidienne.** Mentalement, remplace toujours "$z_k = 1$" par "$z = k$" si ça t'aide. Les deux notations sont équivalentes — la 1-of-K est juste une convention d'écriture standard dans les livres (Bishop, Murphy) qui rend les formules plus compactes.

Et on retrouve bien la marginale :

$$p_\theta(x) = \sum_{z} p(z) \, p_\theta(x \mid z) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k).$$

![[gmm_dag.png]]
*Figure. DAG du modèle GMM. À gauche : représentation dépliée pour $N$ observations. À droite : plate notation. Les $z^{(n)}$ sont latents (non grisés), les $x^{(n)}$ sont observés (grisés). Les paramètres $\pi, \mu, \Sigma$ gouvernent toutes les variables.*

### C. La responsabilité $\gamma_{nk}$

L'objet central pour EM dans un GMM, c'est la **responsabilité** :

$$\gamma_{nk} := p_\theta(z_n = k \mid x_n) = \frac{p(x_n,z=k)}{p(x_n)} = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}.$$

C'est la posterior du modèle pour le point $n$ : sachant qu'on a observé $x_n$, quelle est la probabilité qu'il vienne du cluster $k$ ?

![[Pasted image 20260521161844.png]]

> [!warning] Responsabilité = assignment doux
> $\gamma_{nk}$ est le **pendant probabiliste** de $r_{nk}$ dans K-means :
> 
> - Dans K-means, $r_{nk} \in \{0, 1\}$ : dur, sec.
> - Dans GMM, $\gamma_{nk} \in [0, 1]$ avec $\sum_k \gamma_{nk} = 1$ : doux, gradué.
> 
> Un point "ambigu" entre deux clusters aura par exemple $\gamma_{n1} = 0.6, \gamma_{n2} = 0.4$ — il "appartient à 60% au cluster 1, 40% au cluster 2". On gagne la notion d'incertitude que K-means n'avait pas.

![[gmm_responsibilities.png]]
*Figure. Responsabilités $\gamma_{nk}$ sur un dataset 2D avec 3 clusters. Chaque point est colorié selon ses responsabilités vues comme un mélange RGB. Les points au cœur d'un cluster sont nets (pure couleur), les points aux frontières sont teintés (mélange). Comparer à la coloration dure de K-means qui n'a que 3 couleurs sans nuance.*

### D. L'algorithme EM pour GMM

**Initialisation.** Initialiser $\pi_k, \mu_k, \Sigma_k$ (souvent par K-means pour faciliter la convergence).

**E-step.** Calculer les responsabilités pour le $\theta$ courant :

$$\gamma_{nk} = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}.$$

**M-step.** Mettre à jour les paramètres en utilisant les responsabilités comme poids :

$$\boxed{\;\mu_k^{\text{new}} = \frac{\sum_n \gamma_{nk} \, x_n}{N_k}, \quad \Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_n \gamma_{nk} \, (x_n - \mu_k^{\text{new}})(x_n - \mu_k^{\text{new}})^\top, \quad \pi_k^{\text{new}} = \frac{N_k}{N}\;}$$

où $N_k := \sum_{n=1}^{N} \gamma_{nk}$ est le "nombre effectif" de points dans le cluster $k$.

**Évaluer la log-vraisemblance** :

$$\log p_\theta(X) = \sum_{n=1}^{N} \log\!\left[\sum_{k=1}^{K} \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)\right].$$

Si elle a convergé (variation < tolérance), stopper. Sinon, retourner au E-step.

![[gmm_iterations.png]]
*Figure. EM sur GMM en action. Quatre panneaux : initialisation, après 1 itération, après 5, après convergence (~30). Les ellipses montrent les contours à 1σ et 2σ de chaque gaussienne. Les points sont colorés selon leurs responsabilités. La densité de mélange (heatmap en arrière-plan) s'ajuste progressivement.*

### E. Comparer avec K-means

| Aspect | K-means | EM-GMM |
|---|---|---|
| Variable latente | $r_{nk}$ — hard, 0/1 | $\gamma_{nk}$ — soft, $\in [0, 1]$ |
| Forme du cluster | Sphère (norme $L_2$) | Ellipsoïde (covariance $\Sigma_k$) |
| Cadre | Géométrique (minimiser distortion) | Probabiliste (maximiser vraisemblance) |
| Paramètres par cluster | $D$ (centroïde) | $D + D^2/2 + 1$ (moyenne + cov + poids) |
| Cas particulier | Cas limite de GMM quand $\Sigma_k = \epsilon I$ et $\epsilon \to 0$ | — |

> [!note]- K-means est-il vraiment un cas limite d'EM-GMM ?
> Oui. Si on force $\Sigma_k = \epsilon I$ avec $\epsilon \to 0$, alors la responsabilité $\gamma_{nk}$ devient :
> 
> $$\gamma_{nk} \xrightarrow{\epsilon \to 0} \begin{cases} 1 & \text{si } k = \arg\min_j \|x_n - \mu_j\|^2, \\ 0 & \text{sinon.} \end{cases}$$
> 
> C'est exactement $r_{nk}$ de K-means. Et l'update des $\mu_k$ via la formule GMM redevient la moyenne des points assignés. K-means est donc EM-GMM avec covariance nulle — assignments durs, clusters sphériques.

### F. Issues pratiques

> [!warning] Les pièges classiques du GMM
> **1. Singularités.** Si une composante "colle" sur un seul point ($\mu_k = x_n$, $\Sigma_k \to 0$), sa densité $\mathcal{N}(x_n \mid x_n, \Sigma_k) \to \infty$. La log-vraisemblance diverge — l'optimisation est mal posée. *Solutions* : régulariser (ajouter $\epsilon I$ à $\Sigma_k$), ou faire du bayésien sur les paramètres.
> 
> **2. Non-identifiabilité.** Plusieurs $\theta$ produisent la même densité de mélange :
> - **Label switching** : permuter les clusters donne la même $p(x)$. Pour $K$ clusters, il y a $K!$ minima équivalents.
> - **Composantes vides** : ajouter une composante à poids $\pi_k = 0$ ne change rien.
> - **Composantes dupliquées** : deux composantes identiques peuvent être "fusionnées" sans changer $p(x)$.
> 
> Ce n'est pas un défaut algorithmique — c'est une propriété structurelle des mélanges, qui complique l'interprétation des paramètres optimisés.

### G. Preuves des updates M-step

Les formules $\mu_k^{\text{new}}, \Sigma_k^{\text{new}}, \pi_k^{\text{new}}$ ne sortent pas du chapeau — elles viennent de résoudre $\nabla_\theta \log p_\theta(X) = 0$ en utilisant les responsabilités. Les preuves sont techniques mais éclairantes.

> [!note]- Preuve : update de $\mu_k$
> On veut $\partial \mathcal{L} / \partial \mu_k = 0$ où $\mathcal{L} = \log p_\theta(X)$. La log-vraisemblance s'écrit :
> 
> $$\mathcal{L}(\theta) = \sum_{n=1}^{N} \log \sum_{j=1}^{K} \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j).$$
> 
> En dérivant terme à terme :
> 
> $$\frac{\partial \mathcal{L}}{\partial \mu_k} = \sum_{n=1}^{N} \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)} \cdot \Sigma_k^{-1} (x_n - \mu_k).$$
> 
> On reconnaît la responsabilité $\gamma_{nk}$ dans le facteur de gauche :
> 
> $$\frac{\partial \mathcal{L}}{\partial \mu_k} = \Sigma_k^{-1} \sum_{n=1}^{N} \gamma_{nk}\, (x_n - \mu_k).$$
> 
> En annulant (et en multipliant par $\Sigma_k$ supposée inversible) :
> 
> $$\sum_n \gamma_{nk} x_n = \mu_k \sum_n \gamma_{nk} = \mu_k \, N_k \quad\Longrightarrow\quad \mu_k = \frac{1}{N_k} \sum_n \gamma_{nk} x_n. \;\blacksquare$$

> [!note]- Preuve : update de $\Sigma_k$
> Calcul similaire mais plus technique : on utilise les identités matricielles
> 
> $$\frac{\partial}{\partial \Sigma_k} \det(\Sigma_k)^{-1/2} = -\frac{1}{2} \det(\Sigma_k)^{-1/2} \Sigma_k^{-1}, \quad \frac{\partial}{\partial \Sigma_k} (x_n-\mu_k)^\top \Sigma_k^{-1} (x_n-\mu_k) = -\Sigma_k^{-1}(x_n-\mu_k)(x_n-\mu_k)^\top \Sigma_k^{-1}.$$
> 
> En annulant $\partial \mathcal{L}/\partial \Sigma_k$ et en simplifiant, on obtient :
> 
> $$\Sigma_k = \frac{1}{N_k} \sum_n \gamma_{nk}\, (x_n - \mu_k)(x_n - \mu_k)^\top. \;\blacksquare$$

> [!note]- Preuve : update de $\pi_k$ (avec contrainte $\sum_k \pi_k = 1$)
> On introduit un multiplicateur de Lagrange $\lambda$ pour la contrainte :
> 
> $$\mathfrak{L} = \mathcal{L} + \lambda\Big(\sum_{k=1}^{K} \pi_k - 1\Big).$$
> 
> En dérivant : $\partial \mathfrak{L} / \partial \pi_k = N_k/\pi_k + \lambda = 0$, donc $\pi_k = -N_k/\lambda$. La contrainte $\sum_k \pi_k = 1$ donne $\lambda = -N$, d'où :
> 
> $$\pi_k = \frac{N_k}{N}. \;\blacksquare$$

## IV. La vue abstraite : EM comme coordinate ascent sur l'ELBO

Maintenant qu'on a vu EM "en action" sur K-means et GMM, on remonte vers la **vue générale**. Cette vue répond aux questions qu'on a laissées en suspens :
- *Pourquoi* alterner E-step et M-step fait monter la log-vraisemblance ?
- Que fait *exactement* le E-step en termes mathématiques ?
- Comment EM se généralise à n'importe quel modèle à variable latente ?

### A. La décomposition fondamentale

L'observation clé qui débloque tout : pour **n'importe quelle** distribution $q(z)$ sur les variables latentes, on a l'identité

$$\boxed{\;\log p_\theta(x) \;=\; \mathcal{L}(q, \theta) \;+\; D_{\text{KL}}\!\big(q(z) \,\|\, p_\theta(z \mid x)\big)\;}$$

avec

$$\mathcal{L}(q, \theta) \;:=\; \sum_z q(z) \log \frac{p_\theta(x, z)}{q(z)} \quad\text{et}\quad D_{\text{KL}}(q \| p) \;:=\; -\sum_z q(z) \log \frac{p_\theta(z \mid x)}{q(z)}.$$

$\mathcal{L}(q, \theta)$ est l'**ELBO** (evidence lower bound). $D_{\text{KL}} \geq 0$ toujours, avec égalité ssi $q = p_\theta(z \mid x)$.

> [!note]- Preuve de la décomposition
> Pour toute $q(z)$ avec $\sum_z q(z) = 1$ :
> 
> $$\log p_\theta(x) = \sum_z q(z) \log p_\theta(x) \quad\text{(car }\sum_z q(z) = 1\text{)}.$$
> 
> On injecte $p_\theta(x) = p_\theta(x, z) / p_\theta(z \mid x)$ (Bayes) :
> 
> $$\log p_\theta(x) = \sum_z q(z) \log \frac{p_\theta(x, z)}{p_\theta(z \mid x)} = \sum_z q(z) \log \frac{p_\theta(x, z)}{q(z)} \cdot \frac{q(z)}{p_\theta(z \mid x)}.$$
> 
> En séparant les deux logs :
> 
> $$\log p_\theta(x) = \underbrace{\sum_z q(z) \log \frac{p_\theta(x, z)}{q(z)}}_{= \mathcal{L}(q, \theta)} + \underbrace{\sum_z q(z) \log \frac{q(z)}{p_\theta(z \mid x)}}_{= D_{\text{KL}}(q \| p_\theta(z \mid x))}. \;\blacksquare$$

### B. Conséquences immédiates

1. **L'ELBO est une borne inférieure** : $\mathcal{L}(q, \theta) \leq \log p_\theta(x)$ pour tout $q$, parce que $D_{\text{KL}} \geq 0$.

2. **La borne est serrée quand $q = p_\theta(z \mid x)$** : à ce moment-là $D_{\text{KL}} = 0$ et $\mathcal{L} = \log p_\theta(x)$ exactement.

3. **Si on maximise l'ELBO sur $q$ à $\theta$ fixé, on obtient $q^* = p_\theta(z \mid x)$** — la posterior. Et c'est *exactement* la formule des responsabilités du E-step GMM (§III.C) !

### C. EM comme coordinate ascent

Voilà l'angle qui éclaire tout : EM est l'**ascent coordonné** sur l'ELBO en alternant l'optimisation sur $q$ et sur $\theta$.

> [!warning] EM = maximiser l'ELBO sur $q$ puis sur $\theta$, alternativement
> **E-step** : à $\theta^{\text{old}}$ fixé, maximiser l'ELBO $\mathcal{L}(q, \theta^{\text{old}})$ sur $q$.
> $$q^{\text{new}} = \arg\max_q \mathcal{L}(q, \theta^{\text{old}}) = p_{\theta^{\text{old}}}(z \mid x).$$
> Après le E-step, $D_{\text{KL}} = 0$, donc $\mathcal{L}(q^{\text{new}}, \theta^{\text{old}}) = \log p_{\theta^{\text{old}}}(x)$. **La borne touche.**
> 
> **M-step** : à $q^{\text{new}}$ fixé, maximiser $\mathcal{L}(q^{\text{new}}, \theta)$ sur $\theta$.
> $$\theta^{\text{new}} = \arg\max_\theta \mathcal{L}(q^{\text{new}}, \theta).$$
> L'ELBO monte. Et comme $\log p_\theta(x) \geq \mathcal{L}(q^{\text{new}}, \theta)$, **la log-vraisemblance monte aussi.**

Cette présentation est très abstraite — déroulons-la sur notre histoire GMM à 2 gaussiennes pour la rendre concrète. On garde $\pi_1, \pi_2, \sigma_1, \sigma_2, \mu_2$ fixés à leurs vraies valeurs, et on fait varier **un seul paramètre** $\mu_1$ pour pouvoir tout dessiner sur un graphe 1D.

**Ce qu'on optimise.** L'objectif est $\ell(\mu_1) := \log p_\theta(X) = \sum_n \log[\pi_1 \mathcal{N}(x_n \mid \mu_1, \sigma_1) + \pi_2 \mathcal{N}(x_n \mid \mu_2, \sigma_2)]$ — la **courbe bleue** sur la figure ci-dessous. C'est elle qu'on cherche à maximiser, mais on ne sait pas attaquer directement le $\log \sum$ (cf. §I.A).

**Itération $t = 0$ avec un mauvais $\mu_1^{(0)} = -1$.**

- *E-step.* À $\mu_1^{(0)}$ fixé, on calcule la posterior $q^{(0)}(z) = p_{\mu_1^{(0)}}(z \mid x)$. Concrètement pour GMM : on calcule les responsabilités $\gamma_{nk}^{(0)}$ pour chaque point. Comme $\mu_1^{(0)}$ est mal placé, ces responsabilités sont "mauvaises" en valeur absolue — mais elles sont **cohérentes avec $\mu_1^{(0)}$**, c'est ce qui compte.

- *Construction de l'ELBO comme fonction de $\mu_1$.* On fige $q^{(0)}$ et on regarde l'ELBO $\mathcal{L}(\mu_1 \mid q^{(0)})$ varier quand $\mu_1$ varie. C'est la **parabole verte $g_0$** sur la figure. Trois faits cruciaux :
    - $g_0$ est **sous** la courbe bleue partout (c'est une borne inférieure).
    - $g_0$ **touche** la courbe bleue exactement en $\mu_1^{(0)}$ (parce que $q^{(0)} = p_{\mu_1^{(0)}}(z \mid x)$ ferme la KL à zéro).
    - $g_0$ est une parabole parce que $\log \mathcal{N}(x_n \mid \mu_1, \sigma_1) = -(x_n - \mu_1)^2 / (2\sigma_1^2) + \text{cst}$ est quadratique en $\mu_1$, et une somme pondérée de quadratiques reste quadratique.

- *M-step.* On grimpe au sommet de $g_0$ → on obtient $\mu_1^{(1)}$. Comme $g_0$ touche la courbe bleue en $\mu_1^{(0)}$ et reste dessous ailleurs, monter sur $g_0$ ne peut **pas** faire descendre la vraie log-vraisemblance — au pire elle reste pareille, au mieux elle monte. C'est la garantie de monotonie d'EM.

**Itération $t = 1$ avec $\mu_1^{(1)}$ meilleur.** On recommence : nouveau E-step → nouveau $q^{(1)}$ → nouvelle parabole $g_1$ tangente à la courbe bleue en $\mu_1^{(1)}$ → M-step → $\mu_1^{(2)}$.

**Itération $t = 2$.** Pareil, parabole $g_2$ tangente en $\mu_1^{(2)}$, sommet en $\mu_1^{(3)}$. À ce stade on est très proche du MLE.

> [!note]- L'équation explicite des paraboles $g_t$
> Pour notre setup (un seul $\mu_1$ à optimiser), on peut écrire l'ELBO de façon entièrement explicite. En ne gardant que ce qui dépend de $\mu_1$ :
> 
> $$g_t(\mu_1) \;=\; -\frac{1}{2 \sigma_1^2} \sum_{n=1}^{N} \gamma_{n,1}^{(t)} \, (x_n - \mu_1)^2 \;+\; C^{(t)}$$
> 
> où $C^{(t)}$ regroupe toutes les contributions qui ne dépendent pas de $\mu_1$ (termes en $\mu_2, \pi_k$, entropie de $q^{(t)}$, normalisations). $C^{(t)}$ dépend de $t$ mais ne change pas la **forme** de la parabole — juste son décalage vertical.
> 
> En développant le carré, c'est une parabole standard $g_t(\mu_1) = A^{(t)} \mu_1^2 + B^{(t)} \mu_1 + D^{(t)}$ avec :
> 
> $$A^{(t)} = -\frac{N_1^{(t)}}{2 \sigma_1^2}, \quad B^{(t)} = \frac{1}{\sigma_1^2} \sum_n \gamma_{n,1}^{(t)} x_n, \quad N_1^{(t)} := \sum_n \gamma_{n,1}^{(t)}.$$
> 
> **Ce qui change entre $g_0, g_1, g_2$ : uniquement les $\gamma_{n,1}^{(t)}$.** Tout le reste ($\sigma_1$, $x_n$) est fixe. Quand les responsabilités évoluent (plus tranchées au fil des itérations), les coefficients $A^{(t)}, B^{(t)}$ changent → la parabole se redessine, plus serrée et mieux centrée à chaque tour.
> 
> **Le sommet de la parabole** se trouve en $\mu_1^* = -B^{(t)} / (2 A^{(t)})$, ce qui donne :
> 
> $$\mu_1^{(t+1)} \;=\; \frac{\sum_n \gamma_{n,1}^{(t)} \, x_n}{N_1^{(t)}}.$$
> 
> **C'est exactement la formule du M-step GMM** (§III.D) ! "Grimper au sommet de la parabole $g_t$" = "appliquer la formule M-step". La géométrie et l'algorithme disent la même chose.

![[em_elbo_iterations.png]]
*Figure. EM sur le modèle GMM en faisant varier seulement $\mu_1$ (autres paramètres fixés). **Courbe bleue** : la vraie log-vraisemblance $\ell(\mu_1) = \log p_\theta(X)$ — l'objectif qu'on veut maximiser. **Trois paraboles vertes** $g_0, g_1, g_2$ : les ELBO successives, chacune tangente à la courbe bleue au point $\mu_1^{(t)}$ courant. Le M-step à l'itération $t$ amène au sommet de $g_t$, qui devient le nouveau point $\mu_1^{(t+1)}$ sur la courbe bleue. Init très mauvaise ($\mu_1^{(0)} = -1$) pour rendre la progression visible.*

> [!warning] Récapitulatif des objets à ne pas confondre
> | Objet | Type | En GMM concret |
> |---|---|---|
> | $q^{(t)}(z)$ | Une **distribution sur $z$** (codée par les responsabilités) | $N$ paires $(\gamma_{n,1}^{(t)}, \gamma_{n,2}^{(t)})$ — **des bâtons** |
> | $g_t(\mu_1) = \mathcal{L}(\mu_1 \mid q^{(t)})$ | Une **fonction de $\theta$** (ici de $\mu_1$) calculée à partir de $q^{(t)}$ | Une **parabole** sur la figure |
> | $\mu_1^{(t)}$ | Une valeur scalaire du paramètre courant | Un point sur la courbe bleue |
> 
> $q^{(t)}$ et $g_t$ ne sont **pas** le même objet. $q^{(t)}$ est l'ingrédient ; $g_t$ est ce qu'on en fait. C'est exactement comme la farine et le pain.

**Pourquoi $q$ s'améliore en même temps que $\theta$.** Au début, $\mu_1^{(0)} = -1$ donne des responsabilités $\gamma_{n,1}^{(0)}$ ambiguës et mal placées (la gaussienne centrée en $-1$ est mal positionnée par rapport aux vrais clusters). Au fil des itérations, $\mu_1^{(t)}$ s'approche de la vraie valeur, donc les responsabilités calculées par le E-step deviennent de plus en plus **tranchées** (proches de 0 ou 1). $q^{(t)}$ et $\theta^{(t)}$ se renforcent mutuellement — exactement comme les centroïdes et les assignments dans K-means.

### D. Pourquoi EM converge

À chaque itération :

$$\log p_{\theta^{\text{new}}}(x) \;\geq\; \mathcal{L}(q^{\text{new}}, \theta^{\text{new}}) \;\geq\; \mathcal{L}(q^{\text{new}}, \theta^{\text{old}}) \;=\; \log p_{\theta^{\text{old}}}(x).$$

- La première inégalité : ELBO ≤ log-vraisemblance (définition).
- La seconde : le M-step a maximisé l'ELBO sur $\theta$, donc $\mathcal{L}$ a au moins gardé sa valeur.
- L'égalité : après le E-step, l'ELBO touchait la log-vraisemblance.

Conclusion : $\log p_{\theta^{\text{new}}}(x) \geq \log p_{\theta^{\text{old}}}(x)$. **EM fait monotonement monter la log-vraisemblance**, itération après itération.

Comme la log-vraisemblance est bornée supérieurement (sauf pathologie de singularité), la suite converge vers un **point critique** — typiquement un maximum local. EM ne garantit *pas* le maximum global, mais il garantit la monotonie, ce qui est déjà énorme : pas de divergence, pas d'oscillation.

### E. La formulation équivalente avec $\mathcal{Q}(\theta, \theta^{\text{old}})$

Beaucoup de cours (Bishop, MacKay) présentent le M-step sous une forme **équivalente** :

$$\theta^{\text{new}} = \arg\max_\theta \, \mathcal{Q}(\theta, \theta^{\text{old}}) \quad \text{avec} \quad \mathcal{Q}(\theta, \theta^{\text{old}}) := \mathbb{E}_{z \sim p_{\theta^{\text{old}}}(z \mid x)}\!\big[\log p_\theta(x, z)\big].$$

> [!note]- Pourquoi cette forme est équivalente
> Après le E-step, $q^{\text{new}}(z) = p_{\theta^{\text{old}}}(z \mid x)$. L'ELBO devient :
> 
> $$\mathcal{L}(q^{\text{new}}, \theta) = \sum_z p_{\theta^{\text{old}}}(z \mid x) \log \frac{p_\theta(x, z)}{p_{\theta^{\text{old}}}(z \mid x)} = \mathbb{E}[\log p_\theta(x, z)] - \mathbb{E}[\log p_{\theta^{\text{old}}}(z \mid x)].$$
> 
> Le second terme ne dépend pas de $\theta$ (constant dans le M-step), donc :
> 
> $$\arg\max_\theta \mathcal{L}(q^{\text{new}}, \theta) = \arg\max_\theta \mathbb{E}[\log p_\theta(x, z)] = \arg\max_\theta \mathcal{Q}(\theta, \theta^{\text{old}}). \;\blacksquare$$

C'est la même chose, mais on ne traîne pas le terme constant. Quand on dit "maximiser l'espérance de la log-vraisemblance des données complètes", c'est ça. On retombe d'ailleurs sur les preuves GMM (§III.G) où on a maximisé l'espérance de $\log p_\theta(X, Z)$.

### F. L'algorithme général d'EM

> [!warning] EM en toute généralité
> Soit un modèle à variable latente $p_\theta(x, z) = p(z) \, p_\theta(x \mid z)$ avec posterior $p_\theta(z \mid x)$ calculable.
> 
> 1. **Initialiser** $\theta^{(0)}$.
> 2. **Itérer** jusqu'à convergence :
>    - **E-step** : calculer la posterior $p_{\theta^{(t)}}(z \mid x)$ pour chaque observation.
>    - **M-step** : maximiser $\mathcal{Q}(\theta, \theta^{(t)}) = \mathbb{E}_{z \sim p_{\theta^{(t)}}(z \mid x)}[\log p_\theta(x, z)]$ en $\theta$ → obtenir $\theta^{(t+1)}$.
> 3. **Convergence** : log-vraisemblance stagne → stop.
> 
> Si la posterior n'est **pas** calculable, on ne peut plus faire EM exact. Deux options : approximer (→ Variational Inference, cf. `[[06_Variational_Inference]]`) ou échantillonner (→ Monte Carlo EM, §V.A).

## V. Variations

### A. Monte Carlo EM

Quand la posterior $p_\theta(z \mid x)$ n'est pas calculable analytiquement, on peut l'**échantillonner** par MCMC (cf. `[[MCMC]]`). On remplace l'espérance du M-step par une moyenne empirique :

$$\mathcal{Q}(\theta, \theta^{\text{old}}) \approx \frac{1}{L} \sum_{l=1}^{L} \log p_\theta(x, z^{(l)}), \quad z^{(l)} \sim p_{\theta^{\text{old}}}(z \mid x).$$

**Stochastic EM** : cas limite où on tire **un seul** $z$ par itération (assignment dur stochastique). Utilisé en pratique sur des modèles trop gros pour calculer les responsabilités complètes.

> [!warning] La jointe $p_\theta(x, z)$ n'est presque jamais "gratuite"
> Dire "MCEM s'utilise quand la posterior n'est pas calculable" donne l'impression que la jointe $p_\theta(x, z)$, elle, l'est toujours — comme dans GMM où $p(x, z = k) = \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$ se lit directement. **C'est un cas particulier**, pas la règle.
> 
> Pour la plupart des modèles à variable latente qu'on rencontre en pratique, la jointe est elle aussi piégeuse :
> 
> - **Latent continu + likelihood non-linéaire** (VAE, modèles bayésiens non-conjugués) : on évalue $p_\theta(x, z) = p(z) p_\theta(x \mid z)$ pointwise, mais la marginale $\int p_\theta(x, z) \, dz$ est intractable — et donc la posterior $p_\theta(z \mid x) = p_\theta(x, z) / p_\theta(x)$ aussi.
> - **Energy-based models, MRFs** : $p_\theta(x, z) = \frac{1}{Z_\theta} \exp(-E_\theta(x, z))$ avec $Z_\theta$ inconnu. La jointe n'est même pas évaluable pointwise — seulement à une constante près.
> - **Latents discrets de grande dimension** ($z \in \{0,1\}^D$ avec $D$ grand : RBM, sigmoid belief nets) : la jointe se calcule, mais la marginalisation pour Bayes coûterait $2^D$ termes.
> - **Latents structurés** (HMM, CRF denses) : factorisation locale tractable, marginale globale exponentielle.
> 
> **Pourquoi MCEM marche quand même.** MCMC (cf. `[[MCMC]]`) ne demande pas la posterior normalisée. Metropolis–Hastings n'a besoin que de la jointe **à une constante près**, et Gibbs n'a besoin que des conditionnelles complètes $p_\theta(z_i \mid z_{-i}, x)$ — souvent simples même quand la jointe globale est monstrueuse. C'est ce qui rend MCEM utilisable bien au-delà du cadre confortable du GMM.
> 
> **Ce que GMM rendait invisible.** GMM est le cas où *tout* est gratuit : jointe explicite, marginale finie ($K$ termes), posterior par règle des trois. C'est pour ça qu'on l'introduit en premier — mais ça fausse l'intuition sur ce qui est "facile" en général.

### B. EM pour mixture de Bernoulli

Pour des données **binaires** (par exemple MNIST seuillé), on remplace les gaussiennes par des Bernoulli. Chaque composante $k$ est paramétrée par $\mu_k \in [0, 1]^D$ (probabilité d'activation de chaque dimension) :

$$p(x \mid \mu_k) = \prod_{i=1}^{D} \mu_{ki}^{x_i} (1 - \mu_{ki})^{1 - x_i}.$$

Et le mélange : $p(x \mid \mu, \pi) = \sum_k \pi_k \, p(x \mid \mu_k)$.

**E-step** identique à GMM : $\gamma_{nk} = \pi_k p(x_n \mid \mu_k) / \sum_j \pi_j p(x_n \mid \mu_j)$.

**M-step** :

$$\mu_k = \frac{1}{N_k} \sum_n \gamma_{nk} \, x_n, \qquad \pi_k = \frac{N_k}{N}.$$

L'update de $\mu_k$ est la moyenne pondérée par les responsabilités — exactement comme pour GMM, sauf qu'il n'y a pas de covariance (chaque composante est diagonale par construction).

### C. EM pour données manquantes

Variante très utile en pratique. Si certaines composantes de $x$ sont manquantes, on peut les traiter comme des variables latentes et faire tourner EM dessus : E-step = remplir les valeurs manquantes par leur espérance sous le modèle ; M-step = ré-estimer les paramètres sur les données complétées.

> [!todo] À étoffer dans une future version
> EM pour Bayesian linear regression, HMM (Baum-Welch), Probabilistic PCA (`[[PPCA]]` à venir), Factor Analysis.

## VI. Pourquoi EM compte pour la suite

### A. Pont vers Variational Inference

EM exact suppose que **la posterior $p_\theta(z \mid x)$ est calculable**. Pour les modèles simples (mélanges discrets, PPCA), c'est le cas. Mais pour beaucoup de modèles modernes — VAE en particulier — la posterior est intractable.

**Variational Inference** (cf. `[[06_Variational_Inference]]`) généralise EM à ce cadre : au lieu de prendre $q^* = p_\theta(z \mid x)$ (impossible), on cherche le meilleur $q$ dans une famille paramétrique restreinte (par exemple gaussiennes). On accepte une borne lâche pour gagner la tractabilité.

> [!note] La généalogie
> $$\underbrace{\text{EM exact}}_{p_\theta(z\mid x) \text{ tractable}} \;\xrightarrow{\text{posterior intractable}}\; \underbrace{\text{Variational Inference}}_{q \text{ paramétrique}} \;\xrightarrow{\text{un } q \text{ par exemple}}\; \underbrace{\text{Amortizée (VAE)}}_{q_\phi(z\mid x) \text{ encodeur}}$$

### B. Pont vers VAE

Dans `[[02_VAE]]`, on a vu que :
- L'ELBO du VAE est exactement la même quantité que l'ELBO d'EM ici.
- L'encodeur $q_\phi(z \mid x)$ joue le rôle du E-step amortizé : il approche la posterior intractable.
- Le décodeur $p_\theta(x \mid z)$ joue le rôle du modèle génératif paramétré.
- Le terme reconstruction + KL de la loss VAE est l'ELBO réécrit.

**Le VAE n'est rien d'autre qu'EM appliqué à un modèle dont la posterior est intractable, avec un réseau de neurones pour amortir le E-step.**

C'est *exactement* la même histoire, juste compliquée par (a) un latent continu, (b) un décodeur non-linéaire, (c) une astuce de reparametrization pour rendre le tout différentiable.

---

## Pour aller plus loin

- **Bishop, *Pattern Recognition and Machine Learning*, chapitre 9.** La référence canonique pour EM, GMM, et la vue ELBO. Très détaillé sur les preuves.
- **Dempster, Laird, Rubin (1977).** *Maximum Likelihood from Incomplete Data via the EM Algorithm.* L'article fondateur.
- **Andrew Ng, CS229 notes on EM.** Présentation alternative très claire qui dérive l'ELBO via l'inégalité de Jensen.
- **MacKay, *Information Theory, Inference and Learning Algorithms*, chapitre 22.** Vue plus information-théorique, complémentaire à Bishop.
- **Murphy, *Probabilistic Machine Learning: Advanced Topics*, chapitre 6.** Vue moderne avec liens vers VI, VAE.
