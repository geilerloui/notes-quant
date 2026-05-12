---
title: Applications du transport optimal
date: 2026-05-11
tags: [mathématiques, transport-optimal, applications, machine-learning, finance, vision]
---

## L'idée fondatrice

Les notes précédentes ont construit la **théorie** du transport optimal : Monge, Kantorovich, Brenier, Wasserstein, dualité, Sinkhorn. Cette note change de registre : on regarde **où et comment** OT est utilisé en pratique.

Le transport optimal est un outil **transversal**, utilisé dans plusieurs domaines :

- **Machine learning** : generative models (WGAN, diffusion, flow matching), domain adaptation, fairness, neural OT
- **Finance quantitative** : distributionally robust optimization, model risk, calibration, robust portfolio
- **Statistiques** : tests à deux échantillons, mesure de similarité, two-sample testing
- **Vision par ordinateur** : color transfer, image morphing, shape matching, registration
- **Sciences appliquées** : mécanique des fluides (Benamou-Brenier), météo, biologie computationnelle

Cette note couvre les applications **principales** dans les trois premiers domaines, avec une emphase particulière sur la finance (où OT a explosé depuis 2015) et le ML moderne. Le point commun : à chaque fois, c'est **Sinkhorn** (note 05) qui fait tourner la machinerie en pratique.

## 0. Quand a-t-on besoin de $P$ vs juste $W_p$ ?

Avant de plonger dans les applications, il faut clarifier **une distinction fondamentale** qui structure tout ce qui suit. En sortie d'un solveur de transport optimal, on peut récupérer deux choses (cf. note 03) :

- **Le plan de transport $P$** (le couplage optimal $\gamma^*$) : "qui va où"
- **Le scalaire $W_p$** (la valeur du coût optimal) : "à quel point c'est loin"

**Selon l'application, on veut l'un, l'autre, ou les deux.** Et c'est ce qui décide de **quel outil** utiliser.

### Cas A : On veut juste $W_p$ (scalaire)

Quand on a besoin d'une **mesure de distance différentiable** entre distributions, sans s'intéresser à où va chaque point. C'est typique des applications "loss function" en ML :

| Application | Pourquoi on veut juste $W_p$ |
| :--- | :--- |
| **WGAN** | Loss pour entraîner un générateur ; on dérive par rapport aux paramètres $\theta$ |
| **Calibration de modèles** | Minimiser une "distance" entre distribution prédite et distribution cible |
| **Two-sample testing** | "Ces deux datasets viennent-ils de la même distribution ?" → on compare $W_p$ à un seuil |
| **DRO Wasserstein** | Définir une boule de rayon $\varepsilon$ autour d'une distribution nominale |
| **Wasserstein autoencoders** | Régularisation du latent space |

**Outil typique** : la **dualité de Kantorovich** (note 04). Pour $W_1$, on cherche **une seule fonction** $\varphi$ 1-Lipschitz via Kantorovich-Rubinstein. C'est ce qu'utilise WGAN. Pas besoin du plan, on ne le forme jamais.

### Cas B : On veut le plan $P$ (couplage)

Quand on a besoin de **savoir où va chaque point** : la question scientifique est "qui correspond à quoi". Là, $W_p$ tout seul ne sert à rien :

| Application | Pourquoi on a besoin de $P$ |
| :--- | :--- |
| **Domain adaptation** | Transférer les labels du source vers le target : pour chaque point cible $y_j$, on doit savoir de quel point source $x_i$ il "vient" (lecture de $P_{ij}$) |
| **Color transfer** | Pour chaque pixel de couleur $c_A$ dans l'image, on doit savoir vers quelle couleur $c_B$ il est envoyé |
| **Image morphing** | Pour interpoler entre $A$ et $B$, il faut savoir où chaque pixel de $A$ atterrit dans $B$ (map de Brenier) |
| **Shape matching, registration** | Apparier des points d'un nuage à l'autre (recalage IRM, single-cell biology) |
| **Point cloud matching** | Quelle cellule à $t = 0$ correspond à quelle cellule à $t = 1$ |
| **Wasserstein barycenters** | Indirectement : il faut les plans entre $\mu_k$ et le barycentre courant à chaque itération |

**Outil typique** : **Sinkhorn** (note 05). Il calcule $P^*_\varepsilon$ (un plan régularisé) en $O(n^2)$ par itération, est différentiable, et donne le plan en sortie. C'est devenu le workhorse de l'OT computationnelle en ML depuis 2013.

**Important** : pour ces applications, la dualité K-R **ne suffit pas**. Elle te donnerait juste "les deux distributions sont à distance 3.7" — totalement inutile pour transférer des labels ou interpoler une image. Il faut vraiment former le plan $P$.

### Cas C : On veut les deux (rare mais existe)

Dans certains cas, on s'intéresse aux deux simultanément :

| Application | Pourquoi les deux |
| :--- | :--- |
| **Sinkhorn divergences comme loss** (Genevay-Cuturi 2018) | On utilise la valeur du coût Sinkhorn comme loss différentiable (cas A), mais $P^*_\varepsilon$ est calculé en passant (cas B). Alternative à WGAN sans réseau critique |
| **Analyse exploratoire** | On veut quantifier la dissimilarité ($W_p$) ET comprendre la correspondance ($P$) |

### Récap : quel outil pour quel besoin

| Besoin | Outil | Coût | Output |
| :--- | :--- | :--- | :--- |
| Juste $W_1$ (scalaire) différentiable | Dualité K-R (note 04) | $O(n)$ par évaluation de $\varphi$ | $W_1$ + fonction $\varphi$ |
| Le plan $P$ régularisé | Sinkhorn (note 05) | $O(n^2)$ par itération | $P^*_\varepsilon$ + coût ≈ $W_p^p$ |
| Le plan $P$ exact | Simplex LP / hongrois | $O(n^3)$ | $P^*$ + coût $W_p^p$ |
| $W_2$ entre gaussiennes | Formule fermée (Bures) | $O(d^3)$ ($d$ = dimension) | $W_2$ + map $T$ affine |

> [!note]- Pourquoi cette confusion est légitime quand on découvre OT par WGAN
> Beaucoup de gens découvrent OT à travers WGAN, qui est l'application "phare" en ML. Et WGAN n'a **besoin que du scalaire $W_1$** : la dualité K-R suffit, on ne forme jamais $P$. D'où la question naturelle "à quoi sert Sinkhorn alors ?".
> 
> La réponse : **WGAN n'est qu'une application parmi d'autres**. Toutes les applis qui demandent une **correspondance point à point** (color transfer, domain adaptation, morphing, biology) ont besoin du plan $P$, et là Sinkhorn devient indispensable. La diversité des applis OT en ML va bien au-delà du seul WGAN.

Avec cette grille en tête, on peut maintenant lire les applications spécifiques.

## I. Wasserstein barycenters

### Le problème

Étant donné $N$ distributions $\mu_1, \dots, \mu_N$, comment définir leur **moyenne** ?

L'approche naïve consiste à prendre la moyenne arithmétique $\bar{\mu} = \frac{1}{N} \sum_k \mu_k$. Mais ce **mélange** ne préserve pas la structure des distributions : si tous les $\mu_k$ sont des gaussiennes (mode unique), $\bar{\mu}$ est une **mixture** multimodale qui ne ressemble plus à une gaussienne.

L'**alternative Wasserstein** : on définit le **barycenter** comme la distribution qui minimise la somme des distances de Wasserstein au carré :

$$\boxed{\bar{\mu}_W = \arg\min_\nu \sum_{k=1}^N \lambda_k \, W_2^2(\nu, \mu_k)}$$

(les $\lambda_k \geq 0$ avec $\sum \lambda_k = 1$ sont des poids)

### Exemple : barycentre de deux gaussiennes

Pour $\mu_1 = \mathcal{N}(m_1, \Sigma_1)$ et $\mu_2 = \mathcal{N}(m_2, \Sigma_2)$, le barycentre Wasserstein pour $\lambda = (1/2, 1/2)$ est :

$$\bar{\mu}_W = \mathcal{N}\left( \frac{m_1 + m_2}{2}, \, \bar{\Sigma} \right)$$

où $\bar{\Sigma}$ est la **moyenne géodésique** des covariances (pas la moyenne arithmétique).

Contraste avec le mélange $\bar{\mu} = \frac{1}{2}(\mu_1 + \mu_2)$ : celui-ci est **bimodal**, alors que $\bar{\mu}_W$ est une **vraie gaussienne** unimodale. Le barycentre Wasserstein **préserve la forme**.

### Calcul : Sinkhorn itéré

Le calcul exact est difficile, mais Sinkhorn permet une approximation efficace. L'algorithme [Cuturi-Doucet, 2014] alterne :

1. Pour chaque $\mu_k$, calculer le couplage Sinkhorn entre $\bar{\mu}^{(t)}$ et $\mu_k$
2. Mettre à jour $\bar{\mu}^{(t+1)}$ comme la moyenne barycentrique des push-forwards

### Application phare : color transfer

**Le problème** : transférer la palette de couleurs d'une image source vers une image cible (et inversement).

- L'image source $A$ a une distribution de couleurs $\mu_A$ (histogramme dans l'espace RGB ou Lab)
- L'image cible $B$ a une distribution $\mu_B$
- On veut une image $A'$ avec le contenu de $A$ mais la palette de $B$

**Solution OT** : on calcule la map optimale $T : \mu_A \to \mu_B$ (note 02 : map de Brenier ou approximation Sinkhorn), puis on remplace chaque pixel $c \in A$ par $T(c)$.

Résultat visuel souvent **spectaculaire** : la photo prise au soleil de midi devient celle d'un coucher de soleil, etc. C'est l'application "grand public" la plus célèbre d'OT.

### Application en ML : neural barycenters

Les barycenters apparaissent dans plusieurs contextes ML :
- **Federated learning** : moyenner les distributions de plusieurs clients tout en préservant la structure
- **Multi-task learning** : combiner des représentations issues de tâches différentes
- **Fairness** : trouver une distribution qui interpole entre plusieurs groupes démographiques (de Lara et al., 2021)

## II. Domain adaptation

### Le problème

En ML supervisé classique, on suppose que les données d'entraînement et de test viennent de la **même distribution**. Mais en pratique :

- Le modèle est entraîné sur **dataset A** (par ex. photos de jour)
- Le modèle est déployé sur **dataset B** (par ex. photos de nuit, ou d'une autre caméra)

Le modèle **dégrade** car les distributions diffèrent : c'est le problème de **distribution shift**.

**Domain adaptation** : on a accès aux données source $(X_s, Y_s)$ avec labels, et aux données target $X_t$ **sans labels**. On veut entraîner un modèle qui marche bien sur $X_t$.

### Solution OT : Courty-Flamary 2017

L'idée centrale : trouver une transformation $T$ qui **aligne** la distribution source $\mu_s$ sur la distribution target $\mu_t$, en utilisant le transport optimal.

L'algorithme :

1. Calculer la map de transport $T$ entre $\mu_s$ et $\mu_t$ (via Sinkhorn pour la vitesse)
2. **Transporter** les données source : $\tilde{X}_s = T(X_s)$
3. Entraîner le modèle sur $(\tilde{X}_s, Y_s)$ — on conserve les labels mais les features sont dans le domaine cible

**Pourquoi c'est élégant** : OT donne une **correspondance point à point** entre les deux distributions, qui respecte la géométrie. Contraste avec d'autres méthodes (adversarial alignment, MMD) qui alignent statistiquement sans correspondance explicite.

### Variantes utilisées en pratique

- **OT avec régularisation Laplacienne** : impose que les points proches dans le source soient transportés vers des points proches dans le target (préserve la structure locale)
- **OT avec régularisation par classe** : encourage le transport à respecter les labels (les points de la classe $k$ vont vers des régions probables de la classe $k$)
- **Sinkhorn supplément** : approximation rapide pour passer à l'échelle

OT-domain-adaptation est devenu un benchmark standard, implémenté dans des librairies comme [POT](https://pythonot.github.io/) (Python Optimal Transport).

## III. Finance : DRO et model risk

C'est probablement l'**application la plus impactante d'OT en finance**, avec une littérature en plein essor depuis 2015.

### Le problème : robustesse aux mauvaises distributions

En finance, on travaille toujours avec des **distributions estimées** (returns historiques, distributions calibrées par MLE, distributions implicites par les prix d'options). Ces distributions sont **incertaines** :

- Les returns futurs ne suivront pas exactement la même loi que les returns passés
- Les paramètres sont estimés avec bruit
- Les modèles sont mal-spécifiés

On veut prendre des décisions qui sont **robustes** à cette incertitude.

### Formulation DRO via Wasserstein

**Distributionally Robust Optimization** (DRO) : on minimise sur le **worst case** parmi toutes les distributions "proches" de la nominale.

$$\boxed{\min_\theta \, \sup_{\nu : W_p(\nu, \mu_0) \leq \varepsilon} \, \mathbb{E}_{\nu}[L(\theta, X)]}$$

où :
- $\mu_0$ est la distribution nominale (estimée)
- $\nu$ varie dans une **boule de Wasserstein** de rayon $\varepsilon$ autour de $\mu_0$
- $L(\theta, X)$ est une fonction de perte (par exemple : perte du portefeuille, erreur de prédiction)

**Interprétation économique** : on suppose qu'un "adversaire" peut perturber la vraie distribution dans la limite d'un budget $\varepsilon$ (mesuré en Wasserstein). On veut être robuste à cette perturbation.

### Pourquoi Wasserstein plutôt que KL ?

C'est exactement la raison de la note 03 :

- **KL-DRO** demande que $\nu$ soit absolument continue par rapport à $\mu_0$ (sinon $\text{KL} = \infty$). Trop restrictif : exclut les perturbations physiquement plausibles.
- **Wasserstein-DRO** autorise des $\nu$ à support disjoint de $\mu_0$. On peut perturber les returns historiques par des scenarios extrêmes jamais observés. **C'est ce qu'on veut pour le model risk**.

### Le résultat magique : dualité finie-dimensionnelle

[Mohajerin Esfahani-Kuhn, 2018] et [Blanchet-Murthy, 2019] montrent que le problème DRO Wasserstein (qui est en théorie infini-dimensionnel) admet une **reformulation duale finie-dimensionnelle** :

$$\sup_{W_p(\nu, \mu_0) \leq \varepsilon} \mathbb{E}_\nu[L] = \inf_{\lambda \geq 0} \left\{ \lambda \varepsilon^p + \mathbb{E}_{\mu_0} \left[ \sup_z \left\{ L(z) - \lambda \|z - X\|^p \right\} \right] \right\}$$

Le sup intérieur est une optimisation **point par point** (sur chaque échantillon $X \sim \mu_0$), facile à calculer. Le problème original devient une optimisation 1D sur $\lambda$ + des perturbations locales sur les échantillons.

**Conséquence pratique** : Wasserstein-DRO se résout aussi facilement qu'un problème classique d'optimisation stochastique. C'est ce qui a fait son adoption massive.

### Applications concrètes en finance

- **Portfolio optimization robuste** : la solution de Markowitz est notoirement instable (sensibilité aux paramètres). Wasserstein-DRO produit des portefeuilles stables avec garanties théoriques (Blanchet-Chen-Zhou 2022).
- **Pricing robuste d'options** : worst-case price sous une famille de modèles à distance Wasserstein bornée du modèle calibré (Lütkebohmert-Müller-Schmeck 2022).
- **Risk measures** : CVaR robuste, VaR robuste, sans hypothèse paramétrique forte.
- **Calibration de modèles** : trouver les paramètres qui minimisent le worst-case error sur une boule Wasserstein autour de la distribution de marché.
- **Adversarial training en credit risk** : entraîner des modèles de scoring qui résistent aux perturbations distributionnelles (fraud, market regime change).

### Lien avec ridge regression et regularization

[Blanchet-Kang-Murthy, 2019] : Wasserstein-DRO de la régression linéaire **est exactement équivalent** au LASSO (ou au ridge selon la norme choisie). C'est-à-dire que le rayon $\varepsilon$ de la boule Wasserstein joue le rôle du paramètre de régularisation.

Ce résultat **unifie** des approches qui semblaient sans lien :
- **Robustesse** (DRO)
- **Régularisation** (ridge, LASSO)
- **Sparse estimation**

C'est l'un des résultats les plus profonds du domaine.

## IV. Generative models (mention)

On a déjà discuté de WGAN en note 04. Mentionnons brièvement les **trois grandes familles** de generative models qui utilisent OT :

### WGAN et variantes

- **WGAN** [Arjovsky et al., 2017] : critère $W_1$ via Kantorovich-Rubinstein, critique 1-Lipschitz approximée par weight clipping
- **WGAN-GP** [Gulrajani et al., 2017] : gradient penalty pour imposer 1-Lipschitz proprement
- **Sinkhorn divergence GAN** : utilise directement $S_\varepsilon$ comme loss

### Normalizing flows et OT-Flow

[Onken et al., 2021] propose **OT-Flow** : un normalizing flow où le transport est explicitement régularisé pour être un transport optimal. Combine la souplesse des flows avec la rigueur géométrique d'OT.

### Diffusion models et flow matching

C'est l'**état de l'art actuel** (2023-2025) :
- Les diffusion models [Ho et al., 2020 ; Song et al., 2021] sont **profondément liés à OT** via la formulation Schrödinger bridge
- **Flow matching** [Lipman et al., 2023] et **Conditional Flow Matching** [Tong et al., 2024] sont des reformulations qui utilisent les **géodésiques de Wasserstein** (note 07) comme cible d'entraînement
- Les liens deviennent rigoureux via la dynamique de Benamou-Brenier (note 07)

C'est le sujet où la **théorie d'OT a le plus d'impact pratique aujourd'hui**.

## V. Vision par ordinateur

OT a une longue histoire en vision (avant son explosion en ML). Les applications phares :

### Color transfer et style transfer

Déjà mentionné en section I. Algorithme : on a deux histogrammes 3D (RGB ou Lab), on calcule le transport optimal entre eux, on applique le transport pixel par pixel.

### Image morphing

**Le problème** : créer une animation lisse entre deux images $A$ et $B$ (par exemple : morphing visage A → visage B).

**Solution OT** : si on voit les images comme des distributions (de pixels, de features extraites par un réseau), l'**interpolation de McCann** (note 03)

$$\mu_t = ((1-t)\, \text{Id} + t T)_\# \mu_A$$

donne une trajectoire qui **préserve la masse** et déforme continûment $A$ en $B$. Bien plus réaliste qu'une interpolation linéaire $(1-t) A + t B$ (qui produit du fade-in / fade-out).

### Shape matching et registration

**Le problème** : aligner deux formes 3D (par ex. deux scans IRM du même patient à deux dates) ou deux nuages de points.

**Solution OT** : on calcule le transport optimal entre les deux nuages de points. La map $T$ donne la **correspondance point à point**, qui sert de base à l'alignement (Procrustes + OT).

Applications médicales : suivi de tumeurs, recalage d'imagerie multi-modale, statistiques de formes anatomiques.

### Optical flow et tracking

Le transport optimal entre deux frames consécutives d'une vidéo donne une estimation du **flow optique** (déplacement des pixels d'une frame à l'autre). C'est une approche alternative aux méthodes variationnelles classiques.

## VI. Trois idées à retenir

1. **OT n'est pas qu'un outil ML**. Ses applications principales sont les **generative models** (WGAN, diffusion, flow matching), la **finance quantitative** (DRO, model risk, portfolio robuste), la **vision** (color transfer, morphing, registration), et la **statistique non paramétrique** (tests à deux échantillons).

2. **En finance, Wasserstein-DRO est devenu l'outil standard pour la robustesse**. La reformulation duale finie-dimensionnelle [Mohajerin Esfahani-Kuhn, 2018] permet de résoudre efficacement. Lien profond : DRO Wasserstein de la régression = LASSO/ridge. C'est le pont entre **robustesse** et **régularisation**.

3. **En ML moderne, OT est au cœur des generative models de pointe** (flow matching, diffusion via Schrödinger bridge). La géométrie de Wasserstein (note 07) donne la bonne notion de "déformation continue" entre distributions, ce qui est précisément ce qu'apprennent les flow models.

## VII. Vers la suite

- **Note 07 — Géométrie de Wasserstein** : la structure riemannienne de $\mathcal{P}_2$ qui sous-tend la plupart des applications avancées (flow matching, diffusion, JKO scheme). C'est le pont avec la géométrie différentielle.

### Pour aller plus loin

- **Computational Optimal Transport** [Peyré-Cuturi, 2019] : la référence pour tous les aspects algorithmiques et applications ML
- **Optimal Transport: Old and New** [Villani, 2008] : la bible théorique (mais ardue)
- **Optimal Transport Methods in Economics** [Galichon, 2018] : applications en économie, matching markets, IO
- **Distributionally Robust Optimization** [Rahimian-Mehrotra, 2019] : survey du DRO Wasserstein
