---
title: Normalizing Flows
---
# Normalizing Flows

> Troisième famille génératrice. L'idée tient en une phrase : **on construit une bijection $f_\theta$ entre les données $x$ et un espace latent $z$ de même dimension**, avec un prior simple $p_Z$ (gaussienne, uniforme). Comme $f_\theta$ est inversible, on peut faire les deux trajets — coder $z = f_\theta^{-1}(x)$ et générer $x = f_\theta(z)$ — et surtout calculer la **densité exacte** $p_\theta(x)$ via la formule du changement de variable. C'est le meilleur des deux mondes vu dans les notes précédentes : on garde un **latent** (comme le `[[02_VAE|VAE]]`) **et** une **vraisemblance exacte** (comme les `[[01_Modèles autoregressifs|autorégressifs]]`), sans la borne ELBO ni le décodeur stochastique.

## I. Le projet : une bijection $X \leftrightarrow Z$

### A. Pourquoi les flows ?

Reprenons la grille des `[[00_Fondations#III. Les trois questions fondamentales]]`. On a vu deux compromis :

- **Autorégressifs** : densité $p_\theta(x) = \prod_i p_\theta(x_i \mid x_{<i})$ exacte et tractable, mais **pas de mécanisme d'apprentissage de features** (pas de latent), et génération séquentielle lente.
- **VAE** : un latent $z$ structuré et une génération rapide, mais **densité non exacte** — on n'a que la borne ELBO.

Le **normalizing flow** combine les forces : il apprend une représentation latente *et* donne la log-vraisemblance exacte. Le prix à payer (cf. §VI) : $f_\theta$ doit être **inversible** et **de jacobien calculable**, ce qui contraint fortement l'architecture, et **$x$ et $z$ doivent avoir la même dimension** (contrainte absente du VAE).

### B. La définition

On cherche un mapping $f_\theta : \mathbb{R}^n \to \mathbb{R}^n$ **déterministe et bijectif** tel que :

$$x = f_\theta(z), \qquad z = f_\theta^{-1}(x).$$

Le sens du mot "normalizing" : le flow $f_\theta^{-1}$ **normalise** la distribution complexe des données vers un prior simple (typiquement $\mathcal{N}(0, I)$). Le sens du mot "flow" : on compose plusieurs transformations simples $f = f_K \circ \cdots \circ f_1$ et la densité "coule" à travers chaque étape (cf. §V).

> [!todo] IMAGE — schéma $X \leftrightarrow Z$ en 2D
> *(la belle image : à gauche le Z-space en 2D, à droite l'ensemble des images, avec les flèches $f_\theta$ et $f_\theta^{-1}$ entre les deux)*
>
![[im2-1 (4).png]]
## II. Le changement de variable (le cœur)

Tout repose sur la formule de changement de variable des densités. C'est elle qui rend la vraisemblance exacte calculable.

### A. Cas 1-D (intuition)

Si $Y = f(X)$ avec $f$ inversible et $X = f^{-1}(Y)$, alors :

$$\boxed{\; p_Y(y) = p_X\big(f^{-1}(y)\big) \cdot \left| \frac{d f^{-1}}{dy}(y) \right| \;}$$

L'intuition est géométrique : une transformation **étire ou comprime l'espace**. Là où $f$ dilate l'espace, la densité doit **diminuer** (la même masse de probabilité s'étale sur plus de largeur) ; là où $f$ comprime, la densité **augmente**. Le facteur $|d f^{-1}/dy|$ est exactement le taux de dilatation local, et il garantit que $\int p_Y(y)\, dy = 1$.

> [!warning] Le piège qu'il faut avoir vu une fois
> Si on transforme $X \sim \mathcal{N}(1, 0.1)$ par $f(x) = x^2$ et qu'on oublie le facteur jacobien, l'intégrale de la "densité" obtenue ne vaut **pas** 1 (on trouve ~5.6). C'est le terme $|f^{-1\prime}(y)|$ qui ramène la masse totale à 1. **La densité ne se transporte pas comme une fonction ordinaire : il faut corriger par le taux de déformation local.**

### B. Cas n-D : le déterminant du jacobien

En dimension $n$, le "taux de dilatation local" d'une transformation est le **déterminant de sa matrice jacobienne**. La formule devient :

$$\boxed{\; p_X(x) = p_Z\big(f_\theta^{-1}(x)\big) \cdot \left| \det \frac{\partial f_\theta^{-1}(x)}{\partial x} \right| \;}$$

Le jacobien $J = \partial f^{-1}/\partial x$ est la matrice $n\times n$ des dérivées partielles ; son déterminant mesure le facteur par lequel un volume infinitésimal est dilaté/comprimé par la transformation. Par le **théorème de la fonction inverse**, on peut aussi écrire de façon équivalente :

$$p_X(x) = p_Z\big(f_\theta^{-1}(x)\big) \cdot \left| \det \frac{\partial f_\theta(z)}{\partial z} \right|^{-1}.$$

> [!note] Rappel — jacobien
> Pour $f : \mathbb{R}^n \to \mathbb{R}^n$, le jacobien empile toutes les dérivées partielles : ligne $j$ = sorties, colonne $k$ = entrées.
> $$\mathbf{J} = \begin{bmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_n/\partial x_1 & \cdots & \partial f_n/\partial x_n \end{bmatrix}$$
> Comme la dimension est préservée ($\mathbb{R}^n \to \mathbb{R}^n$), la matrice est **carrée** et le déterminant existe. C'est précisément pourquoi un flow exige $\dim(x) = \dim(z)$.

## III. La fonction objectif

On entraîne par **maximum de vraisemblance directe** (pas de borne, contrairement au VAE). En passant la formule du changement de variable au log :

$$\boxed{\; \max_\theta \; \sum_i \log p_\theta(x^{(i)}) = \max_\theta \; \sum_i \log p_Z\big(f_\theta^{-1}(x^{(i)})\big) + \log \left| \det \frac{\partial f_\theta^{-1}}{\partial x}(x^{(i)}) \right| \;}$$

Lecture des deux termes :

- **$\log p_Z(f_\theta^{-1}(x))$** : "une fois projeté dans l'espace latent, le point doit être probable sous le prior". Pousse $f^{-1}(x)$ vers le centre de la gaussienne.
- **$\log |\det J|$** : terme de volume. Empêche la solution dégénérée où $f$ écrase tout sur un point (ce qui maximiserait le premier terme mais collapserait l'espace).

> [!warning] Les deux conditions pour que ça marche
> Pour que cet objectif soit calculable en pratique, $f_\theta$ doit satisfaire **deux** propriétés simultanément :
> 1. **Inversible**, et l'inverse facile à évaluer (pour le calcul de $f^{-1}(x)$ à l'entraînement, et pour générer à l'inférence).
> 2. **Déterminant du jacobien facile à calculer**. Un déterminant $n\times n$ générique coûte $O(n^3)$ — rédhibitoire. Toute l'ingénierie des flows (§IV) consiste à concevoir des $f$ dont le jacobien est **triangulaire**, donc de déterminant = produit des diagonales en $O(n)$.

## IV. Construire un flow : les briques

Toute l'astuce est de trouver des transformations inversibles **à jacobien triangulaire**. Deux grandes familles.

### A. Exemple 1-D pour fixer les idées : fit d'une bimodale

Le cas le plus intuitif. Données 1-D = mélange de gaussiennes dont on ne connaît que des échantillons. On choisit comme flow $f_\theta = F_X$ = la **CDF d'un mélange de gaussiennes** (à 4 ou 5 composantes). Pourquoi la CDF ? Parce qu'une CDF mappe $\mathbb{R} \to [0,1]$ — exactement le support d'un prior **uniforme** $p_Z = \mathcal{U}(0,1)$.

Le terme jacobien se calcule tout seul ici, car la dérivée d'une CDF **est** la densité :

$$\frac{\partial F_X(x)}{\partial x} = f_X(x) \quad (\text{la pdf du mélange}).$$

Donc la loss se réduit à :

$$\max_\theta \; \sum_i \log p_Z\big(F_X(x^{(i)})\big) + \log f_X(x^{(i)}).$$

En entraînant, $f_\theta$ apprend à "redresser" la bimodale : les $x$ projetés deviennent uniformes sur $[0,1]$. On peut remplacer le prior uniforme par $\mathcal{B}(5,5)$ (même support $[0,1]$) sans rien changer d'autre.

> [!todo] IMAGE — fit bimodale 1-D
> *(les 3 panneaux : à gauche les $x$, au milieu la CDF du mélange, à droite l'histogramme des $z$ uniformes une fois le flow appris ; + éventuellement la courbe de la pdf apprise)*
> `![[...]]`
![[images/3-Apprentissage automatique/Generative Models/flow models/im1-3.png]]
### B. Flows autorégressifs

On garde l'idée de la chain rule des `[[01_Modèles autoregressifs|autorégressifs]]`, mais chaque composante devient une transformation inversible. En 2-D :

$$z_1 = f_\theta(x_1), \qquad z_2 = f_\phi(x_1, x_2).$$

Chaque $z_i$ ne dépend que des $x_{\le i}$. Conséquence directe : le jacobien $\partial z / \partial x$ est **triangulaire** (pas de dépendance "vers le futur"), donc son déterminant est le produit des diagonales. C'est ce qui rend la vraisemblance tractable.

L'inversion se fait composante par composante : connaissant $z_1$ on récupère $x_1$, puis connaissant $z_2$ et $x_1$ on récupère $x_2$, etc. **Génération séquentielle** (lente), mais évaluation de densité parallèle.

> [!note] MAF vs IAF — le compromis vitesse
> Selon qu'on conditionne sur les $x$ ou sur les $z$, on échange la vitesse entre les deux sens :
> - **MAF** (Masked Autoregressive Flow) : densité rapide (1 passe), génération lente (séquentielle). Bon pour l'entraînement / l'estimation de densité.
> - **IAF** (Inverse Autoregressive Flow) : génération rapide, densité lente. Bon quand on veut échantillonner beaucoup (utilisé dans les VAE à postérieur flexible — cf. note VAE).
>
> C'est le même $x_i \mid x_{<i}$ qu'en AR, mais transposé en transformation inversible.

> [!todo] IMAGE — flow autorégressif 2-D
> *(tes belles images du 2D flow — la shape du dataset, puis le résultat après mapping)*
> `![[...]]`

### C. Coupling layers : RealNVP

Le problème des flows autorégressifs : génération séquentielle lente. **RealNVP** résout ça avec une couche de couplage affine qui s'inverse **en une seule passe**.

On coupe le vecteur en deux moitiés. Une moitié passe **inchangée** ; elle sert à calculer une transformation **affine** appliquée à l'autre moitié :

$$\begin{aligned} z_{1:d/2} &= x_{1:d/2} \quad (\text{copiée telle quelle}) \\ z_{d/2:d} &= x_{d/2:d} \cdot s_\theta(x_{1:d/2}) + t_\theta(x_{1:d/2}) \end{aligned}$$

où $s_\theta$ (scale) et $t_\theta$ (translation) sont des **MLP arbitraires** prenant la première moitié en entrée. Pour garantir $s > 0$ (inversibilité), on pose $s_\theta(\cdot) = \exp\{\text{MLP}(\cdot)\}$.

**Pourquoi c'est génial** — le jacobien est triangulaire par construction :

$$\frac{\partial z}{\partial x} = \begin{bmatrix} I & 0 \\ \dfrac{\partial z_{d/2:d}}{\partial x_{1:d/2}} & \operatorname{diag}\big(s_\theta(x_{1:d/2})\big) \end{bmatrix} \;\Longrightarrow\; \det \frac{\partial z}{\partial x} = \prod_{k} s_\theta(x_{1:d/2})_k.$$

Le bloc copié donne l'identité $I$, le bloc transformé donne une diagonale. Déterminant = produit des scales, en $O(n)$. Et l'inversion est triviale : $x_{d/2:d} = (z_{d/2:d} - t_\theta) / s_\theta$ — **une seule passe, pas de séquentialité**, parce que $s_\theta, t_\theta$ ne dépendent que de la moitié copiée qu'on connaît déjà.

> [!warning] Le masking : ne jamais laisser une moitié figée
> Si on ne transformait qu'une moitié, l'autre ne serait jamais modélisée. On **alterne les masques** : couche 1 transforme la 2ᵉ moitié à partir de la 1ʳᵉ, couche 2 transforme la 1ʳᵉ à partir de la 2ᵉ (mask `right` puis `left`). En empilant ces couplages alternés, toutes les dimensions finissent transformées. Pour les images, on utilise un **masque en damier (checkerboard)** plutôt qu'une coupe en deux blocs.

> [!todo] IMAGE — coupling layer RealNVP + checkerboard
> *(tes schémas : la coupe en moitiés / le MLP qui sort log_s et t / le forward, et l'image du damier checkerboard)*
> `![[...]]`

## V. Composition de flows

Une seule transformation simple ne suffit pas à modéliser une distribution complexe. On en **compose plusieurs** :

$$f = f_K \circ f_{K-1} \circ \cdots \circ f_1.$$

La densité se propage à travers la chaîne, et grâce à la propriété multiplicative du déterminant, le log-jacobien total est simplement la **somme** des log-jacobiens de chaque couche :

$$\log \left| \det \frac{\partial f}{\partial x} \right| = \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial h_{k-1}} \right|.$$

C'est l'analogue, côté densité, de l'empilement de couches dans un réseau profond : chaque couche déforme un peu l'espace, et la composition produit une transformation arbitrairement expressive tout en gardant un jacobien tractable couche par couche.

> [!todo] IMAGE — composition de flows (logit + mixture, etc.)
> `![[...]]`

## VI. Passage à la grande dimension (images)

Sur une image $32 \times 32$, l'approche naïve (un MLP par flow conditionnel $z_i = f(x_i \mid x_{<i})$) demanderait 1024 réseaux séparés — impossible à entraîner.

La solution est la même que pour les autorégressifs : **partage de paramètres par masking**. Un **seul** réseau, type **PixelCNN masqué**, calcule simultanément tous les flows en respectant l'ordre des variables. RealNVP avec masques en damier + factorisation multi-échelle est l'instanciation standard pour les images.

> [!todo] IMAGE — schémas PGM + réseau de neurones
> *(les deux belles images : le normalizing flow vu comme modèle graphique probabiliste, et le schéma du réseau)*
>
![[flow2.png]]
## VII. Génération et inférence

> [!warning] Anatomie d'un flow
> - **Estimation de densité (sens $x \to z$).** On passe $x$ dans $f_\theta^{-1}$, on obtient $z = f_\theta^{-1}(x)$, on évalue $p_Z(z)$ + somme des log-jacobiens. Donne $\log p_\theta(x)$ **exact**.
> - **Génération (sens $z \to x$).** On tire $z \sim p_Z = \mathcal{N}(0, I)$, on passe dans $f_\theta$, on obtient $x = f_\theta(z)$. **Déterministe** — pas de tirage stochastique en sortie, contrairement au décodeur du VAE.
>
> Les deux sens utilisent le **même** réseau, parcouru à l'endroit ou à l'envers. C'est la signature des flows.

## VIII. Positionnement

> [!warning] Bilan Normalizing Flows
> **Forces :**
> - **Densité $p_\theta(x)$ exacte** — pas de borne ELBO. Idéal pour estimation de densité, détection d'anomalies, compression.
> - **Latent $z$ bijectif** — interpolation, manipulation, et correspondance $x \leftrightarrow z$ exacte (pas d'approximation comme l'encodeur du VAE).
> - **Entraînement par MLE direct**, stable.
>
> **Faiblesses :**
> - **Contrainte $\dim(z) = \dim(x)$** — pas de compression dimensionnelle, coûteux en mémoire pour les images haute résolution.
> - **Architecture contrainte** — $f$ doit être inversible à jacobien tractable, ce qui limite l'expressivité par couche (d'où le besoin d'empiler beaucoup de couches).
> - **Qualité visuelle** en retrait des GAN / diffusion sur les images naturelles, à budget de calcul comparable.

### Tableau récapitulatif

| Aspect | Autorégressif | VAE | **Flow** | GAN | Diffusion |
|---|---|---|---|---|---|
| Structure | Chain rule | Latent + décodeur | **Bijection $z \leftrightarrow x$** | Latent + générateur | Markov de débruitage |
| Densité $p(x)$ | Exacte | Borne (ELBO) | **Exacte** | Inaccessible | Approchée |
| Apprentissage | MLE direct | ELBO (variational) | **MLE direct** | Adversarial | Score matching |
| Échantillonnage | Lent (séquentiel) | Rapide (1 forward) | **Rapide\*** | Rapide | Itératif (lent) |
| Espace latent | Non | Oui, structuré | **Oui, bijectif** | Oui, non-régulier | Implicite |
| Contrainte dim | — | $\dim z \ll \dim x$ | **$\dim z = \dim x$** | — | — |

\*rapide pour les coupling flows (RealNVP) ; lent pour les flows autorégressifs en génération (MAF).

## IX. Extensions

> [!todo] À étoffer ultérieurement
> - **Continuous Normalizing Flows / Neural ODE** : remplacer la composition discrète $f_K \circ \cdots \circ f_1$ par une **EDO** continue ; le log-déterminant devient une intégrale de la trace du jacobien.
> - **Glow** (Kingma & Dhariwal, 2018) : RealNVP + convolutions $1\times1$ inversibles, génération d'images haute résolution.
> - **Dequantization** : traiter proprement les données discrètes (pixels 8-bit) avec un modèle de densité continue.
> - **Flows comme postérieur de VAE** (IAF) : rendre $q_\phi(z\mid x)$ plus flexible qu'une gaussienne diagonale — lien direct avec la note `[[02_VAE]]`.

---

## Pour aller plus loin

- **RealNVP.** L. Dinh, J. Sohl-Dickstein, S. Bengio. *Density estimation using Real NVP.* ICLR 2017. arXiv:1605.08803.
- **NICE** (le prédécesseur). L. Dinh et al. *NICE: Non-linear Independent Components Estimation.* 2014. arXiv:1410.8516.
- **MAF.** G. Papamakarios et al. *Masked Autoregressive Flow for Density Estimation.* NeurIPS 2017.
- **Glow.** D.P. Kingma, P. Dhariwal. *Glow: Generative Flow with Invertible 1×1 Convolutions.* NeurIPS 2018.
- **Survey.** Papamakarios et al. *Normalizing Flows for Probabilistic Modeling and Inference.* JMLR 2021.
- **Blog.** Lilian Weng, *Flow-based Deep Generative Models* — excellente synthèse visuelle.
