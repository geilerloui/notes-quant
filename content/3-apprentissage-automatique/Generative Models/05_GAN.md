---
title: Generative Adversarial Networks (GAN)
---
# Generative Adversarial Networks (GAN)

> Troisième famille concrète. Les GAN font un saut conceptuel radical par rapport aux autorégressifs, VAE et flows : **ils abandonnent complètement la vraisemblance** comme objectif d'apprentissage. Au lieu de maximiser $\log p_\theta(x)$ ou une borne dessus, ils reformulent la génération comme un *jeu adversariel* entre deux réseaux. C'est cette rupture qui leur a donné — pendant longtemps — la meilleure qualité d'image, au prix d'un entraînement notoirement instable. Aujourd'hui largement remplacés par les diffusion models pour la génération d'images, ils restent essentiels conceptuellement et utilisés dans plein de contextes hybrides (super-résolution, style transfer, image-to-image).

## I. Le saut conceptuel : abandonner la vraisemblance

### A. Récap des familles likelihood-based

Toutes les familles vues jusqu'ici partagent un même squelette d'apprentissage : on définit une densité $p_\theta(x)$, on en évalue la log-vraisemblance sur les données, on maximise.

| Famille       | Densité                                        | Apprentissage |     |     |
| ------------- | ---------------------------------------------- | ------------- | --- | --- |
| Autorégressif | $\prod_i p_\theta(x_i \mid x_{<i})$, exacte    | MLE direct    |     |     |
| VAE           | $\int p_\theta(x \mid z) p(z) dz$, intractable | Borne ELBO    |     |     |
| Flow          | $p_Z(f^{-1}_\theta(x)) \det J$, exacte         | MLE direct    |     |     |

Le principe est toujours le même : la KL entre $p_{\text{data}}$ et $p_\theta$ se ramène (cf. `[[00_Fondations#V bis. Apprentissage par maximum de vraisemblance]]`) à la maximisation de $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]$.

**Question naturelle :** est-ce que maximiser la log-vraisemblance produit toujours des échantillons de bonne qualité ? On va voir que non.

### B. Vraisemblance ≠ qualité des échantillons

Cette question a été tranchée par [Theis, van den Oord & Bethge (2016)](https://arxiv.org/abs/1511.01844). Leur conclusion : **dans un modèle mal spécifié ou imparfaitement optimisé, log-vraisemblance et qualité des échantillons peuvent diverger arbitrairement**. Trois contre-exemples le montrent.

#### Cas 1 — Modèle bruité à haute vraisemblance

Considérons un mélange à 99% bruit et 1% vraies données :

$$p_\theta(x) = 0.01 \, p_{\text{data}}(x) + 0.99 \, p_{\text{noise}}(x).$$

**Qualité des échantillons** : catastrophique. 99 % des tirages produisent du bruit pur.

**Log-vraisemblance** : étonnamment élevée. On peut l'encadrer :

$$\log p_\theta(x) \;\geq\; \log\!\big(0.01 \, p_{\text{data}}(x)\big) \;=\; \log p_{\text{data}}(x) - \log 100.$$

Donc :

$$\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] \;\geq\; \mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)] - \log 100.$$

Et par non-négativité de la KL, on a aussi $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] \leq \mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)]$.

> [!warning] L'argument dimensionnel
> En haute dimension, $|\log p_{\text{data}}(x)|$ croît proportionnellement à $n$ (la dimension de $x$), alors que $\log 100$ reste constant. Donc :
> 
> $$\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] \approx \mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)].$$
> 
> Un modèle qui génère du bruit à 99 % a une log-vraisemblance **quasi-optimale**. La log-vraisemblance est un mauvais proxy de la qualité visuelle.

#### Cas 2 — Échantillons parfaits, log-vraisemblance nulle

Symétrique : on mémorise le dataset d'entraînement et on échantillonne uniformément depuis lui. Les échantillons sont parfaits (ce sont littéralement des vrais échantillons). Mais sur un dataset de test disjoint, la log-vraisemblance est $-\infty$ (probabilité nulle accordée à tout point hors mémorisé).

#### Cas 3 — Découpler les deux

> [!warning] Likelihood-free learning
> Les exemples ci-dessus suggèrent qu'optimiser la log-vraisemblance est un **proxy lâche** de l'objectif réel (générer des échantillons réalistes). Plutôt que de continuer à raffiner la borne sur $\log p_\theta(x)$, peut-on définir un critère d'apprentissage qui *cible directement* la qualité des échantillons ?
> 
> C'est exactement ce que font les GAN : **on n'apprend plus une densité, on apprend à produire des échantillons qui passent un test statistique**. Cette approche s'appelle *likelihood-free learning*.

## II. Comparer deux distributions par leurs échantillons

### A. Le problème : two-sample test

Étant donné deux ensembles d'échantillons $S_1 = \{x \sim P\}$ et $S_2 = \{x \sim Q\}$, peut-on tester si $P = Q$ sans connaître ni $P$ ni $Q$ ?

![[vanilla1.png|359]]

C'est un problème classique de statistique : le **two-sample test**. On définit une statistique $T(S_1, S_2)$ qui mesure une "distance" entre les deux jeux d'échantillons, et on accepte $H_0 : P = Q$ si $T < \alpha$.

> [!note] Observation clé
> La statistique $T$ ne nécessite **jamais d'évaluer les densités** $P$ ou $Q$ — seulement de comparer les échantillons. C'est exactement l'ingrédient *likelihood-free* qu'on cherchait.

### B. L'idée : un discriminateur appris

En haute dimension, choisir manuellement une bonne statistique $T$ (moments, etc.) est très difficile. **L'idée des GAN : laisser un réseau de neurones apprendre la statistique.**

On entraîne un classifieur binaire à distinguer $S_1$ (vrais) de $S_2$ (faux) :
- s'il y arrive bien → les deux distributions sont distinguables → $P \neq Q$,
- s'il n'y arrive pas → on ne peut pas les distinguer → $P \approx Q$.

Et on entraîne **simultanément** le générateur à produire des échantillons qui rendent ce classifieur impuissant. C'est le jeu adversarial.

## III. Le modèle GAN

### A. Les deux joueurs

> [!warning] Anatomie d'un GAN
> Deux réseaux de neurones jouent l'un contre l'autre :
> 
> - **Générateur $G_\phi$** : prend un bruit $z \sim p(z) = \mathcal{N}(0, I)$ en entrée et produit un échantillon $G_\phi(z)$. Objectif : que ses sorties soient indistinguables des vraies données.
> - **Discriminateur $D_\theta$** : prend un échantillon $x$ (vrai ou faux) et sort $D_\theta(x) \in [0, 1]$, sa probabilité estimée que $x$ vienne de $p_{\text{data}}$. Objectif : distinguer correctement les vrais des faux.
> 
> Le générateur cherche à **tromper** le discriminateur ; le discriminateur cherche à **résister** à la tromperie. À l'équilibre, le générateur produit des échantillons indistinguables des vraies données et le discriminateur ne fait pas mieux qu'un tirage à pile ou face.

Le diagramme ci-dessus illustre l'architecture du jeu.

![[vanilla5.png|271]]

### B. Le jeu minimax

On formalise les deux objectifs en une seule fonction de valeur :

$$\min_\phi \max_\theta \; V(G_\phi, D_\theta) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}\!\big[\log D_\theta(x)\big] \;+\; \mathbb{E}_{z \sim p(z)}\!\big[\log\!\big(1 - D_\theta(G_\phi(z))\big)\big].$$

Décodage :

- **Premier terme** : le discriminateur veut $D_\theta(x) \to 1$ pour les vrais $x$ → $\log D_\theta(x) \to 0$ (max).
- **Second terme** : le discriminateur veut $D_\theta(G_\phi(z)) \to 0$ pour les faux → $\log(1 - D_\theta(G_\phi(z))) \to 0$ (max).
- Le générateur veut le contraire sur le second terme : pousser $D_\theta(G_\phi(z))$ vers 1 → $\log(1 - D_\theta(G_\phi(z))) \to -\infty$ (min).

Le premier terme ne dépend pas de $\phi$ ; seul le second est joué par les deux camps.

### C. Algorithme d'entraînement alterné

On ne résout pas le min-max d'un coup. On alterne :

**Étape 1 — gradient ascent sur $D$** (à $G$ fixé) :

$$\theta \leftarrow \theta + \alpha \, \nabla_\theta\!\Big[\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\theta(x)] + \mathbb{E}_{z}[\log(1 - D_\theta(G_\phi(z)))]\Big].$$

**Étape 2 — gradient descent sur $G$** (à $D$ fixé) :

$$\phi \leftarrow \phi - \alpha \, \nabla_\phi \, \mathbb{E}_{z}[\log(1 - D_\theta(G_\phi(z)))].$$

En pratique, on fait un (ou quelques) pas sur $D$, puis un pas sur $G$, et on alterne.

> [!note]- Implémentation : cross-entropy binaire
> Concrètement, on n'écrit jamais $\log D$ et $\log(1-D)$ explicitement dans le code : c'est exactement la **binary cross-entropy** avec labels 1 pour les vrais et 0 pour les faux. La loss du discriminateur s'écrit :
> ```
> loss_D = BCE(D(x_real), 1) + BCE(D(G(z)), 0)
> ```
> et le générateur :
> ```
> loss_G = BCE(D(G(z)), 1)   # note: label 1, voir §V.A
> ```

## IV. Analyse théorique : le point fixe

On va maintenant montrer que ce jeu, joué à l'optimum, ramène à minimiser une **divergence de Jensen-Shannon** entre $p_{\text{data}}$ et la distribution du générateur. C'est le résultat fondateur de [Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661).

### A. Discriminateur optimal pour un $G$ fixé

À $G$ fixé, posons $p_g$ la distribution induite par le générateur sur $x$ (image de $p(z)$ par $G$). On veut maximiser

$$V(G, D) = \int_x p_{\text{data}}(x) \log D(x) \, dx + \int_x p_g(x) \log(1 - D(x)) \, dx.$$

L'intégrande $a \log y + b \log(1 - y)$ (avec $a, b > 0$) est maximisé en $y^* = a / (a + b)$. D'où le **discriminateur optimal** :

$$\boxed{\; D^*_G(x) \;=\; \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \;}$$

> [!note] Interprétation
> Si $p_{\text{data}}(x) \gg p_g(x)$ → $D^*(x) \approx 1$ (le discriminateur classe $x$ comme vrai).  
> Si $p_g(x) \gg p_{\text{data}}(x)$ → $D^*(x) \approx 0$ (classé comme faux).  
> Si $p_{\text{data}}(x) = p_g(x)$ → $D^*(x) = 1/2$ (équiprobable). **Le discriminateur ne peut plus distinguer.**

### B. Substitution dans $V$ : la JSD apparaît

Plugin $D^*_G$ dans $V$ :

$$V(G, D^*_G) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\!\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right].$$

On factorise en faisant apparaître la moyenne $m = (p_{\text{data}} + p_g) / 2$ :

$$V(G, D^*_G) = -\log 4 + D_{\text{KL}}\!\big(p_{\text{data}} \,\big\|\, m\big) + D_{\text{KL}}\!\big(p_g \,\big\|\, m\big).$$

> [!warning] $\min_G V(G, D^*_G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \,\|\, p_g)$
> Par définition de la **divergence de Jensen-Shannon** :
> 
> $$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \,\|\, M) + \frac{1}{2} D_{\text{KL}}(Q \,\|\, M), \quad M = \frac{P + Q}{2}.$$
> 
> Donc le jeu GAN à l'optimum du discriminateur revient à **minimiser la JSD** entre $p_{\text{data}}$ et $p_g$. Comme $\text{JSD} \geq 0$ avec égalité ssi $p_{\text{data}} = p_g$, le minimum global est atteint à
> 
> $$p_g = p_{\text{data}}, \qquad V^* = -\log 4 \approx -1.39.$$
> 
> **Le GAN est mathématiquement bien fondé** : à l'équilibre de Nash du jeu min-max, la distribution du générateur égale la distribution des données. C'est la garantie théorique qui justifie tout l'édifice.

### C. Mais en pratique...

Ce résultat repose sur **deux hypothèses idéales** :
1. le discriminateur est optimal à chaque pas ;
2. les mises à jour du générateur se font dans *l'espace des fonctions* (pas dans l'espace paramétrique).

En pratique, ni l'un ni l'autre n'est vérifié. C'est ce qui cause les pathologies que l'on va voir maintenant.

## V. Les pathologies pratiques

### A. Vanishing gradient & non-saturating loss

Premier problème : à *l'initialisation*, le générateur produit des images aléatoires que le discriminateur classe immédiatement comme fake avec haute confiance ($D(G(z)) \approx 0$). Or la loss du générateur

$$\mathcal{L}_G^{\text{sat}}(\phi) = \mathbb{E}_z[\log(1 - D_\theta(G_\phi(z)))]$$

a un **gradient quasi-nul quand $D(G(z)) \approx 0$** : la sigmoïde sature, $\log(1 - 0) = 0$, et les dérivées partielles s'évanouissent. Le générateur n'apprend rien.

> [!warning] Non-saturating loss (trick de Goodfellow 2014)
> Plutôt que **minimiser** $\log(1 - D(G(z)))$, on **maximise** $\log D(G(z))$ :
> 
> $$\mathcal{L}_G^{\text{ns}}(\phi) = -\mathbb{E}_z[\log D_\theta(G_\phi(z))].$$
> 
> Mathématiquement, les deux ont le même argmax pour $G$ — mais leurs **paysages de gradient sont très différents**. La version non-saturante donne des gradients informatifs même quand $D(G(z)) \approx 0$, parce que $-\log D \to +\infty$ avec un gradient fort.
> 
> En pratique, *tous* les GANs utilisent la non-saturating loss. La forme min-max théorique de §III.B reste pédagogiquement utile, mais ce n'est pas ce que le code optimise.

### B. Mode collapse

Deuxième pathologie : le **mode collapse**, où le générateur "découvre" qu'un petit nombre d'images réussit toujours à tromper le discriminateur, et ne génère plus que celles-là.

> [!example] Le mécanisme du mode collapse
> Supposons que $p_{\text{data}}$ ait plusieurs modes (par ex. les 10 chiffres MNIST). À chaque étape :
> 
> 1. Le générateur trouve un mode pour lequel le discriminateur courant est faible — disons "le 3". Il y va.
> 2. Le discriminateur s'adapte et apprend à mieux classer les 3.
> 3. Le générateur saute sur "le 8". Etc.
> 
> Le générateur ne couvre **jamais** tous les modes en même temps — il oscille de l'un à l'autre, ou pire, se concentre durablement sur quelques modes seulement. Le critère min-max est en théorie satisfait localement, mais la diversité de $p_g$ est très inférieure à celle de $p_{\text{data}}$.

C'est *la* pathologie structurelle des GAN. Beaucoup de variantes (minibatch discrimination, unrolled GAN, WGAN) cherchent à la corriger.

### C. Instabilité et absence de signal de convergence

Troisième problème : la loss du discriminateur **n'est pas un indicateur de convergence**. Contrairement à une MLE classique où la log-vraisemblance descend monotonement, la loss du GAN **oscille** — c'est normal, c'est un jeu. Mais ça veut dire qu'on n'a aucun critère d'arrêt fiable : il faut surveiller la qualité visuelle à la main et stocker des checkpoints fréquents.

> [!note]- Le théorème de convergence et ses limites
> Goodfellow 2014 prouve que *si le discriminateur est optimal à chaque étape et que le générateur est mis à jour dans l'espace des fonctions*, alors $p_g \to p_{\text{data}}$. Aucune des deux hypothèses n'est vérifiée en pratique : on optimise en paramètres avec quelques pas SGD seulement. La théorie ne s'applique pas directement à ce qu'on fait.

## VI. Wasserstein GAN

### A. Pourquoi la JSD est le mauvais choix

Tout vient d'un problème fondamental de la JSD : **elle sature quand les supports de $p_{\text{data}}$ et $p_g$ ne se chevauchent pas**.

> [!warning] La pathologie de la JSD
> Si $\text{supp}(p_{\text{data}}) \cap \text{supp}(p_g) = \emptyset$ (les deux distributions sont disjointes), alors :
> 
> $$\text{JSD}(p_{\text{data}} \,\|\, p_g) = \log 2 \quad \text{(constante).}$$
> 
> Gradient = 0. **Le générateur ne reçoit aucun signal sur la direction dans laquelle il faut bouger.**
> 
> Or au début de l'entraînement, $p_g$ et $p_{\text{data}}$ vivent typiquement sur des sous-variétés de basse dimension dans $\mathbb{R}^n$ — il est *probable* qu'elles soient quasi-disjointes. C'est précisément le régime initial où le générateur a le plus besoin de gradients informatifs, et c'est précisément là que la JSD échoue.
> 
> Cette analyse, due à [Arjovsky & Bottou (2017)](https://arxiv.org/abs/1701.04862), est le point de départ du Wasserstein GAN.

### B. Remplacer JSD par la distance de Wasserstein-1

La **distance de Wasserstein-1** entre deux mesures $P$ et $Q$ varie *continûment* quand les supports se rapprochent — même quand ils sont disjoints :

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|].$$

> [!note] Pour la définition complète et la théorie
> La distance $W_1$, le problème de Monge-Kantorovich, et tout le formalisme du transport optimal sont traités dans le dossier dédié. Voir notamment `[[03_Distance de Wasserstein]]` pour la définition et les propriétés, et `[[04_Dualité de Kantorovich]]` pour la formule duale utilisée plus bas.

L'avantage clé : si $p_g$ s'approche de $p_{\text{data}}$ même sans recouvrement, $W_1(p_g, p_{\text{data}})$ **décroît**, donc fournit un gradient informatif.

### C. La formule duale Kantorovich-Rubinstein

L'inf sur les plans de transport $\gamma$ est intractable en pratique (variables en dimension $n^2$). Mais la **dualité de Kantorovich-Rubinstein** transforme ce problème en un sup sur les fonctions 1-Lipschitz :

$$\boxed{\; W_1(P, Q) \;=\; \sup_{\|f\|_L \leq 1} \; \mathbb{E}_{x \sim P}[f(x)] \;-\; \mathbb{E}_{x \sim Q}[f(x)] \;}$$

> [!note]- La condition de Lipschitz
> $\|f\|_L \leq 1$ signifie que $f$ est **1-Lipschitz** : $|f(x) - f(y)| \leq \|x - y\|$ pour tous $x, y$. Géométriquement, la pente de $f$ est partout bornée par 1. Cette contrainte est ce qui empêche le sup d'exploser à l'infini ; elle est l'analogue dual de la contrainte de marginale dans le primal.
> 
> Preuve de la dualité dans `[[04_Dualité de Kantorovich]]`.

C'est exactement la formule dont on a besoin pour un GAN : un sup sur une famille de fonctions paramétrables par un réseau de neurones.

### D. L'architecture WGAN

On remplace le discriminateur $D_\theta$ d'un GAN classique par un **critic** $f_w$ qui n'est plus un classifieur binaire mais une fonction scalaire $\mathbb{R}^n \to \mathbb{R}$ (pas de sigmoïde, pas de probabilité). La loss devient :

$$\mathcal{L}_{\text{WGAN}}(\phi, w) \;=\; \mathbb{E}_{x \sim p_{\text{data}}}[f_w(x)] \;-\; \mathbb{E}_{z \sim p(z)}[f_w(G_\phi(z))].$$

L'entraînement alterne, comme pour le GAN classique :

- **Critic** : $\max_w$ sous la contrainte $\|f_w\|_L \leq 1$ — cherche à approximer $W_1$.
- **Generator** : $\min_\phi \mathcal{L}_{\text{WGAN}}$ — cherche à réduire $W_1$.

> [!warning] Pourquoi "critic" et pas "discriminateur"
> Sémantiquement le rôle a changé : $f_w$ ne classifie plus (vrai/faux) ; il **note** chaque échantillon pour estimer la distance $W_1$. Plus la note moyenne diffère entre vrais et faux, plus les deux distributions sont éloignées. D'où le nom.

### E. Imposer la contrainte Lipschitz

La grande difficulté pratique : comment garantir que $f_w$ soit 1-Lipschitz ? Deux approches.

**Weight clipping (WGAN original, [Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875)).** Après chaque mise à jour de $w$, on tronque chaque poids dans $[-c, c]$ pour un petit $c$ (typiquement $0.01$). Cela borne grossièrement la constante de Lipschitz du réseau. Brutal et imparfait — peut induire des distributions de poids dégénérées.

**Gradient penalty (WGAN-GP, [Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)).** Au lieu de clipper, on ajoute un terme à la loss qui pénalise les gradients de $f_w$ qui s'éloignent de 1 en norme :

$$\mathcal{L}_{\text{WGAN-GP}} = \mathcal{L}_{\text{WGAN}} + \lambda \, \mathbb{E}_{\hat x}\!\left[(\|\nabla_{\hat x} f_w(\hat x)\|_2 - 1)^2\right],$$

où $\hat x = \varepsilon x + (1 - \varepsilon) G(z)$ est un point tiré uniformément sur le segment entre un vrai et un faux. C'est aujourd'hui la version standard. $\lambda = 10$ marche bien sur la plupart des architectures.

### F. Bénéfices pratiques

> [!warning] Pourquoi WGAN-GP est meilleur que GAN vanilla
> - **Pas de vanishing gradient.** Le critic n'a pas de sigmoïde finale ; ses gradients restent informatifs quel que soit le degré d'entraînement.
> - **Plus de mode collapse en pratique.** $W_1$ est sensible à la couverture des modes, contrairement à JSD qui tolère qu'on ignore une partie de $p_{\text{data}}$ tant qu'on en couvre une autre.
> - **Une loss interprétable.** La loss du critic est une estimation de $W_1(p_{\text{data}}, p_g)$ — elle **décroît** quand l'entraînement progresse. On a enfin un signal de convergence visualisable.
> - **Entraînement plus stable** : on peut entraîner le critic à fond sans risquer d'écraser le générateur (au contraire — un meilleur critic donne une meilleure estimation de $W_1$).

## VII. Bilan et positionnement

### A. Forces des GAN

- **Qualité visuelle** : pendant longtemps (2014–2021), les GAN ont produit les images les plus nettes et photoréalistes — StyleGAN reste un benchmark difficile à battre.
- **Génération rapide** : un seul forward pass du générateur. Pas de séquentialité (vs AR), pas d'itérations multiples (vs diffusion).
- **Flexibilité d'architecture** : pas de contrainte d'invertibilité (vs flows), pas de structure latente imposée (vs VAE).

### B. Faiblesses structurelles

- **Pas de densité accessible.** Contrairement aux AR, VAE, flows, on ne peut pas évaluer $p_\theta(x)$. Impossible donc d'utiliser un GAN pour compression, détection d'anomalie, comparaison rigoureuse de modèles.
- **Mode collapse** (vanilla) — partiellement mitigé par WGAN-GP mais jamais complètement.
- **Pas de signal de convergence** (vanilla) — réglé par WGAN.
- **Sensibilité aux hyperparamètres et à l'architecture** — les GAN ont la réputation justifiée d'être *capricieux* à entraîner.
- **Pas d'espace latent structuré** — le mapping $z \to x$ n'est ni régulier (comme VAE) ni inversible (comme flow). L'interpolation marche en pratique mais n'a pas de garantie théorique.

### C. Pourquoi les diffusion ont gagné

Depuis 2021-2022, les **diffusion models** ont largement remplacé les GAN pour la génération d'images, parce qu'ils :
- atteignent la même qualité voire mieux (DALL-E 2, Stable Diffusion, Midjourney),
- s'entraînent par MLE (objectif stable, monotone, pas de jeu),
- supportent des conditionnements riches (texte, masques, edge maps, etc.),
- couvrent mieux les modes (problème du mode collapse absent par construction).

Le prix : génération itérative (plusieurs forward passes), donc plus lente.

Les GAN restent utilisés en :
- super-résolution (SRGAN, ESRGAN),
- image-to-image (Pix2Pix, CycleGAN),
- style transfer,
- comme *discriminateurs auxiliaires* dans des architectures hybrides.

### Tableau mis à jour

| Aspect | Autorégressif | VAE | Flow | **GAN** | Diffusion |
|---|---|---|---|---|---|
| Structure | Chain rule | Latent + décodeur | Bijection | **Adversariel** | Markov débruitage |
| Densité $p(x)$ | Exacte | Borne (ELBO) | Exacte | **Inaccessible** | Approchée |
| Apprentissage | MLE direct | ELBO | MLE | **Min-max (JSD ou $W_1$)** | Score matching |
| Échantillonnage | Lent (séquentiel) | Rapide | Rapide | **Rapide** | Itératif |
| Espace latent | Non | Oui (structuré) | Oui (bijectif) | **Oui (non-régulier)** | Implicite |
| Stabilité d'entraînement | Excellente | Très bonne | Bonne | **Délicate** | Bonne |

## VIII. Extensions

> [!todo] À étoffer ultérieurement
> Beaucoup de variantes utiles non couvertes ici :
> 
> - **DCGAN** (Radford et al. 2015) — architecture de référence pour générateur et discriminateur sur images : convolutions stridées, BatchNorm, ReLU/LeakyReLU. Précurseur de presque toutes les architectures images modernes.
> - **Conditional GAN** (Mirza & Osindero 2014) — conditionner sur un label : $G(z, y)$ et $D(x, y)$.
> - **StyleGAN / StyleGAN2 / StyleGAN3** (Karras et al. 2019-2021) — l'apogée des GAN pour visages. Architecture progressive + AdaIN.
> - **CycleGAN** (Zhu et al. 2017) — image-to-image non-pairé via cycle consistency. Photos ↔ peintures, chevaux ↔ zèbres.
> - **BigGAN** (Brock et al. 2018) — passage à l'échelle conditionnelle sur ImageNet.
> - **SAGAN** (Zhang et al. 2018) — attention dans le discriminateur et le générateur.

---

## Pour aller plus loin

- **Article fondateur.** I. Goodfellow et al. *Generative Adversarial Nets.* NeurIPS 2014.
- **Critique de la log-vraisemblance comme proxy.** L. Theis, A. van den Oord, M. Bethge. *A note on the evaluation of generative models.* ICLR 2016.
- **Wasserstein GAN.** M. Arjovsky, S. Chintala, L. Bottou. *Wasserstein GAN.* arXiv 2017.
- **WGAN-GP.** I. Gulrajani et al. *Improved Training of Wasserstein GANs.* NeurIPS 2017.
- **Analyse de l'instabilité.** M. Arjovsky, L. Bottou. *Towards Principled Methods for Training Generative Adversarial Networks.* ICLR 2017.
- **Cours.** Stefano Ermon, *CS236 Deep Generative Models*, Stanford. Leçons sur GAN et f-GAN.
- **Pont OT.** Voir le dossier `[[1-Mathématiques/Optimal Transport]]` pour Wasserstein, Kantorovich, dualité — toute la théorie derrière WGAN.
