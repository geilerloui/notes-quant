---
title: Variational Inference (VI)
---
# Variational Inference (VI)

> Cette note prolonge naturellement `[[05_Expectation_Maximization]]`. EM était l'algorithme pour faire du MLE avec variables latentes **quand la posterior $p_\theta(z \mid x)$ est calculable** (cas GMM, K-means, mélanges discrets simples). VI s'attaque au cas où **la posterior n'est plus calculable** : on l'**approxime** par un $q_\phi(z)$ d'une famille paramétrique restreinte, et on optimise sur $\phi$. L'ELBO reste l'objectif central — mais maintenant on a deux jeux de paramètres ($\theta$ pour le modèle, $\phi$ pour l'approximation) au lieu d'un seul.

## Conventions de notation

> [!warning] Notation tenue partout dans cette note
> | Symbole | Sens |
> |---|---|
> | $x$ | Observation (on raisonne sur un point ; généraliser à un dataset $X$ est trivial) |
> | $z$ | Variable latente (typiquement continue en VI, contrairement à EM-GMM) |
> | $\theta$ | Paramètres du modèle génératif $p_\theta(x, z)$ |
> | $p_\theta(z \mid x)$ | **Vraie** posterior — intractable, c'est ce qu'on veut approcher |
> | $q_\phi(z)$ | **Distribution variationnelle** — l'approximation, paramétrée par $\phi$ |
> | $\mathcal{Q}$ | **Famille variationnelle** — l'ensemble des $q_\phi$ qu'on s'autorise |
> | $\mathcal{L}(\phi, \theta)$ | ELBO, fonction des deux jeux de paramètres |
> | $D_{\text{KL}}(q \| p)$ | Divergence de Kullback-Leibler de $q$ vers $p$ |
> 
> **Changement vs EM** : avant, $q$ était une distribution libre (on prenait $q^* = p_\theta(z \mid x)$ pile poil). Maintenant, $q$ est **contraint** à vivre dans $\mathcal{Q}$, donc paramétré par un $\phi$ qu'on optimise.

## I. Le problème : quand la posterior n'est plus calculable

### A. Rappel — pourquoi EM marche sur GMM

Dans `[[05_Expectation_Maximization]]`, l'algorithme EM repose sur un calcul-clé au E-step :

$$q^{\text{new}}(z) = p_{\theta^{\text{old}}}(z \mid x) = \frac{p_\theta(x, z)}{p_\theta(x)} = \frac{p_\theta(x, z)}{\sum_z p_\theta(x, z)}.$$

Pour GMM, c'est faisable parce que $z \in \{1, \ldots, K\}$ est **discret** et $K$ est **petit** (typiquement 2-10). La somme $\sum_z$ est juste une somme finie sur $K$ termes — on l'évalue mécaniquement et on obtient les responsabilités $p_{\theta^{\text{old}}}(z \mid x) = \gamma_{nk}$.

### B. Le problème en haute dimension ou en continu

Dans les modèles modernes, $z$ est typiquement :

- **Continu** : $z \in \mathbb{R}^d$ avec $d$ grand (souvent 32, 128, 512...).
- **Multimodal** : la posterior peut avoir plusieurs bosses séparées.
- **Non-conjugué** : le prior et la likelihood ne se combinent pas en forme fermée.

Dans tous ces cas, la "marginale" $p_\theta(x) = \int p_\theta(x, z) \, dz$ devient une **intégrale en haute dimension** sans forme fermée. Donc le dénominateur de Bayes est intractable, donc $p_\theta(z \mid x)$ est intractable.

> [!warning] L'obstacle structurel
> $$p_\theta(z \mid x) = \frac{p_\theta(x, z)}{\int p_\theta(x, z) \, dz}.$$
> 
> Le numérateur $p_\theta(x, z)$ est **toujours** facile à évaluer pour un $(x, z)$ donné (c'est le produit $p(z) \, p_\theta(x \mid z)$, qu'on sait écrire). Le **dénominateur** est l'obstacle : une intégrale en haute dimension qu'on ne sait ni calculer analytiquement ni approximer correctement par des méthodes naïves.
> 
> **EM ne peut pas démarrer** : on ne peut pas faire un E-step exact.

### C. Trois exemples concrets

**1. Bayesian logistic regression.** Modèle : $p(y_n \mid x_n, w) = \sigma(w^\top x_n)^{y_n} (1-\sigma(w^\top x_n))^{1-y_n}$ avec un prior gaussien $p(w) = \mathcal{N}(0, \alpha^{-1} I)$. La posterior $p(w \mid \text{data})$ n'a **pas de forme fermée** parce que la sigmoïde ne se conjugue pas avec le prior gaussien. C'est l'exemple école de VI.

**2. Modèles à latent continu type PPCA / Factor Analysis.** $z \in \mathbb{R}^d$, $x \mid z \sim \mathcal{N}(Wz + \mu, \Sigma)$. Pour ces cas linéaires gaussiens, on a *quand même* une forme fermée (cas spécial agréable). Mais dès qu'on remplace la moyenne $Wz + \mu$ par un **réseau de neurones non-linéaire** $f_\theta(z)$ → plus de forme fermée → VAE → VI obligatoire.

**3. Topic models (LDA).** Variables latentes : pour chaque document, une distribution sur les topics ; pour chaque mot, un assignment de topic. La posterior conjointe est massive et intractable. CAVI (qu'on verra en §V) est le workhorse standard ici.

### D. La parade : approximer

L'idée centrale de VI est simple : puisqu'on ne peut pas avoir la vraie posterior $p_\theta(z \mid x)$, on en cherche une **approximation** $q_\phi(z)$ dans une famille qu'on contrôle. On paie un prix (la borne n'est plus serrée), mais on gagne la tractabilité.

C'est exactement le passage qu'on évoquait dans la généalogie à la fin de la note EM :

$$\underbrace{\text{EM exact}}_{p_\theta(z \mid x) \text{ tractable}} \;\xrightarrow{\text{posterior intractable}}\; \underbrace{\text{Variational Inference}}_{q_\phi(z) \text{ paramétrique}}.$$

La suite de la note détaille **comment** on choisit $q_\phi$, **quel objectif** on optimise pour le trouver, et **quelles propriétés** ça nous donne.

## II. L'idée de VI : minimiser KL = maximiser ELBO

### A. Approximer la posterior dans une famille restreinte

Formalisons. On se donne une **famille variationnelle** $\mathcal{Q}$ — un ensemble de distributions $q_\phi(z)$ paramétrées par $\phi$. Exemples typiques :

- **Famille gaussienne** : $q_\phi(z) = \mathcal{N}(z \mid \mu_\phi, \Sigma_\phi)$, avec $\phi = (\mu_\phi, \Sigma_\phi)$.
- **Famille mean-field** : $q_\phi(z) = \prod_i q_{\phi_i}(z_i)$ (factorisation totale par composante).
- **Famille mixte** : un mix des deux, par exemple gaussienne par composante avec factorisation.

On cherche le membre de $\mathcal{Q}$ qui est **le plus proche** de la vraie posterior :

$$\phi^* = \arg\min_{\phi}\; D_{\text{KL}}\!\big(q_\phi(z) \,\|\, p_\theta(z \mid x)\big).$$

> [!note] Pourquoi cette direction de KL ?
> Il existe **deux** directions possibles pour la KL : $\text{KL}(q \| p)$ et $\text{KL}(p \| q)$. VI utilise **$\text{KL}(q \| p)$** ("q vers p"). Cette direction n'est pas neutre — elle détermine la nature même de l'approximation (mode-seeking vs mode-covering). On y revient en détail en §IV.

### B. Le serpent qui se mord la queue

Problème immédiat : la KL contient la posterior qu'on ne connaît pas.

$$D_{\text{KL}}(q_\phi \| p_\theta(z\mid x)) = \int q_\phi(z) \log \frac{q_\phi(z)}{p_\theta(z \mid x)} \, dz.$$

Évaluer cette quantité demande $p_\theta(z \mid x)$ — précisément ce qu'on ne peut pas calculer. **Si on pouvait évaluer la KL, on n'aurait pas besoin de VI.**

C'est exactement là que l'**ELBO** entre en scène.

### C. L'identité-clé (rappel d'EM)

Souviens-toi de la décomposition fondamentale (§IV.A de `[[05_Expectation_Maximization]]`) : pour toute $q$,

$$\boxed{\;\log p_\theta(x) \;=\; \mathcal{L}(q, \theta) \;+\; D_{\text{KL}}\!\big(q(z) \,\|\, p_\theta(z \mid x)\big)\;}$$

avec

$$\mathcal{L}(q, \theta) := \int q(z) \log \frac{p_\theta(x, z)}{q(z)} \, dz = \mathbb{E}_{z \sim q}\!\big[\log p_\theta(x, z) - \log q(z)\big].$$

Cette identité est vraie pour **n'importe quelle** $q$ — y compris pour notre $q_\phi$ restreinte à la famille $\mathcal{Q}$.

### D. L'astuce centrale : minimiser KL = maximiser ELBO

On veut minimiser $D_{\text{KL}}(q_\phi \| p_\theta(z \mid x))$ sur $\phi$, mais on ne sait pas la calculer (§II.B). L'astuce consiste à voir que **maximiser l'ELBO** fait *exactement* le même travail. Décortiquons en 5 cliquets logiques.

**Étape 1 — Le point de départ.** L'identité de §II.C, appliquée à notre $q_\phi$, donne :

$$\log p_\theta(x) = \mathcal{L}(\phi, \theta) + D_{\text{KL}}(q_\phi \| p_\theta(z \mid x)).$$

C'est une **égalité exacte** — pas une borne, pas une approximation. Trois quantités : un nombre à gauche, une somme de deux nombres à droite.

**Étape 2 — Qui dépend de $\phi$ et qui n'en dépend pas ?** Regardons chaque terme :

| Terme | Dépend de $\phi$ ? |
|---|---|
| $\log p_\theta(x) = \log \int p_\theta(x, z)\,dz$ | **Non** — aucun $\phi$ dans son écriture. C'est une **constante** quand on bouge $\phi$ (à $\theta, x$ fixés). |
| $\mathcal{L}(\phi, \theta) = \mathbb{E}_{q_\phi}[\log p_\theta(x, z) - \log q_\phi(z)]$ | **Oui** — $q_\phi$ entre dans l'espérance ET dans le terme à l'intérieur. |
| $D_{\text{KL}}(q_\phi \| p_\theta(z\mid x))$ | **Oui** — $q_\phi$ entre dans la KL. |

**Étape 3 — La contrainte du balancier.** Comme la somme est constante en $\phi$ :

$$\mathcal{L}(\phi, \theta) + D_{\text{KL}}(\phi) = \underbrace{\log p_\theta(x)}_{\text{fixé}}.$$

Si tu bouges $\phi$ et que l'ELBO monte de $+0.5$, alors la KL doit descendre de $-0.5$, exactement. C'est **arithmétique**, pas une approximation.

> [!example] Analogie : les deux vases communicants
> Imagine deux vases reliés par un tuyau, niveau total fixé. Si tu vides l'un, l'autre se remplit d'autant. Tu ne peux pas mesurer le vase B (il est dans le noir), mais tu peux mesurer A. Pour minimiser le contenu de B, il te suffit de maximiser celui de A — la contrainte de niveau total s'occupe du reste.
> 
> Ici : **A = ELBO** (mesurable, c'est notre vase éclairé), **B = KL** (intractable, vase dans le noir), **niveau total = $\log p_\theta(x)$** (fixé en $\theta, x$).

**Étape 4 — La conséquence algorithmique.** L'objectif initial était :

$$\phi^* = \arg\min_\phi\; D_{\text{KL}}(q_\phi \| p_\theta(z \mid x)).$$

Comme $D_{\text{KL}}(\phi) = \log p_\theta(x) - \mathcal{L}(\phi, \theta)$, minimiser la KL revient à minimiser $-\mathcal{L}(\phi, \theta)$ (la constante $\log p_\theta(x)$ ne change pas l'argmin), c'est-à-dire **maximiser l'ELBO** :

$$\boxed{\;\arg\min_\phi\; D_{\text{KL}}(q_\phi \| p_\theta(z \mid x)) \;=\; \arg\max_\phi\; \mathcal{L}(\phi, \theta).\;}$$

**Étape 5 — Pourquoi c'est *utile*, pas juste joli.** Les deux objectifs sont équivalents au sens de l'argmax — mais **un seul des deux est calculable** :

- La **KL** s'écrit avec $p_\theta(z \mid x)$ → contient l'évidence intractable → infaisable.
- L'**ELBO** s'écrit avec $p_\theta(x, z)$ et $q_\phi(z)$ → tout est calculable point par point → faisable.

Donc on choisit l'objectif calculable (l'ELBO), on le maximise par n'importe quelle méthode standard (gradient ascent, coordinate ascent, etc.), et la KL descend mécaniquement d'autant. **Sans jamais évaluer la KL.**

> [!warning] Le résultat à retenir
> $$\arg\min_\phi\; D_{\text{KL}}(q_\phi \| p_\theta(z\mid x)) \;=\; \arg\max_\phi\; \mathcal{L}(\phi, \theta).$$
> 
> Justification : $\log p_\theta(x)$ est constante en $\phi$, donc $\mathcal{L}(\phi, \theta) + D_{\text{KL}}(\phi) = \text{cst}$, donc monter l'un descend l'autre d'autant.
> 
> En pratique, on optimise l'ELBO (calculable) au lieu de la KL (intractable). C'est la fondation algorithmique de tout VI — et plus tard, de tout VAE.

![[vi_elbo_decomposition.png]]
*Figure. La décomposition $\log p_\theta(x) = \mathcal{L}(\phi, \theta) + \text{KL}(q_\phi \| p_\theta(z \mid x))$ comme un "thermomètre". La hauteur totale $\log p_\theta(x)$ est **fixe** (en $\theta, x$ donnés). À gauche : $q_\phi^{(0)}$ initial, la KL prend une grosse part du thermomètre, l'ELBO peu. À droite : après optimisation sur $\phi$, la KL a fondu, l'ELBO a monté d'autant. **Maximiser l'ELBO = serrer la borne contre la vraie log-vraisemblance.***

### E. Pourquoi l'ELBO est tractable et la KL ne l'est pas

Développons l'ELBO pour voir explicitement ce qui se passe :

$$\mathcal{L}(\phi, \theta) = \mathbb{E}_{z \sim q_\phi}\!\big[\log p_\theta(x, z)\big] - \mathbb{E}_{z \sim q_\phi}\!\big[\log q_\phi(z)\big].$$

Les deux termes sont **calculables** parce que :

- $\log p_\theta(x, z) = \log p(z) + \log p_\theta(x \mid z)$ — chacun s'évalue à partir de la définition du modèle.
- $\log q_\phi(z)$ — c'est notre $q$, on contrôle sa forme analytique.
- L'espérance $\mathbb{E}_{z \sim q_\phi}[\cdot]$ s'évalue soit en forme fermée (si $\mathcal{Q}$ est bien choisie, par exemple gaussienne avec $p_\theta(x \mid z)$ gaussienne aussi), soit par Monte Carlo en tirant des échantillons $z^{(s)} \sim q_\phi$.

La KL, elle, s'écrit

$$D_{\text{KL}}(q_\phi \| p_\theta(z\mid x)) = \mathbb{E}_{z \sim q_\phi}[\log q_\phi(z)] - \mathbb{E}_{z \sim q_\phi}[\log p_\theta(z \mid x)],$$

et le second terme demande $\log p_\theta(z \mid x) = \log p_\theta(x, z) - \log p_\theta(x)$ — **avec l'évidence $\log p_\theta(x)$ dedans**. C'est précisément l'intégrale intractable qu'on voulait éviter.

> [!note]- Détail : l'évidence sort de l'ELBO
> En réécrivant la KL avec Bayes :
> 
> $$D_{\text{KL}}(q_\phi \| p_\theta(z\mid x)) = \mathbb{E}_q[\log q_\phi(z)] - \mathbb{E}_q[\log p_\theta(x, z)] + \underbrace{\log p_\theta(x)}_{\text{intractable}}.$$
> 
> L'évidence $\log p_\theta(x)$ apparaît explicitement — voilà l'obstacle.
> 
> En **réarrangeant**, on obtient :
> 
> $$\log p_\theta(x) - D_{\text{KL}}(q_\phi \| p_\theta(z\mid x)) = \mathbb{E}_q[\log p_\theta(x, z)] - \mathbb{E}_q[\log q_\phi(z)] = \mathcal{L}(\phi, \theta).$$
> 
> L'ELBO est précisément la quantité où **l'évidence intractable a disparu** par soustraction. C'est pour ça qu'elle est calculable. ∎

### F. Le bilan algorithmique

La stratégie globale de VI tient en deux lignes :

1. Choisir une famille $\mathcal{Q}$ paramétrée par $\phi$.
2. Maximiser $\mathcal{L}(\phi, \theta)$ sur $\phi$ (et éventuellement sur $\theta$ en même temps, si on apprend aussi le modèle).

La borne $\mathcal{L}$ ne sera **jamais** parfaitement serrée (sauf cas dégénéré où $p_\theta(z \mid x) \in \mathcal{Q}$) : il restera toujours un gap $D_{\text{KL}}(q_\phi^* \| p_\theta(z \mid x)) > 0$ qui mesure à quel point notre famille est mal calibrée pour ce problème. C'est le prix qu'on paie pour la tractabilité.

**Reste deux questions critiques** :

- *Comment choisir la famille $\mathcal{Q}$ ?* → §III.
- *Qu'est-ce qu'on perd avec la direction $\text{KL}(q \| p)$ vs $\text{KL}(p \| q)$ ?* → §IV (la propriété mode-seeking, qu'on a déjà mentionnée).

## III. Choisir la famille variationnelle $\mathcal{Q}$

§II nous a dit *quoi* optimiser (l'ELBO) et *sur quoi* (les paramètres $\phi$ d'un $q_\phi$). Reste à fixer **la forme de $q_\phi$** — c'est-à-dire la famille $\mathcal{Q}$ dans laquelle on cherche.

### A. Le grand trade-off : expressivité contre tractabilité

Toute la difficulté du choix de $\mathcal{Q}$ se résume à deux exigences contradictoires :

| Si $\mathcal{Q}$ est **expressive** (riche) | Si $\mathcal{Q}$ est **simple** (pauvre) |
|---|---|
| Bonne approximation : il existe un $q_\phi \in \mathcal{Q}$ proche de la vraie posterior | L'approximation rate les structures fines de la posterior (corrélations, multimodalité) |
| ELBO serré : $D_{\text{KL}}(q_\phi^* \| p_\theta(z \mid x))$ petit | Gap résiduel important |
| Mais : optimisation difficile, gradients lourds | Optimisation facile (souvent forme fermée) |
| Peu de paramètres physiques en commun entre composantes → calculs coûteux | Petit nombre de paramètres → updates rapides |

**Aucun choix n'est universellement meilleur.** Le bon $\mathcal{Q}$ dépend du modèle, de la dimension de $z$, et du compromis qu'on accepte. Les deux familles ci-dessous (mean-field et gaussienne paramétrique) sont les deux choix canoniques.

### B. La famille mean-field

Le choix le plus standard, et historiquement le premier. **Hypothèse** : les composantes de $z = (z_1, \ldots, z_d)$ sont **indépendantes** sous $q$ :

$$q_\phi(z) = \prod_{i=1}^{d} q_{\phi_i}(z_i).$$

Chaque facteur $q_{\phi_i}(z_i)$ est une distribution univariée libre (de famille à préciser — gaussienne, catégorielle, gamma, ce qu'on veut). Le paramètre global $\phi$ se décompose en sous-paramètres $\phi_1, \ldots, \phi_d$, un par composante.

> [!example] Mean-field concret
> Pour $z = (z_1, z_2, z_3) \in \mathbb{R}^3$ avec chaque facteur gaussien :
> 
> $$q_\phi(z_1, z_2, z_3) = \mathcal{N}(z_1 \mid \mu_1, \sigma_1^2) \cdot \mathcal{N}(z_2 \mid \mu_2, \sigma_2^2) \cdot \mathcal{N}(z_3 \mid \mu_3, \sigma_3^2).$$
> 
> Paramètres : $\phi = (\mu_1, \sigma_1, \mu_2, \sigma_2, \mu_3, \sigma_3)$ — six nombres. Comparer à une gaussienne 3D pleine ($\mathcal{N}(\mu, \Sigma)$ avec $\Sigma \in \mathbb{R}^{3 \times 3}$ symétrique) qui aurait 3 + 6 = 9 paramètres et capturerait les corrélations entre les $z_i$. **Mean-field jette toutes les corrélations** pour gagner en simplicité.

**Pourquoi c'est populaire.** Avec mean-field, les updates de chaque $q_{\phi_i}$ ont souvent une **forme fermée** (cf. CAVI en §V). Pas de gradient à calculer ; un calcul d'espérance, et on a directement $\phi_i^{\text{new}}$. C'est la version VI de l'EM-GMM, où le M-step avait aussi des updates en forme fermée.

### C. La famille paramétrique fixe (gaussienne)

Alternative : on fixe une forme paramétrique pour $q_\phi$ tout entier — pas une factorisation, mais une forme analytique avec des paramètres globaux. Le choix classique :

$$q_\phi(z) = \mathcal{N}(z \mid \mu_\phi, \Sigma_\phi), \quad \phi = (\mu_\phi, \Sigma_\phi).$$

Variantes selon comment on paramètre $\Sigma_\phi$ :

- **Covariance pleine** : $\Sigma_\phi \in \mathbb{R}^{d \times d}$ symétrique définie positive. Capture toutes les corrélations entre composantes. $\mathcal{O}(d^2)$ paramètres.
- **Covariance diagonale** : $\Sigma_\phi = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$. Équivalent à mean-field gaussien. $\mathcal{O}(d)$ paramètres.
- **Covariance low-rank** : $\Sigma_\phi = D + UU^\top$ avec $D$ diagonale et $U \in \mathbb{R}^{d \times r}$ avec $r \ll d$. Compromis intéressant.

**Différence pratique avec mean-field.** Avec une gaussienne paramétrique, on n'a **pas** d'updates en forme fermée (sauf modèles très spéciaux). On optimise par **gradient ascent** sur $\mathcal{L}(\phi, \theta)$, en calculant explicitement les dérivées par rapport à $\phi$. C'est plus flexible mais demande des outils de différentiation auto (PyTorch, JAX).

> [!note] Lien avec VAE
> Le VAE utilise une gaussienne paramétrique **amortizée** : au lieu d'avoir un $\phi_n$ par observation $x_n$, on a un encodeur neuronal $f_\phi : x \mapsto (\mu_\phi(x), \Sigma_\phi(x))$ qui prédit les paramètres de $q$ à partir de $x$. Économie massive de paramètres (un seul réseau pour tout le dataset au lieu de $N$ jeux indépendants). Voir `[[02_VAE]]`.

### D. La limite du mean-field : on perd les corrélations

Le pari de mean-field est : "les composantes de $z$ sont à peu près indépendantes sous la posterior". Quand c'est vrai, on perd peu. Quand c'est faux — c'est-à-dire quand la vraie posterior a des **corrélations fortes** entre $z_i$ et $z_j$ — mean-field les écrase brutalement et l'approximation rate la structure.

![[vi_mean_field_factorization.png]]
*Figure. Limite de l'approximation mean-field. **Gauche** : vraie posterior $p_\theta(z_1, z_2 \mid x)$ en 2D, gaussienne corrélée avec une covariance non-diagonale (ellipse oblique). **Droite** : meilleure approximation mean-field $q_\phi(z_1, z_2) = q_{\phi_1}(z_1) \cdot q_{\phi_2}(z_2)$ — gaussienne **axis-aligned** (ellipse alignée sur les axes). Mean-field a correctement attrapé les marginales (variance par axe) mais **a perdu la corrélation**. Conséquence : le gap KL résiduel mesure cette perte d'information.*

> [!warning] La leçon
> Le choix de $\mathcal{Q}$ encode une **hypothèse de modélisation** sur la posterior. Mean-field suppose l'indépendance ; gaussienne pleine suppose une covariance unimodale ; mixture de gaussiennes capture la multimodalité. **Choisir $\mathcal{Q}$ trop pauvre, c'est se condamner à un gap KL incompressible**, peu importe à quel point on optimise bien $\phi$.

### E. Récap : quelle famille pour quoi ?

| Famille | Capture corrélations ? | Capture multimodalité ? | Updates en forme fermée ? | Quand l'utiliser |
|---|---|---|---|---|
| **Mean-field gaussien** | Non | Non | Souvent oui (CAVI) | Modèles à $z$ haute dim peu corrélée ; LDA |
| **Gaussienne pleine** | Oui | Non | Rarement | $z$ corrélée, unimodale ; Bayesian logreg |
| **Gaussienne low-rank** | Partiellement | Non | Non | Compromis pour $d$ très grand |
| **Mixture de gaussiennes** | Oui | Oui | Non | Posterior bimodale ; rare en pratique (cher) |
| **Normalizing flow** | Oui (très) | Oui | Non | Posteriors complexes ; coûteux |
| **Gaussienne amortizée (VAE)** | Diagonale typiquement | Non | Non | $z$ haute dim ; on entraîne aussi $\theta$ |

Quel que soit le choix, le **principe d'optimisation** reste le même : maximiser l'ELBO sur $\phi$ (et éventuellement $\theta$). C'est juste la mécanique de l'optimisation qui change.

**Reste la question du gap résiduel : pourquoi la direction $\text{KL}(q \| p)$ favorise certaines erreurs plutôt que d'autres ?** C'est l'objet de §IV.

## IV. Mode-seeking vs mode-covering : la direction de la KL

Toute la note jusqu'ici utilise $D_{\text{KL}}(q_\phi \| p_\theta(z \mid x))$ — l'argument $q$ à gauche, $p$ à droite. Ce choix n'est pas innocent. La KL est **asymétrique**, et inverser ses deux arguments donne une approximation très différente. Cette section décortique pourquoi.

### A. La KL n'est pas une distance

Rappel — pour deux distributions $q$ et $p$ :

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz \quad\neq\quad D_{\text{KL}}(p \| q) = \int p(z) \log \frac{p(z)}{q(z)} \, dz.$$

Ces deux quantités sont **différentes** en général. Une vraie distance $d$ vérifie $d(q, p) = d(p, q)$ (symétrie), mais la KL non. On parle donc de **divergence**, pas de distance.

> [!warning] Les deux KL portent des noms en VI
> | Direction | Nom usuel | Utilisée par |
> |---|---|---|
> | $D_{\text{KL}}(q \| p)$ | **Reverse KL** ou **exclusive KL** | VI standard (notre cas) |
> | $D_{\text{KL}}(p \| q)$ | **Forward KL** ou **inclusive KL** | Expectation Propagation, MLE |
> 
> "Reverse" / "forward" est relatif à la direction "naturelle" $D_{\text{KL}}(p \| q)$ qui apparaît quand on fait du MLE classique. En VI, on utilise la version "à l'envers" pour des raisons de tractabilité (cf. §II.D).

### B. La mécanique de $D_{\text{KL}}(q \| p)$ — mode-seeking

Écrivons explicitement le coût qu'on minimise :

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_{z \sim q}\!\left[\log \frac{q(z)}{p(z)}\right].$$

**Lecture clé** : l'espérance est prise **sous $q$**. Donc seuls les points $z$ où $q(z)$ est non négligeable contribuent au coût. Là où $q(z) \approx 0$, le terme $q(z) \log(q/p)$ vaut $0 \cdot (\text{quelque chose})$ — la limite donne 0, le terme n'est pas compté.

Maintenant regardons le rapport $q/p$ sous le log :

- **Si $q(z) > 0$ et $p(z) > 0$** (les deux distributions sont d'accord ici) : $\log(q/p)$ vaut un nombre fini, coût modéré.
- **Si $q(z) > 0$ mais $p(z) \to 0$** (q parie sur un endroit où p est nul) : $\log(q/p) \to +\infty$. **Coût gigantesque**, par construction. $q$ est punie sévèrement pour mettre de la masse là où $p$ n'en a pas.
- **Si $q(z) = 0$ mais $p(z) > 0$** (q ignore un mode de p) : pas de coût parce que $q$ ne pèse pas dans l'intégrale à cet endroit.

> [!warning] La règle empirique pour $D_{\text{KL}}(q \| p)$
> **$q$ doit éviter les endroits où $p \approx 0$.** Mais $q$ peut tout à fait ignorer des régions où $p$ est non nulle, ça ne coûte rien.
> 
> Conséquence : si $p$ a plusieurs modes séparés par des "vallées" de basse probabilité, $q$ préférera **se coller sur un seul mode** plutôt que tenter de couvrir les deux (parce que couvrir les deux passe par la vallée, où $p \approx 0$ donc coût explosif).
> 
> On appelle cette tendance **mode-seeking** : $q$ choisit un mode et s'y enferme.

### C. La mécanique de $D_{\text{KL}}(p \| q)$ — mode-covering

Symétriquement :

$$D_{\text{KL}}(p \| q) = \mathbb{E}_{z \sim p}\!\left[\log \frac{p(z)}{q(z)}\right].$$

Maintenant l'espérance est prise **sous $p$**. Les rôles sont inversés :

- **Si $p(z) > 0$ et $q(z) > 0$** : coût modéré.
- **Si $p(z) > 0$ mais $q(z) \to 0$** (q ignore un endroit où p est forte) : $\log(p/q) \to +\infty$. **Coût gigantesque**. $q$ est punie pour ignorer un mode de $p$.
- **Si $p(z) = 0$ mais $q(z) > 0$** (q met de la masse dans une zone vide pour p) : pas de coût.

> [!warning] La règle empirique pour $D_{\text{KL}}(p \| q)$
> **$q$ doit couvrir tous les endroits où $p > 0$.** $q$ peut aussi mettre de la masse là où $p$ n'en a pas, ça ne coûte rien.
> 
> Conséquence : si $p$ a plusieurs modes, $q$ va **s'étaler pour les couvrir tous**, quitte à mettre beaucoup de masse dans les vallées de basse probabilité entre les modes.
> 
> On appelle cette tendance **mode-covering** (aussi **mean-seeking** ou **moment-matching**) : $q$ couvre l'ensemble du support de $p$.

### D. L'image qui résume tout

![[vi_kl_directions.png]]
*Figure. Approximer une posterior bimodale $p$ (bleu) par une gaussienne unimodale $q$ (orange / vert), selon la direction de KL. **Gauche** : $\arg\min_q D_{\text{KL}}(q \| p)$ — mode-seeking. La gaussienne se cale sur **un seul des deux modes**, étroite et centrée. Elle préfère ignorer l'autre mode plutôt que mettre de la masse dans la vallée où $p \approx 0$. **Droite** : $\arg\min_q D_{\text{KL}}(p \| q)$ — mode-covering. La gaussienne **s'étale pour couvrir les deux modes**, large et centrée entre eux, avec beaucoup de masse là où la vraie posterior n'en a pas. Les deux approximations sont "optimales" dans leur sens respectif — mais elles encodent des erreurs très différentes.*

### E. Conséquences pratiques pour VI

VI utilise systématiquement $D_{\text{KL}}(q \| p)$ — pour des raisons de tractabilité (cf. §II.D). Ça implique deux conséquences importantes qu'il faut accepter.

**1. VI peut rater des modes.** Si la vraie posterior est multimodale et que $\mathcal{Q}$ est unimodale (gaussienne, par exemple), VI va se caler sur un mode et ignorer les autres. Pas par défaut de l'optimiseur — par construction de l'objectif. Aucune quantité d'optimisation ne corrigera ça si la famille est unimodale.

**2. VI sous-estime la variance.** Même quand $p$ est unimodale, $q$ optimisée tend à être **plus étroite** que $p$ (parce qu'elle évite les queues où $p$ baisse). Si tu utilises VI pour quantifier de l'incertitude, sache que tu **sous-estimeras** typiquement la variance — c'est un biais systématique connu.

> [!example] Cas concret : posterior bimodale
> Imagine une posterior $p(z \mid x) = 0.5\, \mathcal{N}(-2, 0.5^2) + 0.5\, \mathcal{N}(2, 0.5^2)$ (deux bosses bien séparées). Si on approxime par $q_\phi = \mathcal{N}(\mu, \sigma^2)$ :
> 
> - **VI ($\arg\min \text{KL}(q\|p)$)** donnera $q \approx \mathcal{N}(-2, 0.5^2)$ ou $\mathcal{N}(2, 0.5^2)$ — selon l'initialisation. Un seul mode, l'autre est complètement ignoré. La variance reportée est petite.
> - **MLE / EP ($\arg\min \text{KL}(p\|q)$)** donnerait $q \approx \mathcal{N}(0, 4.25)$ — moyenne entre les deux modes, variance large qui couvre tout le support. Une gaussienne très peu informative.
> 
> Aucune des deux n'est "correcte". Ce sont deux compromis différents, optimaux dans deux sens différents.

### F. Pourquoi VI utilise quand même cette direction

Si $D_{\text{KL}}(q \| p)$ a ces défauts (rater des modes, sous-estimer la variance), pourquoi est-ce le choix standard de VI ?

**Réponse courte** : seul $D_{\text{KL}}(q \| p)$ peut s'écrire comme `cst − ELBO` avec une ELBO **tractable** (cf. §II.D). Inverser la direction donne :

$$D_{\text{KL}}(p \| q) = \int p_\theta(z \mid x) \log \frac{p_\theta(z \mid x)}{q_\phi(z)} \, dz.$$

L'espérance est sous $p_\theta(z \mid x)$ — qu'on ne sait ni évaluer ni échantillonner. Sans posterior, on ne peut pas approximer cette intégrale ni par forme fermée ni par Monte Carlo.

**Bilan** : la direction $\text{KL}(q \| p)$ a des défauts bien connus, mais elle est la **seule tractable** pour le problème qu'on veut résoudre. C'est le compromis fondateur de VI.

> [!note]- Et les autres divergences ?
> Pour échapper aux défauts de $\text{KL}(q \| p)$, on peut utiliser d'autres divergences :
> - **$\alpha$-divergences** (interpolent entre les deux KL).
> - **Stein divergences**, **Wasserstein**.
> - **f-divergences** plus générales.
> 
> Toutes ces alternatives existent mais sont moins standards. Elles compliquent l'optimisation et perdent souvent l'élégance "ELBO = log p − KL" qui rend VI si pratique.

## V. CAVI : Coordinate Ascent Variational Inference

§II nous a dit *quoi* optimiser : l'ELBO sur $\phi$. §III nous a donné une famille concrète : mean-field. **CAVI** (Coordinate Ascent Variational Inference) est l'algorithme historique pour faire cette optimisation **en forme fermée**, sans gradient — l'équivalent VI du M-step de l'EM-GMM (§III.D de `[[05_Expectation_Maximization]]`).

### A. L'idée : optimiser un facteur à la fois

On suppose un $q$ mean-field : $q_\phi(z) = \prod_{i=1}^{d} q_{\phi_i}(z_i)$. L'ELBO est une fonction des $d$ jeux de paramètres $\phi_1, \ldots, \phi_d$.

**Stratégie CAVI** : optimiser **un facteur à la fois**, les autres fixés.

- À l'itération $t$, on garde $q_{\phi_2}, \ldots, q_{\phi_d}$ figés et on cherche le meilleur $q_{\phi_1}$.
- Puis on garde $q_{\phi_1}^{(\text{nouveau})}, q_{\phi_3}, \ldots, q_{\phi_d}$ figés et on cherche le meilleur $q_{\phi_2}$.
- Etc., en cycle.

C'est exactement la philosophie du **coordinate ascent** : au lieu d'optimiser sur tout l'espace $\phi$ d'un coup, on optimise sur un sous-espace à la fois.

### B. La formule générale du CAVI

L'update optimal d'un facteur $q_{\phi_i}(z_i)$ (avec les autres facteurs fixés) a une **forme fermée explicite** :

$$\boxed{\;\log q_{\phi_i}^*(z_i) = \mathbb{E}_{q_{-i}}\!\big[\log p_\theta(x, z)\big] + \text{cst}\;}$$

où $\mathbb{E}_{q_{-i}}[\cdot]$ désigne l'espérance par rapport à **tous les autres facteurs** $q_{\phi_j}$ ($j \neq i$), et la constante normalise pour que $q_{\phi_i}^*$ intègre à 1.

> [!warning] Lecture de la formule
> Le facteur optimal $q_{\phi_i}^*(z_i)$ est obtenu en :
> 1. Prenant le log de la jointe $\log p_\theta(x, z)$.
> 2. Prenant l'espérance par rapport à **tous les autres latents** (sous leurs $q_{\phi_j}$ courants).
> 3. Exponentiant et normalisant pour obtenir une distribution.
> 
> C'est exactement ce qui se passe dans EM-GMM : le calcul des responsabilités $\gamma_{nk}$ revient à intégrer la jointe sur tout sauf l'indice $k$.

> [!note]- Preuve de la formule CAVI
> On part de l'ELBO mean-field :
> 
> $$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p_\theta(x, z)] - \mathbb{E}_{q_\phi}\!\left[\sum_j \log q_{\phi_j}(z_j)\right].$$
> 
> On isole les termes dépendant de $q_{\phi_i}$. Comme $q_\phi = \prod_j q_{\phi_j}$, l'espérance se factorise :
> 
> $$\mathcal{L}(\phi) = \mathbb{E}_{q_{\phi_i}}\!\left[\mathbb{E}_{q_{-i}}[\log p_\theta(x, z)]\right] - \mathbb{E}_{q_{\phi_i}}[\log q_{\phi_i}(z_i)] + \text{(termes en } q_{-i} \text{ seulement)}.$$
> 
> Définissons $\tilde{p}(z_i) := \exp(\mathbb{E}_{q_{-i}}[\log p_\theta(x, z)])$ (à normalisation près). Alors les termes dépendant de $q_{\phi_i}$ s'écrivent :
> 
> $$\mathbb{E}_{q_{\phi_i}}[\log \tilde{p}(z_i)] - \mathbb{E}_{q_{\phi_i}}[\log q_{\phi_i}(z_i)] = -D_{\text{KL}}(q_{\phi_i} \| \tilde{p}) + \text{cst}.$$
> 
> Maximiser sur $q_{\phi_i}$ revient à **minimiser la KL** entre $q_{\phi_i}$ et $\tilde{p}$. L'optimum est $q_{\phi_i}^* = \tilde{p}$, c'est-à-dire :
> 
> $$\log q_{\phi_i}^*(z_i) = \mathbb{E}_{q_{-i}}[\log p_\theta(x, z)] + \text{cst}. \;\blacksquare$$

### C. L'algorithme

> [!warning] CAVI en toute généralité
> Soit un modèle $p_\theta(x, z) = p_\theta(x, z_1, \ldots, z_d)$ avec un $q$ mean-field $q_\phi(z) = \prod_i q_{\phi_i}(z_i)$.
> 
> 1. **Initialiser** chaque $q_{\phi_i}^{(0)}$.
> 2. **Itérer** jusqu'à convergence :
>    Pour chaque $i = 1, \ldots, d$ :
>    $q_{\phi_i}^{(t+1)}(z_i) \;\propto\; \exp\!\left(\mathbb{E}_{q_{-i}^{(t)}}[\log p_\theta(x, z)]\right).$
> 3. **Convergence** : ELBO stagne → stop.
> 
> Comme EM, CAVI est une **ascent monotone** sur l'ELBO. La preuve se calque sur celle de la monotonie EM : chaque update améliore l'ELBO (ou la garde constante à convergence).

![[cavi_iterations.png]]
*Figure. CAVI en action sur une posterior gaussienne 2D corrélée (même setup que `vi_mean_field_factorization.png`). On part d'une init mal placée ($q$ centré en $(-2.5, 3.5)$ alors que $p$ est centrée en $(1.5, 1.0)$). À chaque itération, **un seul facteur** est mis à jour ($q_1$ puis $q_2$, en alternance). Le centre de $q$ se rapproche du centre de $p$ et la KL décroît monotonement. **À convergence**, $q$ est centré sur le bon endroit et a les bonnes marginales — mais reste **axis-aligned**, donc la corrélation de $p$ est perdue (gap KL résiduel, cf. §III.D).*

### D. Quand la forme fermée existe : modèles conjugués

CAVI brille quand la **jointe $\log p_\theta(x, z)$ est dans une famille exponentielle conjuguée** avec chaque facteur $q_{\phi_i}$. Dans ce cas, l'espérance $\mathbb{E}_{q_{-i}}[\log p_\theta(x, z)]$ peut être calculée analytiquement, et $\exp$ de cette espérance retombe sur la même famille que $q_{\phi_i}$ — donc on peut juste extraire les paramètres mis à jour.

**Exemple canonique : Bayesian GMM.** Avec des priors conjugués (Dirichlet sur $\pi$, Normal-Wishart sur $(\mu_k, \Sigma_k)$), CAVI donne des updates analytiques pour chaque $q_{\phi_i}$ — exactement comme EM-GMM mais en cadre bayésien. Les détails sont techniques (cf. Bishop chap. 10) mais le squelette de calcul est identique à ce qu'on a vu en §III.G de `[[05_Expectation_Maximization]]`.

> [!example] Mini-exemple : modèle simple à 2 latents
> Soit $p(z_1, z_2 \mid x)$ une posterior gaussienne 2D. Supposons un $q = q_1(z_1) \cdot q_2(z_2)$ mean-field gaussien.
> 
> L'update CAVI de $q_1$ donne (à constante près) :
> $$\log q_1^*(z_1) = \mathbb{E}_{q_2}[\log p(z_1, z_2 \mid x)] + \text{cst}.$$
> 
> Pour une posterior gaussienne, cette espérance se calcule en forme fermée et donne **une gaussienne** dont la moyenne et la variance dépendent de $\mathbb{E}_{q_2}[z_2]$ et $\text{Var}_{q_2}[z_2]$.
> 
> Puis update de $q_2$ symétriquement, qui dépend de $\mathbb{E}_{q_1}[z_1]$ et $\text{Var}_{q_1}[z_1]$.
> 
> On itère : $q_1, q_2, q_1, q_2, \ldots$. À convergence, $q_1^* \cdot q_2^*$ est la meilleure approximation mean-field gaussienne de $p$ — précisément celle de la figure de §III.D (ellipse axis-aligned).

### E. Liens et limites

**CAVI vs EM.** Vu de haut, CAVI **est** une généralisation d'EM :
- EM-GMM : $q$ est libre, on prend $q = p_\theta(z \mid x)$ exact (forme fermée du E-step).
- CAVI : $q$ est mean-field, on optimise un facteur à la fois.

Dans le cas où la posterior est tractable ET factorise naturellement par les latents, CAVI **redonne EM** — c'est le même algorithme.

**Limites de CAVI.** Trois cas où ça ne marche pas :
1. **Pas conjugué** : l'espérance $\mathbb{E}_{q_{-i}}[\log p_\theta(x, z)]$ n'a pas de forme fermée. On doit passer au gradient (§VI).
2. **Datasets massifs** : chaque itération CAVI revisite toutes les données pour calculer l'update. Pour $N$ très grand, c'est trop cher → SVI (§VI.A).
3. **Mean-field trop pauvre** : si la posterior a des corrélations fortes, CAVI converge vers une mauvaise approximation (cf. §III.D).

## VI. Quand CAVI ne suffit pas : SVI et BBVI

CAVI exige (1) conjugaison pour les forme fermées, (2) un pass sur tout le dataset à chaque itération. Pour les modèles modernes (haute dimension, datasets massifs, formes non-conjuguées), il faut des méthodes plus flexibles. Deux extensions principales : **Stochastic VI** (SVI) et **Black-Box VI** (BBVI).

### A. Stochastic VI (SVI) — passer à l'échelle

**Problème de CAVI.** À chaque itération, un update de $q_{\phi_i}$ demande de sommer/intégrer sur **tout le dataset** $\{x_1, \ldots, x_N\}$. Pour $N = 10^6$, c'est inacceptable.

**Solution.** Remplacer l'update exact par un **update stochastique** basé sur un mini-batch. À chaque itération :

1. Tirer un mini-batch $\mathcal{B} \subset \{1, \ldots, N\}$ de taille $|\mathcal{B}| \ll N$.
2. Calculer un **estimateur du gradient naturel** de l'ELBO sur ce mini-batch.
3. Mettre à jour $\phi$ par un pas de gradient (avec learning rate qui décroît).

C'est l'**équivalent VI de la SGD** : on troque exactitude contre passage à l'échelle. Convergence garantie (avec learning rate Robbins-Monro) mais plus lente en nombre d'itérations — compensé par un coût par itération bien plus faible.

> [!note] Pourquoi "gradient naturel" ?
> SVI utilise typiquement le **gradient naturel** (corrigé par la Fisher information) plutôt que le gradient euclidien. Pour les familles exponentielles, le gradient naturel a une forme particulièrement simple : c'est juste la différence entre les paramètres naturels courants et l'optimum CAVI. SVI = pas de gradient naturel + mini-batch.

### B. Black-Box VI (BBVI) — quand rien n'est conjugué

**Problème.** Si le modèle n'est pas conjugué, ni CAVI ni les formes fermées ne marchent. On a besoin d'une méthode qui **ne fait aucune hypothèse** sur la structure du modèle, juste sur la capacité d'évaluer $\log p_\theta(x, z)$ et $\log q_\phi(z)$ point par point.

**Idée BBVI.** Maximiser l'ELBO **par gradient stochastique**, en estimant les gradients par Monte Carlo. L'ELBO s'écrit :

$$\mathcal{L}(\phi, \theta) = \mathbb{E}_{z \sim q_\phi}[\log p_\theta(x, z) - \log q_\phi(z)].$$

Pour calculer $\nabla_\phi \mathcal{L}$, deux estimateurs principaux :

**1. Score function estimator** (REINFORCE). Marche pour **n'importe quel** $q_\phi$ :

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{q_\phi}\!\big[(\log p_\theta(x, z) - \log q_\phi(z)) \cdot \nabla_\phi \log q_\phi(z)\big].$$

On estime cette espérance par Monte Carlo : tirer $z^{(s)} \sim q_\phi$, calculer le terme entre crochets, moyenner. **Avantage** : aucune contrainte sur $q_\phi$ — ça marche pour des distributions discrètes, mixtes, etc. **Inconvénient** : variance énorme du gradient → convergence lente, beaucoup d'échantillons nécessaires.

**2. Reparametrization trick.** Si $q_\phi$ est continue et reparamétrisable (gaussienne par exemple), on peut écrire $z = g_\phi(\epsilon)$ avec $\epsilon \sim p(\epsilon)$ indépendant de $\phi$ (par ex. $\epsilon \sim \mathcal{N}(0, I)$ et $z = \mu_\phi + \sigma_\phi \epsilon$). Alors :

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{\epsilon}\!\big[\nabla_\phi (\log p_\theta(x, g_\phi(\epsilon)) - \log q_\phi(g_\phi(\epsilon)))\big].$$

**Avantage** : variance bien plus basse que score function. **Inconvénient** : ne marche que si $q_\phi$ est reparamétrisable (typiquement continue, à support fixé).

> [!note] Lien avec VAE
> La reparametrization trick est ce qui rend les VAE entraînables par backpropagation : le gradient $\nabla_\phi$ passe à travers l'échantillonnage $z = g_\phi(\epsilon)$ comme si c'était une opération déterministe. Sans cette astuce, on serait coincé avec score function et son bruit de gradient ingérable. Voir `[[02_VAE]]` pour les détails.

### C. Récap des méthodes d'optimisation VI

| Méthode | Quand l'utiliser | Coût par itération | Convergence |
|---|---|---|---|
| **CAVI** | Modèle conjugué + petit dataset | $\mathcal{O}(N)$ | Rapide (forme fermée) |
| **SVI** | Modèle conjugué + dataset massif | $\mathcal{O}(\|\mathcal{B}\|)$ | Plus lente mais scalable |
| **BBVI (score function)** | Modèle quelconque, $q$ quelconque | $\mathcal{O}(S \cdot N)$ avec $S$ échantillons | Lente (haute variance) |
| **BBVI (reparametrization)** | Modèle quelconque, $q$ continue reparamétrisable | $\mathcal{O}(S \cdot N)$ | Rapide (basse variance) |

Quel que soit le choix, le **principe reste le même** : maximiser l'ELBO sur $\phi$. Ce qui change, c'est juste **comment** on calcule (ou approxime) les updates.

## VII. Récap et place dans la généalogie

### A. Ce qu'on a vu

Le fil de la note :

1. **§I — Le problème** : la posterior $p_\theta(z \mid x)$ est intractable dès que $z$ est continu en haute dimension ou non-conjugué.
2. **§II — L'astuce centrale** : minimiser $\text{KL}(q_\phi \| p)$ = maximiser l'ELBO. Et l'ELBO, contrairement à la KL, est calculable.
3. **§III — Choisir la famille** : trade-off expressivité vs tractabilité. Mean-field, gaussienne paramétrique, etc.
4. **§IV — Direction de la KL** : on utilise $\text{KL}(q \| p)$ (la seule tractable), au prix d'un biais mode-seeking et de variance sous-estimée.
5. **§V — CAVI** : optimisation en forme fermée pour mean-field conjugué.
6. **§VI — SVI / BBVI** : extensions pour datasets massifs et modèles non-conjugués.

### B. Place dans la généalogie

> [!note] La grande chaîne
> $$\underbrace{\text{EM exact}}_{\substack{p_\theta(z \mid x) \text{ tractable} \\ q \text{ libre}}} \;\xrightarrow{\text{posterior intractable}}\; \underbrace{\text{VI}}_{\substack{q_\phi \in \mathcal{Q} \\ \text{CAVI / SVI / BBVI}}} \;\xrightarrow{\text{un } q_\phi \text{ par observation}}\; \underbrace{\text{VAE amortizé}}_{q_\phi(z \mid x) \text{ encodeur NN}}$$

Chaque pas généralise le précédent :
- EM → VI : on accepte une approximation parce que la posterior n'est plus calculable.
- VI → VAE : on amortit le calcul de $\phi$ par un réseau de neurones, permettant de scaler à des datasets massifs avec $z$ haute dimension.

### C. Ce qui change vs EM, en une phrase

**EM** maintenait $q$ libre et faisait des **E-step exacts** (posterior calculable) suivis de M-step exacts. **VI** contraint $q$ dans une famille $\mathcal{Q}$ et fait des "E-step approximatifs" (optimisation sur $\phi$, qui ne donne *jamais* la vraie posterior mais sa meilleure approximation dans $\mathcal{Q}$). Le M-step (si on apprend aussi $\theta$) reste structurellement identique : maximiser l'ELBO sur $\theta$.

C'est *la* différence conceptuelle qui sépare EM de VI — et qui rend VI applicable bien plus largement.

---

## Pour aller plus loin

- **Bishop, *Pattern Recognition and Machine Learning*, chapitre 10.** La référence canonique pour CAVI et Bayesian GMM/PCA.
- **Blei, Kucukelbir, McAuliffe (2017).** *Variational Inference: A Review for Statisticians.* L'article de revue moderne, lecture indispensable.
- **Ranganath, Gerrish, Blei (2014).** *Black-Box Variational Inference.* L'article fondateur de BBVI.
- **Hoffman, Blei, Wang, Paisley (2013).** *Stochastic Variational Inference.* L'article fondateur de SVI.
- **Kingma, Welling (2013).** *Auto-Encoding Variational Bayes.* L'article VAE — application directe de tout ce qu'on a vu ici, avec la reparametrization trick.
- **Murphy, *Probabilistic Machine Learning: Advanced Topics*, chapitre 10.** Vue moderne très claire.
