---
title: Modèles Bayésiens
---
# Modèles Bayésiens

> Ce fichier construit les **modèles bayésiens classiques** à partir des fondations vues dans [[Inférence Bayésienne]] et des techniques de simulation de [[MCMC]]. On commence par formaliser ce qu'est un modèle hiérarchique (forme et représentation graphique), puis on traite les trois régressions bayésiennes (linéaire, logistique, Poisson) et on finit par les modèles hiérarchiques propres — *partial pooling*, random intercept et leurs effets de *shrinkage*.

## I. Modélisation bayésienne — concepts généraux

### A. Forme hiérarchique

Un **modèle bayésien** se spécifie en couches, du bas vers le haut : on commence par décrire la **vraisemblance** (comment les données sont générées à partir des paramètres), puis on monte vers les **priors** (comment les paramètres sont générés). Un modèle simple :

$$
\begin{aligned}
y_i \mid \mu, \sigma^2 &\overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2), \quad i = 1, \dots, n \\
\mu &\sim \mathcal{N}(\mu_0, \sigma_0^2) \\
\sigma^2 &\sim \mathrm{IG}(\nu_0, \beta_0)
\end{aligned}
$$
Lecture : les données viennent d'une normale dont les paramètres sont eux-mêmes aléatoires, avec des distributions connues a priori. C'est la **forme hiérarchique** : chaque couche dépend de la suivante. Cette structure par niveaux est ce qui distingue le bayésien du fréquentiste — ce dernier traite $\mu$ et $\sigma^2$ comme des constantes fixées à estimer.

On peut compliquer en faisant dépendre un prior d'un autre paramètre :

$$
\begin{aligned}
y_i \mid \mu, \sigma^2 &\overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2) \\
\mu \mid \sigma^2 &\sim \mathcal{N}\!\left(\mu_0, \frac{\sigma^2}{\omega_0}\right) \\
\sigma^2 &\sim \mathrm{IG}(\nu_0, \beta_0)
\end{aligned}
$$
Ici $\mu$ dépend de $\sigma^2$ via sa variance — c'est le prior **conjugué conditionnel** vu dans [[Inférence Bayésienne]] section VII.G.

### B. Représentation graphique — DAG et plate notation

Un modèle bayésien se visualise comme un **graphe orienté acyclique** (DAG) où chaque nœud est une variable et chaque arête une dépendance directe. Conventions standards :

- **Cercle simple** : variable aléatoire latente (paramètre).
- **Cercle doublé** ou **grisé** : variable observée (donnée).
- **Carré** : constante ou variable conditionnante (covariable).
- **Flèche** : dépendance directe (le parent générative l'enfant).

**Plate notation.** Quand on a $n$ variables échangeables (même distribution conditionnelle), au lieu de dessiner $n$ nœuds on dessine **un seul représentant** entouré d'un rectangle (la *plate*) annoté "$i = 1, \dots, n$". Ça compacte le schéma sans perdre d'information.

![[bayes_dag_normal_model.png]]
**Figure 1.** DAG du modèle normal $y_i \sim \mathcal{N}(\mu, \sigma^2)$ avec priors indépendants. À gauche : la version dépliée avec les $n$ observations explicites. À droite : la même chose en plate notation, beaucoup plus compacte. Les nœuds grisés sont observés, les blancs sont latents.

### C. Échangeabilité

La plate notation suppose que les observations sont **échangeables** : leur loi conjointe est invariante par permutation des indices. Formellement, $(y_1, \dots, y_n)$ est échangeable si

$$f(y_1, \dots, y_n) = f(y_{\pi(1)}, \dots, y_{\pi(n)})$$
pour toute permutation $\pi$. C'est plus faible que i.i.d. (l'i.i.d. implique l'échangeabilité, l'inverse n'est pas vrai), mais dans la pratique on confond souvent les deux.

> [!note]- Le théorème de de Finetti
> Le **théorème de de Finetti** dit qu'une suite infinie échangeable peut toujours s'écrire comme un mélange : il existe une variable latente $\theta$ telle que les $y_i$ soient i.i.d. *conditionnellement à $\theta$*. C'est la justification mathématique de l'approche bayésienne : dès qu'on accepte l'échangeabilité, on est *forcé* d'introduire une variable latente avec une distribution — exactement ce que fait le bayésien avec le prior.

### D. Posterior conjoint et stratégie d'inférence

Une fois le modèle spécifié, l'objet d'intérêt est le **posterior conjoint** sur tous les paramètres :

$$f(\theta_1, \dots, \theta_d \mid y) \propto \underbrace{f(y \mid \theta_1, \dots, \theta_d)}_{\text{vraisemblance}} \cdot \underbrace{f(\theta_1, \dots, \theta_d)}_{\text{prior conjoint}}$$
Le prior conjoint se factorise selon le DAG — chaque variable est conditionnée sur ses parents. Par exemple pour le modèle à trois couches du A :

$$f(\mu, \sigma^2) = f(\mu \mid \sigma^2) \cdot f(\sigma^2).$$
**Trois scénarios pour exploiter ce posterior :**

1. **Conjugaison** : le posterior est dans une famille standard, on lit ses paramètres directement. C'est le cas pour les régressions linéaire (avec variance connue) et Poisson.
2. **Pas de conjugaison mais semi-tractable** : on peut approximer (ex. Laplace approximation pour la régression logistique).
3. **Pas de conjugaison du tout** : on échantillonne via [[MCMC]]. C'est le cas général pour les modèles hiérarchiques complexes.

Les trois sections suivantes illustrent ces trois régimes.

## II. Régression linéaire bayésienne

### A. Modèle

Données : $\boldsymbol{y} = (y_1, \dots, y_n)^\top$ et matrice de design $\boldsymbol{\Phi} \in \mathbb{R}^{n \times d}$ (lignes = observations, colonnes = features, en incluant l'intercept comme première colonne de 1).

**Vraisemblance** (variance $\sigma^2$ supposée connue pour l'instant) :

$$y_i \mid \boldsymbol{w}, \sigma^2 \overset{ind}{\sim} \mathcal{N}(\boldsymbol{\phi}(\boldsymbol{x}_i)^\top \boldsymbol{w}, \sigma^2)$$
où $\boldsymbol{w} \in \mathbb{R}^d$ est le vecteur de coefficients. Équivalemment en forme matricielle :

$$\boldsymbol{y} \mid \boldsymbol{w}, \sigma^2 \sim \mathcal{N}(\boldsymbol{\Phi} \boldsymbol{w}, \sigma^2 \boldsymbol{I}_n).$$
**Prior conjugué** sur $\boldsymbol{w}$ :

$$\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{m}_0, \boldsymbol{S}_0).$$
Choix typiques : $\boldsymbol{m}_0 = \boldsymbol{0}$ (prior centré sur 0, on n'a pas d'a priori sur le signe des coefficients) et $\boldsymbol{S}_0 = \sigma_w^2 \boldsymbol{I}$ (isotrope, même incertitude sur tous les coefficients).

### B. Posterior — forme fermée

Le Normal-Normal est conjugué (cf. [[Inférence Bayésienne]] section VII.G étendu en multivarié). Le posterior est gaussien :

$$\boxed{\; \boldsymbol{w} \mid \boldsymbol{y}, \boldsymbol{\Phi} \sim \mathcal{N}(\boldsymbol{m}_N, \boldsymbol{S}_N) \;}$$
avec

$$\boldsymbol{S}_N^{-1} = \boldsymbol{S}_0^{-1} + \frac{1}{\sigma^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi}, \qquad \boldsymbol{m}_N = \boldsymbol{S}_N \left( \boldsymbol{S}_0^{-1} \boldsymbol{m}_0 + \frac{1}{\sigma^2} \boldsymbol{\Phi}^\top \boldsymbol{y} \right).$$
> [!note]- Lecture en termes de précisions
> En posant la **matrice de précision** $\boldsymbol{\Lambda} = \boldsymbol{S}^{-1}$ :
>
> $$\boldsymbol{\Lambda}_N = \boldsymbol{\Lambda}_0 + \frac{1}{\sigma^2} \boldsymbol{\Phi}^\top \boldsymbol{\Phi}.$$
>
> Les précisions s'ajoutent (comme dans le cas scalaire). La précision posterior = précision du prior + information apportée par les données ($\boldsymbol{\Phi}^\top \boldsymbol{\Phi} / \sigma^2$ est la matrice d'information de Fisher de la régression).
>
> La moyenne posterior est une moyenne pondérée par les précisions du prior et de l'estimateur des moindres carrés.

**Cas particulier — prior plat ($\boldsymbol{S}_0^{-1} \to 0$).** On retrouve la solution OLS classique :

$$\boldsymbol{m}_N \to (\boldsymbol{\Phi}^\top \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^\top \boldsymbol{y} = \hat{\boldsymbol{w}}^{\text{OLS}}.$$
Le bayésien avec prior plat = le fréquentiste. Mais le bayésien garde en plus la matrice de covariance posterior, donc l'incertitude.

![[bayes_lin_posterior_evolution.png]]
**Figure 2.** Évolution du posterior sur les coefficients $(w_0, w_1)$ d'une régression linéaire simple ($y = w_0 + w_1 x + \varepsilon$, vraies valeurs $w_0 = -0.7$, $w_1 = 0.9$) au fur et à mesure qu'on observe des données. À gauche : prior $\mathcal{N}(\boldsymbol{0}, 0.5 \boldsymbol{I})$, masse étalée. Au milieu : posterior après 1 observation, légèrement contracté. À droite : posterior après 1000 observations, concentré autour des vraies valeurs (point rouge). C'est l'effet d'apprentissage bayésien : plus on observe, plus le posterior se resserre autour de la vérité.

### C. Posterior prédictif

Pour une nouvelle entrée $\boldsymbol{x}_*$, on veut $f(y_* \mid \boldsymbol{x}_*, \boldsymbol{y}, \boldsymbol{\Phi})$. En partant de $y_* = \boldsymbol{\phi}(\boldsymbol{x}_*)^\top \boldsymbol{w} + \varepsilon$ avec $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ et en marginalisant sur $\boldsymbol{w}$ :

$$\boxed{\; y_* \mid \boldsymbol{x}_*, \boldsymbol{y} \sim \mathcal{N}\!\left(\boldsymbol{\phi}(\boldsymbol{x}_*)^\top \boldsymbol{m}_N, \; \boldsymbol{\phi}(\boldsymbol{x}_*)^\top \boldsymbol{S}_N \boldsymbol{\phi}(\boldsymbol{x}_*) + \sigma^2\right) \;}$$
**Décomposition de la variance prédictive :**
- $\boldsymbol{\phi}(\boldsymbol{x}_*)^\top \boldsymbol{S}_N \boldsymbol{\phi}(\boldsymbol{x}_*)$ = **incertitude épistémique** sur les paramètres (diminue quand $n \to \infty$).
- $\sigma^2$ = **incertitude aléatorique** (bruit irréductible des données, ne diminue jamais).

C'est une distinction conceptuelle importante en ML moderne : un bon modèle bayésien quantifie *les deux* sources d'incertitude.

![[bayes_lin_predictive.png|578]]
**Figure 3.** Posterior prédictif d'une régression linéaire bayésienne. La ligne centrale est la moyenne prédictive $\boldsymbol{\phi}(\boldsymbol{x}_*)^\top \boldsymbol{m}_N$. La bande sombre représente l'incertitude épistémique seule (ce que le modèle ne sait pas sur $\boldsymbol{w}$), la bande claire ajoute l'incertitude aléatorique (le bruit $\sigma^2$). Loin des données, l'incertitude épistémique croît — c'est la propriété clé que ne capture pas une régression OLS classique.

### D. Régularisation = MAP avec un prior

C'est le pont conceptuel entre bayésien et ML classique. Le **MAP** (mode du posterior) maximise $\log f(\boldsymbol{w} \mid \boldsymbol{y}) = \log f(\boldsymbol{y} \mid \boldsymbol{w}) + \log f(\boldsymbol{w}) + \text{cst}$ :

$$\hat{\boldsymbol{w}}^{\text{MAP}} = \arg\min_{\boldsymbol{w}} \left[ \frac{1}{2\sigma^2} \|\boldsymbol{y} - \boldsymbol{\Phi}\boldsymbol{w}\|_2^2 - \log f(\boldsymbol{w}) \right].$$
Selon le choix du prior :

| Prior | $-\log f(\boldsymbol{w})$ | Pénalité équivalente | Nom |
|---|---|---|---|
| $\mathcal{N}(\boldsymbol{0}, \sigma_w^2 \boldsymbol{I})$ | $\frac{1}{2\sigma_w^2} \|\boldsymbol{w}\|_2^2$ + cst | $\lambda \|\boldsymbol{w}\|_2^2$ avec $\lambda = \sigma^2 / \sigma_w^2$ | **Ridge / L2** |
| $\mathrm{Laplace}(0, b)$ par coordonnée | $\frac{1}{b} \|\boldsymbol{w}\|_1$ + cst | $\lambda \|\boldsymbol{w}\|_1$ avec $\lambda = \sigma^2 / b$ | **Lasso / L1** |
| Mélange Laplace + Gaussien | combinaison L1 + L2 | Elastic Net | **Elastic Net** |

> 💡 **L'idée en une phrase.** Régulariser un estimateur ML = mettre un prior bayésien centré sur zéro et faire du MAP. La force de la régularisation $\lambda$ correspond à l'inverse de la variance du prior : un prior serré = forte régularisation = on "croit" beaucoup que les coefficients sont petits.

## III. Régression logistique bayésienne

### A. Modèle

**Vraisemblance** Bernoulli avec lien sigmoïde :

$$y_i \mid \boldsymbol{w} \overset{ind}{\sim} \mathrm{Bernoulli}(\sigma(\boldsymbol{\phi}(\boldsymbol{x}_i)^\top \boldsymbol{w})), \qquad \sigma(a) = \frac{1}{1 + e^{-a}}.$$
**Prior gaussien** sur les coefficients :

$$\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{w}_0, \boldsymbol{\Sigma}_0).$$
### B. Pas de conjugaison — d'où vient la difficulté

Le produit Bernoulli $\times$ Gaussien ne donne aucune loi standard. Le posterior est :

$$f(\boldsymbol{w} \mid \boldsymbol{y}, \boldsymbol{\Phi}) \propto \left[\prod_{i=1}^n \sigma(\boldsymbol{\phi}_i^\top \boldsymbol{w})^{y_i} (1 - \sigma(\boldsymbol{\phi}_i^\top \boldsymbol{w}))^{1 - y_i}\right] \cdot \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{w}_0, \boldsymbol{\Sigma}_0).$$
Pas de forme fermée. Trois options :

1. **Laplace approximation** — approximer le posterior par une gaussienne autour du MAP.
2. **MCMC** — échantillonner via [[MCMC]] (Metropolis-Hastings ou HMC).
3. **Variational inference** — approximer par une famille paramétrique tractable.

On détaille la Laplace approximation, la plus simple et historiquement la plus utilisée.

### C. Laplace approximation

**Idée.** Tout posterior unimodal avec mode intérieur ressemble localement à une gaussienne (développement de Taylor d'ordre 2 du log-posterior autour du mode). On approxime donc :

$$f(\boldsymbol{w} \mid \boldsymbol{y}) \approx \mathcal{N}(\boldsymbol{w} \mid \hat{\boldsymbol{w}}^{\text{MAP}}, \boldsymbol{H}^{-1})$$
où $\boldsymbol{H}$ est la **hessienne** du log-posterior négatif évaluée au MAP (la *courbure* du log-posterior à son sommet).

**Étapes pratiques :**

1. **Trouver le MAP** $\hat{\boldsymbol{w}}^{\text{MAP}} = \arg\max_{\boldsymbol{w}} \log f(\boldsymbol{w} \mid \boldsymbol{y})$ par optimisation (Newton-Raphson, gradient avec hessienne).

2. **Calculer la hessienne au MAP.** Pour la régression logistique bayésienne :

$$\boldsymbol{H} = \boldsymbol{\Sigma}_0^{-1} + \sum_{i=1}^n \sigma_i (1 - \sigma_i) \boldsymbol{\phi}_i \boldsymbol{\phi}_i^\top$$
où $\sigma_i = \sigma(\boldsymbol{\phi}_i^\top \hat{\boldsymbol{w}}^{\text{MAP}})$. Lecture : $\boldsymbol{\Sigma}_0^{-1}$ = précision du prior, et le second terme = information apportée par les données au MAP.

3. **Approximation finale :** $\boldsymbol{w} \mid \boldsymbol{y} \sim \mathcal{N}(\hat{\boldsymbol{w}}^{\text{MAP}}, \boldsymbol{H}^{-1})$.

> [!warning] Limites de la Laplace approximation
> - **Capture seulement le voisinage du mode** : si le posterior a des queues lourdes ou est asymétrique, l'approximation est mauvaise loin du mode.
> - **Multi-modal → catastrophe** : la Laplace ne voit qu'un seul mode, elle rate les autres.
> - **Bonne en haute dimension avec beaucoup de données** : le théorème de Bernstein-von Mises garantit la convergence du posterior vers une gaussienne quand $n \to \infty$, donc Laplace devient asymptotiquement exacte.

### D. Posterior prédictif — probit approximation

Prédire la probabilité que $y_* = 1$ pour une nouvelle entrée :

$$\mathbb{P}(y_* = 1 \mid \boldsymbol{x}_*, \boldsymbol{y}) = \int \sigma(\boldsymbol{\phi}_*^\top \boldsymbol{w}) \, f(\boldsymbol{w} \mid \boldsymbol{y}) \, \mathrm{d}\boldsymbol{w}.$$
Même avec l'approximation gaussienne $f(\boldsymbol{w} \mid \boldsymbol{y}) \approx \mathcal{N}(\hat{\boldsymbol{w}}^{\text{MAP}}, \boldsymbol{H}^{-1})$, l'intégrale n'est pas analytique à cause de la sigmoïde. Astuce : **approximer la sigmoïde par une probit (CDF de la gaussienne)** :

$$\sigma(a) \approx \Phi(\lambda a), \quad \lambda^2 = \pi/8.$$
La convolution probit $\times$ gaussienne *est* analytique. Après recalibrage, on obtient :

$$\mathbb{P}(y_* = 1 \mid \boldsymbol{x}_*, \boldsymbol{y}) \approx \sigma\!\left( \kappa(\sigma_a^2) \, \mu_a \right)$$
avec

$$\mu_a = \boldsymbol{\phi}_*^\top \hat{\boldsymbol{w}}^{\text{MAP}}, \quad \sigma_a^2 = \boldsymbol{\phi}_*^\top \boldsymbol{H}^{-1} \boldsymbol{\phi}_*, \quad \kappa(\sigma_a^2) = (1 + \pi \sigma_a^2 / 8)^{-1/2}.$$
**Lecture.** $\mu_a$ est la prédiction "point estimate" qu'on aurait avec une régression logistique fréquentiste (le score linéaire au MAP). $\sigma_a^2$ est l'incertitude posée par le posterior sur $\boldsymbol{w}$. Le facteur $\kappa < 1$ **atténue** la prédiction quand l'incertitude est grande — c'est l'effet bayésien : *plus on est incertain sur les paramètres, plus on est conservateur sur la probabilité finale*.

## IV. Régression Poisson bayésienne

Modèle pour des données de **comptage** : nombre d'occurrences d'un événement par unité (clients par jour, défauts de paiement par mois, etc.).

### A. Modèle

**Vraisemblance** Poisson :

$$Y_i \mid \lambda \overset{iid}{\sim} \mathrm{Poisson}(\lambda), \quad i = 1, \dots, n.$$
(Variante "régression" : $\lambda_i = \exp(\boldsymbol{\phi}_i^\top \boldsymbol{w})$ pour incorporer des covariables. Le cas simple sans covariable suffit pour fixer les idées.)

**Prior conjugué** Gamma sur $\lambda > 0$ :

$$\lambda \sim \mathrm{Gamma}(\alpha, \beta).$$
### B. Posterior — forme fermée

La Gamma est conjuguée à la Poisson (cf. [[Inférence Bayésienne]] section VII.E). Le posterior est :

$$\boxed{\; \lambda \mid \boldsymbol{y} \sim \mathrm{Gamma}\!\left(\alpha + \sum_{i=1}^n y_i, \; \beta + n\right) \;}$$
**Lecture des paramètres :**
- $\alpha + \sum y_i$ = $\alpha$ ("comptes" virtuels du prior) + $\sum y_i$ (comptes observés).
- $\beta + n$ = $\beta$ ("unités d'observation" virtuelles du prior) + $n$ (unités observées).

La taille d'échantillon effective du prior est $\beta$.

### C. Posterior prédictif — Negative Binomial

Prédire un nouveau comptage $y_*$ :

$$f(y_* \mid \boldsymbol{y}) = \int f(y_* \mid \lambda) \, f(\lambda \mid \boldsymbol{y}) \, \mathrm{d}\lambda.$$
C'est une intégrale Poisson $\times$ Gamma, qui se calcule analytiquement et donne une **Negative Binomial** :

$$y_* \mid \boldsymbol{y} \sim \mathrm{NegBin}\!\left(\alpha + \sum y_i, \; \beta + n\right).$$
> [!note]- Pourquoi Negative Binomial ?
> La NegBin est exactement le mélange Poisson-Gamma : si $\Lambda \sim \mathrm{Gamma}(r, p/(1-p))$ et $Y \mid \Lambda \sim \mathrm{Poisson}(\Lambda)$, alors $Y \sim \mathrm{NegBin}(r, p)$. C'est l'analogue discret du Student (qui est le mélange Normal-InverseGamma).
>
> **Conséquence pratique** : le prédictif bayésien Poisson est *plus dispersé* qu'un Poisson de moyenne $\hat\lambda$ (la NegBin a une variance > moyenne, contrairement à la Poisson où variance = moyenne). C'est l'**overdispersion bayésienne** : on prend en compte l'incertitude sur $\lambda$ et donc on augmente la variance prédictive.

### D. Stratégies pour fixer le prior Gamma

Deux philosophies pour choisir $(\alpha, \beta)$ :

1. **Prior informatif.** On a une croyance sur la moyenne et l'incertitude :
   - Moyenne prior : $\alpha / \beta$.
   - Écart-type prior : $\sqrt{\alpha} / \beta$.
   - On résout pour $(\alpha, \beta)$ à partir de ces deux quantités.

2. **Prior vague.** $\mathrm{Gamma}(\varepsilon, \varepsilon)$ avec $\varepsilon$ petit. Moyenne prior = 1, mais variance énorme. Le posterior est alors quasi entièrement dominé par les données :

$$\mathbb{E}[\lambda \mid \boldsymbol{y}] = \frac{\varepsilon + \sum y_i}{\varepsilon + n} \approx \frac{\sum y_i}{n} = \bar{y}.$$
Qui est exactement l'estimateur du maximum de vraisemblance fréquentiste.

## V. Modèles hiérarchiques

### A. Motivation — données groupées

Jusqu'ici on a supposé les données i.i.d. En réalité, beaucoup de jeux de données ont une **structure de groupes naturelle** : observations par région, par client, par période, par usine. Les observations d'un même groupe sont plus similaires entre elles qu'avec celles d'autres groupes.

**Trois façons de modéliser :**

1. **Pooling complet** : on ignore les groupes, on traite tout comme i.i.d. avec un seul $\theta$ global. **Risque** : on rate les différences entre groupes.
2. **Aucun pooling** : un $\theta_g$ par groupe, indépendamment. **Risque** : variance énorme pour les groupes avec peu de données, on ignore l'info commune entre groupes.
3. **Partial pooling** (modèle hiérarchique) : un $\theta_g$ par groupe, mais les $\theta_g$ sont eux-mêmes tirés d'une distribution commune. C'est le compromis bayésien.

![[bayes_pooling_comparison.png]]
**Figure 4.** Trois stratégies sur des données groupées (5 groupes, taille variable). Pooling complet (gauche) : tous les groupes ont la même estimation, on lisse les différences réelles. Aucun pooling (milieu) : chaque groupe a son estimation, mais les groupes avec peu de données ont une variance énorme. Partial pooling (droite) : les estimations par groupe sont "tirées" vers la moyenne globale, d'autant plus que le groupe a peu de données. C'est le **shrinkage** bayésien.

### B. Le modèle hiérarchique Poisson-Gamma

Exemple canonique : **biscuits aux pépites de chocolat**. On produit 150 biscuits répartis entre 5 usines (30 par usine). Le nombre de pépites par biscuit suit (approximativement) une Poisson, mais la moyenne $\lambda$ peut différer entre usines.

**Modèle hiérarchique à trois niveaux :**

$$
\begin{aligned}
y_i \mid \ell_i, \lambda_{\ell_i} &\overset{ind}{\sim} \mathrm{Poisson}(\lambda_{\ell_i}), \quad \ell_i \in \{1, \dots, 5\}, \; i = 1, \dots, 150 \\
\lambda_\ell \mid \alpha, \beta &\overset{iid}{\sim} \mathrm{Gamma}(\alpha, \beta), \quad \ell = 1, \dots, 5 \\
\alpha &\sim p(\alpha), \quad \beta \sim p(\beta)
\end{aligned}
$$
Lectures niveau par niveau :
- **Niveau 1 (likelihood)** : chaque biscuit $i$ vient d'une Poisson dont la moyenne dépend de l'usine $\ell_i$.
- **Niveau 2** : les 5 moyennes par usine $\lambda_1, \dots, \lambda_5$ sont elles-mêmes tirées d'une Gamma commune. C'est ce qui *lie* les usines entre elles.
- **Niveau 3 (hyperpriors)** : $\alpha, \beta$ ont leurs propres priors.

**Inférence.** Le posterior conjoint $f(\lambda_1, \dots, \lambda_5, \alpha, \beta \mid \boldsymbol{y})$ n'a pas de forme fermée (mélange Gamma + hyperpriors complexes). On utilise [[MCMC]], typiquement Gibbs sampling parce que les conditionnelles complètes sont presque toutes dans des familles connues.

![[bayes_hierarchical_dag.png|296]]
**Figure 5.** DAG du modèle hiérarchique cookies. Les hyperparamètres $(\alpha, \beta)$ contrôlent la distribution des moyennes par usine $\lambda_\ell$, qui contrôlent à leur tour les comptes observés $y_i$. La plate interne regroupe les 30 biscuits par usine, la plate externe regroupe les 5 usines. Les flèches descendantes encodent la structure générative.

### C. Random intercept en régression linéaire

Même idée appliquée à la régression. **Exemple : mortalité infantile par pays**, regroupés par région ($r \in \{1, \dots, R\}$).

Modèle **non-hiérarchique** classique :

$$y_i \mid \boldsymbol{x}_i, \boldsymbol{\beta}, \sigma^2 \overset{ind}{\sim} \mathcal{N}(\beta_0 + \beta_1 x_{1i} + \beta_2 x_{2i}, \sigma^2).$$
**Random intercept model.** On laisse l'intercept varier par région :

$$
\begin{aligned}
y_i \mid r_i, \boldsymbol{x}_i, \alpha, \boldsymbol{\beta}, \sigma^2 &\overset{ind}{\sim} \mathcal{N}(\alpha_{r_i} + \beta_1 x_{1i} + \beta_2 x_{2i}, \sigma^2) \\
\alpha_r \mid \mu, \tau^2 &\overset{iid}{\sim} \mathcal{N}(\mu, \tau^2), \quad r = 1, \dots, R \\
\mu &\sim p(\mu), \quad \tau^2 \sim p(\tau^2), \quad \boldsymbol{\beta} \sim p(\boldsymbol{\beta}), \quad \sigma^2 \sim p(\sigma^2)
\end{aligned}
$$
**Lecture des paramètres au niveau 2 :**
- $\mu$ = moyenne globale des intercepts (l'intercept "typique" toutes régions confondues).
- $\tau^2$ = variance des intercepts entre régions. Si $\tau^2 \to 0$, toutes les régions ont le même intercept (= pooling complet). Si $\tau^2 \to \infty$, chaque région est indépendante (= aucun pooling).

**$\tau^2$ est appris depuis les données** : c'est le bayésien qui décide du degré de pooling, pas l'analyste.

> [!note]- Extensions courantes du random intercept
> - **Random slopes** : laisser aussi $\beta_1$ varier par groupe (les pentes diffèrent en plus des intercepts).
> - **Multi-niveaux** : structure imbriquée (élèves dans écoles dans districts).
> - **Crossés** : un même individu appartient à plusieurs groupes simultanément (ex. étudiant $\times$ professeur).
>
> Ces modèles sont la spécialité des packages comme `lme4` (fréquentiste, même si moins flexible) et `Stan` / `PyMC` / `brms` (bayésien complet).

### D. Shrinkage — le bénéfice clé du partial pooling

Le **shrinkage** est l'effet par lequel les estimations par groupe sont "tirées" vers la moyenne globale. Mathématiquement, dans le modèle hiérarchique gaussien, l'estimation posterior pour le groupe $g$ s'écrit :

$$\hat{\alpha}_g^{\text{post}} = w_g \cdot \bar{y}_g + (1 - w_g) \cdot \mu, \quad w_g = \frac{n_g / \sigma^2}{n_g / \sigma^2 + 1/\tau^2}$$
où $\bar{y}_g$ est la moyenne empirique du groupe $g$ et $\mu$ la moyenne globale.

**Lecture :**
- **Groupe avec beaucoup de données** ($n_g$ grand) → $w_g \to 1$ → estimation = moyenne empirique du groupe (peu de shrinkage).
- **Groupe avec peu de données** ($n_g$ petit) → $w_g \to 0$ → estimation tirée vers $\mu$ (fort shrinkage).

> 💡 **L'idée en une phrase.** Le modèle hiérarchique laisse les groupes s'**emprunter de la force** ("borrowing strength") : les groupes avec peu de données bénéficient de l'info des autres groupes via la distribution commune. C'est ce qui rend les modèles hiérarchiques bien plus robustes que des régressions séparées.

**Application en finance.** Pour estimer le risque de défaut par segment de clients (jeunes/seniors, par CSP, par région), un modèle hiérarchique stabilise les estimations pour les segments rares en les tirant vers le comportement global, au lieu de leur donner une variance énorme à cause du faible $n_g$.

## VI. À retenir

> [!summary] Les idées centrales
> 1. **Modèle bayésien = forme hiérarchique.** Spécifier la vraisemblance, puis les priors par couche. La structure se visualise comme un DAG avec plate notation pour les variables échangeables.
> 2. **Régression linéaire bayésienne** : conjuguée (Normal-Normal), posterior gaussien en forme fermée. Le prédictif décompose l'incertitude en épistémique (sur $\boldsymbol{w}$) et aléatorique (sur $\sigma^2$).
> 3. **Régularisation = MAP avec un prior.** Ridge $\leftrightarrow$ Gaussien, Lasso $\leftrightarrow$ Laplace. Le bayésien donne une *justification générative* aux pénalités.
> 4. **Régression logistique bayésienne** : pas de conjugaison → Laplace approximation (gaussienne autour du MAP). Prédictif via probit approximation, qui atténue les prédictions selon l'incertitude posterior.
> 5. **Régression Poisson bayésienne** : conjuguée (Gamma-Poisson), posterior et prédictif (Negative Binomial) en forme fermée. Le prédictif bayésien capture l'overdispersion qu'une Poisson MLE rate.
> 6. **Modèles hiérarchiques = partial pooling.** On laisse les paramètres par groupe varier autour d'une moyenne commune apprise depuis les données. Effet de **shrinkage** : les groupes avec peu de données sont stabilisés par les autres.
> 7. **Inférence en pratique** : conjugué → forme fermée ; quasi-conjugué → Laplace ; cas général (notamment hiérarchiques) → [[MCMC]] (Gibbs si conditionnelles standard, sinon NUTS via Stan/PyMC).
