---
title: Modèle linéaire généralisé
order: 0
---

# Modèle linéaire généralisé (GLM)

> Cette note est la **note chapeau** du dossier *Generalized Linear Models*. Elle présente le cadre unifié dont la régression linéaire et la régression logistique sont des cas particuliers, ainsi que d'autres modèles utiles (Poisson, Gamma...). L'objectif n'est pas de redériver toute la machinerie théorique mais de donner la **vue d'ensemble** et le **vocabulaire** nécessaires pour comprendre comment chaque modèle de régression s'inscrit dans le même framework.

## I. Motivation — pourquoi a-t-on besoin de GLM ?

Le modèle linéaire gaussien (cf. [[01_Régression Linéaire Multiple]]) repose sur **deux hypothèses fortes** :

> [!warning] Les deux hypothèses du modèle linéaire gaussien
> 1. **Random component** : $Y \mid X = x \sim \mathcal{N}(\mu(x), \sigma^2)$ — la cible est continue et gaussienne
> 2. **Regression function** : $\mu(x) = x^T \beta$ — la moyenne est *directement* une combinaison linéaire des prédicteurs

Ces deux hypothèses sont restrictives et **cassent dès que $Y$ n'est plus continue gaussienne**.

#### Trois exemples concrets de violation

> [!example] Cas 1 — $Y$ binaire (classification, présence/absence)
> *Exemple : prédire si un patient présente une déformation post-opératoire (oui/non), si un email est un spam, si un client va faire défaut.*
> 
> $Y \in \{0, 1\}$ donc $\mu(x) = E[Y \mid X = x] \in [0, 1]$ (c'est une probabilité).
> 
> **Problème** : on ne peut pas écrire $\mu(x) = x^T \beta$ parce que le membre de droite parcourt tout $\mathbb{R}$, alors que le membre de gauche est borné dans $[0, 1]$. Il faut une **transformation** pour faire le pont entre les deux.

> [!example] Cas 2 — $Y$ comptage (Poisson)
> *Exemple : nombre d'événements (clics sur une pub, défauts d'un système, accidents par jour, claims d'assurance par mois).*
> 
> $Y \in \mathbb{N} = \{0, 1, 2, \ldots\}$ donc $\mu(x) > 0$ (la moyenne d'un comptage est positive).
> 
> **Problème** : $x^T \beta$ peut prendre des valeurs négatives, ce qui n'a pas de sens pour un comptage.

> [!example] Cas 3 — $Y$ positif continu (durées, montants)
> *Exemple : durées entre événements, taille des sinistres en assurance, temps avant défaut, prix.*
> 
> $Y > 0$ donc $\mu(x) > 0$.
> 
> **Problème** : même problème qu'en Poisson, $x^T \beta$ peut être négatif.

> 💡 **L'idée centrale du GLM.** Au lieu d'imposer $\mu(x) = x^T \beta$ directement, on autorise une **transformation** $g$ entre $\mu(x)$ et $x^T \beta$ : $g(\mu(x)) = x^T \beta$. La transformation $g$ s'appelle **link function** (fonction de lien) et permet de faire vivre $\mu(x)$ dans son domaine naturel ($[0,1]$, $\mathbb{R}_+$, etc.) tout en gardant un prédicteur linéaire.

## II. Les trois ingrédients d'un GLM

> [!warning] Définition d'un GLM
> Un **Generalized Linear Model** (GLM) est défini par **trois composants** :
> 
> 1. **Random component** — la loi conditionnelle de $Y$ étant donné $X$ :
> 
> $$Y \mid X = x \sim \text{distribution dans la famille exponentielle}$$
> 
> (par exemple : Gaussienne, Bernoulli, Poisson, Gamma...)
> 
> 2. **Linear predictor** — toujours linéaire, comme en régression classique :
> 
> $$\eta(x) = x^T \beta$$
> 
> 3. **Link function** $g$ — relie le linear predictor à la moyenne :
> 
> $$g(\mu(x)) = \eta(x) = x^T \beta \qquad \Longleftrightarrow \qquad \mu(x) = g^{-1}(x^T \beta)$$

> 💡 **La régression linéaire classique est un GLM.** C'est le cas particulier où la random component est gaussienne et la link function est l'identité ($g(\mu) = \mu$). On retrouve $\mu(x) = x^T \beta$. Tout le framework GLM est construit pour **étendre** ce cas, pas le remplacer.

#### Schéma d'inférence général

Pour un GLM, le schéma est toujours le même :

1. On choisit une **distribution** pour $Y$ adaptée à la nature de la cible (binaire → Bernoulli, comptage → Poisson, etc.)
2. On choisit une **link function** $g$ qui projette $\mu$ dans le bon domaine
3. On estime $\beta$ par **maximum de vraisemblance** (MLE)
4. On évalue la qualité du fit (deviance, pseudo-R², AUC selon le cas) et on fait des tests d'inférence (Wald, LRT)

## III. Exemple — la prédation prédateur/proie

Pour fixer les idées, voici un exemple concret de comment GLM élargit le champ d'application au-delà du linéaire classique.

**Le contexte.** On modélise le nombre $Y$ de proies qu'un prédateur (par exemple un faucon) capture par jour, en fonction du nombre $X$ de proies disponibles dans son territoire.

**Random component.** $Y$ est un comptage, et empiriquement on observe que sa variance est approximativement égale à sa moyenne — signature d'une loi de Poisson :

$$Y \mid X = x \sim \text{Poisson}(\mu(x))$$

**Regression function.** On suppose une relation **non-linéaire** de type Michaelis-Menten (saturation) :

$$\mu(x) = \frac{m \cdot x}{h + x}$$

où $m$ est le nombre maximum de proies que le prédateur peut traiter par jour (capacité physique), et $h$ est le nombre de proies tel que $\mu(h) = m/2$ (point à mi-saturation).

> [!example] La courbe Michaelis-Menten
> ![[Pasted image 20260505220500.png]]
> 
> **Figure 1.** Fonction de régression $\mu(x) = mx/(h+x)$ pour $m = h = 10$. On voit la **saturation** : à mesure que $x$ augmente, $\mu(x)$ se rapproche asymptotiquement de $m = 10$ sans jamais l'atteindre. Cette forme reflète une contrainte biologique — un prédateur ne peut pas capturer une infinité de proies, même si elles sont nombreuses.

**Link function.** La fonction $\mu(x) = mx/(h+x)$ n'est manifestement pas linéaire en $x$. Mais en prenant la **link reciprocal** $g(\mu) = 1/\mu$, on obtient :

$$g(\mu(x)) = \frac{1}{\mu(x)} = \frac{h + x}{m \cdot x} = \frac{1}{m} + \frac{h}{m} \cdot \frac{1}{x} = \beta_0 + \beta_1 \cdot \frac{1}{x}$$

avec $\beta_0 = 1/m$ et $\beta_1 = h/m$. **Le membre de droite est maintenant linéaire** — en $1/x$ certes, mais linéaire en les paramètres $\beta_0, \beta_1$. C'est tout ce dont on a besoin pour rentrer dans le framework GLM.

> 💡 **Le punch.** Le GLM permet de modéliser des relations **non-linéaires en $x$** mais qui restent **linéaires en $\beta$** après transformation par $g$. C'est ce qui rend le framework si flexible : il englobe énormément de modèles utiles tout en gardant la simplicité de l'inférence du modèle linéaire.

## IV. Tableau récapitulatif des cas usuels

> [!warning] Les GLM les plus courants
> 
> | Distribution de $Y$ | Domaine de $Y$ | Link canonique $g(\mu)$ | Modèle obtenu | Exemple d'application |
> |---|:---:|:---:|---|---|
> | **Gaussienne** | $\mathbb{R}$ | identité : $\mu$ | Régression linéaire | prédiction d'une grandeur continue (mpg, prix, score) |
> | **Bernoulli** | $\{0, 1\}$ | logit : $\log\frac{\mu}{1-\mu}$ | Régression logistique | classification binaire (spam, défaut, malade) |
> | **Binomiale** | $\{0, 1, \ldots, n\}$ | logit | Régression logistique groupée | proportions sur des groupes (taux de succès) |
> | **Poisson** | $\mathbb{N}$ | log : $\log \mu$ | Régression de Poisson | comptages d'événements (clics, accidents) |
> | **Gamma** | $\mathbb{R}_+$ | reciprocal ou log | Régression Gamma | durées, montants, claim sizes |
> | **Inverse Gaussienne** | $\mathbb{R}_+$ | $1/\mu^2$ | Régression Inverse Gaussian | durées avec queue lourde |

> [!note]- Pourquoi appelle-t-on certaines links "canoniques" ?
> Pour chaque distribution de la famille exponentielle, il existe **une link function privilégiée** appelée *link canonique*. Mathématiquement, c'est la link qui rend le **paramètre canonique** $\theta$ de la famille exponentielle directement égal au prédicteur linéaire ($\theta_i = x_i^T \beta$). Cela donne une expression particulièrement simple de la log-vraisemblance et garantit que le MLE est unique (concavité stricte). Pour la régression linéaire, le link canonique est l'identité ; pour la logistique, c'est le logit ; pour Poisson, c'est le log. Ces choix sont en pratique **les choix par défaut** de toutes les libraires (`statsmodels`, `glm()` en R, etc.). On peut utiliser des links non-canoniques (par exemple probit pour Bernoulli au lieu de logit), mais c'est plus rare et techniquement moins propre.

## V. Famille exponentielle — la machinerie sous le capot

Tout le formalisme GLM repose sur une **classe particulière de distributions** : la famille exponentielle. Pas besoin de retenir tous les détails techniques, mais l'idée générale est utile.

> [!note]- La forme canonique de la famille exponentielle (collapsé — pour référence)
> Une distribution est dans la famille exponentielle si sa densité (ou pmf) peut s'écrire :
> 
> $$f_\theta(y) = \exp\left(\frac{y\theta - b(\theta)}{\phi} + c(y, \phi)\right)$$
> 
> où :
> - **$\theta$** est le **paramètre canonique** (ce qu'on cherche à estimer)
> - **$\phi$** est le **paramètre de dispersion** (souvent supposé connu ou estimé séparément)
> - **$b(\theta)$** est la **fonction cumulante** (déterminée par la distribution)
> - **$c(y, \phi)$** est un terme de normalisation
> 
> **Deux résultats clés** qu'on obtient via les identités du score :
> 
> $$E[Y] = b'(\theta) \qquad \text{Var}(Y) = b''(\theta) \cdot \phi$$
> 
> Autrement dit : la moyenne de $Y$ est la dérivée première de $b$, et la variance est la dérivée seconde fois le paramètre de dispersion. Ces deux identités sont la machinerie qui permet de calculer mécaniquement $E[Y]$ et $\text{Var}(Y)$ pour n'importe quelle distribution de la famille.
> 
> **Exemples concrets** :
> 
> | Distribution | $\theta$ | $\phi$ | $b(\theta)$ | $E[Y] = b'(\theta)$ | $\text{Var}(Y) = b''(\theta)\phi$ |
> |---|:---:|:---:|:---:|:---:|:---:|
> | $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ | $\theta^2/2$ | $\theta = \mu$ | $\sigma^2$ |
> | Bernoulli$(p)$ | $\log\frac{p}{1-p}$ | $1$ | $\log(1 + e^\theta)$ | $\frac{e^\theta}{1+e^\theta} = p$ | $p(1-p)$ |
> | Poisson$(\lambda)$ | $\log \lambda$ | $1$ | $e^\theta$ | $e^\theta = \lambda$ | $\lambda$ |
> | Gamma | $-1/\mu$ | scale | $-\log(-\theta)$ | $-1/\theta = \mu$ | $\mu^2 \cdot \phi$ |

> 💡 **Pourquoi cette famille spécifiquement ?** Parce que c'est précisément la classe où :
> 1. Le MLE est bien défini et a une **forme tractable** (log-vraisemblance concave avec link canonique → optimum unique)
> 2. La **variance** est automatiquement déterminée par la moyenne via $\text{Var}(Y) = b''(\theta)\phi$ — pas besoin de l'estimer comme paramètre libre supplémentaire
> 3. L'algorithme **IRLS** (Iteratively Reweighted Least Squares, cf. section VII) converge proprement
> 
> Bref, c'est la classe où **toute la machinerie statistique fonctionne uniformément**.

## VI. Lien canonique — la propriété qui simplifie tout

> [!warning] Définition (Link canonique)
> La **link canonique** $g$ est celle qui relie directement la moyenne $\mu$ au paramètre canonique $\theta$ :
> 
> $$g(\mu) = \theta$$
> 
> Comme $\mu = b'(\theta)$, on a $g = (b')^{-1}$.

#### Pourquoi c'est utile

Avec la link canonique, $\theta_i = x_i^T \beta$ directement, et la log-vraisemblance s'écrit :

$$\ell_n(\beta) = \sum_{i=1}^n \frac{Y_i \cdot x_i^T \beta - b(x_i^T \beta)}{\phi} + \text{cst}$$

Cette expression est **strictement concave** en $\beta$ (dès que $\phi > 0$ et $X$ est de rang plein), ce qui implique :

> 💡 **Conséquence majeure de la concavité stricte.** Avec le link canonique, le MLE $\hat{\beta}$ est **unique** — pas de problème de minima locaux, pas d'ambiguïté. C'est l'analogue de l'unicité de la solution OLS en régression linéaire (où $X^TX$ est inversible). Avec un link non-canonique, on perd cette garantie : il peut y avoir plusieurs optima locaux selon la paramétrisation.

#### Tableau des links canoniques

| Distribution | $b(\theta)$ | Link canonique $g(\mu)$ |
|---|:---:|:---:|
| Normal | $\theta^2/2$ | $\mu$ (identité) |
| Poisson | $e^\theta$ | $\log \mu$ (log) |
| Bernoulli | $\log(1 + e^\theta)$ | $\log\frac{\mu}{1-\mu}$ (logit) |
| Gamma | $-\log(-\theta)$ | $-1/\mu$ (reciprocal) |

C'est ce tableau qui **dicte les choix par défaut** de toutes les libraires statistiques. Quand on appelle `LogisticRegression()` en sklearn ou `glm(family='poisson')` en R, on utilise *implicitement* la link canonique de la distribution choisie.

## VII. Estimation et inférence — vue d'ensemble

#### Pas de solution fermée en général

Contrairement à la régression linéaire (où $\hat{\beta} = (X^TX)^{-1}X^Ty$ est explicite), **la plupart des GLM n'ont pas de solution analytique** pour le MLE. On doit utiliser un **algorithme itératif**.

> [!warning] IRLS (Iteratively Reweighted Least Squares)
> L'algorithme standard pour fitter un GLM est l'**IRLS**. L'idée est élégante : à chaque itération, on linéarise localement le modèle autour de l'estimation courante $\hat{\beta}^{(k)}$ et on résout un **problème de WLS** (Weighted Least Squares, cf. [[01_Régression Linéaire Multiple]] section IV.B). Les poids dépendent de la variance prédite par le modèle courant, et on itère jusqu'à convergence.
> 
> $$\hat{\beta}^{(k+1)} = (X^T W^{(k)} X)^{-1} X^T W^{(k)} z^{(k)}$$
> 
> où $W^{(k)}$ et $z^{(k)}$ sont mis à jour à chaque itération en fonction de $\hat{\beta}^{(k)}$.

> 💡 **L'idée à retenir.** *Fitter un GLM = résoudre une suite de WLS jusqu'à convergence*. C'est pour ça que comprendre WLS en régression linéaire est crucial : c'est la brique de base de toute l'estimation GLM. Et c'est aussi pour ça que la régression linéaire est un GLM trivial : c'est le cas où une seule itération suffit (l'IRLS converge en un coup).

#### Asymptotique du MLE

Le MLE en GLM possède des propriétés asymptotiques classiques :

> [!warning] Normalité asymptotique du MLE
> Sous des conditions de régularité (matrice d'information de Fisher inversible, $n \to \infty$) :
> 
> $$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} \mathcal{N}(0, \mathcal{I}^{-1}(\beta))$$
> 
> où $\mathcal{I}(\beta)$ est la matrice d'information de Fisher.

Cela permet de construire :

- **Wald test** : pour tester $H_0 : \beta_j = 0$, on utilise $\hat{\beta}_j / \widehat{\text{SE}}(\hat{\beta}_j) \sim \mathcal{N}(0, 1)$ asymptotiquement
- **Likelihood Ratio Test (LRT)** : pour tester un ensemble de coefficients (analogue du test de Fisher en linéaire), on compare les log-vraisemblances de deux modèles emboîtés
- **Intervalles de confiance** : via Wald ou via profilage de la log-vraisemblance

> [!note]- Wald vs LRT
> Le **Wald test** est plus simple à calculer (juste $\hat{\beta}/\text{SE}$) mais peut être trompeur en présence de séparation parfaite ou de petits échantillons. Le **LRT** est plus robuste mais nécessite de fitter deux modèles. En pratique, pour un coefficient unique on utilise Wald (c'est ce qui est dans `summary()`) ; pour tester un groupe de coefficients ou des modèles emboîtés, on utilise LRT (la fonction `anova()` en R, `compare_lr_test` en statsmodels).

#### Pas de R² classique — la deviance

En régression linéaire, le R² est central. En GLM, il n'a **pas d'équivalent direct**, parce que la décomposition $\text{SST} = \text{SSE} + \text{SSR}$ ne tient plus (la perte n'est plus quadratique mais log-vraisemblance).

À la place, on utilise la **deviance** :

> [!warning] Deviance
> $$D = -2 \big(\ell(\hat{\beta}) - \ell_{\text{sat}}\big)$$
> 
> où $\ell_{\text{sat}}$ est la log-vraisemblance du **modèle saturé** (un paramètre par observation, fit parfait).
> 
> La deviance joue le rôle du SSR (sum of squared residuals) en GLM : plus elle est petite, mieux c'est. Différence de deviance entre deux modèles emboîtés = statistique du LRT (asymptotiquement $\chi^2$).

Pour avoir un analogue normalisé du R², on utilise des **pseudo-R²** (McFadden, Cox-Snell, Nagelkerke) ou, pour la classification, des métriques discriminatives comme **AUC-ROC**, **KS**, **Gini coefficient** (cf. [[02_Régression logistique]]).

## VIII. Exemples d'applications

Le framework GLM est partout dès qu'on sort du cadre gaussien :

- **Classification binaire** (Bernoulli + logit) : prédire si un email est un spam, si un patient développera une maladie, si un client va faire défaut sur son prêt → c'est la **régression logistique** (cf. [[02_Régression logistique]]).
- **Modélisation de comptages** (Poisson + log) : nombre de clics sur une publicité, nombre d'accidents par jour, nombre de pannes d'une machine, nombre d'événements dans une fenêtre temporelle.
- **Modélisation de durées et de montants** (Gamma + reciprocal/log) : durée entre deux événements, taille des sinistres en assurance, temps de séjour à l'hôpital, prix.
- **Proportions agrégées** (Binomiale + logit) : taux de succès sur des groupes (par exemple taux de conversion par segment, taux de réussite par classe d'élèves).
- **Modélisation de données overdispersed** : quand la variance est plus grande que ce que prédit le modèle Poisson naïf, on utilise des extensions comme **Negative Binomial** ou **quasi-Poisson** (qui restent dans la famille GLM élargie).

## IX. Pour aller plus loin

#### Liens avec d'autres notes

- [[01_Régression Linéaire Multiple]] — cas particulier identité link + Gauss
- [[02_Régression logistique]] — cas particulier logit link + Bernoulli
- [[Maximum de vraisemblance|MLE]] — le cadre d'estimation général sur lequel repose tout GLM
- [[Famille exponentielle]] — la classe de distributions sous-jacente

#### Extensions du framework GLM

- **GLMM** (Generalized Linear Mixed Models) : ajout d'**effets aléatoires** pour gérer les données hiérarchiques / panel / longitudinales
- **GAM** (Generalized Additive Models) : on remplace $\eta = x^T\beta$ par $\eta = \sum_j f_j(x_j)$ avec $f_j$ fonctions lisses estimées non-paramétriquement
- **Quasi-likelihood** : on relâche l'hypothèse de famille exponentielle, on garde juste la relation moyenne-variance — utile pour les données over/underdispersed
- **Régularisation** : Ridge/Lasso/ElasticNet s'étendent naturellement aux GLM (`sklearn.linear_model.LogisticRegression(penalty='l2')`, `glmnet` en R)

> 💡 **À retenir.** GLM est le **framework unifié** qui englobe la régression linéaire et la régression logistique comme deux cas particuliers parmi d'autres. Comprendre ce cadre permet de voir que **tous ces modèles partagent la même structure** : une distribution dans la famille exponentielle, un prédicteur linéaire $\eta = X\beta$, et une link function qui fait le pont entre les deux. Une fois cette ossature en tête, passer d'un modèle à un autre devient mécanique — il suffit de choisir la bonne distribution et la bonne link selon la nature de $Y$.
