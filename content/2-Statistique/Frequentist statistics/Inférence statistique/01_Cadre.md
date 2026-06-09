---
title: Cadre
order: 1
---

# Cadre — Modèles statistiques et estimation

Cette unité introduit la formalisation mathématique de la modélisation statistique pour donner sens à la **trinité de l'inférence statistique**. On veut comprendre les énoncés suivants :

1. **Estimation (point estimate)** :
$$\hat{p} = \bar{R}_n \text{ est un estimateur pour la proportion } p \text{ de couples qui tournent la tête à droite}$$

2. **Intervalles de confiance** :
$$[0.56, 0.73] \text{ est un IC à 95\% pour } p$$

3. **Tests d'hypothèses** :
$$\text{On trouve une preuve statistique que plus de couples tournent la tête à droite en s'embrassant}$$

---

## I. Modèle statistique

> [!warning] Définition — Modèle statistique
> Soit l'issue observée d'une expérience statistique un échantillon $X_1, \ldots, X_n$ de $n$ variables aléatoires i.i.d. dans un espace mesurable $E$ (en général $E \subseteq \mathbb{R}$). On note $\mathbb{P}$ leur distribution commune. Un **modèle statistique** associé est une paire :
> 
> $$(E, (\mathbb{P}_\theta)_{\theta \in \Theta})$$
> 
> où :
> - $E$ est appelé **espace d'échantillonnage** (sample space)
> - $(\mathbb{P}_\theta)_{\theta \in \Theta}$ est une famille de mesures de probabilité sur $E$
> - $\Theta$ est un ensemble appelé **espace des paramètres**

### A. Modèles paramétriques, non-paramétriques, semi-paramétriques

Habituellement, on suppose le modèle statistique **bien spécifié**, c'est-à-dire :

$$\exists \theta \text{ tel que } \mathbb{P} = \mathbb{P}_\theta$$

Ce $\theta$ est appelé le **vrai paramètre**, et il est inconnu. Le but de l'expérience est de trouver $\theta$, ou de vérifier ses propriétés (du type $\theta > 2$ ou $\theta \neq 1/2$).

> [!warning] Définition — Paramétrique
> On suppose souvent $\Theta \subseteq \mathbb{R}^d$ pour un certain $d \ge 1$. Le modèle est alors **paramétrique**.

> [!warning] Définition — Non-paramétrique
> Si $\Theta$ est de dimension infinie, le modèle est **non-paramétrique**.

> [!warning] Définition — Semi-paramétrique
> Si $\Theta = \Theta_1 \times \Theta_2$ avec $\Theta_1$ de dimension finie et $\Theta_2$ de dimension infinie : modèle **semi-paramétrique**. On veut estimer le paramètre fini-dimensionnel et le paramètre infini-dimensionnel est appelé **paramètre de nuisance** (nuisance parameter).

> [!example] Exemples de modèles paramétriques
> 1. Pour $n$ tirages de Bernoulli : $(\{0, 1\}, (\text{Ber}(p))_{p \in (0, 1)})$
> 2. Si $X_1, \ldots, X_n \sim \text{Pois}(\lambda)$ pour $\lambda > 0$ inconnu : $(\mathbb{N}, (\text{Poiss}(\lambda))_{\lambda > 0})$
> 3. Si $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ pour $\mu \in \mathbb{R}$, $\sigma^2 > 0$ inconnus : $(\mathbb{R}, (\mathcal{N}(\mu, \sigma^2))_{(\mu, \sigma^2) \in \mathbb{R} \times (0, \infty)})$
> 4. Si $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}_d(\mu, I_d)$ pour $\mu \in \mathbb{R}^d$ inconnu : $(\mathbb{R}^d, (\mathcal{N}_d(\mu, I_d))_{\mu \in \mathbb{R}^d})$

> [!example] Exemples de modèles non-paramétriques
> 1. Si $X_1, \ldots, X_n \in \mathbb{R}$ sont i.i.d. de pdf unimodale $f$ inconnue : $E = \mathbb{R}$, $\Theta = \{\text{pdf unimodales}\}$, où $\mathbb{P}_\theta = \mathbb{P}_f = \text{distribution de pdf } f$
> 2. Si $X_1, \ldots, X_n \in [0, 1]$ sont i.i.d. de cdf inversible $F$ inconnue : $E = [0, 1]$

> [!example] Autres exemples
> Parfois on n'a pas de notation simple comme $(\mathbb{P}_\theta)_{\theta \in \Theta}$ et il faut être plus explicite.
> 
> 1. **Modèle de régression linéaire** : $(X_1, Y_1), \ldots, (X_n, Y_n) \in \mathbb{R}^d \times \mathbb{R}$ i.i.d. du modèle $Y_i = \beta^T X_i + \varepsilon_i$, $\varepsilon_i \overset{iid}{\sim} \mathcal{N}(0, 1)$ pour $\beta \in \mathbb{R}^d$ inconnu et $X_i \sim \mathcal{N}_d(0, I_d)$ indépendant de $\varepsilon_i$. $E = \mathbb{R}^d \times \mathbb{R}$, $\Theta = \mathbb{R}^d$.
> 
> 2. **Modèle de Cox à risques proportionnels** : $(X_1, Y_1), \ldots, (X_n, Y_n) \in \mathbb{R}^d \times \mathbb{R}$ ; la distribution conditionnelle de $Y$ sachant $X = x$ a une CDF $F$ de la forme :
> 
> $$F(t) = 1 - \exp\left(-\int_0^t h(u) e^{\beta^T x} du\right)$$
> 
> où $h$ est une fonction de nuisance non-négative inconnue et $\beta \in \mathbb{R}^d$ est le paramètre d'intérêt.

### B. Identifiabilité

> [!warning] Définition — Identifiabilité
> Le paramètre $\theta$ est **identifiable** ssi l'application $\theta \in \Theta \mapsto \mathbb{P}_\theta$ est injective :
> 
> $$\theta \neq \theta' \Rightarrow \mathbb{P}_\theta \neq \mathbb{P}_{\theta'}$$
> 
> ou de manière équivalente :
> 
> $$\mathbb{P}_\theta = \mathbb{P}_{\theta'} \Rightarrow \theta = \theta'$$

---

## II. Estimation

### A. Échantillon, statistique, estimateur

> [!warning] Définition — n-échantillon
> Soit $(X_1, \ldots, X_n)$ un n-échantillon : chaque $X_i$ est une variable aléatoire pour $1 \le i \le n$, où $n$ est la taille d'échantillon. Les propriétés suivantes tiennent :
> - **Indépendance** (le i de iid) : $\text{cov}(X_i, X_j) = 0$ si $i \neq j$
> - **Identiquement distribuées** (i.d.) : $E(X_i) = E(X_j) \forall i, j$ et $\text{Var}(X_i) = \text{Var}(X_j) \forall i, j$. Les $X_i$ suivent la même distribution, donc on peut toujours remplacer $E(X_i) = E(X_1)$ et de même pour $\text{Var}$.
> - Conséquence des deux propriétés :
> 
> $$E(\bar{X}_n) = E(X_1) \quad \text{et} \quad \text{Var}(\bar{X}_n) = \text{Var}(X_1)/n = \sigma^2/n$$

> [!warning] Définition — Statistique
> Toute fonction mesurable de l'échantillon, par exemple $\bar{X}_n$, $\max_i X_i$, $X_1 + \log(1 + |X_n|)$, la variance empirique, etc.

> [!warning] Définition — Estimateur et estimation de $\theta$
> Toute statistique dont l'expression ne dépend pas de $\theta$ est un **estimateur**. Une **estimation** est une réalisation de l'expérience.

> [!example] Exemple
> L'estimateur de la moyenne est :
> 
> $$\hat{\mu} = \bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$$
> 
> L'estimation peut être une réalisation de l'expérience : $\hat{\mu}_{\text{obs}} = 3.14$.

![[images/2-Statistiques/A_Frequentist/01_Inference statistique/01_Cadre/im1 (3).png|611]]

### B. Consistance et normalité asymptotique

> [!warning] Définition — Consistance d'un estimateur
> Un estimateur $\hat{\theta}_n$ de $\theta$ est **faiblement (resp. fortement) consistant** si :
> 
> $$\hat{\theta}_n \xrightarrow[n \to \infty]{\mathbb{P} \text{ (resp. p.s.)}} \theta \quad (\text{w.r.t. } \mathbb{P}_\theta)$$
> 
> Un estimateur $\hat{\theta}_n$ de $\theta$ est **asymptotiquement normal** si :
> 
> $$\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, \sigma^2)$$
> 
> $\sigma^2$ est alors appelée la **variance asymptotique** de $\hat{\theta}_n$.

### C. Biais

> [!warning] Définition — Biais
> Biais d'un estimateur $\hat{\theta}_n$ de $\theta$ :
> 
> $$\boxed{\text{biais}(\hat{\theta}_n) = \mathbb{E}[\hat{\theta}_n] - \theta}$$
> 
> Si $\text{biais}(\hat{\theta}) = 0$, on dit que $\hat{\theta}$ est **sans biais** (unbiased).

> [!example] Exemple — biais de l'estimateur de la moyenne
> Soit $X$ une v.a. de distribution inconnue avec $E[X] = \mu$ et $\text{Var}(X) = \sigma^2$. Soit $X_1, \ldots, X_n$ un n-échantillon.
> 
> - $T_1 = X_1$ : $b(T_1) = E(X_1) - \mu = \mu - \mu = 0$
> - $T_2 = \bar{X}_n$ : $b(T_2) = E(\bar{X}_n) - \mu = E(X) - \mu = \mu - \mu = 0$
> - $T_3 = \bar{X}_n^2$ : $b(T_3) = E(\bar{X}_n^2) - \mu$. On utilise :
> 
> $$\text{Var}(\psi) = E(\psi^2) - E(\psi)^2 \Rightarrow E(\psi^2) = \text{Var}(\psi) + E(\psi)^2$$
> 
> D'où :
> 
> $$
> \begin{aligned}
> b(T_3) &= \text{Var}(\bar{X}_n) + E(\bar{X}_n)^2 - \mu \\
> &= \frac{\text{Var}(X)}{n} + E(X)^2 - \mu \\
> &= \frac{\sigma^2}{n} + \mu^2 - \mu
> \end{aligned}
> $$
> 
> L'estimateur est biaisé.
> 
> - $T_5 = \frac{\sum_{i=1}^n X_i}{n-1} = \frac{n}{n-1} \bar{X}_n$. Alors :
> 
> $$
> \begin{aligned}
> b(T_5) &= E\left(\frac{n}{n-1}\bar{X}_n\right) - \mu = \frac{n}{n-1}\mu - \mu \\
> &= \mu \frac{n - (n-1)}{n-1} = \frac{\mu}{n-1}
> \end{aligned}
> $$

### D. Risque quadratique

> [!warning] Définition — Risque quadratique
> On veut des estimateurs avec à la fois un faible biais et une faible variance. On définit le **risque quadratique** d'un estimateur $\hat{\theta}_n \in \mathbb{R}$ comme :
> 
> $$R(\hat{\theta}_n) = \mathbb{E}[|\hat{\theta}_n - \theta|^2]$$
> 
> On peut montrer que :
> 
> $$\boxed{\text{Risque quadratique} = \text{Variance} + \text{Biais}^2}$$
> 
> **Vocabulaire.** Le risque quadratique est exactement l'**erreur quadratique moyenne** (MSE, *Mean Squared Error*) : c'est le même objet $\mathbb{E}[(\hat{\theta}_n - \theta)^2]$. « Risque quadratique » est le terme de la théorie de la décision — où le *risque* $\mathbb{E}_\theta[L(\hat{\theta}_n, \theta)]$ se définit pour une perte $L$ quelconque, le MSE étant le cas particulier de la perte quadratique $L(\hat{\theta}, \theta) = (\hat{\theta} - \theta)^2$ — tandis que « MSE » est le terme usuel en ML. Ici tout est en perte quadratique, donc *risque* = *risque quadratique* = *MSE*.

> [!example] Exemple — risque quadratique
> - $T_1 = X_1$, $b(T_1) = 0$ donc :
> 
> $$R(T_1) = b(T_1)^2 + \text{Var}(T_1) = 0 + \text{Var}(X_1) = \sigma^2$$
> 
> - $T_2 = \bar{X}_n$, $b(T_2) = 0$ donc $R(T_2) = \text{Var}(T_2)$ :
> 
> $$R(T_2) = \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$$
> 
> Comme $R(T_1) = \sigma^2$ et $R(T_2) = \sigma^2/n \xrightarrow[n \to \infty]{} 0$, **$T_2$ est meilleur que $T_1$ au sens du risque quadratique**.

### E. Le compromis biais-variance

La décomposition $R(\hat{\theta}_n) = \text{Var}(\hat{\theta}_n) + \text{biais}(\hat{\theta}_n)^2$ n'est pas qu'une identité algébrique : c'est l'énoncé d'un **arbitrage**. Ces deux quantités mesurent deux choses géométriquement distinctes, et l'erreur classique est de les confondre.

![[biais_variance.png|650]]
> *Figure — Décomposition du risque quadratique. **À gauche**, l'anatomie : l'estimateur $\hat{\theta}_n$ est la cloche entière (sa distribution d'échantillonnage) ; le **biais** est l'écart entre $\theta^\ast$ et le centre $\mathbb{E}[\hat{\theta}_n]$, la **variance** est la largeur de la cloche autour de ce centre (et non autour de $\theta^\ast$). **À droite**, le compromis : l'estimateur biaisé mais resserré B ($R = 0{.}61$) bat l'estimateur sans biais mais étalé A ($R = 1{.}56$) — un peu de biais échangé contre beaucoup de variance en moins.*

> [!tip] Lecture géométrique de la figure
> Sur l'axe des abscisses : les **valeurs possibles de l'estimateur**. L'estimateur $\hat{\theta}_n$ n'est **pas** un point — c'est la *cloche* entière, sa **distribution d'échantillonnage** (sampling distribution).
> - $\theta^\ast$ : la vraie valeur, un point fixe.
> - $\mathbb{E}[\hat{\theta}_n]$ : le **centre** de la cloche.
> - **biais** $= \mathbb{E}[\hat{\theta}_n] - \theta^\ast$ : l'écart entre le centre de la cloche et la vérité (erreur *systématique*).
> - **variance** : la **largeur** de la cloche, c'est-à-dire la dispersion autour de $\mathbb{E}[\hat{\theta}_n]$ — et **non** autour de $\theta^\ast$.
> 
> C'est précisément parce que la variance se mesure autour de $\mathbb{E}[\hat{\theta}_n]$ (et non de $\theta^\ast$) que la décomposition se sépare proprement en deux termes. Si on mesurait la dispersion directement autour de $\theta^\ast$, on obtiendrait le risque quadratique entier d'un coup, sans le décomposer.

La conséquence est contre-intuitive : **le meilleur estimateur n'est pas forcément sans biais**. Accepter un peu de biais pour réduire la variance peut diminuer le risque total. C'est toute la logique du *shrinkage* (régularisation ridge, estimateur de James-Stein, etc.).

> [!note] Le piège de l'oracle
> $c^\ast$ dépend de $\mu$ et $\sigma^2$ **inconnus** : ce n'est donc pas un véritable estimateur (au sens où son expression dépend des quantités à estimer), mais un *oracle*. L'intérêt théorique reste entier : il prouve l'existence d'un estimateur biaisé qui fait mieux. Le passage du résultat oracle à un estimateur *utilisable* — qui estime le shrinkage à partir des données et domine quand même $\bar{X}$ — est exactement l'objet de l'**estimateur de James-Stein** (dimension $\ge 3$).

> [!note] Lien ML — la même décomposition, mais en prédiction
> La décomposition réapparaît au cœur du machine learning, mais avec **trois** termes au lieu de deux (cf. [[01_Fondation#D. Décomposition biais-variance et double descent]]). Ce n'est pas une contradiction : la différence ne tient pas à « stats vs ML » mais à la **cible d'évaluation**.
> 
> - **Ici (estimation)** — la cible est un paramètre déterministe $\theta$, donc
> 
> $$R(\hat{\theta}_n) = \text{Var}(\hat{\theta}_n) + \text{biais}^2 \qquad (\text{2 termes}).$$
> 
> - **En ML (prédiction)** — la cible est une observation future $y_0 = f(x_0) + \varepsilon_0$, *bruitée*, ce qui ajoute un troisième terme :
> 
> $$\text{EPE}(x_0) = \underbrace{\big(f(x_0) - \mathbb{E}[\hat{f}(x_0)]\big)^2 + \text{Var}(\hat{f}(x_0))}_{\text{= la décompo ci-dessus, pour } \hat{f}(x_0) \text{ estimant } f(x_0)} + \underbrace{\text{Var}(\varepsilon)}_{\sigma^2 \text{ irréductible}}.$$
> 
> En une phrase : $\;\text{EPE}(x_0) = \underbrace{\mathbb{E}[(\hat{f}(x_0) - f(x_0))^2]}_{\text{décompo d'estimation}} + \sigma^2$. **La prédiction, c'est l'estimation plus le bruit de l'observation test.**
> 
> | Estimation (cette note) | Prédiction (ML) |
> | :--- | :--- |
> | paramètre vrai $\theta$ (fixe) | valeur vraie $f(x_0)$ (fixe) |
> | estimateur $\hat{\theta}_n$ | prédiction $\hat{f}(x_0)$ |
> | source d'aléa : l'échantillon $X_1, \dots, X_n$ | source d'aléa : le training set $\mathcal{T}$ |
> | cible d'évaluation : $\theta$ | cible d'évaluation : $y_0 = f(x_0) + \varepsilon_0$ |
> | $R = \text{Var} + \text{biais}^2$ | $\text{EPE} = \text{Var} + \text{biais}^2 + \sigma^2$ |
> 
> Le vrai clivage est donc **estimation vs prédiction** : viser une cible déterministe ne donne pas de terme irréductible, viser un tirage futur bruité en donne un. C'est exactement pourquoi la courbe en U du test MSE ne descend jamais sous $\sigma^2$ : quand la complexité monte, biais² ↓ et variance ↑ (les deux termes d'estimation), pendant que $\sigma^2$ reste un plancher constant.
