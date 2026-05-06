---
title: Inférence Bayésienne
---
# Inférence Bayésienne

> Ce fichier couvre les fondations de l'**inférence bayésienne** : ce que veut dire "probabilité" dans le cadre bayésien, comment passer du prior au posterior via Bayes, comment construire des intervalles de crédibilité, et comment exploiter la **conjugaison** (Beta-Bernoulli, Gamma-Poisson, Normale-Normale) pour obtenir des posteriors analytiques.

## I. Les trois paradigmes de la probabilité

Avant Bayes, il faut comprendre ce qu'on entend par "probabilité". Trois cadres coexistent.

### A. Cadre classique

**Idée.** Si les issues d'une expérience sont symétriques (équiprobables), chacune a probabilité $1/n$. Pour un dé équilibré : $\mathbb{P}(\text{face } k) = 1/6$.

**Limite.** Marche pour les jeux de hasard avec symétrie évidente. Casse dès qu'on n'a pas cette symétrie : quelle probabilité que la météo soit pluvieuse demain ? Que ce client fasse défaut sur son crédit ? Pas d'issues équiprobables à compter.

### B. Cadre fréquentiste

**Idée.** La probabilité d'un événement est sa **fréquence limite** dans une suite hypothétique infinie de répétitions. $\mathbb{P}(A) = \lim_{n \to \infty} \frac{\#A}{n}$.

**Limite.** Marche bien quand on peut imaginer répéter l'expérience à l'identique (lancer un dé, tirer une boule). Mais que veut dire "probabilité qu'il pleuve demain" ? Il n'y a qu'**un seul** demain, pas une suite infinie. Idem pour "probabilité que ce dé soit truqué" : sous le cadre fréquentiste, le dé est ou n'est pas truqué — la probabilité est déjà 0 ou 1, on ne sait juste pas laquelle.

### C. Cadre bayésien

**Idée.** La probabilité est une **mesure subjective d'incertitude**. Elle reflète l'état d'information de celui qui raisonne. Deux personnes avec des informations différentes peuvent légitimement assigner des probabilités différentes au même événement.

La probabilité que ce dé soit truqué dépend de **ce que tu sais** : qui te l'a donné, à quoi il ressemble, comment il s'est comporté lors des derniers lancers. Au fur et à mesure que tu observes des données, ta probabilité se met à jour — c'est exactement ce que fait Bayes.

> 💡 **L'idée en une phrase.** Le bayésien traite les paramètres inconnus comme des variables aléatoires sur lesquelles on a une distribution de croyance, pas comme des constantes fixes inconnues. C'est ce qui permet d'écrire $\mathbb{P}(\theta \in [a, b]) = 0.95$, ce que le fréquentiste ne peut pas faire avec ses intervalles de confiance.

> [!note]- Cohérence et Dutch Book
> Pour être *cohérentes*, les probabilités d'un agent doivent suivre les axiomes standards (Kolmogorov). Si elles ne les suivent pas, on peut construire un **Dutch Book** — une série de paris dont l'agent est mathématiquement garanti de sortir perdant. Le théorème de Ramsey-de Finetti montre que **être cohérent ⟺ raisonner comme un bayésien**. C'est l'argument philosophique qui fonde le cadre.

## II. Théorème de Bayes

### A. Cas discret

Pour deux événements $A$ et $B$ avec $\mathbb{P}(B) > 0$ :

$\mathbb{P}(A \mid B) = \frac{\mathbb{P}(B \mid A) \, \mathbb{P}(A)}{\mathbb{P}(B)}.$

Dans une partition $A_1, \dots, A_n$ de l'espace, le dénominateur se développe par la formule des probabilités totales :

$\mathbb{P}(A_k \mid B) = \frac{\mathbb{P}(B \mid A_k) \, \mathbb{P}(A_k)}{\sum_{i=1}^n \mathbb{P}(B \mid A_i) \, \mathbb{P}(A_i)}.$

**Vocabulaire :**
- $\mathbb{P}(A_k)$ = **prior** (notre croyance sur $A_k$ avant d'observer $B$)
- $\mathbb{P}(B \mid A_k)$ = **vraisemblance** (likelihood : la probabilité d'observer $B$ si $A_k$ est vrai)
- $\mathbb{P}(A_k \mid B)$ = **posterior** (notre croyance mise à jour après avoir observé $B$)
- Le dénominateur = constante de normalisation, indépendante de $A_k$

### B. Cas continu

Pour un paramètre $\theta$ et des données $y$ :

$f(\theta \mid y) = \frac{f(y \mid \theta) \, f(\theta)}{f(y)} = \frac{f(y \mid \theta) \, f(\theta)}{\int f(y \mid \theta') \, f(\theta') \, \mathrm{d}\theta'}.$

En pratique, l'intégrale au dénominateur est souvent intractable. Heureusement, comme elle ne dépend pas de $\theta$, on peut travailler à une constante près :

$\boxed{\; f(\theta \mid y) \;\propto\; \underbrace{f(y \mid \theta)}_{\text{vraisemblance}} \cdot \underbrace{f(\theta)}_{\text{prior}} \;}$

C'est cette forme proportionnelle qu'on utilise quasiment tout le temps en pratique. On reconnaît la *forme* du posterior à un facteur près, et la constante se déduit après si nécessaire.

> [!example] Exemple — pile ou face avec un prior uniforme
> On lance une pièce dont le biais $\theta \in [0, 1]$ est inconnu. Prior plat : $f(\theta) = \mathbf{1}_{[0,1]}(\theta)$. On observe **un seul lancer**, qui tombe sur pile ($Y = 1$).
>
> La vraisemblance est $f(Y=1 \mid \theta) = \theta$. Donc à une constante près :
>
> $f(\theta \mid Y=1) \propto \theta \cdot \mathbf{1}_{[0,1]}(\theta).$
>
> Pour normaliser, on calcule $\int_0^1 \theta \, \mathrm{d}\theta = 1/2$. Donc le posterior est $f(\theta \mid Y=1) = 2\theta \, \mathbf{1}_{[0,1]}(\theta)$.
>
> ![[im1 (1) 2.png]]
> **Figure 1.** À gauche : le prior plat (uniforme). À droite : le posterior $2\theta$, qui penche vers $\theta = 1$ parce qu'on a vu pile. C'est l'intuition de Bayes en image : observer pile rend les valeurs élevées de $\theta$ plus crédibles.

### C. Lien avec la vraisemblance fréquentiste

La **vraisemblance** $L(\theta \mid y) = f(y \mid \theta)$ est l'objet central de l'inférence fréquentiste : on cherche $\hat{\theta}^{\text{MLE}} = \arg\max_\theta L(\theta \mid y)$. Le bayésien la pondère par un prior et regarde la *forme entière* du posterior, pas seulement son maximum.

> [!note]- Subtilité : la vraisemblance n'est pas une densité en $\theta$
> $f(y \mid \theta)$ est une densité (ou pmf) en $y$ : si on intègre sur $y$ à $\theta$ fixé, on obtient 1. Mais vue comme fonction de $\theta$ à $y$ fixé, ce n'est *pas* une densité en $\theta$ — son intégrale sur $\theta$ n'a aucune raison de valoir 1. Par contre $f(\theta) \cdot f(y \mid \theta)$, après normalisation, *est* bien une densité en $\theta$ : c'est le posterior.

## III. Fréquentiste vs bayésien — un exemple comparé

Pour fixer les idées, prenons le **même problème** et résolvons-le des deux façons.

**Setup.** Ton frère a deux pièces : une équilibrée ($\theta = 0.5$) et une biaisée ($\theta = 0.7$). Il prend l'une des deux et te propose un pari. Tu peux la lancer 5 fois pour la tester. Tu obtiens **2 piles, 3 faces**. Quelle pièce est-ce ?

### A. Approche fréquentiste

**Modèle.** $X \sim \mathrm{Bin}(5, \theta)$ avec $\theta \in \{0.5, 0.7\}$.

**Vraisemblance** sur les deux hypothèses, évaluée à $X = 2$ :

$L(0.5) = \binom{5}{2} (0.5)^2 (0.5)^3 = 0.3125, \qquad L(0.7) = \binom{5}{2} (0.7)^2 (0.3)^3 = 0.1323.$

**MLE :** $\hat{\theta}^{\text{MLE}} = 0.5$ (la pièce équilibrée maximise la vraisemblance).

**Limite.** Si on demande "quelle est la probabilité que la pièce soit équilibrée ?", le fréquentiste ne peut pas répondre : la pièce *est* ou *n'est pas* équilibrée, c'est un fait, donc $\mathbb{P}(\theta = 0.5) \in \{0, 1\}$. Le MLE donne un point estimé, mais pas une mesure d'incertitude sur ce point.

### B. Approche bayésienne

**Prior.** Tu connais ton frère, tu sais qu'il a 60% de chances de t'avoir filé la pièce truquée :

$\mathbb{P}(\theta = 0.7) = 0.6, \qquad \mathbb{P}(\theta = 0.5) = 0.4.$

**Posterior** par Bayes :

$\mathbb{P}(\theta = 0.5 \mid X = 2) = \frac{L(0.5) \cdot 0.4}{L(0.5) \cdot 0.4 + L(0.7) \cdot 0.6} = \frac{0.0125}{0.0125 + 0.0079} \approx 0.612.$

$\mathbb{P}(\theta = 0.7 \mid X = 2) \approx 0.388.$

**Lecture.** Malgré le prior qui penchait vers la pièce truquée (60%), les données (peu de piles) ont fait basculer la croyance : il y a maintenant **61% de chances que la pièce soit équilibrée**. C'est une vraie probabilité, sur laquelle on peut parier.

> [!summary] Le contraste à retenir
> - **Fréquentiste** : $\theta$ est une constante inconnue, les données sont aléatoires. On obtient un point estimé + des garanties asymptotiques (intervalle de confiance interprété en termes de répétitions).
> - **Bayésien** : $\theta$ est une v.a. avec une distribution de croyance, les données sont fixées (on les a observées). On obtient une **distribution complète** sur $\theta$, à partir de laquelle on calcule moyennes, intervalles, probabilités.
>
> Le bayésien permet d'incorporer **explicitement** des connaissances a priori (le prior). Le fréquentiste prétend être objectif mais cache des choix subjectifs (population de référence, choix du modèle).

## IV. Intervalles de crédibilité

Le pendant bayésien de l'intervalle de confiance fréquentiste s'appelle **intervalle de crédibilité** (credible interval). La différence est philosophique mais importante.

**Intervalle de confiance fréquentiste à 95%.** *"Si on répète l'expérience une infinité de fois et qu'on calcule l'intervalle à chaque fois, 95% de ces intervalles contiendront la vraie valeur."* On ne peut **pas** dire que le vrai $\theta$ a 95% de chances d'être dans *cet* intervalle-ci.

**Intervalle de crédibilité bayésien à 95%.** *"Sachant les données, $\theta$ a 95% de probabilité d'être dans cet intervalle."* C'est une probabilité directe sur $\theta$, exactement ce qu'on veut intuitivement.

Deux façons standard de construire un intervalle à $(1 - \alpha)$ :

### A. Intervalle equal-tailed (à queues égales)

On coupe $\alpha/2$ de probabilité de chaque côté. Pour 95%, on prend les quantiles $q_{0.025}$ et $q_{0.975}$ du posterior.

$\mathrm{IC}_{\text{eq}} = [q_{\alpha/2}, \; q_{1-\alpha/2}].$

**Avantages :** facile à calculer, invariant par transformation monotone (si $[a, b]$ est equal-tailed pour $\theta$, alors $[\log a, \log b]$ l'est pour $\log \theta$).

### B. Intervalle HPD (Highest Posterior Density)

On prend l'intervalle le **plus court** qui contient $1 - \alpha$ de probabilité. Géométriquement : on "baisse une ligne horizontale" sur le graphe de la densité jusqu'à découper $1 - \alpha$ d'aire.

**Avantages :** plus court, et tout point dans l'intervalle a une densité supérieure à tout point en dehors. Plus naturel pour une densité asymétrique.

**Inconvénient :** pas invariant par transformation, et plus dur à calculer (numérique en général).

![[Pasted image 20260502193406.png]]
**Figure 2.** Pour une densité asymétrique : l'intervalle equal-tailed (en bleu) coupe des aires égales aux deux extrémités, mais peut être plus large que nécessaire. L'intervalle HPD (en vert) est plus court et concentré sur la zone de forte densité.

> [!example] Intervalles pour le posterior $f(\theta \mid Y=1) = 2\theta$
> On reprend l'exemple II.B (un lancer, un pile, prior uniforme).
>
> **Equal-tailed à 95%.** On résout $\mathbb{P}(\theta < q) = q^2 = 0.025$ et $q^2 = 0.975$ :
>
> $\mathrm{IC}_{\text{eq}} = [\sqrt{0.025}, \sqrt{0.975}] = [0.158, 0.987].$
>
> **HPD à 95%.** Comme la densité $2\theta$ est croissante, le maximum est en $\theta = 1$. L'HPD est $[q, 1]$ avec $\mathbb{P}(\theta > q) = 0.95$, soit $1 - q^2 = 0.95$ donc $q = \sqrt{0.05} \approx 0.224$ :
>
> $\mathrm{IC}_{\text{HPD}} = [0.224, 1.0].$
>
> L'HPD est plus court (0.776 contre 0.829) et reflète mieux le fait que le posterior penche fortement vers les grandes valeurs.

## V. MAP et lien avec la régularisation ML

Un estimateur ponctuel naturel à partir du posterior : son **mode**, c'est-à-dire le $\theta$ qui maximise le posterior. C'est le **Maximum A Posteriori** (MAP) :

$\hat{\theta}^{\text{MAP}} = \arg\max_\theta f(\theta \mid y) = \arg\max_\theta f(y \mid \theta) \, f(\theta)$

(la constante de normalisation disparaît dans l'argmax).

### A. Lien MAP - MLE

En passant au log :

$\hat{\theta}^{\text{MAP}} = \arg\max_\theta \Big[ \log f(y \mid \theta) + \log f(\theta) \Big] = \arg\max_\theta \Big[ \ell(\theta) + \log f(\theta) \Big]$

le premier terme est la **log-vraisemblance** (ce que maximise le MLE), et le second est un **terme de régularisation** induit par le prior.

> [!warning] Si le prior est uniforme, MAP = MLE
> Quand $f(\theta) \propto \text{cst}$ (prior plat), le second terme est constant en $\theta$ et disparaît. Donc MAP et MLE donnent la même valeur. Le bayésien avec prior plat retombe sur les estimateurs fréquentistes — mais il garde en plus toute la **distribution** posterior, pas juste le mode.

### B. Régularisation = MAP avec un prior

C'est le pont avec le ML moderne. Beaucoup de techniques de régularisation s'interprètent comme des MAP avec un prior particulier sur les paramètres.

| Technique ML | Pénalité | Prior bayésien équivalent |
|---|---|---|
| **Régression ridge / L2** | $\lambda \|\boldsymbol{\theta}\|_2^2$ | $\boldsymbol{\theta} \sim \mathcal{N}(0, \sigma^2 I)$ |
| **Lasso / L1** | $\lambda \|\boldsymbol{\theta}\|_1$ | $\boldsymbol{\theta}_i \sim \mathrm{Laplace}(0, b)$ |
| **Elastic Net** | $\lambda_1 \|\boldsymbol{\theta}\|_1 + \lambda_2 \|\boldsymbol{\theta}\|_2^2$ | mélange Laplace × Gaussien |
| **Early stopping** | implicite | approximativement gaussien |
| **Dropout** | implicite | approximation variationnelle |

**Lecture.** Quand tu fais de la ridge regression, tu fais (sans le savoir) une inférence MAP bayésienne avec un prior gaussien centré sur 0 sur tes coefficients. La force de la régularisation $\lambda$ correspond à l'inverse de la variance du prior $\sigma^2$ : plus $\lambda$ est grand, plus tu "crois" que les coefficients sont proches de zéro.

> [!note]- Pourquoi MAP n'est pas l'estimateur le plus utilisé en bayésien moderne
> Le MAP est souvent un mauvais représentant du posterior :
> - **Pas invariant par reparamétrisation** : MAP de $\theta$ ≠ MAP de $g(\theta)$, contrairement au MLE.
> - **Sensible à la dimension** : en haute dimension, le mode est souvent dans une région de faible "masse" (le volume autour du mode est petit). Tu maximises la densité mais tu rates le typical set.
> - **Cache l'incertitude** : on perd toute l'info de variance.
>
> En bayésien moderne on préfère la **moyenne posterior** $\mathbb{E}[\theta \mid y]$ ou des échantillons du posterior. Le MAP reste utile comme initialisation, comme pont vers la régularisation classique, ou en grande dim quand on ne peut rien d'autre.

## VI. Distributions prédictives

Le posterior $f(\theta \mid y)$ donne notre croyance sur $\theta$. Mais souvent, ce qu'on veut vraiment, c'est **prédire de nouvelles données** $\tilde{y}$. Deux distributions prédictives correspondantes.

### A. Prédictif a priori

Avant d'observer quoi que ce soit, on peut quand même prédire $y$ en marginalisant le prior :

$f(y) = \int f(y \mid \theta) \, f(\theta) \, \mathrm{d}\theta.$

C'est une loi sur $y$ qui prend en compte **toute notre incertitude sur $\theta$**.

> [!example] Prédictif a priori — combien de piles sur 10 lancers ?
> Prior $\theta \sim \mathrm{Unif}[0, 1]$ et $X \mid \theta \sim \mathrm{Bin}(10, \theta)$. Le prédictif a priori est :
>
> $f(x) = \int_0^1 \binom{10}{x} \theta^x (1-\theta)^{10-x} \, \mathrm{d}\theta.$
>
> En reconnaissant le noyau d'une loi $\mathrm{Beta}(x+1, 11-x)$ et en utilisant $\Gamma(n) = (n-1)!$ :
>
> $f(x) = \binom{10}{x} \cdot \frac{x! \, (10-x)!}{11!} = \frac{1}{11}, \qquad x \in \{0, 1, \dots, 10\}.$
>
> **Lecture.** Avec un prior totalement plat sur $\theta$, tous les nombres de piles entre 0 et 10 sont équiprobables. Ça paraît contre-intuitif (on s'attendrait à une cloche autour de 5), mais c'est cohérent : on n'a aucune info sur $\theta$, donc tous les scénarios sont possibles avec poids égal.

### B. Prédictif a posteriori

Après avoir observé $y$, on prédit une nouvelle observation $\tilde{y}$ en marginalisant le **posterior** :

$f(\tilde{y} \mid y) = \int f(\tilde{y} \mid \theta) \, f(\theta \mid y) \, \mathrm{d}\theta.$

Même structure que le prédictif a priori, mais avec le posterior à la place du prior.

> [!example] Prédictif a posteriori — règle de succession de Laplace
> Prior plat sur $\theta$, on observe **un pile** ($y_1 = 1$). Quel est le posterior pour le second lancer ?
>
> Le posterior est $f(\theta \mid y_1 = 1) = 2\theta$ (vu en II.B). Donc :
>
> $\mathbb{P}(Y_2 = 1 \mid Y_1 = 1) = \int_0^1 \theta \cdot 2\theta \, \mathrm{d}\theta = \frac{2}{3}.$
>
> **Lecture.** Après un seul pile observé, on prédit un pile suivant avec probabilité 2/3, pas 1 (on n'est pas certain) ni 1/2 (on a quand même appris quelque chose). C'est la **règle de succession de Laplace** : un prior uniforme "vaut" deux observations virtuelles (un pile + un face), donc après un vrai pile on a virtuellement 2 piles sur 3 lancers.

## VII. Conjugaison

La **conjugaison** est le concept qui rend l'inférence bayésienne tractable analytiquement dans des cas simples.

### A. Le concept

**Définition.** Une famille de distributions $\mathcal{F}$ est dite **conjuguée** pour une vraisemblance $f(y \mid \theta)$ si :

$f(\theta) \in \mathcal{F} \quad \Longrightarrow \quad f(\theta \mid y) \in \mathcal{F}.$

Autrement dit : si on choisit le prior dans $\mathcal{F}$, le posterior reste dans $\mathcal{F}$. On garde la même *forme* paramétrique, seuls les paramètres changent.

**Pourquoi c'est utile.**
1. **Pas d'intégrale à calculer** pour la constante de normalisation : on reconnaît la forme du posterior et on lit ses paramètres.
2. **Mises à jour séquentielles faciles** : le posterior d'aujourd'hui devient le prior de demain, qui est encore dans $\mathcal{F}$.
3. **Interprétation explicite** : les paramètres du posterior sont (presque toujours) un mélange linéaire des paramètres du prior et de statistiques sur les données.

### B. Tableau récapitulatif

| Vraisemblance | Paramètre | Prior conjugué | Posterior |
|---|---|---|---|
| Bernoulli($\theta$) / Binomiale | $\theta \in [0,1]$ | $\mathrm{Beta}(\alpha, \beta)$ | $\mathrm{Beta}(\alpha + \sum y_i, \, \beta + n - \sum y_i)$ |
| Poisson($\lambda$) | $\lambda > 0$ | $\mathrm{Gamma}(\alpha, \beta)$ | $\mathrm{Gamma}(\alpha + \sum y_i, \, \beta + n)$ |
| Exponentielle($\lambda$) | $\lambda > 0$ | $\mathrm{Gamma}(\alpha, \beta)$ | $\mathrm{Gamma}(\alpha + n, \, \beta + \sum y_i)$ |
| Normale($\mu$, $\sigma_0^2$) — $\sigma_0^2$ connu | $\mu \in \mathbb{R}$ | $\mathcal{N}(m_0, s_0^2)$ | $\mathcal{N}(m_n, s_n^2)$ (cf. VII.G) |
| Normale($\mu$, $\sigma^2$) — les deux inconnus | $(\mu, \sigma^2)$ | Normale-Gamma inverse | Normale-Gamma inverse |
| Multinomiale | $(p_1, \dots, p_k)$ | Dirichlet | Dirichlet |
| Gaussienne multivariée | $\Sigma^{-1}$ | Wishart | Wishart |

Les 4 premiers cas sont détaillés ci-dessous. Les 3 derniers sont des extensions multivariées qu'on retrouvera dans des fichiers ultérieurs.

### C. Bernoulli / Binomiale → Beta

C'est *le* cas canonique à comprendre en détail, parce que la mécanique se généralise à tous les autres.

> [!note]- Rappel — la distribution Beta
> Pour $\theta \in [0, 1]$, $\alpha, \beta > 0$ :
>
> $\mathrm{Beta}(\theta; \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} \, \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}.$
>
> **Moyenne :** $\mathbb{E}[\theta] = \dfrac{\alpha}{\alpha + \beta}$.
> **Variance :** $\mathrm{Var}[\theta] = \dfrac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$.
>
> **Formes typiques :**
> - $\alpha = \beta = 1$ : densité plate (Uniforme[0,1]).
> - $\alpha = \beta > 1$ : symétrique, en cloche, centrée en 1/2.
> - $\alpha = \beta < 1$ : en U (deux pics aux bords).
> - $\alpha > \beta$ : penche vers 1.
> - $\alpha < \beta$ : penche vers 0.
>
> ![[beta_shapes.png]]
> **Figure 3.** Les cinq régimes de la Beta selon $(\alpha, \beta)$.

**Setup.** $Y_i \overset{iid}{\sim} \mathrm{Bernoulli}(\theta)$ pour $i = 1, \dots, n$, prior $\theta \sim \mathrm{Beta}(\alpha, \beta)$.

**Vraisemblance.** En posant $s = \sum y_i$ (nombre de succès) :

$f(\boldsymbol{y} \mid \theta) = \prod_{i=1}^n \theta^{y_i} (1 - \theta)^{1 - y_i} = \theta^s (1-\theta)^{n - s}.$

**Posterior** par Bayes (à constante près) :

$f(\theta \mid \boldsymbol{y}) \propto \theta^s (1-\theta)^{n-s} \cdot \theta^{\alpha - 1} (1 - \theta)^{\beta - 1} = \theta^{\alpha + s - 1} (1 - \theta)^{\beta + n - s - 1}.$

On reconnaît le noyau d'une Beta. Donc :

$\boxed{\; \theta \mid \boldsymbol{y} \;\sim\; \mathrm{Beta}(\alpha + s, \; \beta + n - s) \;}$

**Lecture des paramètres.**
- $\alpha + s$ = $\alpha$ (succès virtuels du prior) + $s$ (succès observés)
- $\beta + n - s$ = $\beta$ (échecs virtuels du prior) + $n - s$ (échecs observés)

Le prior $\mathrm{Beta}(\alpha, \beta)$ se comporte comme **$\alpha + \beta$ observations virtuelles**, dont $\alpha$ succès et $\beta$ échecs. C'est la notion de **taille d'échantillon effective** (effective sample size, ESS) du prior.

### D. Moyenne posterior et taille d'échantillon effective

**Décomposition fondamentale.** La moyenne posterior s'écrit comme une **moyenne pondérée** entre la moyenne du prior et la moyenne empirique des données :

$\mathbb{E}[\theta \mid \boldsymbol{y}] = \frac{\alpha + s}{\alpha + \beta + n} = \underbrace{\frac{\alpha + \beta}{\alpha + \beta + n}}_{\text{poids prior}} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{moy. prior}} + \underbrace{\frac{n}{\alpha + \beta + n}}_{\text{poids données}} \cdot \underbrace{\frac{s}{n}}_{\text{moy. empirique}}.$

Les poids somment à 1 et dépendent du rapport $n / (\alpha + \beta)$.

**Conséquences pratiques.**
- Si $n \gg \alpha + \beta$ → le posterior est dominé par les données, le prior n'a quasi pas d'influence.
- Si $n \ll \alpha + \beta$ → le posterior est dominé par le prior, les données comptent peu.
- Le **break-even** est à $n = \alpha + \beta$.

> 💡 **L'idée en une phrase.** Choisir un prior $\mathrm{Beta}(\alpha, \beta)$, c'est dire à ton modèle : *"avant de voir tes données, fais comme si tu avais déjà vu $\alpha + \beta$ observations passées, dont $\alpha$ succès"*. La force du prior est donc *quantifiable* en équivalent-données.

### E. Poisson → Gamma

> [!note]- Rappel — la distribution Gamma
> Pour $x > 0$, $\alpha, \beta > 0$ :
>
> $\mathrm{Gamma}(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} \, x^{\alpha - 1} e^{-\beta x}.$
>
> $\alpha$ s'appelle le **shape**, $\beta$ le **rate** (attention : certaines libs utilisent $1/\beta$ comme "scale").
>
> **Moyenne :** $\mathbb{E}[X] = \alpha / \beta$.
> **Variance :** $\mathrm{Var}[X] = \alpha / \beta^2$.
>
> **Formes :** asymétrique sur $\mathbb{R}_+$, queue droite. Plus $\alpha$ est grand, plus elle ressemble à une gaussienne. Cas particuliers : $\alpha = 1$ donne l'exponentielle, $\alpha = k/2$ avec $\beta = 1/2$ donne la chi-deux à $k$ degrés de liberté.

**Setup.** $Y_i \overset{iid}{\sim} \mathrm{Poisson}(\lambda)$, prior $\lambda \sim \mathrm{Gamma}(\alpha, \beta)$.

> [!note]- Dérivation du posterior
> Vraisemblance :
>
> $f(\boldsymbol{y} \mid \lambda) = \prod_{i=1}^n \frac{\lambda^{y_i} e^{-\lambda}}{y_i!} \propto \lambda^{\sum y_i} e^{-n\lambda}.$
>
> Posterior :
>
> $f(\lambda \mid \boldsymbol{y}) \propto \lambda^{\sum y_i} e^{-n\lambda} \cdot \lambda^{\alpha - 1} e^{-\beta \lambda} = \lambda^{\alpha + \sum y_i - 1} e^{-(\beta + n)\lambda}.$
>
> On reconnaît une Gamma.

**Résultat :**

$\boxed{\; \lambda \mid \boldsymbol{y} \;\sim\; \mathrm{Gamma}\!\left(\alpha + \sum y_i, \; \beta + n\right) \;}$

**Décomposition de la moyenne posterior :**

$\mathbb{E}[\lambda \mid \boldsymbol{y}] = \frac{\alpha + \sum y_i}{\beta + n} = \frac{\beta}{\beta + n} \cdot \frac{\alpha}{\beta} + \frac{n}{\beta + n} \cdot \frac{\sum y_i}{n}.$

La taille d'échantillon effective du prior est **$\beta$** (le paramètre de rate).

### F. Exponentielle → Gamma

**Setup.** $Y_i \overset{iid}{\sim} \mathrm{Exp}(\lambda)$, prior $\lambda \sim \mathrm{Gamma}(\alpha, \beta)$.

> [!note]- Dérivation du posterior
> Vraisemblance :
>
> $f(\boldsymbol{y} \mid \lambda) = \prod_{i=1}^n \lambda e^{-\lambda y_i} = \lambda^n e^{-\lambda \sum y_i}.$
>
> Posterior :
>
> $f(\lambda \mid \boldsymbol{y}) \propto \lambda^n e^{-\lambda \sum y_i} \cdot \lambda^{\alpha - 1} e^{-\beta \lambda} = \lambda^{\alpha + n - 1} e^{-(\beta + \sum y_i)\lambda}.$

**Résultat :**

$\boxed{\; \lambda \mid \boldsymbol{y} \;\sim\; \mathrm{Gamma}\!\left(\alpha + n, \; \beta + \sum y_i\right) \;}$

Noter que les rôles sont **différents** par rapport au cas Poisson : ici $n$ s'ajoute au shape, et $\sum y_i$ au rate. C'est cohérent avec le sens des paramètres : pour l'exponentielle, $1/\lambda$ est le temps moyen entre événements, donc plus on cumule de temps observé ($\sum y_i$), plus on a d'info sur $\lambda$.

### G. Normale (variance connue) → Normale

**Setup.** $X_i \overset{iid}{\sim} \mathcal{N}(\mu, \sigma_0^2)$ avec $\sigma_0^2$ connu, prior $\mu \sim \mathcal{N}(m_0, s_0^2)$.

**Résultat.** Le posterior est gaussien :

$\mu \mid \boldsymbol{x} \;\sim\; \mathcal{N}(m_n, s_n^2)$

avec

$m_n = \frac{\frac{n \bar{x}}{\sigma_0^2} + \frac{m_0}{s_0^2}}{\frac{n}{\sigma_0^2} + \frac{1}{s_0^2}}, \qquad \frac{1}{s_n^2} = \frac{n}{\sigma_0^2} + \frac{1}{s_0^2}.$

> [!note]- Lecture des paramètres en termes de précisions
> En posant la **précision** $\tau = 1/\sigma^2$ (l'inverse de la variance), les formules deviennent élégantes :
>
> $\tau_n = \tau_0 + n \tau_{\text{data}}, \qquad m_n = \frac{\tau_0 m_0 + n \tau_{\text{data}} \bar{x}}{\tau_n}.$
>
> - **Les précisions s'additionnent.** L'info totale du posterior = info du prior + $n$ fois l'info d'une observation.
> - **La moyenne posterior** est une moyenne pondérée par les précisions de la moyenne du prior et de la moyenne empirique.
>
> Taille d'échantillon effective du prior : $\sigma_0^2 / s_0^2$ (le ratio des variances). Plus le prior est "serré" ($s_0^2$ petit), plus il a de poids.

> [!note]- Dérivation par complétion du carré
> On veut montrer que $f(\mu \mid \boldsymbol{x}) \propto f(\boldsymbol{x} \mid \mu) f(\mu)$ a la forme d'une gaussienne.
>
> Vraisemblance :
>
> $f(\boldsymbol{x} \mid \mu) \propto \exp\left(-\frac{1}{2 \sigma_0^2} \sum (x_i - \mu)^2\right) \propto \exp\left(-\frac{n}{2 \sigma_0^2} (\mu - \bar{x})^2\right).$
>
> Prior :
>
> $f(\mu) \propto \exp\left(-\frac{1}{2 s_0^2} (\mu - m_0)^2\right).$
>
> Produit : on factorise les termes en $\mu^2$ et $\mu$ dans l'exponentielle :
>
> $\text{exposant} = -\frac{1}{2}\left[\left(\frac{n}{\sigma_0^2} + \frac{1}{s_0^2}\right) \mu^2 - 2 \left(\frac{n \bar{x}}{\sigma_0^2} + \frac{m_0}{s_0^2}\right) \mu + \text{cst}\right].$
>
> En posant $1/s_n^2 = n/\sigma_0^2 + 1/s_0^2$ et $m_n$ comme dans le résultat, on obtient $-\frac{1}{2 s_n^2}(\mu - m_n)^2 + \text{cst}$. Donc le posterior est bien $\mathcal{N}(m_n, s_n^2)$. $\blacksquare$

## VIII. Priors non-informatifs

Quand on n'a aucune information a priori, on aimerait un prior qui *laisse parler les données*. Plusieurs notions, de plus en plus radicales.

### A. Prior vague

Un prior **vague** est très étalé : sa contribution au posterior est négligeable devant celle des données. Exemples : $\mathcal{N}(0, 10^6)$ pour un paramètre réel, $\mathrm{Beta}(0.001, 0.001)$ pour une probabilité.

Mathématiquement c'est encore un vrai prior (intégrable), mais en pratique il est dominé par les données dès que $n$ n'est pas minuscule.

### B. Prior impropre

À la limite, on peut envisager un "prior" qui n'est **pas une vraie densité** : son intégrale diverge.

**Exemples canoniques :**
- $f(\mu) \propto 1$ (uniforme sur $\mathbb{R}$ entier — non normalisable).
- $f(\sigma^2) \propto 1/\sigma^2$ (uniforme sur $\log \sigma^2$, non normalisable).
- $f(\theta) \propto \theta^{-1}(1-\theta)^{-1}$ ("Beta(0,0)", non normalisable).

> [!warning] Un prior impropre peut produire un posterior propre
> C'est le miracle qui justifie l'usage des priors impropres. Tant que $\int f(y \mid \theta) f(\theta) \, \mathrm{d}\theta < \infty$, le posterior $f(\theta \mid y) \propto f(y \mid \theta) f(\theta)$ est bien défini comme densité, même si $f(\theta)$ ne l'est pas. **Mais il faut vérifier au cas par cas** : certains modèles avec prior impropre donnent un posterior impropre, ce qui est catastrophique (l'inférence n'a plus de sens).

**Exemple gaussien.** Avec $X_i \sim \mathcal{N}(\mu, \sigma_0^2)$ et prior impropre $f(\mu) \propto 1$, le posterior est $\mu \mid \boldsymbol{x} \sim \mathcal{N}(\bar{x}, \sigma_0^2 / n)$. Sa moyenne $\bar{x}$ est exactement le MLE fréquentiste — mais on a en plus l'incertitude sous forme de distribution.

### C. Prior de Jeffreys

Problème : un prior "plat" dépend de la **paramétrisation**. $f(\sigma) \propto 1$ et $f(\sigma^2) \propto 1$ ne sont *pas* le même prior (il faut un jacobien pour passer de l'un à l'autre). Comment choisir un prior "objectif" qui ne dépende pas du choix de paramétrisation ?

**Le prior de Jeffreys.** Défini par

$f_J(\theta) \propto \sqrt{I(\theta)}$

où $I(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log f(y \mid \theta)}{\partial \theta^2}\right]$ est l'**information de Fisher**.

**Propriété clé.** Le prior de Jeffreys est **invariant par reparamétrisation** : si on change $\theta \to \phi = g(\theta)$, le jacobien introduit dans la transformation se compense exactement avec celui de l'information de Fisher. C'est le seul prior "non-informatif" qui a cette propriété.

**Exemples :**
- **Bernoulli** $Y_i \sim \mathrm{Ber}(\theta)$ : Jeffreys = $\mathrm{Beta}(1/2, 1/2)$. Cas rare où Jeffreys est *propre*. ESS = 1.
- **Normale** ($\mu$ connu, $\sigma^2$ inconnu) : Jeffreys = $f(\sigma^2) \propto 1/\sigma^2$. Impropre.
- **Normale** ($\sigma$ connu, $\mu$ inconnu) : Jeffreys = $f(\mu) \propto 1$. Impropre.

> [!note]- Autres approches "objectives"
> - **Reference priors** (Bernardo) : généralisation de Jeffreys, optimale au sens de l'information de Kullback-Leibler entre prior et posterior.
> - **Priors d'entropie maximale** (Jaynes) : on choisit le prior qui maximise l'entropie sous certaines contraintes (e.g. moyenne fixée).
> - **Empirical Bayes** : on estime les hyperparamètres du prior à partir des données. Pratique mais utilise les données deux fois — peut sous-estimer l'incertitude.

## IX. À retenir

> [!summary] Les idées centrales
> 1. **Bayésien = paramètres traités comme v.a.** Le posterior $f(\theta \mid y) \propto f(y \mid \theta) f(\theta)$ donne une distribution complète, pas juste un point estimé.
> 2. **Conjugaison = posterior dans la même famille que le prior**, ce qui rend l'inférence analytique. Beta-Bernoulli, Gamma-Poisson, Gamma-Exp, Normal-Normal sont les paires de base.
> 3. **Moyenne posterior = moyenne pondérée du prior et des données**, avec une taille d'échantillon effective ($\alpha+\beta$ pour Beta, $\beta$ pour Gamma, $\sigma_0^2/s_0^2$ pour Normal). Plus de données → moins de poids au prior.
> 4. **MAP = MLE + régularisation par le prior.** Ridge = MAP gaussien, Lasso = MAP Laplace. C'est le pont avec le ML classique.
> 5. **Intervalles de crédibilité ≠ intervalles de confiance.** Le bayésien permet de dire "$\theta$ est dans $[a, b]$ avec probabilité 95%", chose impossible chez le fréquentiste.
> 6. **Priors non-informatifs** (vagues, impropres, Jeffreys) permettent de faire de l'inférence bayésienne quand on n'a pas d'info a priori, souvent en retombant sur des estimateurs fréquentistes — mais en gardant la distribution complète.
