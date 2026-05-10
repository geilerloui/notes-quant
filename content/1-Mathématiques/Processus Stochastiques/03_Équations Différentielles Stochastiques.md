---
title: Équations Différentielles Stochastiques
date: 2026-05-09
tags: [probabilités, processus-stochastiques, EDS, finance]
---

## L'idée fondatrice

Une **équation différentielle stochastique (EDS)** décrit la dynamique d'un processus aléatoire. La forme générale :

$$dX_t = \underbrace{a(X_t, t)\,dt}_{\text{drift}} + \underbrace{b(X_t, t)\,dW_t}_{\text{diffusion}}$$

- Le **drift** $a(X_t, t)$ est la tendance déterministe (ce que ferait $X_t$ sans bruit)
- La **diffusion** $b(X_t, t)$ est l'intensité du bruit brownien à chaque instant

C'est un **raccourci de notation** pour l'équation intégrale $X_t = X_0 + \int_0^t a(X_s, s)\,ds + \int_0^t b(X_s, s)\,dW_s$ (cf. [[Processus Stochastique]]).

Cette note présente les trois EDS les plus utilisées en finance : **ABM** (drift + bruit constants), **MBG** (Black-Scholes), **OU** (mean-reverting). Pour chacune : équation, intuition, solution, distribution, calibration, simulation.

## I. Mouvement brownien arithmétique (ABM)

### Équation

$$dX_t = \mu\,dt + \sigma\,dW_t$$

Drift et diffusion **constants**. Le processus dérive linéairement avec des fluctuations gaussiennes autour.

### Solution

Les variables se séparent immédiatement (pas besoin d'Itô) :

$$X_T = X_0 + \mu T + \sigma W_T$$

> [!note]- Démonstration
> On intègre directement entre $0$ et $T$ :
> $$\int_0^T dX_t = \int_0^T \mu\,dt + \int_0^T \sigma\,dW_t$$
> Le premier terme donne $X_T - X_0$, le deuxième $\mu T$, le troisième $\sigma(W_T - W_0) = \sigma W_T$ car $W_0 = 0$. $\square$

### Distribution

Comme $W_T \sim \mathcal{N}(0, T)$ et que $X_T$ est une fonction affine de $W_T$ :

$$X_T \sim \mathcal{N}\bigl(X_0 + \mu T,\ \sigma^2 T\bigr)$$

| Quantité | Valeur | Lecture |
|:---:|:---:|:---:|
| $\mathbb{E}[X_T]$ | $X_0 + \mu T$ | la moyenne **dérive linéairement** avec $T$ |
| $\mathrm{Var}[X_T]$ | $\sigma^2 T$ | la variance **grandit linéairement** avec $T$ |
| $\mathrm{Cov}[X_T, X_S]$ | $\sigma^2 \min(T, S)$ | propriété d'un brownien |

L'écart-type grandit en $\sqrt{T}$ : à long terme, le drift $\mu T$ domine le bruit $\sigma\sqrt{T}$ (pour $T$ grand).

![[fig_eds_abm_evolution.png]]
*Figure. Évolution de la distribution $\mathcal{N}(X_0 + \mu T, \sigma^2 T)$ avec $T$ (paramètres FTSE). À mesure que $T$ grandit : la moyenne dérive vers la droite (drift positif), et la distribution s'aplatit (la variance grandit en $T$, donc l'écart-type en $\sqrt{T}$).*

### Calibration (exemple FTSE 100)

Sur le FTSE 100 entre le 13-sept-2018 et le 5-oct-2018, on calcule les différences journalières $\Delta X_i = X_i - X_{i-1}$. Sous l'hypothèse ABM, ces différences sont iid gaussiennes :

$$\Delta X_i \sim \mathcal{N}(m, v)$$

Estimateurs empiriques :

$$m = \frac{1}{n}\sum_{i=1}^n \Delta X_i = 2.3, \qquad v = \frac{1}{n-1}\sum_{i=1}^n (\Delta X_i - m)^2 = 29\,332.2$$

Soit $\sigma \approx 54.1$ par jour.

**Annualisation** (252 jours de trading par an, incréments indépendants) :

$$\mu_a = 252 \cdot m \approx 581.2, \qquad \sigma_a^2 = 252 \cdot v \approx 738\,912 \;\Rightarrow\; \sigma_a \approx 859.6$$

> [!note]- Pourquoi cette annualisation ?
> En continu, $t = 1$ correspond à 1 an. Sur un pas $\Delta t = 1/252$ (un jour) :
> - moyenne : $m_{daily} = \mu_a \cdot \Delta t \;\Rightarrow\; \mu_a = m_{daily} / \Delta t$
> - variance : $v_{daily} = \sigma_a^2 \cdot \Delta t \;\Rightarrow\; \sigma_a^2 = v_{daily} / \Delta t$

![[fig_eds_abm_trajectoires.png]]
*Figure. 80 trajectoires d'ABM simulées à partir de $X_0 = 7\,318.5$ avec les paramètres FTSE calibrés, sur 50 jours. À gauche, les trajectoires divergent autour de la moyenne théorique $X_0 + \mu t$ (rouge). À droite, la gaussienne $\mathcal{N}(X_0 + \mu T, \sigma^2 T)$ tournée à 90° colle à la distribution finale des trajectoires.*

### Simulation

Schéma d'Euler discret :

$$X_{t+\Delta t} = X_t + \mu\,\Delta t + \sigma\,\sqrt{\Delta t}\,Z, \qquad Z \sim \mathcal{N}(0, 1)$$

Avec les paramètres FTSE et $\Delta t = 1/252$ :

$$X_{t+\Delta t} = X_t + 581\,\Delta t + 860\,\sqrt{\Delta t}\,Z$$

Partant de $X_0 = 7\,318.5$ (dernière valeur de l'index), on génère 50 jours en avant : $X_{1/252}, X_{2/252}, \dots, X_{50/252}$.

## II. Mouvement brownien géométrique (MBG)

### Équation

$$dX_t = \mu X_t\,dt + \sigma X_t\,dW_t$$

Drift et diffusion **proportionnels** au prix actuel. C'est le modèle **Black-Scholes** : un actif dont le rendement (et la volatilité) sont en pourcentage du prix, pas en valeur absolue.

### Intuition

Pourquoi proportionnel ? Sur un actif à 100€, une variation de 1€ est petite ; sur le même actif à 10€, c'est énorme. Le MBG modélise des variations en **pourcentage**, pas en montant absolu — c'est ce qui assure que le prix reste **toujours positif** (un actif ne peut pas devenir négatif).

![[fig_eds_mbg_vs_abm.png]]
*Figure. Comparaison ABM vs MBG sur les mêmes incréments browniens. Avec un drift relatif de $5\%$ et une vol relative de $30\%$ pour le MBG, et des valeurs absolues équivalentes pour l'ABM, on voit que **le MBG reste toujours positif** alors que **l'ABM peut traverser zéro**. C'est pourquoi on n'utilise pas l'ABM pour des prix.*

### Solution

On applique la formule d'Itô à $f(X_t) = \log X_t$ (cf. [[Processus Stochastique]]). Le résultat :

$$X_T = X_0 \exp\Bigl[\bigl(\mu - \tfrac{\sigma^2}{2}\bigr) T + \sigma W_T\Bigr]$$

> [!note]- Démonstration via Itô
> Pose $Y_t = \log X_t$. Avec $f(x) = \log x$, $f'(x) = 1/x$, $f''(x) = -1/x^2$, la formule d'Itô donne :
> $$dY_t = \frac{1}{X_t}\,dX_t - \frac{1}{2}\frac{1}{X_t^2}\,(dX_t)^2$$
> En substituant $dX_t = \mu X_t dt + \sigma X_t dW_t$ et $(dX_t)^2 = \sigma^2 X_t^2 dt$ :
> $$dY_t = (\mu - \tfrac{\sigma^2}{2})\,dt + \sigma\,dW_t$$
> C'est un ABM sur $Y_t = \log X_t$. On intègre :
> $$\log X_T - \log X_0 = (\mu - \tfrac{\sigma^2}{2}) T + \sigma W_T$$
> Et on prend l'exponentielle. $\square$

### Distribution

Comme $\log X_T$ est gaussienne, $X_T$ suit une loi **log-normale** :

$$\log X_T \sim \mathcal{N}\bigl(\log X_0 + (\mu - \tfrac{\sigma^2}{2}) T,\ \sigma^2 T\bigr)$$

Soit en notation log-normale : $X_T \sim \mathcal{LN}(\log X_0 + (\mu - \tfrac{\sigma^2}{2})T,\ \sigma^2 T)$.

| Quantité | Valeur |
|:---:|:---:|
| $\mathbb{E}[X_T]$ | $X_0\,e^{\mu T}$ |
| $\mathrm{Var}[X_T]$ | $X_0^2\,e^{2\mu T}\bigl(e^{\sigma^2 T} - 1\bigr)$ |
| $\mathrm{Cov}[X_T, X_S]$ | $X_0^2\,e^{\mu(T+S)}\bigl(e^{\sigma^2 \min(T,S)} - 1\bigr)$ |

**Important** : le prix moyen croît à $X_0 e^{\mu T}$ (pas $X_0 e^{(\mu - \sigma^2/2)T}$). Le terme $-\sigma^2/2$ apparaît dans la **médiane** du log-prix, pas dans la moyenne du prix.

### Calibration (exemple FTSE 100)

Même méthode que l'ABM, mais sur les **log-rendements** $r_i = \log(X_i / X_{i-1})$ au lieu des différences. On obtient (sur les mêmes données FTSE) :

$$\mu \approx 0.0865, \qquad \sigma^2 \approx 0.013, \qquad \sigma \approx 0.114$$

(en pourcentage annualisé : drift $\approx 8.65\%$, vol $\approx 11.4\%$).

![[fig_eds_mbg_trajectoires.png]]
*Figure. 80 trajectoires de MBG sur 50 jours avec les paramètres FTSE calibrés. À droite, la distribution log-normale $X_T \sim \mathcal{LN}$ tournée à 90° — légèrement asymétrique (queue à droite plus longue), c'est la signature d'une log-normale.*

### Simulation

$$X_{t+\Delta t} = X_t \cdot \exp\Bigl[(\mu - \tfrac{\sigma^2}{2})\,\Delta t + \sigma\sqrt{\Delta t}\,Z\Bigr], \qquad Z \sim \mathcal{N}(0,1)$$

Avec FTSE : $X_{t+\Delta t} = X_t\,e^{0.08\,\Delta t + \sqrt{0.013}\,\sqrt{\Delta t}\,Z}$.

## III. Processus d'Ornstein-Uhlenbeck (OU)

### Équation

$$dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$$

Le drift n'est **plus constant** : il dépend de l'écart entre $X_t$ et un niveau cible $\theta$. C'est un processus **mean-reverting** : il revient toujours vers sa moyenne $\theta$.

### Intuition (mean-reversion)

- Si $X_t > \theta$ : le drift $\kappa(\theta - X_t) < 0$ → le processus est **tiré vers le bas**, vers $\theta$
- Si $X_t < \theta$ : le drift $\kappa(\theta - X_t) > 0$ → le processus est **tiré vers le haut**, vers $\theta$

| Paramètre | Rôle |
|:---:|:---|
| $\theta$ | **niveau de retour** (moyenne asymptotique) |
| $\kappa$ | **vitesse de mean-reversion** (plus grand = retour plus rapide) |
| $\sigma$ | intensité du bruit |

Pour $\sigma$ très grand, le drift devient négligeable et on retrouve un brownien rescalé. Le mean-reverting modélise des taux d'intérêt, des spreads de crédit, des écarts de prix entre actifs corrélés (pairs trading), de la volatilité (dans Heston).

![[fig_eds_ou_kappa.png]]
*Figure. Effet du paramètre $\kappa$ sur la vitesse de mean-reversion. Toutes les trajectoires partent de $X_0 = 5$ et ont $\theta = 0$. Avec $\kappa = 0.5$ (gauche), le retour vers $\theta$ est lent et les trajectoires divergent encore après 5 unités de temps. Avec $\kappa = 10$ (droite), le retour est quasi-instantané. La half-life $H = \log 2 / \kappa$ chute de 1.39 à 0.07.*

### Solution

$$X_T = X_0 e^{-\kappa T} + \theta(1 - e^{-\kappa T}) + \sigma\int_0^T e^{-\kappa(T-t)}\,dW_t$$

> [!note]- Démonstration via facteur intégrant
> On réécrit $dX_t + \kappa X_t\,dt = \kappa\theta\,dt + \sigma\,dW_t$. On multiplie par le facteur intégrant $e^{\kappa t}$ :
> $$e^{\kappa t}dX_t + \kappa e^{\kappa t} X_t\,dt = \kappa\theta e^{\kappa t}\,dt + \sigma e^{\kappa t}\,dW_t$$
> Le membre de gauche est exactement $d(e^{\kappa t} X_t)$ (règle du produit, le terme croisé est nul car $e^{\kappa t}$ est déterministe). On intègre entre $0$ et $T$ :
> $$e^{\kappa T} X_T - X_0 = \theta(e^{\kappa T} - 1) + \sigma\int_0^T e^{\kappa t}\,dW_t$$
> Et on multiplie par $e^{-\kappa T}$ pour isoler $X_T$. $\square$

### Distribution

$X_T$ est gaussienne (somme d'une partie déterministe et d'une intégrale d'Itô gaussienne) :

$$X_T \sim \mathcal{N}\bigl(\mathbb{E}[X_T],\ \mathrm{Var}[X_T]\bigr)$$

avec :

| Quantité | Valeur | Limite quand $T \to \infty$ |
|:---:|:---:|:---:|
| $\mathbb{E}[X_T]$ | $X_0 e^{-\kappa T} + \theta(1 - e^{-\kappa T})$ | $\theta$ |
| $\mathrm{Var}[X_T]$ | $\dfrac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa T}\bigr)$ | $\dfrac{\sigma^2}{2\kappa}$ |

**Le processus a une distribution stationnaire** : $\mathcal{N}(\theta,\ \sigma^2/(2\kappa))$. C'est ce qui distingue OU des deux précédents (où la variance grandit indéfiniment).

![[fig_eds_ou_stationnaire.png]]
*Figure. 100 trajectoires d'OU partant de $X_0 = 3$ avec $\theta = 0$, $\kappa = 1$, $\sigma = 1$. Les trajectoires convergent rapidement vers une bande autour de $\theta$ (lignes rouges pointillées à $\pm \sigma_\infty = \pm 0.71$ et $\pm 2\sigma_\infty$). À droite, la gaussienne stationnaire $\mathcal{N}(\theta, \sigma^2/(2\kappa))$ — toujours la même quel que soit $T$ une fois que le régime stationnaire est atteint.*

### Half-life

Combien de temps pour que la moyenne parcoure la moitié du chemin entre $X_0$ et $\theta$ ? On résout $\mathbb{E}[X_H] = (X_0 + \theta)/2$ et on obtient :

$$H = \frac{\log 2}{\kappa}$$

La half-life ne dépend **que de $\kappa$**. Pour $\kappa = 0.5$ → $H \approx 1.4$ ; pour $\kappa = 5$ → $H \approx 0.14$.

### Lien avec AR(1) — important

Le schéma d'Euler discret de l'OU est **exactement** un processus autorégressif AR(1) :

$$X_{t+\Delta t} = X_t e^{-\kappa\Delta t} + \theta(1 - e^{-\kappa\Delta t}) + \sigma\sqrt{\frac{1 - e^{-2\kappa\Delta t}}{2\kappa}}\,Z$$

En posant $b = e^{-\kappa\Delta t}$, $a = \theta(1 - b)$, $\mathrm{SE} = \sigma\sqrt{(1-b^2)/(2\kappa)}$, ça devient :

$$X_{t+\Delta t} = b\,X_t + a + \mathrm{SE}\cdot Z$$

C'est la forme classique d'un **AR(1)**. Donc on peut **calibrer un OU par régression linéaire** : régresser $X_{t+\Delta t}$ sur $X_t$ pour obtenir $a, b, \mathrm{SE}$, puis inverser :

$$\kappa = -\frac{\log b}{\Delta t}, \qquad \theta = \frac{a}{1-b}, \qquad \sigma = \mathrm{SE}\sqrt{\frac{-2\log b}{(1-b^2)\,\Delta t}}$$

Sur les données FTSE : $a = 17.643$, $b = 0.781$, $\mathrm{SE} = 17.637$, soit $\kappa \approx 0.494$, $\theta \approx 80.66$, $\sigma \approx 28.08$.

### Simulation

On utilise directement la mise à jour AR(1) ci-dessus. Plus stable que le schéma d'Euler naïf $X_{t+\Delta t} = X_t + \kappa(\theta - X_t)\Delta t + \sigma\sqrt{\Delta t}\,Z$ pour $\kappa\Delta t$ pas trop petit.

## Récapitulatif

| | **ABM** | **MBG** | **OU** |
|---|---|---|---|
| **Équation** | $dX_t = \mu\,dt + \sigma\,dW_t$ | $dX_t = \mu X_t\,dt + \sigma X_t\,dW_t$ | $dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$ |
| **Solution** | $X_0 + \mu T + \sigma W_T$ | $X_0\,e^{(\mu - \sigma^2/2)T + \sigma W_T}$ | voir ci-dessus |
| **Distribution** | gaussienne | log-normale | gaussienne |
| **$\mathbb{E}[X_T]$** | $X_0 + \mu T$ | $X_0\,e^{\mu T}$ | tend vers $\theta$ |
| **$\mathrm{Var}[X_T]$** | $\sigma^2 T$ | $X_0^2 e^{2\mu T}(e^{\sigma^2 T} - 1)$ | tend vers $\sigma^2/(2\kappa)$ |
| **Stationnaire ?** | non (variance $\to \infty$) | non | **oui** |
| **Domaine** | $\mathbb{R}$ | $\mathbb{R}_+^*$ | $\mathbb{R}$ |
| **Usage typique** | différences de prix | actions (Black-Scholes) | taux, spreads, vol, pairs trading |
| **Discret** | random walk avec drift | random walk multiplicatif | AR(1) |

## Pour aller plus loin

- **Vasicek** : OU appliqué aux taux d'intérêt — $dr_t = \kappa(\theta - r_t)\,dt + \sigma\,dW_t$. Identique à OU mathématiquement, mais utilisé pour pricer des obligations.
- **CIR (Cox-Ingersoll-Ross)** : variante de Vasicek avec diffusion en $\sigma\sqrt{r_t}\,dW_t$ qui assure $r_t \ge 0$.
- **Heston** : MBG sur le prix + OU/CIR sur la variance (volatilité stochastique).
- **MBG fractionnaire** : remplacer $W_t$ par un brownien fractionnaire $W_t^H$ pour modéliser la *rough volatility* (Gatheral-Jaisson-Rosenbaum 2014).

---

## Suite logique

**Précédent ← [[02_Calcul d'Itô]]** : pour la définition rigoureuse de $\int H_s\,dW_s$ et le théorème d'existence/unicité qui fonde l'écriture de chaque EDS de cette note.

**Suivant → [[04_Dynamique de Langevin]]** : on étudie un cas particulier important d'EDS — celle dont le drift est l'opposé du gradient d'un potentiel. Distribution stationnaire connue, lien direct avec le sampling MCMC et les diffusion models en ML.

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. **[[03_Équations Différentielles Stochastiques]]** — (cette note) modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
7. [[07_Contrôle Stochastique]] — HJB, Merton, optimal execution, lien RL
8. [[08_Volterra Signatures]] — (placeholder) extension signature pour mémoire longue
