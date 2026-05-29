---
title: Séries temporelles
order: 2
---
Une série temporelle est une suite de variables aléatoires $\{X_t\}_{t=1}^T$ indexées par le temps. La différence fondamentale avec le cadre i.i.d. classique : en statistiques classiques, on tire $n$ observations indépendantes dans la **même** distribution $N(\mu, \sigma^2)$. En série temporelle, à chaque instant on tire dans une distribution dont les **paramètres dépendent du passé** — $N(\mu_t, \sigma_t^2)$ où $\mu_t$ et $\sigma_t^2$ évoluent selon $X_{t-1}, X_{t-2}, \ldots$

On note $\mathcal{F}_{t-1}$ l'ensemble de toute l'information disponible jusqu'au temps $t-1$. La moyenne et la variance **conditionnelles** s'écrivent alors :

$$\mu_t = \mathbb{E}[X_t \mid \mathcal{F}_{t-1}], \qquad \sigma_t^2 = \mathrm{Var}[X_t \mid \mathcal{F}_{t-1}]$$

Ces deux quantités évoluent dans le temps au fur et à mesure que l'information s'accumule.

---

## (i) White noise — la surprise pure

Le white noise est le bloc de base de toute série temporelle. C'est un processus $\{\varepsilon_t\}$ vérifiant trois propriétés :

$$\mathbb{E}[\varepsilon_t] = 0, \qquad \mathrm{Var}[\varepsilon_t] = \sigma^2, \qquad \mathrm{Cov}(\varepsilon_t,\, \varepsilon_s) = 0 \text{ pour } t \neq s$$

La troisième propriété est la clé : connaître $\varepsilon_{t-1}$ ne dit strictement rien sur $\varepsilon_t$. C'est pour cela qu'on parle de **surprise** — sachant tout le passé, l'espérance est nulle :

$$\mathbb{E}[\varepsilon_t \mid \mathcal{F}_{t-1}] = 0$$

**Le white noise ne suppose rien sur la distribution.** Les trois propriétés ci-dessus n'impliquent pas que $\varepsilon_t$ est gaussien. Il peut être uniforme, t-Student avec fat tails, binaire — peu importe. En pratique on suppose souvent $\varepsilon_t \sim N(0,\sigma^2)$ parce que ça simplifie l'estimation par MLE, mais c'est une hypothèse supplémentaire.



![Trois white noises valides](images/2-Statistiques/A_Frequentist/serie-temporelle/im1.png)
*Figure 1. Trois white noises valides — même structure (moyenne nulle, variance constante, non corrélés), distributions différentes. La t-Student(3) produit des valeurs extrêmes bien plus fréquentes que la gaussienne.*

---

## (ii) Stationnarité

### Définition

Un processus est **stationnaire au sens large** s'il vérifie :

$$\mathbb{E}[X_t] = \mu \quad \text{(constante dans le temps)}$$

$$\mathrm{Var}[X_t] = \sigma^2 \quad \text{(constante dans le temps)}$$

$$\mathrm{Cov}(X_t,\, X_{t-k}) = K(k) \quad \text{(ne dépend que du lag } k\text{, pas de } t\text{)}$$

L'intuition : un processus stationnaire a une **distribution stable**. Il oscille autour d'une moyenne fixe avec une amplitude stable. La stationnarité est ce qui rend l'estimation possible — si la moyenne dérive dans le temps, on ne peut pas l'estimer à partir d'une seule trajectoire.

### Le random walk — contre-exemple classique

$$X_t = X_{t-1} + \varepsilon_t$$

En déroulant la récurrence : $X_t = X_0 + \varepsilon_1 + \varepsilon_2 + \cdots + \varepsilon_t$, donc :

$$\mathrm{Var}[X_t] = t \cdot \sigma^2$$

La variance **croît linéairement avec le temps** — le processus est non stationnaire. Le random walk dérive et ne revient jamais à son niveau de départ.

En finance, les **prix** $S_t$ suivent un random walk. Les **rendements** $r_t = S_t - S_{t-1}$ sont stationnaires. C'est pour cela qu'on travaille toujours avec les rendements.


![Stationnarité vs random walk](images/2-Statistiques/A_Frequentist/serie-temporelle/im2.png)
*Figure 2. À gauche : AR(1) stationnaire avec $|\phi|<1$ — oscille autour de zéro et y revient. À droite : random walk avec $\phi=1$ — dérive sans limite, variance qui grandit sans cesse.*

---

## (iii) Processus AR(p)

### Définition et intuition

Un processus **autorégressif d'ordre $p$** est défini par :

$$X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t$$

C'est exactement une régression linéaire, sauf que les prédicteurs sont les **valeurs passées de la série elle-même**. Estimer un AR(1), c'est régresser $X_t$ sur $X_{t-1}$ par OLS — la pente estimée est $\hat\phi_1$.

La **condition de stationnarité** pour AR(1) : $|\phi_1| < 1$. Si $\phi_1 = 1$, on retrouve le random walk. Pour AR(p), la condition généralise : toutes les racines du polynôme caractéristique doivent être hors du cercle unité.

### La vraie nature du AR : des chocs qui se propagent

On peut réécrire tout AR(1) stationnaire comme une somme infinie de surprises passées. En substituant récursivement $X_{t-1} = \phi X_{t-2} + \varepsilon_{t-1}$, puis $X_{t-2} = \phi X_{t-3} + \varepsilon_{t-2}$, etc. :

$$\boxed{X_t = \varepsilon_t + \phi\,\varepsilon_{t-1} + \phi^2\,\varepsilon_{t-2} + \phi^3\,\varepsilon_{t-3} + \cdots = \sum_{k=0}^{\infty} \phi^k\,\varepsilon_{t-k}}$$

Chaque variable est composée de **couches de surprises passées**, chacune atténuée d'un facteur $\phi$ supplémentaire. $X_{t-2}$ c'est 100 % son propre choc $\varepsilon_{t-2}$. $X_{t-1}$ c'est son propre choc plus $\phi$ fois le choc de $t-2$. $X_t$ accumule trois couches.

<div style="text-align:center">

![Propagation des chocs AR](images/2-Statistiques/A_Frequentist/serie-temporelle/im3.png)

</div>

*Figure 3. La structure en couches d'un AR(1). $X_{t-2}$ est entièrement composé de son propre choc. $X_{t-1}$ hérite de $\phi\cdot\varepsilon_{t-2}$. $X_t$ cumule trois couches, chacune atténuée d'un facteur $\phi$. La flèche en pointillés montre l'effet indirect de $\varepsilon_{t-2}$ sur $X_t$ via $X_{t-1}$.*

Pourquoi $|\phi| < 1$ est indispensable : si $\phi \geq 1$, les couches ne s'atténuent pas — elles grossissent. Le processus explose. La condition $|\phi| < 1$ garantit que chaque choc finit par mourir et que le processus revient vers sa moyenne.

C'est la **représentation MA($\infty$)** d'un AR(1) : tout processus AR stationnaire peut s'écrire comme une somme infinie de surprises passées. Mais dans l'AR, les poids $\phi^k$ sont **contraints** — ils découlent tous d'un seul paramètre $\phi$. Le MA va lever cette contrainte.

---

## (iv) Processus MA(q)

### Définition et intuition

Un processus **moving average d'ordre $q$** est défini par :

$$X_t = \varepsilon_t + \theta_1\,\varepsilon_{t-1} + \theta_2\,\varepsilon_{t-2} + \cdots + \theta_q\,\varepsilon_{t-q}$$

Pas de valeurs passées de $X$ — uniquement des **surprises passées**, pondérées directement. C'est la version explicite et assumée de ce que l'AR faisait implicitement via une somme infinie.

La différence fondamentale avec l'AR :

- **AR(p)** : mémoire **infinie** des chocs, qui s'atténue géométriquement. Les poids $\phi, \phi^2, \phi^3, \ldots$ sont contraints — ils découlent tous d'un seul paramètre $\phi$.
- **MA(q)** : mémoire **finie et exacte** — exactement $q$ périodes, puis le choc disparaît complètement. Les poids $\theta_1, \ldots, \theta_q$ sont **libres** — estimés indépendamment, sans contrainte entre eux.

![AR vs MA structure des poids](images/2-Statistiques/A_Frequentist/serie-temporelle/im4.png)
*Figure 4. À gauche : les poids d'un AR(1) — décroissance géométrique $\phi^k$, jamais exactement nuls, tous contraints par un seul paramètre $\phi$. À droite : les poids d'un MA(2) — libres aux lags 1 et 2, exactement nuls ensuite. La coupure est nette.*

Un MA(q) peut parfois mieux représenter la réalité qu'un AR à beaucoup de lags. Si le vrai mécanisme est "le choc d'hier compte beaucoup, celui d'avant-hier un peu, et tout ce qui précède n'a aucun effet", le MA le capture directement. L'AR serait obligé d'approcher cette coupure nette avec de nombreux lags.

### Stationnarité du MA

Tout MA(q) est **toujours stationnaire**, quelle que soit la valeur des $\theta_k$. C'est une propriété automatique — contrairement à l'AR, il n'y a pas de condition sur les paramètres à vérifier.

---

## (v) Processus ARMA(p,q)

Un **ARMA(p,q)** combine les deux :

$$X_t = \underbrace{\phi_1 X_{t-1} + \cdots + \phi_p X_{t-p}}_{\text{partie AR}} + \underbrace{\varepsilon_t + \theta_1\,\varepsilon_{t-1} + \cdots + \theta_q\,\varepsilon_{t-q}}_{\text{partie MA}}$$

La partie AR capte la dynamique de persistance — la série qui revient à sa moyenne via ses propres valeurs passées. La partie MA capte les effets de court terme — des influences qui s'éteignent exactement après $q$ périodes.

En pratique, l'ARMA est plus **parcimonieux** que l'AR ou le MA seul. Souvent un ARMA(1,1) suffit là où un AR(5) ou MA(5) serait nécessaire. Pour choisir $p$ et $q$ on utilise les critères AIC ou BIC qui pénalisent la complexité :

$$\mathrm{AIC} = -2\ln(\hat{L}) + 2(p+q), \qquad \mathrm{BIC} = -2\ln(\hat{L}) + \ln(T)(p+q)$$

On estime plusieurs ARMA(p,q) pour différentes valeurs de $p$ et $q$ et on retient celui qui minimise le critère.

---

## (vi) ACF et PACF — identifier le bon modèle

### Pourquoi ces outils

Avant de modéliser, on veut savoir : **le passé prédit-il le futur, et jusqu'à combien de lags ?** L'ACF et la PACF répondent à cette question. On les trace systématiquement sur les données brutes pour choisir $p$ et $q$.

### L'ACF — autocorrélation totale

L'ACF au lag $k$ mesure la corrélation entre $X_t$ et $X_{t-k}$, sans distinguer effets directs et indirects :

$$r_k = \frac{\mathrm{Cov}(X_t,\, X_{t-k})}{\mathrm{Var}(X_t)} = \frac{\displaystyle\sum_{t=1}^{T-k}(x_t - \bar{x})(x_{t+k} - \bar{x})}{\displaystyle\sum_{t=1}^{T}(x_t - \bar{x})^2}$$

**Exemple.** Série $[2, 4, 6, 8, 10]$, $\bar{x} = 6$, $\sum(x_t - \bar{x})^2 = 40$.

| Lag $k$ | Paires | $r_k$ |
|---------|--------|-------|
| 1 | $(2,4),(4,6),(6,8),(8,10)$ | $16/40 = 0.4$ |
| 2 | $(2,6),(4,8),(6,10)$ | $-4/40 = -0.1$ |
| 3 | $(2,8),(4,10)$ | $-16/40 = -0.4$ |

### Le problème : effets directs vs indirects

L'ACF au lag 2 capte à la fois l'effet direct de $X_{t-2}$ sur $X_t$ **et** l'effet indirect qui transite par $X_{t-1}$. Si la seule vraie dépendance est $X_{t-1} \to X_t$, l'ACF au lag 2 sera quand même non nulle — à cause de la chaîne $X_{t-2} \to X_{t-1} \to X_t$.

### La PACF — corrélation directe nette

La PACF corrige ce problème. Elle mesure la corrélation entre $X_t$ et $X_{t-k}$ **après avoir retiré l'influence de tous les lags intermédiaires** $X_{t-1}, \ldots, X_{t-k+1}$.

La mécanique pour le lag $k=2$ :

1. Régresser $X_t$ sur $X_{t-1}$ → résidu $e_t^{(1)}$ : ce qui reste de $X_t$ une fois l'effet de $X_{t-1}$ retiré.
2. Régresser $X_{t-2}$ sur $X_{t-1}$ → résidu $e_t^{(2)}$ : ce qui reste de $X_{t-2}$ indépendamment de $X_{t-1}$.
3. PACF(2) = corrélation entre $e_t^{(1)}$ et $e_t^{(2)}$.

Sur l'exemple $[2,4,6,8,10]$ : la série est parfaitement linéaire, $X_{t-1}$ explique $X_t$ à 100 %, les résidus des deux régressions sont nuls, donc PACF(2) = 0. C'est un AR(1) pur.

### La règle de lecture

![ACF et PACF pour AR(1) et AR(2)](images/2-Statistiques/A_Frequentist/serie-temporelle/im5.png)
*Figure 5. Pour un AR(1) et un AR(2) : l'ACF décroît lentement dans les deux cas — elle ne permet pas de distinguer. La PACF coupe nettement après le lag $p$ — c'est elle qui donne directement l'ordre du modèle.*

La règle pratique : **regarder où la PACF rentre dans les bandes de confiance** (tracées à $\pm 2/\sqrt{T}$). Le lag de coupure donne $p$.

Pour les MA, les rôles s'inversent : c'est l'ACF qui coupe net après le lag $q$, et la PACF qui décroît lentement. Pour un ARMA, les deux décroissent — on utilise alors AIC/BIC.



![Tableau récapitulatif ACF PACF](images/serie-temporelle/im6.png)



*Figure 6. Règle de lecture selon le modèle. AR : ACF décroît, PACF coupe après $p$. MA : ACF coupe après $q$, PACF décroît. ARMA : les deux décroissent — utiliser AIC/BIC.*

Le tableau résume tout :

| Modèle | ACF | PACF | Comment choisir |
|--------|-----|------|-----------------|
| AR(p) | Décroît lentement | Coupe après lag $p$ | $p$ = lag de coupure PACF |
| MA(q) | Coupe après lag $q$ | Décroît lentement | $q$ = lag de coupure ACF |
| ARMA(p,q) | Décroît lentement | Décroît lentement | Minimiser AIC/BIC |

---

> **Lien avec ARCH/GARCH.** Dans un AR, MA ou ARMA, le terme $\varepsilon_t$ est supposé être un white noise à variance **constante** $\sigma^2$. En finance, cette hypothèse est systématiquement violée : les résidus ont une variance qui change dans le temps — grande pendant les crises, petite pendant les périodes calmes. C'est le *volatility clustering*. La solution naturelle est de modéliser $\sigma_t^2$ explicitement en fonction des résidus carrés passés — c'est l'objet des modèles ARCH et GARCH.
