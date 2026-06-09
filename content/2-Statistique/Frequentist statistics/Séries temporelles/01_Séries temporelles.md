---
title: Séries temporelles
order: 2
---

# Séries temporelles

> Cette note construit les séries temporelles depuis l'intuition : en quoi elles diffèrent du cadre i.i.d. classique, puis les briques de base (white noise, stationnarité) et les modèles AR / MA / ARMA, avec l'ACF et la PACF pour les identifier. Le fil conducteur : tout processus se lit comme une accumulation de **surprises passées**.

## Statistique classique vs série temporelle

En statistique classique, on a **une seule distribution, fixe**. On tire $n$ observations indépendantes dedans — par exemple les tailles de $n$ adultes pris au hasard : chaque tirage vient de la **même** cloche $N(\mu,\sigma^2)$, et l'ordre des tirages n'a aucune importance. Connaître les 49 premières tailles ne dit rien sur la 50ᵉ.

Une série temporelle casse les deux propriétés d'un coup :
* On perd le $i$ (independent) -> les observations sont **dépendantes** on dit qu'il y'a **autocorrélation** ou une dépendance sérielle (serial dependance).
* On garde le "identically distributed" ... à condition d'être **stationnaire**.
Ainsi la distribution **change à chaque instant**, et son centre **dépend de ce qui s'est déjà produit**. Formellement, c'est une suite de variables aléatoires $\{X_t\}_{t=1}^T$ indexées par le temps.

> 💡 **L'image à garder en tête.** À chaque instant $t$, imagine une cloche gaussienne $N(\mu_t,\sigma_t^2)$, et tu tires **un seul** point dedans : c'est $X_t$. Puis, pour $t+1$, la cloche se **déplace** — son centre $\mu_t$ et sa largeur $\sigma_t^2$ peuvent changer, et ils changent **en fonction du passé** $X_{t-1}, X_{t-2}, \ldots$ En i.i.d. la cloche est clouée au mur, identique à chaque tirage ; en série temporelle elle se balade le long de la trajectoire. Ton intuition de départ était exactement ça.

![i.i.d. vs série temporelle](images/2-Statistiques/A_Frequentist/serie-temporelle/fig_iid_vs_ts.png)
*Figure 0. À gauche, le cadre i.i.d. : la même cloche $N(0,1)$ est redessinée à chaque instant — les points sont tirés indépendamment, sans mémoire, et oscillent autour de $\mu=0$ sans aucune structure. À droite, une série temporelle AR(1) : à chaque $t$, la cloche se re-centre sur $\mu_t = \phi\,X_{t-1}$ (cercle vide, relié au point précédent par la flèche grise), puis on tire $X_t$ dedans (point plein noir). La cloche suit la trajectoire — c'est toute la différence.*

> [!example] L'exemple qui sépare les deux
> **Lancer de dé** *(i.i.d.)* : à chaque lancer, même loi uniforme sur $\{1,\ldots,6\}$, indépendante du passé. Avoir fait un 6 ne change rien à la loi du lancer suivant — la cloche ne bouge pas.
>
> **Température quotidienne** *(série temporelle)* : la loi de la température de demain est **centrée près de celle d'aujourd'hui**. S'il fait 30° aujourd'hui, demain c'est probablement $\sim$ 30°, pas 5°. Le centre de la cloche de demain, $\mu_t$, est en gros la valeur d'aujourd'hui $X_{t-1}$.

C'est précisément cette dépendance au passé qu'on va apprendre à modéliser.

## Moyenne et variance conditionnelles

Pour parler proprement de « la cloche au temps $t$ sachant le passé », il faut d'abord nommer « le passé ».

> 💡 **La tribu $\mathcal{F}_{t-1}$, sans la théorie de la mesure.** $\mathcal{F}_{t-1}$, c'est simplement **toute l'information disponible jusqu'à l'instant $t-1$** : la trajectoire observée $X_1, X_2, \ldots, X_{t-1}$ et tout ce qu'on peut en déduire. Vois-la comme le **dossier** qu'on a en main juste avant de tirer $X_t$. (Le nom mathématique « tribu » / σ-algèbre encode formellement *quelles questions je peux trancher avec l'information dont je dispose* — mais l'intuition « tout ce que j'ai observé jusqu'ici » suffit largement pour toute la note.)

La cloche au temps $t$ — son centre et sa largeur, **sachant le passé** — ce sont exactement la moyenne et la variance **conditionnelles** :

> [!warning] Moments conditionnels
> $$\mu_t = \mathbb{E}[X_t \mid \mathcal{F}_{t-1}], \qquad \sigma_t^2 = \mathrm{Var}[X_t \mid \mathcal{F}_{t-1}]$$
>
> Ce sont la **position** et la **largeur** de la cloche d'où on tire $X_t$, une fois connu tout le passé. Le symbole $\mid \mathcal{F}_{t-1}$ se lit « sachant tout ce qu'on a observé jusqu'à $t-1$ ».

En i.i.d., $\mu_t=\mu$ et $\sigma_t^2=\sigma^2$ pour tout $t$ : la cloche ne bouge jamais. Toute la richesse des séries temporelles vient de la **manière** dont $\mu_t$ et $\sigma_t^2$ dépendent du passé — c'est ce que les modèles qui suivent vont spécifier.

---

## (i) White noise — la surprise pure

Le white noise est le bloc de base de toute série temporelle. C'est un processus $\{\varepsilon_t\}$ vérifiant trois propriétés :

$$\mathbb{E}[\varepsilon_t] = 0, \qquad \mathrm{Var}[\varepsilon_t] = \sigma^2, \qquad \mathrm{Cov}(\varepsilon_t,\, \varepsilon_s) = 0 \text{ pour } t \neq s$$

La troisième propriété est la clé : il n'y a **aucune corrélation linéaire** entre deux instants distincts. C'est la définition du white noise dit *faible* — la seule dont on a besoin pour les modèles AR/MA/ARMA.

**Attention à un piège classique.** Non-corrélation ne veut pas dire « $\varepsilon_{t-1}$ ne dit rien sur $\varepsilon_t$ ». La corrélation ne capte que le lien *linéaire*. Un processus peut être parfaitement non corrélé tout en ayant un passé qui renseigne sur son présent — typiquement sur sa **variance**. C'est exactement le cas des rendements financiers (modèles ARCH/GARCH, cf. fin de note) : non corrélés, mais un $\varepsilon_{t-1}^2$ grand annonce un $\varepsilon_t^2$ grand.

Il faut donc distinguer trois niveaux, du plus faible au plus fort :

$$\underbrace{\mathrm{Cov}(\varepsilon_t,\varepsilon_s)=0}_{\text{white noise faible}} \;\subsetneq\; \underbrace{\mathbb{E}[\varepsilon_t \mid \mathcal{F}_{t-1}] = 0}_{\text{martingale difference}} \;\subsetneq\; \underbrace{\varepsilon_t \perp\!\!\!\perp \mathcal{F}_{t-1}}_{\text{indépendance (white noise fort)}}$$

La propriété du milieu — espérance conditionnelle nulle sachant tout le passé — est ce qui justifie vraiment le mot **surprise** : en moyenne, le passé ne permet de prédire ni le signe ni le niveau de $\varepsilon_t$. Mais ce n'est **pas** une conséquence des trois propriétés de départ, c'est une hypothèse strictement plus forte. L'indépendance est encore au-dessus : là, le passé ne dit *rien du tout*, pas même sur la variance.

**Le white noise ne suppose rien sur la distribution.** Aucun de ces niveaux n'impose la forme de la loi de $\varepsilon_t$ : gaussienne, uniforme, t-Student à fat tails, binaire — peu importe. En pratique on suppose souvent $\varepsilon_t \sim N(0,\sigma^2)$ i.i.d. parce que ça donne l'estimation par MLE, mais c'est une hypothèse supplémentaire, pas une exigence.

![[fig_white_noise.png]]
*Figure 1. Trois white noises faibles valides, tous à variance unité — même structure (moyenne nulle, variance constante, non corrélés), distributions différentes. Les cercles noirs marquent les dépassements de $\pm 2\sigma$. L'uniforme, à support borné $[-\sqrt{3},\sqrt{3}]$, ne peut physiquement jamais sortir de la bande (0 dépassement). La gaussienne en sort rarement ($\approx 5\,\%$). La t-Student(3) en sort bien plus souvent et produit des pics extrêmes : ce sont les fat tails.*

---

## (ii) Stationnarité

### Deux niveaux de stationnarité

L'idée commune : la **loi du processus ne change pas dans le temps**. C'est ce qui récupère le « identiquement distribué » qu'on perd en quittant l'i.i.d. (cf. intro). Il en existe deux versions.

> [!warning] Stationnarité stricte vs faible
> **Stricte** : la loi jointe de $(X_t, X_{t+1}, \ldots, X_{t+k})$ est invariante par translation dans le temps — décaler la fenêtre ne change rien. Très fort, rarement vérifiable en pratique.
>
> **Faible (au sens large, second ordre)** : on n'exige que la stabilité des deux premiers moments :
> $$\mathbb{E}[X_t] = \mu, \qquad \mathrm{Var}[X_t] = \sigma^2, \qquad \mathrm{Cov}(X_t, X_{t-k}) = \gamma(k)$$
> Moyenne et variance constantes, autocovariance qui ne dépend que du **lag** $k$, pas de $t$.

En pratique, « stationnaire » sans précision = **faible**. Les deux notions ne coïncident pas en général, mais deux ponts utiles : stricte + variance finie $\Rightarrow$ faible ; et pour un processus **gaussien**, faible $\Leftrightarrow$ stricte (une loi jointe gaussienne est entièrement déterminée par sa moyenne et sa covariance).

> 💡 **Le lien avec l'i.i.d.** Une série stationnaire, c'est « identiquement distribuée mais pas indépendante » — l'i.i.d. amputé de son premier *i*. La stationnarité te rend le *i.d.* : même loi marginale à chaque $t$, donc un seul $\mu$ et un seul $\sigma$ à estimer. Reste à dompter la dépendance : on demande qu'elle **s'estompe avec le lag** ($\gamma(k)\to 0$ quand $k\to\infty$) — c'est l'**ergodicité**, qui garantit que la moyenne le long d'**une** trajectoire converge vers $\mu$, comme si on avait plein d'échantillons indépendants. Sans ça, observer une seule trajectoire ne servirait à rien.

### Pourquoi c'est indispensable

On ne dispose que d'**une** trajectoire. Si la loi dérive dans le temps, chaque instant a sa propre distribution et on n'a qu'un seul point par distribution — impossible d'estimer quoi que ce soit. La stationnarité est l'hypothèse qui dit « tous ces points viennent de la même loi, je peux les agréger ».

### Le random walk — le contre-exemple à connaître

$$X_t = X_{t-1} + \varepsilon_t \quad\Longleftrightarrow\quad X_t = X_0 + \sum_{s=1}^{t}\varepsilon_s$$

d'où :

$$\mathrm{Var}[X_t] = t\,\sigma^2$$

La variance **croît linéairement** — non stationnaire. La raison profonde : c'est un AR(1) avec $\phi = 1$, donc chaque choc garde un poids de $1$ **pour toujours** (cf. la représentation MA($\infty$), section iii). Les chocs ne meurent jamais, ils s'empilent. À l'opposé, un AR(1) avec $|\phi|<1$ atténue chaque choc géométriquement → la variance se stabilise et le processus **revient à sa moyenne** (mean reversion).

![[fig_stationarity.png]]
*Figure 2. À gauche, 14 trajectoires d'un AR(1) stationnaire ($\phi=0.7$) : elles restent confinées dans une bande $\pm 2\sigma_X$ **constante** et reviennent sans cesse vers zéro. À droite, 14 random walks ($\phi=1$) : ils divergent, contenus seulement par un cône $\pm 2\sqrt{t}\,\sigma$ qui s'élargit — variance qui croît, aucun retour à la moyenne. Même échelle verticale des deux côtés : l'AR(1) paraît tassé précisément parce que sa dispersion est bornée.*

> [!warning] Racine unité
> Le cas $\phi=1$ s'appelle une **racine unité**. C'est la frontière exacte entre stationnaire ($|\phi|<1$, retour à la moyenne) et explosif ($|\phi|>1$). Un processus à racine unité a une mémoire infinie des chocs et une variance qui diverge.

### Rendre stationnaire : la différenciation

Quand une série a une racine unité, on la **différencie** : on passe de $X_t$ à $\Delta X_t = X_t - X_{t-1}$. Différencier un random walk donne $\Delta X_t = \varepsilon_t$ — du white noise, stationnaire. Une série qu'il faut différencier **une** fois pour la rendre stationnaire est dite **intégrée d'ordre 1**, notée $I(1)$. C'est le « **I** » de AR**I**MA : un ARMA appliqué à la série différenciée.

En finance, c'est exactement ce qu'on fait : les **(log-)prix** $S_t$ se comportent comme un random walk ($I(1)$, non stationnaire), les **(log-)rendements** $r_t = \log S_t - \log S_{t-1}$ sont stationnaires. D'où la règle universelle : on modélise toujours les rendements, jamais les prix bruts.

### Comment savoir si c'est stationnaire ?

À l'œil : on regarde si la série dérive (tendance, variance qui enfle). Formellement, deux tests aux hypothèses **inversées** :

- **ADF** (Augmented Dickey-Fuller) : $H_0$ = « il y a une racine unité » (non stationnaire). Rejeter $H_0$ → on conclut à la stationnarité.
- **KPSS** : $H_0$ = « stationnaire ».

Les utiliser ensemble lève l'ambiguïté : si ADF rejette sa $H_0$ **et** KPSS ne rejette pas la sienne, le verdict de stationnarité est solide.










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
