---
title: facteurs barra
order: 1
---

# 1. Cadre général

## 1.1 Cross-section vs série temporelle

> [!note] À retenir
> - **Série temporelle (FF3, CAPM)** : on fixe un actif (ou un portefeuille) $i$, et on régresse $r_{i,t}$ sur les facteurs $f_t$ pour $t = 1, \dots, T$. La matrice $X$ des expositions (ici les $f_t$) est **exogène** et commune à tous les actifs : on apprend les **bêtas** $\beta_i$ propres à chaque actif. Vision **longitudinale**, dynamique des portefeuilles.
> - **Cross-section / coupe transversale (Barra)** : on fixe une date $t$, et on régresse $r_{n,t}$ sur les expositions $X_{n,k,t}$ pour $n = 1, \dots, N$ stocks. À chaque date on **ré-estime** le vecteur de factor returns $f_t$. La matrice $X$ est **endogène** (construite à partir des caractéristiques de chaque stock : secteur, pays, book-to-price, etc.). Vision **transversale**, dynamique du risque.

Intuition : en série temporelle, on demande "comment cet actif se comporte au fil du temps face aux facteurs". En cross-section, on demande "à cette date, quel est le rendement associé au fait d'être value, US, tech, etc. ?". Barra fait *une régression par date* sur tout l'univers.

## 1.2 Le modèle de base

$$
r_n(t) = \sum_k X_{n,k}(t)\, f_k(t) + u_n(t)
$$

- $X_{n,k}(t)$ : exposition de l'actif $n$ au facteur $k$, **connue en $t$** (factor loading).
- $r_n(t)$ : excess return de $t$ à $t+1$.
- $f_k(t)$ : factor return du facteur $k$, **inconnu**, à estimer.
- $u_n(t)$ : specific / idiosyncratic return.

Conventions Barra :
- Returns *forward* : $R_t^+ = (S_{t+1}-S_t)/S_t$.
- Risk-free daily : $r_f^{daily} = (1 + r_f^{yearly}/100)^{1/252} - 1$, converti en USD via le FX spot $F_x^{EUR/USD}$.

## 1.3 Univers et modèles régionaux

MSCI Barra ne publie **pas un modèle unique mondial** mais une **suite de modèles régionaux et globaux**, chacun calibré sur son propre univers avec ses propres facteurs et expositions. Pour un stock donné, plusieurs modèles peuvent s'appliquer (un stock français est dans Europe **et** dans Global), mais ils ne donnent **pas les mêmes loadings**.

### Table des principaux modèles

| Code | Région / Univers | Spécificité |
|---|---|---|
| `bausfastd` | USA | ≈11 000 stocks américains, facteurs locaux US |
| `baeutrd` | Europe | facteurs pays (FR, DE, IT…), secteurs européens, devise EUR |
| `bacne5s` | Chine *national* | A-shares mainland (Shanghai/Shenzhen) |
| `bacxe1s` | Chine *extension* | H-shares (Hong Kong), ADR chinois |
| `baine2l` | Inde | univers domestique indien, INR |
| `bajep4d` | Japon | facteurs locaux JP, secteurs japonais |
| `baase2s` | Asie ex-Japon | reste de l'Asie |
| `bagemtrd` | **Global** (GEM) | filet de sécurité monde entier, facteurs globaux |
| `other` | Fallback | stocks frontier, listings récents, instruments exotiques |

> [!note] Convention de nommage
> Le préfixe `ba` = "Barra". Ensuite : `us`, `eu`, `cn`, `in`, `jp`, `as`, `gem` indique la région. Les suffixes (`fastd`, `trd`, `e5s`…) encodent la version du modèle et la fréquence (daily/monthly).

### Le mécanisme **waterfall**

Pour un portefeuille mondial, on attribue à chaque stock le **modèle le plus spécifique** disponible — un modèle régional étroit capte mieux les facteurs locaux qu'un modèle mondial qui les dilue. La méthode **waterfall** essaie les modèles dans un ordre de spécificité décroissante :

$$
\text{USA} \to \text{Europe} \to \text{Chine N} \to \text{Chine X} \to \text{Inde} \to \text{Japon} \to \text{Asie} \to \text{Global} \to \text{Other}
$$

Chaque stock est associé au **premier modèle qui le contient**. Exemples :
- Apple (US) → capturé par `bausfastd`, pas besoin d'aller plus loin.
- LVMH (FR) → pas dans US, capturé par `baeutrd`.
- Tencent (HK) → pas dans US ni Europe, capturé par `bacxe1s` (Chine extension).
- Une junior mining canadienne → dans aucun modèle régional, attrapée par `bagemtrd`.
- Un IPO frontier exotique → fallback `other`.

> [!note]- Conséquence : matrice $X$ assemblée par bloc
> En pratique la matrice $X$ d'un portefeuille global **n'est pas issue d'un seul modèle** — c'est un assemblage par bloc :
> - les stocks USA prennent leurs expositions depuis `bausfastd`,
> - les stocks Europe depuis `baeutrd`,
> - les stocks chinois depuis `bacne5s` ou `bacxe1s` selon le listing,
> - etc.
>
> Chaque sous-univers vient avec **son propre ensemble de facteurs**, qui ne sont pas nécessairement comparables d'une région à l'autre. Le facteur "Tech US" de `bausfastd` n'est pas le même objet que le facteur "Tech Asie" de `baase2s`. Pour des analyses inter-régions, il faut souvent retomber sur `bagemtrd` (qui mixe les régions dans des facteurs globaux).

### Mimicking portfolio par modèle

Pour chaque modèle, MSCI fournit la matrice $H$ (mimicking portfolio) calibrée sur l'univers correspondant, et le factor return s'obtient simplement par :
$$
\hat{f}(t) = H\, r(t)
$$
C'est-à-dire que **chaque factor return est le rendement d'un portefeuille de l'univers**, dont les poids $H_{k,n}$ sont fixés par le modèle.

> [!note]- Pourquoi c'est puissant
> Une fois $H$ connue, on n'a plus besoin de refaire la régression : il suffit d'observer les rendements $r(t)$ de l'univers à la date $t$, et de calculer $\hat{f}(t) = Hr(t)$ pour avoir tous les factor returns du jour.

### Application : cross-impact

Le waterfall est utilisé de façon similaire dans le calcul du cross-impact d'un basket multi-régions — voir [[(iii) Cross-impact|note Cross-impact]] pour les détails de l'algorithme et de la non-comparabilité des facteurs inter-régions.

## 1.4 Comparaison de portefeuilles dans un univers fixé

Cas typique : on veut comparer deux portefeuilles à expositions et rendements connus, qui sont des **sous-ensembles** de l'univers Barra.

- Univers fixé : $N = 5000$ stocks, donc $r \in \mathbb{R}^{5000}$.
- Portefeuille 1 : 100 stocks. Portefeuille 2 : 25 stocks.
- **Astuce** : on garde le vecteur $r$ de taille 5000, et pour les stocks **non détenus** dans le portefeuille on met simplement $r_n = 0$ (ou plutôt on met le **poids** $h_n = 0$).

Pourquoi : la matrice $H$ et les expositions $X$ sont définies sur l'univers complet. Mettre à zéro les poids hors portefeuille permet d'utiliser le même cadre matriciel pour tous les portefeuilles, et donc de comparer leurs expositions / contributions au risque sur la même base.

# 2. Stage 1 — Choix des facteurs

Les facteurs Barra se rangent en **5 familles** :

| Famille | Type | Exemples |
|---|---|---|
| Risk indices (style) | continu | value, momentum, size, volatility, quality, growth, yield, liquidity, leverage |
| Industries | discret (dummy) | technology, financials, energy… |
| Country | discret (dummy) | USA, France, Japan… |
| Currencies | discret | USD, EUR, JPY… |
| Market ("world") | constante | intercept global (GEMTRD) |

## 2.1 Common factors et descriptors

Les **common factors** (= ce que partagent les actifs) se subdivisent :
- **Discrets** : industries, countries, currencies → exposition $\in \{0, 1\}$.
- **Continus** : style factors → exposition $\in \mathbb{R}$, standardisée (moyenne 0, écart-type 1 sur l'univers).

Chaque style factor est lui-même construit comme une **moyenne pondérée de descriptors**. Exemple pour le facteur **Value** : on agrège book-to-price, earnings yield, dividend yield… → on obtient un score unique $X_{n, \text{Value}}$ par stock.

> [!note]- Pourquoi standardiser les descriptors
> Sans standardisation, les unités sont incomparables (un book-to-price ratio est entre 0 et 5, un earnings yield est en %…). On centre-réduit chaque descriptor sur l'univers à la date $t$, puis on combine. Ça permet aussi de donner une interprétation claire au factor return : $f_{\text{Value}}(t)$ est le rendement associé à *un écart-type d'exposition value* au-dessus de la moyenne.

## 2.2 Lien avec les papiers historiques

- **CAPM (Sharpe 1963)** : 1 facteur → market.
- **Fama-French 3 (1993)** : market + size (SMB) + value (HML).
- **Carhart 4 (1997)** : + momentum.
- **Fama-French 5 (2015)** : + profitability + investment.
- **Barra** : ≈ 10 style factors + industries + countries → granularité bien plus fine, mais l'objectif diffère (gestion de risque vs asset pricing).

## 2.3 Méta-facteurs (modèle global `bagemtrd`)

Spécifique au modèle **global** : MSCI regroupe les ~16 descriptors style en **8 méta-facteurs** lisibles. Ce n'est **pas** une réduction de dimension imposée par la stabilité numérique (le global a plein de stocks, pas de problème d'estimation) — c'est une **réorganisation pour la lisibilité**. Sur un dashboard de risk avec ~16 descriptors style + ~40 industries + ~50 countries, c'est illisible. En 8 méta-facteurs styles, ça devient compréhensible.

### Les 8 méta-facteurs et leurs descriptors

| Méta-facteur | Descriptors |
|---|---|
| **Value** | Book-to-Price, Earnings Yield, LT Reversal |
| **Size** | Mid Cap, Size |
| **Momentum** | Momentum |
| **Quality** | Leverage, Investment Quality, Earnings Variability, Earnings Quality, Profitability |
| **Yield** | Dividend Yield |
| **Volatility** | Beta, Residual Volatility |
| **Growth** | Growth |
| **Liquidity** | Liquidity |

### Formulation mathématique

Soit $D \in \mathbb{R}^{N \times p}$ la matrice des **descriptors** standardisés ($p \approx 16$), et $W \in \mathbb{R}^{p \times K_{\text{meta}}}$ la matrice de poids (sparse) fournie par MSCI, avec $K_{\text{meta}} = 8$. Les expositions aux méta-facteurs sont :
$
X_{\text{meta}} = D \cdot W \quad \in \mathbb{R}^{N \times 8}
$

$W$ est **sparse par construction** : seuls les descriptors d'une même famille contribuent à leur méta-facteur. Par exemple la colonne *Quality* de $W$ a 5 entrées non nulles (Leverage, Investment Quality, Earnings Variability, Earnings Quality, Profitability) et zéro partout ailleurs.

> [!note]- Exemple : structure de $W$ pour Quality
> Si on répertorie les 16 descriptors dans l'ordre (Book-to-Price, Earnings Yield, LT Reversal, Mid Cap, Size, Momentum, Leverage, Investment Quality, Earnings Variability, Earnings Quality, Profitability, Dividend Yield, Beta, Residual Volatility, Growth, Liquidity), la colonne *Quality* de $W$ ressemble à :
> $W_{:, \text{Quality}} = (0, 0, 0, 0, 0, 0, w_7, w_8, w_9, w_{10}, w_{11}, 0, 0, 0, 0, 0)^\top$
> avec $w_7, \dots, w_{11}$ les poids MSCI (typiquement positifs et somme à peu près 1 pour normaliser l'échelle de la combinaison). MSCI calibre ces poids empiriquement pour maximiser la stabilité et l'interprétabilité du méta-facteur.

### Conséquence pour le modèle

Le modèle `bagemtrd` régresse en pratique sur les **méta-facteurs** (8 styles) plutôt que sur les descriptors bruts (16) :
$
r = \underbrace{X_{\text{meta}}}_{N \times 8} f_{\text{meta}} + X_{\text{indus}} f_{\text{indus}} + X_{\text{country}} f_{\text{country}} + u
$

Et l'attribution P&L (cf. §6.4) se fait directement au niveau méta : *"+6k$ vient de Quality, +3k$ de Momentum, −2k$ de Volatility"* — lisible pour un PM, là où *"+1.2k$ Earnings Variability, +2.1k$ Earnings Quality, +0.8k$ Investment Quality…"* serait illisible.

> [!note] Drill-down quand nécessaire
> Si un PM veut comprendre **pourquoi** Quality contribue +6k$, il peut zoomer sur les descriptors sous-jacents. Le méta-facteur aggège mais ne perd pas l'info — elle reste accessible à la demande.

# 3. Stage 2 — Estimation des factor returns

**Principe général.** À chaque date $t$, on a un univers de $N$ stocks et $K$ facteurs. On observe $r \in \mathbb{R}^N$ (rendements ex-post) et $X \in \mathbb{R}^{N \times K}$ (expositions, connues *ex ante*). On veut estimer $f \in \mathbb{R}^K$ (factor returns) par régression cross-section :
$$
r = X f + u, \quad u \sim \mathcal{N}(0, \Sigma_u)
$$
Deux complications par rapport à une OLS standard :
1. **Hétéroscédasticité** : les petits stocks sont plus volatils que les gros → $\Sigma_u$ n'est pas $\sigma^2 I$.
2. **Colinéarité des dummies** : market + industries + countries sont colinéaires (somme des dummies = vecteur de 1 = market) → il faut imposer des contraintes.

On traite (1) par WLS, (2) par une matrice de restriction $R$.

## 3.1 Modèle Market + Industry

### 3.1.1 Hypothèse sur la variance idiosyncratique

Barra suppose que la variance des résidus décroît avec la taille du stock :
$$
\operatorname{Var}(u_i) \propto \frac{1}{\sqrt{\text{Market Cap}_i}}
$$

On définit donc la matrice diagonale de poids WLS :
$$
V = \operatorname{diag}(w_1, \dots, w_N), \quad w_i = \frac{\sqrt{\text{Market Cap}_i}}{\sum_j \sqrt{\text{Market Cap}_j}}
$$

> [!note]- Pourquoi $\sqrt{\text{Cap}}$ et pas $\text{Cap}$ ?
> Pondérer par $\text{Cap}$ donnerait trop de poids aux méga-caps et réduirait quasiment la régression à quelques stocks. Pondérer par $\sqrt{\text{Cap}}$ est un compromis empirique entre equal-weight (qui sur-pondère le bruit des micro-caps) et cap-weight (qui ignore les small/mid). Justification théorique : si la variance idiosyncratique est inversement proportionnelle à $\sqrt{\text{Cap}}$, alors les poids WLS optimaux sont précisément $\sqrt{\text{Cap}}$ (la WLS optimale pondère par l'inverse de la variance).

### 3.1.2 Problème de colinéarité et matrice de restriction

Problème : on a un intercept (market) et des dummies industrie. La somme des dummies industrie vaut le vecteur $\mathbf{1}$, qui est aussi la colonne du market → $X$ n'est pas de rang plein.

Solution : imposer la contrainte de **somme nulle pondérée** sur les industries :
$$
w_1 f_{i_1} + w_2 f_{i_2} + \cdots + w_p f_{i_p} = 0
$$
Intuition : le market capture la moyenne pondérée, donc les industries ne capturent que les **écarts** par rapport à cette moyenne.

On encode ça via une matrice $R$ qui paramétrise le sous-espace des $f$ admissibles : $f = R g$ avec $g$ libre. On résout en $g$ puis on remonte à $f$.

### 3.1.3 Résolution WLS

Le modèle contraint s'écrit $r = X R g + u$. On veut transformer en homoscédastique pour utiliser de l'OLS standard.

> [!note]- Dérivation complète (transformation $V^{1/2}$ et résolution)
> En multipliant par $V^{1/2}$ à gauche :
> $V^{1/2} r = V^{1/2} X R g + V^{1/2} u$
> En posant $\tilde{r} = V^{1/2} r$, $Y = V^{1/2} X R$, $\tilde{u} = V^{1/2} u$ :
> $\tilde{r} = Y g + \tilde{u}, \quad \tilde{u} \sim \mathcal{N}(0, I)$
> Le système est maintenant homoscédastique et $Y$ est de rang plein → OLS classique :
> $\hat{g} = (Y' Y)^{-1} Y' \tilde{r} = (R' X' V X R)^{-1} R' X' V r$
> En remontant via $f = R g$ :
> $\boxed{\hat{f} = R (R' X' V X R)^{-1} R' X' V \, r}$

On définit alors la **matrice des pure factor portfolios** :
$$
\Omega = R (R' X' V X R)^{-1} R' X' V \quad \in \mathbb{R}^{K \times N}
$$
de sorte que $\hat{f} = \Omega r$. C'est exactement la matrice $H$ du §1.3.

> [!note] Interprétation clé
> Chaque ligne $\Omega_{k, \cdot}$ est un **vecteur de poids de portefeuille** sur les $N$ stocks. Le rendement de ce portefeuille est *par construction* le factor return $f_k$. Les poids ne somment pas à 1 (positions long/short).

## 3.2 + Style factors

Les style factors sont des expositions **continues** → pas de problème de colinéarité avec le market → pas besoin de contrainte sur eux.

Le modèle devient :
$$
r = X_s f_s + X_{mi} R g + u
$$
où $X_s$ ($N \times p_s$) sont les expositions style sans contrainte, et $X_{mi} R g$ est le bloc market+industry du §3.1.

### Construction explicite

On empile en une seule matrice $\tilde{X} = [X_s \mid X_{mi} R]$ et on résout pour le vecteur $\tilde{f} = [f_s ; g]$ par WLS comme avant. Une autre façon est de définir une matrice $R$ étendue qui inclut l'identité sur le bloc style.

> [!note]- Exemple : 2 styles, 3 industries, contrainte $f_{i_3} = -f_{i_1} - f_{i_2}$
> Vecteur de facteurs initial : $f = (f_1, f_2, m, f_{i_1}, f_{i_2}, f_{i_3})$ avec $m$ le market.
> Vecteur libre : $g = (f_1, f_2, m, f_{i_1}, f_{i_2})$.
> $R = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & -1 & -1 \end{bmatrix}$
> Les deux premières lignes sont l'identité sur les styles, la dernière ligne encode $f_{i_3} = -f_{i_1} - f_{i_2}$.

### Interprétation des $f_k$ comme moyennes pondérées

Une régression linéaire est, *au fond*, une **moyenne pondérée des $r_n$**. Comme $\hat{f} = \Omega r$, chaque factor return est :
$$
\hat{f}_k = \sum_{n=1}^{N} \Omega_{k,n}\, r_n
$$

- **Pour un facteur industrie** $f_{i_k}$ : $\Omega_{k,n}$ va concentrer ses poids sur les stocks de l'industrie $i_k$ (positifs) et répartir des poids négatifs sur les autres pour neutraliser le market → c'est l'**excès de rendement de l'industrie $i_k$ par rapport au marché**.
- **Pour un facteur style** $f_{s}$ (ex. value) : $\Omega_{k,n}$ est positif pour les stocks à forte exposition value, négatif pour les stocks growth → c'est le rendement d'un portefeuille **long value / short growth**, neutre sur tout le reste.

## 3.3 + Country (modèle complet)

Modèle complet sur un univers global :
$$
r = X_s f_s + X_{mi} f_{mi} + X_c f_c + u
$$
Où :
- $X_s$ : styles (sans contrainte).
- $X_{mi}$ : market + industries (contrainte somme pondérée nulle).
- $X_c$ : countries (contrainte somme pondérée nulle, pour la même raison qu'industries).

### Matrice de restriction simple (poids égaux)

> [!note]- Exemple : 3 industries, 3 countries, 2 styles
> Pour éviter la colinéarité avec le market, on impose $f_{i_3} = -f_{i_1} - f_{i_2}$ et $f_{c_3} = -f_{c_1} - f_{c_2}$.
> $R = \begin{bmatrix}
> 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
> 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
> 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
> 0 & -1 & -1 & 0 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
> 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
> 0 & 0 & 0 & -1 & -1 & 0 & 0 \\
> 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
> 0 & 0 & 0 & 0 & 0 & 0 & 1
> \end{bmatrix}_{9 \times 7}$
> Vecteur libre : $g = (m, f_{i_1}, f_{i_2}, f_{c_1}, f_{c_2}, f_{s_1}, f_{s_2})$.

### Matrice de restriction pondérée (par market cap)

La contrainte "somme nulle" doit en réalité être **pondérée par la capitalisation** :
$$
w_{i_1} f_{i_1} + w_{i_2} f_{i_2} + w_{i_3} f_{i_3} = 0 \quad\Longrightarrow\quad f_{i_3} = -\frac{w_{i_1}}{w_{i_3}} f_{i_1} - \frac{w_{i_2}}{w_{i_3}} f_{i_2}
$$
où $w_{i_k} = \sum_{n \in i_k} w_n$ est la cap totale (relative) de l'industrie $i_k$. Idem pour les countries.

> [!note]- Comment calculer $w_{i_k}$ en pratique
> Si $V = \operatorname{diag}(w_1, \dots, w_N)$ est la matrice des poids stocks et $X_{\text{indus}}$ la matrice des dummies industrie ($N \times 3$), alors :
> $\mathbf{1}_N^\top V X_{\text{indus}} = (w_{i_1}, w_{i_2}, w_{i_3})$
> C'est juste la somme des poids de chaque industrie. Même chose pour les countries.

La matrice $R$ devient :
$$
R = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & -\tfrac{w_{i_1}}{w_{i_3}} & -\tfrac{w_{i_2}}{w_{i_3}} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & -\tfrac{w_{c_1}}{w_{c_3}} & -\tfrac{w_{c_2}}{w_{c_3}} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}_{9 \times 7}
$$

La résolution est ensuite **strictement identique à §3.1.3** : $\hat{f} = R (R' X' V X R)^{-1} R' X' V \, r$.

# 4. Factor mimicking portfolios

## 4.1 Définition

Un portefeuille mimétique du facteur $k$ a pour but :
- exposition **unitaire** au facteur $k$,
- exposition **nulle** à tous les autres facteurs.

C'est-à-dire un portefeuille dont le rendement *isole* la prime du facteur $k$.

**Trois manières** de le construire, par ordre croissant de raffinement :
1. **Full replication** : on prend directement les lignes de $\Omega$.
2. **Total risk optimization** : on cherche le portefeuille qui maximise l'exposition au facteur cible sous contrainte de minimiser le risque total.
3. **Active risk optimization** : idem mais on minimise le tracking error par rapport au full replication, pour réduire le turnover.

## 4.2 Full replication

On a vu en §3.1.3 que $\hat{f} = \Omega r$ avec $\Omega = (X' V X)^{-1} X' V$ (en oubliant momentanément $R$ pour alléger). Donc :
$$
f_k = \sum_{n=1}^{N} \Omega_{k,n}\, r_n
$$
La **ligne $k$ de $\Omega$ est exactement le portefeuille mimétique du facteur $k$** : ses poids donnent par construction une exposition de 1 à $k$ et de 0 aux autres.

> [!note]- Exemple numérique
> Avec :
> $r = \begin{pmatrix} 0.10 \\ 0.08 \\ 0.12 \end{pmatrix}, \quad X = \begin{pmatrix} 1.0 & 0.7 & 1.5 \\ 0.9 & 1.2 & 0.8 \\ 1.1 & 0.5 & 2.0 \end{pmatrix}, \quad V = \operatorname{diag}(0.3, 0.4, 0.3)$
> On calcule $\Omega = (X' V X)^{-1} X' V \in \mathbb{R}^{3 \times 3}$.
> Chaque ligne donne les poids du portefeuille mimétique correspondant : ligne 1 = mimétique du facteur 1 (Size par exemple), ligne 2 = mimétique du facteur 2 (Value), etc.

> [!note] Remarques clés
> - Les poids ne somment **pas à 1** : c'est un portefeuille long/short, pas un portefeuille investi.
> - **"Réplication complète"** veut dire qu'on a un portefeuille mimétique pour *chacun* des $K$ facteurs (les $K$ lignes de $\Omega$).
> - C'est la méthode la plus directe mais elle a un **turnover élevé** : $\Omega$ change à chaque date $t$ (les expositions et la cap bougent), donc les poids du mimétique aussi.

## 4.3 Total risk optimization

Pour réduire le turnover et le risque total, on peut formuler le mimétique du facteur cible $\alpha$ comme un problème d'optimisation :
$$
\max_h \left\{ h^\top X_\alpha - \frac{\lambda}{2} h^\top V h \right\} \quad \text{s.c.} \quad h^\top X_\sigma = 0
$$
Où :
- $h \in \mathbb{R}^N$ : poids du portefeuille,
- $X_\alpha \in \mathbb{R}^N$ : exposition au facteur cible (la colonne de $X$ correspondante),
- $X_\sigma \in \mathbb{R}^{N \times (K-1)}$ : expositions aux autres facteurs (à neutraliser),
- $V$ : matrice de variance-covariance totale des actifs,
- $\lambda$ : paramètre de tolérance au risque.

**Lecture** : on maximise l'exposition au facteur cible (terme $h^\top X_\alpha$) tout en pénalisant le risque total ($h^\top V h$), avec contrainte d'exposition nulle aux autres facteurs.

> [!note]- Astuce : remplacer $V$ par la variance spécifique $D$
> Si on remplace $V$ par $D = \operatorname{diag}(\sigma_{u_1}^2, \dots, \sigma_{u_N}^2)$ (variance idiosyncratique seulement), la solution est **identique à une échelle près**. Pourquoi : la contrainte $h^\top X_\sigma = 0$ a déjà éliminé le risque commun, donc minimiser le risque total = minimiser le risque spécifique.

## 4.4 Active risk optimization

Variante où on minimise l'écart par rapport au portefeuille de full replication $h_F$ :
$$
\max_h \left\{ h^\top X_\alpha - \frac{\lambda}{2} (h - h_F)^\top V (h - h_F) \right\}
$$

Le terme $(h - h_F)^\top V (h - h_F)$ est le **tracking error** par rapport au full replication.

> [!note] Pourquoi cette formulation ?
> Le full replication ($h_F$) a deux défauts :
> 1. **Turnover élevé** → coûts de transaction.
> 2. **Positions extrêmes** (longues/courtes très grandes sur certains stocks) → difficile à implementer en pratique (contraintes de short-selling, de liquidité).
>
> En minimisant le tracking error vs $h_F$ tout en gardant l'exposition $h^\top X_\alpha$, on obtient un **portefeuille plus stable et plus implementable**, au prix d'une légère dégradation de la pureté de l'exposition factorielle.

## 4.5 Contraintes supplémentaires

En pratique on rajoute typiquement :
- **No-short** ou bornes : $h_n \geq 0$ ou $h_n \in [-h_{\max}, h_{\max}]$.
- **Budget** : $\sum_n h_n = 1$.
- **Contraintes de liquidité** : $|h_n| \leq c \cdot \text{ADV}_n$ (pourcentage du volume quotidien).
- **Turnover** : $\sum_n |h_n^{t} - h_n^{t-1}| \leq \tau$.

Ces contraintes transforment le QP en QP avec contraintes linéaires → résolution par solveur standard (CVXPY, Mosek…).

# 5. Stage 3 — Forecasting Risk

Une fois les factor returns $\hat{f}(t)$ estimés pour $t = 1, \dots, T$, on dispose d'une **série temporelle** de factor returns. On peut alors estimer la matrice de covariance des facteurs et construire des prévisions de risque pour n'importe quel portefeuille.

## 5.1 Décomposition du risque

### 5.1.1 Au niveau d'un stock individuel

On part du modèle Barra $r = X f + u$. La covariance des returns se décompose :

> [!note]- Dérivation étape par étape
> Hypothèses : $\operatorname{Cov}(f, u) = 0$ (les factor returns ne sont pas corrélés aux idio), et les $u_n$ sont mutuellement non corrélés ($\operatorname{Cov}(u_n, u_m) = 0$ pour $n \neq m$).
>
> $\operatorname{Var}(r) = \operatorname{Var}(X f + u) = \operatorname{Var}(X f) + \operatorname{Var}(u)$
> Pour le premier terme, $X$ est déterministe (connu en $t$) :
> $\operatorname{Var}(X f) = X \, \operatorname{Var}(f) \, X^\top = X F X^\top$
> Pour le second, par hypothèse d'indépendance des résidus :
> $\operatorname{Var}(u) = \operatorname{diag}(\sigma_{u_1}^2, \dots, \sigma_{u_N}^2) \equiv D$
> D'où :
> $\boxed{V \equiv \operatorname{Var}(r) = \underbrace{X F X^\top}_{\text{risque systématique}} + \underbrace{D}_{\text{risque spécifique}}}$

### 5.1.2 Au niveau du portefeuille (Barra risk)

Pour un portefeuille de poids $h \in \mathbb{R}^N$, le return est $r_p = h^\top r$, et sa variance :
$$
\sigma_p^2 = \operatorname{Var}(h^\top r) = h^\top V h = \underbrace{h^\top X F X^\top h}_{\text{risque commun (factoriel)}} + \underbrace{h^\top D h}_{\text{risque spécifique}}
$$

La **volatilité du portefeuille** (le "Barra risk") est :
$$
\boxed{\sigma_p = \sqrt{h^\top (X F X^\top + D)\, h}}
$$

> [!note] Pourquoi cette décomposition est puissante
> Sans modèle factoriel, estimer $V$ ($N \times N$) demande $\mathcal{O}(N^2)$ paramètres → ingouvernable pour $N = 5000$. Avec le modèle, on a $\mathcal{O}(K^2 + N)$ paramètres, soit $\mathcal{O}(100 + 5000)$ → estimable de façon stable. C'est la raison d'être du modèle Barra côté risk management.

### 5.1.3 Interprétation gaussienne et lien avec la VaR

Si on suppose les returns du portefeuille **gaussiens** :
$$
r_p \sim \mathcal{N}(\mu_p, \sigma_p^2)
$$
alors **toute l'information de risque tient dans le scalaire $\sigma_p$**. En standardisant, $Z = (r_p - \mu_p) / \sigma_p \sim \mathcal{N}(0, 1)$, et tout calcul de risque devient un quantile gaussien.

**VaR (Value-at-Risk) à niveau $\alpha$** : la perte qu'on ne dépassera pas avec probabilité $1 - \alpha$.
$$
\text{VaR}_\alpha = -(\mu_p + z_\alpha \sigma_p) \approx -z_\alpha \sigma_p \quad \text{(si } \mu_p \approx 0\text{)}
$$
Où $z_\alpha$ est le quantile gaussien : $z_{5\%} \approx -1.645$, $z_{1\%} \approx -2.326$.

**En dollars** : $\text{VaR}_\alpha^{\$} = |z_\alpha| \cdot \sigma_p \cdot W$, où $W$ est la valeur du portefeuille.

> [!note] Lecture opérationnelle
> Sur un portefeuille de \$10M avec $\sigma_p = 1\%$ par jour :
> - VaR 95% ≈ $1.645 \times 1\% \times \$10\text{M} = \$164\text{k}$
> - VaR 99% ≈ $2.326 \times 1\% \times \$10\text{M} = \$233\text{k}$
>
> Tout le modèle Barra (5 familles de facteurs, matière fiscale, contraintes WLS, EWMA, Newey-West…) sert *au final* à produire ce **scalaire $\sigma_p$** — qui, sous hypothèse gaussienne, suffit à décrire tout le risque du portefeuille.

> [!note]- Limite de l'hypothèse gaussienne
> En pratique les returns ont des **fat tails** (kurtosis > 3) : les événements extrêmes (krachs) sont plus fréquents qu'une gaussienne ne le prévoit. La VaR gaussienne **sous-estime** donc le risque de tail. Corrections classiques : VaR historique, Cornish-Fisher (corrige skewness/kurtosis), copules, ou EVT (Extreme Value Theory). Mais Barra reste l'épine dorsale du calcul de $\sigma_p$, sur lequel ces ajustements viennent se greffer.

## 5.2 Estimation de $F$ (covariance des facteurs)

### EWMA (Exponentially Weighted Moving Average)

Pour donner plus de poids aux observations récentes :
$$
\hat{F}_t = (1 - \lambda) \sum_{s=0}^{\infty} \lambda^s \, (f_{t-s} - \bar{f})(f_{t-s} - \bar{f})^\top
$$
Avec $\lambda \approx 0.94$ à $0.97$ (RiskMetrics utilise 0.94 pour le quotidien). Half-life $= \ln(0.5) / \ln(\lambda)$, soit ≈ 11 jours pour $\lambda = 0.94$ (court terme), ≈ 23 jours pour $\lambda = 0.97$ (moyen terme).

### Correction Newey-West (autocorrélation)

Les factor returns peuvent présenter de l'autocorrélation (microstructure, liquidité). Newey-West corrige en ajoutant des termes de cross-covariance décalés :
$$
\hat{F}_{NW} = \hat{F}_0 + \sum_{\ell=1}^{L} \left(1 - \frac{\ell}{L+1}\right) (\hat{F}_\ell + \hat{F}_\ell^\top)
$$
Où $\hat{F}_\ell = \operatorname{Cov}(f_t, f_{t-\ell})$. Le poids triangulaire $(1 - \ell/(L+1))$ assure que $\hat{F}_{NW}$ reste semi-définie positive.

### Shrinkage (Ledoit-Wolf)

$\hat{F}$ empirique est bruité, surtout quand $T$ n'est pas très grand devant $K$. On la régularise vers une cible structurée $F_0$ (souvent une matrice diagonale ou à corrélation constante) :
$$
\hat{F}_{\text{shrink}} = \delta F_0 + (1 - \delta) \hat{F}
$$
Ledoit-Wolf donne une formule fermée pour le $\delta$ optimal qui minimise l'erreur Frobenius espérée.

## 5.3 Estimation de $D$ (variance spécifique)

Pour chaque stock $n$ : $\hat{\sigma}_{u_n}^2 = \operatorname{Var}_t(\hat{u}_{n,t})$ avec EWMA. Souvent on **shrink vers une variance moyenne par groupe** (industrie, pays) pour stabiliser les estimations sur les stocks à historique court.

# 6. Applications

## 6.1 Gestion passive : hedging des expositions non désirées

Un fonds qui suit le **MSCI World Small Cap Index** est censé cibler le facteur Size, mais en pratique l'indice porte aussi des expositions résiduelles à d'autres facteurs (Value, Volatility…) → on veut les **neutraliser**.

> [!note]- Exemple chiffré
> Sur 3 small caps avec un portefeuille initial pondéré par cap :
>
> | Stock | Size | Value | Vol | Weight |
> |---|---|---|---|---|
> | A | 1.0 | 0.8 | 1.4 | 0.4 |
> | B | 1.0 | 0.9 | 1.3 | 0.3 |
> | C | 0.95 | 0.85 | 1.4 | 0.3 |
>
> Expositions agrégées **avant hedge** : Size = 0.99, Value = 0.84, Vol = 1.37.
> → L'indice est bien Size (1.0), mais aussi exposé Value (0.84) et Vol (1.37), non désiré.
>
> **Après hedge** (en ajoutant des positions long/short via les portefeuilles mimétiques de Value et Vol) : Size = 0.99 (inchangé), Value = 0, Vol = 0.
>
> Concrètement : on résout pour $h$ tel que $h^\top X_{\text{Value}} = 0$ et $h^\top X_{\text{Vol}} = 0$ tout en gardant $h^\top X_{\text{Size}} \approx 0.99$.

## 6.2 Gestion active : générer de l'alpha

L'investisseur cherche à **surperformer** un benchmark en s'exposant à des facteurs qu'il pense sous-pricés, ou en misé sur sa capacité à stock-picker (alpha pur, non factoriel).

Formulation typique (mean-variance avec contraintes d'exposition) :
$$
\max_h \left\{ h^\top \alpha - \frac{\lambda}{2} (h - h_B)^\top V (h - h_B) \right\}
$$
Où $\alpha$ est le vecteur des alphas prédits par le modèle et $h_B$ le portefeuille benchmark. On peut imposer :
- **Neutralité factorielle** sur certains facteurs : $(h - h_B)^\top X_k = 0$.
- **Tilts contrôlés** sur d'autres facteurs : $|(h - h_B)^\top X_k| \leq c_k$.

## 6.3 Attribution de performance

La performance d'un portefeuille se décompose en contributions factorielles :
$$
h^\top r = \underbrace{h^\top X f}_{\text{commun}} + \underbrace{h^\top u}_{\text{spécifique}} = \sum_k \underbrace{(h^\top X_k)}_{\text{exposition}} \cdot \underbrace{f_k}_{\text{factor return}} + \underbrace{h^\top u}_{\text{stock-picking}}
$$

On peut donc décomposer le rendement réalisé sur une période :
- combien vient de l'exposition Value $\times$ rendement Value ?
- combien de l'exposition Tech $\times$ rendement Tech ?
- combien du stock-picking pur (terme $h^\top u$) ?

C'est la base de l'**attribution de risque ex-ante** (avec $F$ à la place de $f$) et de l'**attribution de performance ex-post** (avec les $f$ réalisés).

> [!note] Lien avec l'IR (Information Ratio)
> Pour un portefeuille actif (vs benchmark), on note $h_a = h - h_B$ les poids actifs. L'IR se décompose en :
> $\text{IR} = \frac{\mathbb{E}[h_a^\top r]}{\sqrt{\operatorname{Var}(h_a^\top r)}}$
> et chaque facteur contribue à la fois au numérateur (via $h_a^\top X_k \cdot \mathbb{E}[f_k]$) et au dénominateur (via $h_a^\top X F X^\top h_a$). C'est la métrique clé pour comparer des stratactives à budgets de risque différents.

## 6.4 Attribution du P&L en dollars

Les sections précédentes raisonnent en *returns* (sans dimension, en %). Sur un desk de trading, on veut **attribuer un P&L en dollars** : "le trader a gagné +10k$ aujourd'hui, c'est dû à quoi ?"

### Dérivation

**Notations** : $h_n$ = nombre de shares du stock $n$, $S_n$ = prix, $\Delta^{cash}_n = h_n \cdot S_n$ = exposition en dollars (notional), $r_n$ = return du stock $n$.

P&L stock-par-stock :
$$
\Delta\text{P\&L}_n = h_n (S_n^{t+1} - S_n^t) = h_n S_n^t \cdot \frac{S_n^{t+1} - S_n^t}{S_n^t} = \Delta^{cash}_n \cdot r_n
$$

Au niveau portefeuille :
$$
\boxed{\Delta\text{P\&L} = (\Delta^{cash})^\top r}
$$

> [!note]- Subtilité : excess return vs total return
> Barra estime ses factor returns sur des **excess returns** $r^{ex} = r - r_f$. Donc rigoureusement :
> $\Delta\text{P\&L} = (\Delta^{cash})^\top r = \underbrace{(\Delta^{cash})^\top \mathbf{1} \cdot r_f}_{\text{P\&L de financement}} + (\Delta^{cash})^\top r^{ex}$
> Le premier terme est le P&L de cash/financement (intérêts). Le second est ce qu'on attribue aux facteurs.
> Pour un portefeuille **dollar-neutral** ($(\Delta^{cash})^\top \mathbf{1} = 0$, long/short équilibrés), le terme de financement disparaît et $\Delta\text{P\&L} \approx (\Delta^{cash})^\top r^{ex}$.

### Substitution du modèle Barra

En injectant $r = X f + u$ :
$$
\Delta\text{P\&L} = (\Delta^{cash})^\top X f + (\Delta^{cash})^\top u
$$

Facteur par facteur :
$$
\Delta\text{P\&L} = \sum_k \underbrace{\big( (\Delta^{cash})^\top X_k \big)}_{\text{exposition }\$\text{ au facteur } k} \cdot \underbrace{f_k}_{\text{factor return}} + \sum_n \underbrace{\Delta^{cash}_n}_{\text{notional}} \cdot \underbrace{u_n}_{\text{idio return}}
$$

### Tableau des unités

| Quantité | Unité | Sens |
|---|---|---|
| $X_{n,k}$ | sans dim. (z-score) | exposition du stock $n$ au facteur $k$ |
| $f_k$ | % | rendement du facteur $k$ |
| $X_{n,k} \cdot f_k$ | % | contribution du facteur $k$ au return du stock $n$ |
| $(\Delta^{cash})^\top X_k$ | $\$ | **exposition en dollars** du portefeuille au facteur $k$ |
| $(\Delta^{cash})^\top X_k \cdot f_k$ | $\$ | **P&L en dollars** dû au facteur $k$ |
| $(\Delta^{cash})^\top u$ | $\$ | **P&L en dollars** dû au stock-picking (idio) |

> [!note]- Exemple : trader qui gagne +10k$
> Portefeuille du jour, expositions § et factor returns observés :
>
> | Facteur | Exposition $\$ | Factor return | Contrib P&L |
> |---|---|---|---|
> | Tech | +200 000 | +3.0% | **+6 000$** |
> | Momentum | +150 000 | +2.0% | **+3 000$** |
> | Value | −50 000 | +1.0% | −500$ |
> | Size | +30 000 | −0.5% | −150$ |
> | … | … | … | … |
> | **Total facteurs** | | | **+8 350$** |
> | **Idiosyncratique** ($\sum_n \Delta^{cash}_n u_n$) | | | **+1 650$** |
> | **Total P&L** | | | **+10 000$** |
>
> Lecture : sur les +10k$, environ 83% vient des paris factoriels (dont +6k$ rien que sur Tech) et 17% du stock-picking pur. Si le trader pense générer de l'**alpha**, il devrait surtout regarder le terme idiosyncratique — le reste, c'est juste une exposition factorielle qu'il aurait pu obtenir avec un ETF.


## Attribution ex-ante (risk budget) vs ex-post (P&L)

- **Ex-post** : on observe $f_k$ réalisés → attribution du **P&L réalisé** comme ci-dessus.
- **Ex-ante** : on remplace $f_k$ par sa **volatilité** $\sigma_{f_k}$ (issue de $F$) → attribution de la **VaR ou du risk budget** entre facteurs avant la prise de position. Permet de répondre à "si je perds gros demain, ce sera probablement à cause de quel facteur ?".

## 6.5 Risque du P&L (vol en $ vs vol en %)

Jusqu'ici on a deux objets qui semblaient séparés :
- En §5.1 : la volatilité du portefeuille $\sigma_p$ en **%** ("le portefeuille a 1.2% de vol par jour").
- En §6.4 : le P&L en **dollars**.

En fait c'est exactement la **même chose**, vue sous deux angles. Le P&L est une variable aléatoire (parce que $r$ l'est), donc on peut directement calculer sa variance :
$$
\sigma_{P\&L}^2 = \operatorname{Var}\big((\Delta^{cash})^\top r\big) = (\Delta^{cash})^\top V (\Delta^{cash})
$$

En injectant Barra :
$$
\boxed{\sigma_{P\&L} = \sqrt{(\Delta^{cash})^\top (X F X^\top + D)\, (\Delta^{cash})}}
$$

C'est **strictement la même formule** que $\sigma_p$ du §5.1.2, sauf qu'on a remplacé $h$ (poids relatifs, $\sum h_n = 1$) par $\Delta^{cash}$ (notional en \$). Le résultat sort donc **directement en dollars**, pas en %.

### Lien explicite

Si $W$ est la valeur totale du portefeuille, alors $\Delta^{cash} = W \cdot h$, d'où :
$$
\sigma_{P\&L} = W \cdot \sigma_p
$$

| Si on met... | $\sigma$ sort en... | Sens opérationnel |
|---|---|---|
| $h$ (poids relatifs) | sans dim. (% par jour) | comparer des portefeuilles à tailles différentes |
| $\Delta^{cash}$ ($\$) | dollars (\$ par jour) | comparer au stop-loss, au risk budget, à la P&L attendue |

La **VaR en dollars** déjà vue en §5.1.3 est juste $\text{VaR}^{\$}_\alpha = |z_\alpha| \cdot \sigma_{P\&L}$.

> [!note]- Exemple : risque du trader à +10k$
> On reprend le portefeuille de l'exemple §6.4 ($+10\text{k}\$ de P&L réalisé). Supposons que le modèle Barra prévoyait :
> $\sigma_{P\&L} = \sqrt{(\Delta^{cash})^\top (X F X^\top + D)\, (\Delta^{cash})} = \$8\,500$
>
> Lectures possibles :
> - **VaR 95% ex-ante** : $1.645 \times 8\,500 \approx \$14\,000$. Le trader risquait de perdre jusqu'à \$14k dans le pire 5% des scénarios.
> - **Sharpe réalisé du jour** : $\$10\,000 / \$8\,500 \approx 1.18$. Bon ratio risque/rendement.
> - **Risk budget** : si le trader a un budget de vol de \$10k/jour, il était dans son enveloppe (\$8.5k < \$10k).
>
> Tout ce que dit Barra in fine, c'est : **"voici un scalaire $\sigma_{P\&L}$ qui résume la distribution du P&L de demain"**. Sous hypothèse gaussienne, $\sigma_{P\&L}$ est tout ce qu'il y a à savoir.

### Décomposition du risque P&L par facteur (Marginal Contribution to Risk)

De la même façon qu'on a décomposé le **P&L réalisé** facteur par facteur en §6.4, on peut décomposer le **risque** du P&L. La contribution marginale du facteur $k$ au risque total est :
$$
\text{MCR}_k = \frac{\partial \sigma_{P\&L}}{\partial (\Delta^{cash}_X)_k} \cdot (\Delta^{cash}_X)_k
$$
où $\Delta^{cash}_X = X^\top \Delta^{cash}$ est le vecteur des expositions en $ aux facteurs.

La propriété clé : $\sum_k \text{MCR}_k + \text{MCR}_{\text{idio}} = \sigma_{P\&L}$ → on a une **vraie décomposition additive** du risque total entre facteurs et idiosyncratique. C'est l'analogue *ex-ante* de l'attribution P&L *ex-post* du §6.4.

## 6.6 P&L multi-régions et waterfall

Les sections §6.3 à §6.5 supposent **un seul modèle Barra** (donc un seul ensemble de facteurs $X, f$). En pratique, un portefeuille global détient des stocks de plusieurs régions, chacune décrite par son propre modèle Barra (cf. §1.3). Le waterfall permet alors de faire l'attribution P&L de façon **rigoureuse, par région**, puis d'agréger.

### Problème

Un fonds long-short global détient :
- 50 stocks américains (50% du book),
- 30 stocks européens (30%),
- 15 stocks chinois (H-shares + A-shares mélangés, 15%),
- 5 stocks japonais (5%).

Question : comment décomposer le P&L total "facteur par facteur" alors qu'**il n'y a pas de facteur Tech commun** entre `bausfastd` et `baase2s` (les deux modèles ont des facteurs Tech, mais ce ne sont pas le même objet) ?

### Méthode : décomposer par région via le waterfall

On applique l'algorithme suivant :

1. **Bucketer chaque stock** dans son modèle Barra le plus spécifique via le waterfall (cf. §1.3) : USA → `bausfastd`, Europe → `baeutrd`, etc.
2. **Calculer l'attribution P&L par région** indépendamment, avec les facteurs et expositions du modèle correspondant.
3. **Agréger** : le P&L total est la somme des P&L régionaux.

$$
\Delta\text{P\&L}_{\text{total}} = \sum_{\text{région } R} \Delta\text{P\&L}_R = \sum_R \Big( \sum_{k \in F_R} (\Delta^{cash}_R)^\top X_k^R \cdot f_k^R + (\Delta^{cash}_R)^\top u_R \Big)
$$

Où $F_R$ est l'ensemble des facteurs du modèle régional $R$, $X^R$ ses expositions, $f^R$ ses factor returns.

> [!note]- Exemple : portefeuille \$100M sur 4 régions, P&L de la journée
>
> | Région | Modèle | Notional | Top facteurs contributeurs | P&L région | Idio |
> |---|---|---|---|---|---|
> | USA | `bausfastd` | +50M\$ | Tech US +30k\$, Momentum US +12k\$ | **+45k\$** | +3k\$ |
> | Europe | `baeutrd` | +30M\$ | DAX −8k\$, EUR/USD +5k\$ | **−2k\$** | −1k\$ |
> | Chine (H) | `bacxe1s` | +10M\$ | Tech CN +6k\$ | **+8k\$** | +2k\$ |
> | Japon | `bajep4d` | +5M\$ | Yen −2k\$ | **−1k\$** | +0.5k\$ |
> | **Total** | | | | **+50k\$** | **+4.5k\$** |
>
> Lecture : sur les +54.5k\$ de la journée, l'essentiel vient des facteurs US (Tech + Momentum). L'Europe a perdu sur l'exposition DAX. Pas de double comptage entre régions parce que chaque stock n'apparait que dans **un seul** modèle (celui sélectionné par le waterfall).

### Pourquoi pas tout passer par `bagemtrd` ?

On pourrait être tenté d'utiliser le modèle global pour tout, ce qui donnerait des facteurs comparables entre régions ("Tech global", "Momentum global"…). Problème : le modèle global **dilue** les spécificités régionales. Si la tech US monte de +3% alors que la tech asiatique baisse de −1%, un facteur "Tech global" verra +1% (moyenne pondérée) et le P&L régional sera mal attribué.

> [!note] Compromis pratique
> - **Attribution intra-région** : on utilise le modèle spécifique (waterfall) → finesse maximale.
> - **Vue inter-région** : on utilise `bagemtrd` en complément, par exemple pour comparer la contribution "Tech" totale entre US et Asie sur des facteurs *commensurables*.
>
> En pratique les desks gardent les **deux vues en parallèle** : waterfall pour l'attribution fine, GEM pour la cohérence inter-régions.

### Lien avec le cross-impact

Le même mécanisme waterfall est utilisé pour le cross-impact d'un basket multi-régions : voir [[(iii) Cross-impact|note Cross-impact §IV]] pour le détail. La logique est la même : non-comparabilité des facteurs inter-régions → décomposition par bloc régional.
