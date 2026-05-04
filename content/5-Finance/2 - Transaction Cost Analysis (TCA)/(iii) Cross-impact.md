---
title: c-Cross-impact
order: 3
---
# Cross-impact

> Les modèles de market impact présentés dans [[(ii) Market Impact - Modèles et calibration|note ii]] traitent **un actif isolé** : impact de mes ordres sur **mon** prix, sans considérer les autres actifs. Cette hypothèse "vase clos" est inadaptée dès qu'on **trade un basket** d'actifs corrélés. En vendant 100M$ d'Apple, je ne fais pas que bouger AAPL — je bouge aussi Microsoft, Google, Nvidia, parce que ces stocks partagent des facteurs de risque communs (tech, momentum, growth…). Le **cross-impact** est précisément la modélisation de ces effets croisés. Cette note présente la construction la plus naturelle : passer par un **modèle factoriel** (de type Barra) pour projeter l'impact dans l'espace des facteurs, puis le redistribuer aux autres actifs via leurs expositions. Référence canonique : Tomas, Mastromatteo, Benzaquen (2022), *How to build a cross-impact model from first principles*.

## I. Pourquoi le cross-impact ?

### A. La somme des coûts n'est pas le coût de la somme

Considérons un portefeuille à liquider sur 5 large-caps tech américaines (AAPL, MSFT, GOOG, NVDA, META). Les modèles de self-impact donnent un coût pour chaque actif pris isolément. La tentation naturelle serait d'écrire :

$$\text{Coût total} \;\overset{?}{=}\; \sum_{i=1}^{5} c_{\text{self}}^{i}.$$

> [!warning] Cette formule est fausse en pratique
> Quand on liquide les 5 stocks **simultanément**, l'impact de chacun se renforce mutuellement à travers les facteurs communs. Vendre AAPL fait baisser le facteur tech ; cette baisse de facteur se transmet à MSFT, GOOG, NVDA, META — **avant même** qu'on ait commencé à vendre ces stocks. Quand on commence à vendre MSFT, on part d'un prix déjà déprimé par la vente d'AAPL. La somme des coûts individuels **sous-estime** le coût total.

> 💡 **L'image canonique.** Le self-impact répond à la question *"si je suis seul à trader, combien me coûte mon trade ?"*. Le cross-impact répond à *"si je trade plusieurs stocks corrélés en même temps, combien me coûte mon basket ?"*. Sur un basket décorrélé (ex. AAPL + une mining junior canadienne), les deux questions ont la même réponse. Sur un basket très corrélé (5 large-caps tech), les deux divergent fortement.

### B. Décomposition du coût total

La décomposition naturelle du coût total est :

$$\boxed{\;\text{Coût total} = \underbrace{\sum_{i} c_{\text{self}}^{i}}_{\text{self-impact (}\\textit{vase clos}\text{)}} + \underbrace{\sum_{i} c_{\text{cross}}^{i}}_{\text{cross-impact (interactions)}}\;}$$

Le self-impact est calculé stock par stock par les modèles de [[(ii) Market Impact - Modèles et calibration|note ii]] (square-root, Almgren). Le cross-impact $c_{\text{cross}}^{i}$ capture **l'impact additionnel** sur le stock $i$ provoqué par les trades simultanés sur les **autres** stocks $j \neq i$ — via les facteurs communs.

> [!note]- Pourquoi pas de spread cost dans le cross-impact ?
> Le spread cost est un coût d'**exécution** : tu le payes parce que tu traverses le bid-ask sur **ton** ordre. Il n'y a pas de mécanisme par lequel mon ordre sur AAPL me ferait payer le spread de MSFT — ces sont deux exécutions séparées avec leurs spreads propres. Le cross-impact, lui, est un effet de **prix** : mon ordre sur AAPL déforme le prix d'équilibre de MSFT via les facteurs communs. Le coût additionnel sur MSFT vient de cette déformation, pas d'un coût de transaction.
> 
> Mathématiquement : le cross-impact est une déformation de la **trajectoire de prix** des autres actifs, pas un surcoût d'exécution. Le self-impact contient bien une composante spread cost (qu'on traverse en exécutant) ; le cross-impact n'en contient pas.

## II. L'idée centrale : passer par les facteurs

### A. Modèle factoriel de base

On part du modèle factoriel standard de type Barra :

> [!warning] Modèle factoriel
> $$r = X f + u$$
> 
> où :
> - $r \in \mathbb{R}^n$ : vecteur des **returns** des $n$ stocks de l'univers,
> - $X \in \mathbb{R}^{n \times d}$ : matrice des **expositions factorielles** (loadings) — $X_{i,k}$ = exposition du stock $i$ au facteur $k$,
> - $f \in \mathbb{R}^d$ : vecteur des **rendements factoriels** ($d$ facteurs : industrie, style, pays…),
> - $u \in \mathbb{R}^n$ : **résidu spécifique** au stock (idiosyncratique).

L'idée centrale : si $r$ et $f$ sont liés par $r = Xf + u$, alors une perturbation des returns (causée par mes trades) se transmet à une perturbation des facteurs, puis re-transmet aux autres stocks via leurs propres expositions. Le cross-impact n'est rien d'autre que cette transmission.

### B. Factor mimicking portfolio

Pour passer des returns aux facteurs, on a besoin de l'**inverse** de la relation $r = Xf + u$. Sous une régression cross-sectionnelle GLS standard, on obtient :

> [!warning] Factor mimicking portfolio
> $$f = H r$$
> 
> où $H \in \mathbb{R}^{d \times n}$ est la matrice du **factor mimicking portfolio**, obtenue par cross-section régression GLS :
> 
> $$H = (X' \Omega^{-1} X)^{-1} X' \Omega^{-1}$$
> 
> avec $\Omega$ la matrice de covariance des résidus spécifiques (typiquement diagonale, contenant les *specific risks* par stock).

Concrètement, la ligne $k$ de $H$ donne les poids du portefeuille qui réplique le facteur $k$ — d'où le nom *factor mimicking*. C'est l'objet standard des modèles factoriels de Barra et de la littérature de portefeuille (cf. Connor-Goldberg-Korajczyk).

### C. La chaîne de transmission

On suppose que mes trades en masse sur une courte durée provoquent une variation des returns proportionnelle au self-impact $\Delta c$ (en bps) :

$$\Delta r = \alpha \cdot \Delta c, \qquad \alpha = 1$$

Le choix $\alpha = 1$ exprime que *l'impact que je provoque sur les returns est exactement le coût en bps que j'identifie comme self-impact* — on travaille dans les mêmes unités. À partir de là, la transmission se déroule en deux étapes :

> [!warning] Chaîne de transmission
> **Étape 1 — Returns → facteurs.** Mes trades déforment les returns, qui déforment à leur tour les facteurs :
> $$\Delta f = H \, \Delta c$$
> 
> **Étape 2 — Facteurs → returns (sur les autres stocks).** Les facteurs perturbés se retransmettent à tous les stocks via leurs expositions :
> $$\Delta r = X \, \Delta f = X H \, \Delta c$$

L'opérateur $XH$ est une matrice $n \times n$ qui exprime **comment l'impact se propage entre stocks** via les facteurs. C'est l'objet central du cross-impact.

> 💡 **Lecture géométrique.** $H$ projette les returns sur l'espace des facteurs (de dimension $d \ll n$). $X$ relifte des facteurs vers les returns. Composer les deux donne $XH$ : un opérateur de **rang faible** ($\leq d$) qui ne propage que la composante "factorielle" de mon impact, en filtrant la composante idiosyncratique. C'est cohérent avec l'intuition : un trade sur AAPL n'impacte pas un mining junior canadien (pas de facteur commun), il impacte les autres tech (facteurs communs).

## III. Algorithme : calculer le cross-impact stock par stock

### A. La formule par stock

Pour calculer le cross-impact subi par le stock $i$, on isole les contributions des **autres** stocks. Concrètement :

> [!warning] Cross-impact du stock $i$
> $$\Delta c^{i}_{\text{cross}} = X_{i,:} \cdot H \cdot \Delta c^{(-i)}$$
> 
> où :
> - $X_{i,:}$ est la **ligne $i$** de la matrice d'expositions (vecteur ligne de dimension $d$),
> - $H$ est le factor mimicking portfolio ($d \times n$),
> - $\Delta c^{(-i)} \in \mathbb{R}^n$ est le **vecteur complet** des self-impacts $(\Delta c_1, \ldots, \Delta c_n)$ avec **la composante $i$ remplacée par 0**.

L'idée : on prend tous les self-impacts du portefeuille, on annule celui du stock $i$ lui-même (parce qu'on ne veut pas compter son self-impact dans son cross-impact — ce serait du double comptage), on projette le reste sur l'espace des facteurs ($H \Delta c^{(-i)}$), et on relifte sur le stock $i$ via son exposition $X_{i,:}$.

### B. Algorithme

L'algorithme global est une simple boucle sur les stocks du portefeuille :

```
Inputs:  n stocks, vecteur des self-impacts Δc ∈ ℝⁿ
         Matrice X (n × d) des expositions Barra
         Matrice H (d × n) du factor mimicking portfolio

Pour i = 1, ..., n:
    Δc_temp ← copie de Δc
    Δc_temp[i] ← 0                            # on annule le stock courant
    Δc_cross[i] ← X[i, :] · H · Δc_temp       # cross-impact subi par le stock i

Coût total = Σ Δc[i] (self) + Σ Δc_cross[i] (cross)
```

### C. Lecture interprétative via $\Delta f$

Le calcul intermédiaire $\Delta f = H \Delta c$ a une **lecture indépendante très utile** : il dit *quels facteurs ai-je bougés à travers mes trades*.

> [!example] Diagnostic factoriel d'un trade
> Si je liquide un basket tech-heavy et que mon $\Delta f$ donne :
> 
> | Facteur | $\Delta f$ (bps) |
> |---|:---:|
> | Industry: Tech | −18 |
> | Style: Momentum | −7 |
> | Style: Growth | −5 |
> | Country: USA | −1 |
> | Industry: Energy | +0.2 |
> 
> Lecture : *mon trade a tiré le facteur tech vers le bas de 18 bps, le momentum de 7 bps, le growth de 5 bps. L'effet pays USA est négligeable, et l'énergie est inchangée*.
> 
> C'est un outil de diagnostic puissant : on identifie immédiatement **quels facteurs** sont stressés par notre exécution, et donc quels autres stocks (non explicitement tradés) vont subir le cross-impact le plus fort — typiquement ceux à forte exposition tech / momentum / growth.

## IV. Choix du modèle factoriel : la cascade waterfall

### A. Le problème pratique

MSCI Barra ne fournit pas un modèle factoriel unique mondial — il publie une **suite de modèles régionaux et globaux**, chacun calibré sur un univers donné avec ses propres facteurs et ses propres expositions. Pour un stock donné, plusieurs modèles peuvent s'appliquer (un stock français est dans le modèle Europe **et** dans le modèle Global), mais ils ne donnent **pas les mêmes loadings**.

Le choix : utiliser **le modèle le plus spécifique** possible pour chaque stock — un modèle régional étroit captera mieux les facteurs locaux qu'un modèle mondial qui les dilue.

### B. La cascade waterfall

La méthode **waterfall** consiste à essayer les modèles dans un ordre de spécificité décroissante, et à associer chaque stock au premier modèle qui le contient :

> [!warning] Cascade des modèles MSCI Barra (du plus spécifique au plus général)
> 1. **`bausfastd`** — modèle USA
> 2. **`baeutrd`** — modèle Europe
> 3. **`bacne5s`** — modèle Chine national (mainland A-shares)
> 4. **`bacxe1s`** — modèle Chine extérieur (H-shares, ADR)
> 5. **`baine2l`** — modèle Inde
> 6. **`bajep4d`** — modèle Japon
> 7. **`baase2s`** — modèle Asie (ex-Japon)
> 8. **`bagemtrd`** — modèle Global (Global Equity Model)
> 9. **`other`** — fallback pour les stocks non couverts par les modèles ci-dessus

> 💡 **Logique de la cascade.** Un stock français se retrouvera dans `baeutrd` (Europe) plutôt que dans `bagemtrd` (Global) parce que le modèle Europe contient des facteurs spécifiques (pays France, secteurs européens, devises EUR) qui captent mieux son risque que les facteurs globaux. Un stock indien tombera dans `baine2l` plutôt que `baase2s` (Asie) parce que le modèle Inde est plus fin sur l'univers domestique. Le `bagemtrd` sert de **filet de sécurité** : si un stock n'est dans aucun modèle régional, on l'attrape avec le modèle global. Le tag `other` capture les rares cas (stocks frontier, listings très récents, instruments exotiques) absents même du modèle global.

### C. Implication pour la matrice $X$

En pratique, la matrice $X$ utilisée dans l'algorithme **n'est pas issue d'un modèle factoriel unique** — c'est une matrice **assemblée par bloc** :

- les stocks USA prennent leurs expositions depuis `bausfastd`,
- les stocks Europe depuis `baeutrd`,
- les stocks chinois depuis `bacne5s` ou `bacxe1s` selon le listing,
- etc.

Chaque sous-univers vient avec son propre ensemble de facteurs et ses propres expositions. Le calcul du cross-impact s'effectue alors **par sous-univers** (stocks USA entre eux, stocks Europe entre eux…), parce que le cross-impact entre un stock USA et un stock chinois est mal défini si on n'a pas de facteur commun aux deux modèles.

> [!note]- Cross-impact inter-régions ?
> Pour traiter les baskets vraiment globaux (ex. un fonds long-short qui détient AAPL et Tencent simultanément), il existe deux approches :
> 1. **Tout passer par `bagemtrd`** (modèle global). Avantage : tous les stocks ont des expositions sur les mêmes facteurs, le cross-impact est calculable sur toute la population. Inconvénient : on perd la finesse régionale — l'effet "tech US" et l'effet "tech Asie" sont mélangés dans un facteur tech global.
> 2. **Combiner cascade + global** : utiliser la cascade pour le cross-impact intra-région (où c'est le plus précis), et `bagemtrd` pour l'inter-région (où c'est mieux que rien). Plus complexe à implémenter, mais plus précis en pratique.

## V. Considérations pratiques

### A. Pas de spread dans le cross-impact

Comme évoqué en I.B (callout collapsible), le spread cost n'apparaît que dans le self-impact. Le cross-impact est purement un effet de prix, pas un coût d'exécution. Conséquence : la formule $\Delta c^i_{\text{cross}} = X_{i,:} H \Delta c^{(-i)}$ utilise des $\Delta c$ **hors spread** — concrètement, on retire la composante spread cost des self-impacts avant de les injecter dans le cross-impact.

### B. Magnitude relative self vs cross

En pratique, sur un basket large-cap diversifié, le cross-impact est **du même ordre de grandeur** que le self-impact, voire supérieur sur un basket très concentré sectoriellement. Quelques règles de pouce empiriques :

> [!example] Ordres de grandeur typiques
> | Type de basket | Ratio cross / self |
> |---|:---:|
> | Basket très diversifié (50 stocks, secteurs variés) | 10–30% |
> | Basket sectoriel (10 stocks tech US) | 50–100% |
> | Basket factor-pur (long momentum, short value) | 80–150% |
> 
> Le cas extrême est le basket *factor-pur* : par construction, tous les stocks chargent les mêmes facteurs avec le même signe, donc l'effet de transmission est maximal. À l'inverse, un basket bien diversifié bénéficie de l'annulation partielle des effets factoriels (les exposures des longs et shorts se compensent partiellement dans $H \Delta c$).

### C. Non-additivité et ordre d'exécution

Le modèle ci-dessus suppose que les $n$ stocks sont tradés **simultanément** sur la même fenêtre. En pratique, les exécutions s'étalent dans le temps et ne sont pas synchrones. Deux régimes :

- **Trades simultanés** : l'algorithme tel quel s'applique. Le cross-impact est instantané.
- **Trades séquentiels** : si on trade AAPL puis MSFT, le cross-impact d'AAPL sur MSFT a déjà été partiellement digéré par le marché au moment où on commence MSFT. Il faut ajouter une **décroissance temporelle** (à la propagator), ce qui sort du cadre simple présenté ici.

Pour un usage pre-trade (estimation de coût avant exécution), l'hypothèse simultanée est raisonnable. Pour un usage post-trade fin (décomposition d'un coût observé), il faut un modèle de propagator.

## VI. Limites et extensions

### A. Linéarité

Le modèle $\Delta r = XH \Delta c$ est **linéaire** en $\Delta c$. Cela cohabite mal avec la non-linéarité (concavité) du self-impact lui-même : on a $c_{\text{self}}(Q) \propto \sqrt{Q}$ ou $\ln(Q)$ selon le modèle, donc une grande quantité ne produit pas une grande variation linéaire de prix. Le cross-impact présenté **suppose** que les $\Delta c$ d'entrée sont déjà calculés (par les modèles non-linéaires de note ii), puis les transmet linéairement via les facteurs.

Cette hybridation (self non-linéaire + cross linéaire) est la pratique standard. Des modèles plus rigoureux (Tomas et al. 2022) dérivent un cross-impact non-linéaire à partir de premiers principes, mais sont plus lourds à calibrer.

### B. Stationnarité de $XH$

L'opérateur $XH$ est calibré sur un historique de returns/factor returns. Il est **supposé stationnaire** — c'est-à-dire que la structure de propagation factorielle est la même hier, aujourd'hui, et demain. C'est raisonnable en régime normal de marché, mais peut casser en période de **stress** (crise 2008, COVID mars 2020, où les corrélations explosent et les facteurs habituels se réorganisent).

### C. Au-delà du factor model : approche propagator

L'approche factor-based présentée ici est l'une des deux grandes familles de cross-impact. L'alternative est l'approche **propagator** :
- Mastromatteo, Benzaquen, Eisler, Bouchaud (2017) — *Trading lightly: Cross-impact and optimal portfolio execution*
- Benzaquen, Mastromatteo, Eisler, Bouchaud (2017) — *Dissecting cross-impact on stock markets: An empirical analysis*

Ces modèles renoncent au passage explicite par les facteurs et estiment directement la matrice de cross-impact $\Lambda$ (telle que $\Delta P = \Lambda \cdot Q$) à partir de tick data, avec un noyau de décroissance temporelle. Plus flexible, plus puissant sur des univers où les facteurs ne capturent pas tout — mais beaucoup plus lourd à calibrer (besoin de tick data sur tous les stocks de l'univers).

---

## Résumé — fil logique

| Section | Contenu |
|---|---|
| **I** | Le self-impact en vase clos sous-estime le coût d'un basket : il ignore les transmissions inter-stocks via les facteurs communs. Décomposition coût total = $\sum$ self + $\sum$ cross. Pas de spread dans le cross. |
| **II** | Modèle factoriel $r = Xf + u$, factor mimicking $f = Hr$ avec $H = (X'\Omega^{-1}X)^{-1}X'\Omega^{-1}$. Chaîne de transmission : $\Delta r = XH \Delta c$. |
| **III** | Algorithme : pour chaque stock $i$, $\Delta c^i_{\text{cross}} = X_{i,:} H \Delta c^{(-i)}$ avec $\Delta c^{(-i)}$ le vecteur des self-impacts avec la composante $i$ annulée. Diagnostic via $\Delta f$ : "quels facteurs ai-je bougés ?". |
| **IV** | Cascade waterfall MSCI Barra : USA → Europe → Chine N → Chine X → Inde → Japon → Asie → Global → Other. Chaque stock prend ses expositions depuis le 1er modèle qui le contient. |
| **V** | Pas de spread, ratio cross/self typique 10–150% selon diversification, cas simultané vs séquentiel. |
| **VI** | Linéarité (cross linéaire, self non-linéaire), stationnarité de $XH$, alternative propagator (Mastromatteo, Benzaquen, Bouchaud). |

## Références

- **Tomas, Mastromatteo, Benzaquen** (2022). *How to build a cross-impact model from first principles*. Quantitative Finance.
- **Mastromatteo, Benzaquen, Eisler, Bouchaud** (2017). *Trading lightly: Cross-impact and optimal portfolio execution*.
- **Benzaquen, Mastromatteo, Eisler, Bouchaud** (2017). *Dissecting cross-impact on stock markets: An empirical analysis*. Journal of Statistical Mechanics.
- **MSCI Barra** — documentation officielle des modèles factoriels (référence par code modèle).
- **Connor, Goldberg, Korajczyk** (2010). *Portfolio Risk Analysis*. Princeton University Press — référence standard sur les modèles factoriels.
