---
title: c-Cross-impact
order: 3
---
# Cross-impact

> Les modèles de market impact présentés dans [[02_Market Impact - Modèles et calibration|note ii]] traitent **un actif isolé** : impact de mes ordres sur **mon** prix, sans considérer les autres actifs. Cette hypothèse "vase clos" est inadaptée dès qu'on **trade un basket** d'actifs corrélés. En vendant 100M$ d'Apple, je ne fais pas que bouger AAPL — je bouge aussi Microsoft, Google, Nvidia, parce que ces stocks partagent des facteurs de risque communs (tech, momentum, growth…). Le **cross-impact** est précisément la modélisation de ces effets croisés. Cette note présente d'abord la construction la plus naturelle — passer par un **modèle factoriel** (de type Barra) pour projeter l'impact dans l'espace des facteurs, puis le redistribuer aux autres actifs via leurs expositions — puis présentera dans une seconde section un algorithme alternatif. Référence canonique : Tomas, Mastromatteo, Benzaquen (2022), *How to build a cross-impact model from first principles*.

## I. Pourquoi le cross-impact ?

### A. La somme des coûts n'est pas le coût de la somme

Considérons un portefeuille à liquider sur 5 large-caps tech américaines (AAPL, MSFT, GOOG, NVDA, META). Les modèles de self-impact donnent un coût pour chaque actif pris isolément. La tentation naturelle serait d'écrire :

$$\text{Coût total} \;\overset{?}{=}\; \sum_{i=1}^{5} c_{\text{self}}^{i}.$$

> [!warning] Cette formule est fausse en pratique
> Quand on liquide les 5 stocks **simultanément**, l'impact de chacun se renforce mutuellement à travers les facteurs communs. Vendre AAPL fait baisser le facteur tech ; cette baisse de facteur se transmet à MSFT, GOOG, NVDA, META — **avant même** qu'on ait commencé à vendre ces stocks. Quand on commence à vendre MSFT, on part d'un prix déjà déprimé par la vente d'AAPL. La somme des coûts individuels **sous-estime** le coût total.

> 💡 **L'image canonique.** Le self-impact répond à la question *"si je suis seul à trader, combien me coûte mon trade ?"*. Le cross-impact répond à *"si je trade plusieurs stocks corrélés en même temps, combien me coûte mon basket ?"*. Sur un basket décorrélé (ex. AAPL + une mining junior canadienne), les deux questions ont la même réponse. Sur un basket très corrélé (5 large-caps tech), les deux divergent fortement.

### B. Décomposition du coût total

La décomposition naturelle du coût total est :

$$\boxed{\;\text{Coût total} = \underbrace{\sum_{i} c_{\text{self}}^{i}}_{\text{self-impact (}\textit{vase clos}\text{)}} + \underbrace{\sum_{i} c_{\text{cross}}^{i}}_{\text{cross-impact (interactions)}}\;}$$

Le self-impact est calculé stock par stock par les modèles de [[02_Market Impact - Modèles et calibration|note ii]] (square-root, Almgren). Le cross-impact $c_{\text{cross}}^{i}$ capture **l'impact additionnel** sur le stock $i$ provoqué par les trades simultanés sur les **autres** stocks $j \neq i$ — via les facteurs communs.

> [!note]- Pourquoi pas de spread cost dans le cross-impact ?
> Le spread cost est un coût d'**exécution** : tu le payes parce que tu traverses le bid-ask sur **ton** ordre. Il n'y a pas de mécanisme par lequel mon ordre sur AAPL me ferait payer le spread de MSFT — ces sont deux exécutions séparées avec leurs spreads propres. Le cross-impact, lui, est un effet de **prix** : mon ordre sur AAPL déforme le prix d'équilibre de MSFT via les facteurs communs. Le coût additionnel sur MSFT vient de cette déformation, pas d'un coût de transaction.
> 
> Mathématiquement : le cross-impact est une déformation de la **trajectoire de prix** des autres actifs, pas un surcoût d'exécution. Le self-impact contient bien une composante spread cost (qu'on traverse en exécutant) ; le cross-impact n'en contient pas.

## II. Algorithme factor-based (Barra)

### A. Modèle factoriel et factor mimicking portfolio

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

### B. Chaîne de transmission

On suppose que mes trades en masse sur une courte durée provoquent une variation des returns proportionnelle au self-impact $\Delta c$ (en bps) :

$$\Delta r = \alpha \cdot \Delta c, \qquad \alpha = 1$$

Le choix $\alpha = 1$ exprime que *l'impact que je provoque sur les returns est exactement le coût en bps que j'identifie comme self-impact* — on travaille dans les mêmes unités. À partir de là, la transmission se déroule en deux étapes :

> [!warning] Chaîne de transmission
> **Étape 1 — Returns → facteurs.** Mes trades déforment les returns, qui déforment à leur tour les facteurs :
> $$\Delta f = H \, \Delta c$$
> 
> **Étape 2 — Facteurs → returns (sur les autres stocks).** Les facteurs perturbés se retransmettent à tous les stocks via leurs expositions :
> $$\Delta r = X \, \Delta f = X H \, \Delta c$$

L'opérateur $XH$ est une matrice $n \times n$ qui exprime **comment l'impact se propage entre stocks** via les facteurs. C'est l'objet central du cross-impact factor-based.

> 💡 **Lecture géométrique.** $H$ projette les returns sur l'espace des facteurs (de dimension $d \ll n$). $X$ relifte des facteurs vers les returns. Composer les deux donne $XH$ : un opérateur de **rang faible** ($\leq d$) qui ne propage que la composante "factorielle" de mon impact, en filtrant la composante idiosyncratique. C'est cohérent avec l'intuition : un trade sur AAPL n'impacte pas un mining junior canadien (pas de facteur commun), il impacte les autres tech (facteurs communs).

### C. Cross-impact stock par stock

Pour calculer le cross-impact subi par le stock $i$, on isole les contributions des **autres** stocks. Concrètement :

> [!warning] Cross-impact du stock $i$
> $$\Delta c^{i}_{\text{cross}} = X_{i,:} \cdot H \cdot \Delta c^{(-i)}$$
> 
> où :
> - $X_{i,:}$ est la **ligne $i$** de la matrice d'expositions (vecteur ligne de dimension $d$),
> - $H$ est le factor mimicking portfolio ($d \times n$),
> - $\Delta c^{(-i)} \in \mathbb{R}^n$ est le **vecteur complet** des self-impacts $(\Delta c_1, \ldots, \Delta c_n)$ avec **la composante $i$ remplacée par 0**.

L'idée : on prend tous les self-impacts du portefeuille, on annule celui du stock $i$ lui-même (parce qu'on ne veut pas compter son self-impact dans son cross-impact — ce serait du double comptage), on projette le reste sur l'espace des facteurs ($H \Delta c^{(-i)}$), et on relifte sur le stock $i$ via son exposition $X_{i,:}$.

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

### D. Lecture interprétative via $\Delta f$

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

### E. Choix du modèle factoriel : cascade waterfall

MSCI Barra ne fournit pas un modèle factoriel unique mondial — il publie une **suite de modèles régionaux et globaux**, chacun calibré sur un univers donné avec ses propres facteurs et ses propres expositions. Pour un stock donné, plusieurs modèles peuvent s'appliquer (un stock français est dans le modèle Europe **et** dans le modèle Global), mais ils ne donnent **pas les mêmes loadings**.

Le choix : utiliser **le modèle le plus spécifique** possible pour chaque stock — un modèle régional étroit captera mieux les facteurs locaux qu'un modèle mondial qui les dilue.

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

### F. Considérations pratiques

**Pas de spread dans le cross-impact.** Comme évoqué en I.B (callout collapsible), le spread cost n'apparaît que dans le self-impact. Le cross-impact est purement un effet de prix, pas un coût d'exécution. Conséquence : la formule $\Delta c^i_{\text{cross}} = X_{i,:} H \Delta c^{(-i)}$ utilise des $\Delta c$ **hors spread** — concrètement, on retire la composante spread cost des self-impacts avant de les injecter dans le cross-impact.

**Magnitude relative self vs cross.** En pratique, sur un basket large-cap diversifié, le cross-impact est **du même ordre de grandeur** que le self-impact, voire supérieur sur un basket très concentré sectoriellement. Quelques règles de pouce empiriques :

> [!example] Ordres de grandeur typiques
> | Type de basket | Ratio cross / self |
> |---|:---:|
> | Basket très diversifié (50 stocks, secteurs variés) | 10–30% |
> | Basket sectoriel (10 stocks tech US) | 50–100% |
> | Basket factor-pur (long momentum, short value) | 80–150% |
> 
> Le cas extrême est le basket *factor-pur* : par construction, tous les stocks chargent les mêmes facteurs avec le même signe, donc l'effet de transmission est maximal. À l'inverse, un basket bien diversifié bénéficie de l'annulation partielle des effets factoriels (les exposures des longs et shorts se compensent partiellement dans $H \Delta c$).

**Non-additivité et ordre d'exécution.** Le modèle ci-dessus suppose que les $n$ stocks sont tradés **simultanément** sur la même fenêtre. En pratique, les exécutions s'étalent dans le temps et ne sont pas synchrones. Deux régimes :

- **Trades simultanés** : l'algorithme tel quel s'applique. Le cross-impact est instantané.
- **Trades séquentiels** : si on trade AAPL puis MSFT, le cross-impact d'AAPL sur MSFT a déjà été partiellement digéré par le marché au moment où on commence MSFT. Il faut ajouter une **décroissance temporelle** (à la propagator), ce qui sort du cadre simple présenté ici.

Pour un usage pre-trade (estimation de coût avant exécution), l'hypothèse simultanée est raisonnable. Pour un usage post-trade fin (décomposition d'un coût observé), il faut un modèle de propagator.

### G. Limites de l'approche factor-based

**Linéarité.** Le modèle $\Delta r = XH \Delta c$ est **linéaire** en $\Delta c$. Cela cohabite mal avec la non-linéarité (concavité) du self-impact lui-même : on a $c_{\text{self}}(Q) \propto \sqrt{Q}$ ou $\ln(Q)$ selon le modèle, donc une grande quantité ne produit pas une grande variation linéaire de prix. Le cross-impact présenté **suppose** que les $\Delta c$ d'entrée sont déjà calculés (par les modèles non-linéaires de note ii), puis les transmet linéairement via les facteurs.

Cette hybridation (self non-linéaire + cross linéaire) est la pratique standard. Des modèles plus rigoureux (Tomas et al. 2022) dérivent un cross-impact non-linéaire à partir de premiers principes, mais sont plus lourds à calibrer.

**Stationnarité de $XH$.** L'opérateur $XH$ est calibré sur un historique de returns/factor returns. Il est **supposé stationnaire** — c'est-à-dire que la structure de propagation factorielle est la même hier, aujourd'hui, et demain. C'est raisonnable en régime normal de marché, mais peut casser en période de **stress** (crise 2008, COVID mars 2020, où les corrélations explosent et les facteurs habituels se réorganisent).

**Au-delà du factor model : approche propagator.** L'approche factor-based présentée ici est l'une des deux grandes familles de cross-impact. L'alternative est l'approche **propagator** :
- Mastromatteo, Benzaquen, Eisler, Bouchaud (2017) — *Trading lightly: Cross-impact and optimal portfolio execution*
- Benzaquen, Mastromatteo, Eisler, Bouchaud (2017) — *Dissecting cross-impact on stock markets: An empirical analysis*

Ces modèles renoncent au passage explicite par les facteurs et estiment directement la matrice de cross-impact $\Lambda$ (telle que $\Delta P = \Lambda \cdot Q$) à partir de tick data, avec un noyau de décroissance temporelle. Plus flexible, plus puissant sur des univers où les facteurs ne capturent pas tout — mais beaucoup plus lourd à calibrer (besoin de tick data sur tous les stocks de l'univers).

## III. Algorithme de scheduling : POV adaptatif à horizon commun

> Cette section présente un **algorithme de scheduling** — pas d'estimation de cross-impact. Le problème : étant donné un basket de $n$ titres à liquider sur plusieurs jours, comment répartir l'exécution dans le temps de manière à ce que (1) tous les titres se terminent simultanément et (2) le POV par titre reste maîtrisé.

### A. Le problème : exécuter un basket en $T$ jours

Setup : $n$ titres à liquider, position initiale $Q_i^0$, volume moyen quotidien $V_i$. On cherche un **schedule** $Q_i^t$ qui :

- **Démarre** à $Q_i^0$ pour tous les titres,
- **Se termine** simultanément à $Q_i^T = 0$ pour tous les titres (horizon commun $T$),
- Maintient une **fraction de volume** $\text{pov}_i$ raisonnable titre par titre (typiquement < 5% pour limiter le market impact).

Deux méthodes naïves apparaissent naturellement, et chacune **échoue** à satisfaire ces trois contraintes simultanément.

### B. Deux méthodes naïves et leur incompatibilité

**Méthode (i) — exécution linéaire.** On impose une décroissance linéaire de la position avec un horizon $T$ commun :

$$Q_i^t = Q_i^0 \left(1 - \frac{t}{T}\right), \qquad t \in [0, T]$$

Le quantité exécutée par jour est $Q_i^0 / T$, donc la **participation rate** sur le titre $i$ vaut :

$$\text{pov}_i = \frac{Q_i^0 / T}{V_i} = \frac{Q_i^0}{V_i \cdot T}$$

L'horizon est commun, mais la participation rate **n'est pas contrôlée** : elle dépend de la liquidité de chaque titre. Sur un titre très illiquide ($V_i$ petit), $\text{pov}_i$ peut exploser.

**Méthode (ii) — POV uniforme.** On impose au contraire un POV constant et identique pour tous les titres (par exemple 5%) :

$$\text{pov}_i = \text{prate} = 5\%$$

La quantité exécutée par jour vaut alors $V_i \cdot \text{prate}$, et après $t$ jours :

$$Q_i^t = Q_i^0 - t \cdot V_i \cdot \text{prate}$$

Le POV est maîtrisé, mais l'horizon **dépend du titre** :

$$T_i = \frac{Q_i^0}{V_i \cdot \text{prate}}$$

Chaque titre se termine à un $T_i$ différent — pas un schedule de basket.

> [!warning] Le piège classique : l'algo "vendre les liquides en premier"
> Une variante intuitive de (ii) consiste à liquider les titres les plus liquides d'abord, en plein POV, et à passer aux suivants quand les premiers sont terminés. Ça marche sur le papier — mais en fin de schedule, **il ne reste que les titres les plus illiquides**. Le POV moyen du basket explose en queue d'exécution, et le market impact sur ces derniers titres dégrade massivement le coût total. C'est un anti-pattern à éviter : **les titres illiquides doivent être tradés tôt**, en parallèle des liquides, pas concentrés en fin de schedule.

### C. Fusion : POV adaptatif à $T$ commun

L'idée centrale : imposer **simultanément** la décroissance linéaire (i) et le contrôle du POV (ii), en laissant le POV varier par titre. On définit le ratio :

> [!warning] Ratio de jours-de-volume
> $$P_v^i = \frac{Q_i^0}{V_i}$$
> 
> $P_v^i$ s'interprète comme **le nombre de jours de volume** que représente la position sur le titre $i$ à pleine participation. Si $P_v^i = 0{,}3$, ma position vaut 0.3 jour de volume du titre — facile à liquider. Si $P_v^i = 20$, elle représente 20 jours de volume — illiquide.

En égalisant les deux expressions du schedule (forme (i) avec horizon $T$ commun, forme (ii) avec POV adaptatif), on trouve :

> [!warning] Formule du POV adaptatif
> $$\text{pov}_i = \frac{P_v^i}{T}$$
> 
> avec $P_v^i = Q_i^0 / V_i$ le ratio de jours-de-volume.

**Lecture.** Plus un titre est illiquide ($P_v^i$ grand) → plus son POV est élevé. Plus il est liquide ($P_v^i$ petit) → plus son POV est faible. C'est exactement l'inverse de l'intuition naïve "vendre les liquides en premier".

> 💡 **Propriété clé.** Avec ce schedule, **tous les titres terminent en $T$ jours simultanément**, et il n'y a plus de concentration d'illiquides en fin d'exécution. Les illiquides sont tradés à POV élevé **dès le jour 1**, en parallèle des liquides à POV faible. Le market impact se répartit uniformément sur l'horizon plutôt que d'exploser en queue.

### D. Choix de $T$ : le 80e percentile

Reste à choisir l'horizon $T$. Pour un basket de $n$ titres, on a une **distribution** des $P_v^i$. Le choix doit arbitrer entre deux extrêmes :

- $T$ petit : tous les POV sont élevés, market impact lourd partout.
- $T$ grand : POV faibles, mais on porte le risque de marché plus longtemps (drift, exposition au beta).

Une règle pragmatique répandue : fixer $T$ tel que **80% des titres respectent un POV cible** (typiquement 5%), en acceptant que les 20% les plus illiquides dépassent ce seuil :

> [!warning] Choix de l'horizon
> $$T = \frac{P_v^{(80)}}{\text{prate}}, \qquad \text{prate} = 5\%$$
> 
> où $P_v^{(80)}$ est le 80e percentile de la distribution des $P_v^i$ sur le basket.

> [!note]- Pourquoi 80 et pas 100 ?
> Si on prend le **max** au lieu du 80e percentile, on garantit que **tous** les titres respectent 5% POV, mais $T$ devient potentiellement énorme. Sur un basket avec un outlier à $P_v = 50$, on aurait $T = 1000$ jours — délirant en pratique. Le 80e percentile est un **compromis empirique** (validé par backtest dans les desks d'exécution équités) : on accepte de "sacrifier" les 20% les plus illiquides — qui paieront un peu plus de market impact — en échange d'un horizon raisonnable pour le reste du basket.
> 
> Le percentile exact est un paramètre de tuning : 70, 80, 90 selon le profil de risque et les contraintes de l'execution desk.

### E. Implémentation et exemple chiffré

**En pratique**, le ratio $P_v^i$ se calcule en USD à partir de quantités déjà disponibles dans les systèmes :

$$P_v^i = \frac{|\text{bsDelta}_i|}{\text{mdv21}_i \cdot \text{priceUSD}_i} = \frac{|Q_i \cdot P_i|}{V_i \cdot P_i} = \frac{Q_i}{V_i}$$

où `bsDelta` est la valeur dollar de la position (signée long/short, on prend la valeur absolue), `mdv21` est le median daily volume sur 21 jours (en shares), et `priceUSD` est le prix en USD. Le prix se simplifie au numérateur et au dénominateur : $P_v^i$ est une grandeur **adimensionnelle**, exprimée en jours.

**Algorithme :**

```
Inputs:  n titres, vecteur des positions Q_0 ∈ ℝⁿ, volumes V ∈ ℝⁿ
         POV cible prate (typiquement 0.05)
         Percentile cible q (typiquement 80)

1. P_v[i] ← Q_0[i] / V[i]                 # vecteur des jours-de-volume
2. m_p   ← percentile(P_v, q)             # 80e percentile
3. T     ← m_p / prate                    # horizon commun
4. pov[i] ← P_v[i] / T                    # POV par titre

Return T, pov
```

> [!example] Exemple chiffré : basket de 10 titres
> On prend une distribution synthétique des $P_v^i$ :
> 
> $$P_v = [0{,}3,\; 0{,}5,\; 1{,}1,\; 1{,}2,\; 2{,}4,\; 3{,}0,\; 4{,}5,\; 7{,}5,\; 12{,}0,\; 20{,}0]$$
> 
> Le 80e percentile vaut $P_v^{(80)} = 7{,}5$ jours, donc avec $\text{prate} = 5\%$ :
> 
> $$T = \frac{7{,}5}{0{,}05} = 150 \text{ jours}$$
> 
> Les POV résultants par titre :
> 
> | Titre | $P_v^i$ | POV ($= P_v^i / 150$) | Statut |
> |:---:|:---:|:---:|:---:|
> | A | 0.3  | 0.20%  | très liquide |
> | B | 0.5  | 0.33%  | liquide |
> | C | 1.1  | 0.73%  | liquide |
> | D | 1.2  | 0.80%  | liquide |
> | E | 2.4  | 1.60%  | normal |
> | F | 3.0  | 2.00%  | normal |
> | G | 4.5  | 3.00%  | normal |
> | H | 7.5  | **5.00%** | au seuil (80e percentile) |
> | I | 12.0 | **8.00%** | illiquide (au-dessus du seuil) |
> | J | 20.0 | **13.33%** | très illiquide (au-dessus du seuil) |
> 
> Les 8 premiers titres ont un POV ≤ 5%. Les 2 derniers (titres I et J, les 20% les plus illiquides du basket) dépassent le seuil — c'est le compromis assumé du 80e percentile.

**Trajectoires d'exécution.** En traçant $Q_i^t / V_i = P_v^i (1 - t/T)$ — c'est-à-dire la position résiduelle exprimée en jours-de-volume — on visualise la propriété fondamentale de l'algo :

![[scheduling_trajectories.png|561]]
**Figure 1.** Toutes les trajectoires partent de leur $P_v^i$ propre (à gauche) et convergent vers 0 à $t = T = 150$ jours. La pente de chaque courbe vaut exactement $-P_v^i / T = -\text{pov}_i$. Le titre J (jaune, $P_v=20$) a la pente la plus raide — il consomme 20 jours de volume en 150 jours réels, donc POV = 13.33%. Le titre A (violet, $P_v=0.3$) a la pente la plus faible — POV = 0.20%. **Tous les titres se terminent en même temps**, sans concentration d'illiquides en fin de schedule.

**Distribution des POV.** Le barplot suivant rend visible le compromis du 80e percentile :

![[scheduling_povs.png|590]]
**Figure 2.** 8 titres respectent le POV cible de 5% (en bleu), 2 titres le dépassent (en rouge — ce sont les 20% les plus illiquides du basket). Le titre H se trouve pile au seuil par construction (c'est le 80e percentile lui-même). Le compromis est lisible directement : on accepte que les 2 titres illiquides paient plus de market impact, en échange d'un horizon $T$ raisonnable pour le reste du basket.

### F. Limites

**Heuristique, pas optimisation.** Le schedule est construit à partir d'une règle simple ($T$ via le 80e percentile, POV linéaire) — il ne minimise pas explicitement une fonctionnelle de coût (self-impact + risque de marché). C'est un compromis pragmatique calibré empiriquement, pas un optimum au sens mathématique.

**Indépendance vis-à-vis de la section II.** Cet algorithme et l'algo factor-based de la section II opèrent à des **horizons différents** et répondent à des questions différentes :

- Le **factor-based (II)** vit sur une fenêtre courte (intraday, 1 jour max) : *étant donnés des trades simultanés sur un basket, quel est le cross-impact entre titres via les facteurs ?* Il ne dit **rien** sur le nombre de jours nécessaires à l'exécution.
- Le **scheduling (III)** vit sur plusieurs jours : *comment dimensionner l'horizon $T$ et le POV par titre pour étaler proprement le basket ?* Il ne dit **rien** sur le cross-impact intra-fenêtre.

Les deux ne se composent pas naturellement. Articuler les deux dans un cadre unifié (cross-impact-aware sur un horizon multi-jours) est un sujet à part.

---

## Références

- **Tomas, Mastromatteo, Benzaquen** (2022). *How to build a cross-impact model from first principles*. Quantitative Finance.
- **Mastromatteo, Benzaquen, Eisler, Bouchaud** (2017). *Trading lightly: Cross-impact and optimal portfolio execution*.
- **Benzaquen, Mastromatteo, Eisler, Bouchaud** (2017). *Dissecting cross-impact on stock markets: An empirical analysis*. Journal of Statistical Mechanics.
- **MSCI Barra** — documentation officielle des modèles factoriels (référence par code modèle).
- **Connor, Goldberg, Korajczyk** (2010). *Portfolio Risk Analysis*. Princeton University Press — référence standard sur les modèles factoriels.
