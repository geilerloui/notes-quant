---
title: a-Fondamentaux TCA
order: 1
---
# TCA — Fondamentaux : coûts et mesures

> La **Transaction Cost Analysis (TCA)** est l'analyse des coûts payés par un investisseur pour exécuter ses ordres sur le marché. C'est un sujet central pour les gérants institutionnels parce que les coûts d'exécution rongent directement la performance : un alpha de 50 bps peut être complètement effacé par 30 bps de coûts mal maîtrisés. Cette note pose les bases : comment décomposer un coût total, comment le mesurer en bps, quels benchmarks d'exécution utiliser, et la tension fondamentale entre trader vite (impact élevé) et trader lentement (risque de marché). La modélisation du market impact lui-même — le coût caché le plus important — est traitée dans [[(ii) Market Impact - Modèles et calibration]].


# Faut rajouter un truc 

tu sais toute l'histoire de market order, limit order, order book les trucs de base de la liquidité de la microstructure quoi 

## I. Structure des coûts de transaction

### A. Décomposition générale

Le **coût total de transaction** (Transaction Cost, TC) se décompose en trois grands blocs :

$$\text{TC} = \text{execution costs} + \text{delay costs} + \text{commissions}.$$

Cette vue "haut niveau" cache une réalité plus fine : à l'intérieur de chaque bloc, certains coûts sont **directement observables** et d'autres sont **cachés** — ce sont précisément ces derniers qui dominent en pratique et qui justifient l'existence de la TCA comme discipline.

### B. Les 9 composantes (taxonomie de Kissell)

Kissell propose une décomposition en **9 sources de coût**, ordonnées du plus visible au plus caché :

> [!warning] Les 9 stages du coût de transaction
> | | Composante | Type | Visibilité |
> |---|---|:---:|:---:|
> | 1 | **Commissions** | fixed | visible |
> | 2 | **Fees** | fixed | visible |
> | 3 | **Taxes** | variable | visible |
> | 4 | **Spreads** | variable | visible |
> | 5 | **Investment delays** (delay cost) | variable | caché |
> | 6 | **Price appreciation** | variable | caché |
> | 7 | **Market impact** | variable | caché |
> | 8 | **Timing risk** | variable | caché |
> | 9 | **Opportunity cost** | variable | caché |

> 💡 **Pourquoi cette décomposition compte.** Les coûts visibles (commissions, fees, taxes, spread affiché) sont faciles à mesurer mais représentent **la plus petite partie** du coût total. Les coûts cachés (à partir de l'investment delay) sont **variables, opaques, et corrélés entre eux**. C'est l'inverse de l'intuition naïve qui consisterait à se concentrer sur les commissions.

> [!note]- Pourquoi les coûts cachés sont corrélés
> Les composantes cachées s'interconnectent : un délai d'investissement long laisse le prix "appréciation" (price appreciation) jouer contre vous, mais réduit le market impact (vous pouvez trader plus lentement). À l'inverse, exécuter rapidement réduit le timing risk mais augmente le market impact. **C'est cette interdépendance qui rend la TCA difficile** — on ne peut pas optimiser une composante sans en dégrader une autre. C'est précisément l'objet du *trader's dilemma* (cf. III.C).

### C. Direct vs Indirect

Une autre manière de regrouper, plus opérationnelle :

- **Coût direct** : ce qu'on paye explicitement → fees, commissions, taxes. Facturé, observable sur les relevés.
- **Coût indirect** (alias *trading cost*) : ce qu'on paye implicitement à travers le prix d'exécution → spread cost + market impact + bruit de marché.

Mathématiquement, sur un ordre exécuté :

$$\text{coût indirect} = \underbrace{\text{spread cost}}_{\text{traverser bid-ask}} + \underbrace{\text{realized impact}}_{\text{déplacer le prix}} + \underbrace{\varepsilon}_{\text{bruit marché + volatilité}}$$

avec $\text{realized impact} = \text{temporary} + \text{permanent}$ (cf. [[(ii) Market Impact - Modèles et calibration|note ii]] pour la décomposition complète).

> 💡 **Lecture pratique du bruit $\varepsilon$.** Pendant ton exécution, le marché bouge naturellement — ni à cause de toi, ni de manière prévisible. Si tu achètes pendant que le marché monte (drift favorable), $\varepsilon < 0$ ; si tu achètes pendant qu'il descend, $\varepsilon > 0$. Sur un grand nombre d'ordres, ces effets se compensent, mais sur un ordre donné ils peuvent dominer le coût observé. C'est pour ça qu'on parle de *realized impact ≈ market impact théorique + ε* : la mesure brute mélange ton effet et celui du marché.

## II. Mesure en basis points (bps)

### A. Convention bps

Les financiers expriment les coûts en **points de base** (basis points, bps) plutôt qu'en pourcentage, parce que les variations sont typiquement petites :

> [!warning] Conversions à connaître par cœur
> $$1\% = 100\,\text{bps} \qquad 0.1\% = 10\,\text{bps} \qquad 0.01\% = 1\,\text{bps}$$

> [!example] Lecture sur le marché obligataire
> Si les taux passent de **1.50% à 1.51%**, on dit que les taux ont monté de **1 bp**, pas de "0.01%". C'est plus lisible et évite les confusions avec les variations relatives.

### B. Trading cost en bps

Pour un ordre exécuté à un prix moyen $\bar{P}$ partant d'un prix initial $P_0$ :

$$\text{tcost (bps)} = \frac{\bar{P} - P_0}{P_0} \cdot 10^4.$$

Le facteur $10^4$ vient de la double multiplication par 100 : une fois pour passer du ratio brut au pourcentage ($\times 100$), une fois pour passer du pourcentage aux bps ($\times 100$).

> [!example] Achat de 1M$ d'actions
> Prix initial $P_0 = 30.04\,\$$, prix moyen exécuté $\bar{P} = 30.15\,\$$.
> 
> $$\text{tcost} = \frac{30.15 - 30.04}{30.04} \cdot 10^4 = \frac{0.11}{30.04} \cdot 10^4 \approx 32\,\text{bps}.$$
> 
> Lecture : *l'exécution t'a coûté 0.32% de plus que le prix que tu espérais payer au départ*. Sur 1M$ d'achat, ça représente
> 
> $$1\,000\,000 \cdot 0.0032 = 3\,200\,\$ \;\text{de surcoût.}$$

### C. Décomposition tcost = market impact + mouvement naturel

Le coût observé ne vient pas que de toi : pendant ton exécution, le prix de référence du marché bouge aussi. On décompose :

$$\text{tcost} = \underbrace{\text{Market Impact (MI)}}_{\text{ton effet}} + \underbrace{\text{Mouvement Naturel (MN)}}_{\text{drift de marché}}$$

Algébriquement, en notant $S_T^{\text{rétro}}$ le prix qu'aurait eu l'actif à l'instant $T$ **sans ton ordre** :

$$\frac{\bar{S} - S_0}{S_0} = \underbrace{\frac{\bar{S} - S_T^{\text{rétro}}}{S_0}}_{\text{MI}} + \underbrace{\frac{S_T^{\text{rétro}} - S_0}{S_0}}_{\text{MN}}$$

> [!example] Décomposition d'un tcost de 32 bps
> Si le tcost mesuré est de **32 bps**, on peut typiquement le décomposer en :
> - **16 bps** dus au mouvement naturel du marché (le prix serait monté même sans toi).
> - **16 bps** dus à ton market impact (le surcoût que tu as provoqué en tradant en masse).
> 
> En entretien : le piège classique est de dire que tout le tcost est du MI. C'est faux — il faut savoir séparer ta contribution du contexte de marché.

> 💡 **Le problème fondamental de la mesure.** $S_T^{\text{rétro}}$ n'est **pas observable** : on ne peut pas voir le prix qu'aurait eu l'actif sans notre ordre. C'est ce qui motive toute la modélisation du market impact dans [[(ii) Market Impact - Modèles et calibration|note ii]] — on ne peut pas mesurer le MI directement, on est obligé de le **modéliser**.

## III. Benchmarks d'exécution

Pour évaluer la qualité d'une exécution, on a besoin d'un point de référence — un *benchmark*. Trois benchmarks dominent en pratique : **VWAP** (le plus simple), **arrival price**, et **Implementation Shortfall** (le plus complet).

### A. VWAP (Volume-Weighted Average Price)

Le VWAP d'une période est le prix moyen pondéré par les volumes échangés :

> [!warning] VWAP
> $$\text{VWAP} = \frac{\sum_i Q_i \cdot P_i}{\sum_i Q_i}$$
> 
> où $Q_i$ et $P_i$ sont la quantité et le prix de chaque trade $i$ exécuté pendant la période. C'est le *prix moyen pondéré* qu'on aurait obtenu en achetant proportionnellement au volume.

**Usage.** Le VWAP est le benchmark le plus utilisé en algotrading institutionnel. Battre le VWAP signifie *"j'ai exécuté à un prix meilleur que la moyenne pondérée du marché sur la période"*. Les algos VWAP sont conçus pour répliquer cette moyenne en étalant l'ordre sur la journée selon le profil de volume historique.

> 💡 **Limite.** Le VWAP est un benchmark *passif* : il dit si on a bien suivi le flux du marché, pas si on a bien exécuté par rapport à la décision d'investissement. Si le prix a fortement monté après ta décision mais avant que tu commences à trader, battre le VWAP ne te sauve pas — tu as déjà payé un coût caché (l'investment delay) que le VWAP ne capture pas.

### B. Implementation Shortfall (IS)

L'**Implementation Shortfall** mesure l'écart entre le rendement *idéal* (sur papier) qu'aurait donné une exécution instantanée à $P_0$, et le rendement *réel* obtenu en tradant.

> [!warning] Implementation Shortfall
> $$\text{IS} = \text{Paper return} - \text{Actual return}$$
> 
> avec :
> - **Paper return** = rendement qu'on aurait eu si on avait pu trader instantanément à $P_0$
> - **Actual return** = rendement réellement réalisé en tenant compte du prix moyen exécuté $\bar{P}$ et des fees
> 
> L'IS est le *slippage total* — le manque à gagner global lié à l'écart entre l'idéal et la réalité d'implémentation.

> [!example] Achat de 5000 actions, calcul d'IS
> Données : $P_d = 10\,\$$ (prix au moment de la décision), $P_n = 11\,\$$ (prix de fin de période), $\bar{P} = 10.50\,\$$ (prix moyen exécuté), $\text{Fees} = 100\,\$$.
> 
> **Paper return** (exécution instantanée à $P_d$) :
> $$S \cdot (P_n - P_d) = 5000 \cdot (11 - 10) = 5\,000\,\$.$$
> 
> **Actual return** (rendement réel après exécution moyenne) :
> $$S \cdot (P_n - \bar{P}) - \text{Fees} = 5000 \cdot (11 - 10.50) - 100 = 2\,400\,\$.$$
> 
> **IS = slippage total** :
> $$\text{IS} = 5\,000 - 2\,400 = 2\,600\,\$.$$

> 💡 **Sources du shortfall.** L'IS capture **toutes** les sources de coût en un nombre :
> - tu attends trop longtemps avant de trader → le prix monte (delay cost / price appreciation)
> - tu impactes le prix en tradant en masse → market impact
> - tu payes des commissions et le spread
> - tu rates une partie de l'ordre → opportunity cost
> 
> C'est le benchmark le plus complet, mais aussi le plus difficile à décomposer. C'est pour ça qu'on le complète avec des modèles de market impact qui isolent la composante MI dans le total IS.

### C. Trader's dilemma

Pour un ordre donné, le trader fait face à un **arbitrage fondamental** entre deux risques opposés :

> [!warning] Trader's dilemma
> | Stratégie | Effet positif | Effet négatif |
> |---|---|---|
> | **Trader vite (agressif)** | Réduit le timing risk (moins exposé aux mouvements de marché) | Augmente le market impact (on consomme la profondeur du carnet) |
> | **Trader lentement (passif)** | Réduit le market impact (on étale dans le temps) | Augmente le timing risk (le prix peut bouger contre nous) |

![[Pasted image 20260504145303.png]]

**Figure 1.** En abscisse, la durée d'exécution (order interval). En ordonnée, le coût en bps. La courbe **market impact** décroît avec la durée (tradar plus lentement réduit l'impact). La courbe **timing risk** croît avec la durée (plus on étale, plus on s'expose à la volatilité du marché). Le coût total est la somme — il atteint un minimum à une durée optimale, qui est l'objectif des algos d'**optimal execution** (Almgren-Chriss, cf. [[(ii) Market Impact - Modèles et calibration|note ii]]).

> 💡 **Le pivot conceptuel.** Le trader's dilemma est *la* tension qui structure tout l'algotrading. L'optimisation d'exécution (Almgren-Chriss) consiste à trouver le point optimal sur cette courbe en U, pour un profil d'aversion au risque donné. Plus le trader est averse au risque, plus il préférera trader vite (privilégier la certitude au prix d'un MI plus élevé) ; moins il l'est, plus il étalera.

## IV. Best execution

### A. Le triangle Price / Time / Size

La **best execution** est un concept réglementaire et opérationnel : c'est l'obligation (et la pratique) d'exécuter les ordres clients dans les meilleures conditions possibles, en arbitrant entre trois dimensions :

- **Price** : obtenir le meilleur prix possible.
- **Time** : exécuter dans un délai raisonnable.
- **Size** : exécuter le volume demandé en entier.

> 💡 **Pourquoi un triangle.** On ne peut pas optimiser les trois simultanément. Vouloir le meilleur prix peut imposer d'attendre (sacrifie Time) ou de fragmenter l'ordre (sacrifie Size). Vouloir une exécution rapide peut forcer à payer plus cher (sacrifie Price). La best execution est l'art de faire le bon arbitrage selon le contexte.

### B. Transaction Cost Management process

La best execution n'est pas un acte ponctuel mais un **processus**, articulé en plusieurs étapes du cycle d'investissement :

1. **Pre-trade analysis** : avant l'exécution, estimer le coût attendu (cost estimation, *ex-ante*) et choisir la stratégie (algo VWAP, IS, POV…).
2. **Trade execution** : exécuter selon la stratégie choisie, en surveillant les conditions de marché.
3. **Post-trade analysis** : après l'exécution, mesurer le coût réalisé (cost measurement, *ex-post*) et le décomposer.
4. **Feedback loop** : utiliser les résultats post-trade pour calibrer les modèles ex-ante et améliorer les exécutions futures.

> [!warning] Ex-ante vs ex-post
> - **Ex-ante** = avant l'exécution, on **estime** le coût attendu à partir d'un modèle (typiquement market impact + spread cost + timing risk). Sert à choisir la stratégie d'exécution.
> - **Ex-post** = après l'exécution, on **observe** le coût réalisé et on le décompose. Sert à évaluer la qualité de l'exécution et à recalibrer les modèles.

## V. Lectures du market impact : absolu vs relatif

Le market impact peut être lu sous deux angles très différents selon qu'on est trader ou risk manager.

### A. MI absolu : le point de vue trader

Du point de vue du trader exécutant un ordre, la question est :

> *"Combien va me coûter ce trade en bps ? Est-ce que ce coût est inférieur à mon alpha attendu ?"*

C'est une lecture **absolue** : on regarde le coût d'un ordre en valeur, pour décider si le trade est rentable compte tenu de l'alpha estimé. Si tu attends 50 bps de gain et que le MI estimé est de 15 bps, tu trades ; si le MI est de 60 bps, tu passes.

### B. MI relatif : le point de vue risk

Du point de vue du risk manager surveillant un portefeuille, la question est :

> *"Quels stocks sont les plus risqués à liquider ? Sur lesquels la banque porte-t-elle un delta cash important par rapport à leur liquidité ?"*

C'est une lecture **relative** : on compare les stocks entre eux pour identifier ceux qui posent le plus de risque de liquidation. Un stock peu liquide sur lequel on a un gros notional est dangereux, même si individuellement chaque trade reste exécutable.

> 💡 **Exemple concret.** Imaginons deux positions de 10M$ chacune, l'une sur Apple (AAPL, ADV ≈ 60M actions/jour) et l'autre sur un mid-cap européen (ADV ≈ 200K actions/jour). Le notional est identique, mais le ratio position/ADV est radicalement différent — la deuxième position pose un risque de liquidation bien plus important. La lecture *relative* du MI est exactement ce qui permet de quantifier ce risque et de fixer des limites par stock.

### C. Cas applicatifs

La lecture relative motive plusieurs cas applicatifs concrets, traités dans [[(ii) Market Impact - Modèles et calibration|note ii]] :

- **Max delta par stock** : pour un budget de coût $c^*$ donné, déterminer la taille maximale tradable. Ça revient à *inverser* le modèle de MI.
- **Stress test vol/volume** : étudier comment le coût évolue quand la volatilité ou le volume changent (en cas de stress de marché).
- **Reporting de portefeuille** : agréger des métriques par actif (gross delta, %ADV, distributions par bucket) pour identifier les concentrations de risque liquidité.
- **Cross-impact** (cf. [[(iii) Cross-impact|note iii]]) : sur un basket d'actifs corrélés, le coût de trader plusieurs noms ensemble n'est pas la somme des coûts individuels — il faut tenir compte des corrélations.

---

## Résumé — fil logique

| Section | Contenu |
|---|---|
| **I** | Coût total = direct (commissions, fees, taxes, spread visible) + indirect (spread cost, market impact, bruit). 9 stages de Kissell : les coûts cachés dominent. |
| **II** | Conversion bps : 1% = 100 bps. Tcost = $(\bar{P} - P_0)/P_0 \cdot 10^4$. Décomposition tcost = MI + mouvement naturel ; le MI n'est pas directement observable → modélisation nécessaire. |
| **III** | Benchmarks : VWAP (passif, suit le marché), IS (slippage total, capture toutes les sources), trader's dilemma (MI vs timing risk → courbe en U avec optimum). |
| **IV** | Best execution = arbitrage Price / Time / Size. Process : pre-trade (ex-ante) → execution → post-trade (ex-post) → feedback. |
| **V** | MI absolu (trader : coût d'un trade vs alpha) vs MI relatif (risk : comparer les stocks pour identifier les concentrations). Motivation des cas applicatifs (max delta, stress tests, reporting). |
