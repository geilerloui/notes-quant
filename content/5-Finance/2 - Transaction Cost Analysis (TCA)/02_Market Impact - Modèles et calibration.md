---
title: b-Market Impact - Modèles et calibration
order: 2
---
# Market Impact — Modèles et calibration

> Cette note suit la chronologie historique de la modélisation du market impact (MI). On part du **problème de mesure** : on ne peut pas observer directement le coût qu'on a soi-même provoqué — il faut donc modéliser. On dérive d'abord la **square-root law** (loi en racine carrée), longtemps standard de l'industrie, puis on présente sa **critique empirique** (Zarinelli et al. 2015) qui motive des décompositions plus fines (impact en fonction de la participation rate **et** de la duration). On formalise ensuite le **modèle d'Almgren et al. (2005)** qui sépare proprement les composantes temporary et permanent, et on détaille sa **calibration empirique**. La note se termine sur les **considérations data** et **cas applicatifs** (max delta, stress tests, reporting), et annonce les limites qui motivent le **cross-impact** (cf. [[03_Cross-impact|note iii]]).

## I. Le problème de mesure : pourquoi modéliser ?

### A. La définition naturelle est inobservable

La définition la plus naturelle du market impact serait :

$$\text{MI} = (\text{prix observé avec ton ordre}) - (\text{prix qu'on aurait observé sans ton ordre}).$$

Le problème : le second terme **n'existe pas** dans le monde réel. On ne peut pas exécuter le même ordre dans deux univers parallèles, l'un avec et l'autre sans le trade. C'est une analogie avec le **principe d'incertitude de Heisenberg** : observer le système modifie le système. Mesurer le MI nécessiterait de connaître un contrefactuel qui, par définition, ne s'est pas produit.

> 💡 **Conséquence fondamentale.** Le market impact n'est pas une grandeur directement mesurable — c'est une grandeur **modélisée**. Toute la littérature sur le MI consiste à construire des modèles paramétrés, à les calibrer sur des données historiques d'exécution, et à valider leur pouvoir prédictif. C'est pour ça qu'il y a tant de modèles concurrents : il n'existe pas de "vrai MI" qu'on pourrait observer pour départager.

### B. Décomposition opérationnelle du coût

À défaut du contrefactuel, on travaille sur le **coût total observé**. La décomposition standard est :

$$\text{tcost} = \underbrace{\text{spread cost}}_{\text{traverser bid-ask}} + \underbrace{\text{realized impact}}_{\text{déplacer le prix}} + \underbrace{\varepsilon}_{\text{bruit marché}}.$$

Le **realized impact** est ce que la modélisation cherche à prédire. Il se décompose lui-même en deux composantes (cf. II) — temporary et permanent — qui ont des origines microstructurelles différentes.

> [!warning] Les trois mesures qu'on confond souvent
> | Concept | Ce que c'est | Observable ? |
> |---|---|:---:|
> | **MI théorique** | différence prix avec/sans ordre (contrefactuel) | non |
> | **Realized impact** | $I^{\text{real}} = \text{MI théorique} + \varepsilon$ | oui (mais bruité) |
> | **Spread cost** | demi-spread payé en traversant le carnet | oui |
> 
> En pratique, on **observe** le realized impact $I^{\text{real}}$ et on **estime** le MI théorique $I(Q)$ par modélisation. Le bruit $\varepsilon = I^{\text{real}} - I(Q)$ capture les mouvements de marché et la volatilité non liés à l'ordre.

## II. Décomposition temporary + permanent

### A. Le schéma temporel

Quand on exécute un gros ordre d'achat sur un horizon $[0, T]$, le prix suit typiquement la trajectoire suivante :

<div style="text-align:center; margin: 1.5rem 0;">
  <img src="images/tca/temp_perm.png" style="max-width:75%;" alt="Décomposition temporary + permanent du market impact"/>
</div>

**Figure 1.** Le prix part de $P_0 = 30\,\$$, monte pendant l'exécution sous la pression d'achat, atteint un pic $P_{\text{peak}} = 30.25\,\$$ à la fin de l'exécution, puis **redescend partiellement** vers un nouveau niveau d'équilibre $P_\infty = 30.05\,\$$ (le marché digère l'ordre, certaines positions courtes reviennent fournir de la liquidité). L'impact total ($P_{\text{peak}} - P_0 = 0.25\,\$$) se décompose en deux parties :

$$\underbrace{(P_{\text{peak}} - P_0)}_{\text{impact total}} = \underbrace{(P_{\text{peak}} - P_\infty)}_{\text{temporary}} + \underbrace{(P_\infty - P_0)}_{\text{permanent}}.$$

Sur l'exemple : $0.25 = 0.20 + 0.05$, soit 80% de temporary et 20% de permanent.

### B. Origines microstructurelles

> [!warning] Les deux composantes
> **Temporary impact.** *La partie du mouvement de prix qui se résorbe après la fin du trade.* Origine microstructurelle : tu consommes la depth du carnet d'ordres en achetant les niveaux successifs de l'ask. Le prix monte parce que les vendeurs disponibles immédiatement sont plus chers que les vendeurs qui se manifesteraient si on attendait. C'est une **prime d'urgence** payée pour exécuter rapidement.
> 
> **Permanent impact.** *La partie qui reste après la résorption.* Origine informationnelle : le marché apprend de ton trade. Un gros achat est interprété comme un signal — soit qu'il y a une information privée (insider), soit que l'actif était sous-évalué. Le prix d'équilibre se reforme à un nouveau niveau. C'est l'effet Kyle classique.

> 💡 **L'image canonique.** Le temporary impact, c'est *consommer la liquidité disponible dans le carnet à l'instant t*. Le permanent impact, c'est *modifier la perception collective du prix d'équilibre*. Le premier est mécanique et se résorbe, le second est informationnel et reste.

### C. Spread cost ≠ temporary impact

Une confusion fréquente : le **spread cost** et le **temporary impact** sont deux choses différentes.

> [!note]- Spread cost vs temporary impact
> - **Spread cost** = demi-spread bid-ask au moment où on envoie l'ordre. C'est l'**impact immédiat** : la différence entre le mid et le prix d'exécution sur le premier niveau du carnet.
> - **Temporary impact** = pression de prix qui se construit **pendant** l'exécution et se résorbe **après**. Couvre tout l'historique du trade, pas juste l'instant initial.
> 
> Exemple : best bid = 99.8, best ask = 100.5, mid = 100.15. Si tu achètes immédiatement au best ask (100.5), ton **spread cost** est $(100.5 - 100.15) / 100.15 \approx 35$ bps. Si en plus ton ordre est gros et fait monter le prix d'exécution moyen à 100.7 sur la durée, le **temporary impact** est $(100.7 - 100.15) / 100.15 \approx 55$ bps (qui *inclut* le spread cost initial mais ajoute la pression cumulée).

## III. La square-root law

### A. La formule

La loi empirique la plus connue en market impact est la **square-root law** :

> [!warning] Square-root law of market impact
> $$I(Q) = Y \cdot \sigma \cdot \sqrt{\frac{Q}{V}}$$
> 
> où :
> - $Q$ : taille de l'ordre (en actions ou en notional),
> - $V$ : volume de référence (typiquement le volume journalier moyen, ADV ou MDV),
> - $\sigma$ : volatilité de l'actif (souvent annualisée puis ramenée à l'horizon journalier),
> - $Y$ : constante d'ordre 1, calibrée empiriquement (typiquement entre 0.5 et 1.5).
> 
> L'**exposant 1/2** sur $Q/V$ est ce qui donne son nom à la loi : le coût croît **comme la racine** de la taille relative de l'ordre, pas linéairement.

### B. Lecture intuitive

L'argument microstructurel pour l'exposant 1/2 est le suivant : plus on va profond dans le carnet, **moins il y a de trades** disponibles à chaque niveau de prix. Un petit ordre consomme un niveau ; un gros ordre consomme plusieurs niveaux, mais l'impact cumulatif augmente moins vite que la taille parce que la liquidité est concave dans le carnet.

> 💡 **Conséquence pratique.** Doubler la taille de l'ordre n'augmente le coût que de $\sqrt{2} \approx 1.41 \times$, pas de $2\times$. Quadrupler ne double pas le coût. Cette concavité est la raison pour laquelle l'industrie utilise massivement cette loi : elle est simple, monotone, robuste sur des ordres "normaux".

### C. Calibration linéaire

La forme square-root se calibre par **régression linéaire** après transformation log-log :

$$\ln I(Q) = \ln Y + \ln \sigma + \frac{1}{2} \ln \frac{Q}{V}.$$

En traitant $\sigma$ comme un input observable (pas un paramètre), on régresse $\ln I^{\text{real}}$ contre $\ln(Q/V)$ et on attend une **pente proche de 0.5** pour valider la square-root.

### D. La critique : square-root ne distingue pas la durée

Voici le problème fondamental que la square-root law **ne capture pas**. Considérons un trader qui doit exécuter $Q = 10\,000$ actions sur un actif d'ADV $V = 100\,000$. Deux stratégies opposées :

> [!example] Même Q/V, deux stratégies opposées
> | Stratégie | Description | Q/V | Coût square-root |
> |---|---|:---:|:---:|
> | (a) **VWAP** | étaler l'ordre sur la journée entière | 0.10 | $Y \sigma \sqrt{0.10}$ |
> | (b) **Agressif** | balancer tout d'un coup en quelques minutes | 0.10 | $Y \sigma \sqrt{0.10}$ |
> 
> **Les deux stratégies donnent le même coût prédit** par la square-root law. Ce qui est manifestement absurde : (b) consomme la depth du carnet en quelques minutes, (a) la consomme étalée sur 6h pendant que le carnet se reconstitue continuellement.

Le problème mathématique : la square-root ne dépend que du **ratio cumulé** $Q/V$, pas de la **vitesse d'exécution**. Elle ignore donc la **participation rate** (POV = $Q$ / volume effectivement traité pendant l'exécution) et la **duration** (durée d'exécution rapportée au jour).

> 💡 **La leçon.** Pour modéliser correctement le market impact, il ne suffit pas de connaître la taille relative de l'ordre — il faut connaître **comment** on l'exécute (vite ou étalé). C'est ce qui motive les modèles plus fins de la section IV.

## IV. Au-delà de la square-root : Zarinelli et al. (2015)

### A. Le constat empirique

Zarinelli, Treccani, Farmer, Lillo (*Market Microstructure and Liquidity*, 2015) — *"Beyond the Square Root: Evidence for Logarithmic Dependence of Market Impact on Size and Participation Rate"* — testent la square-root law sur un dataset couvrant **5 ordres de grandeur** de $Q/V$, beaucoup plus large que les études antérieures qui couvraient typiquement 2 ordres de grandeur.

> [!warning] Le résultat central
> - La **square-root law** fitte bien sur **environ 2 ordres de grandeur** ($Q/V \in [10^{-3}, 10^{-1}]$ environ — la zone "normale" des metaorders).
> - **Aux extrêmes** (très petits ou très grands metaorders), la loi **casse** : elle surestime systématiquement l'impact des très petits ordres et sous-estime celui des très grands.
> - Une forme **logarithmique** (plus concave) fitte beaucoup mieux sur les **5 ordres de grandeur**.

### B. L'impact surface : participation × duration

Zarinelli et al. introduisent l'idée d'une **impact surface** : le market impact n'est pas une fonction de la seule variable $\pi = Q/V$, mais une fonction de **deux** variables :

$$\pi = \frac{Q}{V_D} = \underbrace{\frac{Q}{V_P}}_{F\;:\;\text{participation rate}} \cdot \underbrace{\frac{V_P}{V_D}}_{\eta\;:\;\text{duration ratio}}$$

où :
- $V_P$ = volume échangé sur le marché **pendant** la durée du metaorder,
- $V_D$ = volume échangé sur la **journée entière** (ADV),
- $F = Q/V_P$ = **participation rate** : fraction du volume qu'on représente *pendant qu'on trade*,
- $\eta = V_P/V_D$ = **duration ratio** : fraction de la journée que dure notre exécution.

> 💡 **Ce que ça résout.** La décomposition $\pi = F \cdot \eta$ sépare proprement *l'agressivité* ($F$ : à quelle vitesse on consomme la liquidité) de *l'étalement* ($\eta$ : combien de temps on étale). Reprenons l'exemple VWAP vs agressif :
> 
> | Stratégie | $Q/V_D$ | $F$ | $\eta$ |
> |---|:---:|:---:|:---:|
> | (a) VWAP étalé sur la journée | 0.10 | $\approx 0.10$ | $\approx 1$ |
> | (b) Agressif sur 5 min | 0.10 | $\approx 1$ | $\approx 0.01$ |
> 
> Maintenant les deux stratégies sont **distinctes** dans l'espace $(F, \eta)$ : (b) a un participation rate dramatiquement plus élevé, et c'est précisément ça qui fait son coût plus élevé.

### C. La forme fonctionnelle

Zarinelli et al. proposent un fit empirique sous forme **logarithmique** plutôt que power law :

$$I(Q, F) \;\propto\; \sigma \cdot \ln\Big(1 + a\,F\Big) \cdot g(\eta)$$

avec une fonction $g$ qui capture l'effet de duration. Le détail des paramètres dépend de la calibration (et du dataset) — pour notre usage, ce qui importe c'est le **principe** : *l'impact dépend conjointement de la participation et de la duration, pas uniquement de la taille relative*.

## V. Modèle d'Almgren et al. (2005)

### A. La formule

Almgren, Thum, Hauptmann, Li (*Direct Estimation of Equity Market Impact*, Risk Magazine 2005) proposent un modèle qui sépare explicitement les composantes temporary et permanent :

> [!warning] Modèle Almgren et al. (2005)
> $$\text{MI} = \underbrace{b_1 \cdot I^* \cdot \text{POV}^{a_4}}_{\text{temporary}} + \underbrace{(1 - b_1) \cdot I^*}_{\text{permanent}}$$
> 
> avec :
> 
> $$I^* = a_1 \cdot \left(\frac{Q}{\text{ADV}}\right)^{a_2} \cdot \sigma^{a_3}, \qquad \text{POV} = \frac{Q}{V + Q}$$
> 
> Paramètres :
> - $a_1, a_2, a_3$ : paramètres du **scaling** de l'impact en fonction de la taille et de la volatilité.
> - $a_4$ : exposant de la **participation rate**. Souvent fixé à 1 dans la calibration de base.
> - $b_1 \in [0, 1]$ : **fraction temporary** vs permanent. $b_1 = 0$ → tout est permanent, $b_1 = 1$ → tout est temporary.

### B. Lecture des paramètres

> 💡 **Décomposition conceptuelle.** $I^*$ est l'**impact total** d'un metaorder de taille $Q$ — sans distinction temporary/permanent. Une fois $I^*$ posé, $b_1$ joue le rôle de **clé de répartition** entre les deux composantes. Le facteur $\text{POV}^{a_4}$ multiplie uniquement la composante temporary, ce qui colle à l'intuition : la prime d'urgence (temporary) dépend de la vitesse d'exécution, alors que l'effet informationnel (permanent) dépend juste de la taille totale tradée.

> [!note]- Pourquoi $\text{POV} = Q/(V+Q)$ et pas $Q/V$ ?
> La normalisation $V + Q$ (au dénominateur) évite la divergence quand on devient un participant majeur. Si $Q \gg V$, alors POV → 1 (saturation), pas l'infini. Mathématiquement, c'est une fonction logistique tronquée qui borne POV dans $[0, 1]$. C'est plus stable numériquement et plus réaliste.

### C. Lien avec la square-root

Si on prend le cas particulier $a_2 = 1/2$, $a_3 = 1$, on retrouve l'esprit de la square-root law dans $I^*$ :

$$I^* = a_1 \cdot \sqrt{\frac{Q}{\text{ADV}}} \cdot \sigma$$

Le modèle Almgren **généralise** la square-root en (i) ajoutant la dépendance en POV pour le temporary, (ii) séparant temporary et permanent, (iii) laissant les exposants libres à la calibration au lieu de les fixer à 1/2.

## VI. Calibration empirique du modèle Almgren

### A. Régression linéaire en POV

Dans le cas simplifié $a_4 = 1$ (POV en linéaire), le modèle s'écrit :

$$\text{MI} = (1 - b_1) I^* + b_1 I^* \cdot \text{POV} = \alpha_0 + \alpha_1 \cdot \text{POV}$$

avec $\alpha_0 = (1 - b_1) I^*$ et $\alpha_1 = b_1 \cdot I^*$. C'est une **régression linéaire simple** de MI sur POV — facile à calibrer.

> [!example] Exemple de calibration
> Dataset (artificiel pour l'illustration) : pour 4 buckets de POV différents, on observe les valeurs moyennes de MI suivantes (en bps) :
> 
> | Bucket | POV | MI moyen (bps) |
> |:---:|:---:|:---:|
> | 1 | 0.03 | 10.6 |
> | 2 | 0.05 | 16.5 |
> | 3 | 0.07 | 23.5 |
> | 4 | 0.04 | 30 |
> 
> En forme matricielle :
> $$\begin{pmatrix} 10.6 \\ 16.5 \\ 23.5 \\ 30 \end{pmatrix} = \begin{pmatrix} 1 & 0.03 \\ 1 & 0.05 \\ 1 & 0.07 \\ 1 & 0.04 \end{pmatrix} \begin{pmatrix} \alpha_0 \\ \alpha_1 \end{pmatrix}$$
> 
> La régression OLS donne typiquement $\hat{\alpha}_0 \approx 5.0$ et $\hat{\alpha}_1 \approx 250$.

### B. Méthode du bucketing pour réduire le bruit

Les trades individuels sont **très bruités** — le bruit $\varepsilon$ (mouvement de marché) peut dominer le signal MI sur un trade donné. La parade standard est le **bucketing** :

1. On groupe les trades par tranches de POV similaires (buckets).
2. Dans chaque bucket, on **moyenne** les realized impacts.
3. La moyenne par bucket réduit l'écart-type du bruit comme $1/\sqrt{n_{\text{trades}}}$, ce qui rend le signal MI extractible.

> 💡 **Intuition.** Sur un trade isolé, $\varepsilon$ peut faire dévier l'observation de plusieurs dizaines de bps par rapport au vrai MI. Sur 1000 trades dans un bucket, $\varepsilon$ moyen est divisé par $\sqrt{1000} \approx 31$ — le bruit s'estompe et le signal MI émerge.

### C. Récupérer $b_1$ à partir de $\alpha_0, \alpha_1$

Une fois la régression effectuée, on récupère le coefficient $b_1$ par algèbre élémentaire :

> [!note]- Calcul de $b_1$
> On a $\alpha_0 = (1 - b_1) I^*$ et $\alpha_1 = b_1 \cdot I^*$, donc :
> 
> $$\frac{\alpha_1}{\alpha_0} = \frac{b_1}{1 - b_1} \;\Longrightarrow\; b_1 = \frac{\alpha_1}{\alpha_0 + \alpha_1}.$$
> 
> Et l'impact total $I^*$ se retrouve par :
> 
> $$I^* = \alpha_0 + \alpha_1 = \alpha_0 + \alpha_1.$$
> 
> Sur l'exemple précédent ($\alpha_0 = 5.0$, $\alpha_1 = 250$) :
> 
> $$b_1 = \frac{250}{5 + 250} \approx 0.98, \qquad I^* = 255 \text{ bps}.$$
> 
> Lecture : 98% de l'impact est temporary, 2% permanent — un ratio caractéristique d'un actif liquide où l'effet informationnel est faible.

### D. Estimation de $a_1, a_2, a_3$ par log-log

La régression POV ci-dessus calibre $\alpha_0, \alpha_1$ pour un actif (ou un sous-univers) donné. Pour calibrer la **forme générale** $I^* = a_1 (Q/\text{ADV})^{a_2} \sigma^{a_3}$ qui décrit comment $I^*$ varie d'un actif à l'autre, on passe en **log-log** :

> [!note]- Régression log-log multivariée
> $$\ln I^* = \ln a_1 + a_2 \ln \frac{Q}{\text{ADV}} + a_3 \ln \sigma + \text{erreur}.$$
> 
> C'est une régression linéaire multivariée standard. Sur le cross-section des actifs (ou sur l'historique d'un actif sur plusieurs régimes de volatilité), les coefficients $a_2$ et $a_3$ s'estiment directement. Empiriquement, $a_2 \in [0.4, 0.7]$ (pas exactement 1/2 comme la square-root pure) et $a_3 \approx 1$ (impact proportionnel à la volatilité).

## VII. Considérations pratiques sur les inputs data

Au-delà de la formule mathématique, le choix des inputs détermine la qualité du modèle. Trois points clés.

### A. Volume de référence : qu'est-ce que l'ADV ?

Le volume "journalier" d'un actif n'est pas une grandeur unique — il dépend de ce qu'on **agrège**. Composantes possibles :

- **Lit** : continuous trading sur la lit market (carnet visible).
- **Auctions** : open auction et close auction (volume échangé en début et fin de journée).
- **Dark trades** : trades exécutés dans des dark pools (pas visibles avant exécution).
- **OTC** : trades de gré à gré entre institutionnels, hors marché central.
- **Hidden** : ordres iceberg, partiellement visibles selon le vendor.

> [!warning] Le piège du volume "non consolidé"
> Beaucoup de feeds de données fournissent par défaut un volume **lit-only** (continuous trading uniquement). Ce volume **sous-estime systématiquement** la liquidité totale de l'actif :
> - Pour des large-caps US, le lit représente ~40-60% du volume total — le reste est dark + auctions + OTC.
> - Sur des mid-caps européens, l'écart peut être encore plus marqué.
> 
> **Conséquence** : un modèle calibré sur un volume lit-only **surestime** systématiquement le coût d'exécution, parce que le dénominateur $V$ est artificiellement petit. La pratique professionnelle est d'utiliser un **volume consolidé** (lit + auctions + dark + OTC) comme référence ADV — c'est plus cohérent avec la liquidité réellement disponible.

> 💡 **Comment le détecter.** Un signe que le volume utilisé est lit-only : si on compare l'ADV d'une source académique (souvent lit-only) à l'ADV d'un vendor pro (Reuters, Bloomberg, vendor consolidé), on observe un facteur 2-3× d'écart sur des large-caps. C'est un point de gouvernance data important quand on construit un modèle interne.

### B. Volatilité : fenêtre et vol floor

La volatilité $\sigma$ apparaît comme multiplicateur dans toutes les formules de MI. Deux choix méthodologiques sont cruciaux.

**Fenêtre de calcul.** La volatilité réalisée se calcule sur une fenêtre glissante. Choix typique : 21 jours (≈ 1 mois de trading) ou 50 jours (≈ 2.5 mois).

> [!note]- Trade-off court terme / long terme
> | Fenêtre | Avantages | Inconvénients |
> |---|---|---|
> | **Court terme (21j)** | Réactif aux changements de régime de volatilité | Très bruité, instable d'un jour à l'autre |
> | **Long terme (50j+)** | Stable, robuste statistiquement | Lent à réagir aux chocs de volatilité réels |
> 
> En risk management, on préfère typiquement la fenêtre longue : on accepte la lenteur en échange de la stabilité, parce que les limites risk basées sur un MI bruité changeraient tous les jours, ce qui est ingérable opérationnellement.

**Vol floor.** Cas pathologique : un stock est **en cours d'acquisition** (annonce de tender offer). Son prix se fige à un niveau proche du prix d'offre, sa volatilité réalisée tombe à $\sigma \approx 0$. Conséquence mécanique : le modèle prédit un impact $\approx 0$, ce qui est absurde — l'actif n'est pas devenu "infiniment liquide", il est juste figé en attendant l'issue de l'opération.

> [!warning] Vol floor
> $$\sigma_{\text{utilisé}} = \max(\sigma_{\text{réalisée}}, \sigma_{\text{floor}})$$
> 
> où $\sigma_{\text{floor}}$ est typiquement la volatilité moyenne du secteur ou un seuil minimum (ex. 10% annualisé). Empêche les cas pathologiques de produire des estimations de coût absurdement faibles.

### C. Spread cost : conventions de mesure

Le spread cost peut se mesurer de plusieurs manières :

- **Quoted spread** instantané : (ask − bid) au moment du trade. Très bruité, dépend du timing exact.
- **Time-weighted spread** : moyenne du spread pondérée par le temps, sur une fenêtre (ex. 5 jours). Plus stable.
- **Volume-weighted spread** : moyenne pondérée par le volume échangé. Représentatif du spread effectivement payé.

> 💡 **Le choix de fenêtre.** En pratique, une fenêtre de 5 jours (time-weighted ou volume-weighted) est un bon compromis : assez longue pour lisser le bruit intraday, assez courte pour refléter les conditions actuelles. Plus la fenêtre est longue, plus le spread cost devient un *paramètre structurel* de l'actif (caractéristique de sa microstructure) plutôt qu'une mesure ponctuelle.

## VIII. Cas applicatifs

Une fois le modèle de MI calibré, il sert dans plusieurs contextes opérationnels.

### A. Max delta : inversion du modèle pour limites risk

**Le besoin.** Le risk management veut imposer des limites par stock du type *"on ne peut pas porter une position dont le coût de liquidation dépasserait $c^* = 1\,000\,\$$"*. Pour traduire cette contrainte en limite de notional, il faut **inverser** le modèle de MI.

**La formulation.** Pour un actif donné, on a :

$$c_{\text{self}}(X) = f(\sigma, X, \text{ADV}, \text{spread})$$

où $c_{\text{self}}$ est le coût de liquidation d'une position de taille $X$. La question est : quelle est la **taille maximale** $X^*$ telle que $c_{\text{self}}(X^*) \leq c^*$ ?

**Sous le modèle Almgren** avec $a_4 = 1$, $c_{\text{self}}$ est une fonction croissante et convexe de $X$ (en racine carrée pour la square-root, en log pour Zarinelli), donc l'inversion est unique. Numériquement, on résout :

$$X^* = \arg\min_X \big(c_{\text{self}}(X) - c^*\big)^2 \quad \text{s.c.} \quad c_{\text{self}}(X) \leq c^*.$$

> [!example] Reparamétrisation en notional
> Le notional d'une position est $\text{Notional} = P \cdot X$. Dans la pratique, le risk manager fixe le seuil en USD (ex. $c^* = 1\,000\,\$$ de coût) et obtient un **max notional** par stock — qui peut ensuite être exprimé en max delta cash dans les systèmes de risk.

### B. Stress test vol/volume

**Le besoin.** Quand on stresse la volatilité ($\sigma \to \sigma + \Delta\sigma$), comment évolue le coût de liquidation du portefeuille ? Le piège : $\sigma$ et $V$ sont **corrélés** — quand la volatilité monte, le volume monte aussi typiquement. Donc stresser $\sigma$ sans toucher $V$ surestime l'impact du stress.

**Étude empirique typique.** On régresse $\Delta V$ contre $\Delta \sigma$ sur l'historique :

$$\Delta V_t = \beta \cdot \Delta \sigma_t + \varepsilon_t$$

> [!warning] Résultat empirique
> Sur des univers liquides (S&P 500, EuroStoxx, indices chinois), la régression jour-à-jour donne typiquement un **$R^2$ très faible** (souvent < 0.05). La corrélation $\sigma$–$V$ existe à long terme (régimes de volatilité), mais elle est **non exploitable à court terme** — le bruit domine.
> 
> **Conclusion pratique** : on ne peut pas modéliser de façon fiable la réponse du volume à un choc de volatilité au niveau jour-à-jour. On stresse $\sigma$ et $V$ **séparément**, en se basant sur des scénarios historiques (ex. "volatilité × 2 et volume × 1.5 comme en mars 2020") plutôt que sur une régression.

### C. Reporting portefeuille

Au niveau portfolio, on agrège des métriques pour identifier les concentrations de risque liquidité :

- **Net delta** : delta total signé (longs − shorts). Mesure l'exposition directionnelle.
- **Gross delta** : somme des |delta| par stock. Mesure la taille brute du portefeuille (pertinent quand longs/shorts ne se compensent pas en termes de risque liquidité).
- **%ADV** : ratio (position / ADV) par stock. Mesure la taille de la position rapportée à la liquidité — directement lié au coût de liquidation.
- **Distribution buckets** : histogramme du gross delta par tranches de %ADV. Permet de voir si le portefeuille est concentré sur des stocks peu liquides (queue de distribution lourde côté %ADV élevé).

> 💡 **Lecture risk.** Deux portefeuilles de même net delta peuvent avoir des profils de liquidité radicalement différents. Un portefeuille concentré sur 5 large-caps liquides est facilement liquidable ; un portefeuille répartissant le même notional sur 200 mid-caps peu liquides peut être inliquidable en pratique. Le reporting par buckets de %ADV est l'outil standard pour identifier ces situations.

## IX. Limites et extensions

### A. Inventory model vs propagator

Les modèles présentés (square-root, Almgren) font partie de la famille des **inventory models** (alias *closed-form models*) : ils donnent une **formule fermée** liant les inputs (taille, vol, spread) à un coût scalaire.

Une famille alternative est celle des **propagator models** : on modélise dynamiquement la **trajectoire de prix** en fonction de la séquence d'ordres, avec un noyau de propagation qui décrit comment chaque ordre individuel impacte le prix sur tous les instants futurs (avec décroissance temporelle).

> [!note]- Inventory vs propagator — différence conceptuelle
> | | Inventory model | Propagator model |
> |---|---|---|
> | **Sortie** | Coût agrégé scalaire | Trajectoire de prix complète |
> | **Inputs** | Taille, vol, spread, POV | Séquence d'ordres + noyau de décroissance |
> | **Calibration** | Régression sur metaorders | Calibration dynamique sur tick data |
> | **Force** | Simple, robuste, calibrable | Capture l'effet de la structure temporelle |
> | **Faiblesse** | Ignore la trajectoire intra-trade | Beaucoup plus de paramètres, instable |
> 
> Référence canonique côté propagator : Bouchaud, Bonart, Donier, Gould (2018), *Trades, Quotes and Prices*. Côté inventory : Almgren et al. (2005), Grinold & Kahn, Kissell.

### B. Le modèle est en "vase clos"

Toutes les formules de cette note traitent **un actif isolé** : impact de mes ordres sur **mon** prix, sans référence aux autres actifs ni aux autres agents. C'est une limite majeure :

- Sur un **basket** corrélé (ex. 30 stocks tech qu'on liquide ensemble), le coût total n'est pas la somme des coûts individuels — il y a des effets de **cross-impact** (mes ventes sur Apple bougent aussi le prix de Microsoft via la corrélation).
- L'environnement de marché n'est pas neutre : d'autres agents tradent simultanément, et leurs ordres interagissent avec les miens. Le modèle "vase clos" ignore complètement ces interactions.

C'est précisément ce que cherchent à capturer les modèles de **cross-impact**, traités dans [[03_Cross-impact|note iii]].

---

## Résumé — fil logique

| Section | Contenu |
|---|---|
| **I** | Heisenberg : MI = différence avec/sans ordre est inobservable → on doit modéliser. |
| **II** | Realized impact = temporary (mécanique, depth du carnet, prime d'urgence) + permanent (informationnel, prix d'équilibre déformé). Spread cost ≠ temporary. |
| **III** | Square-root law : $I = Y \sigma \sqrt{Q/V}$. Calibrable en log-log. **Limite cruciale** : ne distingue pas VWAP d'agressif (même $Q/V$). |
| **IV** | Zarinelli (2015) : square-root casse aux extrêmes ; impact surface $\pi = F \cdot \eta$ (participation × duration). |
| **V** | Modèle Almgren : MI = $b_1 I^* \text{POV}^{a_4}$ (temp) + $(1-b_1) I^*$ (perm), avec $I^* = a_1 (Q/\text{ADV})^{a_2} \sigma^{a_3}$. |
| **VI** | Calibration : régression linéaire MI = $\alpha_0 + \alpha_1$ POV (cas $a_4 = 1$) + bucketing pour réduire le bruit + log-log pour $a_1, a_2, a_3$. |
| **VII** | Inputs data : ADV consolidé (lit + auctions + dark + OTC), vol fenêtre longue + floor, spread time-weighted/volume-weighted. |
| **VIII** | Cas applicatifs : max delta (inversion), stress test vol/volume ($R^2$ faible jour-à-jour), reporting %ADV par bucket. |
| **IX** | Limites : modèle scalaire (inventory) vs trajectoire (propagator) ; vase clos → motive le cross-impact. |
