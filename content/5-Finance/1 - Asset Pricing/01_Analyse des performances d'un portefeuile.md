---
title: 01-Analyse des performances d'un portefeuille
order: 0
---
# Analyse des performances d'un portefeuille

> Avant de modéliser quoi que ce soit en finance, on regarde la data. Cette note pose les outils statistiques de base pour analyser une série financière : pourquoi on travaille sur les **rendements** plutôt que les prix, pourquoi le prix ne se prédit pas mais la **volatilité** si, comment l'**hypothèse de marché efficient (EMH)** encadre ces faits, et quelles précautions prendre quand on fait tourner une **régression linéaire** sur de la data financière.

## I. Prix ou rendements ?

**Définitions.** À partir d'une série de prix $P_t$, on définit le rendement (return) simple

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

et le rendement logarithmique

$$\ell_t = \ln\frac{P_t}{P_{t-1}}.$$

Pour des variations faibles, $\ell_t \approx r_t$. Le log-return a l'avantage d'être additif dans le temps : $\sum_t \ell_t = \ln(P_T / P_0)$.

**Pourquoi ne pas étudier directement les prix ?** Les prix possèdent deux défauts qui rendent l'analyse statistique pénible :
- ils sont **non stationnaires** (leur moyenne et leur variance dérivent dans le temps),
- leur niveau dépend du passé arbitrairement loin (le prix d'aujourd'hui mémorise toute l'histoire).

Les rendements gomment ces deux problèmes en première approximation : ils oscillent autour d'une moyenne approximativement constante, et leur niveau ne mémorise pas le passé long. C'est *la* raison structurelle pour laquelle l'industrie travaille sur les returns.

![[images/5-Finance/0_analyse_performances/im1.png]]

**Figure 1.** Une même série financière vue en niveau (à gauche) et en rendement (à droite). À gauche, le prix dérive — sa moyenne et sa dispersion ne sont pas stables. À droite, les rendements oscillent autour de 0 et restent dans une bande approximativement constante.

## II. Stationnarité

> [!warning] Stationnarité (au sens faible)
> Une série $(X_t)$ est stationnaire si sa moyenne $\mathbb{E}[X_t]$ et sa fonction d'autocovariance $\mathrm{Cov}(X_t, X_{t+h})$ ne dépendent pas de $t$. Toute la théorie statistique classique (loi des grands nombres, théorèmes de convergence) suppose la stationnarité.

**Le contraste empirique.** Pour vérifier la stationnarité d'une série, on calcule une **moyenne glissante** et un **écart-type glissant** sur une fenêtre fixe (ex : 1 an) et on regarde si ces statistiques sont stables dans le temps.

![[images/5-Finance/0_analyse_performances/im2.png]]

**Figure 2.** Moyennes et écarts-types glissants sur 1 an. **Prix** : moyenne fortement dérivante (de 95 à 20), écart-type qui bouge également → série très clairement non stationnaire. **Rendements** : moyenne stable autour de 0 — la stationnarité en moyenne tient. **Mais** l'écart-type des rendements bouge significativement dans le temps : la variance n'est pas constante. C'est le **volatility clustering**, qu'on traite plus loin.

> [!note] Variance conditionnelle vs inconditionnelle
> Dire "les returns sont stationnaires" mérite une précision. Formellement :
> - **Stationnarité faible (inconditionnelle)** : $\mathbb{E}[r_t] = \mu$ et $\text{Var}(r_t) = \sigma^2$ constants en moyenne sur le long terme — approximativement vraie pour des returns.
> - **Hétéroscédasticité conditionnelle** : $\text{Var}(r_t \mid \mathcal{F}_{t-1}) = \sigma_t^2$ — la variance *instantanée* connue à $t-1$ dépend du temps. C'est ce que capte le volatility clustering.
>
> Les returns sont approximativement stationnaires au sens faible inconditionnel, mais **conditionnellement hétéroscédastiques**. C'est tout l'objet des modèles ARCH/GARCH : modéliser $\sigma_t^2$ comme fonction de l'historique récent, dans un cadre globalement stationnaire.

## III. L'hypothèse de marché efficient (EMH)

L'**EMH** (Fama, 1970) est le cadre théorique qui justifie la difficulté à prédire les prix. Idée centrale : un prix de marché incorpore déjà toute l'information disponible, donc tout signal exploitable a déjà été arbitré.

> [!warning] Trois formes de l'EMH
> - **Forme faible** : les prix passés n'ont aucun pouvoir prédictif. L'analyse technique pure ne marche pas.
> - **Forme semi-forte** : toute information publique (résultats, news, ratios) est déjà dans le prix. Seule l'information privée pourrait donner un avantage.
> - **Forme forte** : même l'information privée est intégrée. Personne ne peut battre le marché systématiquement.
>
> La forme **semi-forte** est celle qui résume le compromis empirique : la plupart des données publiques ont un pouvoir prédictif très faible, mais pas exactement nul (sinon il n'y aurait pas de quant research).

**Conséquence statistique.** Si l'EMH faible tient, alors les rendements ne sont pas prévisibles à partir de leur propre passé. Formellement, $\mathbb{E}[r_{t+1} \mid \mathcal{F}_t] \approx 0$ (martingale-like), ce qui implique que l'**autocorrélation des rendements doit être nulle** à tous les lags.

![[images/5-Finance/0_analyse_performances/im3.png]]

**Figure 3.** ACF et PACF des rendements. Quasiment toutes les barres sont à l'intérieur de l'intervalle de confiance à 95 % (bande grise) : statistiquement on ne rejette pas $\rho_k = 0$. C'est la signature empirique de la forme faible de l'EMH — les rendements ne se prédisent pas eux-mêmes.

> [!warning] Ce que ça **n'**implique **pas**
> L'EMH ne dit pas que rien ne prédit les rendements — elle dit que les **rendements passés** ne prédisent pas les rendements futurs. D'autres variables (facteurs fondamentaux, microstructure, sentiment) peuvent porter du signal — c'est le terrain de jeu de la quant research.

## IV. Volatility clustering — la vol, elle, se prédit

L'imprédictibilité des returns ne signifie pas que la série est *iid*. Si l'on regarde les **rendements au carré** ou en valeur absolue, on observe un fait empirique massif et universel :

> Les jours de grande variation sont suivis de jours de grande variation, et les jours calmes de jours calmes.

C'est le **volatility clustering**, identifié par Mandelbrot (1963).

![[images/5-Finance/0_analyse_performances/im4.png]]

**Figure 4.** À gauche, $|r_t|$ — on voit clairement des grappes de jours de forte amplitude. À droite, l'ACF de $r_t^2$ montre une autocorrélation **positive et significative** sur de nombreux lags : la variance d'aujourd'hui est très bien prédite par la variance d'hier.

> [!warning] La règle empirique fondamentale
> - $r_t$ n'a **pas** d'autocorrélation → **les prix ne se prédisent pas**.
> - $r_t^2$ (ou $|r_t|$) a une **forte autocorrélation** → **la volatilité se prédit**.
>
> C'est sur cette asymétrie que reposent toutes les familles de modèles de volatilité conditionnelle (ARCH, GARCH, stochastic volatility) — qui constituent une part majeure de la recherche en finance quantitative.

## V. Régression linéaire en finance — les pièges

La régression linéaire est l'outil de base pour expliquer un rendement à partir de plusieurs **facteurs** (autres returns, ratios fondamentaux, signaux techniques). Le modèle s'écrit

$$r_t = \beta_0 + \beta_1 X^{(1)}_t + \cdots + \beta_p X^{(p)}_t + \varepsilon_t.$$

Le théorème de **Gauss-Markov** garantit que l'estimateur OLS est BLUE (best linear unbiased estimator) **sous quatre hypothèses** : linéarité, exogénéité, **homoscédasticité** et **absence d'autocorrélation des résidus**. En finance, les deux dernières s'effondrent presque systématiquement, et trois autres pathologies viennent s'ajouter. Tour d'horizon.

### A. Multicolinéarité — le *factor zoo*

Les facteurs financiers se ressemblent énormément. Momentum 20j et momentum 25j sont quasi identiques ; value HML et quality HML se chevauchent ; un secteur et le marché global sont corrélés. Quand deux régresseurs sont fortement corrélés, OLS ne peut plus séparer leurs contributions et les coefficients deviennent **instables**, parfois énormes en valeur absolue avec des signes opposés qui se compensent.

![[images/5-Finance/0_analyse_performances/im5.png|388]]

**Figure 5.** Matrice de corrélation entre 5 régresseurs simulés. $X_1 \approx X_2$ et $X_3 \approx X_4$ forment deux clusters de colinéarité ; $X_5$ est indépendant. OLS attribuera des coefficients arbitraires à $X_1$ et $X_2$ (et à $X_3, X_4$), seule leur somme étant identifiée.

> [!note] Solutions
> - **Ridge** : pénalité $\lambda \|\beta\|_2^2$ qui distribue la charge entre régresseurs corrélés. Standard en quant.
> - **Lasso** : pénalité $\lambda \|\beta\|_1$ qui **annule** des coefficients. Utile quand on veut sélectionner un sous-ensemble parmi un grand nombre de facteurs.
> - **PCA / orthogonalisation** : décorréler les régresseurs en amont.

### B. Autocorrélation des résidus

Si les rendements de marché sont approximativement non autocorrélés (Figure 3), les **résidus d'une régression** ne le sont pas forcément. En particulier dès qu'un facteur prédictif important est omis (ou mal capté), sa persistance s'évacue dans les résidus.

![[images/5-Finance/0_analyse_performances/im6.png]]

**Figure 6.** ACF des résidus OLS. La barre à lag 1 est ≈ 0.35, à lag 2 ≈ 0.14 — bien au-dessus de la bande de confiance gris-clair. Les résidus sont fortement autocorrélés, ce qui viole l'hypothèse iid de Gauss-Markov.

**Conséquence.** Les écarts-types des coefficients calculés par OLS sont **sous-estimés** — on conclut à la significativité de variables qui ne le sont pas. C'est *la* source classique de fausses découvertes en quant research.

> [!note] Solution standard : Newey-West (HAC)
> Les **erreurs-types HAC** (Heteroscedasticity and Autocorrelation Consistent) de Newey & West corrigent simultanément l'autocorrélation et l'hétéroscédasticité des résidus. À utiliser systématiquement quand on régresse sur de la data financière. En `statsmodels` : `model.fit(cov_type='HAC', cov_kwds={'maxlags': L})`.

### C. Hétéroscédasticité

Le volatility clustering (section IV) se transmet directement aux résidus : la variance de $\varepsilon_t$ n'est pas constante dans le temps.

![[images/5-Finance/0_analyse_performances/im7.png]]

**Figure 7.** À gauche les résidus bruts $\hat\varepsilon_t$, à droite leur carré. La dispersion des résidus n'est manifestement pas constante : on voit nettement des périodes calmes et des périodes de forte variance. C'est le pendant régressionnel du volatility clustering.

**Conséquence.** Même sans autocorrélation, des résidus hétéroscédastiques rendent les SE OLS biaisés.

> [!note] Solutions
> - **White standard errors** (HC0, HC1) : robustes à l'hétéroscédasticité seule.
> - **Newey-West (HAC)** : robustes à hétéroscédasticité + autocorrélation, donc le choix par défaut en finance.
> - **WLS** : si l'on peut modéliser la variance conditionnelle (ex : via GARCH), pondérer par $1/\hat\sigma_t^2$.

### D. Non-normalité — *fat tails*

Les rendements financiers ont des **queues de distribution plus épaisses** que la normale. Kurtosis empirique > 3, fréquemment ≥ 5. Les résidus de régression héritent de cette propriété.

![[images/5-Finance/0_analyse_performances/im8.png|340]]

**Figure 8.** QQ-plot des résidus standardisés contre la loi normale. Si les résidus étaient gaussiens, les points seraient alignés sur la diagonale. On observe au contraire que les points décrochent dans les deux queues : les extrêmes sont plus fréquents et plus grands que ce qu'une normale prédit.

**Conséquence.** Les tests t et F classiques reposent sur la normalité des résidus en petit échantillon. En présence de fat tails, ces tests sous-estiment l'incertitude des coefficients, surtout en queue (ce qui matterait pour des stratégies tail-risk).

> [!note] Solutions
> - **Bootstrap** (par blocs si autocorrélation) : intervalles de confiance qui ne supposent pas la normalité.
> - **Robust regression** : Huber, RANSAC, ou M-estimators réduisent l'influence des outliers.
> - **Distribution conditionnelle non gaussienne** (Student-t, EGB2) pour le modèle d'erreur.

### E. Stationnarité des régresseurs

Dernière pathologie, plus subtile : si les régresseurs eux-mêmes sont **non stationnaires** (ex : on régresse un prix sur un autre prix, ou un return cumulé sur un autre), on obtient des **régressions fallacieuses** (*spurious regressions*, Granger & Newbold 1974). Le $R^2$ peut être élevé et les coefficients très significatifs alors qu'aucune relation véritable n'existe.

> [!warning] Règle de pouce
> Avant de régresser une variable financière sur une autre, vérifier la stationnarité (test ADF, KPSS) et, si nécessaire, travailler sur les **différences** (returns) plutôt que sur les **niveaux** (prix). C'est le pendant statistique du choix prix vs rendements de la section I.

## VI. Récapitulatif et ouverture

| Fait empirique | Implication |
|---|---|
| Prix non stationnaires | On travaille sur les rendements |
| Rendements stationnaires en moyenne | OLS sur returns techniquement valide |
| Rendements non autocorrélés (EMH faible) | Les prix ne se prédisent pas |
| $r_t^2$ très autocorrélé (vol clustering) | La volatilité, elle, se prédit → GARCH |
| Régresseurs corrélés (factor zoo) | Multicolinéarité → Ridge / Lasso |
| Résidus autocorrélés | SE OLS biaisés → Newey-West |
| Résidus hétéroscédastiques | SE OLS biaisés → White / HAC |
| Fat tails | Tests t classiques peu fiables → bootstrap |

**Prochaines briques** :
- **Markowitz** : construction de portefeuille moyenne-variance à partir de ces rendements.
- **CAPM** : premier modèle d'équilibre liant rendement attendu et risque systématique.
- **Modèles factoriels** (Fama-French, Barra) : régression à l'échelle cross-sectionnelle pour décomposer rendements en expositions à des facteurs communs.
- **Modèles de volatilité conditionnelle** (GARCH) : exploitation directe du volatility clustering.
