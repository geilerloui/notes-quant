---
title: "Le CAPM — Capital Asset Pricing Model"
date: 2025-03-21
tags: [asset-pricing, capm, markowitz, portfolio-theory]
---

Le CAPM (*Capital Asset Pricing Model*), ou MEDAF en français (*Modèle d'Évaluation des Actifs Financiers*), est le premier modèle à donner un **prix au risque d'un actif individuel**. Développé par Treynor (1961), Sharpe (1964), Lintner (1965) et Mossin (1966), il repose sur les travaux d'optimisation de Markowitz (1952) et de Tobin (1958).

---

## 1. De Markowitz au CAPM — la chaîne logique

### (i) Markowitz et la frontière efficiente

Markowitz montre qu'en combinant des actifs, on peut réduire le risque sans sacrifier le rendement. L'ensemble des portefeuilles optimaux forme la **frontière efficiente** : pour chaque niveau de risque $\sigma$, c'est le portefeuille maximisant $\mathbb{E}[R]$.

**Le blocage pratique.** Pour $N$ actions, il faut estimer $N(N-1)/2$ covariances. Pour 500 actions, cela représente 124 750 paramètres — ingérable et trop bruité pour être fiable.

<div style="text-align:center">
<img src="images/capm/im1.png" style="max-width:75%" alt="Frontière efficiente et CML" />
</div>

*Figure 1. Frontière efficiente et Capital Market Line. Tout investisseur rationnel détient le portefeuille tangent $T$ pour sa partie risquée, dosé avec $R_f$ selon son aversion au risque.*

### (ii) Le Single Index Model — le chaînon manquant

Sharpe (1963) propose une hypothèse simplificatrice : deux actions ne sont corrélées *qu'à travers le marché*. Les résidus $\varepsilon_i$ sont indépendants entre eux. La covariance entre deux actifs devient alors :

$$\text{Cov}(R_i, R_j) = \beta_i \cdot \beta_j \cdot \sigma_m^2$$

Pour 500 actions, on passe de 124 750 paramètres à **501 paramètres** (500 bêtas + 1 variance de marché). Markowitz devient calculable à grande échelle.

### (iii) L'équilibre de marché — T = marché

Si tous les investisseurs sont rationnels (hypothèse H1), ils détiennent tous le même portefeuille risqué optimal : le **portefeuille tangent** $T$ (celui qui maximise le ratio de Sharpe). Tobin (1958) montre que chaque investisseur dose son exposition en mixant $T$ avec l'actif sans risque $R_f$ selon son aversion au risque.

À l'équilibre de marché (H2), si tout le monde veut $T$, les prix s'ajustent jusqu'à ce que $T$ soit le portefeuille de marché — chaque actif pondéré par sa capitalisation boursière. Ce n'est pas une hypothèse supplémentaire : c'est une conséquence logique.

> **Exemple.** Markowitz donne $T = 40\%$ Apple $\cdot$ $35\%$ Total $\cdot$ $25\%$ LVMH. En agrégeant les portefeuilles de tous les investisseurs, ces poids deviennent les poids du marché.

**Ce qui en découle.** Le seul risque non diversifiable d'un actif est sa corrélation avec le marché — mesurée par $\beta_i = \text{Cov}(R_i, R_m) / \text{Var}(R_m)$. C'est le seul risque rémunéré à l'équilibre.

<details>
<summary>Les trois hypothèses du modèle</summary>

**H1 · Agents rationnels.** Chaque investisseur détient un portefeuille sur la frontière efficiente de Markowitz. Tous veulent $T$ pour leur partie risquée, dosé avec $R_f$ selon leur aversion au risque.

**H2 · Équilibre de marché ($S = D$).** Si tout le monde veut $T$, les prix s'ajustent pour que $T$ soit le marché entier — chaque actif pondéré par sa capitalisation boursière.

**H3 · Marchés parfaits.** Pas de coûts de transaction, pas de taxes, accès au même taux sans risque $R_f$ pour tous, anticipations homogènes sur les rendements futurs. Ces hypothèses sont irréalistes en pratique, mais nécessaires pour obtenir une formule fermée propre.

</details>

---

## 2. La formule du CAPM

$$\boxed{\mathbb{E}[R_i] = R_f + \beta_i \cdot \left(\mathbb{E}[R_m] - R_f\right)}$$

| Terme | Définition | Ordre de grandeur |
|---|---|---|
| $R_f$ | Taux sans risque — rémunération du temps. OAT 10 ans, T-Bill US. | $\sim 3\%$ |
| $\mathbb{E}[R_m] - R_f$ | Prime de risque de marché. | $\sim 5$–$7\%$/an historiquement |
| $\beta_i$ | $\text{Cov}(R_i, R_m) / \text{Var}(R_m)$ — sensibilité de l'actif au marché. | $2.0$ pour Tesla |

**Application numérique.** Avec $R_f = 3\%$ et prime $= 5\%$ :

$$\mathbb{E}[R_\text{Apple}] = 3\% + 1.5 \times 5\% = 10.5\%$$
$$\mathbb{E}[R_\text{Total}] = 3\% + 0.8 \times 5\% = 7\%$$
$$\mathbb{E}[R_\text{LVMH}] = 3\% + 0.6 \times 5\% = 6\%$$

**Vérification de cohérence.** Avec le portefeuille tangent $T = (40\%, 35\%, 25\%)$ :

$$0.40 \times 10.5\% + 0.35 \times 7\% + 0.25 \times 6\% = 8\% = \mathbb{E}[R_m] \checkmark$$

### (i) Interprétation du bêta

| $\beta$ | Comportement | Exemples | Qualificatif |
|---|---|---|---|
| $< 0$ | Monte quand le marché chute | Or, VIX | Assurance |
| $0 \to 1$ | Moins volatile que le marché | McDonalds ($\beta = 0.7$) | Défensif |
| $= 1$ | Identique au marché | ETF S\&P 500 | Marché |
| $> 1$ | Amplifie les mouvements | Tesla ($\beta \approx 2$), semi-conducteurs | Offensif |

### (ii) La Security Market Line (SML)

La SML est la représentation graphique du CAPM : axe $x = \beta$, axe $y = \mathbb{E}[R]$. Elle passe par $(0, R_f)$ et $(1, \mathbb{E}[R_m])$. Le point marché est **toujours sur la droite par construction**.

$$\mathbb{E}[R_i] = R_f + \beta_i \cdot \underbrace{(\mathbb{E}[R_m] - R_f)}_{\text{pente de la SML}}$$

- **Au-dessus de la SML** : $\alpha > 0$ — l'actif offre plus que ce que son risque justifie → sous-évalué, signal d'achat.
- **En dessous de la SML** : $\alpha < 0$ — l'actif offre moins → sur-évalué, signal de vente.

<div style="text-align:center">
<img src="images/capm/im2.png" style="max-width:75%" alt="Security Market Line" />
</div>

*Figure 2. Security Market Line. Le point marché ($\beta=1$) est sur la droite par construction. Les actifs au-dessus ont $\alpha > 0$ (sous-évalués), ceux en dessous ont $\alpha < 0$ (sur-évalués).*

---

## 3. Décomposition du risque total

Le modèle empirique s'écrit $R_i - R_f = \alpha_i + \beta_i(R_m - R_f) + \varepsilon_i$. En prenant la variance :

$$\underbrace{(\sigma^i)^2}_{\text{Risque total}} = \underbrace{(\beta_i)^2 \, \text{Var}(R_m)}_{\text{Risque systématique}} + \underbrace{\text{Var}(\varepsilon_i)}_{\text{Risque idiosyncratique}}$$

> **Attention.** C'est une décomposition de la **variance**, pas de la volatilité. On ne peut pas écrire $\sigma_\text{total} = \sigma_\text{sys} + \sigma_\text{idio}$ car $\sqrt{a+b} \neq \sqrt{a} + \sqrt{b}$.

**Risque systématique** $(\beta_i)^2 \, \text{Var}(R_m)$ — lié au marché global (récessions, inflation, crises). **Non diversifiable.** C'est le seul risque rémunéré par le CAPM.

**Risque idiosyncratique** $\text{Var}(\varepsilon_i)$ — spécifique à l'entreprise (scandale CEO, brevet raté, usine qui brûle). **Diversifiable** : avec environ 30 actifs décorrélés, il disparaît. Non rémunéré car évitable gratuitement.

> **Exemple.** McDonalds ($\beta = 0.70$, $\sigma = 1.05$) et Amazon ($\beta = 1.46$, $\sigma = 1.02$) ont une volatilité totale similaire, mais de nature opposée. Amazon est quasi entièrement systématique — son risque est non diversifiable et justifie un rendement attendu plus élevé. McDonalds a surtout du risque propre qu'on peut éliminer en diversifiant.

<div style="text-align:center">
<img src="images/capm/im4.png" style="max-width:75%" alt="Décomposition du risque total" />
</div>

*Figure 4. Décomposition du risque total par actif. La part systématique (non diversifiable, rémunérée) croît avec $\beta$. La part idiosyncratique (diversifiable, non rémunérée) varie selon les caractéristiques propres de chaque titre.*

### (i) Preuve mathématique de la diversification

Considérons un portefeuille équipondéré de $N$ actions (poids $1/N$). En développant la variance :

$$\text{Var}(R_p) = \frac{1}{N} \cdot \overline{\text{Var}} + \left(1 - \frac{1}{N}\right) \cdot \overline{\text{Cov}}$$

Quand $N \to \infty$ :
- Le terme $\frac{1}{N} \cdot \overline{\text{Var}} \to 0$ — le risque idiosyncratique disparaît car les $\varepsilon_i$ indépendants s'annulent mutuellement.
- Le terme $\left(1 - \frac{1}{N}\right) \cdot \overline{\text{Cov}} \to \overline{\text{Cov}}$ — les covariances convergent vers leur moyenne, c'est le plancher systématique.

<div style="text-align:center">
<img src="images/capm/im3.png" style="max-width:75%" alt="Diversification du risque" />
</div>

*Figure 3. Diversification du risque de portefeuille. Le risque spécifique décroît en $1/N$ et s'annule avec une trentaine d'actions. Le risque systématique constitue un plancher incompressible.*

**Pourquoi seul le systématique est rémunéré.** Si tu peux éliminer le risque idiosyncratique gratuitement (en ajoutant des actions), le marché ne te paiera jamais pour le porter. Un investisseur rationnel diversifie toujours — à l'équilibre, le risque spécifique n'est donc jamais rémunéré.

---

## 4. L'alpha — deux usages à ne pas confondre

### (i) Alpha ex ante — mispricing (forward-looking)

$$\alpha_i^{ex\ ante} = \hat{\mathbb{E}}[R_i] - \underbrace{\left[R_f + \beta_i \cdot \text{ERP}\right]}_{k_\text{CAPM}}$$

L'alpha ex ante répond à la question : *"Ce stock est-il mal pricé aujourd'hui ?"* Il confronte :
- Le **rendement estimé** $\hat{\mathbb{E}}[R_i]$ — issu d'une source externe (analystes, Gordon-Shapiro, modèle quant).
- Le **rendement requis** $k_\text{CAPM}$ — la barre fixée par le CAPM pour ce niveau de risque.

> **Exemple.** Tesla $\beta = 1.5$, $R_f = 3\%$, ERP $= 5\%$ → requis CAPM $= 10.5\%$. Les analystes estiment $13\%$. Donc $\alpha^{ex\ ante} = +2.5\%$ → signal d'achat.

### (ii) Alpha de Jensen — performance (backward-looking)

$$\alpha_p^{Jensen} = R_p^{\text{réalisé}} - \left[R_f + \beta_p \cdot (R_m - R_f)\right]$$

Ou via régression : $R_p - R_f = \alpha_p + \beta_p(R_m - R_f) + \varepsilon_p$.

L'alpha de Jensen répond à : *"La performance observée venait-elle de skill ou juste d'exposition au marché ?"*

> **Exemple.** Marché $+10\%$, $R_f = 2\%$, $\beta_p = 1.2$ → rendement CAPM attendu $= 2\% + 1.2 \times 8\% = 11.6\%$. Le fonds a fait $+16\%$ → $\alpha^{Jensen} = +4.4\%$ : vraie surperformance ajustée du risque.

### (iii) La notion d'horizon — point critique

L'alpha ex ante n'existe pas "dans l'absolu". Il existe toujours **pour un horizon $H$ donné** :

$$\alpha_{t,H}^{ex\ ante} = \hat{\mathbb{E}}_t[R_{t \to t+H}] - k_{t,H}$$

La **règle absolue** : $\hat{\mathbb{E}}[R_i]$ et $k$ doivent être définis sur le **même horizon**. Mélanger les horizons rend l'alpha sans signification.

| Méthode d'estimation de $\hat{\mathbb{E}}[R_i]$ | Horizon implicite | $k$ required à utiliser |
|---|---|---|
| Gordon-Shapiro ($D_1/P_0 + g$) | Long terme / annuel | CAPM annualisé |
| Target price analyst 12M | 12 mois | CAPM annualisé |
| Signal quant mensuel | 1 mois | CAPM $/ 12$ |
| Signal quant daily | 1 jour | CAPM $/ 252$ |

> **Cas bancal.** Gordon-Shapiro (rendement annuel long terme) $-$ required return mensuel du modèle factoriel : on compare des pommes et des oranges.

<details>
<summary>Tableau de lecture rapide β / α Jensen</summary>

| $\beta$ | $\alpha$ Jensen | Lecture |
|---|---|---|
| Élevé | Positif | Exposition agressive ET vrai stock-picking — le gérant idéal |
| Élevé | $\approx 0$ | A surfé le marché — un ETF à levier aurait suffi |
| Élevé | Négatif | Prise de risque max + mauvais stock-picking — le pire cas |
| Faible | Positif | Défensif et génère de l'alpha — rare et précieux |

</details>

---

## 5. Estimer $\hat{\mathbb{E}}[R_i]$ — les trois méthodes

### (i) Target price analyste

$$\hat{\mathbb{E}}[R_i] = \frac{TP + \text{Div} - P_0}{P_0}$$

Horizon implicite : 12 mois (convention des TP analyste). Rapide et scalable sur un univers large. Dépend de la qualité du consensus.

### (ii) Modèle de Gordon-Shapiro (1956)

Le cours d'une action égalise la valeur actuelle de tous ses dividendes futurs. Sous hypothèse de croissance constante $g$ des dividendes :

$$P_0 = \frac{D_1}{k - g} \implies \hat{\mathbb{E}}[R_i] = k = \frac{D_1}{P_0} + g$$

Le $k$ ainsi obtenu est le **rendement implicite dans le prix** : "compte tenu de mes hypothèses $(D_1, g)$ et du prix actuel $P_0$, quel rendement le marché semble-t-il anticiper ?"

**Conditions d'application.** Le modèle requiert $g < k$ et que l'entreprise verse des dividendes. Inapplicable aux *growth stocks* sans dividende (Amazon, early-stage tech).

> **Exemple — screener 4 actions** ($R_f = 4\%$, ERP $= 5\%$)
>
> | Action | $D_1/P_0 + g$ | $\beta$ | Requis CAPM | $\alpha^{ex\ ante}$ | Signal |
> |---|---|---|---|---|---|
> | Pernod Ricard | $9.6\%$ | $0.9$ | $8.5\%$ | $+1.1\%$ | Acheter |
> | Hermès | $8.7\%$ | $0.8$ | $8.0\%$ | $+0.7\%$ | Acheter |
> | Renault | $9.5\%$ | $1.1$ | $9.5\%$ | $\approx 0$ | Neutre |
> | EADS | $10.5\%$ | $1.4$ | $11.0\%$ | $-0.5\%$ | Éviter |

### (iii) DCF complet

Modélisation explicite des cash flows sur un horizon défini (5–10 ans) plus une valeur terminale. Précis mais long et non scalable sur un univers large.

---

## 6. Usages pratiques

### (i) Screener d'actifs

Calcule le rendement requis CAPM pour chaque action. Compare à $\hat{\mathbb{E}}[R_i]$ via l'une des trois méthodes ci-dessus. Trie par score $= \hat{\mathbb{E}}[R_i] - k$. Les scores positifs sont les candidats sous-évalués.

> Le CAPM seul ne dit rien — il faut impérativement une source externe pour $\hat{\mathbb{E}}[R_i]$.

### (ii) WACC et corporate finance

Le CAPM donne le coût des fonds propres :

$$k_e = R_f + \beta \cdot \text{ERP}$$

Ce $k_e$ entre dans le WACC, qui est le taux d'actualisation du DCF :

$$\text{WACC} = k_e \cdot \frac{E}{D+E} + k_d(1-t) \cdot \frac{D}{D+E}$$

C'est l'usage le plus répandu en dehors de la gestion — présent dans chaque dossier M\&A, valorisation d'entreprise, fairness opinion. Un projet cyclique ($\beta$ élevé) aura un hurdle rate plus élevé.

### (iii) Évaluation de performance

**Évaluer un fonds.** L'alpha de Jensen isole la surperformance résiduelle au-delà de ce que $\beta$ aurait mécaniquement produit.

> **Exemple — 3 gérants, même année.** Marché $+10\%$, $R_f = 2\%$, $\beta = 1.6$ pour tous → attendu CAPM $= 14.8\%$.
>
> | Gérant | Perf réalisée | $\alpha$ Jensen | Verdict |
> |---|---|---|---|
> | G1 | $+15\%$ | $\approx 0$ | A surfé le marché — ETF à levier suffisait |
> | G2 | $+19\%$ | $+4.2\%$ | Vrai stock-picking |
> | G3 | $+11\%$ | $-3.8\%$ | Exposition agressive + mauvais choix |

**Valider une stratégie quantitative.** La régression Jensen détermine si un signal génère de l'alpha ou charge simplement des titres high-beta.

| Stratégie | $\beta$ | $\alpha$ Jensen | $p$-value | Verdict |
|---|---|---|---|---|
| "Top croissance des ventes" | $1.5$ | $\approx 0$ | $> 0.05$ | Beta déguisé en alpha |
| "Top ROE/Price" | $0.9$ | $+3.2\%$ | $< 0.05$ | Signal validé |

### (iv) Construction de portefeuille et hedge

Le $\beta$ d'un portefeuille est la moyenne pondérée des $\beta$ individuels :

$$\beta_p = \sum_i w_i \beta_i$$

Pour neutraliser l'exposition marché (stratégie *market-neutral*), on shorte des futures indice à hauteur de $\beta_p \times \text{capital}$ : le stock-picking $(\varepsilon_i)$ est conservé, l'exposition systématique est annulée.

### (v) Market timing

Si l'on a une conviction sur la direction du marché, on ajuste $\beta_p$ :

- **Anticipation de hausse** → portefeuille agressif ($\beta > 1$) : surpondérer tech, cycliques.
- **Anticipation de baisse** → portefeuille défensif ($\beta < 1$) : surpondérer utilities, pharma, or.

> **Ajustement via futures.** Pour passer de $\beta_p = 1.2$ à $\beta_p = 0.8$ sur un portefeuille de 10M€ : shorter $0.4 \times 10\text{M} = 4\text{M}€$ de futures indice. Toutes les positions actions restent intactes.

---

## 7. Limitations

**$\beta$ instable dans le temps.** Estimé sur données historiques, le $\beta$ change avec les régimes de marché. Un $\beta = 1.5$ en période haussière peut devenir $2.5$ en crise.

**$\alpha$ et $\beta$ souvent non significatifs.** En pratique, beaucoup d'estimations ont une $p$-value $> 0.05$ — le modèle s'ajuste mal à certains actifs. Toujours vérifier la significativité statistique.

**Résidus non gaussiens.** Les rendements réels ont des queues épaisses (*fat tails*). Le test de Shapiro-Wilk rejette souvent la normalité des $\varepsilon_i$.

**Un seul facteur de risque.** Le marché n'explique pas tout. D'autres facteurs systématiques existent : taille, value, momentum. Ce sera l'objet du modèle de Fama-French.

**Critique de Roll (1977).** Le vrai portefeuille de marché est inobservable — il devrait inclure l'immobilier, le capital humain, les actifs privés. Le S\&P 500 utilisé comme proxy est imparfait, ce qui rend le modèle non testable au sens strict.

---

## Synthèse

Le CAPM repose sur une chaîne logique en trois étapes : Markowitz identifie le portefeuille tangent optimal, Tobin montre que tous les investisseurs le détiennent, et l'équilibre de marché implique que ce portefeuille *est* le marché. Il en découle que le seul risque rémunéré est la corrélation d'un actif avec le marché — son $\beta$.

$$\mathbb{E}[R_i] = R_f + \beta_i \cdot (\mathbb{E}[R_m] - R_f)$$

L'alpha ex ante (écart entre rendement estimé et requis, **pour un horizon donné**) est le carburant de toute gestion active. L'alpha de Jensen mesure ex post si un gérant ou une stratégie a créé de la valeur au-delà de son exposition marché.

> **La suite.** Années 1970 : Black, Jensen \& Scholes (1972) et Fama-MacBeth (1973) testent le modèle sur données réelles — les résultats sont mitigés. Roll (1977) pose la critique fondamentale sur l'inobservabilité du marché. Ces travaux mèneront à Fama-French (1992), qui ajoute les facteurs taille et value à l'équation de pricing.
