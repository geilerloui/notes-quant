# Fama-French 3 Facteurs

---

## 1. Le point de départ — l'échec empirique du CAPM

Le CAPM prédit une chose simple : plus le bêta d'un actif est élevé, plus son rendement espéré est élevé. FF testent ça sur 30 ans de données (1963–1990) et trouvent une relation plate — le bêta n'est pas rémunéré dans les données.

<div style="display: flex; gap: 1.5rem; margin: 1.5rem 0;">
  <div style="flex: 1; text-align: center;">
    <img src="../../images/fama-french/beta-capm-prediction.svg" alt="CAPM prediction" style="max-width: 100%;"/>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="../../images/fama-french/beta-actual-data.svg" alt="Actual data" style="max-width: 100%;"/>
  </div>
</div>

*Figure 1. Chaque point = un décile d'actions trié par bêta (1963–1990). À gauche : la prédiction du CAPM. À droite : les données réelles — pente non significativement différente de zéro.*

FF s'appuient alors sur la littérature des anomalies des années 1980 — une pléthore de papiers avait documenté que certaines caractéristiques des entreprises (taille, B/M, momentum...) prédisaient les rendements mieux que le bêta, sans que personne ne sache vraiment pourquoi. FF prennent les deux anomalies les plus robustes et les formalisent en un modèle cohérent.

---

## 2. Les deux variables explicatives

### (i) Rappel comptable — le bilan et les notations

![Balance sheet](../../images/fama-french/imZ.png)

*Figure 2. Balance sheet — rappel comptable des notations utilisées.*

Les notations utilisées dans tout ce qui suit :

- $K_0$ = Capitaux propres (book value of equity)
- $A_t$ = Valeur totale des actifs au temps $t$
- $D_t$ = Dettes totales au temps $t$
- $S_t$ = Prix de l'action au temps $t$
- $\theta_t$ = Nombre d'actions en circulation au temps $t$

L'identité comptable fondamentale :

$$A_t = K_0 + D_t \quad \Longrightarrow \quad K_0 = A_t - D_t$$

$K_0$ c'est ce qui appartient comptablement aux actionnaires — la valeur des actifs une fois toutes les dettes remboursées. Valeur **historique**, dans les livres.

> **Attention à l'ambiguïté française.** En comptabilité, "passif" désigne tout le côté droit du bilan (dettes + capitaux propres). Quand FF disent Book Value = Actif − Passif, ils veulent dire Actif − **dettes uniquement**, ce qui donne bien les capitaux propres $K_0$.

### (ii) La taille : $ME_t = S_t \times \theta_t$

$$ME_t = S_t \times \theta_t$$

C'est la capitalisation boursière — ce que le marché pense que les fonds propres valent aujourd'hui. Les petites capitalisations surperforment les grandes sur le long terme. C'est le *size premium*.

La coupure Small/Big se fait à la **médiane de $ME$ calculée sur les actions NYSE uniquement**. Si on utilisait tout le marché incluant AMEX et NASDAQ remplis de micro-caps, la médiane tomberait trop bas et la quasi-totalité des actions se retrouverait en "Big" — sans sens économique.

### (iii) Le ratio Book-to-Market : $B/M = K_0 / ME_t$

$$B/M = \frac{K_0}{ME_t} = \frac{A_t - D_t}{S_t \times \theta_t}$$

Le numérateur regarde en arrière (valeur comptable historique). Le dénominateur regarde en avant (ce que les investisseurs anticipent). L'écart entre les deux représente tout ce que le marché valorise au-delà des actifs tangibles : croissance future, marque, brevets non comptabilisés.

| B/M | Interprétation | Nom |
|-----|----------------|-----|
| Faible | Marché paye bien au-dessus de la valeur comptable | **Growth stock** (ex. Tesla, startups tech) |
| Élevé | Marché valorise proche de la valeur comptable | **Value stock** (entreprise décotée, souvent en difficulté) |

Les value stocks surperforment les growth stocks sur le long terme. C'est le *value premium*.

> **Note pratique.** Les professionnels utilisent P/B $= 1/(B/M)$. Un B/M de 0.5 = un P/B de 2.

---

## 3. La construction des facteurs

### (i) Un facteur = un portefeuille long-short

En ML, une feature est une observation passive — l'âge, le salaire. Elle existe dans le monde, tu la mesures. FF **construisent** leurs facteurs. SMB et HML sont des portefeuilles long-short — des stratégies actives qui génèrent un rendement chaque mois. Ce rendement mensuel, c'est le facteur.

![Long-short portfolio](../../images/fama-french/long-short-portfolio.svg)

*Figure 3. Construction du facteur SMB comme portefeuille long-short.*

Le short **annule** l'exposition au marché : si le marché monte de 2%, les longs montent de 2% et les shorts perdent 2% — les deux s'annulent. Ce qui reste dans le rendement du portefeuille, c'est **uniquement** la différence entre les deux groupes, purifiée du mouvement de marché.

### (ii) La grille 2×3 — le double tri

Chaque juillet, FF trient toutes les actions selon deux critères indépendants : médiane $ME$ → Small/Big, et 30ᵉ/70ᵉ percentiles $B/M$ → Growth/Neutral/Value. L'intersection donne 6 portefeuilles.

![Grille 2x3](../../images/fama-french/grid-2x3.svg)

*Figure 4. Les 6 portefeuilles issus du double tri taille × B/M.*

### (iii) Les formules de SMB et HML

$$SMB = \frac{1}{3}(R_{SV} + R_{SN} + R_{SG}) - \frac{1}{3}(R_{BV} + R_{BN} + R_{BG})$$

$$HML = \frac{1}{2}(R_{SV} + R_{BV}) - \frac{1}{2}(R_{SG} + R_{BG})$$

### (iv) Orthogonalité des facteurs

> 📌 SMB et HML sont quasi-orthogonaux (corrélation < 0.15 en pratique) grâce au double tri. En moyennant SMB sur les 3 lignes B/M, l'effet valeur se compense des deux côtés et disparaît — il reste uniquement de la taille pure. Symétriquement, HML moyenné sur Small et Big fait disparaître l'effet taille. Sans ce double tri, les deux facteurs seraient corrélés (les small caps sont naturellement plus souvent des value stocks) et la régression ne pourrait pas les distinguer l'un de l'autre — problème de multicolinéarité.

### (v) La timeline — rebalancement annuel

![Timeline rebalancement](../../images/fama-french/timeline-rebalancing.svg)

*Figure 5. Rebalancement annuel en juillet — les compositions sont fixes 12 mois.*

SMB et HML changent chaque mois — non pas parce que la composition des portefeuilles change, mais parce que les rendements réalisés des 6 portefeuilles changent chaque mois.

---

## 4. Le modèle et la régression

### (i) L'équation

$$R_i - R_f = \alpha_i + \beta_i(R_m - R_f) + s_i \cdot SMB + h_i \cdot HML + \varepsilon_i$$

### (ii) Interprétation des coefficients

| Coefficient | Valeur | Interprétation |
|-------------|--------|----------------|
| $\beta_i$ | $\approx 1$ | Exposition au risque de marché (hérité du CAPM) |
| $s_i > 0$ | | Covarie avec les small caps → porte du risque taille |
| $s_i < 0$ | | Se comporte comme une large cap |
| $h_i > 0$ | | Covarie avec les value stocks → porte du risque valeur |
| $h_i < 0$ | | Se comporte comme une growth stock |
| $\alpha_i$ | idéalement 0 | Surperformance non expliquée par les 3 facteurs |

### (iii) Structure matricielle — 25 régressions

Pour le portefeuille $i$, la régression s'écrit :

$$\underbrace{\begin{pmatrix} R_{i,1} - R_f \\ R_{i,2} - R_f \\ \vdots \\ R_{i,T} - R_f \end{pmatrix}}_{\mathbf{R}_i \ (T \times 1)} = \underbrace{\begin{pmatrix} 1 & R_{m,1}-R_f & SMB_1 & HML_1 \\ 1 & R_{m,2}-R_f & SMB_2 & HML_2 \\ \vdots & \vdots & \vdots & \vdots \\ 1 & R_{m,T}-R_f & SMB_T & HML_T \end{pmatrix}}_{\mathbf{X} \ (T \times 4) \ \text{— identique pour les 25}} \underbrace{\begin{pmatrix} \alpha_i \\ \beta_{m,i} \\ s_i \\ h_i \end{pmatrix}}_{\boldsymbol{\beta}_i \ (4 \times 1)} + \underbrace{\begin{pmatrix} \varepsilon_{i,1} \\ \varepsilon_{i,2} \\ \vdots \\ \varepsilon_{i,T} \end{pmatrix}}_{\boldsymbol{\varepsilon}_i \ (T \times 1)}$$

L'estimateur OLS : $\hat{\boldsymbol{\beta}}_i = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{R}_i$ pour chaque $i \in \{1, \ldots, 25\}$.

Ce qui change entre les 25 régressions : uniquement $\mathbf{R}_i$. La matrice $\mathbf{X}$ est rigoureusement identique pour tous.

---

## 5. Les résultats empiriques

### (i) Panel A — rendements moyens

![Panel A Summary Statistics](../../images/fama-french/imX.png)

*Figure 6. Panel A — moyenne et écart-type des rendements mensuels des 25 portefeuilles.*

Les rendements augmentent dans deux directions : de gauche à droite (Growth → Value) et de haut en bas (Big → Small). La valeur 0.31 pour Small Growth est la moyenne des rendements mensuels de ce portefeuille sur toute la période — le rendement le plus faible de toute la grille.

Le fait que les rendements bougent selon ces deux axes indépendants est exactement ce qui motive l'existence des deux facteurs.

### (ii) Panel B — coefficients de régression

![Panel B Regression Results](../../images/fama-french/imY.png)

*Figure 7. Panel B — coefficients $a$, $b$, $s$, $h$ et leurs t-stats pour les 25 portefeuilles.*

**Panel b (bêta de marché) :** tous les coefficients sont entre 0.9 et 1.1. Le bêta n'apporte aucune information discriminante entre les 25 portefeuilles — preuve visuelle de son insuffisance.

**Panel s (loadings SMB) :** $s$ décroît régulièrement de haut en bas (Small $\approx$ 1.2–1.5, Big $\approx$ −0.2), quelle que soit la colonne $B/M$. SMB capture la taille et rien d'autre. T-stats entre 30 et 65.

**Panel h (loadings HML) :** $h$ croît régulièrement de gauche à droite (Growth $\approx$ −0.3 à −0.5, Value $\approx$ +0.6 à +0.8), quelle que soit la ligne de taille. Les growth stocks ont un $h$ négatif — elles évoluent à l'opposé des value stocks.

**Panel a (alphas) :** le test ultime. 23 alphas sur 25 ne sont pas significatifs ($|t| < 1.96$). Deux exceptions : Small Growth ($\alpha = -0.45$, $t = -4.19$) et Big Growth ($\alpha = +0.20$, $t = 3.14$).

### (iii) Le F-test de Gibbons (GRS)

Le GRS teste si les 25 alphas sont **simultanément** tous nuls :

$$H_0 : \alpha_1 = \alpha_2 = \cdots = \alpha_{25} = 0$$

Il rejette — FF3 n'explique donc pas parfaitement tous les rendements. Ce résultat honnête que FF reconnaissent eux-mêmes motivera FF5 en 2015.

---

## 6. Interprétation et débat

### (i) "The anomalies largely disappear"

Dans le CAPM, une stratégie "acheter des small caps" génère un alpha positif — le modèle ne comprend pas ce rendement, c'est une anomalie. Dans FF3, cette même stratégie a juste un $s_i$ élevé. L'alpha tombe à zéro parce que le modèle reconnaît que l'investisseur porte du risque taille et le rémunère.

**Ce qui était une anomalie inexplicable devient une prime de risque légitime.**

Le bêta était un *proxy* — il captait indirectement les effets taille et valeur parce que les actions à bêta élevé ont tendance à être des small caps et des value stocks. Quand FF contrôlent pour SMB et HML, le bêta perd tout pouvoir explicatif résiduel.

### (ii) Risque ou mispricing ?

<details>
<summary>Interprétation risque (Fama)</summary>

Les primes SMB et HML rémunèrent des risques systématiques réels — petites entreprises plus fragiles en récession, value stocks souvent en difficulté financière. Les primes sont rationnelles et persistantes.

</details>

<details>
<summary>Interprétation comportementale (Thaler, Shiller)</summary>

Les investisseurs surpayent les growth stocks (belles histoires de croissance) et sous-paient les value stocks. Ce n'est pas du risque, c'est du mispricing durable. Fama n'a jamais réussi à identifier précisément quel choc macroéconomique justifie ces primes — ce qui nourrit ce camp.

</details>

---

## 7. Applications pratiques

### (i) Évaluer un gérant

Un gérant fait +14%/an contre +10% pour le marché. Régression FF3 : $s = +0.62$, $h = +0.41$, $\alpha = +0.08\%$/mois ($t = 0.9$, non significatif). Les +4% s'expliquent presque entièrement par du biais small cap et value — réplicable passivement avec un ETF pour quasiment zéro frais. Pas de vrai talent.

### (ii) Factor investing

Si tu crois que les primes sont persistantes, tu construis un portefeuille qui les capture explicitement par un tri systématique annuel. C'est ce que font les ETF smart beta et Dimensional Fund Advisors — pas de stock picking, juste du factor tilting discipliné.

### (iii) Risk attribution

$s$ élevé signifie que tu souffriras lors des crises de liquidité (2008, mars 2020) où les small caps chutent brutalement. FF3 te dit exactement d'où vient le risque de ton portefeuille — tu peux hedger ou assumer en connaissance de cause.

### (iv) La suite — FF5 (2015)

FF ajoutent deux facteurs supplémentaires : **RMW** (Robust Minus Weak — les entreprises très profitables surperforment) et **CMA** (Conservative Minus Aggressive — les entreprises qui investissent peu surperforment). Ces deux facteurs absorbent notamment les imperfections résiduelles de Small Growth et Big Growth identifiées dans FF3.

---

## Données

Ken French met à disposition gratuitement toutes les données sur son site à Dartmouth. Le fichier "Fama/French 3 Factors" contient quatre colonnes : `Mkt-RF`, `SMB`, `HML`, et `RF`, disponibles depuis 1926 en fréquence mensuelle ou quotidienne.

[mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
