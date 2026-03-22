# Fama-French 3 Facteurs

---

## 1. Le point de départ — l'échec empirique du CAPM

Le CAPM prédit une chose simple : plus le bêta d'un actif est élevé, plus son rendement espéré est élevé. FF testent ça sur 30 ans de données (1963–1990) et trouvent une relation plate — le bêta n'est pas rémunéré dans les données.

| | |
|:---:|:---:|
| ![CAPM prediction](images/facteurs-fondamentaux/beta-capm-prediction.svg) | ![Actual data](images/facteurs-fondamentaux/beta-actual-data.svg) |

*Figure 1. Chaque point = un décile d'actions trié par bêta (1963–1990). En haut : la prédiction du CAPM. En bas : les données réelles — pente non significativement différente de zéro.*

FF s'appuient alors sur la littérature des anomalies des années 1980 — une pléthore de papiers avait documenté que certaines caractéristiques des entreprises (taille, B/M, momentum...) prédisaient les rendements mieux que le bêta, sans que personne ne sache vraiment pourquoi. FF prennent les deux anomalies les plus robustes et les formalisent en un modèle cohérent.

---

## 2. Les deux variables explicatives

**(i) Rappel comptable — le bilan et les notations**

![Balance sheet](images/facteurs-fondamentaux/imZ.png)

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

**(ii) La taille : $ME_t = S_t \times \theta_t$**

$$ME_t = S_t \times \theta_t$$

C'est la capitalisation boursière — ce que le marché pense que les fonds propres valent aujourd'hui. Les petites capitalisations surperforment les grandes sur le long terme. C'est le *size premium*.

La coupure Small/Big se fait à la **médiane de $ME$ calculée sur les actions NYSE uniquement**. Si on utilisait tout le marché incluant AMEX et NASDAQ remplis de micro-caps, la médiane tomberait trop bas et la quasi-totalité des actions se retrouverait en "Big" — sans sens économique.

**(iii) Le ratio Book-to-Market : $B/M = K_0 / ME_t$**

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

**(i) Un facteur = un portefeuille long-short**

En ML, une feature est une observation passive — l'âge, le salaire. Elle existe dans le monde, tu la mesures. FF **construisent** leurs facteurs. SMB et HML sont des portefeuilles long-short — des stratégies actives qui génèrent un rendement chaque mois. Ce rendement mensuel, c'est le facteur.

![Long-short portfolio](images/facteurs-fondamentaux/long-short-portfolio.svg)

*Figure 3. Construction du facteur SMB comme portefeuille long-short.*

Le short **annule** l'exposition au marché : si le marché monte de 2%, les longs montent de 2% et les shorts perdent 2% — les deux s'annulent. Ce qui reste dans le rendement du portefeuille, c'est **uniquement** la différence entre les deux groupes, purifiée du mouvement de marché.

**(ii) La grille 2×3 — le double tri**

Chaque juillet, FF trient toutes les actions selon deux critères indépendants : médiane $ME$ → Small/Big, et 30ᵉ/70ᵉ percentiles $B/M$ → Growth/Neutral/Value. L'intersection donne 6 portefeuilles.

![Grille 2x3](images/facteurs-fondamentaux/grid-2x3.svg)

*Figure 4. Les 6 portefeuilles issus du double tri taille × B/M.*

**(iii) Les formules de SMB et HML**

$$SMB = \frac{1}{3}(R_{SV} + R_{SN} + R_{SG}) - \frac{1}{3}(R_{BV} + R_{BN} + R_{BG})$$

$$HML = \frac{1}{2}(R_{SV} + R_{BV}) - \frac{1}{2}(R_{SG} + R_{BG})$$

**(iv) Orthogonalité des facteurs**

> 📌 SMB et HML sont quasi-orthogonaux (corrélation < 0.15 en pratique) grâce au double tri. En moyennant SMB sur les 3 lignes B/M, l'effet valeur se compense des deux côtés et disparaît — il reste uniquement de la taille pure. Symétriquement, HML moyenné sur Small et Big fait disparaître l'effet taille. Sans ce double tri, les deux facteurs seraient corrélés (les small caps sont naturellement plus souvent des value stocks) et la régression ne pourrait pas les distinguer l'un de l'autre — problème de multicolinéarité.

**(v) La timeline — rebalancement annuel**

![Timeline rebalancement](images/facteurs-fondamentaux/timeline-rebalancing.svg)

*Figure 5. Rebalancement annuel en juillet — les compositions sont fixes 12 mois.*

SMB et HML changent chaque mois — non pas parce que la composition des portefeuilles change, mais parce que les rendements réalisés des 6 portefeuilles changent chaque mois.

---

## 4. Le modèle et la régression

**(i) L'équation**

$$R_i - R_f = \alpha_i + \beta_i(R_m - R_f) + s_i \cdot SMB + h_i \cdot HML + \varepsilon_i$$

**(ii) Interprétation des coefficients**

| Coefficient | Valeur | Interprétation |
|-------------|--------|----------------|
| $\beta_i$ | $\approx 1$ | Exposition au risque de marché (hérité du CAPM) |
| $s_i > 0$ | | Covarie avec les small caps → porte du risque taille |
| $s_i < 0$ | | Se comporte comme une large cap |
| $h_i > 0$ | | Covarie avec les value stocks → porte du risque valeur |
| $h_i < 0$ | | Se comporte comme une growth stock |
| $\alpha_i$ | idéalement 0 | Surperformance non expliquée par les 3 facteurs |

**(iii) Structure matricielle — 25 régressions**

Pour le portefeuille $i$, la régression s'écrit :

$$\underbrace{\begin{pmatrix} R_{i,1} - R_f \\ R_{i,2} - R_f \\ \vdots \\ R_{i,T} - R_f \end{pmatrix}}_{\mathbf{R}_i \ (T \times 1)} = \underbrace{\begin{pmatrix} 1 & R_{m,1}-R_f & SMB_1 & HML_1 \\ 1 & R_{m,2}-R_f & SMB_2 & HML_2 \\ \vdots & \vdots & \vdots & \vdots \\ 1 & R_{m,T}-R_f & SMB_T & HML_T \end{pmatrix}}_{\mathbf{X} \ (T \times 4) \ \text{— identique pour les 25}} \underbrace{\begin{pmatrix} \alpha_i \\ \beta_{m,i} \\ s_i \\ h_i \end{pmatrix}}_{\boldsymbol{\beta}_i \ (4 \times 1)} + \underbrace{\begin{pmatrix} \varepsilon_{i,1} \\ \varepsilon_{i,2} \\ \vdots \\ \varepsilon_{i,T} \end{pmatrix}}_{\boldsymbol{\varepsilon}_i \ (T \times 1)}$$

L'estimateur OLS : $\hat{\boldsymbol{\beta}}_i = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{R}_i$ pour chaque $i \in \{1, \ldots, 25\}$.

Ce qui change entre les 25 régressions : uniquement $\mathbf{R}_i$. La matrice $\mathbf{X}$ est rigoureusement identique pour tous.

---

## 5. Les résultats empiriques

**(i) Panel A — rendements moyens**

![Panel A Summary Statistics](images/facteurs-fondamentaux/imX.png)

*Figure 6. Panel A — moyenne et écart-type des rendements mensuels des 25 portefeuilles.*

Les rendements augmentent dans deux directions : de gauche à droite (Growth → Value) et de haut en bas (Big → Small). La valeur 0.31 pour Small Growth est la moyenne des rendements mensuels de ce portefeuille sur toute la période — le rendement le plus faible de toute la grille.

Le fait que les rendements bougent selon ces deux axes indépendants est exactement ce qui motive l'existence des deux facteurs.

**(ii) Panel B — coefficients de régression**

![Panel B Regression Results](images/facteurs-fondamentaux/imY.png)

*Figure 7. Panel B — coefficients $a$, $b$, $s$, $h$ et leurs t-stats pour les 25 portefeuilles.*

**Panel b (bêta de marché) :** tous les coefficients sont entre 0.9 et 1.1. Le bêta n'apporte aucune information discriminante entre les 25 portefeuilles — preuve visuelle de son insuffisance.

**Panel s (loadings SMB) :** $s$ décroît régulièrement de haut en bas (Small $\approx$ 1.2–1.5, Big $\approx$ −0.2), quelle que soit la colonne $B/M$. SMB capture la taille et rien d'autre. T-stats entre 30 et 65.

**Panel h (loadings HML) :** $h$ croît régulièrement de gauche à droite (Growth $\approx$ −0.3 à −0.5, Value $\approx$ +0.6 à +0.8), quelle que soit la ligne de taille. Les growth stocks ont un $h$ négatif — elles évoluent à l'opposé des value stocks.

**Panel a (alphas) :** le test ultime. 23 alphas sur 25 ne sont pas significatifs ($|t| < 1.96$). Deux exceptions : Small Growth ($\alpha = -0.45$, $t = -4.19$) et Big Growth ($\alpha = +0.20$, $t = 3.14$).

**(iii) Le F-test de Gibbons (GRS)**

Le GRS teste si les 25 alphas sont **simultanément** tous nuls :

$$H_0 : \alpha_1 = \alpha_2 = \cdots = \alpha_{25} = 0$$

Il rejette — FF3 n'explique donc pas parfaitement tous les rendements. Ce résultat honnête que FF reconnaissent eux-mêmes motivera FF5 en 2015.

---

## 6. Interprétation et débat

**(i) "The anomalies largely disappear"**

Dans le CAPM, une stratégie "acheter des small caps" génère un alpha positif — le modèle ne comprend pas ce rendement, c'est une anomalie. Dans FF3, cette même stratégie a juste un $s_i$ élevé. L'alpha tombe à zéro parce que le modèle reconnaît que l'investisseur porte du risque taille et le rémunère.

**Ce qui était une anomalie inexplicable devient une prime de risque légitime.**

Le bêta était un *proxy* — il captait indirectement les effets taille et valeur parce que les actions à bêta élevé ont tendance à être des small caps et des value stocks. Quand FF contrôlent pour SMB et HML, le bêta perd tout pouvoir explicatif résiduel.

**(ii) Risque ou mispricing ?**

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

**(i) Évaluer un gérant**

Un gérant fait +14%/an contre +10% pour le marché. Régression FF3 : $s = +0.62$, $h = +0.41$, $\alpha = +0.08\%$/mois ($t = 0.9$, non significatif). Les +4% s'expliquent presque entièrement par du biais small cap et value — réplicable passivement avec un ETF pour quasiment zéro frais. Pas de vrai talent.

**(ii) Factor investing**

Si tu crois que les primes sont persistantes, tu construis un portefeuille qui les capture explicitement par un tri systématique annuel. C'est ce que font les ETF smart beta et Dimensional Fund Advisors — pas de stock picking, juste du factor tilting discipliné.

**(iii) Risk attribution**

$s$ élevé signifie que tu souffriras lors des crises de liquidité (2008, mars 2020) où les small caps chutent brutalement. FF3 te dit exactement d'où vient le risque de ton portefeuille — tu peux hedger ou assumer en connaissance de cause.

**(iv) La suite — FF5 (2015)**

FF ajoutent deux facteurs supplémentaires : **RMW** (Robust Minus Weak — les entreprises très profitables surperforment) et **CMA** (Conservative Minus Aggressive — les entreprises qui investissent peu surperforment). Ces deux facteurs absorbent notamment les imperfections résiduelles de Small Growth et Big Growth identifiées dans FF3.

---

## Données

Ken French met à disposition gratuitement toutes les données sur son site à Dartmouth. Le fichier "Fama/French 3 Factors" contient quatre colonnes : `Mkt-RF`, `SMB`, `HML`, et `RF`, disponibles depuis 1926 en fréquence mensuelle ou quotidienne.

[mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)



# Novy Marx
## (i) Le point de départ : une anomalie dans les données

Fama et French ont montré dans les années 90 que les value stocks (ratio $B/M$ élevé) surperforment le marché — c'est le facteur HML. Novy Marx observe quelque chose en plus dans les données : des entreprises très profitables surperforment aussi, indépendamment de leur $B/M$. Ce n'est pas expliqué par Fama-French. Sa question : quelle mesure de profitabilité capture le mieux cette prime ?

> **Rappel — le ratio $B/M$.** $B$ c'est le book value, c'est-à-dire les capitaux propres comptables (actifs $-$ dettes). $M$ c'est la valeur de marché (prix de l'action $\times$ nombre d'actions). Un $B/M$ élevé veut dire que le marché valorise peu l'entreprise par rapport à ce qu'elle vaut sur le papier — c'est une value stock, considérée comme "bon marché". Un $B/M$ faible veut dire que le marché valorise beaucoup l'entreprise au-delà de sa valeur comptable — c'est une growth stock, considérée comme "chère".

---

## (ii) La justification théorique

Il part de la formule de valorisation de base — une action vaut ses dividendes futurs actualisés :

$$S_t = \sum_{\tau=1}^{\infty} \frac{D_{t+\tau}}{(1+r)^\tau}$$

Problème : les dividendes sont un mauvais signal. Beaucoup d'entreprises n'en versent pas (Amazon pendant 20 ans), et ils peuvent être manipulés indépendamment de la performance réelle. Il réécrit la formule avec l'identité comptable du **clean surplus** :

$$B_t = B_{t-1} + E_t - D_t$$

Ce qui dit : capitaux propres comptables cette année $=$ capitaux propres l'an dernier $+$ résultat net $-$ dividendes versés. On isole $D_t$ :

$$D_t = E_t - (B_t - B_{t-1}) = E_t - dB_t$$

On substitue dans la formule de prix :

$$S_t = \sum_{\tau=1}^{\infty} \frac{\mathbb{E}_t[E_{t+\tau} - dB_{t+\tau}]}{(1+r)^\tau}$$

Cette formule dit : **le prix d'une action aujourd'hui dépend des bénéfices futurs attendus**. Donc si tu trouves un bon prédicteur des bénéfices futurs $E$, tu identifies les actions sous-évaluées.

> **Ce que veut dire l'espérance.** On est en $t$, on ne connaît pas les bénéfices futurs. $\mathbb{E}_t[\cdot]$ c'est "mon estimation aujourd'hui de ce que ça sera demain", basée sur toute l'information disponible maintenant.

> **Attention à la notation $D_t$.** Dans le clean surplus, $D_t$ désigne les dividendes versés. Dans un bilan comptable, $D$ désigne souvent les dettes. Ce sont deux usages différents de la même lettre — ici on est dans la logique du clean surplus donc $D_t =$ dividendes partout dans cette section.

**Exemple numérique.** Supposons une action qui cote 10 dollars sur le marché. Tu estimes ses bénéfices futurs et tu appliques la formule — tu obtiens une valeur fondamentale de 13 dollars. Le marché sous-évalue cette action : il ne voit pas encore que ses bénéfices futurs justifient un prix de 13 dollars. Elle surperformera quand le marché corrigera son erreur. À l'inverse, une valeur fondamentale de 8 dollars pour une action cotée à 10 dollars signifie qu'elle est surévaluée.

Tout le papier revient à trouver le meilleur moyen d'estimer les bénéfices futurs aujourd'hui pour repérer ces écarts.

---

## (iii) Le choix du proxy : pourquoi GP/A ?

La formule parle de bénéfices futurs $E$ mais ne dit pas comment les mesurer aujourd'hui. L'argument central : le compte de résultat est une cascade, et chaque étage rajoute des décisions comptables qui bruitent le signal.

![Cascade du compte de résultat — chaque étage rajoute du bruit comptable.](images/facteurs-fondamentaux/novy_cascade_compte_resultat.png)

*Figure 1. La cascade du compte de résultat. Plus on descend, plus on accumule de décisions comptables discrétionnaires.*

Le bénéfice net accumule amortissements, provisions, impôts différés — deux boîtes identiques économiquement peuvent avoir des résultats nets très différents selon leurs choix comptables. Le FCF varie selon les décisions d'investissement — une boîte qui construit une usine a un FCF négatif sans être moins profitable. La marge brute (Sales $-$ COGS) est le signal le plus proche de la réalité économique brute.

Il définit donc :

$$\text{GP/A} = \frac{\text{Sales} - \text{COGS}}{\text{Assets}}$$

> **Les trois candidats en données concrètes.** Pour chaque action $i$, on calcule :
> - $\text{IB/A}$ = résultat net / actifs totaux
> - $\text{FCF/A}$ = (résultat net + amortissements $-$ variation BFR $-$ capex) / actifs totaux
> - $\text{GP/A}$ = (Sales $-$ COGS) / actifs totaux
>
> On divise toujours par les actifs totaux pour normaliser et comparer des entreprises de tailles différentes. Une grande boîte a forcément une grosse marge brute en valeur absolue — diviser par les actifs retire cet effet de taille.

---

## (iv) La preuve empirique — Table 1 (Fama-McBeth)

Il estime chaque mois la régression suivante sur toutes les actions du marché américain :

$$r_{i,t} = \alpha + \beta_1 \cdot \text{GP/A}_i + \beta_2 \cdot \text{IB/A}_i + \beta_3 \cdot \text{FCF/A}_i + \beta_4 \cdot \log(B/M)_i + \beta_5 \cdot \log(ME)_i + \varepsilon_{i,t}$$

Il moyenne ensuite les coefficients sur toutes les périodes. La t-stat teste si la moyenne de $\hat{\beta}$ est significativement différente de zéro — au-dessus de 2 c'est significatif. Les coefficients dans la table sont multipliés par $10^2$.

> **Ce que cette régression fait concrètement.** Il n'y a pas de portefeuille ici — c'est une régression sur les actions individuelles. On prend toutes les actions du marché américain (plusieurs milliers), on régresse leurs rendements mensuels sur leurs caractéristiques fondamentales, et on regarde quelles caractéristiques ont un $\hat{\beta}$ significativement différent de zéro. C'est un test statistique pur pour identifier quelles variables prédisent les rendements en coupe transversale.

![Table 1 — Régressions Fama-McBeth. 7 spécifications différentes, une par colonne.](images/facteurs-fondamentaux/novy_fama_mcbeth.png)

*Figure 2. Fama-McBeth regression. Encadrés rouges : une seule mesure testée seule. Encadrés bleus : GP/A testé conjointement avec une autre mesure.*

La table présente **7 régressions différentes**, une par colonne.

**Encadrés rouges — une seule mesure à la fois :**
- Colonne (1) : GP/A seul $\rightarrow$ $t = 5.49$. Très significatif.
- Colonne (2) : bénéfice net seul $\rightarrow$ $t = 0.84$. Non significatif.
- Colonne (3) : FCF seul $\rightarrow$ $t = 2.28$. Significatif.

Le FCF seul est effectivement significatif — il prédit aussi les rendements en isolation. Mais ce n'est pas ça que Novy Marx veut montrer. Il veut savoir lequel est le meilleur signal.

**Encadrés bleus — deux mesures ensemble :**
- Colonne (4) : GP/A + bénéfice net $\rightarrow$ GP/A reste à $t = 5.22$, bénéfice net tombe à $t = 0.31$.
- Colonne (5) : GP/A + FCF $\rightarrow$ GP/A reste à $t = 4.63$, FCF tombe à $t = 1.64$.

Résultat clé : quand GP/A est dans la régression, les autres mesures perdent leur pouvoir prédictif. GP/A **absorbe** leur information — le FCF semblait marcher seul uniquement parce qu'il était corrélé avec GP/A.

> **Sur $r_{1,0}$ et $r_{12,2}$.** Ce sont des variables de contrôle pour le momentum. $r_{1,0}$ c'est le rendement du mois précédent, son coefficient négatif ($-5.57$, $t = -13.8$) capte l'effet de retournement à court terme. $r_{12,2}$ c'est le rendement sur les 12 derniers mois hors le dernier mois, son coefficient positif ($0.76$, $t = 3.87$) capte le momentum classique. Novy Marx les inclut pour s'assurer que l'effet GP/A n'est pas du momentum déguisé.

---

## (v) La découverte bonus — Table 2 et corrélations

Ici Novy Marx change d'approche : il **crée de vrais portefeuilles**. Il trie toutes les actions en 5 groupes par GP/A. Les 20% les moins profitables vont dans le portefeuille "Low", les 20% les plus profitables dans "High". Il calcule le rendement moyen de chaque groupe puis régresse ces rendements sur les trois facteurs Fama-French :

$$r_{i,t} - r_f = \alpha_i + \beta_i^{MKT} \cdot MKT_t + \beta_i^{SMB} \cdot SMB_t + \beta_i^{HML} \cdot HML_t + \varepsilon_{i,t}$$

> **Ce que sont MKT, SMB et HML.** Ce sont les trois facteurs de Fama-French, eux-mêmes construits comme des portefeuilles long/short :
> - $MKT$ = rendement du marché $-$ taux sans risque.
> - $SMB$ (Small Minus Big) = rendement des petites caps $-$ rendement des grandes caps.
> - $HML$ (High Minus Low) = rendement des value stocks (B/M élevé) $-$ rendement des growth stocks (B/M faible).
>
> Le coefficient sur HML indique à quel point un portefeuille ressemble à des value stocks (positif) ou à des growth stocks (négatif).

![Table 2 — Portefeuilles triés par GP/A (Panel A) et par B/M (Panel B).](images/facteurs-fondamentaux/novy_linear_regression.png)

*Figure 3. Portefeuilles triés par profitabilité (Panel A) et par B/M (Panel B). Les rendements augmentent dans les deux cas de Low à High. Mais les coefficients HML vont dans des directions opposées.*

Les rendements augmentent de Low GP/A ($0.31\%$ par mois) à High GP/A ($0.62\%$). Mais le résultat surprenant c'est le coefficient sur HML :

- Portefeuille Low GP/A : $\beta^{HML} = +0.15$
- Portefeuille High GP/A : $\beta^{HML} = -0.29$

Les entreprises très profitables se comportent comme des growth stocks, à l'opposé des value stocks qui ont $\beta^{HML} = +0.51$. C'est une observation visuelle qui donne l'intuition. Pour le vérifier formellement, il construit la matrice de données suivante :

![Matrice de données : n actions × d caractéristiques fondamentales.](images/facteurs-fondamentaux/novy_matrice_donnees.png)

*Figure 4. Structure des données utilisées pour calculer la table de corrélation. Chaque ligne est une action, chaque colonne une caractéristique fondamentale. La colonne B/M est en rouge car sa corrélation avec GP/A est négative.*

À partir de cette matrice il calcule la corrélation de Pearson entre chaque paire de colonnes.

![Figure 18 — Table de corrélation entre les caractéristiques fondamentales.](images/facteurs-fondamentaux/novy_correlation.png)

*Figure 5. Table de corrélation (Figure 18 du papier). La ligne GP/A est encadrée en rouge. Le chiffre clé : corrélation GP/A vs B/M $= -0.18$, $t = -17.2$.*

Chiffres clés sur la ligne GP/A :
- GP/A vs IB/A : $+0.45$ — corrélation positive, logique, les deux mesurent la profitabilité.
- GP/A vs FCF/A : $+0.31$ — idem.
- GP/A vs B/M : $\mathbf{-0.18}$, $t = -17.2$ — les actions avec GP/A élevé ont systématiquement un B/M faible.

> **Ce que cette corrélation prouve.** La stratégie value achète les actions avec B/M élevé. La stratégie GP/A achète les actions avec GP/A élevé. Puisque GP/A et B/M sont négativement corrélés, les deux stratégies misent sur des entreprises opposées. Quand la stratégie value performe bien, la stratégie GP/A performe moins bien. D'où une corrélation négative entre leurs rendements : $\rho \approx -0.5$ dans les données.

---

## (vi) La conclusion : le portefeuille mixte

Novy Marx construit le portefeuille final : $50\%$ long sur le facteur GP/A (portefeuille High $-$ Low GP/A) et $50\%$ long sur le facteur value (portefeuille High $-$ Low B/M). Comme les deux sont anticorrélés, la variance du portefeuille combiné est plus faible.

Le rendement attendu :

$$\mathbb{E}[R_p] = 0.5 \times \mathbb{E}[R^1] + 0.5 \times \mathbb{E}[R^2] = 0.5\%$$

Inchangé — c'est la moyenne des deux. La variance :

$$\mathbb{V}[R_p] = x_1^2 \sigma_1^2 + x_2^2 \sigma_2^2 + 2 x_1 x_2 \cdot \text{cov}(R^1, R^2)$$

Avec $\sigma_1 = \sigma_2 = 3\%$ et $\text{cov} = \rho \cdot \sigma_1 \cdot \sigma_2 = -0.5 \times 0.03 \times 0.03 = -0.00045$ :

$$\mathbb{V}[R_p] = 0.25 \times 0.0009 + 0.25 \times 0.0009 + 2 \times 0.25 \times (-0.00045)$$

$$= 0.000225 + 0.000225 - 0.000225 = 0.000225$$

$$\sigma_p = \sqrt{0.000225} = 1.5\%$$

La volatilité est divisée par deux pour le même rendement. Le Sharpe ratio :

$$\text{Sharpe seul} = \frac{0.5\%}{3\%} = 0.17 \qquad \text{Sharpe mixte} = \frac{0.5\%}{1.5\%} = 0.33$$

Le Sharpe double.

![Figure 17 — Performance du portefeuille mixte de 1963 à 2010.](images/facteurs-fondamentaux/novy_hedging.png)

*Figure 6. Sharpe ratio annualisé glissant de 1963 à 2010. Le mix 50/50 (trait plein) est systématiquement plus stable que la stratégie value seule (pointillés) et la stratégie profitabilité seule (tirets). La stratégie value s'effondre en 2000 pendant la bulle tech — la profitabilité compense.*

> **Pourquoi ces mispricing existent-ils ?** Lakonishok et al. (1994) montrent que le marché sur-paie systématiquement les growth stocks (médiatisées, excitantes) et sous-paie les value stocks (ennuyeuses, mal-aimées). Novy Marx ajoute : le marché rate aussi le signal GP/A — il ne voit pas que les entreprises très profitables vont continuer à générer des bénéfices élevés. La stratégie exploite ces deux inefficiences comportementales simultanément, avec un risque réduit grâce à leur anticorrélation naturelle.

La stratégie concrète : **acheter des value stocks profitables, vendre des growth stocks peu profitables**.



# Fama-French 5 Facteurs

---

## 1. Le point de départ — les limites du FF3

Le GRS test du FF3 rejette l'hypothèse que les 25 alphas sont simultanément nuls. FF reconnaissent eux-mêmes que leur modèle est incomplet. Deux anomalies persistent :

- Les entreprises très profitables surperforment, indépendamment de leur $B/M$ — c'est ce que Novy-Marx (2013) a formalisé avec $GP/A$.
- Les entreprises qui investissent peu surperforment celles qui investissent beaucoup.

Ces deux anomalies ne sont pas des accidents dans les données. Elles ont une justification théorique directe dans le modèle de valorisation de base. C'est ce que FF (2015) exploitent pour construire deux nouveaux facteurs.

---

## 2. Le fondement théorique — le DDM réécrit

FF partent de la même identité que Novy-Marx. En substituant le clean surplus $D_t = E_t - dB_t$ dans la formule de valorisation :

$$M_t = \sum_{\tau=1}^{\infty} \frac{\mathbb{E}(E_{t+\tau} - dB_{t+\tau})}{(1+r)^\tau}$$

On divise des deux côtés par $B_t$ :

$$\frac{M_t}{B_t} = \frac{\sum_{\tau=1}^{\infty} \mathbb{E}(E_{t+\tau} - dB_{t+\tau}) / (1+r)^\tau}{B_t}$$

Cette équation est une identité comptable — elle est vraie par construction. Elle relie le ratio market-to-book observé aujourd'hui aux bénéfices futurs attendus et au taux de rendement requis $r$. C'est de cette identité que découlent les trois effets empiriques du FF5.

**Effet profitabilité.** Fixe $M_t/B_t$. Si $E_{t+\tau}$ augmente, le numérateur augmente. Pour que l'égalité tienne, $(1+r)^\tau$ doit augmenter aussi — donc $r \uparrow$. Les actions d'entreprises très profitables ont empiriquement des rendements moyens plus élevés.

**Effet investissement.** $dB_{t+\tau}$ est soustrait dans le numérateur. Si l'investissement augmente, le numérateur diminue. Pour compenser, $r$ doit baisser — donc $r \downarrow$. Les entreprises qui investissent agressivement ont des rendements plus faibles.

**Effet valeur.** Si $B_t$ est grand relativement à $M_t$, autrement dit $B/M \uparrow$, le dénominateur est petit. Pour que l'égalité tienne, $r \uparrow$. C'est le value premium du FF3, vu sous le même angle.

> **Ce que veut dire $r$ ici.** $r$ n'est pas le rendement de l'entreprise — c'est le taux de rendement requis par les investisseurs sur cette action. Une entreprise très profitable est perçue comme plus attractive, les investisseurs l'achètent, et son rendement moyen observé historiquement est plus élevé. C'est une relation empirique capturée par le facteur, pas une causalité directe.

---

## 3. Les deux nouvelles variables explicatives

**(i) Rappel comptable**

On reprend les notations du bilan introduites pour le FF3. L'identité fondamentale reste :

$$A_t = K_0 + D_t \quad \Longrightarrow \quad K_0 = A_t - D_t$$

où $K_0$ sont les capitaux propres comptables (book equity), $A_t$ les actifs totaux et $D_t$ les dettes totales.

**(ii) La profitabilité opérationnelle : $OP$**

FF mesurent la profitabilité par le ratio suivant :

$$OP = \frac{\text{Revenues} - \text{COGS} - \text{SG\&A} - \text{Interest}}{\text{Book equity}}$$

C'est une mesure proche du $GP/A$ de Novy-Marx, mais normalisée par les capitaux propres comptables plutôt que par les actifs totaux, et avec des charges supplémentaires soustraites. L'intuition reste la même : plus ce ratio est élevé, plus l'entreprise génère de valeur à partir de ses ressources.

**(iii) L'investissement : $INV$**

FF mesurent l'investissement par la croissance des actifs totaux d'une année à l'autre :

$$INV = \frac{A_t - A_{t-1}}{A_{t-1}}$$

Ce ratio capture l'agressivité de l'investissement. Une entreprise qui double ses actifs en un an ($INV$ élevé) dilue mécaniquement le numérateur du DDM — les cash flows futurs sont répartis sur une base d'actifs plus grande. Le marché anticipe un rendement plus faible.

> **Attention au look-ahead.** $INV$ est calculé à la fin de chaque exercice fiscal, avec un décalage de six mois avant d'être utilisé dans les portefeuilles. Ce décalage garantit que l'information comptable est publiquement disponible au moment du tri.

---

## 4. La construction des facteurs — le triple tri

**(i) De la grille 2×3 au triple tri indépendant**

Dans le FF3, les portefeuilles sont construits par un double tri : taille × $B/M$. Pour le FF5, FF veulent isoler l'effet de $OP$ et de $INV$ indépendamment de la taille. La solution : un triple tri indépendant.

Chaque juillet, FF trient toutes les actions selon trois critères **indépendants** :

- **Taille** : médiane $ME$ NYSE → Small / Big
- **$B/M$** : 30e/70e percentiles NYSE → Low / Neutral / High
- **$OP$ ou $INV$** : 30e/70e percentiles NYSE → Weak / Neutral / Robust (pour $OP$) ou Conservative / Neutral / Aggressive (pour $INV$)

L'intersection donne $2 \times 3 \times 3 = 18$ portefeuilles pour chaque dimension supplémentaire.

> **Pourquoi des tris indépendants ?** Si on triait séquentiellement — d'abord par taille, puis par $OP$ à l'intérieur de chaque groupe de taille — les portefeuilles résultants auraient des tailles très inégales. Les tris indépendants garantissent que chaque critère est appliqué sur l'ensemble du marché, sans que la coupure $OP$ ne soit conditionnée par la coupure taille. C'est la même logique qui rendait SMB et HML quasi-orthogonaux dans le FF3 : en moyennant sur les autres dimensions, on purge chaque facteur des effets croisés.

**(ii) La formule de RMW**

RMW (Robust Minus Weak) est construit à partir des 6 portefeuilles issus du tri taille × $OP$ :

$$RMW = \frac{1}{2}(R_{SR} + R_{BR}) - \frac{1}{2}(R_{SW} + R_{BW})$$

où $SR$ = Small Robust, $BR$ = Big Robust, $SW$ = Small Weak, $BW$ = Big Weak. On moyenne sur Small et Big pour que RMW capture uniquement l'effet profitabilité, purgé de l'effet taille.

**(iii) La formule de CMA**

CMA (Conservative Minus Aggressive) est construit à partir des 6 portefeuilles issus du tri taille × $INV$ :

$$CMA = \frac{1}{2}(R_{SC} + R_{BC}) - \frac{1}{2}(R_{SA} + R_{BA})$$

où $SC$ = Small Conservative, $BC$ = Big Conservative, $SA$ = Small Aggressive, $BA$ = Big Aggressive.

**(iv) Le rebalancement**

Les 18 portefeuilles sont rebalancés chaque juillet, avec les données comptables de l'exercice fiscal se terminant en décembre de l'année précédente. Le décalage de six mois garantit l'absence de look-ahead bias — exactement comme dans le FF3.

---

## 5. Le modèle FF5

$$R_i - R_f = \alpha_i + \beta_i(R_m - R_f) + s_i \cdot SMB + h_i \cdot HML + r_i \cdot RMW + c_i \cdot CMA + \varepsilon_i$$

| Coefficient | Signe positif | Signe négatif |
|-------------|---------------|---------------|
| $\beta_i$ | Exposition au risque de marché | — |
| $s_i$ | Se comporte comme une small cap | Se comporte comme une large cap |
| $h_i$ | Covarie avec les value stocks | Covarie avec les growth stocks |
| $r_i$ | Covarie avec les entreprises robustes | Covarie avec les entreprises fragiles |
| $c_i$ | Covarie avec les firmes conservatrices | Covarie avec les firmes agressives |
| $\alpha_i$ | Surperformance non expliquée | Sous-performance non expliquée |

---

## 6. La Table 1 — validation empirique des patterns univariés

**(i) Ce que prédit le DDM**

Rappelons l'identité :

$$\frac{M_t}{B_t} = \frac{\sum_{\tau=1}^{\infty} \mathbb{E}(E_{t+\tau} - dB_{t+\tau}) / (1+r)^\tau}{B_t}$$

En fixant $M_t/B_t$, les trois prédictions sont :

$$B/M \uparrow \;\Rightarrow\; r \uparrow \qquad OP \uparrow \;\Rightarrow\; r \uparrow \qquad INV \uparrow \;\Rightarrow\; r \downarrow$$

**(ii) Ce que montrent les données — les microcaps**

La Table 1 présente les rendements moyens mensuels des portefeuilles triés par taille × $B/M$ (Panel A), taille × $OP$ (Panel B), et taille × $INV$ (Panel C). Pour les microcaps (ligne Small), les patterns sont nets.

![Table 1 — rendements moyens des portefeuilles univariés](images/facteurs-fondamentaux/ff5_table1.png)

*Figure 8. Table 1 — rendements moyens mensuels des portefeuilles taille × B/M (Panel A), taille × OP (Panel B), taille × INV (Panel C). En bleu : patterns conformes à la théorie DDM. En rouge : anomalies pour les megacaps.*

**Panel A — $B/M \uparrow \Rightarrow r \uparrow$**

| | Low | 2 | 3 | 4 | High |
|--|-----|---|---|---|------|
| Small | 0.26 | 0.81 | 0.85 | 1.01 | 1.15 |
| Big | 0.46 | 0.51 | 0.48 | 0.56 | 0.62 |

Le rendement mensuel moyen passe de $0.26\%$ à $1.15\%$ quand $B/M$ augmente de gauche à droite pour les microcaps. C'est exactement ce que prédit le DDM : les value stocks ($B/M$ élevé) ont un $r$ plus élevé. Pour les megacaps, la progression existe mais est très atténuée.

**Panel B — $OP \uparrow \Rightarrow r \uparrow$**

| | Weak | 2 | 3 | 4 | Robust |
|--|------|---|---|---|--------|
| Small | 0.56 | 0.94 | 0.90 | 0.95 | 0.88 |
| Big | 0.39 | 0.33 | 0.43 | 0.47 | 0.57 |

La progression est moins monotone mais la direction générale est confirmée : les entreprises robustes surperforment les entreprises fragiles parmi les microcaps.

**Panel C — $INV \uparrow \Rightarrow r \downarrow$**

| | Conservative | 2 | 3 | 4 | Aggressive |
|--|-------------|---|---|---|------------|
| Small | 1.01 | 0.98 | 0.99 | 0.89 | 0.35 |
| Big | 0.71 | 0.52 | 0.49 | 0.48 | 0.42 |

Le rendement chute de $1.01\%$ à $0.35\%$ quand l'investissement devient agressif parmi les microcaps. C'est la prédiction DDM la plus nette empiriquement — la décroissance est presque monotone.

**(iii) Pourquoi ça casse pour les megacaps**

Pour les grandes capitalisations, les patterns s'affaiblissent ou disparaissent. Les megacaps sont couvertes par des centaines d'analystes — toute l'information sur leur profitabilité et leur investissement est déjà incorporée dans les prix. Les inefficiences que les facteurs cherchent à capturer n'ont pas le temps de se former.

**(iv) Le problème de pollution**

La Table 1 trie selon un seul critère à la fois. Mais $B/M$, $OP$ et $INV$ sont corrélés entre eux : une value stock ($B/M$ élevé) a tendance à être peu profitable et à peu investir. Un tri univarié sur $B/M$ capture donc aussi indirectement les effets $OP$ et $INV$, sans qu'on puisse les démêler.

> C'est précisément pourquoi le triple tri indépendant est nécessaire — il isole l'effet net de chaque variable en contrôlant pour les deux autres. La Table 1 montre que les effets existent. La Table 4 montrera que chacun est indépendant.

---

## 7. La Table 2 — HML est redondant

**(i) La redondance empirique**

Le test GRS mesure si un modèle laisse des alphas significatifs sur les portefeuilles tests. FF montrent qu'en ajoutant RMW et CMA au modèle $\{R_m - R_f,\ SMB,\ HML\}$, le GRS baisse — le modèle s'améliore. Mais l'observation centrale est différente : **ajouter HML à un modèle qui contient déjà RMW et CMA n'améliore pas significativement le GRS dans aucun des panels testés**.

![Table 2 — test GRS selon les combinaisons de facteurs](images/facteurs-fondamentaux/ff5_table2.png)

*Figure 9. Table 2 — statistique GRS et mesures d'adéquation pour différentes combinaisons de facteurs, sur les trois panels de portefeuilles tests. Les combinaisons encadrées en rouge montrent que RMW CMA sans HML fait aussi bien que le modèle complet.*

HML n'apporte pas d'information indépendante une fois que RMW et CMA sont présents.

**(ii) L'intuition économique**

Pourquoi HML est-il capturé par RMW et CMA ? Parce que les trois variables mesurent des facettes liées de la même réalité économique. Une value stock ($B/M$ élevé) est typiquement une entreprise dont le marché anticipe peu de croissance — elle investit peu ($INV$ faible, donc $CMA$ positif) et sa profitabilité est faible ($OP$ faible, donc $RMW$ négatif). HML capture indirectement ce que RMW et CMA capturent directement et séparément.

> **Nuance importante.** HML reste utile à la marge — le GRS baisse quand on l'ajoute, même en présence de RMW et CMA. Mais il n'est pas un facteur indépendant au sens où il capturerait un risque que les quatre autres ne capturent pas. En pratique, beaucoup de travaux empiriques utilisent le modèle à quatre facteurs $\{R_m - R_f,\ SMB,\ RMW,\ CMA\}$ sans HML.

---

## 8. La Table 3 — régression des facteurs les uns sur les autres

**(i) La question**

Si HML est redondant, ça veut dire qu'il est expliqué par les autres facteurs. On peut le vérifier directement en régressant chaque facteur sur les quatre autres :

$$HML_t = \alpha + \beta^{MKT}(R_{m,t} - R_{f,t}) + \beta^{SMB} \cdot SMB_t + \beta^{RMW} \cdot RMW_t + \beta^{CMA} \cdot CMA_t + \varepsilon_t$$

Et symétriquement pour SMB, RMW, CMA, et le marché.

![Table 3 — régressions croisées des facteurs](images/facteurs-fondamentaux/ff5_table3.png)

*Figure 10. Table 3 — chaque ligne régresse un facteur sur les quatre autres. La ligne HML est encadrée en bleu : $R^2 = 0.51$, $\beta^{CMA} = 1.04$ ($t = 23.03$).*

**(ii) Les résultats pour HML**

| | $\beta^{RMW}$ | $t$ | $\beta^{CMA}$ | $t$ | $R^2$ |
|--|--------------|-----|--------------|-----|-------|
| HML | 0.23 | 5.36 | 1.04 | 23.03 | 0.51 |

Un $R^2$ de $0.51$ signifie que RMW et CMA expliquent plus de la moitié de la variance de HML. Le coefficient sur CMA est $1.04$ avec $t = 23.03$ — extrêmement significatif. Le coefficient sur RMW est positif : les value stocks ont tendance à être peu profitables, ce qui est cohérent avec l'intuition économique.

**(iii) Les résultats pour les autres facteurs**

Les $R^2$ des autres régressions sont bien plus faibles. SMB, RMW et CMA ne sont pas expliqués par les autres facteurs — ils capturent chacun une dimension indépendante du risque. C'est HML seul qui est redondant.

---

## 9. La Table 4 — le résultat du triple tri

**(i) La question**

La Table 1 montrait des effets univariés potentiellement pollués par les corrélations entre $B/M$, $OP$ et $INV$. La Table 4 répond à la vraie question : une fois qu'on contrôle simultanément pour les trois variables, les effets persistent-ils ?

![Table 4 — rendements des portefeuilles du triple tri](images/facteurs-fondamentaux/ff5_table4.png)

*Figure 11. Table 4 — rendements moyens des portefeuilles issus du triple tri taille × B/M × OP et taille × B/M × INV. Chaque cellule contrôle simultanément pour les trois dimensions.*

**(ii) Ce que montre le triple tri**

Les effets persistent après contrôle. Dans chaque cellule $B/M$ fixé, les rendements augmentent avec $OP$ et diminuent avec $INV$ — exactement comme prédit par le DDM. Et inversement, dans chaque cellule $OP$ fixée, les rendements augmentent avec $B/M$.

Ce résultat est la preuve que les trois variables capturent des dimensions **indépendantes** du rendement attendu. Ce n'est pas le même phénomène mesuré trois fois.

**(iii) La conclusion de FF**

Le FF5 améliore le FF3 sur deux fronts : il absorbe les anomalies de profitabilité et d'investissement que le FF3 laissait inexpliquées, et il fournit une justification théorique unifiée via le DDM. Le prix à payer est la complexité — cinq facteurs au lieu de trois, et une construction par triple tri plus délicate à répliquer.

> **Ce que le FF5 ne résout pas.** Le GRS rejette encore le FF5 — il reste des alphas significatifs, notamment sur les portefeuilles de petites capitalisations et les portefeuilles extrêmes. Le modèle est meilleur que le FF3, pas parfait. Le momentum (WML) en particulier reste une anomalie non capturée, ce qui motivera des extensions ultérieures.
