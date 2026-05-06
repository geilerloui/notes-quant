# Fama-French 3 Facteurs

---

## 1. Le point de départ — l'échec empirique du CAPM

Le CAPM prédit une chose simple : plus le bêta d'un actif est élevé, plus son rendement espéré est élevé. FF testent ça sur 30 ans de données (1963–1990) et trouvent une relation plate — le bêta n'est pas rémunéré dans les données.

| | |
|:---:|:---:|
| ![CAPM prediction](beta-capm-prediction.svg) | ![Actual data](beta-actual-data.svg) |

*Figure 1. Chaque point = un décile d'actions trié par bêta (1963–1990). En haut : la prédiction du CAPM. En bas : les données réelles — pente non significativement différente de zéro.*

FF s'appuient alors sur la littérature des anomalies des années 1980 — une pléthore de papiers avait documenté que certaines caractéristiques des entreprises (taille, B/M, momentum...) prédisaient les rendements mieux que le bêta, sans que personne ne sache vraiment pourquoi. FF prennent les deux anomalies les plus robustes et les formalisent en un modèle cohérent.

---

## 2. Les deux variables explicatives

**(i) Rappel comptable — le bilan et les notations**

![Balance sheet](imZ.png)

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

![Long-short portfolio](long-short-portfolio.svg)

*Figure 3. Construction du facteur SMB comme portefeuille long-short.*

Le short **annule** l'exposition au marché : si le marché monte de 2%, les longs montent de 2% et les shorts perdent 2% — les deux s'annulent. Ce qui reste dans le rendement du portefeuille, c'est **uniquement** la différence entre les deux groupes, purifiée du mouvement de marché.

**(ii) La grille 2×3 — le double tri**

Chaque juillet, FF trient toutes les actions selon deux critères indépendants : médiane $ME$ → Small/Big, et 30ᵉ/70ᵉ percentiles $B/M$ → Growth/Neutral/Value. L'intersection donne 6 portefeuilles.

![Grille 2x3](grid-2x3.svg)

*Figure 4. Les 6 portefeuilles issus du double tri taille × B/M.*

**(iii) Les formules de SMB et HML**

$$SMB = \frac{1}{3}(R_{SV} + R_{SN} + R_{SG}) - \frac{1}{3}(R_{BV} + R_{BN} + R_{BG})$$

$$HML = \frac{1}{2}(R_{SV} + R_{BV}) - \frac{1}{2}(R_{SG} + R_{BG})$$

**(iv) Orthogonalité des facteurs**

> 📌 SMB et HML sont quasi-orthogonaux (corrélation < 0.15 en pratique) grâce au double tri. En moyennant SMB sur les 3 lignes B/M, l'effet valeur se compense des deux côtés et disparaît — il reste uniquement de la taille pure. Symétriquement, HML moyenné sur Small et Big fait disparaître l'effet taille. Sans ce double tri, les deux facteurs seraient corrélés (les small caps sont naturellement plus souvent des value stocks) et la régression ne pourrait pas les distinguer l'un de l'autre — problème de multicolinéarité.

**(v) La timeline — rebalancement annuel**

![Timeline rebalancement](timeline-rebalancing.svg)

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

![Panel A Summary Statistics](imX.png)

*Figure 6. Panel A — moyenne et écart-type des rendements mensuels des 25 portefeuilles.*

Les rendements augmentent dans deux directions : de gauche à droite (Growth → Value) et de haut en bas (Big → Small). La valeur 0.31 pour Small Growth est la moyenne des rendements mensuels de ce portefeuille sur toute la période — le rendement le plus faible de toute la grille.

Le fait que les rendements bougent selon ces deux axes indépendants est exactement ce qui motive l'existence des deux facteurs.

**(ii) Panel B — coefficients de régression**

![Panel B Regression Results](imY.png)

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

![Cascade du compte de résultat — chaque étage rajoute du bruit comptable.](novy_cascade_compte_resultat.png)

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

![Table 1 — Régressions Fama-McBeth. 7 spécifications différentes, une par colonne.](novy_fama_mcbeth.png)

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

![Table 2 — Portefeuilles triés par GP/A (Panel A) et par B/M (Panel B).](novy_linear_regression.png)

*Figure 3. Portefeuilles triés par profitabilité (Panel A) et par B/M (Panel B). Les rendements augmentent dans les deux cas de Low à High. Mais les coefficients HML vont dans des directions opposées.*

Les rendements augmentent de Low GP/A ($0.31\%$ par mois) à High GP/A ($0.62\%$). Mais le résultat surprenant c'est le coefficient sur HML :

- Portefeuille Low GP/A : $\beta^{HML} = +0.15$
- Portefeuille High GP/A : $\beta^{HML} = -0.29$

Les entreprises très profitables se comportent comme des growth stocks, à l'opposé des value stocks qui ont $\beta^{HML} = +0.51$. C'est une observation visuelle qui donne l'intuition. Pour le vérifier formellement, il construit la matrice de données suivante :

![Matrice de données : n actions × d caractéristiques fondamentales.](novy_matrice_donnees.png)

*Figure 4. Structure des données utilisées pour calculer la table de corrélation. Chaque ligne est une action, chaque colonne une caractéristique fondamentale. La colonne B/M est en rouge car sa corrélation avec GP/A est négative.*

À partir de cette matrice il calcule la corrélation de Pearson entre chaque paire de colonnes.

![Figure 18 — Table de corrélation entre les caractéristiques fondamentales.](novy_correlation.png)

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

![Figure 17 — Performance du portefeuille mixte de 1963 à 2010.](novy_hedging.png)

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

![Table 1 — rendements moyens des portefeuilles univariés](ff5_table1.png)

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

![Table 2 — test GRS selon les combinaisons de facteurs](ff5_table2.png)

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

![Table 3 — régressions croisées des facteurs](ff5_table3.png)

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

![Table 4 — rendements des portefeuilles du triple tri|424](ff5_table4.png)

*Figure 11. Table 4 — rendements moyens des portefeuilles issus du triple tri taille × B/M × OP et taille × B/M × INV. Chaque cellule contrôle simultanément pour les trois dimensions.*

**(ii) Ce que montre le triple tri**

Les effets persistent après contrôle. Dans chaque cellule $B/M$ fixé, les rendements augmentent avec $OP$ et diminuent avec $INV$ — exactement comme prédit par le DDM. Et inversement, dans chaque cellule $OP$ fixée, les rendements augmentent avec $B/M$.

Ce résultat est la preuve que les trois variables capturent des dimensions **indépendantes** du rendement attendu. Ce n'est pas le même phénomène mesuré trois fois.

**(iii) La conclusion de FF**

Le FF5 améliore le FF3 sur deux fronts : il absorbe les anomalies de profitabilité et d'investissement que le FF3 laissait inexpliquées, et il fournit une justification théorique unifiée via le DDM. Le prix à payer est la complexité — cinq facteurs au lieu de trois, et une construction par triple tri plus délicate à répliquer.

> **Ce que le FF5 ne résout pas.** Le GRS rejette encore le FF5 — il reste des alphas significatifs, notamment sur les portefeuilles de petites capitalisations et les portefeuilles extrêmes. Le modèle est meilleur que le FF3, pas parfait. Le momentum (WML) en particulier reste une anomalie non capturée, ce qui motivera des extensions ultérieures.

---

## Asness, Moskowitz & Pedersen (2013) — Value and Momentum Everywhere

Le papier étudie deux stratégies d'investissement sur **8 classes d'actifs** simultanément : la stratégie **value** (acheter des actifs sous-évalués) et la stratégie **momentum** (acheter des actifs qui ont bien performé récemment).

La découverte centrale : ces deux stratégies sont **négativement corrélées**. Les auteurs exploitent cette propriété pour construire un meilleur portefeuille. Le fonds **AQR** applique cette stratégie en conditions réelles.

> Tables à regarder en priorité : **I, II, IV**.

---

### (i) Le concept clé : pourquoi la corrélation négative est précieuse

Le rendement d'un portefeuille combinant deux stratégies $A$ et $B$ :

$$R_p = a \cdot R_A + (1-a) \cdot R_B$$

La variance (= le risque) de ce portefeuille :

$$\text{Var}(R_p) = a^2 \sigma^2_A + (1-a)^2 \sigma^2_B + 2a(1-a)\underbrace{\text{Cov}(A,B)}_{\text{clé}}$$

Le terme de covariance est déterminant :

- Si $A$ et $B$ sont **positivement corrélés** → covariance positive → la diversification ne sert à rien.
- Si $A$ et $B$ sont **négativement corrélés** → covariance négative → le risque **baisse**, même avec seulement 2 actifs.

C'est exactement le cas ici : combiner value et momentum **réduit le risque sans sacrifier le rendement espéré**.

---

### (ii) Les données utilisées

Le papier couvre 8 classes d'actifs : actions US, UK, Europe et Japon (grandes capitalisations uniquement), futures sur 18 indices actions mondiaux, devises, obligations gouvernementales, et matières premières.

Les small stocks sont exclus car trop illiquides pour implémenter ces stratégies en pratique. Les auteurs construisent au total **48 portefeuilles tests** : 3 groupes × 2 stratégies × 8 classes d'actifs.

> ⚠️ **Limite importante.** Risque de *cherry picking* sur le choix des 8 classes d'actifs — voir section (vii).

---

### (iii) Construction des portefeuilles

Pour chaque classe d'actifs, tous les actifs sont triés en 3 groupes égaux. **P1** regroupe les "perdants", **P2** le milieu, **P3** les "gagnants". Deux portefeuilles zero-cost sont construits :

- **P3 − P1** : long P3, short P1 — le spread brut.
- **Factor** : même logique mais chaque actif est pondéré par son rang (*rank-weighted*) — version plus lisse.

**Pour les actions — Value.** Ratio book-to-market : valeur comptable laggée de 6 mois divisée par le prix de marché actuel. Ratio élevé → sous-évalué → on achète. Ratio faible → surévalué → on vend.

**Pour les actions — Momentum.** Performance cumulée de $t-12$ à $t-2$ (le dernier mois est exclu pour éviter le *mean reversion* de court terme). On achète les winners, on vend les losers.

**Pour les devises et matières premières — Value.** Pas de ratio B/M possible. Les auteurs utilisent le rendement sur 5 ans inversé. Si un actif vaut 50 à $t = -5$ ans et 20 aujourd'hui :

$$\text{rendement}_{5\text{ans}} = \frac{20 - 50}{50} = -60\% \implies \text{proxy value} = +60\%$$

On trie tous les actifs ainsi, on achète les plus "bon marché", on vend les plus "chers".

---

### (iv) Résultats — Table I

![Table I — Performance des portefeuilles value et momentum](value_momentum_table1.png)

*Figure 1. Performance des portefeuilles value, momentum et 50/50 sur les actions US (01/1972–07/2011). Rendements annualisés en excès du T-bill.*

![Résumé Table I](value_momentum_cards_table1.png)

Le **4.6 %** et le **t-stat de 3.98** (colonnes P3−P1) et le **5.8 %** avec **t-stat 5.40** (colonne Factor) correspondent au portefeuille 50/50.

L'observation cruciale : le t-stat du 50/50 (5.40) est **bien plus élevé** que celui de chaque stratégie prise séparément (1.66 et 2.84), alors que le rendement de 5.8 % est juste la moyenne des deux. L'explication :

$$t\text{-stat} = \frac{\mathbb{E}[R_p]}{\sigma(R_p)} \qquad \text{avec} \qquad \sigma^2(R_p) = \frac{1}{4}\sigma^2_V + \frac{1}{4}\sigma^2_M + \frac{1}{2}\underbrace{\text{Cov}(V,M)}_{<\, 0}$$

La covariance négative **soustrait** du risque total : l'écart-type s'effondre de ~15 % à 6.8 %, le t-stat monte mécaniquement, et le Sharpe ratio passe de ~0.35 pour chaque stratégie seule à **0.86 pour le 50/50**. C'est l'argument central du papier.

---

### (v) Structure des corrélations — Table II

#### Pourquoi $0.68$ et pas $1$ sur la diagonale ?

Dans une matrice de corrélation classique, la diagonale est toujours égale à 1 car on compare un actif à lui-même. Ici, ce n'est pas le cas. Chaque cellule compare la **série de rendement moyenne** d'une stratégie agrégée sur plusieurs marchés. Le chiffre $0.68$ pour "Stock Value" est la corrélation entre la stratégie value aux USA et la moyenne des stratégies value au UK, en Europe et au Japon.

**Interprétation business.** Cela prouve que la value est un facteur **mondial** ("Everywhere"). Si le chiffre était proche de 0, cela voudrait dire que la value est un phénomène purement local.

![Table II — Matrice de corrélations](value_momentum_table2.png)

*Figure 2. Matrice de corrélations des rendements moyens entre stratégies value et momentum, toutes classes d'actifs confondues. \* = statistiquement significatif.*

#### La structure en block matrix

La matrice révèle une organisation très symétrique en trois blocs :

- **Bloc haut-gauche positif ($0.68$).** Toutes les stratégies value bougent ensemble à travers le monde → la value est un phénomène global.
- **Bloc bas-droit positif ($0.65$).** Toutes les stratégies momentum bougent ensemble → même logique.
- **Bloc haut-droit négatif ($-0.53$).** Value et momentum vont systématiquement dans des directions opposées — c'est le cœur du papier.

> Cette structure en blocs suggère l'existence d'un **facteur commun unique** qui drive les deux familles dans des directions opposées, d'où l'idée naturelle d'utiliser une **PCA**.

---

### (vi) Analyse en composantes principales (PCA)

La PCA sur les 48 stratégies confirme que le **PC1** explique environ $0.4$ de la variance pour les stocks US et UK.

- **Chargements (*loadings*).** Le PC1 charge **positivement sur le momentum** et **négativement sur la value**.
- **Identification du facteur.** Les auteurs identifient ce facteur sous-jacent unique comme étant la **liquidité de financement** (*funding liquidity*).
- **Logique économique.** Le momentum est une stratégie qui nécessite du levier et beaucoup de transactions. En cas de choc de liquidité (quand le crédit se resserre), le momentum est massivement liquidé, tandis que la value (actifs délaissés) réagit différemment.

---

### (vii) Les facteurs macro n'expliquent rien — Table III

![Table III — Régression sur facteurs macro](value_momentum_table3.png)

*Figure 3. Régression des rendements value et momentum sur des facteurs macroéconomiques (PIB, term spread, inflation…). Les t-stats sont proches de 0 dans l'ensemble.*

Les t-stats proches de 0 et les $R^2$ très faibles confirment que les facteurs macro n'expliquent ni le value premium ni le momentum premium. Ce résultat n'est pas une surprise — c'est connu dans la littérature depuis longtemps.

---

### (viii) La vraie explication : la liquidité de financement — Table IV

La **funding liquidity** n'est pas la liquidité de marché (bid-ask spread). C'est la facilité avec laquelle les investisseurs peuvent emprunter pour financer leurs positions. Elle est mesurée via le **TED spread** (LIBOR − T-bills) et d'autres proxies de stress sur les marchés de financement.

Le TED spread est très autocorrélé : si le spread est élevé aujourd'hui, il le sera probablement demain. Pour extraire les chocs purs, les auteurs modélisent un AR(2) :

$$R^{TED}_t = \alpha + \gamma_1 R^{TED}_{t-1} + \gamma_2 R^{TED}_{t-2} + \varepsilon_t$$

Le résidu $\varepsilon_t$ est le **choc de liquidité pur**, débarrassé de toute autocorrélation.

![TED spread historique](value_momentum_ted_spread.png)

*Figure 4. TED spread de 1990 à 2011. Le spread est quasi-plat pendant des années, puis explose lors des crises (LTCM 1998, crise financière 2008) avant de redescendre lentement — comportement typique d'un processus autocorrélé.*

Ils font ensuite une PCA sur l'ensemble des mesures de liquidité (TED spread, LIBOR spread, etc.) et gardent le premier composant principal comme proxy unique — la PCA élimine le bruit et retient ce qui est **commun** à toutes ces séries.

![Table IV — Régression sur la liquidité de financement](value_momentum_table4.png)

*Figure 5. Régression des rendements value et momentum sur les mesures de liquidité de financement. La value est négativement corrélée aux chocs, le momentum positivement.*

**Résultats clés.**

- La value est **négativement** corrélée aux chocs de liquidité : quand le crédit se resserre, la value souffre.
- Le momentum est **positivement** corrélé aux mêmes chocs.
- Le PC de funding liquidity est le driver commun des deux stratégies dans des directions opposées, aussi bien au niveau US que global.

![Figure 1 — Chocs de liquidité historiques](value_momentum_figure1.png)

*Figure 6. Épisodes historiques de stress de liquidité de financement et leur impact sur les stratégies value et momentum.*

---

### (ix) Le nouveau modèle de pricing — Figure 6

Pour tester leur modèle, les auteurs utilisent les 48 stratégies comme actifs tests. Pour chaque stratégie, les régresseurs sont le facteur marché et toutes les autres stratégies sauf celle testée.

Sur chaque graphe, l'axe X est le **rendement prédit** par le modèle et l'axe Y est le **rendement réel observé**. Si le modèle était parfait, tous les points seraient sur la diagonale.

![Figure 2 — Tests de pricing cross-sectionnel](value_momentum_figure2.png)

*Figure 7. Rendements réels vs rendements prédits pour les 48 portefeuilles value et momentum, sous quatre modèles : CAPM, Fama-French 4 facteurs, Fama-French 6 facteurs, et AMP 3 facteurs.*

| Modèle | $R^2$ | Alpha moyen |
|---|---|---|
| CAPM | 0.45 | Élevé |
| Fama-French 4 facteurs | 0.55 | Significatif |
| Fama-French 6 facteurs | 0.60 | Significatif |
| **AMP 3 facteurs** | **0.71** | **Plus faible** |

Le modèle AMP atteint $R^2 = 0.71$ — meilleur fit parmi les quatre. Cependant, tous les modèles ont un **GRS F-stat significatif** ($p = 0$) : aucun ne capture parfaitement tous les rendements, les alphas restent non nuls. Le message des auteurs est que leur modèle est le moins mauvais, mais qu'aucun modèle ne clôt complètement le débat.

---

### (x) Limites du papier

Le professeur soulève quatre critiques majeures :

1. **Réplication impossible.** Il n'a pas réussi à reproduire exactement les résultats des auteurs, ce qui pose une question de fiabilité des données originales.

2. **Risque de *cherry picking*.** Il est possible que les auteurs aient sélectionné les 8 classes d'actifs qui fonctionnaient le mieux pour obtenir une matrice de corrélation aussi "propre". Beaucoup de papiers en sciences sociales ne se répliquent pas sur un échantillon plus large.

3. **Saut interprétatif sur le PC1.** Le lien entre le facteur mathématique PC1 et la *funding liquidity* n'est pas explicitement démontré dans une table de données — c'est une interprétation théorique des auteurs.

4. **Biais de capitalisation.** L'exclusion des *small stocks* limite la portée des résultats aux grandes capitalisations uniquement, ignorant une partie du marché où la liquidité est pourtant cruciale.

---

> **Résumé.** Value et momentum sont négativement corrélées parce qu'elles réagissent en sens opposé aux chocs de liquidité de financement. Combiner les deux en 50/50 fait passer le Sharpe de ~0.35 à **0.86** en faisant s'effondrer le risque grâce à la covariance négative.


# Tests internationaux du modèle à cinq facteurs

> Fama, E.F. & French, K.R. (2017). *International tests of a five-factor asset pricing model*. **Journal of Financial Economics**, 123(3), 441–463.

---

## 1. Motivation et positionnement

Le papier **FF (2015)** propose un modèle à cinq facteurs testé exclusivement sur le marché actions américain. La question naturelle est : *est-ce que ce modèle fonctionne hors des US ?* Ce papier constitue un **out-of-sample test** sur quatre régions :

- **NAM** — North America (US + Canada)
- **Europe**
- **Japan**
- **AP** — Asia Pacific

Deux références justifient pourquoi on ne peut pas simplement extrapoler les résultats US :

- **Fama & French (2008)** — *Dissecting Anomalies* : les résultats en finance sont souvent *sample-specific* ; toute étude hors des US doit citer ce papier.
- **FF (2012)** — première étude internationale sur size, value et momentum, qui motive la structure en quatre régions retenue ici.

---

## 2. Choix méthodologiques

### (i) Mesure des rendements en dollars

Les auteurs adoptent la perspective d'un investisseur américain et **mesurent tous les rendements en dollars**. Ce choix introduit mécaniquement un *dollar factor* dans les données : convertir les rendements locaux en dollars revient à multiplier l'indice par le taux de change, si bien que les résultats peuvent être en partie *driven* par l'appréciation ou la dépréciation du dollar sur la période.

### (ii) Facteurs locaux vs facteurs globaux

Le papier teste deux architectures :

- **Facteurs globaux** : un seul SMB global, un seul RMW global, etc., construits sur l'ensemble des marchés.
- **Facteurs locaux** : des facteurs propres à chaque région (SMB_EUR, RMW_JAP, etc.).

La conclusion est sans ambiguïté : **l'intégration des marchés financiers n'est pas assez avancée** pour que les facteurs globaux fonctionnent. Ce sont les facteurs locaux qu'il faut utiliser.

---

## 3. Le modèle et l'objectif des tests

Le modèle testé est une régression time-series pour chaque actif $i$ :

$$
R_{i,t} - R_{f,t} = \alpha_i + \beta_1 \, \text{MKT}_t + \beta_2 \, \text{SMB}_t + \beta_3 \, \text{HML}_t + \beta_4 \, \text{RMW}_t + \beta_5 \, \text{CMA}_t + \varepsilon_{i,t}
$$

L'objectif est de tester si $\alpha_i = 0$. Si le modèle décrit correctement la cross-section des rendements espérés, tous les alphas doivent être nuls — tout ce qui est systématique est capté par les cinq facteurs. Un alpha significativement différent de zéro signale un **mispricing systématique** que le modèle ne parvient pas à expliquer.

### Le test GRS

Le **test GRS (Gibbons, Ross & Shanken, 1989)** teste conjointement la nullité de tous les alphas sur les $N = 25$ portefeuilles tests :

$$
H_0 : \alpha_1 = \alpha_2 = \cdots = \alpha_{25} = 0
$$
$$
H_1 : \exists \, i \text{ tel que } \alpha_i \neq 0 \quad \Rightarrow \quad \text{mispricing systématique}
$$

Si $p(\text{GRS}) = 0.00$, on rejette $H_0$ : le modèle ne capte pas toute la cross-section. Mais rejeter le GRS ne signifie pas que le modèle est inutile — la question pertinente est de savoir *de combien* les alphas s'écartent de zéro, et si le FF5 fait mieux que le FF3.

---

## 4. Résultats

### (i) Table 1 — Primes de risque moyennes par région

**Panel A** reporte les moyennes des facteurs sur chaque marché.

<div style="display: flex; flex-direction: column; gap: 0.5rem;">

![Table 1 Panel A — Sample averages des facteurs par région](ff5_international_table1_panel_a.png)

*Figure 1. Table 1, Panel A. Moyennes des primes de risque (annualisées, en %) et t-statistiques pour les quatre régions.*

</div>

Résultats saillants :

- **NAM** : le facteur de marché est significatif (t-stat ≈ 2.63), mais le **size premium (SMB) est faible** (t-stat ≈ 1.05) — pas d'effet taille robuste.
- **Japan** : **seul HML est significatif** — ni la profitabilité (RMW) ni l'investissement (CMA) ne semblent pricer les actions japonaises. Le Japon est un cas à part dans toute la littérature internationale.

**Panel B** décompose HML en *small stocks* vs *big stocks*.



![Table 1 Panel B — Décomposition HML small vs big](ff5_international_table1_panel_b.png)

*Figure 2. Table 1, Panel B. Décomposition de la prime value selon la taille des entreprises.*



Contrairement à l'intuition habituelle (les anomalies seraient concentrées dans les petites capitalisations), le point estimate de HML est principalement **driven par les big stocks** pour toutes les régions sauf le Japon.

**Panel C** donne les matrices de corrélation des facteurs *entre régions*.



![Table 1 Panel C — Corrélations des facteurs entre régions](ff5_international_table1_panel_c.png)

*Figure 3. Table 1, Panel C. Matrices de corrélation des facteurs entre les quatre régions.*



Résultat notable : **la corrélation de RMW entre régions est quasi nulle**. Implication pratique : implémenter une stratégie *long-profitable* indépendamment dans chaque marché produit un portefeuille global dont les composantes sont quasi-orthogonales — c'est une opportunité de diversification quasi gratuite.

---

### (ii) Table 2 — Portefeuilles double-triés

**Panel A** réplique les sorts de FF (2015) sur les données internationales.

<div style="display: flex; flex-direction: column; gap: 0.5rem;">

![Table 2 Panel A — Portefeuilles double-triés](ff5_international_table2_panel_a.png)

*Figure 4. Table 2, Panel A. Rendements moyens des portefeuilles triés sur Size × B/M, Size × OP et Size × Inv pour chaque région.*

</div>

Les rendements moyens augmentent bien lorsqu'on passe des groupes 3–5 (big) vers les groupes 1–2 (small), confirmant un effet taille. Ils augmentent également avec le B/M et la profitabilité. **Mais ce schéma ne tient pas pour tous les groupes** — notamment les groupes 1–2 en Europe — et le **Japon présente des patterns très faibles** pour la profitabilité et l'investissement.

Le problème identifié dans FF (2015) persiste à l'international : les **small stocks non-profitables qui investissent beaucoup** génèrent des rendements anormalement bas que le modèle ne parvient pas à expliquer.

---

### (iii) Table 3 — Régressions des facteurs (test de redondance)

<div style="display: flex; flex-direction: column; gap: 0.5rem;">

![Table 3 — Factor regressions](ff5_international_table3.png)

*Figure 5. Table 3. Régressions de chaque facteur sur les quatre autres, par région. L'intercept teste la redondance.*

</div>

Dans **FF (2015)**, la régression de HML sur les quatre autres facteurs donnait un intercept significatif → conclusion : **HML est redondant** une fois qu'on contrôle RMW et CMA.

Le résultat est différent à l'international. Pour certains marchés, l'intercept de HML est négatif (par exemple $-0.28$) et non significatif → **HML n'est plus redondant**. En revanche, c'est désormais le facteur **CMA (investissement) qui est *on shaky ground*** — son rôle marginal est remis en question selon les régions. Les patterns diffèrent sensiblement d'un marché à l'autre.

---

### (iv) Table 4 — Test GRS : FF3 vs FF5

<div style="display: flex; flex-direction: column; gap: 0.5rem;">

![Table 4 — GRS statistics](ff5_international_table4.png)

*Figure 6. Table 4. Statistiques GRS et p-values pour le modèle FF3 et le modèle FF5, sur différents ensembles de portefeuilles tests, par région.*

</div>

Pour NAM, sur les 25 portefeuilles $\text{Size} \times \text{B/M}$ :

| Modèle | GRS | p-value |
|--------|-----|---------|
| FF3    | 2.85 | 0.00   |
| FF5    | 2.34 | ≈ 0.00 |

Le FF5 **améliore systématiquement le GRS** par rapport au FF3, mais le test reste rejeté. L'interprétation correcte n'est pas que le modèle est mauvais — avec 25 actifs tests, on rejette presque toujours le GRS. La question pertinente est l'**amélioration économique** : les alphas résiduels sont-ils plus petits et moins systématiques avec le FF5 ? La réponse est oui pour NAM, Europe et AP. Pour le Japon, le gain est plus limité.

---

### (v) Table 5 — À compléter



![Table 5 — partie 1](ff5_international_table5.png)

*Figure 7. Table 5, partie 1.*




![Table 5 — partie 2](ff5_international_table5_part2.png)

*Figure 8. Table 5, partie 2.*



---

## 5. Conclusions

Le modèle FF5 constitue une amélioration substantielle sur le FF3 pour **NAM, Europe et Asia Pacific** — il capte les patterns de B/M, profitabilité et investissement dans les rendements moyens. Trois points importants à retenir :

1. **Les facteurs globaux ne fonctionnent pas** — l'intégration des marchés financiers est insuffisante pour un modèle unifié. Il faut des facteurs locaux par région.

2. **Le Japon est un cas à part** : seul HML est robustement pricé ; la profitabilité et l'investissement n'ont pas de pouvoir explicatif significatif sur les rendements japonais.

3. **Le facteur value (HML) n'est plus redondant** à l'international, contrairement à la conclusion de FF (2015). C'est en revanche le facteur **investissement (CMA)** dont le rôle apparaît fragile selon les régions.

Le problème principal du modèle — déjà identifié dans FF (2015) — persiste : les **petites entreprises non-profitables qui investissent beaucoup** génèrent des rendements que le modèle sous-estime systématiquement.

# Factor investing — Low-risk anomaly et volatilité

Ce chapitre couvre quatre papiers fondateurs sur la *low-risk anomaly* et la gestion dynamique de la volatilité, présentés dans leur ordre chronologique et logique.

| Papier | Contribution clé |
|---|---|
| Frazzini & Pedersen (2014) | Documenter l'anomalie, construire BAB |
| Asness, Frazzini, Gormsen & Pedersen (2020) | Décomposer BAB en BAC + BAV |
| Moreira & Muir (2017) | Volatility-Managed Portfolios (in-sample) |
| Cederburg, O'Doherty, Wang & Yan (2020) | Test OOS des VMP |

---

## 01 — Betting Against Beta (BAB)

*Frazzini & Pedersen — Journal of Financial Economics, 2014*

### (i) L'anomalie de départ

Le CAPM prédit une relation positive et linéaire entre $\beta$ et rendement espéré — la *Security Market Line* (SML). Black, Jensen & Scholes (1972) sont les premiers à tester cette prédiction sur données réelles et observent que la droite empirique est systématiquement **plus plate** que la SML théorique.

$$\text{Prédit par le CAPM :} \quad E[r_i] = r_f + \beta_i (E[r_m] - r_f)$$

En pratique : les titres à faible $\beta$ génèrent un $\alpha > 0$ (sous-évalués par le CAPM) et les titres à fort $\beta$ génèrent un $\alpha < 0$ (surévalués).

![[volatility_flat_sml.png]]

*Figure 1. Flat security market line. La droite empirique (bleue) est plus plate que la SML théorique (pointillée). Les titres à faible $\beta$ surperforment le modèle ($\alpha > 0$) et les titres à fort $\beta$ sous-performent ($\alpha < 0$).*

L'idée centrale de BAB : acheter les titres à faible $\beta$ (sous-évalués) et shorter les titres à fort $\beta$ (surévalués), en neutralisant l'exposition nette au marché via un scaling des deux jambes.

### (ii) Théorie — contraintes d'effet de levier

Pourquoi cette anomalie persiste-t-elle ? Certains investisseurs institutionnels (fonds de pension, assureurs) sont **contraints en effet de levier** — leur règlement leur interdit d'emprunter pour leverager un portefeuille diversifié. Pour atteindre leur cible de rendement, ils n'ont qu'une seule option : surpondérer directement les actifs à fort $\beta$.

> **Exemple concret — Jean (fonds de pension) vs Léo (trader).** Le marché va faire +10%. Jean veut faire +20% pour battre ses concurrents. *Option A (interdite)* : acheter des actions sûres ($\beta = 1$) avec un levier ×2, soit $2 \times 10\% = 20\%$. Mais son règlement interdit d'emprunter. *Option B (sa seule solution)* : acheter des actions à forte corrélation avec le marché ($\beta = 2$). Si le marché fait +10%, ces actions font naturellement +20%.

Comme tous les investisseurs contraints se ruent sur les actions à haute corrélation, leur prix monte — et leur rendement futur baisse. Si Léo peut utiliser du levier, il achète les actions à faible corrélation (délaissées), les leverage, et encaisse l'alpha que Jean a abandonné. La prédiction testable : le spread entre titres à faible et fort $\beta$ doit être positif et corrélé avec le niveau de *margin debt* dans l'économie.

### (iii) Univers d'actifs

Le papier teste la stratégie sur sept classes d'actifs : indices actions (Australia, Germany, Japan, UK, US…), obligations souveraines internationales, US Treasury bonds (différentes maturités), corporate bonds (IG & HY), credit indices (CDS), paires de devises, et commodities. Dans chaque classe, le $\beta$ est calculé par rapport au portefeuille de marché spécifique à cette classe.

### (iv) Décomposition du bêta

Le bêta est une identité algébrique qui se décompose en deux dimensions :

$$\beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)} = \frac{\rho_{im} \cdot \sigma_i \cdot \sigma_m}{\sigma_m^2} = \rho_{im} \cdot \frac{\sigma_i}{\sigma_m}$$

En notation estimée :

$$\hat{\beta}_i = \underbrace{\hat{\rho}_{im}}_{\text{BAC}} \cdot \underbrace{\dfrac{\hat{\sigma}_i}{\hat{\sigma}_m}}_{\text{BAV}}$$

où $\hat{\rho}_{im}$ = corrélation du titre $i$ avec le marché (le titre bouge-t-il en même temps que le marché ?) et $\hat{\sigma}_i / \hat{\sigma}_m$ = volatilité relative (le titre bouge-t-il plus fort en amplitude que le marché ?). Ces deux dimensions seront exploitées séparément par BAC et BAV (section 02).

![[volatility_beta_decomposition.png]]

*Figure 2. Décomposition $\hat{\beta}_i = \hat{\rho}_{im} \times (\hat{\sigma}_i / \hat{\sigma}_m)$. La dimension corrélation sera exploitée par BAC (contraintes d'effet de levier) et la dimension volatilité relative par BAV (lottery demand).*

### (v) Estimation du bêta en deux étapes

Le bêta est estimé en deux étapes avec des fenêtres différentes, justifiées par le fait que les corrélations varient plus lentement que les volatilités :

**Étape 1 — bêta time-series :**

$$\hat{\beta}_i^{ts} = \hat{\rho} \cdot \frac{\hat{\sigma}_i}{\hat{\sigma}_m}$$

- $\hat{\rho}_{im}$ estimé sur **5 ans** de returns agrégés sur 3 jours ($r_{i,t}^{3d} = \sum_{k=0}^{2} \ln(1 + r_{i,t+k})$) pour corriger le *nonsynchronous trading*
- $\hat{\sigma}_i$ estimé sur **1 an** de returns quotidiens
- Minimum requis : 6 mois pour $\hat{\sigma}$, 3 ans pour $\hat{\rho}$

**Étape 2 — shrinkage vers la moyenne cross-sectionnelle :**

$$\hat{\beta}_i = w_i \hat{\beta}_i^{TS} + (1 - w_i) \hat{\beta}^{XS}$$

avec $w_i = 0.6$ et $\hat{\beta}^{XS} = 1$ fixes pour tous les actifs et toutes les périodes. Sans shrinkage, un titre estimé à $\beta = 0.2$ serait leveragé d'un facteur 5 — trop sensible au bruit d'estimation. Avec shrinkage : $0.6 \times 0.2 + 0.4 \times 1 = 0.52$, soit un levier de ≈1.9×. Le shrinkage ne change pas le classement des titres mais réduit l'amplitude du scaling.

### (vi) Construction du facteur BAB

Sans scaling, acheter P1 (low-$\beta$) et shorter P10 (high-$\beta$) crée un $\beta$ net positif : si le marché monte de 10%, P1 monte de 6.4% et P10 monte de 17% — la jambe short explose. Le facteur BAB divise chaque jambe par son $\beta$ pour ramener les deux à $\beta = 1$, rendant le portefeuille market-neutral :

$$r_{BAB,t+1} = \frac{1}{\hat{\beta}^L}\left(r^L_{t+1} - r^f\right) - \frac{1}{\hat{\beta}^H}\left(r^H_{t+1} - r^f\right)$$

où $\hat{\beta}^L$ = bêta du portefeuille low-$\beta$, $\hat{\beta}^H$ = bêta du portefeuille high-$\beta$. Le $\beta$ net = $1 - 1 = 0$ → market-neutral.

**Exemple chiffré avec les données de la table :**

| Jambe | $\beta$ ex ante | Scaling | CAPM $\alpha$ | Contribution |
|---|---|---|---|---|
| Longue (P1, low-$\beta$) | 0.64 | $\times$ 1.56 | +0.52 %/mois | ≈ +0.81% |
| Courte (P10, high-$\beta$) | 1.70 | $\times$ 0.59 | −0.10 %/mois | ≈ +0.06% |

Le −0.10% de P10 n'est pas un coût d'emprunt : c'est l'observation que les titres à fort $\beta$ rapportent 0.10%/mois de moins que ce que le CAPM prédit. En shortant P10, cette sous-performance devient un gain. Contribution totale ≈ 0.87%/mois, proche des 0.73% observés (le calcul exact porte sur tous les déciles pondérés).

### (vii) Résultats empiriques — actions US (Table 3, 1926–2012)

Les titres sont triés en 10 déciles de $\beta$. La colonne BAB est le rendement du portefeuille long/short décrit ci-dessus, issu de la régression :

$$r_{BAB,t} = \alpha_{BAB} + \beta_{BAB}(r_{mt} - r_{ft}) + \epsilon_t$$

| Mesure | P1 (low) | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 (high) | BAB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Excess return | 0.91 | 0.98 | 1.00 | 1.03 | 1.05 | 1.10 | 1.05 | 1.08 | 1.06 | 0.97 | **0.70** |
| *(t-stat)* | *(6.37)* | *(5.73)* | *(5.16)* | *(4.88)* | *(4.49)* | *(4.37)* | *(3.84)* | *(3.74)* | *(3.27)* | *(2.55)* | *(7.12)* |
| CAPM alpha | **0.52** | **0.48** | **0.42** | **0.39** | **0.34** | **0.34** | 0.22 | 0.21 | 0.10 | −0.10 | **0.73** |
| *(t-stat)* | *(6.30)* | *(5.99)* | *(4.91)* | *(4.43)* | *(3.51)* | *(3.20)* | *(1.94)* | *(1.72)* | *(0.67)* | *(−0.48)* | *(7.44)* |
| 3-factor alpha | **0.40** | **0.35** | **0.26** | **0.21** | **0.13** | 0.11 | −0.03 | −0.06 | **−0.22** | **−0.49** | **0.73** |
| *(t-stat)* | *(6.25)* | *(5.95)* | *(4.76)* | *(4.13)* | *(2.49)* | *(1.94)* | *(−0.59)* | *(−1.02)* | *(−2.81)* | *(−3.68)* | *(7.39)* |
| Beta (ex ante) | 0.64 | 0.79 | 0.88 | 0.97 | 1.05 | 1.12 | 1.21 | 1.31 | 1.44 | 1.70 | **0.00** |
| Volatility (%) | 15.70 | 18.70 | 21.11 | 23.10 | 25.56 | 27.58 | 29.81 | 31.58 | 35.52 | 41.68 | 10.75 |
| Sharpe ratio | **0.70** | 0.63 | 0.57 | 0.54 | 0.49 | 0.48 | 0.42 | 0.41 | 0.36 | **0.28** | **0.78** |

**Lecture de la colonne BAB :**

- **Alpha de 0.73% (/mois)** : part du rendement non expliquée par le marché. Environ 8.7%/an d'excès de rendement pur.
- **t-stat de 7.44** : l'un des t-stats les plus élevés de la littérature empirique. La probabilité que ce résultat soit dû au hasard est quasi nulle.
- **Beta de 0.00** : selon le CAPM, un portefeuille à $\beta = 0$ devrait avoir $\alpha = 0$ (il ne rapporte que $r_f$). Avoir $\alpha = 0.73\%$ prouve que le CAPM est incapable d'expliquer ces rendements — c'est une **anomalie pure**. L'alpha reste stable à 0.73% quel que soit le modèle de contrôle (CAPM, 3 facteurs, 4 facteurs).
- **Sharpe ratio de 0.78** : supérieur à tous les déciles (0.70 pour P1, 0.28 pour P10). La stratégie BAB est mathématiquement plus efficiente que détenir un décile seul.
- **Flat SML** : les excess returns sont quasi-plats de P1 à P10 (0.91 à 0.97) — le marché ne rémunère pas le $\beta$ supplémentaire.
- **Volatilité BAB faible (10.75%)** : la neutralité au marché élimine le risque commun aux deux jambes.

Résultats répliqués dans 18 des 19 pays MSCI développés testés.

---

## 02 — Betting Against Correlation (BAC), Betting Against Volatility (BAV) & SMAX

*Asness, Frazzini, Gormsen & Pedersen — Journal of Financial Economics, 2020*

**Données** : 58 415 stocks dans 24 pays (MSCI World Developed), janvier 1926 – décembre 2015.

Le BAB est validé. Mais d'où vient l'anomalie exactement ? Le bêta est le produit de deux dimensions : $\hat{\beta}_i = \hat{\rho}_{im} \cdot (\hat{\sigma}_i / \hat{\sigma}_m)$. Le BAB les mélange. Les auteurs veulent les isoler.

### (i) Le problème : ρ et σᵢ sont corrélés

Empiriquement, $\hat{\rho}_{im}$ et $\hat{\sigma}_i$ sont positivement corrélés dans le cross-section (corrélation ≈ 0.33 en moyenne) : un titre qui bouge fort en absolu a tendance à aussi beaucoup bouger avec le marché. Trier sur $\rho$ seul capture donc aussi un effet $\sigma_i$ — impossible de savoir quelle dimension drive l'alpha. D'où deux facteurs séparés.

$$BAB \rightarrow BAC \text{ (isole } \hat{\rho}_{im}) + BAV \text{ (isole } \hat{\sigma}_i/\hat{\sigma}_m)$$

![[volatility_bab_decomp_schema.png]]

*Figure 3. BAB est décomposé en deux facteurs orthogonaux. BAC isole l'effet corrélation (moteur : contraintes d'effet de levier) et BAV isole l'effet volatilité relative (moteur : lottery demand).*

### (ii) Table II — justification empirique du double-tri (US, 1930–2015)

Les 25 portefeuilles issus du double-tri confirment que les deux dimensions contribuent indépendamment au $\beta$ et aux alphas.

**Panel A — Bêtas** : tri d'abord par $\hat{\sigma}_i$ (5 quintiles = colonnes), puis par $\hat{\rho}_{im}$ à l'intérieur (5 quintiles = lignes). Les t-stats ne sont reportés que pour les spreads LS (standard dans le papier) :

| | Vol Q1 | Vol Q2 | Vol Q3 | Vol Q4 | Vol Q5 | LS (corr) |
|---|---|---|---|---|---|---|
| Corr Q1 (faible) | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | **0.4** *(24.9)* |
| Corr Q2 | 0.7 | 0.9 | 1.0 | 1.1 | 1.2 | **0.5** *(26.2)* |
| Corr Q3 | 0.7 | 1.0 | 1.2 | 1.3 | 1.4 | **0.7** |
| Corr Q4 | 0.8 | 1.0 | 1.2 | 1.3 | 1.6 | **0.8** |
| Corr Q5 (haute) | 0.8 | 1.0 | 1.1 | 1.3 | 1.6 | **0.8** |
| **LS (vol)** | **0.4** *(8.3)* | **0.5** *(12.6)* | **0.5** *(15.3)* | **0.5** *(17.0)* | **0.7** *(21.1)* | |

**Panel B — CAPM alphas (%/mois)** :

| | Vol Q1 | Vol Q2 | Vol Q3 | Vol Q4 | Vol Q5 | LS (corr) |
|---|---|---|---|---|---|---|
| Corr Q1 (faible) | **0.4** *(5.6)* | **0.3** *(3.9)* | **0.2** *(3.2)* | **0.1** *(2.1)* | **0.1** *(2.3)* | **−0.3** *(−3.6)* |
| Corr Q2 | **0.3** *(3.3)* | **0.2** *(2.1)* | 0.1 *(1.6)* | 0.1 *(0.7)* | −0.1 *(−0.7)* | **−0.3** *(−3.0)* |
| Corr Q3 | **0.4** *(4.1)* | **0.3** *(2.8)* | 0.1 *(0.6)* | 0.0 *(−0.2)* | **−0.2** *(−2.3)* | **−0.6** *(−4.4)* |
| Corr Q4 | **0.4** *(3.3)* | **0.3** *(2.3)* | 0.0 *(0.3)* | −0.1 *(−1.0)* | **−0.3** *(−2.2)* | **−0.7** *(−4.2)* |
| Corr Q5 (haute) | **0.3** *(1.4)* | 0.1 *(0.2)* | 0.1 *(0.4)* | **−0.3** *(−1.7)* | **−0.5** *(−2.7)* | **−0.8** *(−3.3)* |
| **LS (vol)** | −0.1 *(−0.5)* | −0.2 *(−1.0)* | −0.2 *(−0.8)* | **−0.4** *(−2.2)* | **−0.6** *(−3.0)* | |

![[volatility_table2_doubletri.png]]

*Figure 4. Visualisation de la Table II Panel B. Chaque cellule représente l'alpha CAPM du portefeuille correspondant. La colonne LS (spread corr) est négative partout — les titres à haute corrélation sous-performent même à volatilité constante. La ligne LS (spread vol) est également négative dans les quintiles de haute volatilité.*

**Lecture clé** : regarde la colonne Vol Q3 (volatilité identique pour tous les titres de cette colonne). L'alpha va de +0.2 (Corr Q1) à −0.2 (Corr Q5). Même à $\sigma_i$ constant, l'anomalie de corrélation existe et est significative. C'est la preuve empirique que BAC capture quelque chose d'indépendant de la volatilité.

### (iii) Construction de BAC

BAC exploite uniquement la dimension $\hat{\rho}_{im}$, en contrôlant la volatilité via le double-tri.

**Procédure au début de chaque mois :**

1. **Tri par $\hat{\sigma}_i$** : tous les stocks → 5 quintiles de volatilité réalisée (1 an de returns quotidiens).
2. **Tri par $\hat{\rho}_{im}$ à l'intérieur** : dans chaque quintile de vol $q$, rank par corrélation (5 ans, returns 3 jours) → portefeuille low-corr ou high-corr.
3. **Pondération par rang** (*rank weighting*) : les poids sont proportionnels à l'écart au rang médian — plus la corrélation s'écarte de la médiane, plus le poids est élevé.
4. **Scaling $\beta = 1$** : les deux portefeuilles sont (de)leveragés pour avoir $\beta = 1$ à la formation, comme dans BAB.
5. **Agrégation** : BAC = moyenne equal-weighted des 5 facteurs BAC(q).

**Poids — rank weighting :**

$$w_H^q = k^q (z^q - \bar{z}^q)^+ \qquad w_L^q = k^q (z^q - \bar{z}^q)^-$$

où $z^q$ est le vecteur des rangs de corrélation dans le quintile $q$, $\bar{z}^q$ le rang moyen, et $k^q = 2 / \mathbf{1}^\prime |z^q - \bar{z}^q|$ normalise pour que les poids somment à 1. La notation $(x)^+$ désigne la partie positive d'un vecteur (valeurs négatives remplacées par 0) et $(x)^-$ la partie négative.

**Rendement BAC dans le quintile $q$ :**

$$r_{t+1}^{BAC(q)} = \frac{1}{\beta_t^{L,q}}\left(r_{t+1}^{L,q} - r^f\right) - \frac{1}{\beta_t^{H,q}}\left(r_{t+1}^{H,q} - r^f\right)$$

où $r_{t+1}^{L,q}$ = rendement du portefeuille low-corr dans le quintile $q$ (pondéré par $w_L^q$), $r_{t+1}^{H,q}$ = rendement du portefeuille high-corr (pondéré par $w_H^q$).

**Facteur BAC final :**

$$r_{t+1}^{BAC} = \frac{1}{5} \sum_{q=1}^{5} r_{t+1}^{BAC(q)}$$

### (iv) Construction de BAV

BAV est construit **exactement comme BAC, avec les rôles de corrélation et volatilité inversés** : tri d'abord par corrélation (5 quintiles = colonnes), puis tri par volatilité à l'intérieur de chaque quintile de corrélation.

**Rendement BAV dans le quintile $q$ (où $q$ est un quintile de *corrélation*) :**

$$r_{t+1}^{BAV(q)} = \frac{1}{\beta_t^{L,q}}\left(r_{t+1}^{L,q} - r^f\right) - \frac{1}{\beta_t^{H,q}}\left(r_{t+1}^{H,q} - r^f\right)$$

Ici $L$ = portefeuille **low-vol** (pondéré par rang de vol), $H$ = portefeuille **high-vol**, à corrélation constante — contrairement à BAC où $L$ = low-corr.

**Facteur BAV final :**

$$r_{t+1}^{BAV} = \frac{1}{5} \sum_{q=1}^{5} r_{t+1}^{BAV(q)}$$

<details>
<summary>Récapitulatif — différence entre BAC et BAV</summary>

| | BAC | BAV |
|---|---|---|
| Tri primaire (colonnes) | Volatilité $\hat{\sigma}_i$ | Corrélation $\hat{\rho}_{im}$ |
| Tri secondaire (long/short) | Corrélation $\hat{\rho}_{im}$ | Volatilité $\hat{\sigma}_i$ |
| Dimension isolée | $\hat{\rho}_{im}$ | $\hat{\sigma}_i / \hat{\sigma}_m$ |
| Théorie testée | Contraintes d'effet de levier | Lottery demand |
| Scaling | $\beta = 1$ pour chaque jambe | Idem |
| Agrégation | $\frac{1}{5}\sum_q BAC(q)$ | $\frac{1}{5}\sum_q BAV(q)$ |

</details>

### (v) Décomposition formelle : BAB = f(BAC, BAV)

Les auteurs vérifient que BAC et BAV expliquent entièrement BAB via la régression :

$$BAB_t = a_0 + a_1 \, BAC_t + a_2 \, BAV_t + \varepsilon_t$$

| Échantillon | $a_1$ (BAC) | $a_2$ (BAV) | $R^2$ | $a_0$ |
|---|---|---|---|---|
| US (1963–2015) | **0.71** | **0.51** | **85%** | ≈ 0 |
| Global | **0.84** | **0.49** | **96%** | ≈ 0 |

**Conclusion clé** : $R^2 = 96\%$ et les intercepts sont statistiquement nuls. Cela signifie que le BAB n'a **aucune existence propre** en dehors de BAC et BAV. C'est une coquille vide : enlevez l'effet "levier" (BAC) et l'effet "loto" (BAV), il ne reste plus rien.

### (vi) Résultats de BAC — Table IV (US, 1963–2015)

| Quintile vol | Q1 | Q2 | Q3 | Q4 | Q5 | BAC total |
|---|---|---|---|---|---|---|
| 5-factor alpha (%/mois) | **0.39** | **0.63** | **0.57** | **0.68** | **1.25** | **0.70** |
| *(t-stat)* | *(3.56)* | *(5.50)* | *(4.26)* | *(4.08)* | *(4.96)* | *(5.45)* |
| Sharpe ratio (ann.) | 0.60 | 0.90 | 0.87 | 0.81 | 0.80 | **0.93** |
| SMB loading | 0.62 | 0.61 | 0.58 | 0.58 | 0.61 | **0.60** |

Alpha de 0.70%/mois (t-stat 5.45), robuste au modèle 5 facteurs de Fama-French. SR annualisé = 0.93. Le loading élevé sur SMB (0.60) reflète que les titres à faible corrélation tendent à être plus petits.

### (vii) Lottery demand — intuition de BAV

> **Action "Bon Père de Famille" vs Action "Ticket de Loto".** L'Action Bon Père de Famille (utilities, basse vol) gagne 0.5% par mois régulièrement. L'Action Loto (biotech, crypto) ne gagne rien 11 mois sur 12, mais peut faire +500% en un mois. Les investisseurs surpaient l'Action Loto par biais psychologique. En moyenne, ces actions ont un rendement misérable car on paie "l'espoir". En vendant les actions volatiles (short BAV) et en achetant les ennuyeuses (long BAV), on encaisse une prime de régularité.

**Pourquoi ça marche aussi sur les large caps** : un titre ne devient "loto" que temporairement, quand sa volatilité explose — annonce de résultats, tweet du patron. Même Tesla ou Meta ont des phases "loto" où les retailers achètent en masse, poussant le prix au-delà de sa valeur fondamentale.

### (viii) BAV — facteurs comportementaux : LMAX, SMAX, IVOL

BAV est implémenté via trois facteurs comportementaux :

**LMAX** (*Low MAX factor*, Bali et al. 2016) : long les stocks avec le plus faible MAX (moyenne des 5 meilleurs returns quotidiens du mois), short les stocks avec le plus haut MAX. Construit via intersection de 6 portefeuilles triés sur taille et MAX.

**SMAX** (*Scaled MAX factor*) : LMAX scalé par la volatilité du stock, ce qui isole l'effet momentum de l'effet volatilité :

$$\text{SMAX}_{i,t} = \frac{\bar{r}^{\,\text{top-5}}_{i,\,t-1}}{\hat{\sigma}_{i,\,t-1}}$$

**IVOL** (*Idiosyncratic Volatility factor*, Ang, Hodrick, Xing & Zhang 2006) : volatilité résiduelle de la régression Fama-French des returns quotidiens du mois :

$$R_{i,t}^{ex} = \alpha_i + \beta_{1i} R_{m,t}^{ex} + \beta_{2i} \, \text{SMB}_t + \beta_{3i} \, \text{HML}_t + \varepsilon_{i,t}$$

Le résidu $\varepsilon_{i,t}$ est la volatilité idiosyncratique. Long low-IVOL, short high-IVOL. Aux US : facteurs Fama-French (1993) ; hors US : facteurs Asness & Frazzini (2013).

### (ix) Résultats et conclusion pratique

- BAC (dimension $\rho$) est corrélé avec la *margin debt* → valide les contraintes d'effet de levier
- SMAX et IVOL (dimension $\sigma_i$) sont corrélés avec le sentiment des investisseurs → valide la *lottery demand*
- Les deux effets coexistent de manière indépendante

<details>
<summary>Implications pour un gestionnaire de fonds</summary>

| Situation | Avant (BAB seul) | Avec BAC + BAV séparés |
|---|---|---|
| Vouloir du rendement | Acheter des actions risquées ($\beta$ élevé) | Acheter des actions ennuyeuses avec du levier |
| Chercher de l'alpha | Chercher le prochain Google | Vendre les titres "loto" surpayés |
| Contrôle des facteurs | BAB très exposé SMB (small caps) | BAC très exposé SMB, BAV moins — contrôle plus précis |

</details>

---

## 03 — Volatility-Managed Portfolios (VMP)

*Moreira & Muir — Journal of Finance, 2017*

### (i) Idée centrale

Peut-on améliorer **n'importe quel facteur existant** (market, SMB, HML, MOM…) en gérant dynamiquement son exposition selon sa volatilité récente ? Intuition : quand la volatilité est basse, le ratio rendement/risque est favorable → on leverage. Quand la volatilité est haute, on réduit l'exposition.

$$f_t^{*} = \underbrace{\frac{c}{\hat{\sigma}_{t-1}^{2}}}_{S_t} \cdot f_t$$

où $S_t = c / \hat{\sigma}_{t-1}^2$ est le scalaire de gestion, $c$ est une constante choisie pour que $\text{Var}(f_t^*) = \text{Var}(f_t)$ (variance normalisée), et $\hat{\sigma}_{t-1}^2$ est la variance estimée sur les ≈22 jours de trading du mois précédent.

![[volatility_vmp_scalaire.png]]

*Figure 5. Comportement du scalaire $S_t = c / \hat{\sigma}_{t-1}^2$. Quand la volatilité est basse (zone verte), $S_t > 1$ et le facteur est leveragé. Quand la volatilité est haute (zone rouge), $S_t < 1$ et l'exposition est réduite.*

### (ii) Résultat in-sample

Moreira & Muir montrent que $f_t^*$ améliore le Sharpe Ratio de nombreux facteurs in-sample via la régression :

$$f_t^{*} = \alpha + \beta \, f_t + \varepsilon_t$$

Si $\alpha > 0$, alors $f_t^*$ *expands the mean-variance frontier*. La question centrale est :

$$SR[f_t^{*}, f_t] > SR[f_t] \; ?$$

---

## 04 — Performance des VMP out-of-sample

*Cederburg, O'Doherty, Wang & Yan — Journal of Financial Economics, 2020*

### (i) Le biais ex-post de Moreira & Muir

Le résultat impressionnant de Moreira & Muir (2017) repose sur une faille : la constante $c$ est estimée sur toute la période d'étude. Cela signifie qu'on utilise des données futures pour construire la stratégie. Un investisseur réel ne connaît pas $c$ ex-ante. Le résultat in-sample est donc potentiellement trompeur.

### (ii) Procédure out-of-sample

Cederburg et al. estiment tous les paramètres uniquement sur les données passées via une rolling window de 120 mois :

![[volatility_oos_rolling.png]]

*Figure 6. Procédure out-of-sample à fenêtre roulante. Les 120 premiers mois servent à estimer $\hat{c}$, $\hat{\alpha}$, $\hat{\beta}$. Ces paramètres sont figés avant de calculer $f_{t+1}^*$. La fenêtre glisse d'un mois à chaque itération.*

1. Sur les 120 premiers mois : estimer $f_t^*$ → obtenir $\hat{c}$
2. Régression $f_t^* = \hat{\alpha} + \hat{\beta} f_t + \varepsilon_t$ → obtenir $[\hat{c}, \hat{\alpha}, \hat{\beta}]$
3. Figer les paramètres — on ne regarde plus jamais les données futures
4. Pour $t+1$ : calculer $f_{t+1}^*$ avec les paramètres figés
5. La fenêtre glisse d'un mois → série OOS $\{\hat{f}_{r=1}, \ldots, \hat{f}_{r=T-120}\}$

Test final :

$$SR[f_t^*, f_t] > SR[f_t] \; ?$$

### (iii) Résultat — la nuance OOS

| | In-sample (Moreira & Muir 2017) | Out-of-sample (Cederburg et al. 2020) |
|---|---|---|
| SR amélioré ? | Oui, nettement, pour la majorité des facteurs | **Résultat nuancé** — pas systématique |
| Biais | $c$ calculé ex-post sur toute la période | Estimé uniquement sur données passées |
| Conclusion | Impressionnant mais optimiste | Amélioration réelle mais plus modeste et variable |

Le message central : les VMP ne sont pas une martingale. L'amélioration du SR in-sample est robuste, mais out-of-sample les résultats dépendent du facteur, de la période, et du choix de la fenêtre d'estimation. C'est pourquoi le test OOS est devenu un **standard de rigueur incontournable** dans cette littérature — et la raison pour laquelle tout papier sur le timing de volatilité doit passer par cette procédure.

---

## Synthèse

Ces quatre papiers forment une progression logique et critique.

| Papier | Message central | Résultat clé |
|---|---|---|
| Frazzini & Pedersen (2014) | Anomalie low-risk inexplicable par le CAPM | $\alpha_{BAB} = 0.73\%$/mois, t-stat 7.44, $\beta = 0.00$ |
| Asness et al. (2020) | BAB est une coquille vide — deux sources distinctes | $BAB = 0.71 \cdot BAC + 0.51 \cdot BAV$, $R^2 = 85\%$ |
| Moreira & Muir (2017) | La gestion dynamique de vol améliore le SR | $\alpha > 0$ in-sample pour la majorité des facteurs |
| Cederburg et al. (2020) | Biais ex-post de Moreira & Muir | OOS : résultat réel mais plus modeste et variable |
