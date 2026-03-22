# CLAUDE.md — contexte du projet notes.quant

## Le projet

Site de notes personnelles de maths/stats/finance quantitative, hébergé sur GitHub Pages via Quartz.

URL : https://geilerloui.github.io/notes-quant
Repo : https://github.com/geilerloui/notes-quant (branche v4)

## Stack

- Quartz v4.5.2 (static site generator, Markdown + KaTeX)
- GitHub Actions pour le déploiement automatique
- Workflow : éditer les fichiers dans content/ → cd C:\Users\geile\mon-site → npx quartz sync
- IMPORTANT : toujours lancer npx quartz sync depuis la racine du projet, pas depuis un sous-dossier

## Design

- Typo : Lora (body), DM Sans (headings)
- Palette : fond crème #f7f5f0, accent bordeaux #c0392b
- CSS custom : quartz/styles/custom.scss
- Images : max-width 75%, dans content/images/<nom-chapitre>/
- Les chemins d'images sont résolus depuis la racine de content/ — utiliser images/<dossier>/im1.png

## Structure du contenu
```
content/
├── 1-probabilites/
│   └── geometrie-L2.md
├── 2-statistiques/
│   ├── regression-lineaire.md
│   ├── serie-temporelle.md
│   └── volatilite.md           (squelette vide)
├── 3-apprentissage-automatique/
│   └── fondation.md
├── 4-asset-pricing/
│   ├── markowitz.md
│   ├── capm.md
│   ├── anomalies.md
│   └── modeles-factoriels/
│       ├── facteurs-fondamentaux.md  (FF3 + Novy-Marx)
│       ├── facteurs-macro.md         (squelette vide)
│       ├── facteurs-barra.md         (squelette vide)
│       └── ML factor investing.md    (squelette vide)
├── 5-produits-derives/               (vide)
└── images/
    ├── geometrie-L2/     (im1–im4)
    ├── regression-lineaire/ (im1–im4)
    ├── serie-temporelle/ (im1–im5)
    ├── fondation/        (im1–im5)
    ├── markowitz/        (im1–im6)
    ├── capm/             (im1–im7)
    ├── anomalies/        (im1–im4)
    └── facteurs-fondamentaux/ (SVG + PNG divers)
```

## Conventions

- LaTeX : $...$ inline, $$...$$ display
- Captions : *Figure N. Description complète avec point final.*
- Pas de LaTeX dans les captions à l'intérieur des div HTML
- Markdown pur pour les captions hors div HTML
- Titres de sections : ## pour les grandes parties, ### (i) (ii) (iii) pour les sous-parties
- Dropdowns : balise <details><summary>Titre</summary>contenu</details>
- Tester que le KaTeX se rend bien dans les <details>

## Contenu couvert

### 1. Géométrie dans L² (1-probabilites/geometrie-L2.md)
- L'idée fondatrice : dictionnaire Rⁿ ↔ L²
- L'espace L² et Hilbert
- (i) E(X) = projection sur Δ
- (ii) ρ = cos θ entre X̃ et Ỹ centrés
- (iii) E(Y|X) = projection sur L²_X, variance totale = Pythagore

### 2. Régression linéaire (2-statistiques/regression-lineaire.md)

**Chapitre 1 — Le modèle**
- (i) Vue classique — résidus comme distances verticales
- (ii) Vue géométrique dans L² — Y = E(Y|X) + ε, orthogonalité
- (iii) Vue probabiliste — à venir
- (iv) Dérivation des estimateurs MCO (= OLS) par orthogonalité
- (v) Interprétation des coefficients — ceteris paribus, corrélations partielles dans le cas gaussien

**Chapitre 2 — Diagnostic** (généralisé à la régression multiple)
- (i) Hypothèses de Gauss-Markov H1–H5
- (ii) BLUE — Best Linear Unbiased Estimator
- (iii) Multicolinéarité — VIF, Ridge/Lasso en mention
  - 📌 À compléter : lien géométrique R² ajusté avec agrandissement L²_X

**Chapitre 3 — Inférence & Tests**
- (i) Distribution de β̂ ~ N(β, σ²/S_XX)
- (ii) R² = cos²θ, SCT = SCE + SCR, R² ajusté, 3 pièges
- (iii) Test de Student, règle 1.96, p-value, IC sur β

**Chapitre 4 — Prédiction**
- (i) Intervalle de confiance sur E(Y|X=x*)
- (ii) Intervalle de prédiction sur Y(x*)
- (iii) Pourquoi IP > IC — le +1 irréductible

### 3. Séries temporelles (2-statistiques/serie-temporelle.md)
- Cadre : µ_t et σ_t² conditionnelles à F_{t-1}
- (i) White noise — surprise pure, E[ε_t|F_{t-1}] = 0
- (ii) Stationnarité au sens large — définition + random walk comme contre-exemple
- (iii) Processus AR(p) — propagation des chocs, racines unitaires, stationnarité
- (iv) Processus MA(q) — chocs passés, toujours stationnaire
- (v) Processus ARMA(p,q)
- (vi) ACF et PACF — identifier le bon modèle (règles de lecture)

### 4. Volatilité (2-statistiques/volatilite.md)
- Squelette vide — à rédiger (GARCH, ARCH, volatilité stochastique)

### 5. Fondations du ML (3-apprentissage-automatique/fondation.md)
- Inférer vs prédire
- EPE = erreur réductible + σ² irréductible
- Décomposition biais-variance (dérivation, interprétation, trade-off)
- Régularisation : Ridge (L2), Lasso (L1), différence géométrique
- Double descent — au-delà du trade-off classique
- Interpolation vs extrapolation
- Évaluation : train/val/test, K-fold, walk-forward (séries temporelles), data leakage

### 6. Markowitz (4-asset-pricing/markowitz.md)
- (i) Fonction d'utilité u(W), aversion au risque, concavité
- (ii) Prime de risque, équivalent certain, Arrow-Pratt
- (iii) Von Neumann & Morgenstern (1944)
- (iv) Rupture de Markowitz (1952)
- (v) Richesse finale et rendement du portefeuille
- (vi) Hypothèse quadratique et score d'utilité
- (vii) Pourquoi la diversification marche
- (viii) Allocation optimale
- (ix) Frontière efficiente sans actif sans risque + min variance portfolio
- (x) Avec actif sans risque — CML (Capital Market Line)
- (xi) Ratio de Sharpe et portefeuille tangent
- (xii) Théorème des deux fonds

### 7. CAPM / MEDAF (4-asset-pricing/capm.md)
- 1. De Markowitz au CAPM : Single Index Model, équilibre de marché T = marché
- 2. Formule CAPM, interprétation β, Security Market Line (SML)
- 3. Décomposition risque total : systématique (β²σ²_m) + idiosyncratique
- 4. Alpha — deux usages : ex ante (mispricing) vs Jensen (performance)
- 5. Estimer E[R_i] : target price, Gordon-Shapiro, DCF
- 6. Usages pratiques : screener, WACC, évaluation perf, hedge, market timing
- 7. Limitations
- 8. Tests empiriques : BJS (1972), Fama-MacBeth (1973), Roll (1977)

### 8. Anomalies (4-asset-pricing/anomalies.md)
- Contexte : CRSP, années 1980, recherche d'alpha au-delà de β
- Méthode des portefeuilles triés (sorted portfolios)
- (ii) Effet taille — Banz (1981)
- (iii) Effet janvier — Keim & Roll (1983)
- (iv) Effet value — Rosenberg, Reid & Lanstein (1985)
- (v) Momentum — Jegadeesh & Titman (1993)
- (vi) Synthèse — ce que le CAPM ne peut pas expliquer

### 9. Modèles factoriels — Facteurs fondamentaux (4-asset-pricing/modeles-factoriels/facteurs-fondamentaux.md)
**Fama-French 3 Facteurs**
- 1. Échec empirique du CAPM (relation β-rendement plate)
- 2. Variables : taille (SmB) + value (HmL), rappel comptable bilan
- 3. Construction des facteurs SmB et HmL (6 portefeuilles 2×3)
- 4. Modèle FF3 et régression, interprétation des loadings
- 5. Résultats empiriques
- 6. Interprétation et débat (risque vs mispricing)
- 7. Applications pratiques

**Novy-Marx — Gross Profitability**
- (i) Anomalie dans les données : GP/A prédit les rendements
- (ii) Justification théorique (Miller-Modigliani, valeur de croissance)
- (iii) Choix du proxy GP/A
- (iv) Preuve empirique — Fama-McBeth Table 1
- (v) Corrélations et portefeuille mixte value+profitability

### 10. Modèles factoriels — Macro / BARRA / ML (squelettes vides)
- facteurs-macro.md, facteurs-barra.md, ML factor investing.md — à rédiger

## Prochains chapitres prévus

- 2-statistiques/ : volatilite.md (GARCH, ARCH)
- 3-apprentissage-automatique/ : reduction-dimension.md (PCA)
- 4-asset-pricing/modeles-factoriels/ : facteurs-macro.md, facteurs-barra.md, ML factor investing.md
- 5-produits-derives/ : introduction.md, black-scholes.md