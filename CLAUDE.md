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
│   ├── geometrie-L2.md       (FAIT)
│   └── images/geometrie-L2/  (im1.png à im4.png)
├── 2-statistiques/
│   ├── regression-lineaire.md (FAIT)
│   └── images/regression-lineaire/ (im1.png à im4.png)
├── 3-apprentissage-automatique/
├── 4-asset-pricing/
└── 5-produits-derives/
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

## Prochains chapitres prévus

- 2-statistiques/ : ar-ma.md, garch.md
- 3-apprentissage-automatique/ : reduction-dimension (PCA)
- 4-asset-pricing/ : markowitz.md, capm.md, fama-french.md
- 5-produits-derives/ : introduction.md, black-scholes.md