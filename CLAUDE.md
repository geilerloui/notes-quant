\# CLAUDE.md — contexte du projet notes.quant



\## Le projet

Site de notes personnelles de maths/stats/ML, hébergé sur GitHub Pages via Quartz.

URL : https://geilerloui.github.io/notes-quant

Repo : https://github.com/geilerloui/notes-quant (branche v4)



\## Stack

\- Quartz v4.5.2 (static site generator, Markdown + KaTeX)

\- GitHub Actions pour le déploiement automatique

\- Workflow : éditer les fichiers dans content/ → npx quartz sync



\## Design

\- Typo : Lora (body), DM Sans (headings)

\- Palette : fond crème #f7f5f0, accent bordeaux #c0392b

\- CSS custom : quartz/styles/custom.scss

\- Images : max-width 75%, dans content/images/<chapitre>/



\## Structure du contenu

\- content/geometrie-L2.md — Géométrie dans L² (FAIT)

&#x20; - Images : content/images/geometrie-L2/im1.png à im4.png

\- À venir : régression linéaire (vue géométrique, MCO, R², prédiction)



\## Conventions

\- LaTeX : $...$ inline, $$...$$ display

\- Captions : \*Figure N. Description complète avec point final.\*

\- Images dans les div flex HTML : pas de LaTeX dans les <p> caption (ne se rend pas)

\- Markdown pur pour les captions hors div HTML

\- Titres de sections : ## pour les grandes parties, ### (i) (ii) (iii) pour les sous-parties



\## Contenu couvert

1\. Géométrie dans L²

&#x20;  - L'idée fondatrice : dictionnaire Rⁿ ↔ L²

&#x20;  - L'espace L² et Hilbert

&#x20;  - (i) E(X) = projection sur Δ

&#x20;  - (ii) ρ = cos θ entre X̃ et Ỹ centrés

&#x20;  - (iii) E(Y|X) = projection sur L²\_X, variance totale = Pythagore



\## Prochain chapitre

Régression linéaire — vue géométrique (y = Xβ comme projection sur col(X), hat matrix, MCO)

