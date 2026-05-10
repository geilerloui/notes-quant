---
title: Volterra Signatures (placeholder à creuser)
date: 2026-05-10
tags: [probabilités, processus-stochastiques, volterra, signatures, à-développer, placeholder]
---

> [!warning] 🔖 Placeholder — note à développer un jour
> Cette note est un **marqueur** pour ne pas oublier le sujet. À creuser quand j'ai un projet ML / kaggle qui le justifie empiriquement (mémoire longue avérée dans les données).

## Pourquoi je note ça

Sujet repéré pendant le travail sur [[06_Rough Paths]]. Potentiellement utile pour :
- **Kaggle de prédiction de volatilité** (Optiver, JPX, etc.) où la mémoire longue est avérée
- **Feature engineering pour séries temporelles à mémoire longue** (order flow, signaux macro, vol réalisée)
- **Calibration ML de modèles rough volatility** (rough Heston, rough Bergomi)

À ne **pas** creuser si :
- Les données sont markoviennes / mémoire courte (rendements d'actifs, prix observés)
- L'autocorrélation décroît exponentiellement → lags classiques + EWMA suffisent

## L'idée centrale (résumé)

### Processus de Volterra

Généralisation du brownien :

$$X_t = \int_0^t K(t, s)\,dW_s$$

où $K(t, s)$ est un **noyau déterministe** qui encode la structure de mémoire. Cas particuliers :
- $K(t, s) = 1$ → brownien standard
- $K(t, s) = e^{-\theta(t-s)}$ → OU stationnaire
- $K(t, s) = (t-s)^{H-1/2}$ → fBm (cf [[06_Rough Paths]] §II)

### Volterra signature

Extension de la **path signature** classique (cf [[06_Rough Paths]] §V.2) : au lieu de calculer les intégrales itérées brutes du chemin, on les pondère par le noyau de Volterra :

$$\int_0^t K(t, r_1) \int_0^{r_1} K(r_1, r_2) \ldots dX_{r_n}$$

Ces features captent **directement la structure de mémoire** que le modèle ML doit respecter. Pour le noyau $K_H$, on parle de **rough signature** ou **fractional signature**.

## Pourquoi ça pourrait booster un score Kaggle vol

**Hypothèse à valider empiriquement** : si la vol réalisée a une autocorrélation en loi de puissance (ce qui est connu depuis Bayer-Friz-Gatheral 2016, $H \approx 0.1$), alors des features qui encodent cette structure de mémoire **devraient** mieux performer que des features classiques (lags, rolling stats) ou la signature naïve.

**Avant de coder Volterra signature**, tester :
1. **Autocorrélation de la target** sur 100-500 lags, en log-log
2. Si décroissance exponentielle → laisser tomber, lags classiques suffisent
3. Si décroissance en loi de puissance → essayer dans l'ordre :
   - **EWMA avec plusieurs $\alpha$** (couvre des échelles de mémoire variées) — gain rapide
   - **Rolling stats sur fenêtres en loi de puissance** (1, 5, 22, 100, 500)
   - **Path signature classique** (avec `signatory` ou `iisignature`)
   - **Volterra signature** seulement si tout ce qui précède plafonne

## Bibliographie à explorer

**Théorie** :
- Bayer-Friz-Gassiat-Martin-Stemper (2019) — *A regularity structure for rough volatility*
- Cuchiero-Salvi (2020+) — papers sur Volterra signatures et applications ML

**Applications ML** :
- Bayer-Stemper (2019) — *Deep calibration of rough stochastic volatility models*
- Horvath-Muguruza-Tomas (2021) — *Deep learning volatility*
- Cuchiero-Möller-Svaluto-Ferro-Schmocker (2024) — extensions récentes

**Implémentations** :
- Pas de librairie standardisée (contrairement à `signatory` pour la signature classique). Code à écrire ou récupérer dans des repos académiques.

## Ce qui manque dans cette note

Si je décide un jour de transformer ce placeholder en vraie note :

- [ ] Définition rigoureuse des processus de Volterra (espace, conditions sur le noyau)
- [ ] Construction explicite de la Volterra signature avec exemples
- [ ] Théorèmes universels (équivalent du théorème de Hambly-Lyons pour Volterra)
- [ ] Implémentation Python d'un calcul simple sur synthetic data
- [ ] Benchmark sur un dataset Kaggle vol (Optiver) vs signature classique
- [ ] Figures comparant noyaux Volterra et leur effet sur la mémoire

## Statut

🔖 **Placeholder, pas une vraie note**. À reprendre quand :
- Un projet concret le justifie (Kaggle vol, projet pro Schneider sur smart grid avec mémoire longue, etc.)
- J'ai du temps pour creuser sérieusement (3-5 jours de travail)
- Les outils ont mûri (peut-être qu'une librairie standardisée émergera)

---

## Suite logique

**Précédent ← [[07_Contrôle Stochastique]]** : on vient de voir HJB et le contrôle d'EDS. Ici on revient au feature engineering pour ML, en généralisant la signature de [[06_Rough Paths]].

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998), path signature
7. [[07_Contrôle Stochastique]] — HJB, Merton, optimal execution, lien RL
8. **[[08_Volterra Signatures]]** — (placeholder) extension de la signature pour processus à mémoire longue
