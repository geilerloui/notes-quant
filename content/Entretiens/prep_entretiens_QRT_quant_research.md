# Plan de prép — Entretiens QRT Quant Research

> Objectif : couvrir les thèmes des rounds techniques QRT, en priorisant le **vrai chantier** (proba) et en capitalisant sur tes **atouts** (ML, stats, finance/microstructure).
> Légende statut : 🟢 tu maîtrises (à structurer) · 🟡 à rafraîchir · 🔴 chantier prioritaire

---

## Le process (rappel)

1. **Screen / OA** — ✅ fait (85%)
2. **Réexplication du code + 1er entretien technique** (quant researcher senior) — proba, Python, régression, + live coding type LeetCode
3. **Entretien(s) technique(s)** (managers) — maths, stats, analyse de données, ML, parfois projet
4. **Management / team fit** — quelle équipe, culture

Compter ~3-4 rounds techniques + RH + final. Les thèmes ci-dessous sont **répartis** dans ces rounds, pas un round par thème.

---

## 1. Probabilité & brain teasers analytiques — 🔴 CHANTIER PRIORITAIRE

C'est ton vrai trou (« j'ai pas révisé les maths »). À travailler en premier.

- **Bases** : espérance, variance, covariance ; linéarité de l'espérance (l'outil le plus rentable des brain teasers)
- **Probas conditionnelles & Bayes** : formule de Bayes, indépendance, loi des probabilités totales
- **Combinatoire** : arrangements, combinaisons, principe d'inclusion-exclusion
- **Lois classiques** : uniforme, binomiale, géométrique, Poisson, exponentielle, normale — savoir quand chacune s'applique et leurs espérances/variances par cœur
- **Expected value games** : dés, cartes, pièces, paris — calculer la valeur d'un jeu, stratégies optimales
- **Brain teasers types** : tirages avec/sans remise, marches aléatoires, problème du secrétaire, paradoxes (Monty Hall, anniversaires), espérance du nombre de tirages jusqu'à un événement
- **Niveau supérieur** (si ça va loin) : espérance conditionnelle, martingales, chaînes de Markov simples

**Méthode (clé) :** raisonner **à voix haute**, poser tes hypothèses, structurer. Les brain teasers testent ta démarche sous pression, pas le résultat seul.

---

## 2. Statistiques & régression — 🟢 tu maîtrises, à structurer pour l'oral

Confirmé chez QRT : OLS, multicolinéarité, ridge vs lasso.

- **Régression linéaire / OLS** : les hypothèses de Gauss-Markov, ce qui casse si elles tombent, interprétation des coefficients
- **Multicolinéarité** : détection (VIF, corrélations), impact sur l'instabilité/l'interprétation des coefficients
- **Régularisation** : ridge (L2) vs lasso (L1) — quand l'un, quand l'autre, pourquoi lasso fait de la sélection de variables
- **Biais-variance** : le tradeoff, et comment la régularisation le déplace
- **Inférence** : tests d'hypothèses, p-values, intervalles de confiance, R²
- **Estimateurs** : biais, variance, consistance, maximum de vraisemblance
- **Séries temporelles** : stationnarité, AR/MA/ARMA, autocorrélation
- **Bayésien** : prior/posterior, l'intuition (ton vault couvre ça)

---

## 3. Machine Learning — 🟢 ton terrain (PhD)

Ne pas réviser le fond, juste **savoir l'expliquer simplement et vite**.

- Arbres, random forests, boosting (tu maîtrises — ton vault)
- Overfitting, régularisation, cross-validation, fuite de données
- Métriques (selon contexte : RMSE, AUC, precision/recall)
- Feature engineering & sélection
- Le piège classique : expliquer un concept avec une **intuition claire**, pas une définition de manuel

---

## 4. Mathématiques — 🟡 à rafraîchir

Confirmé chez QRT : algèbre linéaire, calcul.

- **Algèbre linéaire** : valeurs/vecteurs propres, matrices symétriques définies positives, projections, décompositions (SVD), interprétation géométrique
- **Calcul / analyse** : dérivées, intégrales, optimisation, un peu d'équations différentielles
- **Optimisation** : convexité, descente de gradient, multiplicateurs de Lagrange

---

## 5. Python — connaissance fine du langage — 🟡 à rafraîchir

Pas de l'algo : la mécanique « interne » du langage (mentionnée dans les retours QRT).

- **Generators** (`yield`), itérateurs, lazy evaluation
- **GIL** (Global Interpreter Lock) : ce que c'est, son impact sur le threading
- **Decorators** : comment ça marche, écrire un decorator simple
- **Structures** : différences list / tuple / set / dict, et leurs complexités
- `@staticmethod` vs `@classmethod` vs méthode d'instance
- Compréhensions, `lambda`, `map`/`filter`
- **pandas / numpy** : manipulation, vectorisation, groupby

---

## 6. Finance & microstructure de marché — 🟢 TON ARME (Morgan Stanley)

C'est ta différenciation rare. À transformer en récit solide.

- **Microstructure** : carnet d'ordres, bid-ask spread, market impact, liquidité
- **Factor models (Barra)** : ton expérience directe — sache l'expliquer de bout en bout
- **Market impact / TCA / cross-impact** : ton expertise MS, sous-exploitée jusqu'ici
- **Récit projets** : prépare 2 histoires solides (1) un projet MS (factor models OU market impact), (2) le challenge ENS — avec le problème, ton approche, les choix, les résultats, ce que tu ferais différemment

---

## 7. Live coding (LeetCode-style) — 🟢 réamorcé aujourd'hui

Les patterns algo servent ici (pas dans le screen). Maintien :

- Garde la fiche des 10 patterns sous la main
- Refais 1-2 problèmes par jour pour ne pas re-rouiller
- Réflexes : brute force d'abord, vérifier `i=0`, le `==`, le `return`, ne pas sur-construire, raisonner à voix haute

---

## Priorisation (ordre conseillé)

1. **Proba & brain teasers** (🔴) — le seul vrai trou, le plus rentable
2. **Python interne** (🟡) — rapide à rafraîchir, gros ROI
3. **Maths** (🟡) — algèbre linéaire surtout
4. **Récit finance/MS** (🟢) — à structurer, c'est ton arme
5. **Stats & ML** (🟢) — révision légère, tu les as
6. **Live coding** — entretien d'1-2 problèmes/jour en fond

---

## Ressources clés

- **Proba / brain teasers** : *A Practical Guide to Quant Finance Interviews* (Xinfeng Zhou — « le green book ») ; *Heard on the Street* (Crack) ; les puzzles Jane Street
- **Questions quant générales** : *150 Most Frequently Asked Questions on Quant Interviews* (Stefanica, Radoičić, Wang)
- **Stats / ML** : tu as déjà ton vault (ESL-niveau) — révision ciblée seulement
- **Entraînement actif** : faire les problèmes en **timed + à voix haute**, pas en lecture passive
