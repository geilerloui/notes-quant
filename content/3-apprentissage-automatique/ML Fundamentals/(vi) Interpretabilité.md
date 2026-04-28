# (vi) Interprétabilité

> Comprendre **pourquoi** un modèle fait telle prédiction, et plus généralement quelles features influencent la sortie. Deux niveaux : **global** (le modèle dans son ensemble) et **local** (une prédiction donnée).

---

## 1. Pourquoi se soucier de l'interprétabilité

- **Réglementaire** : credit scoring, médical, justice prédictive. Le client/patient/justiciable a le droit de savoir pourquoi il a été refusé/diagnostiqué/condamné.
- **Debugging** : un modèle peut avoir une performance excellente sur un mauvais signal (target leakage, biais dans les données). L'interprétabilité aide à le détecter.
- **Confiance utilisateur** : un trader ne fera pas confiance à un signal qu'il ne comprend pas, même très performant.
- **Découverte scientifique** : comprendre quelles features sont prédictives peut révéler des patterns métier inconnus.

---

## 2. Interprétabilité globale

### 2.1 Modèles intrinsèquement interprétables

- **Régression linéaire/logistique** : les coefficients $\beta_j$ se lisent directement (effet d'une augmentation d'une unité de $x_j$ sur $y$).
- **Arbres de décision** : règles si-alors lisibles à condition de rester peu profonds.
- **GAM** (Generalized Additive Models) : $y = f_1(x_1) + f_2(x_2) + \ldots$, on peut tracer chaque $f_j$.

> **Attention aux régressions linéaires** : interprétation des coefficients valable uniquement si features standardisées et indépendantes. La multicolinéarité fait n'importe quoi avec les signes.

### 2.2 Feature importance

Plusieurs méthodes pour ranker les features par contribution au modèle.

- **Importance des arbres** (RF, gradient boosting) : moyenne de la réduction d'impureté (Gini ou MSE) qu'apporte chaque split sur cette feature.
   - **Piège** : biais en faveur des features à haute cardinalité.
- **Permutation importance** : on shuffle une feature et on mesure la dégradation de performance. Plus honnête, model-agnostic, mais coûteux.
- **Coefficients standardisés** (régression linéaire) : $|\beta_j| \cdot \text{std}(x_j)$.

### 2.3 Partial Dependence Plot (PDP)

*À développer.*

Pour une feature $x_j$, on trace la prédiction moyenne du modèle $\hat{f}(x_j, x_{-j})$ en faisant varier $x_j$ et en moyennant sur les autres features. Donne la forme fonctionnelle de la dépendance.

- **Limite** : suppose les features indépendantes (sinon on évalue le modèle sur des combinaisons jamais vues).
- **Variante ICE** (Individual Conditional Expectation) : tracer une courbe par observation au lieu de moyenner. Révèle l'hétérogénéité des effets.

---

## 3. Interprétabilité locale

### 3.1 SHAP

*À développer.*

Décomposition de chaque prédiction individuelle comme somme des contributions de chaque feature, basée sur les valeurs de Shapley de la théorie des jeux. Garantit additivité et cohérence.

$$\hat{f}(x) = \mathbb{E}[\hat{f}(X)] + \sum_{j=1}^p \phi_j(x)$$

où $\phi_j(x)$ est la contribution de la feature $j$ à la prédiction sur $x$.

- **Standard de fait** aujourd'hui pour expliquer une prédiction individuelle.
- Implémentation efficace pour les modèles à arbres : `shap.TreeExplainer`.
- Coûteux pour les autres modèles (`KernelExplainer`).

### 3.2 LIME

*À développer.*

Idée : autour d'un point $x$, on génère des perturbations, on regarde comment le modèle réagit, et on fit un modèle simple (linéaire) **localement**. Les coefficients du modèle local donnent l'explication.

- Plus rapide que SHAP mais moins théoriquement fondé.
- Sensible au choix du voisinage et aux perturbations.

---

## 4. Interprétabilité ≠ causalité

> **Piège fondamental.** Toutes ces méthodes donnent des **corrélations conditionnelles**, pas des effets causaux. Si une feature a une grande importance SHAP, ça ne veut **pas** dire qu'agir sur elle changera la sortie en production. Pour ça, il faut un cadre causal (DAG, do-calculus, expérimentation).
