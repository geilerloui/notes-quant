# (v) Inférence prédictive

> *Predictive inference* = ce qu'on peut dire de **fiable** sur la prédiction d'un point individuel, au-delà du score brut. Deux outils principaux qui ne servent pas exactement la même chose : **calibration** et **conformal prediction**.

---

## 1. Calibration

https://arxiv.org/html/2501.19047v2

### 1.1 Le problème

Beaucoup de modèles de classification produisent des "probabilités" qui ne sont **pas vraiment des probabilités**. Un modèle qui dit $\hat{p} = 0.9$ pour 100 clients devrait, si bien calibré, voir environ 90 d'entre eux être réellement positifs. Ce n'est pas garanti.

- **Logreg** (sans repondération) : naturellement bien calibrée
- **Random Forest, SVM, Naive Bayes** : mal calibrés (scores poussés vers les extrêmes ou écrasés au milieu)
- **Neural networks profonds** : tendance à l'**overconfidence** (prédisent 0.99 souvent à tort)

### 1.2 Diagnostic : reliability diagram

*À développer.*

On bucketise les prédictions par intervalles de probabilité (ex : [0, 0.1], [0.1, 0.2], …) et on trace la fréquence empirique des positifs dans chaque bucket vs le centre du bucket. Un modèle parfaitement calibré tombe sur la diagonale.

### 1.3 Méthodes de recalibration

*À développer.*

- **Platt scaling** : ajuste une logreg sur les scores du modèle. Hypothèse de forme sigmoïdale.
- **Isotonic regression** : non-paramétrique, plus flexible, mais a besoin de plus de données pour être stable.
- **Temperature scaling** (NN) : un seul paramètre $T$ qui divise les logits avant softmax.

> **Quand calibrer ?** Quand on a besoin d'une **PD interprétable** (credit scoring, risk management, médical) ou pour combiner les sorties de plusieurs modèles.

---

## 2. Conformal prediction

### 2.1 Le problème

Calibration donne un score qu'on peut interpréter en moyenne. **Conformal prediction donne autre chose** : pour chaque point individuel, un **intervalle de prédiction** (régression) ou un **ensemble de classes possibles** (classification) avec une **garantie statistique** :

$$P(y_{\text{vrai}} \in C(x_{\text{nouveau}})) \geq 1 - \alpha$$

valable **sans hypothèse sur la distribution**, juste l'échangeabilité (≈ i.i.d.).

### 2.2 Idée

*À développer.*

1. Entraîner le modèle sur le train.
2. Calculer un score de non-conformité (ex : $|y - \hat{y}|$ en régression) sur un set de calibration séparé.
3. Pour un nouveau point, l'intervalle de prédiction est centré sur $\hat{y}$ avec largeur = quantile $1-\alpha$ des scores de calibration.

### 2.3 Différence avec la calibration

| | Calibration | Conformal prediction |
| :--- | :--- | :--- |
| Question | "Mon $\hat{p}$ correspond-il à la vraie probabilité ?" | "Sur quelle plage de valeurs la vraie sortie a-t-elle de fortes chances d'être ?" |
| Type de garantie | Marginale, en moyenne | Marginale, en moyenne sur la calibration |
| Sortie | Score recalibré | Intervalle ou ensemble de classes |
| Hypothèse | Échantillon de calibration | Échangeabilité |

Les deux peuvent se combiner : modèle calibré → conformal sur les scores calibrés → intervalle plus serré.
