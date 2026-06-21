---
title: Algorithmes de Base
description: Régression linéaire et logistique en mode prédictif, kNN, SVM — vue ML plutôt que statistique
---

# Algorithmes de Base

> Cette note couvre quatre algorithmes fondamentaux côté **prédiction** plutôt que côté **inférence**. Pour la régression linéaire/logistique, le détail inférentiel complet (hypothèses Gauss-Markov, tests, intervalles de confiance) est dans [[01_Régression Linéaire Multiple]] et [[02_Régression logistique]] — ici on regarde juste "comment ça prédit et pourquoi", en restant proche du langage ML plutôt que stats.

---

## A. Régression linéaire — vue ML

On cherche $\hat f(x) = \hat\beta_0 + \hat\beta^T x$ qui minimise l'erreur quadratique moyenne sur le training set :

$$\hat\beta = \arg\min_\beta \frac{1}{n}\sum_{i=1}^n (y_i - \beta_0 - \beta^T x_i)^2$$

C'est tout. Pas d'hypothèse sur les résidus, pas de test, pas d'intervalle de confiance — juste un problème d'optimisation convexe avec une solution fermée $\hat\beta = (X^TX)^{-1}X^Ty$ (cf. [[01_Régression Linéaire Multiple#C. Estimation sur échantillon]] pour la dérivation complète).

> [!warning] Ce qu'on garde en mode ML
> - **L'objectif** : minimiser le risque empirique (MSE)
> - **Le modèle** : linéaire en les paramètres
> - **L'évaluation** : MSE/RMSE sur un test set, pas un $R^2$ ajusté ou des p-values
>
> Ce qu'on **laisse de côté** : hypothèses Gauss-Markov, tests de Student/Fisher, diagnostics de résidus. Si l'objectif est purement prédictif, rien de tout ça n'est nécessaire — c'est utile uniquement si on veut interpréter $\hat\beta$ comme un effet causal ou faire de l'inférence.

**Le lien avec le réseau de neurones.** Une régression linéaire est un MLP **sans couche cachée** : une seule couche, fonction d'activation identité, loss MSE. C'est le cas le plus simple du spectre de modèles vu en [[00_Perceptron Multi-Couches]] — zéro non-linéarité, zéro composition de fonctions.

$$\text{Régression linéaire} \;=\; \text{MLP}\big(L=0,\ g(z) = z,\ \mathcal{L} = \text{MSE}\big)$$

> [!note]- Pourquoi un MLP à 1 couche cachée généralise la régression linéaire
> Si tu mets une couche cachée avec activation **linéaire** (identité), la composition de deux transformations linéaires reste linéaire — $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$ est encore une fonction affine de $x$. Donc empiler des couches linéaires n'ajoute **aucune expressivité** par rapport à une seule couche — c'est seulement la non-linéarité (ReLU, tanh, sigmoid) qui rend les couches profondes utiles. La régression linéaire est donc le point de départ naturel de tout le spectre des MLP.

**Quand utiliser ça en prédiction pure.** Quand on suspecte une relation globalement linéaire, ou comme **baseline rapide** avant d'essayer des modèles plus complexes — toujours comparer un gradient boosting ou un réseau à une régression linéaire simple en baseline, c'est souvent suffisant et ça donne un point de référence pour juger si la complexité supplémentaire est justifiée.

---

## B. Régression logistique — vue ML

Même logique que A, mais pour la classification binaire. On modélise une probabilité via la sigmoïde :

$$\hat p(x) = \sigma(\beta_0 + \beta^T x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

et on minimise la **log-loss** (cross-entropy binaire) au lieu de la MSE :

$$\hat\beta = \arg\min_\beta -\frac{1}{n}\sum_{i=1}^n \big[y_i \log \hat p(x_i) + (1-y_i)\log(1-\hat p(x_i))\big]$$

Pas de solution fermée (contrairement à OLS) — on résout par descente de gradient ou Newton-Raphson (IRLS). Le détail probabiliste complet (odds ratio, MLE, tests de Wald) est dans [[02_Régression logistique]].

> [!warning] Ce qu'on garde en mode ML
> - **La frontière de décision** : $\beta_0 + \beta^T x = 0$ — un **hyperplan**. La régression logistique est un classifieur **linéaire**.
> - **La sortie** : une probabilité $\hat p(x) \in [0,1]$, pas juste une classe — utile pour ranker, calibrer un seuil, ou combiner avec d'autres scores.
> - **L'évaluation** : accuracy, AUC-ROC, log-loss sur test set.

**Le lien avec le réseau de neurones.** La régression logistique est un MLP **sans couche cachée** avec activation sigmoïde en sortie et loss cross-entropy :

$$\text{Régression logistique} \;=\; \text{MLP}\big(L=0,\ g_{\text{out}} = \sigma,\ \mathcal{L} = \text{cross-entropy}\big)$$

C'est littéralement **un seul neurone** — la brique de base historique (Rosenblatt 1957, cf. [[00_Perceptron Multi-Couches#I — Origine et Architecture]]) avant l'empilement en couches.

> [!note]- Pourquoi la frontière reste linéaire même avec la sigmoïde
> La sigmoïde est non-linéaire, mais elle est **monotone** — elle ne fait que "compresser" la sortie dans $[0,1]$ sans changer où se trouve la frontière de décision. $\hat p(x) = 0.5 \iff \sigma(z) = 0.5 \iff z = 0 \iff \beta_0 + \beta^T x = 0$. La frontière reste l'hyperplan $\beta^T x + \beta_0 = 0$, exactement comme en régression linéaire. C'est pour ça qu'on classe la régression logistique parmi les **classifieurs linéaires**, au même titre que LDA ou le perceptron — malgré la non-linéarité apparente de la sigmoïde.

**Quand l'utiliser en prédiction pure.** Excellente baseline pour la classification binaire, surtout sur peu de données ou quand l'interprétabilité (même approximative) compte. Moins performante que XGBoost/réseaux dès que les frontières sont non-linéaires — mais sa simplicité et sa rapidité d'entraînement en font un point de départ systématique.

---

## C. k-Nearest Neighbors (kNN)

### Principe

Aucune phase d'entraînement à proprement parler — kNN est un algorithme **paresseux** (*lazy learning*) : il stocke simplement le training set, et au moment de prédire pour un nouveau point $x_0$, il regarde les $k$ points les plus proches dans le training set.

> [!warning] Règle de décision
> **Classification** : on prédit la classe **majoritaire** parmi les $k$ plus proches voisins de $x_0$.
> $$\hat f(x_0) = \underset{c}{\arg\max} \sum_{i \in N_k(x_0)} \mathbb{1}[y_i = c]$$
>
> **Régression** : on prédit la **moyenne** des $y_i$ des $k$ plus proches voisins.
> $$\hat f(x_0) = \frac{1}{k} \sum_{i \in N_k(x_0)} y_i$$
>
> où $N_k(x_0)$ désigne l'ensemble des $k$ observations du training set les plus proches de $x_0$, typiquement au sens de la distance euclidienne.

### Le choix de $k$ — biais-variance

> [!example] Effet de $k$
> - $k = 1$ : la prédiction colle exactement au point d'entraînement le plus proche. Frontière de décision très irrégulière (chaque point individuel compte), **variance maximale, biais minimal**.
> - $k$ grand (proche de $n$) : la prédiction se rapproche de la moyenne globale (en régression) ou de la classe majoritaire globale (en classification). Frontière très lisse, **biais maximal, variance minimale**.
>
> Le choix de $k$ se fait par cross-validation — c'est le levier biais-variance central de kNN.

![[knn_k_effect.png]]
**Figure.** *Frontière de décision kNN pour $k=1$ (gauche, très irrégulière), $k=15$ (milieu), $k=50$ (droite, quasi-linéaire et lisse) sur un dataset 2D à deux classes.*

### Le rôle de la distance

kNN dépend entièrement de la notion de "proche" — le choix de la métrique de distance change radicalement le comportement du modèle.

- **Distance euclidienne** $\|x - x_0\|_2$ : la plus courante. Implicitement, elle suppose que toutes les features ont la même échelle.
- **Distance de Manhattan** $\|x - x_0\|_1$ : moins sensible aux outliers individuels sur une feature.
- **Distance de Mahalanobis** : tient compte de la corrélation entre features (utilise la matrice de covariance inverse).

> [!warning] Standardisation obligatoire
> Si une feature est en kg (valeurs ~50-100) et une autre en mètres (valeurs ~1.5-2), la distance euclidienne est **dominée** par la feature aux plus grandes valeurs absolues — pas parce qu'elle est plus informative, juste parce que son échelle est plus grande. **Toujours standardiser les features avant kNN.**

### kNN et la malédiction de la dimension

kNN est l'exemple canonique de modèle qui souffre de la malédiction de la dimension — on l'a vu en détail dans [[01_Fondation#E. Curse of dimensionality]] : en haute dimension, le voisin le plus proche s'éloigne exponentiellement vite, et le 1-NN finit par souffrir d'un **biais** énorme (pas de la variance, contre-intuitivement).

> [!note]- Rappel express
> Pour capturer une fraction $r$ constante de points, l'arête du voisinage doit croître en $r^{1/d}$ — en dimension 10, capturer 1% des points nécessite déjà de couvrir 63% de l'étendue de chaque axe. Le voisinage n'est plus "local", et kNN perd son avantage principal. C'est pour ça que kNN est rarement utilisé sur des features tabulaires en grande dimension sans réduction de dimension préalable (PCA, sélection de features).

### Avantages / limites

| Avantages | Limites |
|---|---|
| Aucune hypothèse sur la forme de $f$ | Lent en prédiction sur grand $n$ (calcul de distance à tous les points) |
| Frontières de décision arbitrairement complexes | Souffre fortement de la malédiction de la dimension |
| Simple à comprendre et interpréter localement | Sensible aux features non pertinentes et à l'échelle |
| Pas d'entraînement | Demande beaucoup de mémoire (stocke tout le training set) |

**Variantes utiles.** *Weighted kNN* : pondère les votes des voisins par l'inverse de leur distance (un voisin très proche compte plus qu'un voisin à la limite du rayon $k$). *KD-tree / Ball-tree* : structures de données qui accélèrent la recherche des $k$ plus proches voisins de $O(n)$ à $O(\log n)$ en moyenne — indispensables dès que $n$ est grand.

---

## D. Support Vector Machines (SVM)

### Principe — la marge maximale

SVM cherche l'hyperplan séparateur qui maximise la **marge** — la distance entre l'hyperplan et les points les plus proches de chaque classe (les *support vectors*).

> [!warning] Cas séparable — formulation primale
> On cherche $(w, b)$ qui définissent l'hyperplan $w^Tx + b = 0$, tel que toutes les observations soient correctement classées avec une marge d'au moins 1 :
>
> $$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 \ \ \forall i$$
>
> Minimiser $\|w\|^2$ équivaut à **maximiser la marge** $2/\|w\|$ — plus $w$ est petit, plus la marge entre les deux classes est large.

![[svm_margin.png]]
**Figure.** *Hyperplan séparateur (ligne pleine) et marge (lignes pointillées). Les points sur les lignes pointillées sont les support vectors — ce sont les seuls points qui déterminent la position de l'hyperplan.*

> 💡 **Pourquoi seuls les support vectors comptent.** Une fois l'hyperplan optimal trouvé, retirer n'importe quel point qui n'est **pas** un support vector ne change rien à la solution — il était déjà correctement classé avec une marge confortable. Seuls les points les plus proches de la frontière (les support vectors) déterminent $(w, b)$. C'est une différence fondamentale avec la régression logistique, où **tous** les points contribuent au gradient.

### Cas non séparable — marge souple (soft margin)

En pratique, les classes ne sont presque jamais parfaitement séparables. On introduit des variables de relâchement $\xi_i \geq 0$ qui autorisent certains points à violer la marge, pénalisées par un hyperparamètre $C$ :

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^Tx_i+b) \geq 1-\xi_i,\ \ \xi_i \geq 0$$

> [!note]- Le rôle de $C$ — biais-variance encore
> - **$C$ petit** : on tolère beaucoup de violations de marge → marge large, modèle plus régularisé → **biais élevé, variance faible**.
> - **$C$ grand** : on pénalise fortement les violations → marge étroite, le modèle colle aux données → **biais faible, variance élevée** (risque d'overfitting).
>
> $C$ se choisit par cross-validation, exactement comme $k$ pour kNN ou $\lambda$ pour Ridge/Lasso. C'est le même principe de régularisation qu'on retrouve partout — juste un nom différent selon le modèle.

### Le kernel trick

Pour des frontières non-linéaires, SVM utilise l'astuce du noyau (kernel trick) : au lieu de transformer explicitement $x$ vers un espace de plus haute dimension $\phi(x)$, on remplace le produit scalaire $x_i^Tx_j$ par une fonction noyau $k(x_i, x_j) = \phi(x_i)^T\phi(x_j)$, **sans jamais calculer $\phi$ explicitement**.

> [!warning] Kernels usuels
> - **Linéaire** : $k(x_i,x_j) = x_i^Tx_j$ — équivalent au SVM linéaire de base.
> - **Polynomial** : $k(x_i,x_j) = (x_i^Tx_j + c)^d$ — frontières polynomiales de degré $d$.
> - **RBF (Gaussien)** : $k(x_i,x_j) = \exp(-\gamma\|x_i-x_j\|^2)$ — le plus utilisé en pratique, correspond à un espace de dimension **infinie**. $\gamma$ contrôle la "portée" de chaque point : $\gamma$ grand → frontière très locale et irrégulière (proche de 1-NN), $\gamma$ petit → frontière lisse (proche du linéaire).

![[svm_kernel_comparison.png]]
**Figure.** *Frontières de décision SVM sur un dataset non-linéairement séparable (deux anneaux concentriques). Linéaire (gauche) échoue totalement — aucune droite ne peut séparer les anneaux. RBF (droite) capture la structure circulaire grâce au kernel trick.*

> [!note]- Pourquoi ça marche sans jamais calculer $\phi(x)$
> La solution du SVM ne dépend de $x$ que via des produits scalaires $x_i^Tx_j$ — jamais des coordonnées individuelles. Donc remplacer chaque produit scalaire par $k(x_i,x_j)$ revient exactement à résoudre le problème dans l'espace transformé $\phi(x)$, **sans jamais le construire**. Pour le kernel RBF, $\phi(x)$ vivrait dans un espace de dimension infinie — totalement impossible à calculer explicitement, mais le kernel trick le rend gratuit. C'est la même logique de fond que les Gaussian Processes — le kernel encode une notion de similarité, sans jamais matérialiser l'espace de features sous-jacent.

### Lien avec la malédiction de la dimension

Le kernel RBF projette implicitement vers une dimension infinie — mais ça ne casse **pas** la malédiction de la dimension pour autant. Le kernel encode une notion de similarité **locale** (comme une distance), donc un SVM à noyau RBF souffre du même problème géométrique que kNN en très grande dimension : si la dimension *intrinsèque* des données (cf. [[01_Fondation#F. Manifold Hypothesis]]) est grande, le kernel RBF perd son pouvoir discriminant pour la même raison que toutes les distances se concentrent en haute dimension.

### Avantages / limites

| Avantages | Limites |
|---|---|
| Marge maximale → bonne généralisation théorique | Lent à entraîner sur grand $n$ ($O(n^2)$ à $O(n^3)$) |
| Kernel trick → frontières non-linéaires sans exploser la dimension explicite | Sensible au choix du kernel et de ses hyperparamètres ($C$, $\gamma$) |
| Robuste en haute dimension *si* peu d'observations ($n < d$) | Pas de probabilités natives (nécessite un calibrage, ex: Platt scaling) |
| Peu d'hyperparamètres à régler ($C$, kernel, $\gamma$) | Largement remplacé par le gradient boosting sur données tabulaires modernes |

> [!note]- Où SVM reste pertinent aujourd'hui
> Sur données tabulaires avec grand $n$, le gradient boosting (XGBoost/LightGBM) domine quasi systématiquement. SVM reste compétitif quand $n$ est **petit** et $d$ est **grand** — typiquement en bioinformatique (peu d'échantillons, beaucoup de gènes), en classification de texte avec peu d'exemples, ou pour des problèmes où la théorie de la marge maximale donne des garanties statistiques utiles.

---

## Récapitulatif — où se placer sur le spectre

| Modèle | Frontière | Hyperparamètre clé | Coût prédiction |
|---|---|---|---|
| **Régression linéaire** | Hyperplan (régression) | — (ou $\lambda$ si régularisé) | $O(d)$ |
| **Régression logistique** | Hyperplan (classification) | — (ou $\lambda$ si régularisé) | $O(d)$ |
| **kNN** | Arbitrairement complexe | $k$ | $O(n \cdot d)$ ou $O(\log n)$ avec KD-tree |
| **SVM linéaire** | Hyperplan à marge max | $C$ | $O(d)$ |
| **SVM kernel (RBF)** | Non-linéaire, locale | $C$, $\gamma$ | $O(n_{SV} \cdot d)$ |

> 💡 **Le fil conducteur.** Tous ces modèles cherchent à minimiser une fonction de perte régularisée — MSE, log-loss, ou marge — sur un espace de fonctions plus ou moins riche. Régression linéaire/logistique imposent une frontière linéaire stricte (biais fort, variance faible). kNN ne fait aucune hypothèse de forme (biais faible, variance forte, et plombé par la malédiction de la dimension). SVM se place entre les deux : linéaire par défaut, mais capable d'emprunter la flexibilité de kNN via le kernel trick, au prix d'un hyperparamètre supplémentaire à régler.
