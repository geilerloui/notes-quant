# Boosting Algorithms

Le boosting part d'une idée radicalement différente du bagging (cf [[02_Bagging_Methods]]) : au lieu de moyenner $M$ arbres construits **en parallèle** (chacun indépendamment sur un bootstrap), on construit les arbres **séquentiellement** — chaque nouvel arbre corrige les erreurs des précédents. Résultat : on réduit le **biais** (alors que le bagging réduit la variance), avec des arbres volontairement faibles (*weak learners*, typiquement profondeur 3-8).

Deux familles historiques :
- **AdaBoost** (Freund & Schapire 1997) : repondère les observations mal classées à chaque tour, weak learner = stump
- **Gradient Boosting** (Friedman 2001) : fit chaque arbre sur le **gradient négatif de la loss**, ce qui généralise AdaBoost à toute loss différentiable

Les implémentations modernes (XGBoost, LightGBM, CatBoost) sont toutes des extensions de Gradient Boosting avec des optimisations algorithmiques et des régularisations en plus.

Les hyperparamètres se rangent toujours en deux familles : ceux hérités du Decision Tree (`max_depth`, `min_samples_leaf`, etc., qui s'appliquent à chaque arbre faible), et ceux spécifiques au boosting (`n_estimators`, `learning_rate`, `loss`).

## I. AdaBoost — Freund & Schapire, 1997

### Le cadre général : stagewise additive modeling

Avant d'entrer dans l'algorithme, il faut comprendre le cadre dans lequel AdaBoost s'inscrit. L'idée du boosting c'est de construire $f$ comme une **somme de fonctions simples** (les stumps) :

$$f(x) = \sum_{t=1}^T \alpha_t h_t(x)$$

En théorie, on pourrait optimiser tous les $\alpha_t$ et $h_t$ simultanément — mais c'est computationnellement cauchemardesque. Le *stagewise additive modeling* propose à la place de construire $f$ **terme par terme, de façon greedy** : à chaque étape $t$, on cherche le meilleur $(\alpha_t, h_t)$ à ajouter à $f_{t-1}$ déjà fixé :

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i L\!\left(y_i,\ f_{t-1}(x_i) + \alpha\, h(x_i)\right)$$

puis on met à jour $f_t = f_{t-1} + \alpha_t h_t$, sans jamais retoucher les termes précédents.

L'analogie : imagine que tu veux approximer une courbe mystère ($y = 30 + x + \sin(x)$). Au lieu de trouver la formule d'un coup, tu procèdes par étapes — d'abord une constante $f_1 = 30$ qui capture la moyenne, puis tu ajoutes une droite pour capter la tendance linéaire, puis un sinus pour les ondulations. Chaque terme corrige ce que le précédent a raté. C'est exactement ce que fait AdaBoost, avec des stumps à la place.
![[Pasted image 20260416162642.png|209]]
qu'on approxime avec 
![[Pasted image 20260416162659.png]]



Il reste à choisir la **loss $L$**. AdaBoost utilise la **loss exponentielle** :

$$L(y, f(x)) = e^{-y f(x)}$$

Le produit $y \cdot f(x)$ mesure si la prédiction est correcte : s'il est positif (bonne prédiction), la loss est $< 1$ et faible ; s'il est négatif (erreur), la loss explose exponentiellement. C'est ce choix précis qui va faire tomber toutes les formules d'AdaBoost.

---

L'idée centrale d'AdaBoost est de construire itérativement une forêt de stumps où **chaque stump se concentre sur les erreurs du précédent**. Pour forcer cela, on associe à chaque observation un **poids $w_i^{(t)}$** qui mesure à quel point elle est difficile à classer au tour $t$ — plus une observation a été mal classée aux tours précédents, plus son poids est élevé, et plus le prochain stump sera forcé de la prendre en compte.

### Dérivation des formules

> [!note]- Preuve complète
> **Initialisation.** Le cadre stagewise définit $w_i^{(t)} = e^{-y_i f_{t-1}(x_i)}$. À $t=1$, $f_0 = 0$ donc $w_i^{(1)} = 1$ pour tout $i$. On normalise par $n$ : $w_{i,1} = 1/n$.
>
> **Calcul de $\alpha_t$ et $\varepsilon_t$.** À l'étape $t$, on cherche $(\alpha_t, h_t)$ qui minimisent :
> $( \alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i w_i^{(t)} \cdot e^{-\alpha y_i h(x_i)}$
> Comme $y_i, h(x_i) \in \{-1, +1\}$, on a $y_i h(x_i) = 1 - 2\cdot\mathbf{1}[y_i \neq h(x_i)]$, donc :
> $= e^{-\alpha} \sum_i w_i^{(t)} \cdot e^{2\alpha \cdot \mathbf{1}[y_i \neq h(x_i)]}$
> $= e^{-\alpha} \left[ \sum_{y_i = h(x_i)} w_i^{(t)} + e^{2\alpha} \sum_{y_i \neq h(x_i)} w_i^{(t)} \right]$
> On pose :
> $\boxed{\varepsilon_t = \sum_{y_i \neq h(x_i)} w_i^{(t)}}$
> ce qui donne :
> $= e^{-\alpha}\left[(1 - \varepsilon_t) + e^{2\alpha} \varepsilon_t\right]$
> On dérive par rapport à $\alpha$ et on annule :
> $-e^{-\alpha}(1 - \varepsilon_t) + e^{\alpha}\varepsilon_t = 0 \implies e^{2\alpha} = \frac{1-\varepsilon_t}{\varepsilon_t} \implies \boxed{\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}}$
>
> **Mise à jour des poids.** On pose $f_t = f_{t-1} + \alpha_t h_t$ et on développe $w_i^{(t+1)} = e^{-y_i f_t(x_i)}$ :
> $w_i^{(t+1)} = e^{-y_i(f_{t-1}(x_i) + \alpha_t h_t(x_i))} = \underbrace{e^{-y_i f_{t-1}(x_i)}}_{w_i^{(t)}} \cdot e^{-y_i \alpha_t h_t(x_i)}$
> $\boxed{w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}}$


---

### Tour 1 — Illustration à $t=1$

**Étape 1 — Initialisation** : $w_{i,1} = \dfrac{1}{n}$ pour tout $i$.

![[Pasted image 20260417171440.png|305]]
Table. XXXX

**Étape 2 — Choix du stump optimal** : on choisit $h_t$ qui minimise l'erreur pondérée : $\varepsilon_t = \sum_{i=1}^n w_{i,t} \cdot \mathbf{1}\left[h_t(x_i) \neq y_i\right]$

On construit trois stumps (tronc d'arbre) et au lieu de choisir celui qui minimise l'Index de Gini on va prendre celui qui minimise l'erreur pondérée. Ainsi avec $n = 8$, $w_{i,1} = 1/8$ pour tout $i$. En testant les trois features :
- Chest Pain : 3 erreurs $\Rightarrow \varepsilon_1 = 3/8$
- Blocked Arteries : 4 erreurs $\Rightarrow \varepsilon_1 = 4/8 = 1/2$
- Weight $> 176$ : 1 erreur $\Rightarrow \varepsilon_1 = 1/8$

On retient le stump "**Weight $> 176$**"
![[Pasted image 20260416105200.png]]
**Étape 3 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$

On va ensuite assigner un poids à ce "stump" : $\alpha_1 = \frac{1}{2}\ln(\frac{1-0.125}{0.125}) =\frac{1}{2}\ln(7) \approx 0.97$.

A CACHER: L'interprétation de $\alpha_t$ est naturelle : un stump parfait ($\varepsilon_t \to 0$) obtient un $\alpha_t$ très grand ; un stump aléatoire ($\varepsilon_t = 1/2$) obtient $\alpha_t = 0$ ; un stump qui se trompe systématiquement obtient un $\alpha_t$ négatif — son vote est inversé.

![[Pasted image 20260416110650.png|240]]

**Étape 4 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$

Mise à jour des poids :

$\text{mal classé :} \quad \tfrac{1}{8} \cdot e^{+0.97} \approx 0.33 \qquad \text{bien classé :} \quad \tfrac{1}{8} \cdot e^{-0.97} \approx 0.05$

A CACHER: Ces deux figures nous disent que 

![[Pasted image 20260416111212.png|466]]

On va ensuite modifier les poids dans la table qu'on normalise ensuite

![[Pasted image 20260416150127.png|467]]
Table X

---

### Tour 2 — Illustration à $t=2$

**Étape 1 — Choix du stump optimal** : on choisit $h_t$ qui minimise l'erreur pondérée : $\varepsilon_t = \sum_{i=1}^n w_{i,t} \cdot \mathbf{1}\left[h_t(x_i) \neq y_i\right]$

Après mise à jour des poids, on recommence. En recalculant $\varepsilon_t$ pour chaque feature :

- Weight $> 176$ (gagnant du tour 1) : 1 erreur à poids $0.49$ $\Rightarrow \varepsilon_2 = 0.49$
- Chest Pain : 3 erreurs à poids $0.07$ chacune $\Rightarrow \varepsilon_2 = 0.21$
- Blocked Arteries : 4 erreurs à poids $0.07$ $\Rightarrow \varepsilon_2 = 0.28$

**Étape 2 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$

On retient **Chest Pain**. Son poids : $\alpha_2 = \frac{1}{2}\ln(\frac{1-0.21}{0.21}) =\frac{1}{2}\ln(3.76) \approx 0.66$.

A CACHER: L'interprétation de $\alpha_t$ est naturelle : un stump parfait ($\varepsilon_t \to 0$) obtient un $\alpha_t$ très grand ; un stump aléatoire ($\varepsilon_t = 1/2$) obtient $\alpha_t = 0$ ; un stump qui se trompe systématiquement obtient un $\alpha_t$ négatif — son vote est inversé.

![[Pasted image 20260416110650.png|240]]

**Étape 3 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$

La mise à jour des poids au tour 2 : l'individu $n°4$ (poids $0.49$, bien classé cette fois) passe à $\approx 0.25$ ; les 3 erreurs de Chest Pain (poids $0.07$) passent à $\approx 0.13$ ; les autres tombent à $\approx 0.035$.

A CACHER: Ces deux figures nous disent que 

![[Pasted image 20260416111212.png|475]]

On modifie la table en conséquence

![[Pasted image 20260416151745.png|382]]
Table X.

---

### Prédiction finale

Après $T$ itérations :

$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

Pour un patient avec Weight $= 205$ et Chest Pain $=$ Yes :

$$H(x) = \text{sign}\left[0.97 \times (+1) + 0.66 \times (+1)\right] = \text{sign}(1.63) = +1 \implies \text{Yes Heart Disease}$$

![[Pasted image 20260416153745.png]]


## II. Gradient Boosting — Friedman, 2001

**(a) Adaboost:** Pour rappel, AdaBoost construisait son modèle selon la récurrence :

$F_t(x) = F_{t-1}(x) + \alpha_t h_t(x)$

où à chaque étape on résolvait :

$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i L\!\left(y_i,\ F_{t-1}(x_i) + \alpha\, h(x_i)\right)$

avec la **loss exponentielle**. La limite : cette loss est très sensible aux outliers, et on ne peut pas en changer.

**(b) Gradient boosting:** Gradient Boosting garde exactement la même structure :

$F_t(x) = F_{t-1}(x) + \nu \cdot \gamma_t \cdot h_t(x)$

où $h_t$ est un arbre CART, $\gamma_t$ le pas optimal calculé à cette étape, et $\nu \in (0,1]$ un learning rate qui contrôle la contribution de chaque arbre. À chaque étape on résout :

$(\gamma_t, h_t) = \arg\min_{\gamma, h} \sum_i L\!\left(y_i,\ F_{t-1}(x_i) + \nu \cdot \gamma\, h(x_i)\right)$

La différence n'est pas dans la forme — elle est dans **ce sur quoi on entraîne $h_t$** : plutôt que de résoudre ce problème directement (difficile pour une loss quelconque), on l'approche en entraînant $h_t$ sur le **gradient négatif de la loss**, ce qui permet d'utiliser n'importe quelle loss différentiable.

L'algorithme se déroule en 3 étapes : initialisation de $F_0$, puis $T$ itérations de correction, puis prédiction finale $F_T(x)$.

---

### Étape 1 — Initialisation : $F_0(x)$

À $t=0$, il n'y a pas encore de modèle ($F_{t-1} = 0$) ni d'arbre à entraîner ($h = 1$, constante). La forme générale se réduit donc à chercher simplement la meilleure constante $\gamma$ qui minimise la loss sur tout le dataset :

$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$

Ici $\gamma$ joue le rôle de $F(x)$ — c'est littéralement $F_0(x) = \gamma$ pour tout $x$, donc on substitue directement dans la loss. Avec le MSE $L(y, F) = \frac{1}{n}\sum_i (y_i - F(x_i))^2$, on dérive par rapport à $\gamma$ et on annule :

$\frac{\partial}{\partial \gamma} \sum_{i=1}^n (y_i - \gamma)^2 = \sum_{i=1}^n -2(y_i - \gamma) = 0 \implies \boxed{\gamma = \frac{1}{n}\sum_{i=1}^n y_i}$

$F_0$ est donc simplement la **moyenne des $y_i$** — une feuille unique qui prédit la même valeur pour tout le monde.

Sur notre exemple (Height, Favorite Color, Gender $\to$ Weight) :

$F_0(x) = \frac{88 + 76 + 56}{3} = 73.3 \text{ kg}$

📌 *[Image : feuille unique avec valeur 73.3]*

---

### Étape 2 — Itérations : construire les arbres correctifs

À chaque itération $t = 1, \ldots, T$, on procède en quatre sous-étapes.

**A — Calcul des pseudo-résidus**

Rappelons la descente de gradient classique dans l'espace des **paramètres** :

$$\theta_t = \theta_{t-1} - \eta \cdot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

À chaque étape, on corrige $\theta$ en allant dans la direction opposée au gradient de la loss. GBM fait exactement la même chose, mais dans l'espace des **fonctions** : au lieu de mettre à jour un vecteur $\theta$, on met à jour une fonction $F$. Le "gradient" devient alors le gradient de la loss par rapport aux valeurs prédites $F(x_i)$, et on l'annote $r_{i,t}$ :

$r_{i,t} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$

La mise à jour idéale serait 

$$
F_t(x_i) =  F_{t-1}(x_i) + \nu \cdot \Big(-\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}\Big)=F_{t-1}(x_i) + \nu \cdot r_{i,t}
$$
— exactement le pendant fonctionnel de la descente de gradient. Le problème : $r_{i,t}$ n'est défini que sur les $n$ points d'entraînement, pas sur tout l'espace des $x$. C'est pour ça qu'on **entraîne un arbre $h_t$ pour approximer ces pseudo-résidus** — l'arbre généralise la direction de descente à tout $x$.

Avec le MSE $L = \frac{1}{2}(y_i - F(x_i))^2$ :

$r_{i,t} = y_i - F_{t-1}(x_i)$

MAIS DU COUP VU QUE CES DES ARBRES DE REGRESION CEST DES INDICATRICES LES F_t-1 ?

Ce sont les résidus classiques — c'est une coïncidence du MSE. Avec une autre loss les pseudo-résidus seraient différents, d'où le terme "pseudo".

Sur notre exemple, à $t=1$ avec $F_0 = 73.3$ :

$r_{1,1} = 88 - 73.3 = 14.7 \qquad r_{2,1} = 76 - 73.3 = 2.7 \qquad r_{3,1} = 56 - 73.3 = -17.3$

📌 *[Image : tableau avec colonne $r_{i,1}$]*

**B — Entraînement d'un arbre sur les pseudo-résidus**

On entraîne un arbre CART sur les $r_{i,t}$ — on cherche à **prédire les résidus**, pas les $y_i$ directement. L'arbre crée $J_t$ feuilles (régions terminales $R_{j,t}$). Sur notre exemple on obtient un stump avec $J_1 = 2$ feuilles :

- $R_{1,1}$ : Height $< 1.55$ $\to$ résidu $-17.3$
- $R_{2,1}$ : Height $\geq 1.55$ $\to$ résidus $14.7, 2.7$

📌 *[Image : stump]*

**C — Calcul du pas optimal $\gamma_{j,t}$ par feuille**

Pour chaque feuille $R_{j,t}$, on cherche le $\gamma$ qui minimise la loss sur les observations qui tombent dans cette feuille :

$\gamma_{j,t} = \arg\min_\gamma \sum_{x_i \in R_{j,t}} L\!\left(y_i,\ F_{t-1}(x_i) + \nu \cdot \gamma\right)$

$$
\begin{aligned}
\gamma_{jm} &= \underset{\gamma}{argmin} \sum_{x_i \in R_{ij}} L(y_i, F_{m-1}(x_i)+\gamma) \\
&= \partial_{\gamma} \sum_{x_i \in R_{ij}} (y_i - (F_{m-1}(x_i) + \gamma)^2) \\
&= \sum_{x_i \in R_{ij}} y_i - F_{m-1}(x_i) - \gamma = 0 
\end{aligned}
$$

Avec le MSE, cela revient à calculer la **moyenne des résidus** dans la feuille :

$\gamma_{j,t} = \frac{1}{|R_{j,t}|} \sum_{x_i \in R_{j,t}} r_{i,t-1}$

Sur notre exemple :

$\gamma_{1,1} = -17.3 \qquad \gamma_{2,1} = \frac{14.7 + 2.7}{2} = 8.7$

**D — Mise à jour du modèle**

On met à jour $F_t$ en ajoutant la contribution de l'arbre :

$F_t(x) = F_{t-1}(x) + \nu \sum_{j=1}^{J_t} \gamma_{j,t} \cdot \mathbf{1}(x \in R_{j,t})$

Avec $\nu = 0.1$ sur notre exemple :

$F_1(x_1) = 73.3 + 0.1 \times 8.7 = 74.2$
$F_1(x_2) = 73.3 + 0.1 \times 8.7 = 74.2$
$F_1(x_3) = 73.3 + 0.1 \times (-17.3) = 71.6$

📌 *[Image : $F_1(x)$]*


## III. XGBoost Chen & Guestrin 2016


## IV. LightGBM - Ke et al 2017 Microsoft


## V. CatBoost Prokorenkova 2018 Yandex
