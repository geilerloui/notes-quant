# Boosting Algorithms

## I. Introduction

Le boosting part d'une idée radicalement différente du bagging (cf [[02_Bagging_Methods]]) : au lieu de moyenner $M$ arbres construits **en parallèle** (chacun indépendamment sur un bootstrap), on construit les arbres **séquentiellement** — chaque nouvel arbre corrige les erreurs des précédents. Résultat : on réduit le **biais** (alors que le bagging réduit la variance), avec des arbres volontairement faibles (*weak learners*, typiquement profondeur 3-8).

Deux familles historiques structurent le domaine :

- **AdaBoost** (Freund & Schapire 1997) : repondère les observations mal classées à chaque tour, weak learner = stump (§III)
- **Gradient Boosting** (Friedman 2001) : fit chaque arbre sur le **gradient négatif de la loss**, ce qui généralise AdaBoost à toute loss différentiable (§IV)

Les implémentations modernes (XGBoost §V, LightGBM §VI, CatBoost §VII) sont toutes des extensions de Gradient Boosting avec des optimisations algorithmiques et des régularisations en plus.

Les hyperparamètres se rangent toujours en deux familles : ceux hérités du Decision Tree (`max_depth`, `min_samples_leaf`, etc., qui s'appliquent à chaque arbre faible — cf [[01_Decision_Trees]]), et ceux spécifiques au boosting (`n_estimators`, `learning_rate`, `loss`).

**Plan.** Avant d'attaquer un algorithme particulier, on pose le **cadre commun** qui sous-tend *tous* les modèles de boosting (§II) : le *forward stagewise additive modeling*. AdaBoost et Gradient Boosting en sont deux instanciations, déterminées par le choix de la loss $L$ — c'est tout ce qui les différencie sur le fond.

---

## II. Le cadre commun : forward stagewise additive modeling

Le boosting, dans toutes ses variantes, est une instanciation d'un même cadre théorique : on construit le prédicteur final comme une **somme de fonctions simples**, ajoutées **une par une** de façon greedy. Chaque algorithme (AdaBoost, GBM, XGBoost…) ne diffère que par **deux choix** : la loss $L$ et la famille des $h_t$ — ce qui détermine si l'argmin de chaque étape se résout en fermé ou doit être approché numériquement.

### 1. Modèle additif et construction stagewise

> [!warning] Définition — Modèle additif
> Un modèle additif est une fonction de la forme
> 
> $$f(x) = \sum_{t=1}^T \alpha_t \, h_t(x)$$
> 
> où chaque $h_t : \mathcal{X} \to \mathbb{R}$ est un **weak learner** (typiquement un stump ou un arbre peu profond) et $\alpha_t \in \mathbb{R}$ son coefficient.

En théorie, on pourrait optimiser tous les couples $(\alpha_t, h_t)_{t=1, \ldots, T}$ **simultanément** — mais c'est computationnellement cauchemardesque (problème non-convexe en très haute dimension). Le *forward stagewise additive modeling* propose à la place de construire $f$ **terme par terme**, sans jamais retoucher les termes précédents.

> [!warning] Définition — Construction stagewise (greedy)
> Soit $f_0 \equiv 0$. À chaque étape $t = 1, \ldots, T$, on résout :
> 
> $$(\alpha_t, h_t) = \arg\min_{\alpha,\, h} \sum_{i=1}^n L\!\left(y_i,\; f_{t-1}(x_i) + \alpha\, h(x_i)\right)$$
> 
> puis on met à jour
> 
> $$f_t(x) = f_{t-1}(x) + \alpha_t\, h_t(x)$$
> 
> Les couples $(\alpha_s, h_s)_{s < t}$ déjà fixés ne sont **jamais ré-optimisés**.

L'adjectif *forward* (vers l'avant) souligne ce dernier point : on n'avance qu'à coup de corrections additives, sans retour en arrière. C'est ce qui rend l'algorithme tractable — à chaque étape, on n'a qu'un seul couple $(\alpha, h)$ à trouver.

### 2. Intuition : approximation par briques successives

Pour fixer les idées, oublions un instant les arbres et imaginons qu'on cherche à approximer une **courbe cible** $y = 30 + x + \sin(x)$ comme une somme de fonctions simples.

![[Pasted image 20260416162642.png|209]]
*Figure. La courbe cible $y = 30 + x + \sin(x)$ qu'on veut approximer.*

Une approche stagewise typique procéderait comme suit :

1. $f_1(x) = 30$ — une constante qui capture le **niveau moyen** de la courbe ;
2. $f_2(x) = f_1(x) + x = 30 + x$ — on ajoute une droite pour capter la **tendance linéaire** ;
3. $f_3(x) = f_2(x) + \sin(x)$ — on ajoute un sinus pour les **ondulations résiduelles**.

À chaque étape, le nouveau terme corrige ce que la somme des précédents n'avait pas encore capté. C'est exactement ce que feront AdaBoost et GBM, à ceci près que les "briques" $h_t$ ne seront pas une constante / une droite / un sinus, mais à chaque fois un **arbre** entraîné de façon adaptative sur les erreurs courantes.

![[Pasted image 20260416162659.png]]
*Figure. Approximation stagewise de la courbe : à chaque étape, un nouveau terme $\alpha_t h_t(x)$ vient corriger ce que la somme des précédents n'a pas encore capté.*

### 3. Le choix de la loss détermine l'algorithme

Tout l'intérêt du cadre stagewise est qu'il est **paramétré par la loss $L$** : selon le choix de $L$, l'argmin de la définition prend une forme différente, et donne naissance à un algorithme différent. Les deux cas que l'on étudie dans la suite :

| Loss $L(y, f)$ | Argmin résoluble en fermé ? | Algorithme | Section |
|:---:|:---:|:---:|:---:|
| $e^{-y f}$ (exponentielle) | **oui** — formules pour $\alpha_t, \varepsilon_t$ | **AdaBoost** | §III |
| $\frac{1}{2}(y - f)^2$, $\log(1 + e^{-yf})$, etc. | **non** en général → on l'approche par descente de gradient fonctionnel | **Gradient Boosting** | §IV |

> [!note]- Pourquoi la loss exponentielle est "facile"
> La loss exponentielle a une propriété multiplicative magique : $e^{-y(f_{t-1} + \alpha h)} = e^{-y f_{t-1}} \cdot e^{-\alpha y h}$. Le poids $e^{-y f_{t-1}}$ se factorise et ne dépend plus de $(\alpha, h)$ — l'argmin devient un simple problème de minimisation sur $(\alpha, h)$ avec des poids fixes, qui se résout en fermé. C'est exactement ce qu'on exploite en §III pour faire tomber les formules d'AdaBoost.
> 
> Avec une loss générale (MSE, log-loss, Huber…), cette factorisation n'a pas lieu et l'argmin n'a pas de solution analytique. Friedman (2001) propose alors de **remplacer l'argmin exact par une étape de descente de gradient** dans l'espace des fonctions — c'est l'objet de §IV.

Autrement dit : **AdaBoost et GBM ne sont pas deux algorithmes distincts**, mais deux instanciations du même cadre stagewise, l'une où l'argmin se résout en fermé, l'autre où on l'approxime numériquement.

---

## III. AdaBoost — Freund & Schapire, 1997

> [!example] Fil rouge — diagnostic cardiaque
> Pour ancrer chaque étape, on utilisera un dataset de **diagnostic cardiaque** à 8 patients, avec trois symptômes (Chest Pain : oui/non, Blocked Arteries : oui/non, Weight : continu) et une étiquette cible Heart Disease ($+1$ si malade, $-1$ sinon). On veut construire un classifieur AdaBoost qui prédit la maladie à partir des symptômes.

AdaBoost est l'**instanciation du cadre stagewise** (cf §II) avec la **loss exponentielle**. C'est ce choix précis qui va faire tomber toutes les formules sous forme analytique.

> [!warning] Définition — Loss exponentielle
> Pour $y \in \{-1, +1\}$ et $f(x) \in \mathbb{R}$ :
> 
> $$L(y, f(x)) = e^{-y\, f(x)}$$
> 
> Le produit $y \cdot f(x)$ s'appelle la **marge** : positive si la prédiction est correcte (loss $< 1$, faible), négative sinon (loss $> 1$, qui explose exponentiellement avec l'amplitude de l'erreur).

### Mise en place : trois objets et où sortent les poids

À chaque tour $t$, AdaBoost manipule trois objets qu'il faut bien distinguer :

| Objet | Type | Rôle |
|:---:|:---:|:---|
| $h_t : \mathcal{X} \to \{-1, +1\}$ | un **stump** (arbre de profondeur 1) | classifieur faible du tour $t$ |
| $\alpha_t \in \mathbb{R}$ | un scalaire | poids de vote attribué à $h_t$ dans l'ensemble final |
| $w_i^{(t)} \in \mathbb{R}_+$ | un scalaire par observation | difficulté courante du point $i$ au tour $t$ |

Le **classifieur final** est la majorité pondérée des votes :

$$H(x) = \text{sign}\!\left(\sum_{t=1}^T \alpha_t\, h_t(x)\right)$$

**D'où sortent les poids $w_i^{(t)}$ ?** Ils ne sont pas un ingrédient *ad hoc* : ils tombent **mécaniquement** de la substitution $L = e^{-yf}$ dans le cadre stagewise. À l'étape $t$, on cherche

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i e^{-y_i (f_{t-1}(x_i) + \alpha h(x_i))} = \arg\min_{\alpha, h} \sum_i \underbrace{e^{-y_i f_{t-1}(x_i)}}_{=:\, w_i^{(t)}} \cdot e^{-\alpha y_i h(x_i)}$$

Le facteur $e^{-y_i f_{t-1}(x_i)}$ est **complètement déterminé** par ce qu'on a déjà construit — il ne bouge pas dans l'argmin. C'est lui qu'on baptise $w_i^{(t)}$ : un point bien classé jusqu'ici a $w_i^{(t)} < 1$ (faible), un point mal classé a $w_i^{(t)} > 1$ (fort). Le prochain stump $h_t$ devra donc se concentrer sur les points à fort poids.

**Stratégie de résolution.** L'argmin est en deux variables : on le résout en **séquence**. D'abord on choisit $h_t$ qui minimise l'erreur **pondérée** par les $w_i^{(t)}$, puis connaissant $h_t$ on dérive l'argmin par rapport à $\alpha$ et on tombe sur une formule fermée pour $\alpha_t$.

### Dérivation des formules

> [!note]- Preuve complète
> **Initialisation.** Par définition, $w_i^{(1)} = e^{0} = 1$, qu'on normalise à $w_i^{(1)} = 1/n$.
> 
> **Calcul de $\varepsilon_t$ et $\alpha_t$.** À l'étape $t$, on minimise :
> $\sum_i w_i^{(t)} \cdot e^{-\alpha y_i h(x_i)}$
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
> 
> En pratique on **renormalise** ensuite pour que $\sum_i w_i^{(t+1)} = 1$. $\square$

> [!warning] Formules clés d'AdaBoost
> Au tour $t$, connaissant les poids normalisés $w_i^{(t)}$ :
> 
> 1. **Choix du stump** : $h_t = \arg\min_h \varepsilon_t(h)$ avec $\varepsilon_t = \sum_i w_i^{(t)} \cdot \mathbf{1}[y_i \neq h_t(x_i)]$
> 2. **Poids de vote du stump** : $\alpha_t = \frac{1}{2} \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$
> 3. **Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i\, \alpha_t\, h_t(x_i)}$, puis renormalisation.

L'**interprétation de $\alpha_t$** est naturelle : un stump parfait ($\varepsilon_t \to 0$) obtient un $\alpha_t$ très grand ; un stump aléatoire ($\varepsilon_t = 1/2$) obtient $\alpha_t = 0$ ; un stump systématiquement faux ($\varepsilon_t \to 1$) obtient un $\alpha_t$ négatif — son vote est inversé.

![[Pasted image 20260416110650.png|240]]
*Figure. Allure de $\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ en fonction de $\varepsilon_t$. Croisement en zéro pour $\varepsilon_t = 1/2$, divergence en $0$ et $1$.*

L'**interprétation de la mise à jour** : si $i$ est bien classé ($y_i h_t(x_i) = +1$), $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t}$ — le poids diminue. S'il est mal classé, $w_i^{(t+1)} = w_i^{(t)} \cdot e^{+\alpha_t}$ — il augmente. Les points ratés deviennent plus visibles pour $h_{t+1}$.

---

### Tour 1 — Illustration à $t=1$

**Étape 1 — Initialisation** : $w_{i,1} = \dfrac{1}{n}$ pour tout $i$.

![[Pasted image 20260417171440.png|305]]
*Table 1. Dataset cardiaque initial : 8 patients, 3 features (Chest Pain, Blocked Arteries, Weight), poids uniformes $w_{i,1} = 1/8$.*

**Étape 2 — Choix du stump optimal** : $h_t = \arg\min_h \varepsilon_t$ avec $\varepsilon_t = \sum_{i=1}^n w_{i,t} \cdot \mathbf{1}[h_t(x_i) \neq y_i]$.

On construit trois stumps candidats (un par feature) et on retient celui qui minimise l'erreur pondérée — pas l'index de Gini comme en §[[01_Decision_Trees]]. Avec $n=8$ et $w_{i,1} = 1/8$ :

- Chest Pain : 3 erreurs $\Rightarrow \varepsilon_1 = 3/8$
- Blocked Arteries : 4 erreurs $\Rightarrow \varepsilon_1 = 4/8 = 1/2$
- Weight $> 176$ : 1 erreur $\Rightarrow \varepsilon_1 = 1/8$

On retient le stump **Weight $> 176$**.

![[Pasted image 20260416105200.png]]
*Figure. Le stump retenu au tour 1 (Weight $> 176$) commet une seule erreur sur les 8 patients.*

**Étape 3 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$.

$$\alpha_1 = \frac{1}{2}\ln\frac{1 - 0.125}{0.125} = \frac{1}{2}\ln(7) \approx 0.97$$

**Étape 4 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$.

$$\text{mal classé :} \quad \tfrac{1}{8} \cdot e^{+0.97} \approx 0.33 \qquad \text{bien classé :} \quad \tfrac{1}{8} \cdot e^{-0.97} \approx 0.05$$

![[Pasted image 20260416111212.png|466]]
*Figure. Effet de la mise à jour au tour 1 : le seul point mal classé voit son poids passer de $0.125$ à $\approx 0.33$, les sept bien classés tombent à $\approx 0.05$.*

On modifie les poids dans la table puis on renormalise pour que $\sum_i w_{i,2} = 1$ :

![[Pasted image 20260416150127.png|467]]
*Table 2. Poids après normalisation à la fin du tour 1 : le patient mal classé concentre $\approx 0.49$ du poids total, les autres $\approx 0.07$ chacun.*

---

### Tour 2 — Illustration à $t=2$

**Étape 1 — Choix du stump optimal** : $h_t = \arg\min_h \varepsilon_t$.

Avec les nouveaux poids, on recalcule $\varepsilon_t$ pour chaque feature :

- Weight $> 176$ (gagnant du tour 1) : 1 erreur à poids $0.49$ $\Rightarrow \varepsilon_2 = 0.49$
- Chest Pain : 3 erreurs à poids $0.07$ $\Rightarrow \varepsilon_2 = 0.21$
- Blocked Arteries : 4 erreurs à poids $0.07$ $\Rightarrow \varepsilon_2 = 0.28$

On retient **Chest Pain**. Noter que Weight, optimal au tour 1, est maintenant le **pire** choix : son unique erreur pèse trop lourd.

**Étape 2 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$.

$$\alpha_2 = \frac{1}{2}\ln\frac{1 - 0.21}{0.21} = \frac{1}{2}\ln(3.76) \approx 0.66$$

**Étape 3 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$.

Le patient $n°4$ (poids $0.49$, bien classé cette fois) passe à $\approx 0.25$ ; les 3 erreurs de Chest Pain (poids $0.07$) montent à $\approx 0.13$ ; les autres tombent à $\approx 0.035$.

![[Pasted image 20260416151745.png|382]]
*Table 3. Poids après normalisation à la fin du tour 2 : le patient $n°4$ a vu son poids retomber, les trois nouvelles erreurs (sur Chest Pain) se voient amplifiées.*

---

### Prédiction finale

Après $T$ itérations :

$$H(x) = \text{sign}\!\left(\sum_{t=1}^T \alpha_t\, h_t(x)\right)$$

Pour un patient avec Weight $= 205$ et Chest Pain $=$ Yes :

$$H(x) = \text{sign}\!\left[0.97 \times (+1) + 0.66 \times (+1)\right] = \text{sign}(1.63) = +1 \implies \text{Heart Disease}$$

![[Pasted image 20260416153745.png]]
*Figure. Le classifieur AdaBoost après 2 tours : combinaison pondérée de Weight $> 176$ (poids $0.97$) et Chest Pain (poids $0.66$).*

---

## IV. Gradient Boosting — Friedman, 2001

> [!example] Fil rouge — prédire le poids
> Pour ancrer l'algorithme, on utilisera un dataset à 3 patients avec trois features (Height, Favorite Color, Gender) et une cible continue Weight (kg). On veut construire un Gradient Boosting de régression qui prédit le poids à partir des features.

Gradient Boosting (GBM) est l'**instanciation du cadre stagewise** (cf §II) avec **n'importe quelle loss différentiable** — pas seulement la loss exponentielle d'AdaBoost. Le prix à payer : l'argmin de chaque étape n'a plus de solution en fermé comme pour AdaBoost (cf §III). L'idée de Friedman est de **remplacer l'argmin exact par un pas de descente de gradient dans l'espace des fonctions**.

### Mise en place : descente de gradient fonctionnelle

**Rappel — descente de gradient classique.** Si on cherche à minimiser $\mathcal{L}(\theta)$ par rapport à $\theta \in \mathbb{R}^p$, on itère :

$$\theta_t = \theta_{t-1} - \nu \cdot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

À chaque étape, on corrige $\theta$ d'un petit pas dans la direction opposée au gradient.

**L'idée de Friedman.** Transposer cette logique à l'espace des **fonctions**. On cherche $F : \mathcal{X} \to \mathbb{R}$ qui minimise $\mathcal{L}(F) = \sum_i L(y_i, F(x_i))$. Le "gradient" devient le gradient de $L$ par rapport aux **valeurs prédites** $F(x_i)$, évalué en chaque point d'entraînement.

> [!warning] Définition — Pseudo-résidus
> À l'étape $t$, le **pseudo-résidu** de l'observation $i$ est défini par
> 
> $$r_{i,t} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$$
> 
> C'est l'opposé du gradient de la loss au point $i$, évalué au modèle courant $F_{t-1}$.

L'analogue de la descente de gradient serait alors $F_t(x_i) = F_{t-1}(x_i) + \nu \cdot r_{i,t}$. **Mais il y a un problème** : $r_{i,t}$ n'est défini que sur les $n$ points d'entraînement, pas sur tout l'espace $\mathcal{X}$. Pour prédire sur de nouveaux points, on doit **étendre** le gradient à tout $\mathcal{X}$. La solution de Friedman : entraîner un **arbre de régression $h_t$** dont la cible est les pseudo-résidus. L'arbre généralise la direction de descente à tout $x$.

> [!warning] Définition — Les trois objets de chaque tour
> À chaque tour $t$, GBM construit trois objets :
> 
> | Objet | Type | Rôle |
> |:---:|:---:|:---|
> | $h_t$ | un **arbre CART** à $J_t$ feuilles | définit les régions $R_{j,t}$ par split sur les pseudo-résidus |
> | $\gamma_{j,t}$ | un scalaire **par feuille** $j$ | valeur optimale dans la feuille pour la **vraie loss** |
> | $\nu \in (0, 1]$ | scalaire fixe (hyperparamètre) | **learning rate** / shrinkage |
> 
> La mise à jour finale est :
> 
> $$F_t(x) = F_{t-1}(x) + \nu \sum_{j=1}^{J_t} \gamma_{j,t} \cdot \mathbf{1}[x \in R_{j,t}]$$

**Pourquoi $\gamma_{j,t}$ et pas juste $h_t$ ?** L'arbre $h_t$ est entraîné par MSE sur les pseudo-résidus — c'est un proxy. Avec une loss finale autre que MSE (MAE, log-loss, Huber…), le proxy n'est plus aligné. La recherche $\gamma_{j,t}$ corrige ça : on garde les **régions** définies par l'arbre, mais on **ré-optimise les valeurs** par feuille pour la vraie loss. Avec MSE en loss finale (notre cas), $\gamma_{j,t}$ se réduit à la moyenne des pseudo-résidus dans la feuille — c'est le seul cas où arbre brut et $\gamma_{j,t}$ coïncident.

L'algorithme se déroule en 3 étapes : initialisation de $F_0$, $T$ itérations de correction, prédiction finale.

---

### Étape 1 — Initialisation : $F_0(x)$

![[gb1.png|282]]
*Table 4. Dataset fil rouge : 3 patients, features Height / Favorite Color / Gender, cible Weight.*

À $t=0$, il n'y a pas encore de modèle ($F_{-1} \equiv 0$) ni d'arbre à entraîner. La forme générale se réduit à chercher la meilleure **constante** $\gamma$ qui minimise la loss sur tout le dataset :

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$$

Avec le MSE $L(y, F) = \frac{1}{2}(y - F)^2$, on dérive par rapport à $\gamma$ et on annule :

$$\frac{\partial}{\partial \gamma} \sum_i \tfrac{1}{2}(y_i - \gamma)^2 = -\sum_i (y_i - \gamma) = 0 \implies \boxed{\gamma = \frac{1}{n}\sum_i y_i}$$

$F_0$ est donc simplement la **moyenne des $y_i$** — une feuille unique qui prédit la même valeur pour tout le monde. Sur le fil rouge :

$$F_0(x) = \frac{88 + 76 + 56}{3} = 73.3 \text{ kg}$$

![[gb2.png|108]]
*Figure. Le modèle initial $F_0$ : une feuille unique qui prédit $73.3$ pour tout patient.*

---

### Étape 2 — Itération $t$ : construire l'arbre correctif

À chaque itération $t = 1, \ldots, T$, on procède en quatre sous-étapes A → D.

**A — Calcul des pseudo-résidus** : $r_{i,t} = -\left[\partial L / \partial F(x_i)\right]_{F = F_{t-1}}$.

Avec le MSE $L = \frac{1}{2}(y_i - F(x_i))^2$ :

$$r_{i,t} = -\frac{\partial}{\partial F(x_i)} \tfrac{1}{2}(y_i - F(x_i))^2 = y_i - F_{t-1}(x_i)$$

Ce sont les résidus classiques — **c'est une coïncidence du MSE**. Avec une autre loss (MAE, log-loss…) les pseudo-résidus seraient différents : c'est pour ça qu'on les appelle "pseudo".

**B — Entraînement d'un arbre sur les pseudo-résidus.** On entraîne un arbre CART (cf [[01_Decision_Trees]] §B) sur les $r_{i,t}$ — l'arbre cherche à **prédire les résidus**, pas les $y_i$ directement. L'arbre crée $J_t$ feuilles, c'est-à-dire $J_t$ régions terminales $R_{j,t}$ de $\mathcal{X}$.

**C — Calcul du pas optimal $\gamma_{j,t}$ par feuille.** Pour chaque feuille $R_{j,t}$, on cherche le $\gamma$ qui minimise la loss sur les observations qui y tombent :

$$\gamma_{j,t} = \arg\min_\gamma \sum_{x_i \in R_{j,t}} L\!\left(y_i,\; F_{t-1}(x_i) + \gamma\right)$$

> [!note]- Dérivation pour le MSE
> Avec $L(y, F) = \frac{1}{2}(y - F)^2$ :
> $\partial_\gamma \sum_{x_i \in R_{j,t}} \tfrac{1}{2}\big(y_i - (F_{t-1}(x_i) + \gamma)\big)^2 = -\sum_{x_i \in R_{j,t}}\big(y_i - F_{t-1}(x_i) - \gamma\big) = 0$
> $\implies \gamma_{j,t} = \frac{1}{|R_{j,t}|} \sum_{x_i \in R_{j,t}}\big(y_i - F_{t-1}(x_i)\big) = \frac{1}{|R_{j,t}|} \sum_{x_i \in R_{j,t}} r_{i,t}$
> 
> Avec MSE, $\gamma_{j,t}$ est donc simplement la **moyenne des pseudo-résidus** dans la feuille — c'est ce que l'arbre prédit déjà par construction. $\square$

**D — Mise à jour du modèle** :

$$F_t(x) = F_{t-1}(x) + \nu \sum_{j=1}^{J_t} \gamma_{j,t} \cdot \mathbf{1}[x \in R_{j,t}]$$

Le learning rate $\nu \in (0, 1]$ contrôle l'amplitude de la correction — il sert de **régularisation**, pas d'optimisation. Une valeur faible ($\nu \approx 0.1$) shrinke les contributions, ce qui force GBM à utiliser plus d'arbres mais évite l'overfitting.

---

### Tour 1 — Illustration à $t=1$

**A — Pseudo-résidus.** Avec $F_0 = 73.3$ pour tout le monde :

$$r_{1,1} = 88 - 73.3 = 14.7 \qquad r_{2,1} = 76 - 73.3 = 2.7 \qquad r_{3,1} = 56 - 73.3 = -17.3$$

![[gb3.png|330]]
*Table 5. Dataset augmenté de la colonne $r_{i,1}$ : pseudo-résidus du tour 1.*

**B — Arbre sur les pseudo-résidus.** L'arbre CART trouve le meilleur split. Sur cet exemple, il splitte sur **Height $< 1.55$** et obtient un stump à $J_1 = 2$ feuilles :

- $R_{1,1}$ : Height $< 1.55$ → contient $\{r_{3,1}\} = \{-17.3\}$
- $R_{2,1}$ : Height $\geq 1.55$ → contient $\{r_{1,1}, r_{2,1}\} = \{14.7, 2.7\}$

![[gb4.png|513]]
*Figure. Le stump $h_1$ entraîné sur les pseudo-résidus du tour 1.*

**C — $\gamma_{j,1}$ par feuille** (moyenne des résidus, cas MSE) :

$$\gamma_{1,1} = -17.3 \qquad \gamma_{2,1} = \frac{14.7 + 2.7}{2} = 8.7$$

![[gb5.png|162]]
*Figure. Le stump $h_1$ avec les valeurs $\gamma_{j,1}$ inscrites dans chaque feuille.*

**D — Mise à jour de $F_1$**, avec $\nu = 0.1$ :

$$F_1(x_1) = 73.3 + 0.1 \times 8.7 = 74.2$$
$$F_1(x_2) = 73.3 + 0.1 \times 8.7 = 74.2$$
$$F_1(x_3) = 73.3 + 0.1 \times (-17.3) = 71.6$$

Les prédictions ont **bougé dans la bonne direction** mais sont encore loin des cibles ($88, 76, 56$) — c'est l'effet du shrinkage $\nu = 0.1$, qui force des corrections lentes.

| Modèle courant après tour 1 | Table avec $F_1(x_i)$ |
| --------------------------- | --------------------- |
| ![[gb6.png\|275]]           | ![[gb1 (1).png\|276]] |

*Figures. À gauche : le modèle $F_1 = F_0 + \nu \cdot h_1$ vu comme une combinaison de l'initialisation (feuille à $73.3$) et du stump du tour 1. À droite : table avec la colonne $F_1(x_i)$ ajoutée.*

---

### Tour 2 — Illustration à $t=2$

On répète exactement les quatre sous-étapes A → D avec $F_1$ comme modèle courant. Les pseudo-résidus deviennent $r_{i,2} = y_i - F_1(x_i)$, on entraîne un nouvel arbre $h_2$ dessus, on calcule $\gamma_{j,2}$ par feuille, et on met à jour :

$$F_2(x) = F_1(x) + \nu \sum_{j} \gamma_{j,2} \cdot \mathbf{1}[x \in R_{j,2}]$$

![[gb7.png|543]]
*Figure. Construction du tour 2 : nouveaux pseudo-résidus $r_{i,2}$, nouvel arbre $h_2$, nouvelles valeurs $\gamma_{j,2}$ par feuille. Le modèle complet $F_2$ est l'addition $F_0 + \nu \cdot h_1 + \nu \cdot h_2$.*

À chaque tour, les pseudo-résidus diminuent en valeur absolue : on s'approche progressivement des cibles. En répétant $T \sim 100$ fois, on convergerait pratiquement vers les vraies valeurs.

---

### Étape 3 — Prédiction finale

Après $T$ itérations, le modèle complet est la somme cumulée des contributions :

$$F_T(x) = F_0 + \nu \sum_{t=1}^T \sum_{j=1}^{J_t} \gamma_{j,t} \cdot \mathbf{1}[x \in R_{j,t}]$$

Pour prédire sur un nouveau patient, on traverse les $T$ arbres : à chaque tour $t$, on identifie la feuille $R_{j,t}$ où il tombe, on récupère $\gamma_{j,t}$, et on cumule.

![[gb8.png|262]]
*Figure. Nouveau patient à prédire, avec ses features.*

En traversant les deux arbres construits ($M = 2$ ici), on cumule les contributions à $F_0 = 73.3$ :

![[gb9.png|456]]
*Figure. Calcul de $F_2(x)$ pour le nouveau patient : on suit le chemin dans chaque arbre, on récupère $\gamma_{j,t}$, et on additionne avec le shrinkage. Résultat : prédiction de $\approx 70$ kg.*

---

### Hyperparamètres principaux

Récapitulatif des hyperparamètres de `GradientBoostingRegressor` (scikit-learn) :

- **`n_estimators`** ($T$) : nombre d'arbres. Plus $T$ est grand, plus l'ensemble est expressif — mais aussi plus susceptible d'overfitter. Typiquement $100$–$1000$.
- **`learning_rate`** ($\nu$) : shrinkage. Petit $\nu$ + grand $T$ régularise mais coûte du calcul. Compromis classique : $\nu = 0.1$, $T = 500$.
- **`max_depth`** : profondeur des arbres faibles. Pour GBM, profondeur **3-8** typiquement (vs. 1 pour AdaBoost qui utilise des stumps).
- **`subsample`** : fraction du dataset utilisée à chaque tour (Stochastic Gradient Boosting, Friedman 2002). Réduit la corrélation entre arbres et améliore la généralisation.
- **`loss`** : la loss $L$. Pour la régression : `squared_error` (MSE), `absolute_error` (MAE), `huber` (robuste). Pour la classification binaire : `log_loss` (deviance, équivalente à la log-loss).

### Différences avec AdaBoost

| | AdaBoost (§III) | Gradient Boosting (§IV) |
|:---|:---:|:---:|
| Loss | Exponentielle (fixe) | Quelconque différentiable |
| Argmin par tour | **Fermé** (formule pour $\alpha_t$) | **Approché** (descente de gradient + arbre) |
| Weak learner | Stump (profondeur 1) | Arbre profondeur 3-8 |
| Tâche | Classification binaire $\{-1,+1\}$ | Régression + classification (toute loss) |
| Mécanisme de focus | Repondération des observations $w_i^{(t)}$ | Pseudo-résidus = gradient |

---

## V. XGBoost — Chen & Guestrin, 2016

XGBoost (*eXtreme Gradient Boosting*, [Chen & Guestrin, KDD 2016](https://arxiv.org/abs/1603.02754)) est l'implémentation moderne dominante du gradient boosting. Sur le fond, c'est du GBM (§IV) ; sur la forme, **trois différences clés** :

1. **Régularisation explicite** dans la loss (terme $\gamma T + \tfrac{1}{2}\lambda \sum w_j^2$).
2. **Approximation de la loss au second ordre** (Taylor 2) — au lieu de juste suivre le gradient, on utilise aussi la Hessienne, ce qui permet de calculer la valeur optimale par feuille en fermé.
3. **Optimisations algorithmiques massives** (greedy approché, weighted quantile sketch, sparsity-aware, parallel/cache/out-of-core) qui rendent l'algorithme scalable à des datasets de millions de lignes.

### V.1 La loss XGBoost et ses hyperparamètres

> [!warning] Définition — Loss XGBoost régularisée
> Pour un ensemble de $T$ arbres $\{f_k\}_{k=1}^T$ produisant la prédiction $\hat{y}_i = \sum_k f_k(x_i)$ :
> 
> $$\mathcal{L} = \frac{1}{n} \sum_{i=1}^n \ell\!\left(y_i,\, \hat{y}_i\right) + \sum_{k=1}^T \Omega(f_k), \qquad \Omega(f) = \gamma T_f + \frac{1}{2}\lambda \sum_{j=1}^{T_f} w_j^2$$
> 
> où $T_f$ est le nombre de feuilles de l'arbre $f$ et $w_j$ la valeur prédite par la feuille $j$. **Deux pénalités** : $\gamma$ pénalise le nombre de feuilles (complexité), $\lambda$ pénalise les valeurs par feuille (shrinkage L2).

**Hyperparamètres principaux** :

| Paramètre | Défaut | Rôle |
|:---|:---:|:---|
| `eta` ($\eta$) | $0.3$ | Learning rate (shrinkage) — analogue du $\nu$ de GBM |
| `gamma` ($\gamma$) | $0$ | Coût minimum pour ouvrir une nouvelle branche (pruning) |
| `lambda` ($\lambda$) | $1$ | Régularisation L2 sur les valeurs de feuille |
| `max_depth` | $6$ | Profondeur maximale de chaque arbre |
| `min_child_weight` | $1$ | Cover minimum par feuille (cf §V.2) |
| `subsample` | $1$ | Fraction des observations échantillonnée par arbre |
| `colsample_bytree` | $1$ | Fraction des features échantillonnée par arbre |

`colsample_bytree` est emprunté à Random Forest et est, selon les auteurs, **plus efficace que le sous-échantillonnage en lignes** pour prévenir l'overfitting.

### V.2 Régression — fil rouge Drug Dosage

> [!example] Fil rouge — Drug Dosage
> On utilise un dataset à 4 observations : abscisse Drug Dosage, ordonnée Drug Effectiveness. On veut un XGBoost de régression qui prédit l'efficacité du médicament selon la dose. La loss est le MSE.

#### Étape 1 — Prédiction initiale

Par défaut, XGBoost commence avec une **prédiction constante de $0.5$** pour tous les points — convention bizarre mais sans importance (n'importe quelle constante marche, on convergera vers la même solution). On calcule les résidus initiaux $r_i = y_i - 0.5$.

![[xgb-1.png|275]]
*Figure. Les 4 points du dataset (effectiveness vs dosage) avec la prédiction initiale constante à $0.5$ et les résidus en pointillés.*

#### Étape 2 — Construction d'un arbre XGBoost

À chaque tour, XGBoost fit un arbre **sur les résidus** — comme GBM. Mais le critère de split n'est ni Gini, ni MSE classique : c'est un **score de similarité** qui tombe directement de la dérivation Taylor (cf §V.4).

> [!warning] Définition — Similarity Score et Gain
> Pour un nœud contenant les observations d'indices $I$ :
> 
> $$\text{Similarity}(I) = \frac{1}{2} \cdot \frac{\big(\sum_{i \in I} g_i\big)^2}{\sum_{i \in I} h_i + \lambda}$$
> 
> où $g_i, h_i$ sont les dérivées première et seconde de la loss au point $i$. Pour le MSE : $g_i = -(y_i - \hat{y}_i)$ et $h_i = 1$, donc la similarity se réduit à $(\text{somme des résidus})^2 / (\text{nombre de résidus} + \lambda)$.
> 
> Le **Gain** d'un split (parent $I$ → gauche $I_L$, droite $I_R$) est :
> 
> $$\boxed{\;\text{Gain} = \text{Similarity}(I_L) + \text{Similarity}(I_R) - \text{Similarity}(I)\;}$$
> 
> On retient le split qui maximise le Gain — exactement comme on retient le split qui maximise la réduction d'impureté en CART.

**(i) Similarity au nœud racine.** On met tous les résidus dans une seule feuille initiale et on calcule sa similarity :

![[xgb2.png|408]]
*Figure. Tous les résidus du dataset dans la feuille initiale. Avec $\lambda = 0$, on a $\text{Similarity} = (-10.5 + 6.5 + 7.5 - 7.5)^2 / 4 = 4$.*

**(ii) Calcul du Gain pour chaque candidat de split.** On teste différents seuils sur Dosage et on calcule le Gain pour chacun. Sur cet exemple, le premier split optimal donne :

$$\text{Gain} = \text{Sim}_L + \text{Sim}_R - \text{Sim}_{\text{root}} = 110.25 + 14.08 - 4 = 120.33$$

**(iii) Splitting récursif.** On continue à splitter la feuille de droite tant que `max_depth` n'est pas atteint :

![[xgb4.png|340]]
*Figure. Split récursif de la feuille de droite. Nouveau Gain : $98 + 56.25 - 14.08 = 140.17$.*

**(iv) Pruning par $\gamma$.** Une fois l'arbre construit, on **prune** en remontant des feuilles vers la racine : pour chaque branche, on calcule $\text{Gain} - \gamma$. Si c'est **négatif**, on supprime la branche ; sinon, on la garde. Avec $\gamma = 130$ :

$$\text{Gain} - \gamma = 140.17 - 130 = 10.17 > 0 \implies \text{on garde}$$

![[xgb5.png|360]]
*Figure. Arbre final après pruning par $\gamma = 130$ : toutes les branches sont conservées.*

#### Étape 3 — Prédiction avec learning rate

Une fois l'arbre construit, **la valeur prédite par la feuille $j$** est donnée par la même formule qui sort de la dérivation Taylor :

$$w_j^\star = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Pour le MSE, c'est essentiellement la **moyenne des résidus** dans la feuille (à un signe et un facteur de régularisation près). La prédiction est ensuite shrinkée par le **learning rate** $\eta = 0.3$ (défaut) :

$$\hat{y}^{\text{new}}_i = \hat{y}^{\text{old}}_i + \eta \cdot w_j^\star \quad \text{si } x_i \in \text{feuille } j$$

![[xgb6.png|393]]
*Figure. Après ajout du premier arbre scalé par $\eta = 0.3$ : la nouvelle prédiction (ligne noire) s'approche du point. Le résidu devient plus petit.*

On répète le processus pour construire le second arbre sur les nouveaux résidus, puis le troisième, etc.

![[xgb7.png|356]]
*Figure. Après plusieurs arbres : la prédiction (en escalier, somme de tous les arbres) suit progressivement la cible.*

### V.3 Classification — log-loss et Cover

Pour la classification binaire, on change la loss en **log-loss** (cross-entropy binaire) :

$$\ell(y_i, \hat{y}_i) = -\big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\big]$$

où $p_i = \sigma(\hat{y}_i)$ est la probabilité prédite via sigmoïde, et $\hat{y}_i$ vit dans l'espace des **log-odds** (logit). L'arbre prédit donc dans l'espace logit, et on convertit en probabilité à la prédiction finale.

#### Dérivées de la log-loss

> [!warning] $g_i, h_i$ pour la classification
> $$g_i = -(y_i - p_i), \qquad h_i = p_i (1 - p_i)$$
> 
> Conséquence importante : $h_i$ **n'est plus constant à 1** comme en régression. Il dépend de la prédiction courante $p_i$. Et il est **maximal** ($= 0.25$) quand $p_i = 0.5$ (point indécis) et **minimal** ($\to 0$) quand $p_i \to 0$ ou $1$ (point bien classé).

#### Construction de l'arbre

Le processus est identique à la régression — similarity score, Gain, splitting récursif — mais avec les nouveaux $g_i, h_i$.

![[xgb8.png|267]]
*Figure. Setup classification : 4 observations (Dosage en x, P(effective) en y), prédiction initiale constante $p = 0.5$ et résidus.*

**Feuille initiale.** Avec $p_i = 0.5$ pour tous, les résidus s'annulent (autant de positifs que de négatifs) et la similarity de la racine est nulle :

![[xgb9.png|252]]
*Figure. Tous les résidus dans la feuille racine. Numérateur = $(-0.5 + 0.5 + 0.5 - 0.5)^2 = 0$, donc Similarity $= 0$.*

**Premier split.** On teste $\text{Dosage} < 15$. Le calcul de la similarity de la feuille de gauche (3 résidus à $-0.5, +0.5, +0.5$) donne :

$$\frac{(-0.5 + 0.5 + 0.5)^2}{3 \times (0.5)(1 - 0.5) + \lambda} = \frac{0.25}{0.75 + \lambda} = 0.33 \text{ pour } \lambda = 0$$

![[xgb10.png|315]]
*Figure. Premier split à $\text{Dosage} < 15$. Gain = $0.33 + 1 - 0 = 1.33$.*

On teste les autres seuils, aucun ne donne un meilleur Gain — on retient $\text{Dosage} < 15$.

**Splitting récursif sur la feuille gauche** :

![[xgb11.png|316]]
*Figure. Split supplémentaire $\text{Dosage} < 5$ sur la feuille de gauche. La profondeur est limitée à 2, on s'arrête.*

**Pruning** : avec $\gamma = 2$, $\text{Gain} - \gamma = 2.66 - 2 = 0.66 > 0$ — on garde l'arbre.

![[xgb13.png|302]]
*Figure. Arbre final après pruning par $\gamma = 2$.*

#### Prédiction : logit → probabilité

L'arbre prédit dans l'espace logit. Pour un point qui tombe dans une feuille de valeur $w^\star$ :

$$\text{logit}_{\text{new}} = \text{logit}_{\text{old}} + \eta \cdot w^\star, \qquad p_{\text{new}} = \sigma(\text{logit}_{\text{new}}) = \frac{1}{1 + e^{-\text{logit}_{\text{new}}}}$$

| Étape                                                        |       Figure        |
| :----------------------------------------------------------- | :-----------------: |
| Conversion proba initiale $0.5$ → log-odds $0$               | ![[xgb14.png\|114]] |
| Ajout du terme $\eta \cdot w^\star = 0.3 \times (-2) = -0.6$ | ![[xgb15.png\|283]] |
| Conversion log-odds $-0.6$ → proba $0.35$ via sigmoïde       | ![[xgb16.png\|280]] |

#### Cover en classification — un point délicat

> [!warning] Cover = $\sum h_i$
> En **régression** : $h_i = 1$ pour tout $i$, donc $\text{Cover} = $ nombre de résidus dans la feuille. La contrainte par défaut `min_child_weight = 1` est triviale (au moins 1 résidu par feuille).
> 
> En **classification** : $h_i = p_i(1 - p_i)$, donc $\text{Cover} = \sum_{i \in I} p_i(1 - p_i)$. Plus les prédictions courantes sont **confiantes** ($p_i$ proche de $0$ ou $1$), plus le Cover est petit — et la contrainte `min_child_weight = 1` devient **très restrictive**.

Pour le premier arbre où tous les $p_i = 0.5$, chaque résidu apporte $0.25$ au Cover. Une feuille avec 3 résidus a $\text{Cover} = 0.75 < 1$ — XGBoost refuse de la créer avec la valeur par défaut.

![[xgb12.png|495]]
*Figure. Effet du Cover en classification : avec `min_child_weight = 1` (défaut), aucune feuille n'est autorisée (toutes ont $\text{Cover} < 1$). Solution pratique pour ce petit exemple : mettre `min_child_weight = 0`.*

### V.4 Dérivation Taylor à l'ordre 2

> [!note]- D'où viennent Similarity et $w^\star$ ?
> XGBoost approxime la loss à l'ordre 2 autour de la prédiction courante $\hat{y}_i$. En écrivant le **prochain arbre** comme $f_t$ qui ajoute $w_{q(x_i)}$ à $\hat{y}_i$ (où $q(x_i)$ est la feuille où tombe $x_i$ et $w_j$ sa valeur) :
> 
> $$\ell(y_i, \hat{y}_i + w_{q(x_i)}) \approx \ell(y_i, \hat{y}_i) + g_i\, w_{q(x_i)} + \tfrac{1}{2} h_i\, w_{q(x_i)}^2$$
> 
> En sommant sur les observations et en regroupant par feuille $j$ (avec $I_j$ l'ensemble des observations qui tombent dans la feuille $j$) :
> 
> $$\mathcal{L}_t \approx \text{cste} + \sum_{j=1}^{T} \left[\Big(\sum_{i \in I_j} g_i\Big) w_j + \tfrac{1}{2} \Big(\sum_{i \in I_j} h_i + \lambda\Big) w_j^2\right] + \gamma T$$
> 
> **Optimal par feuille** (dérivée par rapport à $w_j$, annulée) :
> 
> $$\boxed{w_j^\star = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}}$$
> 
> **En substituant $w_j^\star$ dans la loss approchée** :
> 
> $$\mathcal{L}_t^\star = -\frac{1}{2}\sum_{j=1}^T \frac{\big(\sum_{i \in I_j} g_i\big)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$
> 
> Chaque feuille contribue à la loss par $-\tfrac{1}{2} \cdot \frac{(\sum g_i)^2}{\sum h_i + \lambda}$ — c'est exactement la **Similarity** (au signe et facteur $\tfrac{1}{2}$ près).
> 
> **Gain d'un split** = gain de Similarity dû au split, moins le coût $\gamma$ d'ouvrir une feuille supplémentaire :
> 
> $$\text{Gain} = \tfrac{1}{2}\!\left[\frac{(\sum_{I_L} g_i)^2}{\sum_{I_L} h_i + \lambda} + \frac{(\sum_{I_R} g_i)^2}{\sum_{I_R} h_i + \lambda} - \frac{(\sum_I g_i)^2}{\sum_I h_i + \lambda}\right] - \gamma$$
> 
> En pratique XGBoost compare juste les Similarity (sans le $-\gamma$ pendant la construction), et applique le pruning par $\gamma$ après coup. $\square$

Ce qui sort de cette dérivation, en résumé : **XGBoost n'est pas qu'un wrapper de GBM**, c'est un GBM avec une approximation au second ordre qui permet de calculer $w_j^\star$ **en fermé** et d'utiliser un critère de split (Similarity) qui colle exactement à la loss régularisée.

### V.5 Optimisations algorithmiques

C'est ce qui rend XGBoost scalable.

#### (a) Approximate Greedy Algorithm

Pour trouver le meilleur split sur une feature continue, l'algorithme exact teste **tous les seuils possibles** — coûteux dès que $n$ est grand. La solution XGBoost : tester seulement les **quantiles** de la feature comme candidats de split (environ **33 quantiles** par défaut). On ne teste plus que ~33 seuils par feature, gain énorme en performance avec perte négligeable de qualité.

#### (b) Weighted Quantile Sketch

Petit raffinement subtil : les quantiles habituels mettent **autant d'observations** par bin. Mais en classification, les observations à fort $h_i = p_i(1 - p_i)$ (les "indécises") portent plus de signal pour le split. XGBoost utilise donc des **quantiles pondérés par les Hessiens** $h_i$ — la somme des poids est constante par bin, pas le nombre d'observations.

![[xgb22.png|544]]
*Figure. Quantile sketch sur dataset distribué : chaque machine calcule un histogramme local, on les combine pour obtenir un histogramme global approximatif sur lequel les quantiles sont calculés.*

| Régression ($h_i = 1$) | Classification ($h_i = p_i(1-p_i)$) |
|:---:|:---:|
| ![[xgb23.png\|400]] | ![[xgb24.png\|279]] |

*Figures. En régression, $h_i = 1$ uniforme ⇒ quantiles pondérés = quantiles normaux. En classification, les points "incertains" ($p_i \approx 0.5$, $h_i$ grand) attirent les bornes de bins, et les points "confiants" ($p_i \approx 0$ ou $1$, $h_i$ petit) en attirent peu.*

#### (c) Sparsity-Aware Split Finding

Sur du sparse data ou avec des valeurs manquantes, XGBoost a une astuce : pour chaque split candidat, il teste **deux orientations par défaut** pour les manquants (envoyer tous les manquants à gauche, vs tous à droite), garde celle qui donne le meilleur Gain, et la **mémorise** dans l'arbre. À la prédiction, un point avec valeur manquante suit automatiquement l'orientation apprise.

![[xgb25.png|497]]
*Figure. Dataset avec valeurs manquantes (Dosage non-renseigné pour certains points). Les résidus restent calculables (basés sur la prédiction initiale $0.5$).*

![[xgb26.png|535]]
*Figure. On split le dataset en deux tables : observations avec Dosage connu (qu'on trie pour tester les seuils) et observations à Dosage manquant (qu'on traite séparément).*

![[xgb27.png|449]]
*Figure. Pour chaque seuil candidat, on calcule deux Gains : un en envoyant les manquants à gauche ($\text{Gain}_L$), un en les envoyant à droite ($\text{Gain}_R$).*

![[xgb28.png|491]]
*Figure. Arbre final : à chaque nœud, on a appris **l'orientation par défaut** (flèche bleue) à suivre pour un point à valeur manquante. À la prédiction, un nouveau point sans Dosage suit automatiquement cette direction.*

#### (d) Parallel Learning, Cache-Aware Access, Out-of-Core Computation

Trois optimisations bas-niveau :

- **Parallel Learning** : la recherche du split optimal sur chaque feature est indépendante ⇒ paralléliser sur les features (un thread par feature). Important : c'est la recherche de **split** qui est parallélisée, pas la construction de plusieurs arbres en même temps (séquentiel par nature).
- **Cache-Aware Access** : XGBoost organise les gradients et hessiens en mémoire pour maximiser les hits dans le cache L1/L2 du CPU pendant le calcul des Similarity.
- **Blocks for Out-of-Core Computation** : quand le dataset ne tient pas en RAM, XGBoost le stocke en blocs **compressés** sur disque et streame en utilisant les CPU en parallèle pour la décompression. Lecture disque = goulot d'étranglement, donc on compresse le plus possible.

---

## VI. LightGBM — Ke et al., 2017 (Microsoft)

[Ke et al., NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) — viser le même résultat que XGBoost (gradient boosting régularisé) mais **beaucoup plus vite**. Trois innovations algorithmiques structurantes : **histogram-based splitting**, **GOSS** (gradient sampling), **EFB** (feature bundling). Plus un détail de stratégie d'arbre : **leaf-wise growth** au lieu du level-wise.

### VI.1 Histogram-based splitting

Pour chaque feature continue, on **bucketise** une fois pour toutes les valeurs dans (typiquement) $255$ bins. Pendant la construction des arbres, on cherche le meilleur split **uniquement sur les bornes des bins**, pas sur toutes les valeurs uniques.

- **Coût de recherche d'un split** : $O(\text{\#data} \times \text{\#features})$ pour XGBoost exact, $O(\text{\#bins} \times \text{\#features})$ pour LightGBM.
- **Coût mémoire** : on stocke des `uint8` (256 bins) au lieu de `float32`, soit **4× moins de RAM**.

XGBoost a ajouté un mode `hist` (équivalent algorithmiquement) en 2017 en réponse à LightGBM.

![[Pasted image 20260529175702.png|263]]![[Pasted image 20260529175720.png|216]]

### VI.2 GOSS — Gradient-based One-Side Sampling

**Idée** : les observations à **fort gradient** $|g_i|$ sont mal modélisées par les arbres actuels (résidus élevés), elles portent donc beaucoup d'information pour le prochain arbre. Les observations à **faible gradient** sont déjà bien modélisées, contribuent peu.

**Procédure** :
1. Trier les observations par $|g_i|$ décroissant.
2. Garder les **top $a\%$** systématiquement (typiquement $a = 20$).
3. Sampler aléatoirement $b\%$ parmi les $(1-a)\%$ restantes (typiquement $b = 10$).
4. Pour préserver l'estimateur du Gain, **multiplier les gradients du subsample** par $(1-a)/b$.

**Effet** : on entraîne sur ~30% des données mais avec quasi pas de perte de qualité — speedup typique 2-3×.

![[Pasted image 20260529175826.png|306]]![[Pasted image 20260529175843.png|270]]

### VI.3 EFB — Exclusive Feature Bundling

**Constat** : dans les datasets très sparse (one-hot encoding, texte, RecSys…), beaucoup de features sont **mutuellement exclusives** — elles ne sont jamais non-nulles en même temps. Par exemple, dans un one-hot à 100 catégories, exactement **une** feature est non-nulle par observation.

**Idée** : on peut "fusionner" plusieurs features mutuellement exclusives en **une seule feature combinée** sans perte d'information. Si feature A prend des valeurs dans $\{0, 1, 2\}$ et feature B mutuellement exclusive dans $\{0, 1, 2, 3\}$, on les combine en une feature unique dans $\{0, 1, 2, 3, 4, 5, 6\}$ : valeurs $1$–$2$ = A non-nul, $3$–$6$ = B non-nul. Plus de combinaisons que de bins individuels mais **bien moins** que les features séparées.

**Effet** : sur du sparse, peut diviser le nombre effectif de features par 5-10×.

![[Pasted image 20260529175951.png|315]]

### VI.4 Leaf-wise vs level-wise growth

| | Level-wise (XGBoost classique) | Leaf-wise (LightGBM) |
|:---|:---:|:---:|
| Stratégie | Tous les nœuds d'une profondeur sont splittés avant de passer à la suivante | À chaque étape, on splitte **la feuille qui donne le plus grand Gain**, peu importe sa profondeur |
| Arbres typiques | Équilibrés (BFS) | Asymétriques (best-first) |
| Profondeur effective | Bornée par `max_depth` | Bornée par `num_leaves` (plus naturel) |
| Overfitting | Moins sensible | Plus prone à overfitter sur petit dataset |

LightGBM atteint typiquement la **même loss en moins d'arbres** que XGBoost level-wise — mais demande plus de prudence sur les hyperparamètres (limiter `num_leaves`, augmenter `min_data_in_leaf`) sur petit dataset.

![[Pasted image 20260529180219.png]]

### VI.5 Quand préférer LightGBM ?

- **Très gros dataset** (millions+ de lignes) : LightGBM scale typiquement 5-10× plus vite que XGBoost.
- **Beaucoup de features sparse** ou one-hot : EFB est très efficace.
- **Compétition Kaggle** sur dataset tabulaire moyen-gros : LightGBM gagne souvent en pratique grâce à sa vitesse d'itération.

XGBoost reste préférable sur **petits datasets** (overfitting plus facile à contrôler en level-wise) et quand on a besoin de **stabilité numérique maximale**.

---

## VII. CatBoost — Prokhorenkova et al., 2018 (Yandex)

[Prokhorenkova et al., NeurIPS 2018](https://arxiv.org/abs/1706.09516). Le pitch CatBoost : *"on prend GBM, on règle le problème du leakage des variables catégorielles avec une astuce d'ordonnancement, et on utilise des arbres oblivious pour aller vite"*. Deux contributions majeures : **Ordered Target Encoding** et **Ordered Boosting**.

### VII.1 Le problème du target leakage

Pour encoder une variable catégorielle (couleur préférée, ville, métier…) en numérique, le **target encoding** classique remplace chaque catégorie par la **moyenne de la cible** sur les observations de cette catégorie :

$$\text{encoded}(\text{cat}) = \frac{1}{n_{\text{cat}}} \sum_{i :\, \text{cat}_i = \text{cat}} y_i$$

**Problème** : on utilise $y_i$ pour encoder la feature de l'observation $i$. **Leakage direct**. Le modèle peut "tricher" et le test score s'effondre.

> [!example] Exemple jouet du leakage
> Dataset à 4 lignes, feature `Favorite Color` (3 lignes "blue", 1 ligne "red"), cible binaire :
> 
> ![[Pasted image 20260510215629.png|225]] ![[Pasted image 20260510215847.png|229]]
> 
> Le target encoding naïf encode "blue" par la moyenne de $y$ sur les 3 lignes blue = $0.33$, et "red" par $1.0$. Le classifieur apprend trivialement la règle *"si encoded = 0.33 alors classe 1, sinon classe 0"* — qui est en réalité juste *"si Favorite Color = blue alors classe 1, sinon classe 0"*. Le modèle a appris une règle qui inclut $y$ dans son input.

Le **k-fold target encoding** réduit le leakage (on encode chaque fold avec les autres folds), mais ne l'élimine pas complètement.

### VII.2 Ordered Target Encoding

**Idée** : ordonner aléatoirement les observations, puis encoder chaque ligne en utilisant **uniquement les lignes précédentes** dans l'ordre. C'est essentiellement un *"online learning du target encoding"*.

> [!warning] Formule — Ordered Target Encoding
> Pour la ligne $i$ dans l'ordre aléatoire, avec catégorie $\text{cat}_i$ :
> 
> $$\text{encoded}_i = \frac{\text{OptionCount}_i + \alpha}{n_i + 1}$$
> 
> où
> - $n_i$ = nombre de lignes **précédentes** (j < i dans l'ordre) ayant la même catégorie $\text{cat}_i$
> - $\text{OptionCount}_i$ = nombre de ces lignes précédentes pour lesquelles $y_j = 1$ (classification binaire)
> - $\alpha$ = **prior** ($\approx 0.05$ par défaut), évite que les premières lignes (sans historique) aient un encoding dégénéré

**Calcul pas à pas sur l'exemple** (avec $\alpha = 0.05$) :

| Ligne (ordre aléatoire) | Favorite Color | $y$ | $n_i$ (prev. blue) | OptionCount | Encoded |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | blue | 1 | $0$ | $0$ | $(0 + 0.05)/(0 + 1) = 0.05$ |
| 2 | blue | 1 | $1$ | $1$ | $(1 + 0.05)/(1 + 1) = 0.525$ |
| 3 | blue | 0 | $2$ | $2$ | $(2 + 0.05)/(2 + 1) = 0.683$ |
| 4 | red  | 1 | $0$ | $0$ | $(0 + 0.05)/(0 + 1) = 0.05$ |

![[Pasted image 20260510220326.png|188]] ![[Pasted image 20260510220248.png|183]]
*Figure. Table avant / après application de l'Ordered Target Encoding. Chaque encoding n'utilise que les valeurs de $y$ des lignes précédentes ⇒ pas de leakage.*

![[Pasted image 20260510220404.png|188]]
*Figure. Résultat final : la colonne `Favorite Color` est remplacée par sa version encodée, qu'on peut ensuite traiter comme une feature numérique standard.*

> [!important] Pourquoi ça résout le leakage
> Comme l'encoding de la ligne $i$ ne dépend **pas** de $y_i$, il n'y a plus de contamination directe target → feature. L'algorithme est l'analogue exact de la **validation par série temporelle** appliquée à l'encoding catégoriel.

### VII.3 Construction d'un arbre — exemple à la main

Une fois les catégorielles encodées, on construit les arbres GBM. Reprenons un fil rouge avec une feature continue $X$ et une cible continue $y$.

**Préparation.** On découpe $y$ en bins (pour appliquer l'Ordered Target Encoding même si la cible est continue) :

![[Pasted image 20260510220626.png|119]] ![[Pasted image 20260510220818.png|153]]
*Figure. Découpage de $y$ en deux bins pour permettre l'encoding ordonné même sur cible continue.*

On ajoute les colonnes **Predictions** (initialisée à $0$) et **Residuals** ($= y - \hat{y}$, donc initialement $= y$) :

![[Pasted image 20260510220907.png|238]]
*Figure. Table augmentée avec Predictions et Residuals à $t = 0$.*

**Construction de l'arbre.** On trie les valeurs de $X$ et on calcule la moyenne entre paires successives — ce sont les **seuils candidats** :

![[Pasted image 20260510221118.png|253]]
*Figure. Seuils candidats sur la feature $X$ (moyennes entre valeurs successives triées).*

Pour chaque seuil, on construit un stump et on calcule le **Leaf Output** par feuille = moyenne des résidus dans la feuille :

![[Pasted image 20260510221243.png|265]] ![[Pasted image 20260510221339.png|270]]
*Figure. Deux stumps candidats (deux seuils différents) avec leurs Leaf Outputs respectifs.*

### VII.4 Cosine similarity pour comparer les arbres

> [!warning] Critère de sélection CatBoost
> Pour choisir entre plusieurs stumps candidats, CatBoost utilise la **similarité cosinus** entre le vecteur des résidus et le vecteur des Leaf Outputs (dupliqué par appartenance à chaque feuille) :
> 
> $$\cos\big(\vec{r},\, \vec{w}\big) = \frac{\vec{r}^\top \vec{w}}{\|\vec{r}\| \, \|\vec{w}\|}$$
> 
> On retient le stump qui **maximise** ce cosinus — c'est celui dont les Leaf Outputs *pointent dans la même direction* que les résidus à corriger.

|                  Stump 1                  |                  Stump 2                  |
| :---------------------------------------: | :---------------------------------------: |
| ![[Pasted image 20260510221449.png\|250]] | ![[Pasted image 20260510221508.png\|242]] |

*Figure. Calcul de la similarité cosinus pour les deux stumps candidats. On retient celui qui donne le plus grand cosinus.*

> [!note]- Pourquoi cosine et pas juste somme des carrés des résidus améliorée ?
> Intuitivement, le cosinus mesure l'**alignement directionnel** entre la correction proposée et le résidu à corriger, indépendamment de l'amplitude. Comme l'amplitude sera ensuite contrôlée par le learning rate $\eta$, l'amplitude du Leaf Output est moins importante que sa **direction**. Le cosinus capture exactement cette intuition.
> 
> En pratique, quand on a beaucoup de données, les premières lignes (peu d'historique pour l'encoding ordonné) sont des estimations bruitées des Leaf Outputs. CatBoost **les ignore** dans le calcul du cosinus — il ne prend en compte que les lignes "stables" avec assez d'historique.

### VII.5 Mise à jour et tour suivant

Une fois le stump retenu (avec son Leaf Output $w^\star$), on met à jour les prédictions :

$$\hat{y}_i^{\text{new}} = \hat{y}_i^{\text{old}} + \eta \cdot w^\star \quad \text{(avec } \eta = 0.1 \text{ typiquement)}$$

Puis on recalcule les résidus, on refait le ré-encoding ordonné (les bins de $y$ peuvent avoir changé), et on construit le stump suivant :

![[Pasted image 20260510221913.png|333]]
*Figure. Update des prédictions et des résidus après le premier arbre.*

![[Pasted image 20260510222152.png|341]]
*Figure. Ré-encoding ordonné pour le tour 2 avec les bins mis à jour.*

![[Pasted image 20260510222222.png|253]] ![[Pasted image 20260510222249.png|287]] ![[Pasted image 20260510222313.png|305]]
*Figures. Tour 2 — recherche du nouveau seuil optimal, construction de l'arbre, table mise à jour.*

![[Pasted image 20260510222419.png|496]] ![[Pasted image 20260510222451.png|394]]
*Figures. Prédiction additive et mise à jour cumulative après plusieurs arbres.*

### VII.6 Symmetric (Oblivious) Trees

> [!warning] Définition — Oblivious tree
> Un arbre est **symétrique** (ou **oblivious**) si **tous les nœuds d'un même niveau utilisent le même split**. À profondeur 3, l'arbre n'a que 3 seuils (un par niveau) au lieu de $1 + 2 + 4 = 7$ comme un arbre CART classique.

![[Pasted image 20260510222552.png|280]]
*Figure. Arbre oblivious de profondeur 2 : les deux nœuds du niveau 1 utilisent exactement le même split (Age < 12). On obtient des arbres très réguliers, équivalents à un **vote pondéré sur quelques tests indépendants**.*

**Pourquoi cette restriction** :

1. **Vitesse de prédiction** : un oblivious tree peut être implémenté comme une **table de lookup** indexée par les $d$ tests binaires — $O(d)$ avec $d$ = profondeur, sans branche dynamique. Hyperprévisible pour le CPU, très rapide.
2. **Régularisation implicite** : l'oblivious est un weak learner plus faible (moins flexible). En accord avec la philosophie boosting *"combiner beaucoup de learners très faibles"*, c'est en fait un atout pour la généralisation.
3. **Robustesse à l'overfitting** : moins de degrés de liberté ⇒ moins de capacité à mémoriser le bruit.

### VII.7 Quand préférer CatBoost ?

- **Données avec beaucoup de catégorielles à haute cardinalité** (villes, produits, IDs…) : l'Ordered Target Encoding fait des miracles.
- **Datasets petits-moyens** où le leakage est un risque réel : CatBoost est plus robuste que XGBoost/LightGBM avec target encoding naïf.
- **Latence de prédiction critique** (déploiement temps réel) : les oblivious trees sont 2-10× plus rapides à évaluer que des arbres CART classiques de même profondeur.

XGBoost et LightGBM gardent l'avantage sur des datasets purement numériques où l'encoding catégoriel n'est pas un enjeu, et sur des stacks Kaggle où la diversité de modèles est valorisée.

### VII.8 Comparatif final

| | XGBoost | LightGBM | CatBoost |
|:---|:---:|:---:|:---:|
| Année / Auteurs | 2016, Chen & Guestrin | 2017, Ke et al. (Microsoft) | 2018, Prokhorenkova et al. (Yandex) |
| Innovation principale | Régularisation L1/L2 explicite + Taylor 2 | Histogram + GOSS + EFB | Ordered TE + Oblivious trees |
| Croissance d'arbre | Level-wise (défaut) | Leaf-wise (best-first) | Oblivious / symmetric |
| Catégorielles natives | Non (encoding externe) | Partiel (`category` dtype) | **Oui, ordered TE intégré** |
| Sparse / missing | Sparsity-aware | EFB | Standard |
| Vitesse training | Référence | **~5-10× plus rapide** | Comparable XGBoost |
| Vitesse prédiction | Standard | Standard | **Très rapide (oblivious)** |
| Robustesse petit dataset | Bonne | Plus prone overfit (leaf-wise) | **Très bonne** |
