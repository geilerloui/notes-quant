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

## V. XGBoost Chen & Guestrin 2016


## VI. LightGBM - Ke et al 2017 Microsoft



https://www.youtube.com/watch?v=A2Xf8YgFdko



## VII. CatBoost Prokorenkova 2018 Yandex

il rappelle que target encoding est pas ouf car on fait du leakage; modèle efficace on training data mais pas testing data. Donc on a parlé de k-fold target encoding pour réduire le leakage. Après si tu lis le manuscript Catboost ils disent que si tu as que une seule catégorie eg tt le monde a "favorite color" qui est a blue - bon pas très claire mais en gros le mec te dit que la tu vois bien que classe 1 c'est favorite color = 0.33 sinon c'est 0.5 donc on a un méga leakage

![[Pasted image 20260510215629.png|225]]
![[Pasted image 20260510215847.png|229]]


il dit que c un exemple débile qui devrait pas se produire car si ta favorite color = blue pr tt le monde ben tu mets tt le monde a 1 et c tout. Mais les mecs qui ont fait le papier de catboost eux se sont dit que ça fait pas sens donc ils vont trouver une façon pr résoudre ce pb. Catboost = categorical boosting, 

catboost évite le leakge en etranine chaque row data as it it were fed recursively in the algorithm. 
Par exemple au lieu d'utiliser an overall mean it uses a user defined prior that in the examples I saw was set to 0.05 aussi le dénominateur on rajoute +1 lutot qu'un weight

$$
\begin{aligned}
& \text { CatBoost } \\
& \text { Encoding }
\end{aligned}=\frac{\text { OptionCount }+0.05}{n+1}
$$$n=$ Number of rows that have already been seen that have the same value for Favorite Color

il dit ligne 1 j'ai 0+0.05/(0+1) = 0.05
il dit ligne 2 = 0+0.05/(0+1)=0.05
il dit ligne 3 = 0+0.05/(0+1)=0.05
il dit ligne 4 = 1+0.05/(1+1)=0.525


![[Pasted image 20260510220326.png|220]]

![[Pasted image 20260510220248.png|225]]

et c'est comme ça comme catboost perform target encoding - on dit Ordered Target encoding.

![[Pasted image 20260510220404.png|228]]

je pense le trucr qui est pas expliqué c'est comme tu fais sur le validation/test set ? 

#### Using trees


Gross différence par rapport à avant maintenant (y) est continu et pas catégorielle. il dit en gros on va définir deux bins déjà et ensuite on pourra utiliser notre ordered target encoding qu'on a vu

![[Pasted image 20260510220626.png|173]]


![[Pasted image 20260510220818.png|181]]

Une fois qu'on a fait le preprocessing du ordered target encoding on vire la colonne Bin # et on rajoute deux colonnes supplémentaires : Predictions et Residuals = y - \hat{y} mais vu que \hat{y} est est à zero au début on a égalité entre y et residuals

![[Pasted image 20260510220907.png|315]]

Puis on va définir un arbre au début il prenne la colonne X il la sort par ordre et on calcule la moyenne des deux valeurs successives : 

![[Pasted image 20260510221118.png|388]]


Après cest un peu random les mecs te mettent les residuals dans chacune des feuilles et ils calculenet la moyenne ce qu'ils appellent Leaf output



![[Pasted image 20260510221243.png|313]]
ok
![[Pasted image 20260510221339.png|352]]

puis pour mesurer si la prédiction est bonne il calcule cosinus(residuals, leaf output)


![[Pasted image 20260510221449.png|250]]

et 
![[Pasted image 20260510221508.png|242]]

conclusion on choisit le second car il a une cosinus plus élevé

apres je sais pas ce qu'il raconte: it doesnt make a lot of sense to include leaf output values taht are not based on data in the cosine similairty calculation. So in practice when you have a lot of data, catbost simply ignores the first bunch of rows when calculating the cosine similarity

une fois qu'on a fait le choix on va updater les predictions 

new prediction = prediction + (learning rate x leaf output) ici lr=0.1

il dit on a une amélioration faible mais c'est toujours mieux que de prédire que tout le monde a height=0 qui est ce qu'on prédisait au tout début.

puis en gros il te dit que apres maintenant qu'on a choisi l'arbre on va pouvoir update la colonne Predictions et donc aussi les residuals  et comme on a fait précédemement on remets Favorite color pas en mode ordered target encoding

![[Pasted image 20260510221913.png|508]]


on refait le orderd target encoding en ayant remodifier les bins


![[Pasted image 20260510222152.png|525]]

et on refait le sorting on onbtient un seuil a 0.29
![[Pasted image 20260510222222.png|371]]



oué l'arbre
![[Pasted image 20260510222249.png|364]]

et la table
![[Pasted image 20260510222313.png|361]]

ensuite prédiction
![[Pasted image 20260510222419.png]]

et ensuite on add up les values des trees : c'est très mauvaois mais c'est que des moini arbres 

![[Pasted image 20260510222451.png|448]]



il dit que catboost builds oblivious or symmetric decision trees - a symmetric = uses the exact the same threshold for the same node in the same level eg both node uses the same threshold age <12 

![[Pasted image 20260510222552.png|280]]
y'a deux raisons : 
* ça empire les prédictions de l'arbres
* remember teh whole idea of gradient boostin si to combine a bunch of weak learners to make decisions and symmetric decision trees are just a weaker type of learner