# Decision Trees

Un arbre de décision découpe l'espace des features en régions rectangulaires alignées sur les axes (les *feuilles*), et associe à chaque feuille une prédiction. C'est l'un des modèles supervisés les plus simples conceptuellement, mais aussi la **brique de base** des méthodes d'ensemble (Random Forest, Gradient Boosting) qu'on verra dans [[02_Bagging_Methods]] et [[03_Boosting_Algorithms]].

Deux variantes selon la cible :
- **Classification** ($y$ catégoriel) : la feuille prédit la classe majoritaire — §A
- **Régression** ($y$ continu) : la feuille prédit la moyenne des $y_i$ — §B

L'algorithme de construction (CART, Breiman 1984) est essentiellement le même : à chaque nœud, on cherche le split qui minimise une **mesure d'impureté** (Gini en classification, SSR en régression).

---

## A. Classification

> [!example] Fil rouge : diagnostic grippe
> Pour ancrer chaque définition, on utilisera tout au long de la section un dataset de **diagnostic grippe** (flu) à 10 patients, avec trois symptômes binaires (Fatigue, Fièvre, Toux) et une étiquette cible Grippe (oui/non). On veut construire un arbre de décision qui prédit la grippe à partir des symptômes. Cet exemple suffit à illustrer toute la mécanique de CART : impureté, splits, élagage, feature importance.

**Rappel** (cf [[01_Fondation]] §I.B.3 et §I.C.2). En classification on cherche $\hat f : \mathcal{X} \to \{1, \ldots, K\}$ qui minimise le taux d'erreur empirique. Trois familles de modèles existent : génératifs, discriminatifs, et *discriminant functions*. Les arbres de décision sont des **discriminant functions** : ils prédisent directement une classe, sans probabilité intermédiaire.

Concrètement, un arbre découpe l'espace $\mathcal{X}$ en régions disjointes $R_1, \ldots, R_J$ (les feuilles), et associe à chaque feuille une classe. Tout se joue donc en deux temps :

1. **Décider quoi prédire dans chaque feuille** une fois l'arbre construit (§1)
2. **Choisir un critère pour mesurer la qualité d'une feuille** : Gini ou entropie (§2)
3. **Construire l'arbre** lui-même : choisir les coupures qui rendent les feuilles le plus *pures* possible (§3)

---

### 1. Décision dans une feuille

Avant de parler de comment construire l'arbre, posons d'abord la mécanique de base : à quoi ressemble une feuille, et que prédit-on quand un patient y tombe ?

Une feuille $R_j$ est une région de l'espace définie par les coupures qui mènent à elle (par exemple : "Fatigue = oui ET Fièvre = non"). Dans cette région tombent un certain nombre d'observations du training set, et on note leur composition par classe.

> [!warning] Définition — Proportion de classe dans une feuille
> Soit $R_j$ une feuille contenant $|R_j|$ observations, et $K$ classes possibles. La **proportion de la classe $k$** dans $R_j$ est définie par
> 
> $$p_k(R_j) = \frac{1}{|R_j|} \sum_{i \in R_j} \mathbf{1}[y_i = k]$$
> 
> C'est simplement la fréquence empirique de chaque classe dans la feuille. Par construction, $\sum_k p_k(R_j) = 1$ et $p_k(R_j) \in [0, 1]$.

Une fois l'arbre construit, la règle de prédiction est simple : pour un nouveau patient qui tombe dans la feuille $R_j$, on prédit la classe **majoritaire** dans cette feuille.

> [!warning] Définition — Règle de décision dans une feuille
> Pour toute feuille $R_j$, on assigne la classe
> 
> $$c_j = \underset{k \in \{1, \ldots, K\}}{\arg\max} \; p_k(R_j)$$
> 
> et tout point $x \in R_j$ se voit prédire $\hat f(x) = c_j$. C'est ce qui fait des arbres une *discriminant function* : on sort une classe directement, sans probabilité.

> [!example] Grippe — décision dans une feuille
> Considérons une feuille $R_j$ contenant 10 patients dont 7 sains et 3 malades :
> 
> $$p_0(R_j) = 0.7, \qquad p_1(R_j) = 0.3 \;\Rightarrow\; c_j = \arg\max\{0.7, 0.3\} = 0 \text{ (sain)}$$
> 
> Tout nouveau patient qui tombera dans cette feuille sera donc prédit *sain*. Les 3 patients malades qui y tombaient seront mal classés — c'est ce qu'on appelle l'**erreur de la feuille**, $1 - \max_k p_k(R_j) = 0.3$.

![[Pasted image 20260418131332.png|338]]
*Figure. Une feuille $R_j$ contenant des observations de deux classes. La règle de décision retient la classe majoritaire (ici les ronds) ; les croix qui tombent dans cette feuille seront mal classées.*

---

### 2. Mesurer l'impureté : Gini vs Entropie

Maintenant qu'on sait quoi prédire dans une feuille, on veut construire l'arbre. Pour ça, il nous faut un **critère** qui mesure à quel point une feuille est "bien" — autrement dit, à quel point elle est homogène : si tous les patients qui y tombent ont la même étiquette, la feuille est *pure* et l'erreur est nulle ; si les classes sont mélangées, la feuille est *impure* et l'erreur est forte.

#### Origine

L'indice de Gini est introduit en économie par Corrado Gini en 1912 pour mesurer les inégalités de revenus dans une population. Soixante-douze ans plus tard, Breiman le réutilise dans CART (Classification and Regression Trees, 1984) comme **critère de split** : à chaque nœud de l'arbre, on choisit la coupure qui réduit le plus l'impureté de Gini des feuilles.

Deux variantes apparaîtront ensuite, basées sur la théorie de l'information plutôt que sur Gini :

- **ID3** (Iterative Dichotomiser 3, Quinlan 1986) : utilise l'**entropie** $H = -\sum_k p_k \log p_k$ et le **gain d'information** comme critère de split.
- **C4.5** (Quinlan 1993) : extension industrielle d'ID3 — gestion des valeurs manquantes, des features continues, élagage post-construction. C'est la base de nombreuses implémentations historiques.

Dans cette note on suit la formulation CART (Gini) qui est la plus utilisée aujourd'hui (notamment par défaut dans scikit-learn). Gini et entropie donnent en pratique des arbres très similaires — on revient sur leur comparaison à la fin.

#### Construction de l'impureté de Gini

L'idée de Breiman pour mesurer l'impureté : on tire deux patients au hasard dans la feuille, et on regarde la probabilité qu'ils soient de **classes différentes**. Plus cette probabilité est grande, plus la feuille est mélangée — donc impure.

> [!note]- Construction par tirage de deux patients
> On tire deux patients $A$ et $B$ de façon aléatoire et **indépendante** dans la feuille $R_j$, et on cherche $\mathbb{P}(A \neq B)$, la probabilité qu'ils soient de classes différentes.
> 
> **Approche directe.** On énumère les cas favorables : soit $A=1, B=0$, soit $A=0, B=1$. Ces événements sont disjoints, donc :
> 
> $$\mathbb{P}(A \neq B) = \mathbb{P}(A=1, B=0) + \mathbb{P}(A=0, B=1)$$
> 
> Par indépendance, chaque terme se factorise. Avec $p_1 = p_1(R_j)$ pour alléger :
> 
> $$\mathbb{P}(A \neq B) = p_1(1-p_1) + (1-p_1)p_1 = 2p_1(1-p_1)$$
> 
> **Approche par complémentaire (plus élégante).** On calcule plutôt $\mathbb{P}(A = B)$, puis on prend $1$ moins :
> 
> $$\mathbb{P}(A = B) = \sum_{k=0}^{K-1} \mathbb{P}(A = k, B = k) = \sum_{k=0}^{K-1} p_k^2$$
> 
> par indépendance. Donc :
> 
> $$\mathbb{P}(A \neq B) = 1 - \sum_{k=0}^{K-1} p_k(R_j)^2$$
> 
> Cette deuxième forme se généralise immédiatement à $K$ classes — c'est elle qu'on retient comme définition. $\square$

> [!warning] Définition — Impureté de Gini
> L'**impureté de Gini** d'une feuille $R_j$ est la probabilité que deux observations tirées indépendamment dans $R_j$ soient de classes différentes :
> 
> $$G(R_j) = 1 - \sum_{k=1}^{K} p_k(R_j)^2$$
> 
> - **Cas pur** ($G = 0$) : une seule classe est présente, donc $p_k = 1$ pour un certain $k$ et $0$ pour les autres. Deux patients tirés au hasard sont **toujours** de la même classe.
> - **Cas le plus impur** : la distribution est uniforme ($p_k = 1/K$ pour tout $k$), et $G = 1 - K \cdot (1/K)^2 = 1 - 1/K$. Pour $K = 2$, ça donne $G_{\max} = 1/2$.

> [!example] Grippe — Gini de la feuille racine
> Avant tout split, la feuille racine contient les 10 patients du dataset, dont 4 ont la grippe et 6 ne l'ont pas. Donc $p_0 = 0.6$, $p_1 = 0.4$ et :
> 
> $$G(R_{\text{racine}}) = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48$$
> 
> On est proche du maximum théorique $G_{\max} = 0.5$ — la feuille racine est très mélangée, ce qui est logique : on n'a encore fait aucune coupure. C'est cette valeur de **0.48 qu'on cherche à diminuer** en splittant.

#### Comparaison avec l'entropie

L'autre mesure classique d'impureté est l'**entropie de Shannon** :

> [!warning] Définition — Entropie de Shannon
> $$H(R_j) = -\sum_{k=1}^K p_k(R_j) \log p_k(R_j)$$
> 
> avec la convention $0 \log 0 = 0$. Comme Gini, l'entropie est nulle quand la feuille est pure et maximale pour la loi uniforme.

Gini et entropie partagent les mêmes propriétés qualitatives : nulles quand la feuille est pure, maximales pour la loi uniforme, concaves en $p_k$. La courbe en fonction de $p_1$ (cas binaire) est très similaire — Gini est essentiellement une approximation polynomiale de l'entropie. En pratique, les deux donnent des arbres quasi identiques ; **Gini est légèrement moins coûteux à calculer** (pas de logarithme), c'est pourquoi il est par défaut dans la plupart des implémentations.

![[Pasted image 20260415193533.png|427]]
*Figure. Comparaison Gini vs entropie en fonction de $p_1$ pour $K=2$. Les deux courbes sont nulles aux extrêmes (feuille pure) et maximales en $p_1 = 0.5$ (feuille uniforme). Note : l'entropie est ici renormalisée pour permettre la comparaison visuelle.*

---

### 3. Construire l'arbre : algorithme CART

Maintenant qu'on a un critère d'impureté, on peut construire l'arbre. CART (Classification and Regression Trees, Breiman 1984) procède de façon **gloutonne** : à chaque nœud, on cherche la coupure qui réduit le plus l'impureté des feuilles enfants, puis on récurse.

---
**Le critère : Gini pondéré sur un split**

Pour une variable $j$ et un seuil $s$, le split sépare la feuille parente en deux régions :

$$R_1 = \{X \in \mathbb{R}^p : X_j \leq s\}, \quad R_2 = \{X \in \mathbb{R}^p : X_j > s\}$$

(Pour une variable binaire comme Fatigue, $s$ est trivial : oui d'un côté, non de l'autre.) On calcule l'impureté de Gini dans chaque feuille enfant, puis on les moyenne **pondérées par la taille** de chaque feuille :

> [!warning] Définition — Gini pondéré d'un split
> Pour un split qui produit deux feuilles enfants $R_1, R_2$ avec $n_1, n_2$ observations et $N = n_1 + n_2$ :
> 
> $$G_{\text{split}} = \frac{n_1}{N} \cdot G(R_1) + \frac{n_2}{N} \cdot G(R_2)$$
> 
> Le **gain** apporté par le split par rapport à la feuille parente $R_p$ est :
> 
> $$\text{Gain} = G(R_p) - G_{\text{split}}$$

L'algorithme glouton cherche, à chaque nœud, le couple $(j, s)$ qui **minimise** $G_{\text{split}}$ (équivalent : qui maximise le gain).

#### Étape 1 - Construction de l'arbre

**(i) Gini initial**

![[Pasted image 20260415185926.png|261]]
*Table 1. Le dataset grippe : 10 patients, 3 symptômes binaires (Fatigue, Fièvre, Toux), étiquette cible Grippe.*

Sur le dataset complet *(Table 1)*, on calcule le Gini sans split sur la cible Grippe :

$$G_{\text{initial}} = 1 - (0.6^2 + 0.4^2) = 0.48$$

**(ii) Choix du 1er split**

On teste chaque feature pour trouver celle qui réduit le plus l'impureté.

**Feature "Fatigue ?".** Le split sépare en deux feuilles :
- Fatigue = oui (4 patients, 3 grippe / 1 sain) :
  $$G_{\text{Oui}} = 1 - \left(\tfrac{3}{4}^2 + \tfrac{1}{4}^2\right) = 1 - (0.5625 + 0.0625) = 0.375$$
- Fatigue = non (6 patients, 1 grippe / 5 sains) :
  $$G_{\text{Non}} = 1 - \left(\tfrac{1}{6}^2 + \tfrac{5}{6}^2\right) = 1 - (0.0278 + 0.6944) = 0.278$$

Gini pondéré :
$$G_{\text{Fatigue}} = \tfrac{4}{10} \cdot 0.375 + \tfrac{6}{10} \cdot 0.278 = 0.150 + 0.167 = 0.317$$

On répète pour "Fièvre ?" et "Toux ?". Le minimum est obtenu pour Fatigue : c'est **la feature retenue** pour le 1er split. Le gain est :

$$\text{Gain}_{\text{Fatigue}} = 0.48 - 0.317 = 0.163$$

![[Pasted image 20260415185817.png|583]]
*Figure. Comparaison des trois stumps candidats (Fatigue, Fièvre, Toux) au 1er split. On retient celui qui minimise le Gini pondéré : Fatigue.*

**(iii) Choix du 2nd split**

On récurse sur chaque feuille de l'étape précédente. Sur la branche "Fatigue = oui" et la branche "Fatigue = non", on teste les features restantes (Fièvre, Toux). Le calcul du Gini pondéré désigne **Fièvre** comme meilleur split sur (au moins) une des branches.

$$\text{Gain}_{\text{Fièvre}} = 0.317 - 0.25 = 0.067$$

![[Pasted image 20260415191756.png|354]]
*Figure. Le 2nd split sur Fièvre.*

**(iv) Choix des 3e et 4e splits**

On continue à splitter récursivement jusqu'à ce que chaque feuille soit pure (ou qu'un critère d'arrêt soit atteint, voir §5). Sur le dataset grippe, deux splits supplémentaires suffisent à obtenir un arbre où toutes les feuilles sont pures.

![[Pasted image 20260418102957.png|482]]
*Figure. Les 3e et 4e splits — l'arbre est complètement développé, chaque feuille est pure.*

**(v) Critères d'arrêt**

Sur ce petit dataset on a pu arrêter parce que **toutes les feuilles sont devenues pures**. En général c'est rare, et on doit ajouter des critères d'arrêt explicites pour éviter un arbre trop profond qui overfitte. Les principaux :

- **`max_depth`** : profondeur maximale de l'arbre. Limite directe sur la complexité.
- **`min_samples_split`** : nombre minimum d'observations dans un nœud pour qu'il soit splittable. Empêche les splits sur des feuilles trop petites (peu de données = bruit).
- **`min_samples_leaf`** : nombre minimum d'observations qu'une feuille doit contenir. Garantit une statistique minimale dans chaque prédiction.
- **`min_impurity_decrease`** : si le meilleur split apporte un gain d'impureté inférieur à ce seuil, on s'arrête. Évite les splits inutiles.

#### Étape 2 — Élagage de l'arbre

Même avec des critères d'arrêt, on peut produire un arbre $T_0$ inutilement profond. L'élagage (pruning) consiste à **simplifier l'arbre a posteriori** en fusionnant certaines feuilles dans leur parent.

> [!warning] Définition — Cost complexity pruning
> Pour un arbre $T$ avec $|T|$ feuilles, on définit la fonction de coût :
> 
> $$C_\alpha(T) = \sum_{j=1}^{|T|} \big[1 - \hat{p}_{c_j}(R_j)\big] + \alpha \cdot |T|$$
> 
> - Le 1er terme est l'**erreur d'entraînement** de l'arbre (somme des proportions mal classées dans chaque feuille).
> - Le 2nd terme est une **pénalité** proportionnelle à la taille de l'arbre, contrôlée par $\alpha \geq 0$.
> 
> On cherche $T \subseteq T_0$ qui minimise $C_\alpha(T)$.

L'idée du pruning est exactement celle d'une régularisation $\ell_1$ sur le nombre de feuilles : pour $\alpha = 0$ on garde l'arbre complet $T_0$ ; quand $\alpha$ augmente, la pénalité $\alpha |T|$ devient plus contraignante et on coupe les feuilles dont le gain en erreur ne compense pas leur coût. À la limite $\alpha \to \infty$, l'arbre se réduit à sa racine (une seule feuille = la classe majoritaire globale).

**Comment choisir $\alpha$.** $\alpha$ est un hyperparamètre, choisi par **cross-validation** (typiquement 5 ou 10 folds). En pratique : on trie les sous-arbres possibles obtenus en élaguant successivement les feuilles les plus faibles (weakest link pruning), on évalue chacun par CV, et on garde celui qui minimise l'erreur moyenne sur les folds de validation.

#### Étape 3 — Visualisation des plans

Une fois l'arbre construit, chaque feuille correspond à une **région rectangulaire** de l'espace des features (intersection des coupures qui mènent à la feuille). C'est une particularité des arbres : ils découpent l'espace en boîtes alignées sur les axes.

| Graphe complet                            | Feuilles                                  |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260418133309.png\|305]] | ![[Pasted image 20260418131332.png\|310]] |
|                                           |                                           |
*Figures. Visualisation des plans de partition de l'espace des features par l'arbre de décision. Chaque rectangle correspond à une feuille, et reçoit la classe majoritaire des points qui y tombent.*

#### Étape 4 — Feature importance

Une dernière chose qu'un arbre nous donne gratuitement : une mesure de **quelle feature est la plus utile** pour la prédiction. L'idée : sommer le gain d'impureté apporté par chaque split, pondéré par la taille de la feuille parente.

> [!warning] Définition — Feature importance (Mean Decrease Impurity)
> Pour une feature $X_j$, son importance est la somme, sur tous les splits qui utilisent $X_j$, du gain pondéré apporté :
> 
> $$\text{Imp}(X_j) = \sum_{\text{split sur } X_j} \frac{n_{\text{parent}}}{N} \cdot \big[G(R_{\text{parent}}) - G_{\text{split}}\big]$$
> 
> Plus une feature est utilisée tôt et apporte de gros gains, plus son importance est grande.

> [!example] Grippe — calcul des importances
> Pour ton arbre (split racine sur Fatigue, puis Fièvre dans la branche gauche, Toux dans la branche droite, puis Toux pour finir de splitter la sous-branche Fatigue=Oui ∧ Fièvre=Non), il y a 4 splits internes. On somme la contribution de chacun avec $N = 10$ :
> 
> | Feature | Splits utilisés | Calcul | Importance |
> |:---:|:---|:---|:---:|
> | **Fatigue** | racine ($n=10$, gain=$0.163$) | $\tfrac{10}{10} \times 0.163$ | $0.163$ |
> | **Fièvre** | branche Fatigue=Oui ($n=4$, gain=$0.125$) | $\tfrac{4}{10} \times 0.125$ | $0.050$ |
> | **Toux** | branche Fatigue=Non ($n=6$, gain=$0.111$) + sous-branche F=Oui $\wedge$ Fi=Non ($n=2$, gain=$0.5$) | $\tfrac{6}{10} \times 0.111 + \tfrac{2}{10} \times 0.5$ | $0.167$ |
> 
> **Toux et Fatigue sont quasi à égalité** ($0.167$ vs $0.163$), Fièvre loin derrière ($0.050$). Toux est utilisée deux fois (dans deux sous-arbres différents), c'est ce qui la rend légèrement dominante.
> 
> > [!note]- Vérification de la conservation
> > Sur un arbre sans feuille parfaitement pure (ici la sous-branche Fatigue=Non $\wedge$ Toux=Oui contient encore $\{5, 7\}$ avec 1G/1S si on s'arrête là), on a la propriété :
> > 
> > $G_{\text{init}} = \sum_{\text{features}} \text{Imp}(X_j) + \sum_{\text{feuilles}} \frac{n_f}{N} G(R_f)$
> > 
> > Ici : $0.163 + 0.050 + 0.167 = 0.380$, plus le Gini résiduel des feuilles impures $= \frac{2}{10} \times 0.5 = 0.100$. Total : $0.380 + 0.100 = 0.480 = G_{\text{init}}$ $\checkmark$. C'est rassurant : chaque parcelle d'impureté initiale est soit absorbée par un split (et compte dans une feature importance), soit reste dans une feuille finale.
> 
> Sklearn renvoie par défaut les importances **normalisées** (divisées par leur somme $0.380$), ce qui donne :
> 
> | Feature | Importance normalisée |
> |:---:|:---:|
> | Toux | $0.439$ |
> | Fatigue | $0.430$ |
> | Fièvre | $0.132$ |

#### Hyperparamètres principaux

Récapitulatif des hyperparamètres d'un Decision Tree (scikit-learn) :

- **`max_depth`** : profondeur maximale.
- **`min_samples_split`** : taille minimale d'un nœud pour le splitter.
- **`min_samples_leaf`** : taille minimale d'une feuille.
- **`min_impurity_decrease`** : gain minimum pour autoriser un split.
- **`ccp_alpha`** : paramètre $\alpha$ du cost complexity pruning.
- **`criterion`** : `gini` (par défaut) ou `entropy`.

---
## B. Régression

> [!example] Fil rouge : dosage et score santé
> Pour la régression, on change de cadre : la cible $y$ est désormais **continue**. On reprend un dataset à 6 patients avec deux features ($X_1$ = Dosage, $X_2$ = Score Santé) et $y$ une mesure de réponse. L'objectif est le même que pour la classification — découper l'espace en régions $R_j$ — mais cette fois **chaque feuille prédit une valeur réelle** (la moyenne des $y$ de la feuille), et le critère de qualité change : on remplace l'impureté de Gini par le **SSR** (Sum of Squared Residuals).

La structure est rigoureusement la même qu'en classification :

1. **Décider quoi prédire dans chaque feuille** : la moyenne des $y_i$ qui tombent dans la feuille (§1)
2. **Choisir un critère pour mesurer la qualité d'une feuille** : le SSR (§2)
3. **Construire l'arbre** : choisir les splits qui minimisent le SSR pondéré (§3)

---

### 1. Décision dans une feuille

En régression, la cible $y \in \mathbb{R}$. Pour une feuille $R_j$ contenant un certain nombre d'observations, le choix naturel de prédiction est la **moyenne** des valeurs $y_i$ qui y tombent — c'est ce qui minimise la MSE dans la feuille (cf [[01_Fondation]] §I.B.1, l'espérance conditionnelle est le minimiseur de la MSE).

> [!warning] Définition — Règle de décision dans une feuille (régression)
> Pour toute feuille $R_j$ contenant $|R_j|$ observations, on assigne la valeur
> 
> $$\bar y_j = \frac{1}{|R_j|} \sum_{i \in R_j} y_i$$
> 
> et tout point $x \in R_j$ se voit prédire $\hat f(x) = \bar y_j$.

L'erreur sur la feuille est mesurée par la **somme des carrés des résidus** (SSR) — chaque observation $y_i$ s'écarte de la prédiction $\bar y_j$, on somme les écarts au carré :

$$\text{SSR}(R_j) = \sum_{i \in R_j} (y_i - \bar y_j)^2$$

Une feuille pure en classification (toutes les obs de la même classe) devient ici une feuille **homogène** : tous les $y_i$ sont proches, donc $\text{SSR} \to 0$.

---

### 2. Mesurer l'erreur : SSR

L'analogue du Gini en régression est directement le SSR. Plus une feuille a un SSR élevé, plus elle est "impure" au sens régression (les $y$ sont éparpillés autour de la moyenne).

> [!warning] Définition — SSR pondéré d'un split
> Pour un split qui produit deux feuilles enfants $R_1, R_2$ avec $n_1, n_2$ observations :
> 
> $$\text{SSR}_{\text{split}} = \text{SSR}(R_1) + \text{SSR}(R_2)$$
> 
> Le **gain** apporté par le split est :
> 
> $$\text{Gain} = \text{SSR}(R_{\text{parent}}) - \text{SSR}_{\text{split}}$$

Note : contrairement à Gini, la formule SSR pondérée est implicite — chaque carré est déjà pondéré par le nombre d'observations dans sa feuille (c'est une somme, pas une moyenne).

> [!example] Dosage et Score Santé — SSR initial
> Avant tout split, sur les 6 patients du dataset :
> 
> $$\text{SSR}_{\text{initial}} = 1156 + 81 + 1296 + 1849 + 1 + 1369 = \mathbf{5752}$$
> 
> ![[Pasted image 20260415195718.png|446]]
> 
> ![[Pasted image 20260415200316.png]]

---

### 3. Construire l'arbre : algorithme CART

Comme pour la classification, CART procède de façon **gloutonne** : à chaque nœud, on cherche le couple (variable, seuil) qui minimise le SSR pondéré des feuilles enfants, puis on récurse.

---
**Le critère : SSR pondéré sur un split**

Pour une variable continue $X_j$ et un seuil $s$, le split sépare la feuille parente en deux régions :

$$R_1 = \{X \in \mathbb{R}^p : X_j \leq s\}, \quad R_2 = \{X \in \mathbb{R}^p : X_j > s\}$$

L'algorithme glouton cherche, à chaque nœud, le couple $(j, s)$ qui **minimise** $\text{SSR}_{\text{split}}$ (équivalent : maximise le gain en SSR).

#### Étape 1 — Construction de l'arbre

**(i) SSR initial**

On a déjà calculé $\text{SSR}_{\text{initial}} = 5752$ sur le dataset complet (cf §2).

**(ii) Choix du 1er split**

On teste différents seuils sur chaque feature et on garde celui qui minimise le SSR pondéré.

![[Pasted image 20260415200754.png]]

**Test sur le Dosage ($x_1$).** On tente un split "médian" pour séparer la montée de la descente, par exemple à $x_1 = 8.5$ (entre les patients 2 et 3).
- Groupe Gauche ($x_1 < 8.5$) : Patients 1, 2.
  - Valeurs $y$ : $\{15, 40\}$. Moyenne $\bar y_G = 27.5$.
  - $\text{SSR}_G = (15 - 27.5)^2 + (40 - 27.5)^2 = 156.25 + 156.25 = \mathbf{312.5}$
- Groupe Droite ($x_1 > 8.5$) : Patients 3, 4, 5, 6.
  - Valeurs $y$ : $\{85, 92, 50, 12\}$. Moyenne $\bar y_D = 59.75$.
  - $\text{SSR}_D = (85 - 59.75)^2 + (92 - 59.75)^2 + (50 - 59.75)^2 + (12 - 59.75)^2 \approx 637 + 1040 + 95 + 2280 = \mathbf{4052}$
- **SSR total après split** $x_1 = 312.5 + 4052 = \mathbf{4364.5}$
- **Gain** : $5752 - 4364.5 = \mathbf{1387.5}$

**Test sur le Score Santé ($x_2$).** On tente un seuil $x_2 = 40$.
- Groupe Gauche ($x_2 < 40$) : Patients 1, 3, 5.
  - Valeurs $y$ : $\{15, 85, 50\}$. Moyenne $\bar y_G = 50$.
  - $\text{SSR}_G = (15 - 50)^2 + (85 - 50)^2 + (50 - 50)^2 = 1225 + 1225 + 0 = \mathbf{2450}$
- Groupe Droite ($x_2 > 40$) : Patients 2, 4, 6.
  - Valeurs $y$ : $\{40, 92, 12\}$. Moyenne $\bar y_D = 48$.
  - $\text{SSR}_D = (40 - 48)^2 + (92 - 48)^2 + (12 - 48)^2 = 64 + 1936 + 1296 = \mathbf{3296}$
- **SSR total après split** $x_2 = 2450 + 3296 = \mathbf{5746}$
- **Gain** : $5752 - 5746 = \mathbf{6}$. (C'est presque inutile !)

Le 1er split retient **Dosage à $x_1 = 8.5$** (gain 1387.5 contre 6 pour Score Santé).

![[Pasted image 20260415201016.png|407]]

**(iii) Choix du 2nd split**

On récurse sur la branche Droite (Patients 3, 4, 5, 6) qui a encore un SSR élevé (4052).

![[Pasted image 20260415201127.png|441]]

L'algorithme cherche à séparer ces 4 points. Le split le plus logique pour capturer la "chute" de la cloche est entre le Patient 4 et le Patient 5. Seuil : $(15.5 + 22.0) / 2 = \mathbf{18.75}$.

- Sous-groupe Droite-Gauche ($8.5 \leq x_1 < 18.75$) : Patients 3, 4.
  - $y = \{85, 92\}$. Moyenne $\bar y_{DG} = 88.5$.
  - $\text{SSR}_{DG} = (85 - 88.5)^2 + (92 - 88.5)^2 = 12.25 + 12.25 = \mathbf{24.5}$
- Sous-groupe Droite-Droite ($x_1 \geq 18.75$) : Patients 5, 6.
  - $y = \{50, 12\}$. Moyenne $\bar y_{DD} = 31$.
  - $\text{SSR}_{DD} = (50 - 31)^2 + (12 - 31)^2 = 361 + 361 = \mathbf{722}$

**SSR total de l'arbre** : $312.5$ (branche de gauche inchangée) $+ 24.5 + 722 = \mathbf{1059}$. On est passé d'un SSR de $5752$ (départ) à $1059$ — l'arbre devient très précis.

![[Pasted image 20260415201153.png|320]]

**(iv) Arbre final**

![[Pasted image 20260415201352.png|437]]
*Figure. L'arbre de régression complet.*

**(v) Critères d'arrêt**

Comme en classification, on arrête de splitter quand un critère d'arrêt est atteint :

- **`max_depth`** : profondeur maximale.
- **`min_samples_split`** : nombre minimum d'observations dans un nœud pour qu'il soit splittable.
- **`min_samples_leaf`** : nombre minimum d'observations qu'une feuille doit contenir.
- **`min_impurity_decrease`** : gain minimum (en SSR) pour autoriser un split.

#### Étape 2 — Élagage de l'arbre

La construction ci-dessus se fait sur le training set, mais le SSR du training ne fait que diminuer à chaque split — l'arbre risque l'overfitting. Pour évaluer la vraie qualité, il faut un **validation set** qui calcule son propre SSR, et c'est de là que sort le paramètre $\alpha$ du cost complexity pruning : on élague l'arbre pour minimiser $\text{SSR}(T) + \alpha |T|$, en choisissant $\alpha$ par cross-validation.

![[Pasted image 20260415204452.png]]

#### Étape 3 — Visualisation des plans

| Graphe complet                            | Plan de partition                         |
| ----------------------------------------- | ----------------------------------------- |
| ![[Pasted image 20260415202158.png\|313]] | ![[Pasted image 20260415203227.png\|313]] |
|                                           |                                           |
*Figures. À gauche : visualisation 3D de la fonction prédite $\hat f(x_1, x_2)$ — chaque "marche" correspond à une feuille. À droite : plan de partition $(x_1, x_2)$ — chaque rectangle est une feuille colorée par la valeur prédite.*

#### Étape 4 — Feature importance

La formule MDI s'adapte directement à la régression : on remplace simplement le gain en Gini par le gain en SSR.

> [!warning] Définition — Feature importance en régression
> Pour une feature $X_j$, son importance est la somme, sur tous les splits qui utilisent $X_j$, du gain en SSR pondéré par la taille de la feuille parente :
> 
> $$\text{Imp}(X_j) = \sum_{\text{split sur } X_j} \big[\text{SSR}(R_{\text{parent}}) - \text{SSR}_{\text{split}}\big]$$
> 
> En régression, le poids $n_{\text{parent}}/N$ est implicite dans la définition du SSR (qui est une somme, pas une moyenne).

> [!example] Dosage et Score Santé — calcul des importances
> Reprenons les étapes de construction pour calculer l'importance cumulée :
> 
> 1. **Dosage ($X_1$)** :
>    - Split 1 ($t_1 = 8.5$) : Réduction de $5485.5$
>    - Split 2 ($t_2 = 18.75$) : Réduction de $313.3$
>    - **Total Dosage** : $5485.5 + 313.3 = \mathbf{5798.8}$
> 2. **Score Santé ($X_2$)** :
>    - Split 3 ($t_3 = 45.5$) : Réduction de $722.0$
>    - **Total Santé** : $\mathbf{722.0}$
> 
> Dosage est de loin la feature dominante, ce qui colle avec l'intuition : c'est elle qui structure la "cloche" en $y$.

![[Pasted image 20260415203850.png]]
*Figure. Feature importance plot : Dosage domine largement Score Santé.*

#### Hyperparamètres principaux

Mêmes hyperparamètres qu'en classification, sauf que `criterion` change :

- **`max_depth`** : profondeur maximale.
- **`min_samples_split`** : taille minimale d'un nœud pour le splitter.
- **`min_samples_leaf`** : taille minimale d'une feuille.
- **`min_impurity_decrease`** : gain minimum pour autoriser un split.
- **`ccp_alpha`** : paramètre $\alpha$ du cost complexity pruning.
- **`criterion`** : `squared_error` (par défaut, équivalent au SSR), `friedman_mse`, ou `absolute_error` (analogue robuste, MAE au lieu de MSE).

---
