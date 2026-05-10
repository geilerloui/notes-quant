# Bagging Methods

Les méthodes de bagging partent d'un constat simple : un decision tree (cf [[01_Decision_Trees]]) a un faible biais mais une variance énorme. L'idée — due à Breiman — est de construire beaucoup d'arbres et d'agréger leurs prédictions, ce qui réduit la variance sans toucher au biais. **Random Forest** est l'incarnation la plus connue de cette idée ; pour la suite (boosting), voir [[03_Boosting_Algorithms]].

## Du compromis biais-variance à l'agrégation

Un decision tree complètement développé a un **biais très faible** (il peut représenter n'importe quelle frontière de décision rectangulaire) mais une **variance énorme** : si on change quelques observations du training set, l'arbre construit peut être complètement différent. C'est typique d'un modèle non-paramétrique flexible.

L'élagage (cost complexity pruning) est une première manière de réduire cette variance, mais elle a ses limites : on perd en biais ce qu'on gagne en variance.

L'**idée géniale de Breiman** est différente : plutôt que de réduire la variance d'un seul arbre, on en construit **beaucoup** — chacun entraîné sur un échantillon légèrement différent — puis on **agrège** leurs prédictions (vote majoritaire en classification, moyenne en régression). La variance de la moyenne décroît en $1/M$ (où $M$ est le nombre d'arbres) sans toucher au biais : c'est ce qui rend les méthodes ensemble si efficaces en pratique.

**Bagging** (Bootstrap AGGregatING, Breiman 1996) est le cadre général : prendre un *base learner* (n'importe quel modèle, pas nécessairement un arbre), l'entraîner $M$ fois sur des échantillons bootstrap différents, puis agréger. **Random Forest** (Breiman 2001) est l'application spécifique aux arbres CART, avec en plus une astuce de *random subspace* (cf §A) pour diminuer encore la corrélation entre arbres.

> [!note]- Pourquoi ça marche : variance de la moyenne d'arbres iid
> Supposons d'abord un cas idéal : $M$ arbres $h_1, \ldots, h_M$ entraînés sur des datasets **indépendants** (pas juste des bootstraps du même dataset). On note $\bar h(x_0) = \frac{1}{M} \sum_m h_m(x_0)$ la prédiction moyennée en un point $x_0$.
> 
> **Espérance** (le biais ne change pas) :
> 
> $\mathbb{E}[\bar h(x_0)] = \frac{1}{M} \sum_m \mathbb{E}[h_m(x_0)] = \mathbb{E}[h_1(x_0)]$
> 
> Moyenner $M$ arbres iid donne le **même biais** qu'un seul arbre.
> 
> **Variance** (divisée par $M$) : par indépendance, la variance d'une somme est la somme des variances :
> 
> $\text{Var}(\bar h(x_0)) = \frac{1}{M^2} \sum_m \text{Var}(h_m(x_0)) = \frac{\sigma^2}{M}$
> 
> où $\sigma^2 = \text{Var}(h_1(x_0))$ est la variance d'un arbre individuel. **La variance décroît en $1/M$**, sans toucher au biais. Plus on a d'arbres, plus la prédiction agrégée est stable.
> 
> C'est l'argument pur de l'agrégation : si on pouvait avoir des arbres vraiment indépendants, la variance tendrait vers 0 quand $M \to \infty$, et il ne resterait que le biais (incompressible avec un seul modèle donné). En pratique les arbres sont **corrélés** (entraînés sur des bootstraps du même dataset), et c'est là qu'intervient l'astuce spécifique de Random Forest — voir §A.

## I. Random Forest — Breiman 2001

Random Forest applique l'idée du bagging avec CART comme **base learner**, plus une astuce supplémentaire : à chaque split, on ne considère qu'un **sous-ensemble aléatoire de features**. C'est ce qu'on appelle le *random subspace*, et on va voir pourquoi c'est crucial.

> [!note]- Pourquoi le bagging seul ne suffit pas : variance corrélée
> Reprenons la preuve de l'agrégation, mais cette fois en supposant les arbres **corrélés** (cas réaliste : ils sont entraînés sur des bootstraps du même dataset, donc ils partagent beaucoup d'observations). Soit $\rho$ la corrélation entre deux arbres :
> 
> $\rho = \text{Corr}(h_i(x_0), h_j(x_0)), \quad i \neq j$
> 
> La variance de la moyenne devient (Breiman 2001) :
> 
> $\text{Var}(\bar h(x_0)) = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$
> 
> Deux termes :
> - Le **2nd terme** $\frac{1-\rho}{M}\sigma^2$ tend vers 0 quand $M \to \infty$ — c'est le bénéfice classique de la moyenne.
> - Le **1er terme** $\rho \sigma^2$ **ne dépend pas de $M$** — c'est un plancher irréductible. Plus les arbres sont corrélés, plus ce plancher est haut.
> 
> **Conséquence** : pour réduire vraiment la variance, il ne suffit pas d'ajouter des arbres — il faut aussi **décorréler** les arbres (réduire $\rho$). C'est exactement ce que vise le random subspace ci-dessous.

L'idée du **random subspace** : à chaque split, au lieu de tester toutes les $d$ features, on en tire un sous-ensemble aléatoire de taille $m \ll d$, et on choisit le meilleur split parmi celles-ci uniquement. Le choix typique est $m = \sqrt{d}$ en classification et $m = d/3$ en régression. En forçant les arbres à se passer de leurs features dominantes de temps à autre, on les rend **structurellement différents** — donc moins corrélés.

### A. Construction de la forêt

#### Étape 1 — Bootstrapping

À partir du dataset original $\mathcal{D}$ de taille $n$, on tire $M$ échantillons bootstrap $\mathcal{D}_1, \ldots, \mathcal{D}_M$ : chacun est obtenu en tirant $n$ observations **avec remise** dans $\mathcal{D}$. Du coup chaque bootstrap contient certaines observations en double et **en exclut d'autres** — ces dernières forment le set **out-of-bag** de l'arbre, on s'en servira au §2.

![[Pasted image 20260415223833.png|556]]
*Figure. Bootstrapping : à partir du dataset original, on génère $M$ échantillons en tirant avec remise. Les observations exclues d'un bootstrap forment son OOB.*

#### Étape 2 — Construire un arbre par bootstrap, avec random subspace

Sur chaque bootstrap $\mathcal{D}_m$, on entraîne un arbre CART **non élagué** (volontairement profond pour avoir un faible biais — on compte sur l'agrégation pour réduire la variance). La spécificité Random Forest : à **chaque split** de l'arbre, on tire aléatoirement $m = \sqrt{d}$ features parmi les $d$ disponibles, et on cherche le meilleur split uniquement sur ce sous-ensemble.

![[Pasted image 20260415224923.png|483]]
*Figure. Premier split de l'arbre 1 : sur les 3 features (Fièvre, Fatigue, Toux), on en tire $\sqrt{3} \approx 2$ aléatoirement. Ici Fatigue et Toux ont été tirées, et on choisit le meilleur split entre les deux.*

![[Pasted image 20260415224939.png|509]]
*Figure. Au split suivant, on tire à nouveau 2 features aléatoirement, indépendamment du tirage précédent. Et on itère ainsi à chaque nœud jusqu'à finir l'arbre.*

Une fois les $M$ arbres construits, on obtient une **forêt** :

![[Pasted image 20260415225510.png|474]]
*Figure. La forêt finale : $M$ arbres, chacun entraîné sur un bootstrap différent et avec un random subspace à chaque split.*

#### Étape 3 — Visualisation des plans

**(i) Classification** : la frontière de décision globale de la forêt n'est plus rectangulaire (alignée sur les axes) comme un arbre individuel — c'est l'union pondérée des frontières de tous les arbres, ce qui produit des frontières plus lisses, parfois en "escalier diagonal".

**(ii) Régression** : la surface prédite est la moyenne des $M$ surfaces individuelles.

> 💡 **L'image en 3D.** On peut visualiser ça en reprenant le graphe 3D de la régression du decision tree (cf I.B §3) : c'est comme si on générait $M$ graphes en parallèle — chacun avec ses propres marches — et qu'on faisait la moyenne (ou le vote) point par point. Le résultat est une surface beaucoup plus lisse et plus robuste qu'un seul arbre. Le random subspace renforce cet effet : en forçant les arbres à utiliser des features différentes, leurs marches se positionnent à des endroits différents — donc la moyenne capture mieux la forme sous-jacente.

|        Arbre 1         |        Arbre 2         |        Arbre 3         |
| :--------------------: | :--------------------: | :--------------------: |
| ![[fig_bag_tree1.png]] | ![[fig_bag_tree2.png]] | ![[fig_bag_tree3.png]] |

![[fig_bag_mean.png|325]]
*Figure. Trois arbres CART (max_depth=3) entraînés sur trois bootstraps différents du même dataset 2D. Chacun produit une surface en escalier avec ses propres marches grossières, légèrement différentes d'un arbre à l'autre (variance élevée d'un seul arbre). La **moyenne** des trois (en bas) lisse les marches et produit une surface beaucoup plus expressive : c'est exactement l'effet du bagging — réduction de la variance par agrégation, sans toucher au biais.*

#### Étape 4 — Prédiction finale

**(i) Classification** : chaque arbre vote pour une classe, et la forêt retient la classe **majoritaire** :

$\hat y_{\text{RF}}(x) = \text{mode}\big(h_1(x), \ldots, h_M(x)\big)$

![[Pasted image 20260416094358.png]]
*Figure. Pour une nouvelle observation, on la fait descendre dans chacun des $M$ arbres, on récupère $M$ votes, et la majorité l'emporte.*

**(ii) Régression** : on prend la **moyenne** des $M$ prédictions :

$\hat y_{\text{RF}}(x) = \frac{1}{M} \sum_{m=1}^M h_m(x)$

---

### B. Mesurer l'erreur : Out-of-Bag (OOB)

Car chaque arbre est entraîné sur seulement ~63% des observations (les autres ~37% sont OOB), Random Forest fournit gratuitement un estimateur de l'erreur de généralisation **sans avoir besoin d'un validation set séparé** — c'est un des grands avantages pratiques de la méthode.

> [!note]- Pourquoi ~37% des observations sont OOB par arbre
> Sur un bootstrap de taille $n$ tiré avec remise dans un dataset de taille $n$, la probabilité qu'une observation donnée **ne soit jamais tirée** est :
> 
> $\Big(1 - \frac{1}{n}\Big)^n \xrightarrow[n \to \infty]{} \frac{1}{e} \approx 0.368$
> 
> Donc en moyenne, environ **36.8% des observations sont OOB** pour chaque arbre. Inversement, ~63.2% sont "in-bag".

> [!warning] Définition — Erreur OOB
> Pour chaque observation $i$ du dataset original, on collecte les **arbres où $i$ était OOB** (donc qui n'ont jamais vu $i$ pendant l'entraînement). On agrège leurs prédictions sur $i$ — vote majoritaire en classification, moyenne en régression — pour obtenir une **prédiction OOB** $\hat y_i^{\text{OOB}}$. L'erreur OOB est ensuite :
> 
> $\text{Err}_{\text{OOB}} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\big[\hat y_i^{\text{OOB}} \neq y_i\big] \quad \text{(classification)}$
> 
> $\text{Err}_{\text{OOB}} = \frac{1}{n} \sum_{i=1}^n (\hat y_i^{\text{OOB}} - y_i)^2 \quad \text{(régression)}$

**En pratique**, sur l'observation $n°2$ par exemple (Fièvre=Oui, Fatigue=Oui, Toux=Non, vraie classe = Grippe) :

1. **Collecte** : on identifie les arbres où l'obs 2 était OOB (mettons les arbres 1, 4, 5 sur 5 arbres).
2. **Passage** : on fait descendre l'obs 2 dans ces 3 arbres-là.
3. **Vote local** : on récupère ses 3 prédictions, par exemple 2 "Grippe" + 1 "Sain".
4. **Prédiction OOB** : majorité = Grippe.
5. **Comparaison** : prédiction OOB = vraie classe ? Si oui, bien classée ; sinon, erreur.

On répète pour les $n$ observations, et la fraction d'erreurs donne $\text{Err}_{\text{OOB}}$. Cet estimateur est **non biaisé** (chaque obs n'est prédite que par des arbres qui ne l'ont jamais vue), équivalent moralement à une cross-validation gratuite.

---

### C. Feature importance : MDI vs Permutation

Random Forest fournit deux mesures d'importance des features, qui mesurent des choses différentes et n'ont pas les mêmes biais.

#### MDI — Mean Decrease Impurity (par défaut sklearn)

C'est l'extension directe du MDI vu en §I (decision tree). Pour chaque arbre $m$ et chaque feature $X_j$, on calcule l'importance MDI dans cet arbre (somme des gains pondérés sur les splits qui utilisent $X_j$). Puis on **moyenne sur les $M$ arbres** :

> [!warning] Définition — Importance MDI
> $\text{Imp}_{\text{MDI}}(X_j) = \frac{1}{M} \sum_{m=1}^M \text{Imp}_m(X_j)$
> 
> où $\text{Imp}_m(X_j)$ est l'importance MDI de $X_j$ dans l'arbre $m$ (formule du §I).

**Avantages** : très rapide à calculer (déjà fait pendant l'entraînement), pas de coût supplémentaire.

**Limite connue** : MDI est **biaisée vers les features à beaucoup de modalités** (numériques continues, catégorielles à grand cardinal). Une feature avec plein de seuils possibles offre plus d'opportunités de split, donc accumule plus de gains de manière mécanique — même si elle n'est pas vraiment informative. C'est pourquoi sklearn recommande la **permutation importance** ci-dessous quand on doute.

#### Permutation importance (plus fiable)

Idée intuitive : **si une feature est importante, alors casser ses valeurs (les mélanger aléatoirement) doit faire chuter la performance**. Si la performance ne bouge pas, c'est que la feature ne servait à rien.

L'algorithme :

1. On calcule la performance de référence $\text{Err}_{\text{ref}}$ (typiquement l'erreur OOB ou sur un test set).
2. Pour chaque feature $X_j$ :
   - On **permute aléatoirement** la colonne $X_j$ dans le dataset (on garde les autres colonnes intactes), ce qui détruit le lien entre $X_j$ et $y$ tout en préservant la distribution marginale de $X_j$.
   - On recalcule la performance $\text{Err}_{\text{perm}, j}$ sur ce dataset permuté.
   - L'importance est la **chute de performance** : $\text{Imp}_{\text{perm}}(X_j) = \text{Err}_{\text{perm}, j} - \text{Err}_{\text{ref}}$.
3. Pour réduire le bruit, on répète la permutation $K$ fois ($K = 5$ ou $10$ par défaut) et on moyenne.

> [!warning] Définition — Permutation importance
> $\text{Imp}_{\text{perm}}(X_j) = \mathbb{E}_{\pi}\big[\text{Err}\big(\hat f, \mathcal{D}_{\pi(j)}\big)\big] - \text{Err}\big(\hat f, \mathcal{D}\big)$
> 
> où $\mathcal{D}_{\pi(j)}$ est le dataset avec la colonne $j$ permutée selon une permutation aléatoire $\pi$.

**Avantages** :
- **Pas de biais** lié au type ou cardinal de la feature
- Mesure l'importance **par rapport à la performance** (ce qui nous intéresse vraiment), pas un proxy d'impureté
- Marche pour n'importe quel modèle, pas que Random Forest

**Inconvénient** : coûteux — il faut $K \times d$ passes de prédiction sur le dataset entier.

**En pratique** : commencer par regarder MDI (gratuit), puis confirmer avec Permutation importance sur les top features quand les enjeux sont importants.

---

### D. Hyperparamètres principaux

Les hyperparamètres d'un `RandomForestClassifier` / `RandomForestRegressor` (sklearn) se rangent en deux familles :

#### (i) Hérités du Decision Tree (s'appliquent à chaque arbre individuel)

Ce sont les mêmes que ceux vus en §I — ils contrôlent la structure de chaque arbre dans la forêt :

- **`max_depth`** : profondeur maximale des arbres. En RF, on le laisse souvent à `None` (arbres complets, faible biais) car l'agrégation gère la variance — au contraire d'un decision tree seul où il faut souvent élaguer.
- **`min_samples_split`** : nombre minimum d'observations dans un nœud pour le splitter.
- **`min_samples_leaf`** : nombre minimum d'observations par feuille. Sert à régulariser légèrement chaque arbre.
- **`min_impurity_decrease`** : gain minimum pour autoriser un split.
- **`criterion`** : `gini`/`entropy` en classification, `squared_error`/`absolute_error` en régression.

#### (ii) Spécifiques au Random Forest (contrôlent l'agrégation)

Ces hyperparamètres n'existent que parce qu'on a une **forêt** — ils gouvernent comment on construit et combine les arbres :

- **`n_estimators`** : nombre d'arbres $M$ dans la forêt. Plus il y en a, mieux c'est, jusqu'à un plateau de stabilité (typiquement 200-500 suffit). Coût linéaire en $M$.
- **`max_features`** : taille du random subspace $m$ à chaque split. Défaut : `sqrt` ($\sqrt{d}$) en classification, `1.0` (toutes les features) en régression dans sklearn récent — mais $d/3$ historiquement recommandé en régression. C'est le bouton qui contrôle le trade-off **décorrélation vs force** (cf le callout sur $\rho\sigma^2 + (1-\rho)\sigma^2/M$).
- **`bootstrap`** : si `False`, désactive le bootstrap (chaque arbre voit le dataset complet). Mettre `True` (défaut) pour avoir le vrai Random Forest et l'OOB.
- **`oob_score`** : si `True`, calcule automatiquement l'erreur OOB pendant l'entraînement (gratuit, cf §2).
