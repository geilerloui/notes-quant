---
title: Fondations - Supervised Learning
---
# Fondations du Supervised Learning

> Cette note pose le **cadre théorique** du supervised learning — ce qu'on cherche, pourquoi, et avec quels garde-fous — puis enchaîne sur la **méthodologie** d'un projet ML concret. C'est le socle sur lequel s'appuient ensuite tous les algorithmes spécifiques (régression linéaire, arbres, NN, etc.).

## I. Cadre théorique

### A. La vraie fonction $f$

**Le point de départ.** Tout le supervised learning repose sur une seule idée : il existe une fonction $f$ inconnue qui lie les entrées aux sorties, et tout l'enjeu est de **construire une approximation $\hat f$ de $f$ à partir d'un échantillon fini**. Toute la machinerie statistique qui suit (EPE, ERM, biais-variance, CV...) ne fait qu'opérationnaliser cette idée.

$$\boxed{\text{Le supervised learning} \;=\; \text{Approximation de la fonction inconnue } f}$$

![[Pasted image 20260425184115.png|433]]
**Figure 1.** La vraie fonction $f$ (en bleu) est inconnue. On observe seulement les points $(x_i, y_i)$, et $\varepsilon_i = y_i - f(x_i)$ est l'écart vertical entre l'observation et la courbe.

> [!warning] Définition formelle
> Soit $Y$ une réponse quantitative et $p$ prédicteurs $X = (X_1, \ldots, X_p)$. On suppose
> 
> $$Y = f(X) + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0, \quad \varepsilon \perp X.$$
> 
> - $f$ : fonction fixée mais **inconnue**, déterministe en $X$.
> - $\varepsilon$ : terme d'erreur aléatoire qui capture ce que $X$ ne peut pas expliquer.

> [!example] Fil rouge — prédire le prix d'un appartement
> Pour ancrer chaque concept, on utilisera tout au long de cette note l'exemple de la **prédiction du prix d'un appartement** à partir de quelques features simples : surface ($X_1$, en m²), nombre de pièces ($X_2$), arrondissement ($X_3$). La cible $Y$ est le prix de vente, en €. Les chiffres seront volontairement simples pour rester lisibles, pas calibrés sur des données réelles.
> 
> La vraie fonction $f$ qui lie ces variables au prix est inconnue — elle dépend de mille facteurs qu'on ne peut ni mesurer ni écrire explicitement. Tout ce qu'on aura, c'est un dataset fini de transactions passées.

**Que représente le $\varepsilon$ ?** On le décompose en **deux sources** distinctes, qui ont des conséquences différentes :

1. **Variables non mesurées mais explicatives.** Aucun modèle ne peut les utiliser puisqu'elles ne sont pas dans $X$. *Pour notre fil rouge : la luminosité de l'appartement, l'état de la cuisine, la qualité de la vue — toutes ces variables affectent le prix mais ne sont pas dans nos features.*
2. **Variabilité intrinsèque** du phénomène. Bruit irréductible même avec un modèle parfait. *Deux appartements identiques sur le papier ne se vendent jamais exactement au même prix : aléa du marché, négociation, urgence du vendeur.*

> 💡 **$\varepsilon$ est une hypothèse sur le monde, pas sur le modèle.** Que tu utilises une régression linéaire ou un random forest, ce bruit reste dans les données. Aucune sophistication algorithmique ne peut le faire disparaître — c'est précisément la borne inférieure que rencontrera tout modèle, quel qu'il soit. On verra en I.D que cette borne porte un nom : l'**erreur irréductible**.

---
### B. EPE et espérance conditionnelle

**Le besoin.** On a posé l'existence de $f$, mais pour la définir précisément, il faut un **critère**. Quelle fonction $f^*$ veut-on viser exactement ? La réponse vient de la théorie de la décision statistique : on se donne une fonction de coût $L(Y, f(X))$ qui mesure l'écart entre la vraie valeur $Y$ et la prédiction $f(X)$, et on définit la fonction optimale comme celle qui minimise l'erreur en espérance.

> [!warning] Expected Prediction Error (EPE)
> $$f^* \;=\; \arg\min_f \; \mathbb{E}\big[L(Y, f(X))\big]$$
> 
> C'est un cadre **général** : à chaque choix de loss correspond une fonction optimale différente. On va voir trois cas — MSE, MAE, 0/1 — qui couvrent l'essentiel de ce qu'on rencontre en pratique.

> 💡 **L'hypothèse cachée.** Tout ce cadre suppose qu'on connaît la distribution jointe $P(X, Y)$. C'est purement théorique : en pratique on ne connaît jamais $P$, on a juste un échantillon fini. C'est exactement ce qu'on contournera en I.C avec l'**ERM** — remplacer l'espérance théorique par sa moyenne empirique.

#### B.1 MSE → espérance conditionnelle

On choisit la **MSE** (Mean Squared Error) comme fonction de coût :

$$L(Y, f(X)) = (Y - f(X))^2.$$

L'EPE devient $\text{EPE}(f) = \mathbb{E}\big[(Y - f(X))^2\big]$, et on minimise sur **toutes** les fonctions $f$ mesurables (cadre général, sans restriction de classe). La solution est **l'espérance conditionnelle** :

> [!warning] Fonction de régression
> $$\boxed{f^*(x) \;=\; \mathbb{E}[Y \mid X = x]}$$
> 
> aussi appelée **fonction de régression** de $Y$ sur $X$.

> [!note]- Dérivation
> On utilise la propriété de tour de l'espérance :
> 
> $$\text{EPE}(f) = \mathbb{E}\big[(Y - f(X))^2\big] = \mathbb{E}_X\Big[\mathbb{E}_{Y \mid X}\big[(Y - f(X))^2 \mid X\big]\Big].$$
> 
> Comme l'intégrande est positive, minimiser l'intégrale globale revient à minimiser **point par point** pour chaque $x$ :
> 
> $$f^*(x) = \arg\min_c \; \mathbb{E}\big[(Y - c)^2 \mid X = x\big].$$
> 
> C'est un problème scalaire classique : on dérive par rapport à $c$ et on annule, ce qui donne $c^* = \mathbb{E}[Y \mid X = x]$.

**Lien avec I.A.** En I.A on a posé $Y = f(X) + \varepsilon$ avec $\mathbb{E}[\varepsilon] = 0$ et $\varepsilon \perp X$. Sous ces hypothèses,

$$\mathbb{E}[Y \mid X] = f(X) + \mathbb{E}[\varepsilon \mid X] = f(X).$$

Le $f$ qu'on a posé au départ est donc **exactement** l'espérance conditionnelle. Les deux objets coïncident — c'est ce qui rend la MSE si naturelle pour la régression.

> [!example] Fil rouge — l'espérance conditionnelle pour les appartements
> $f^*(x) = \mathbb{E}[Y \mid X = x]$ se lit : *"pour un appartement de surface 50 m², 2 pièces, dans le 11ème, le prix optimal à prédire est la **moyenne** des prix de vente de tous les appartements ayant exactement ces caractéristiques"*. C'est intuitif : si plusieurs appartements identiques se sont vendus à 380k, 410k, 395k, la meilleure prédiction (au sens MSE) pour le prochain est leur moyenne, ~395k.

**Vue géométrique (spécifique à la MSE).**

> 💡 Cette vue géométrique vient du fait que la MSE est la norme $L^2$, ce qui donne au problème une structure d'**espace de Hilbert** et une interprétation en termes de projection orthogonale. Elle ne se généralise pas aux autres loss.

On travaille dans $L^2$, muni du produit scalaire $\langle X, Y \rangle = \mathbb{E}[XY]$. Dans cet espace, **les vecteurs sont des variables aléatoires**.

L'idée centrale : $\mathbb{E}(Y \mid X)$ est la **projection orthogonale** de $Y$ sur le sous-espace $L^2_X$ des fonctions de $X$. Cela donne la décomposition :

$$Y = \mathbb{E}(Y \mid X) + \varepsilon.$$

Le résidu $\varepsilon = Y - \mathbb{E}(Y \mid X)$ est **orthogonal à $L^2_X$**, ce qui signifie $\mathbb{E}[\varepsilon \cdot f(X)] = 0$ pour toute fonction $f$. Deux conséquences immédiates :

- $\mathbb{E}(\varepsilon) = 0$ — en prenant $f = 1$.
- $\text{cov}(\varepsilon, X) = 0$ — en prenant $f = X$.

Ces propriétés ne sont pas des hypothèses qu'on impose : elles **découlent directement de la géométrie**.

![Interprétation géométrique dans L²|440](images/2-Statistiques/A_Frequentist/regression-lineaire/im2.png)
**Figure 2.** Interprétation géométrique dans $L^2$. $\mathbb{E}(Y \mid X)$ est la projection orthogonale de $Y$ sur le sous-espace $L^2_X$ des fonctions (mesurables) de $X$. Le résidu $\varepsilon = Y - \mathbb{E}(Y \mid X)$ est orthogonal à tout le sous-espace $L^2_X$.

#### B.2 MAE → médiane conditionnelle

Si on remplace la MSE par la **MAE** (Mean Absolute Error) :

$$L(Y, f(X)) = |Y - f(X)|, \qquad \text{EPE}(f) = \mathbb{E}\big[|Y - f(X)|\big],$$

le minimiseur devient la **médiane conditionnelle** :

$$\boxed{f^*(x) \;=\; \text{median}(Y \mid X = x)}$$

> 💡 **Pourquoi MAE plutôt que MSE ?** La médiane est **robuste aux outliers** : un appartement vendu à un prix aberrant (héritage bradé, vente forcée) tire violemment la moyenne mais peu la médiane. En présence d'outliers, MAE donne une fonction-cible plus représentative du marché "normal".

#### B.3 0/1 loss → classifieur de Bayes

En classification, $Y$ prend des valeurs dans $\{1, \ldots, K\}$. La loss naturelle est la **0/1 loss** :

$$L(Y, f(X)) = \mathbf{1}\{Y \neq f(X)\}, \qquad \text{EPE}(f) = P(Y \neq f(X)).$$

Le minimiseur est le **classifieur de Bayes** : on prédit la classe la plus probable conditionnellement à $X$.

> [!warning] Classifieur de Bayes
> $$\boxed{f^*(x) \;=\; \arg\max_{k \in \{1, \ldots, K\}} P(Y = k \mid X = x)}$$
> 
> **Erreur de Bayes.** Le taux d'erreur du classifieur de Bayes,
> 
> $$\mathbb{E}_X\big[1 - \max_k P(Y = k \mid X)\big],$$
> 
> est l'**erreur irréductible** en classification — l'analogue de $\sigma^2$ en régression. Aucun classifieur ne peut faire mieux.

#### B.4 Tableau récapitulatif

Le minimiseur de l'EPE dépend de la loss. Tous les algorithmes qu'on verra plus tard sont en fait des stratégies pour **estimer** un de ces minimiseurs à partir d'un échantillon fini.

| Loss | Cadre | Minimiseur |
|---|---|---|
| $L_2$ : $(Y - f(X))^2$ | Régression | $f^*(x) = \mathbb{E}[Y \mid X=x]$ |
| $L_1$ : $\lvert Y - f(X) \rvert$ | Régression robuste | $f^*(x) = \text{median}(Y \mid X=x)$ |
| 0/1 : $\mathbf{1}\{Y \neq f(X)\}$ | Classification | $f^*(x) = \arg\max_k P(Y=k \mid X=x)$ — **classifieur de Bayes** |

---

### C. De $f^*$ à $\hat f$ : Empirical Risk Minimization

**Le problème.** En I.B, $f^*$ minimise l'EPE — mais l'EPE est une espérance sur la vraie distribution $P(X, Y)$ qu'on ne connaît pas. En pratique, on a juste un dataset fini

$$\mathcal{T} = \{(x_1, y_1), \ldots, (x_n, y_n)\}.$$

**L'idée unique.** On remplace l'espérance théorique par sa **moyenne empirique** sur $\mathcal{T}$. C'est l'**Empirical Risk Minimization (ERM)** — l'opérationnalisation pratique de l'EPE.

> [!warning] Empirical Risk Minimization
> Le **risque empirique** est
> 
> $$\hat R(f) \;=\; \frac{1}{n} \sum_{i=1}^n L(y_i, f(x_i)).$$
> 
> C'est un estimateur de l'EPE par la loi des grands nombres (sous l'hypothèse i.i.d.). On définit alors
> 
> $$\hat f \;=\; \arg\min_{f \in \mathcal{F}} \hat R(f).$$

#### C.1 Le choix de la classe $\mathcal{F}$

On ne minimise pas sur **toutes** les fonctions — sinon $\hat f$ interpolerait les $n$ points et serait inutile en généralisation (overfitting extrême). On restreint à une classe $\mathcal{F}$. Deux grandes familles :

- **Paramétrique** : $\mathcal{F} = \{f_\theta : \theta \in \Theta\}$. On réduit l'estimation de fonction à l'estimation d'un vecteur de paramètres $\theta$. Ex : régression linéaire ($\theta = \beta$), réseaux de neurones ($\theta$ = poids).
- **Non-paramétrique** : pas de paramètres finis à estimer, on utilise directement la structure locale des données. Ex : KNN, kernel methods, arbres.

> 💡 **Le compromis.** Plus $\mathcal{F}$ est riche, plus on peut approcher $f^*$ — mais plus on risque l'overfitting. La taille de $\mathcal{F}$ est exactement le levier qu'on règle pour naviguer le trade-off **biais-variance** (I.D).

#### C.2 Cadre concret par tâche

**Régression.** Avec la loss MSE $L(y, \hat y) = (y - \hat y)^2$, le risque empirique devient le **RSS** (Residual Sum of Squares) :

$$\text{RSS}(f) \;=\; \sum_{i=1}^n (y_i - f(x_i))^2.$$

![[Pasted image 20260425194646.png|279]]
**Figure 3.** Cas particulier où la vraie fonction de régression est affine : $\mathbb{E}[Y \mid X = x] = a^* + b^* x$.

> [!example] Fil rouge — la régression linéaire en pratique
> On choisit la MSE comme loss et la classe affine $\mathcal{F} = \{f(x) = a + bx\}$ pour notre problème de prix d'appartement, avec $X =$ surface en m². L'ERM s'écrit
> 
> $$\hat f(x) = \hat a + \hat b \cdot x, \qquad (\hat a, \hat b) \;=\; \arg\min_{a, b} \sum_{i=1}^n (y_i - a - b x_i)^2.$$
> 
> Sur un échantillon parisien typique, on pourrait obtenir par exemple $\hat a \approx 50\,000$ et $\hat b \approx 9\,500$ — le prix prédit pour 50 m² serait alors $50\,000 + 9\,500 \times 50 = 525\,000$ €. Cet $\hat f$ est l'**estimateur** ERM de la vraie droite de population $f^*_{\text{lin}}(x) = a^* + b^* x$, qu'on ne connaîtra jamais exactement.

**Classification.** Avec la loss 0/1, le risque empirique devient le **taux d'erreur empirique** :

$$\hat R(f) \;=\; \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{y_i \neq f(x_i)\}.$$

On a vu en I.B.3 que le minimiseur théorique est le **classifieur de Bayes**. En pratique, on ne connaît pas $P(Y = k \mid X)$, donc on doit l'estimer. Car pour rappel on a vu que $f^*(x) \;=\; \arg\max_{k \in \{1, \ldots, K\}} P(Y = k \mid X = x)$. Il existe **trois grandes stratégies** (Bishop, PRML §1.5.4) qui structurent toute la zoologie des modèles de classification :

| Approche | Stratégie | Exemples |
|---|---|---|
| **Génératifs** | Estimer $\hat P(X \mid Y=k)$ et $\hat P(Y=k)$, recombiner par Bayes : $\hat P(Y=k \mid X) \propto \hat P(X \mid Y=k) \hat P(Y=k)$ | Naive Bayes, LDA, QDA |
| **Discriminatifs** | Estimer $\hat P(Y=k \mid X)$ directement, sans passer par $\hat P(X \mid Y=k)$ | Régression logistique, MLP avec sigmoïde/softmax (cf [[00_Perceptron Multi-Couches]]) |
| **Discriminant function** | Apprendre directement $\hat f : \mathcal{X} \to \{1, \ldots, K\}$, sans probabilités | Arbres (cf [[(i) Modèles d'Arbres]]), perceptron strict, SVM |

> 💡 **Pourquoi cette distinction est importante.** Selon ce qu'on choisit de modéliser, on obtient des familles très différentes de méthodes. Les **génératifs** sont plus riches (on peut générer des données synthétiques) mais souvent moins précis pour la prédiction. Les **discriminatifs** sont focalisés sur la tâche de prédiction et généralement plus performants. Les **discriminant functions** sont les plus directs mais perdent l'information probabiliste.

---

### D. Décomposition biais-variance et double descent

#### D.1 Décomposition biais-variance

**Le contexte.** On dispose d'un training set $\mathcal{T} = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ pour estimer $\hat f$, et on évalue la qualité de l'estimation sur un point de test $(x_0, y_0)$ qu'on n'a pas vu.

> 💡 **Précision théorique cruciale.** $\hat f$ dépend du dataset $\mathcal{T}$, qui est lui-même aléatoire (tiré de la vraie distribution). Donc $\hat f$ est une variable aléatoire indexée par $\mathcal{T}$. L'EPE en un point $x_0$ moyenne donc sur **deux sources d'aléa** : la nouvelle observation $Y_0$ **et** le tirage de $\mathcal{T}$. C'est cette double source d'aléa qui va se factoriser en biais et variance.

> [!warning] EPE en un point + erreur réductible/irréductible
> $$\text{EPE}(x_0) \;=\; \mathbb{E}_{\mathcal{T}, Y_0}\big[(Y_0 - \hat f_{\mathcal{T}}(x_0))^2 \mid X_0 = x_0\big] \;=\; \underbrace{\mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big]}_{\text{réductible}} + \underbrace{\text{Var}(\varepsilon)}_{\text{irréductible}}$$
> 
> - **Réductible** : on peut diminuer l'écart entre la vraie fonction et notre estimation en changeant de méthode, en ayant plus de données, etc.
> - **Irréductible** : le $\sigma^2$ du $\varepsilon$ — limite ultime, aucun modèle ne peut faire mieux.

> [!note]- Dérivation
> On part de $Y_0 = f(x_0) + \varepsilon$ avec $\varepsilon \perp \hat f$ (le bruit du test est indépendant du training set) et $\mathbb{E}[\varepsilon] = 0$. On développe :
> 
> $$(Y_0 - \hat f(x_0))^2 = \big((f(x_0) - \hat f(x_0)) + \varepsilon\big)^2.$$
> 
> En prenant l'espérance et en utilisant $\mathbb{E}[\varepsilon] = 0$ et l'indépendance, le terme croisé s'annule :
> 
> $$\mathbb{E}\big[(Y_0 - \hat f(x_0))^2\big] = \mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big] + \mathbb{E}[\varepsilon^2] = \mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big] + \text{Var}(\varepsilon).$$

> [!warning] Décomposition biais-variance
> En ajoutant et soustrayant $\mathbb{E}[\hat f(x_0)]$ dans la partie réductible :
> 
> $$\boxed{\text{EPE}(x_0) \;=\; \underbrace{\big(f(x_0) - \mathbb{E}[\hat f(x_0)]\big)^2}_{\text{biais}^2} + \underbrace{\text{Var}(\hat f(x_0))}_{\text{variance}} + \text{Var}(\varepsilon)}$$
> 
> - **Biais** : erreur systématique. *En moyenne sur tous les datasets possibles, est-ce que $\hat f$ vise juste ?* Un modèle linéaire sur une vraie relation quadratique aura un biais structurel non nul, peu importe la quantité de données.
> - **Variance** : sensibilité aux données. *Si je ré-entraîne sur un autre dataset tiré de la même distribution, est-ce que $\hat f(x_0)$ change beaucoup ?*


![[Pasted image 20260425192714.png|405]]




> [!note]- Dérivation
> On note $\mu = \mathbb{E}[\hat f(x_0)]$ et on écrit :
> 
> $$f(x_0) - \hat f(x_0) = \underbrace{(f(x_0) - \mu)}_{\text{constante}} + \underbrace{(\mu - \hat f(x_0))}_{\text{v.a., moyenne 0}}.$$
> 
> En développant le carré et en prenant l'espérance, le terme croisé s'annule (la constante sort de l'espérance, multipliée par une v.a. de moyenne 0) :
> 
> $$\mathbb{E}\big[(f(x_0) - \hat f(x_0))^2\big] = (f(x_0) - \mu)^2 + \mathbb{E}\big[(\mu - \hat f(x_0))^2\big] = \text{biais}^2 + \text{Var}(\hat f(x_0)).$$

> [!warning] Overfitting & Underfitting
> - Modèle trop simple → **fort biais** → **underfitting** : $\text{MSE}^{\text{train}}$ et $\text{MSE}^{\text{test}}$ tous deux élevés.
> - Modèle trop complexe → **forte variance** → **overfitting** : $\text{MSE}^{\text{train}}$ très faible mais $\text{MSE}^{\text{test}} \gg \text{MSE}^{\text{train}}$ (le modèle mémorise les fluctuations du train).

> [!warning] Généralisation
> La **généralisation** d'un modèle, c'est sa capacité à bien performer sur des données **jamais vues**, tirées de la même distribution que le train. Formellement, l'**erreur de généralisation** (*generalization error*) est le test error quand le test set tend vers l'infini — c'est exactement l'**EPE** définie en I.B.
>
> $\text{generalization error} \;=\; \lim_{n_{\text{test}} \to \infty} \text{test error} \;=\; \text{EPE}$
>
> Un modèle qui **overfit** a une excellente performance sur le train mais une mauvaise généralisation — il a mémorisé le bruit du train plutôt que d'apprendre la vraie structure de $f$. À l'inverse, un modèle qui **underfit** généralise mal aussi, mais pour une raison différente : il n'a jamais capturé $f$, même sur le train. Toute la section II (cross-validation, learning curves) sert à **estimer** cette erreur de généralisation sans disposer d'un jeu de test infini.

![[Pasted image 20260425191515.png]]
**Figure 4.** Cas $f$ non linéaire. (a) La vraie fonction (noir) est une sinusoïde tronquée. (b) On fait varier les degrés de liberté du modèle (splines) en abscisse, du moins flexible (gauche) au plus flexible (droite). Le **train MSE** (gris) ne fait que chuter. Le **test MSE** (rouge) est en forme de U.

> [!example] Cas 1 — $f$ non linéaire (générique)
> La courbe en U du test MSE de la Figure 4 se décompose en deux régimes :
> - **À gauche** : modèle trop simple → **biais² élevé** (régression linéaire qui ne peut pas capturer la sinusoïde).
> - **À droite** : modèle trop complexe → **variance élevée** (splines qui collent aux fluctuations du train).
> 
> La somme atteint son minimum quelque part au milieu, mais reste **bornée par en bas par $\text{Var}(\varepsilon)$**. C'est le scénario générique — la majorité des problèmes ML réels.

![[Pasted image 20260425215218.png]]
**Figure 5.** Cas $f$ linéaire. Pas de forme en U pour le test : le biais reste à zéro pour toute classe contenant les fonctions linéaires, et la variance ne fait qu'augmenter avec la complexité.

> [!example] Cas 2 — $f$ linéaire
> Si la vraie fonction est elle-même affine (Figure 5), toute classe $\mathcal{F}$ contenant les fonctions affines a un **biais nul** : $\mathbb{E}[\hat f(x_0)] = f(x_0)$ pour tout $x_0$. Reste la variance, qui ne fait que croître avec la flexibilité du modèle. **Pas de trade-off ici** — le minimum du test MSE est atteint pour le modèle le plus simple capable de capturer $f$ (la régression linéaire elle-même), et toute complexité supplémentaire ne fait qu'augmenter la variance sans rien améliorer.

> [!example] Fil rouge — trois modèles pour le prix d'appartement
> Pour fixer les idées, comparons trois choix de classe $\mathcal{F}$ sur notre problème :
> 
> | Modèle | Biais | Variance | Régime |
> | :--- | :--- | :--- | :--- |
> | Constante $\hat f(x) = \bar y$ | Très élevé (ignore $x$) | Très faible | Underfitting |
> | Régression linéaire $\hat f(x) = \hat a + \hat b x$ | Modéré (la vraie relation surface→prix n'est sans doute pas exactement linéaire) | Faible | Souvent un bon point de départ |
> | KNN avec $K=1$ | Nul (passe par tous les points) | Énorme (un seul outlier change tout) | Overfitting catastrophique |
> 
> Le bon choix se situe entre ces extrêmes — typiquement régression polynomiale modérée, random forest, ou gradient boosting. C'est précisément ce qu'on règlera en pratique avec la **cross-validation** (II.B).

#### D.2 Le double descent

**Le contexte.** Le cadre biais-variance qu'on vient de voir suppose qu'on travaille en régime **underparameterized** ($p < n$, moins de paramètres que d'observations). Quand on dépasse l'**interpolation threshold** ($p = n$, le modèle fit parfaitement les données d'entraînement, $\text{MSE}^{\text{train}} = 0$), il se passe quelque chose qui n'est pas prédit par la courbe en U classique.

![[Pasted image 20260425194227.png|443]]
**Figure 6.** Double descent. En abscisse, le ratio $p/n$ (paramètres / observations). À gauche, le régime classique underparameterized : courbe en U du test error (biais-variance). À l'interpolation threshold ($p/n = 1$), pic du test error. À droite, le régime overparameterized ($p \gg n$) : le test error redescend et peut passer en dessous du minimum du régime classique. L'asymptote inférieure reste $\sigma^2$.

> [!warning] Trois régimes
> - **Underparameterized** ($p < n$) : courbe en U classique. Plus de paramètres → biais ↓, variance ↑.
> - **At the interpolation threshold** ($p \approx n$) : il existe **exactement une** solution qui interpole les données. Cette solution est forcée et fragile — petit changement dans le train → grand changement dans $\hat f$. Variance explose, **pic du test error**.
> - **Overparameterized** ($p > n$) : il existe **une infinité** de solutions qui interpolent. La descente de gradient en sélectionne automatiquement une particulière — la solution de **norme minimale**, plus lisse et plus régulière. Le test error redescend.

**Régularisation implicite.** Le mécanisme central : **pas besoin d'ajouter un $\lambda \|w\|^2$ explicite**, l'algorithme d'optimisation lui-même régularise en sélectionnant la solution la plus simple parmi celles qui interpolent. C'est cette régularisation implicite qui rend les modèles overparameterized étonnamment performants.

> 💡 **Pas spécifique aux NN.** Le double descent est démontré analytiquement sur la régression linéaire (Belkin et al. 2019) et observable sur les random features, les random forests, le boosting. Mais c'est sur les **réseaux de neurones profonds** modernes qu'il est le plus spectaculaire — ces modèles vivent par défaut dans le régime $p \gg n$.

> 💡 **Le biais-variance trade-off est-il mort ?** Non. Le cadre classique reste valide dans son régime (underparameterized). Le double descent montre simplement qu'**au-delà de l'interpolation threshold**, un autre régime prend le relais où l'intuition "plus complexe = pire" devient fausse. La pratique moderne (LLM, ResNet, ViT) vit dans ce second régime — d'où l'importance de connaître les deux.

---

### E. Curse of dimensionality

> Cette section part d'un fait géométrique pur — l'espace se vide exponentiellement vite quand $d$ monte — et montre pourquoi ça casse le kNN en régression, puis comment on s'en sort.

**Le setup.** On place $N$ points uniformément dans le cube unité $[0,1]^d$. L'uniformité est volontaire : c'est le **pire cas**, celui sans aucune structure à exploiter. Toute la malédiction qu'on va décrire s'applique dans ce monde sans hypothèse. En pratique les données ne sont jamais vraiment uniformes — c'est précisément ce qui sauve le ML réel, et on y revient en I.F avec la Manifold Hypothesis.

**Manifestation 1 — le voisinage n'est plus local.** On veut faire du 1-NN en régression : pour prédire en $x_0$, on cherche le point d'entraînement le plus proche et on retourne sa valeur $Y$. Pour que cette moyenne locale ait un sens, il faut que les voisins soient *proches*. Posons la question : pour capturer une fraction $r$ des $N$ points autour de $x_0$, quelle est la taille du sous-cube nécessaire ?

Comme les points sont uniformes, capturer une fraction $r$ du volume suffit. Un sous-cube d'arête $e$ a un volume $e^d$. On veut $e^d = r$, soit :

$$\boxed{e_d(r) = r^{1/d}}$$

> [!warning] La malédiction en chiffres
> | dimension $d$ | arête pour $r=1\%$ | arête pour $r=10\%$ |
> |---|---|---|
> | 1 | 0.01 | 0.10 |
> | 3 | 0.22 | 0.46 |
> | 10 | **0.63** | **0.80** |
> | 100 | 0.95 | 0.98 |
> 
> En 10 dimensions, pour capturer 1% des points, le sous-cube couvre déjà 63% de l'étendue de chaque axe. Le voisinage n'est plus local.

Le double piège : réduire $r$ pour rester local → peu de points → variance explose. Augmenter $r$ pour avoir assez de points → voisinage non-local → biais. **En haute dimension, il n'existe plus de réglage qui donne à la fois localité et assez d'échantillons.**

![[cube_ball_basics.png|627]]
**Figure.** *Placeholder — sous-cube qui capture $r=10\%$ des points en $d=2$ (arête 0.32) vs $d=3$ (arête 0.46), avec la courbe $e_d(r)$ en fonction de $d$.*

**Manifestation 2 — les points fuient vers les bords.** Dans ce même cube, on inscrit la boule de rayon 0.5 centrée en $(0.5, \ldots, 0.5)$. Cette boule touche le cube au milieu de chaque face — distance constante 0.5 du centre. Mais la distance du centre à un coin vaut :

$\text{distance centre} \to \text{coin} = \sqrt{d \cdot 0.5^2} = \frac{\sqrt{d}}{2}$

En $d=2$ : 0.71. En $d=3$ : 0.87. En $d=10$ : 1.58. Le cube se transforme en oursin — ses pointes ($2^d$ coins) s'éloignent pendant que la boule reste figée. Tout le volume migre vers les coins.

Conséquence concrète : un point tiré uniformément dans $[0,1]^d$ a une distance au centre qui vaut en espérance $\sqrt{d/12}$. En $d=3$ c'est pile 0.5 (le bord de la boule). Dès $d > 3$, **le point typique est hors de la boule** — et par concentration (somme de $d$ variables indépendantes), quasi tous les points sont à la même distance $\approx \sqrt{d/12}$ du centre. La fraction de points dans la boule inscrite tombe à 0 dès $d \approx 10$.

![[Pasted image 20260617211930.png|428]]
**Figure.** *Placeholder — fraction des points dans la boule inscrite en fonction de $d$ (courbe empirique), montrant la chute vers 0 dès $d \sim 10$.*

le truc des oursins la
![[Pasted image 20260617212210.png]]

**Conséquence sur la prédiction — l'exemple de la cloche.** Pour voir ce que ça coûte en erreur, on pose un cas concret (ESL §2.5) : $N=1000$ points tirés uniformément dans $[-1,1]^d$, vraie relation $f(X) = e^{-8\|X\|^2}$ (cloche gaussienne valant 1 au centre, décroissant vite vers 0), sans bruit. On veut prédire en $x_0 = 0$ par 1-NN.

La prédiction est $\hat y_0 = f(x_{\text{NN}}) = e^{-8\|x_{\text{NN}}\|^2}$. Si le NN est à distance $r$ de l'origine, la prédiction vaut $e^{-8r^2}$. On vient de montrer que $r$ croît avec $d$ — donc la prédiction descend vers 0, alors que la vraie valeur est $f(0) = 1$.

=> la c'étiat très mal expliqué: faut quand même préciser que notre échantillon de test est à la localisation $x_0=0$ et qu'on va chercher un point dans T (training set) le plus proche. Plus la dimension est grande et plus ce point sera dans les épines comme sur l'image en haut et donc la distance va exploser ce qui va faire qu'on va prédire 0 au lieu de 1 c'est donc totalement biaisé.

On décompose l'erreur en biais et variance :

$\text{MSE}(x_0) = \underbrace{\left(f(0) - \mathbb{E}[\hat y_0]\right)^2}_{\text{biais}^2} + \underbrace{\text{Var}(\hat y_0)}_{\approx 0}$

> [!warning] Ce qu'on observe
> - **Biais** : domine et explose. Le NN s'éloigne de l'origine → $f(x_{\text{NN}}) \to 0$ → sous-estimation systématique. En $d=10$, plus de 99% des tirages ont leur NN à distance $> 0.5$, où $f \approx 0.13$. Le biais plafonne à 1 ($= f(0) - 0$), la MSE aussi.
> - **Variance** : quasi nulle partout. Les données sont uniformes donc peu importe le tirage de $\mathcal{T}$, le NN tombe toujours à peu près à la même distance. La prédiction est stable — mais stablement fausse.

C'est le visage de la malédiction en régression : **le 1-NN meurt par biais, pas par variance.**



![[Pasted image 20260617211745.png|291]]![[Pasted image 20260617211819.png|268]]
**Figure.** *Placeholder — courbes MSE, biais² et variance en fonction de $d$ pour cet exemple. MSE et biais² montent et plafonnent à 1, variance reste quasi nulle.*

**Échappatoire — le modèle linéaire.** Le kNN fait de la moyenne locale, et c'est exactement là que la malédiction frappe. Le modèle linéaire $Y = X^T\beta + \varepsilon$ (OLS) échappe à ce piège : il utilise **tous** les points du training set pour estimer $\beta$, sans notion de voisinage. En moyennant sur $x_0$ avec $N$ grand :

$\mathbb{E}_{x_0}[\text{EPE}(x_0)] = \sigma^2 + \sigma^2 \frac{p}{N}$

L'EPE croît **linéairement** en $p$, pas exponentiellement. La malédiction est tuée — au prix d'une hypothèse forte : si la vraie $f$ n'est pas linéaire, le biais structurel peut être énorme. C'est le tradeoff fondamental : **localité sans hypothèse (kNN) vs structure imposée (linéaire)**.

**Borne formelle — Lipschitz et le corollaire de Mallat.** On peut se demander si la malédiction est évitable avec un meilleur estimateur. La réponse est non, sous hypothèse Lipschitz. Si $f$ est uniformément Lipschitz de constante $C$ sur $\Omega$ compact, alors pour **tout** estimateur $\tilde f$ :

$\sup_{f \in \mathcal{F}} \|f - \tilde f\|_\infty \geq C \frac{\sqrt{d}}{2} \sqrt{\frac{2}{\pi e}} \left(1 + O\left(\frac{\log d}{d}\right)\right) \cdot n^{-1/d}$

Pour atteindre une erreur $C\epsilon$, il faut donc :

$n \geq \frac{\epsilon^{-d}\, d^{d/2}}{(2\pi e)^{d/2}}$

C'est une **borne minimax** : pas "un mauvais modèle se plante", mais "le meilleur modèle possible se plante au moins autant". L'explosion exponentielle $\epsilon^{-d}$ est inévitable sous cette seule hypothèse de régularité. La sortie n'est donc pas "plus de données" — c'est **changer la classe de fonctions**, en imposant une structure plus forte que Lipschitz. C'est exactement ce que font les réseaux de neurones via la compositionnalité et les invariances, ce qui leur permet de casser cette borne.

> [!note]- Pourquoi Lipschitz ne suffit pas
> Lipschitz dit seulement "$f$ ne varie pas trop vite". C'est une hypothèse faible qui ne dit rien sur la **structure** de $f$. Les hypothèses plus fortes qui cassent la borne sont par exemple : additivité ($f(x) = \sum_i f_i(x_i)$, qui ramène à $d$ problèmes 1D), compositionnalité (la vraie $f$ se décompose en couches, chaque couche simple), ou faible dimension intrinsèque (les données vivent sur une sous-variété de dimension $k \ll d$ — ce qu'on formalise en I.F).

> [!note]- Récap — malédiction de la dimension par modèle
> | Modèle | Comment il échappe (ou pas) à la malédiction |
> |---|---|
> | **Régression linéaire** | Échappe par hypothèse structurelle forte. EPE croissant en $p/N$, pas exponentiel. Biais nul si $f$ est vraiment linéaire, sinon biais structurel élevé. |
> | **kNN** | Ne s'échappe pas. Localité sans hypothèse → le voisin s'éloigne exponentiellement vite, biais explose (cf. exemple ci-dessus). |
> | **Réseaux de neurones** | Échappent **si** l'architecture encode les bonnes symétries (CNN → translation, RNN → translation temporelle). Sinon (MLP générique) restent dans le pire cas, comme le kNN. Cf. [[00_Perceptron Multi-Couches#VI — Pourquoi ça marche : Mallat]]. |
> | **SVM (kernel)** | Le kernel trick projette en très haute dimension, mais le **kernel** encode une notion de similarité (souvent locale, comme RBF) — même problème que kNN si le kernel n'exploite pas de structure. Les kernels structurés (ex : kernel de chaînes pour texte) s'en sortent mieux. |
> | **Arbres / Random Forest** | Partitionnement récursif de l'espace — souffre aussi de la sparsité en haute dimension (chaque split a de moins en moins de points), mais le bagging réduit la variance. |
>
> Le facteur commun : un modèle échappe à la malédiction **uniquement s'il impose une structure** (linéarité, symétrie, additivité) qui réduit la dimension effective du problème. Sans ça, tout modèle retombe dans le pire cas.

> [!note]- Lien avec l'overfitting (cf. I.D)
> La malédiction de la dimension et l'overfitting sont **deux symptomes du même problème sous-jacent** : pas assez de données par rapport à la complexité effective du problème. La décomposition biais-variance de I.D s'applique directement ici — le 1-NN en haute dimension est un cas extrême de la courbe en U de la Figure 4 : biais énorme, variance quasi nulle. C'est un point important car l'intuition naïve associe souvent kNN/modèles flexibles à "forte variance" — mais en haute dimension, c'est le **biais** qui domine pour le 1-NN (cf. décomposition ci-dessus). La malediction de la dimension est donc un cas où l'overfitting au sens classique (variance) n'est **pas** le problème principal — c'est l'incapacité structurelle à capturer $f$ qui domine.

---

### F. Manifold Hypothesis

> La section E a décrit le pire cas : données uniformes dans $\mathbb{R}^d$, malédiction pleine. Cette section explique pourquoi le ML fonctionne malgré tout en pratique — et quelle hypothèse fondamentale le justifie.

**Le constat.** Les données réelles ne remplissent pas $\mathbb{R}^d$ de façon uniforme. Des images de chats sont techniquement des vecteurs de $\mathbb{R}^{10^6}$ (pixels), mais elles ne peuvent pas avoir n'importe quelle valeur : elles forment une structure très fine dans cet espace énorme.

> [!warning] Manifold Hypothesis
> Les données réelles vivent sur une **sous-variété** de dimension intrinsèque $k \ll d$, plongée dans l'espace ambiant $\mathbb{R}^d$. La malédiction de la dimension joue en $k$, pas en $d$.

**L'intuition de Bishop.** Pourquoi $k$ est-il petit ? Parce que les données sont générées par un petit nombre de **causes physiques**, même si leur représentation est en haute dimension. Pour des images d'un objet, les degrés de liberté réels sont par exemple : l'orientation de l'objet, sa position dans la scène, la distance à la caméra. Trois paramètres physiques — pas un million. Toutes les images de cet objet forment donc une surface de dimension $k=3$ dans $\mathbb{R}^{10^6}$, pas un nuage qui remplit l'espace.

![[manifold_hypothesis.png|355]]
**Figure.** *Placeholder — sous-variété (surface courbe) dans un espace ambiant $\mathbb{R}^3$, avec deux classes (chats / chiens) comme régions sur cette variété et une frontière de décision dessus. Les trois axes représentent les paramètres physiques de Bishop : orientation, position, distance caméra.*

**Ce que ça change pour les modèles.** Le ML ne réduit pas la dimension ambiante $d$ — un réseau de neurones prend bien $d$ features en entrée. Ce qu'il apprend, c'est à **exploiter** la structure de dimension $k$ pour faire sa prédiction. La malédiction joue en $k$ parce que c'est la vraie complexité du problème.

Deux grandes familles selon comment elles traitent la variété :

- **PCA** : cherche une sous-variété **linéaire** (un sous-espace plat). Elle approxime la variété par le meilleur hyperplan de dimension $k$. C'est la version linéaire de la Manifold Hypothesis.
- **Réseaux de neurones** : apprennent une sous-variété **non-linéaire** pendant l'entraînement. Chaque couche déforme l'espace des représentations pour "déplier" la variété et rendre la tâche linéairement soluble en sortie.

> [!note]- Dimension intrinsèque et Lipschitz
> Le corollaire de Mallat (I.E) donne une borne en $\epsilon^{-d}$ sous hypothèse Lipschitz seule. La Manifold Hypothesis est précisément l'hypothèse structurelle plus forte qui **casse cette borne** : si les données vivent sur une variété de dimension $k$, la borne devient $\epsilon^{-k}$. Tout le gain vient de $d \to k$. C'est pourquoi la compositionnalité et les invariances des réseaux de neurones ne sont pas juste des astuces d'ingénieur — elles sont la façon dont le modèle exploite implicitement que $k \ll d$.

> [!note]- La variété n'est pas prouvée, c'est un pari
> La Manifold Hypothesis est une hypothèse empirique, pas un théorème. On ne sait pas calculer $k$ exactement pour un dataset réel, et la variété n'est pas visualisable directement dès que $d > 3$. Ce qu'on sait : empiriquement, les modèles qui l'exploitent (neural nets, kernel methods, PCA) fonctionnent bien malgré $d$ grand. Si les données étaient vraiment uniformes ($k = d$), aucun modèle ne s'en sortirait — pas même les réseaux de neurones profonds.

---

## II. Méthodologie de projet

### A. Vue d'ensemble : l'algo complet

> [!warning] Paramètres vs Hyperparamètres
> - **Paramètres** : appris par ERM (ex : poids $\theta$, $\beta$ d'une régression).
> - **Hyperparamètres** : choisis avant l'entraînement (ex : $\lambda$ de ridge, profondeur d'arbre, $K$ de KNN).

**Le squelette d'un projet ML.** On commence par isoler le **test set** du reste. Sur les données restantes (train + validation), on fait une **boucle externe** d'hyperparameter tuning, et **dans chaque itération**, une **boucle interne** de k-fold cross-validation. On choisit les hyperparamètres qui minimisent la CV error.

> [!note]- Pseudo-code (algo complet)
> ```
> 1. Splitter Dataset → Train+Val (80%) / Test (20%)
>    Test mis au coffre-fort.
> 
> 2. Pour chaque λ ∈ {1, 2, 3}:                    ← grid search
>        Pour chaque k ∈ {1, ..., 5}:              ← k-fold CV
>             Entraîner sur 4 folds avec λ
>             Évaluer sur fold k → MSE_k
>        CV(λ) = moyenne(MSE_1, ..., MSE_5)
>    
> 3. λ* = argmin CV(λ)                              ← model selection
> 
> 4. Réentraîner UN modèle avec λ* sur tout Train+Val.
> 
> 5. Évaluer ce modèle sur Test set.                ← model assessment
>    → C'est le chiffre que tu annonces.
> ```

> 💡 **L'idée à retenir.** Toute la suite de cette partie déroule cet algo : la **CV** (II.B) implémente la boucle interne, l'**hyperparameter tuning** (II.C) implémente la boucle externe, et le **data leakage** (II.D) liste les pièges à éviter pour que ce cadre reste valide.

---

### B. Cross-validation

**Motivation.** $f$ est inconnue, donc impossible de calculer le test MSE directement (cf I.D). Les **méthodes de resampling** contournent ça en réutilisant les données d'entraînement : on met de côté une portion, on entraîne sur le reste, on évalue sur la portion mise de côté. Répété intelligemment, ça donne une estimation du test error.

> [!warning] Test error
> $$\text{Err} \;=\; \mathbb{E}\big[L(Y_0, \hat f(X_0))\big]$$
> 
> L'erreur moyenne d'un modèle entraîné sur $\mathcal{T}$ quand on l'applique à un nouveau point $(X_0, Y_0)$ tiré de la **même distribution**. C'est exactement l'EPE de I.D.

#### B.1 Model selection vs Model assessment

Deux usages distincts qu'il faut absolument séparer :

- **Model selection** : choisir entre plusieurs candidats (degré de polynôme, $\lambda$ de ridge, profondeur d'arbre…). On compare leurs estimations de test error et on prend le meilleur.
- **Model assessment** : estimer la performance du modèle final, une fois choisi. C'est ce qu'on annonce en production : *"mon modèle aura environ X% d'erreur"*.

> 💡 **Pourquoi cette distinction est non négociable.** Si on utilise le même set pour les deux, l'assessment est **biaisé vers le bas** : on a choisi le minimum d'un ensemble bruité, donc l'erreur affichée est trop optimiste. D'où le **3-way split** — train / validation / test — où le test set ne sert **qu'à la fin**, jamais touché pendant la sélection.

![[Pasted image 20260426091303.png|435]]
**Figure 7.** Le 3-way split classique. Train pour fit, validation pour choisir le modèle, test pour assesser sa performance finale.

**Le problème pratique.** Le 3-way split suppose qu'on a assez de données pour les couper en trois. En pratique, on n'en a jamais assez. La cross-validation résout ça en **réutilisant** les données pour faire à la fois train et validation.

#### B.2 K-fold cross-validation

> [!warning] Définition
> On découpe le dataset en $K$ parties (folds) de taille à peu près égale. Pour chaque fold $k = 1, \ldots, K$, on entraîne le modèle sur les $K-1$ autres folds et on évalue sur le fold $k$. On moyenne les $K$ erreurs obtenues.
> 
> Soit $\kappa : \{1, \ldots, N\} \to \{1, \ldots, K\}$ la **fonction de partition** qui assigne chaque observation $i$ à un fold $\kappa(i)$, et $\hat f_\theta^{-k}$ le modèle entraîné en retirant le fold $k$. La **CV error** est :
> 
> $$\boxed{\text{CV}(\hat f_\theta) \;=\; \frac{1}{N} \sum_{i=1}^N L\big(y_i, \hat f_\theta^{-\kappa(i)}(x_i)\big)}$$

> [!note]- Cas particuliers : MSE et misclassification error
> **Régression (MSE).** $L(y, \hat y) = (y - \hat y)^2$ donne :
> 
> $$\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K \text{MSE}_k, \qquad \text{MSE}_k = \frac{1}{|F_k|} \sum_{i \in F_k} (y_i - \hat f_\theta^{-k}(x_i))^2$$
> 
> où $F_k$ est l'ensemble des indices du fold $k$.
> 
> **Classification (misclassification error).** $L(y, \hat y) = \mathbf{1}\{y \neq \hat y\}$ donne :
> 
> $$\text{CV}_{(K)} = \frac{1}{K} \sum_{k=1}^K \text{Err}_k, \qquad \text{Err}_k = \frac{1}{|F_k|} \sum_{i \in F_k} \mathbf{1}\{y_i \neq \hat f_\theta^{-k}(x_i)\}.$$

![[Pasted image 20260426091723.png|413]]
**Figure 8.** K-fold cross-validation avec $K = 5$. À chaque itération, un fold différent sert de validation (zone bleue) tandis que les 4 autres servent à l'entraînement.

**Cas particuliers.** Selon le choix de $K$ :
- $K = 2$ : **validation set approach** (la moitié train, la moitié validation).
- $K = N$ : **leave-one-out** (LOOCV). Chaque observation est validée seule.
- $K = 5$ ou $10$ : valeurs standard recommandées en pratique.

#### B.3 Choix de $K$ : trade-off biais-variance de la CV elle-même

> [!warning] Le compromis
> - **$K$ grand (LOOCV)** : faible biais (chaque modèle est entraîné sur quasi tout le dataset) mais **variance élevée** (les $N$ training sets sont quasi identiques entre eux, leurs erreurs sont fortement corrélées). Et coût computationnel énorme.
> - **$K$ petit ($K=2$)** : variance basse mais **biais élevé** (les modèles sont entraînés sur la moitié des données — si la learning curve n'a pas encore plafonné, la CV surestime l'erreur).
> - **$K = 5$ ou $10$** : compromis recommandé.

> 💡 **Subtilité importante.** La CV estime bien l'**erreur attendue** $\text{Err}$ (moyennée sur tous les training sets possibles), pas l'**erreur conditionnelle** $\text{Err}_\mathcal{T}$ pour ton training set spécifique. Pour la plupart des applications, c'est ce qu'on veut.

![[Pasted image 20260426092050.png|456]]
**Figure 9.** Récapitulatif des méthodes de resampling : validation set, LOOCV, k-fold.

---

### C. Hyperparameter tuning

**Motivation.** La k-fold CV te donne, pour **un** candidat d'hyperparamètres $\theta$, une estimation du test error. Reste à savoir comment **générer la liste de candidats** à comparer. C'est le rôle du **hyperparameter tuning** (ou **HPO**, *hyperparameter optimization*).

> [!warning] Trois grandes méthodes
> - **Grid search** (recherche par grille). On fixe à l'avance une grille de valeurs pour chaque hyperparamètre, et on teste **toutes les combinaisons**. Ex : $\lambda \in \{0.01, 0.1, 1, 10\}$ × profondeur $\in \{3, 5, 7\}$ → 12 candidats. Simple, exhaustif, mais le coût explose en haute dimension (curse of dimensionality dans l'espace des hyperparams).
> 
> - **Random search** (recherche aléatoire). On tire $N$ valeurs aléatoirement dans l'espace des hyperparamètres. **Souvent meilleur que grid search à budget égal** : Bergstra & Bengio (2012) ont montré qu'avec $N$ tirages, random search couvre $N$ valeurs distinctes par dimension, contre $\sqrt[d]{N}$ pour grid search en dimension $d$.
> 
> - **Bayesian optimization** (optimisation bayésienne). On modélise $\theta \mapsto \text{CV}(\theta)$ comme un **processus gaussien** et on choisit le prochain $\theta$ à tester en équilibrant **exploration** (zones inconnues) et **exploitation** (zones prometteuses). Beaucoup plus efficace en nombre de candidats testés, mais plus complexe à mettre en place. Outils standards : `optuna`, `hyperopt`, `scikit-optimize`.

> 💡 **Critère de sélection.** Quelle que soit la méthode, on choisit $\theta^* = \arg\min_\theta \text{CV}(\theta)$. **On ne regarde pas l'écart train/CV** pour décider — un modèle underfitté a aussi un faible écart, ce critère n'est donc pas suffisant. Le seul critère qui marche est de **minimiser la CV error**.

---

### D. Data leakage

> [!warning] Définition
> Le **data leakage** désigne toute situation où de l'information du validation/test set "fuit" dans le train, ce qui rend l'estimation du test error **trop optimiste**. Tu obtiens un super CV score, mais le modèle marche mal en production.

#### D.1 Les quatre cas classiques

- **Preprocessing avant le split.** Tu fais un `StandardScaler` sur **tout le dataset** avant de splitter → la moyenne et l'écart-type calculés ont vu les données du test. Il faut **fit le scaler sur le train uniquement**, puis appliquer (`transform`) au test. Idem pour l'imputation des `NaN` par la moyenne.

- **Feature selection avant la CV.** Tu sélectionnes les top-$K$ features les plus corrélées avec $Y$ sur tout le dataset, puis tu fais une CV. La sélection elle-même a vu le validation fold. Il faut faire la sélection **dans chaque fold**.

- **Target leakage** (le plus vicieux). Une de tes features est en réalité une fonction de la target qui ne sera pas disponible en production. Ex : prédire "patient malade" en utilisant "nombre de jours d'hospitalisation". CV impeccable, modèle inutilisable.

- **Time leakage** (séries temporelles). Faire une CV aléatoire sur des données temporelles = utiliser le futur pour prédire le passé. Il faut une **time-series split** (entraîner sur $t < T$, valider sur $t \geq T$).

> 💡 **Règle d'or.** Tout preprocessing qui "regarde" $Y$ ou les statistiques globales du dataset doit être encapsulé dans le pipeline et fitté **uniquement sur le train fold** à chaque itération de CV. En sklearn, c'est le rôle de `Pipeline` + `ColumnTransformer`.

> [!example] Fil rouge — un cas de target leakage évident
> Pour notre problème de prix d'appartement, imaginons qu'une feature s'appelle `prix_au_m2`. C'est tentant : c'est très corrélé au prix, le modèle aura un score impeccable. Mais cette feature est **calculée à partir du prix lui-même** ($\text{prix\_au\_m2} = \text{prix} / \text{surface}$). En production, pour un nouvel appartement, on n'a pas encore le prix — donc pas la feature. Le modèle ne sert à rien.
> 
> C'est un cas exagéré, mais le même piège existe sous des formes plus subtiles : agrégations temporelles qui incluent des dates futures, indicateurs construits a posteriori, identifiants de transaction qui encodent l'ordre de vente, etc.

---

### E. Learning curves (diagnostic)

**Motivation.** $f$ est inconnue, donc impossible de calculer biais et variance directement. Andrew Ng propose un diagnostic visuel : on trace $\text{MSE}^{\text{train}}$ et $\text{MSE}^{\text{cv}}$ en fonction de la **taille du training set** $n$, et on lit le problème dans la forme des courbes.

> 💡 **Notation.** Andrew Ng note ces erreurs $J_{\text{train}}$ et $J_{\text{cv}}$ et la taille du training set $m$. On utilise ici $\text{MSE}^{\text{train}}, \text{MSE}^{\text{cv}}$ et $n$ pour rester cohérent avec le reste de la note.

> [!warning] Définition
> - $\text{MSE}^{\text{train}}(n)$ : erreur moyenne sur le training set de taille $n$ utilisé pour entraîner.
> - $\text{MSE}^{\text{cv}}(n)$ : erreur moyenne sur un validation set fixe.
> 
> On entraîne le même modèle sur des sous-ensembles de taille $n = 1, 2, \ldots, N$ (où $N$ est la taille totale disponible) et on trace les deux courbes.

#### E.1 Cas high bias (underfitting)

![[Pasted image 20260425194100.png|324]]
**Figure 10.** High bias. $\text{MSE}^{\text{train}}$ et $\text{MSE}^{\text{cv}}$ convergent rapidement vers une valeur élevée et se collent l'une à l'autre. Le modèle est structurellement trop simple — ajouter des données ne servira à rien.

**Diagnostic** : ajouter des données est inutile.
**Action** : modèle plus complexe, ajouter des features, réduire la régularisation.

#### E.2 Cas high variance (overfitting)

![[Pasted image 20260425194136.png|462]]
**Figure 11.** High variance. $\text{MSE}^{\text{train}}$ reste très basse, $\text{MSE}^{\text{cv}}$ reste élevée mais descend lentement. Gros écart entre les deux courbes — signature de l'overfitting.

**Diagnostic** : ajouter des données va probablement aider — la variance se réduit avec $n$.
**Action** : plus de données, modèle plus simple, augmenter la régularisation.

> 💡 **À retenir.** Les learning curves donnent une réponse directe à la question pratique *"est-ce que ça vaut le coup de collecter plus de données ?"*. Si tu es en high bias, non. Si tu es en high variance, oui.

---

### F. Hypothèse i.i.d. et extrapolation

**Hypothèse i.i.d.** Tout le cadre théorique qu'on a construit suppose que les données de test sont tirées de la **même distribution** que les données d'entraînement (i.i.d. = independent and identically distributed). Cette hypothèse définit une frontière de validité précise.

> [!warning] Interpolation vs extrapolation
> - **Interpolation** : $x_{\text{new}}$ appartient au domaine couvert par les données d'entraînement. Le modèle est dans son régime de validité, l'EPE estimé est représentatif.
> - **Extrapolation** : $x_{\text{new}}$ est en dehors. L'EPE ne dit plus rien — on n'a aucune garantie sur le comportement de $\hat f$.

> [!example] Âge → salaire
> On entraîne une régression linéaire sur des données collectées uniquement entre 18 et 40 ans. Dans la zone observée, le modèle est bien calé. Mais quand on l'évalue à 60 ou 70 ans, il continue mécaniquement sa droite et prédit des salaires de plus en plus élevés — alors que la vraie relation plafonne puis redescend. **Le modèle n'a aucun moyen de le savoir** : il n'a jamais vu de données là-bas.

![[Pasted image 20260425224841.png|412]]
**Figure 12.** Interpolation vs extrapolation. Zone verte (18–40 ans) : données observées, le modèle linéaire (trait plein bleu) approxime bien la vraie relation (pointillés gris). Zone rouge (40–80 ans) : aucune donnée. Le modèle (trait pointillé bleu) extrapole linéairement et diverge dramatiquement de la vraie relation, qui plafonne autour de 60 ans avant de redescendre.

> 💡 **Le piège en haute dimension.** En 1D la zone d'interpolation est facile à voir (un intervalle). En haute dimension, les données couvrent une **région très creuse** de l'espace. Un nouveau point peut sembler raisonnable sur chaque feature individuellement, mais correspondre à une **combinaison jamais vue** — donc être en extrapolation sans qu'on s'en rende compte.

> 💡 **Distribution shift.** Le terme générique pour ce problème — quand la distribution de test diffère de la distribution d'entraînement — est le **distribution shift**. C'est un champ de recherche actif en ML (domain adaptation, out-of-distribution generalization). Aucune quantité de paramètres ou de régularisation n'élimine ce problème : si la distribution change, le modèle entraîné dans l'ancienne ne donne aucune garantie dans la nouvelle.

---

## Annexe — Vocabulaire

> [!note]- 📖 Loss / cost / risk / objective
> Les termes pour désigner "ce qu'on minimise" sont souvent utilisés de manière interchangeable. Les distinctions techniques :
> 
> | Terme | Définition | Échelle |
> |---|---|---|
> | **Loss function** $L(y, \hat y)$ | Mesure d'erreur sur **une seule observation** | 1 point |
> | **Cost function** $J$ | Moyenne de la loss sur **un ensemble d'observations**. On peut la calculer sur le train ($J_{\text{train}}$), validation ($J_{\text{cv}}$), ou test ($J_{\text{test}}$) | Échantillon fini |
> | **Empirical risk** $\hat R(f)$ | Synonyme de cost function évaluée sur le training set | Échantillon fini |
> | **Risk** $R(f)$ ou **EPE** | Espérance de la loss sur la **vraie distribution** $P(X,Y)$ | Population infinie |
> | **Objective function** | Ce qu'on minimise effectivement, peut inclure une régularisation : $J(\theta) + \lambda \|\theta\|^2$ | Échantillon fini |
> 
> En pratique, **Andrew Ng** dit "cost function" pour ce que **Hastie** appelle "empirical risk", et **Bishop** dit "error function". Reconnais les synonymes.

> [!note]- 📖 Autres synonymes utiles
> - **Hyperparameters** ($\lambda$ de ridge, profondeur d'arbre, $K$ de KNN…) = choisis par CV
> - **Parameters** ($\theta$, $\beta$, poids) = appris par ERM
> - **Training error** = erreur sur le train
> - **Test error** = erreur sur un test set fini
> - **Generalization error** = test error quand le test set → ∞ (= EPE)
