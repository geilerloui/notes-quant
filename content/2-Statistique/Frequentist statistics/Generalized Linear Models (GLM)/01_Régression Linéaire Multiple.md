---
title: Régression linéaire
order: 1
---

# Régression linéaire

> Cette note couvre la régression linéaire : le modèle, les hypothèses de Gauss-Markov, l'inférence, les variantes d'OLS et la régularisation. Le fil rouge pour les diagnostics est le dataset `Auto` (ISLR), qui présente naturellement plusieurs des pathologies classiques (non-linéarité, hétéroscédasticité, clustering, multicolinéarité).

## I. Le modèle

### A. Le problème de régression — la solution générale

Avant de parler de régression *linéaire*, posons le **problème général** : on a deux variables aléatoires $X$ et $Y$, et on cherche à approximer $Y$ par une fonction de $X$. Quelle est la **meilleure** fonction $f(X)$ ?

> [!warning] Le problème de régression
> On cherche la fonction $f$ qui minimise l'erreur quadratique moyenne :
> 
> $f^* = \arg\min_f E\big[(Y - f(X))^2\big]$
> 
> **La solution est l'espérance conditionnelle** : $f^*(X) = E(Y \mid X)$.

> 💡 **Pourquoi $E(Y|X)$ ?** C'est un théorème général de probabilité, indépendant de la régression linéaire. La meilleure prédiction de $Y$ à partir de $X$ au sens MSE est la moyenne de $Y$ conditionnellement à $X$. Cette fonction $f^*(X)$ s'appelle la **fonction de régression** — c'est l'objet qu'on cherche à approximer en pratique.

#### Vue géométrique dans $L^2$

On travaille dans $L^2$, muni du produit scalaire $\langle X, Y \rangle = E(XY)$. Dans cet espace, **les vecteurs sont des variables aléatoires**.

> [!warning] Décomposition par projection orthogonale
> $E(Y|X)$ est la **projection orthogonale** de $Y$ sur le sous-espace $L^2_X$ des fonctions de $X$. Cela donne une décomposition naturelle de $Y$ en deux parties orthogonales :
> 
> $Y = E(Y|X) + \varepsilon$
> 
> Le résidu $\varepsilon = Y - E(Y|X)$ est **orthogonal à $L^2_X$** : $E[\varepsilon \cdot f(X)] = 0$ pour toute fonction $f$.

Deux conséquences immédiates :

- $E(\varepsilon) = 0$ — en prenant $f = 1$
- $\text{cov}(\varepsilon, X) = 0$ — en prenant $f = X$

![Interprétation géométrique dans L²|407](images/2-Statistiques/A_Frequentist/regression-lineaire/im2.png)

**Figure 1.** Interprétation géométrique dans $L^2$. $E(Y|X)$ est la projection orthogonale de $Y$ sur $L^2_X$ (le sous-espace de **toutes** les fonctions de $X$). Le résidu $\varepsilon = Y - E(Y|X)$ est orthogonal à tout le sous-espace $L^2_X$.

> 💡 **À ce stade, rien n'est encore "linéaire".** $E(Y|X)$ peut être n'importe quelle fonction de $X$ — courbe, polynomiale, en escalier. La régression *linéaire* arrive à la section suivante : on **restreint** la recherche aux fonctions affines.

### B. Restriction au cas linéaire — les vrais paramètres $\alpha, \beta$

On restreint la recherche aux **fonctions affines de $X$** : $f(X) = \alpha + \beta X$. C'est le **modèle de régression linéaire** au niveau **population**.

> [!warning] Modèle de régression linéaire (population)
> On suppose que la fonction de régression est affine :
> 
> $Y = \alpha + \beta X + \varepsilon$
> 
> Géométriquement, on projette $Y$ sur le sous-espace 1D des fonctions affines de $X$ (au lieu de tout $L^2_X$ comme en I.A).

![[Pasted image 20260507105136.png|408]]

**Figure 2.** Distinction population vs échantillon. La **vraie droite** $y = \alpha + \beta x$ (en bleu) est définie au niveau de la loi de $(X, Y)$ avec $\alpha, \beta$ les vrais paramètres. La **droite estimée** $\hat{y} = \hat{\alpha} + \hat{\beta}x$ (en rouge) est calculée à partir d'un échantillon fini d'observations $(x_i, y_i)$ via MCO. Les deux droites diffèrent légèrement : $\hat{\alpha}, \hat{\beta}$ sont des estimateurs de $\alpha, \beta$, donc aléatoires — ils convergent vers les vrais paramètres quand $n \to \infty$.

#### Calcul des vrais paramètres $\alpha, \beta$

Les paramètres $\alpha, \beta$ sont définis par les **conditions d'orthogonalité** : le résidu $\varepsilon = Y - \alpha - \beta X$ doit être orthogonal au sous-espace engendré par $1$ et $X$ dans $L^2$.

> [!warning] Vrais paramètres de la régression linéaire (population)
> $\boxed{\alpha = E(Y) - \beta E(X)} \qquad \boxed{\beta = \frac{\text{cov}(X,Y)}{V(X)}}$
> 
> **Attention** : ce sont les **vrais paramètres** au niveau de la loi de $(X, Y)$, **pas des estimateurs**. Ils utilisent les vraies espérances et covariances — qu'on ne connaît pas en pratique.

> [!note]- Preuve — par orthogonalité dans $L^2$
> **Condition 1 :** $\langle \varepsilon, 1 \rangle = 0$
> 
> $E(Y - \alpha - \beta X) = 0 \implies \alpha = E(Y) - \beta E(X)$
> 
> **Condition 2 :** $\langle \varepsilon, X \rangle = 0$
> 
> $E\big[(Y - \alpha - \beta X)X\big] = 0 \implies E(YX) - \alpha E(X) - \beta E(X^2) = 0$
> 
> On substitue $\alpha = E(Y) - \beta E(X)$ :
> 
> $E(YX) - E(Y)E(X) = \beta\big(E(X^2) - E(X)^2\big)$
> 
> Or $E(XY) - E(X)E(Y) = \text{cov}(X,Y)$ et $E(X^2) - E(X)^2 = V(X)$, donc :
> 
> $\beta = \frac{\text{cov}(X,Y)}{V(X)}$

> 💡 **Lecture des deux formules.** La droite passe toujours par le point $\big(E(X), E(Y)\big)$ — $\alpha$ n'est qu'un ajustement de position. $\beta$ mesure combien de la variation de $X$ se transfère à $Y$, normalisée par la dispersion de $X$.

### C. Estimation sur échantillon — Moindres Carrés Ordinaires (MCO)

En pratique, on **ne connaît pas** la loi de $(X, Y)$ — on dispose seulement d'un **échantillon** de $n$ observations $(x_i, y_i)$. On ne peut pas calculer $E(X)$, $V(X)$, $\text{cov}(X, Y)$ directement, on doit les **estimer**.

> [!warning] Le critère MCO
> On choisit $\hat{\alpha}, \hat{\beta}$ qui minimisent la somme des carrés des résidus sur l'échantillon :
> 
> $(\hat{\alpha}, \hat{\beta}) = \arg\min_{\alpha, \beta} \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2= \arg\min_{\alpha, \beta} \sum_{i=1}^n (\varepsilon_i)^2$
> 
> **MCO** = Moindres Carrés Ordinaires (OLS en anglais). C'est l'**estimation empirique** des paramètres $\alpha, \beta$ de la section B.

![[Pasted image 20260507105940.png|362]]

**Figure 3.** Visualisation du critère MCO. Les segments verticaux pointillés rouges représentent les résidus $\varepsilon_i = y_i - \hat{y}_i$ — la distance verticale entre chaque observation $y_i$ et sa valeur prédite $\hat{y}_i = \hat{\alpha} + \hat{\beta}x_i$ sur la droite. MCO choisit $\hat{\alpha}, \hat{\beta}$ qui minimisent **la somme des carrés** de ces segments. Élever au carré pénalise davantage les grands résidus (un résidu deux fois plus grand pèse quatre fois plus dans le critère).

#### Estimateurs des moindre carrés (régression simple)

En remplaçant les espérances et covariances de la section B par leurs **versions empiriques** (sommes au lieu d'intégrales), on obtient :

> [!warning] Estimateurs MCO (régression simple)
> $\boxed{\hat{\beta} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2}} \qquad \boxed{\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}}$
> 
> Ce sont les **estimateurs** des vrais $\alpha, \beta$. Quand $n \to \infty$, par la loi des grands nombres, $\hat{\beta} \to \beta$ et $\hat{\alpha} \to \alpha$.

> 💡 **Le parallèle avec B est exact.** Population : $\beta = \text{cov}(X,Y)/V(X)$, $\alpha = E(Y) - \beta E(X)$. Échantillon : $\hat{\beta} = \widehat{\text{cov}}/\hat{V}$, $\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$. Mêmes formules, juste les moments empiriques à la place des vrais.

#### Forme matricielle (régression multiple)

Avec $X \in \mathbb{R}^{n \times (p+1)}$ (la première colonne étant des 1 pour l'intercept), $\beta \in \mathbb{R}^{p+1}$ et $y \in \mathbb{R}^n$, le critère OLS s'écrit :

$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2$

> [!warning] Estimateur OLS sous forme matricielle
> $\boxed{\hat{\beta} = (X^T X)^{-1} X^T y}$
> 
> C'est *la* formule à connaître par cœur en entretien. Valable tant que $X^T X$ est inversible (i.e. pas de multicolinéarité parfaite, cf. H5).

> [!note]- Preuve — gradient nul
> On développe le critère :
> 
> $\|y - X\beta\|^2 = (y - X\beta)^T(y - X\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta$
> 
> Le gradient par rapport à $\beta$ :
> 
> $\nabla_\beta \|y - X\beta\|^2 = -2 X^T y + 2 X^T X \beta$
> 
> Annuler le gradient donne les **équations normales** :
> 
> $X^T X \hat{\beta} = X^T y$
> 
> Si $X^T X$ est inversible : $\hat{\beta} = (X^T X)^{-1} X^T y$.

#### Vue géométrique dans R^n


COURS DE : geometric interpretaiton of linear regression Nipun Batra


Attention : on **change d'espace**. La vue $L^2$ de la section A travaillait avec des **variables aléatoires**. Ici on travaille dans $\mathbb{R}^n$ avec des **vecteurs d'observations**.

> 💡 **Deux géométries distinctes.** 
> - **$L^2$ (population, section A)** : vecteurs = variables aléatoires, sous-espace $L^2_X$ = fonctions de $X$, projection = $E(Y|X)$.
> - **$\mathbb{R}^n$ (échantillon, ici)** : vecteurs = données observées ($y \in \mathbb{R}^n$, colonnes $X_1, \ldots, X_p \in \mathbb{R}^n$), sous-espace = espace colonne de $X$, projection = $\hat{y} = X\hat{\beta}$.
> 
> C'est la même *idée* (projection orthogonale qui minimise une distance) mais dans deux espaces différents.

![[geo-1.png|430]]

**Figure 4.** Estimation OLS comme projection orthogonale dans $\mathbb{R}^n$. Le vecteur $y$ (des $n$ observations) est projeté sur l'espace colonne de $X$ (engendré par les vecteurs-colonnes $X_1, X_2$). Le résidu estimé $\hat{\varepsilon}$ est la distance orthogonale entre $y$ et sa projection $\hat{y} = X\hat{\beta}$.

> [!note]- Lecture détaillée de $\hat{\beta} = (X^T X)^{-1} X^T y$
> - $X^T y$ est le **vecteur des produits scalaires** entre chaque colonne de $X$ et $y$ : $(X^T y)_j = \langle X_j, y \rangle$. Mesure comment $y$ se projette sur chaque direction $X_j$ individuellement.
> - $X^T X$ est la **matrice de Gram** des colonnes de $X$ : $(X^T X)_{jk} = \langle X_j, X_k \rangle$. Si les colonnes sont orthogonales (corrélations nulles), $X^T X$ est diagonale et l'inverse devient triviale ; sinon $(X^T X)^{-1}$ corrige les corrélations entre les $X_j$ pour décorréler les coefficients.

> 💡 **Lecture géométrique des équations normales.** $X^T X \hat{\beta} = X^T y$ signifie $X^T (y - X\hat{\beta}) = 0$, soit $X^T \hat{\varepsilon} = 0$ : les résidus estimés sont **orthogonaux aux colonnes de $X$**. C'est l'analogue empirique de la condition $\text{cov}(\varepsilon, X) = 0$ de la section A — projection orthogonale, version échantillon.

> [!note]- La hat matrix $H = X(X^T X)^{-1} X^T$
> En substituant $\hat{\beta}$, on obtient $\hat{y} = X \hat{\beta} = H y$ avec $H = X(X^T X)^{-1} X^T$. Cette matrice $H$ ("hat matrix" parce qu'elle "met le chapeau" sur $y$) est la **matrice de projection orthogonale** sur l'espace colonne de $X$. Propriétés : symétrique, idempotente ($H^2 = H$), $\text{tr}(H) = p+1$. Elle réapparaîtra dans le calcul du leverage (cf. II.C).

### D. Vue probabiliste — OLS comme maximum de vraisemblance

Les sections B et C ont justifié OLS par la **géométrie** (projection orthogonale). On peut aussi le justifier par la **probabilité** : sous une hypothèse de bruit gaussien, OLS coïncide avec l'estimateur du maximum de vraisemblance (MLE).

> [!warning] Hypothèse probabiliste
> On suppose les résidus indépendants et gaussiens :
> 
> $\varepsilon_i \sim \mathcal{N}(0, \sigma^2) \quad \text{i.i.d.}$
> 
> Cela revient à modéliser $y_i \mid x_i \sim \mathcal{N}(x_i^T \beta,\ \sigma^2)$.

La log-vraisemblance des observations s'écrit :

$\ell(\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i^T \beta)^2$

Maximiser $\ell$ par rapport à $\beta$ revient à **minimiser** $\sum (y_i - x_i^T \beta)^2$ — c'est exactement le critère OLS.

![[Pasted image 20260506225347.png]]



![[Pasted image 20260506225403.png|490]]

> 💡 **OLS = MLE sous bruit gaussien.** Les moindres carrés ne sont pas un choix arbitraire. Trois lectures du même objet :
> - **Minimisation** (vue classique) : minimiser la somme des carrés des résidus
> - **Projection** (vue géométrique) : projeter $y$ orthogonalement sur l'espace colonne de $X$ dans $\mathbb{R}^n$
> - **Vraisemblance** (vue probabiliste) : maximiser la vraisemblance sous bruit gaussien

> [!note]- Lien avec Gauss-Markov
> Gauss-Markov donne le caractère BLUE de OLS **sans hypothèse gaussienne** — il faut juste H1–H5. L'hypothèse gaussienne supplémentaire apporte deux choses : (1) OLS devient MLE, donc *efficient* parmi tous les estimateurs (pas juste les linéaires), (2) la distribution de $\hat{\beta}$ devient exactement gaussienne, ce qui permet les tests de Student exacts (cf. III.A).

> [!note]- Récapitulatif des trois niveaux
> | Niveau | Objet | Espace géométrique | Projection sur |
> |---|---|---|---|
> | **A — Général** (population) | $E(Y\|X)$ | $L^2$ | sous-espace de toutes les fonctions de $X$ |
> | **B — Linéaire** (population) | $\alpha, \beta$ vrais | $L^2$ | sous-espace des fonctions affines de $X$ |
> | **C — Estimation** (échantillon) | $\hat{\alpha}, \hat{\beta}$ | $\mathbb{R}^n$ | espace colonne de la matrice $X$ |
> 
> A et B vivent au niveau **population** (vraies espérances) ; C vit au niveau **échantillon** (sommes empiriques). Ne jamais les confondre — c'est un piège pédagogique classique parce que les formules se ressemblent.

### E. Interprétation des coefficients

#### Prédicteurs continus

$\beta$ se lit ainsi : **si $X$ augmente d'une unité, $Y$ augmente en moyenne de $\beta$ unités**, à valeurs des autres prédicteurs constantes.

![[Pasted image 20260507125521.png]]

**Figure 5.** Interprétation visuelle de $\hat{\beta}_1$ sur Auto (`mpg ~ horsepower`). **À gauche** : vue globale, droite OLS rouge ajustée sur les 392 observations — le rectangle bleu pointillé marque la zone zoomée à droite. **À droite** : zoom sur la droite autour de hp = 100, avec le triangle "rise over run" pour $\Delta\text{hp} = +1$ : un cheval supplémentaire fait baisser la consommation prédite de $\hat{\beta}_1 \approx -0.158$ mpg. Le coefficient OLS est exactement la **pente** géométrique de la droite : $\hat{\beta}_1 = \Delta\widehat{\text{mpg}} / \Delta\text{hp}$.

> 💡 **L'effet pur en régression multiple.** Si le modèle est $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$, alors $\beta_1$ mesure l'effet de $X_1$ sur $Y$, **une fois l'effet de $X_2$ retiré**. C'est la conséquence directe de la géométrie : MCO projette $Y$ sur le sous-espace engendré par $X_1$ et $X_2$ conjointement. Si $X_1$ et $X_2$ sont orthogonaux dans $L^2$ — c'est-à-dire $\text{cov}(X_1, X_2) = 0$ — alors $\beta_1$ et $\beta_2$ sont exactement les coefficients qu'on obtiendrait en régressant $Y$ sur chacun séparément. L'orthogonalité annule toute interférence entre les variables.

> [!note]- Interprétation probabiliste sous loi normale multivariée
> Dans le cas particulier où $(Y, X_1, \ldots, X_p)$ suit une **loi normale multivariée**, les coefficients $\beta_j$ ont une interprétation probabiliste précise :
> 
> $$\beta_j = \rho_{Y X_j | X_{-j}} \cdot \frac{\sigma_Y}{\sigma_{X_j}}$$
> 
> où $\rho_{Y X_j | X_{-j}}$ est la **corrélation partielle** entre $Y$ et $X_j$ conditionnellement à tous les autres prédicteurs. La corrélation partielle mesure exactement ce qui reste de la relation entre $Y$ et $X_j$ une fois qu'on a retiré l'effet linéaire de toutes les autres variables. C'est le fondement probabiliste de l'interprétation "à autres prédicteurs constants".


en fait ici ce serait bien d'avoir le même graphe mais avec les données de notre mpg la 

+aussi une table avec t-stat p value pr l'interprétation




#### Prédicteurs catégoriels — les variables indicatrices (dummies)

Quand un prédicteur $X$ est catégoriel (par exemple `origin` ∈ {USA, Europe, Japon}), on ne peut pas l'inclure tel quel dans la régression — il faut l'**encoder en variables indicatrices** (one-hot encoding).

##### Cas à deux groupes — l'idée fondatrice

L'intuition la plus simple part du cas à **deux groupes** A et B. On définit une seule variable indicatrice :

$$x_i = \begin{cases} 0 & i \in A \\ 1 & i \in B \end{cases}$$

et on régresse $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$.

Cas 1 : on pose que $x_i=0$ on a donc $y_i=\beta_0+\varepsilon_i$ et $\frac{1}{n_A}\sum_{i=1}^{n}y_i = \frac{1}{n_A}\sum_{i=1}^{n} (\beta_0 + \varepsilon_i)$ en développant on obtient $\bar{y}_A=\beta_0$ car le bruit est nulle en moyenne. 
Cas 2 : on pose que $x_i=1$ on a donc $y_i=\beta_0+\beta_1 + \varepsilon_i$ on fait la même technique $\frac{1}{n_B}\sum_{i=1}^{n}y_i = \frac{1}{n_A}\sum_{i=1}^{n} (\beta_0 + \beta_1 + \varepsilon_i)$ on obtient $\bar{y}_B =\beta_0+\beta_1$ ainsi $\beta_1=\bar{y}_B-\bar{y}_A$ 



> [!warning] Lecture des coefficients (cas 2 groupes)
> - **$\beta_0$** : pour $x = 0$, on a $y_i = \beta_0 + \varepsilon_i$ → $\beta_0 =$ **moyenne du groupe A** (groupe de référence)
> - **$\beta_1$** : la pente vaut $\frac{\Delta y}{\Delta x} = \frac{\Delta y}{1} = \Delta y$ → $\beta_1 =$ **différence des moyennes** entre B et A
> 
> $$\beta_0 = \overline{y}_A, \qquad \beta_1 = \overline{y}_B - \overline{y}_A$$

![[Pasted image 20260505212641.png|619]]

**Figure 6.** Régression sur dummy = comparaison de moyennes. Les deux nuages de points représentent les deux groupes ; les losanges noirs marquent les moyennes empiriques. La droite OLS passe **exactement** par les deux moyennes : $\beta_0$ est la hauteur de la moyenne du groupe A (à $x=0$), et $\beta_1$ est l'écart vertical entre les deux moyennes.

> 💡 **À retenir.** Sur des dummies, OLS retrouve mécaniquement les moyennes par groupe — la régression "contient" les comparaisons de moyennes comme cas particulier. (Le lien formel régression ↔ t-test est traité en III.C.)

##### Généralisation à $K$ groupes

Quand le prédicteur catégoriel a $K$ modalités (par exemple 3 origines : USA, Europe, Japon), on généralise naturellement le modèle à deux groupes. Chaque observation $y_{ij}$ (observation $i$ du groupe $j$) s'écrit comme la somme d'un niveau de base + un effet de groupe + un résidu individuel.

> 💡 **Le problème d'identification.** Le modèle a $K + 1$ paramètres ($\mu, \tau_1, \ldots, \tau_K$) mais on n'observe que $K$ moyennes de groupe. Il y a donc **un paramètre de trop** — on peut ajouter une constante $c$ à $\mu$ et la retirer à tous les $\tau_j$ sans changer les prédictions. Pour rendre le modèle identifiable, il faut imposer **une contrainte** sur les $\tau_j$. Selon la contrainte choisie, l'interprétation des coefficients change — mais les **prédictions sont toujours les mêmes**.

**(i) La paramétrisation par défaut — treatment constraint**

> [!warning] Treatment constraint (par défaut dans Python/R)
> On fixe **$\tau_1 = 0$** (le premier groupe devient la **catégorie de référence**). Le modèle devient :
> 
> $$y_i = \beta_0 + \beta_2 \mathbb{1}_{i \in \text{groupe 2}} + \beta_3 \mathbb{1}_{i \in \text{groupe 3}} + \cdots + \beta_K \mathbb{1}_{i \in \text{groupe K}} + \varepsilon_i$$
> 
> Lecture des coefficients :
> - **$\beta_0$** : moyenne du **groupe de référence** (groupe 1)
> - **$\beta_k$** pour $k \geq 2$ : **différence** entre la moyenne du groupe $k$ et celle du groupe de référence

*Exemple sur Auto* avec `origin` ∈ {USA, Europe, Japon} et USA = référence :

$$\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Europe}} + \beta_3 \mathbb{1}_{\text{Japon}}$$

- $\beta_0$ : mpg moyen pour une voiture **USA** à hp = 0
- $\beta_2$ : différence moyenne de mpg entre **Europe et USA**, à hp égale
- $\beta_3$ : différence moyenne de mpg entre **Japon et USA**, à hp égale

> 💡 **Pourquoi la treatment constraint est le défaut.** C'est la plus interprétable en pratique : on choisit un groupe "naturel" comme référence (le plus fréquent, ou un baseline théorique) et tous les autres coefficients se lisent comme des **écarts par rapport à cette référence**. C'est ce que font `sklearn.preprocessing.OneHotEncoder(drop='first')`, `pd.get_dummies(drop_first=True)`, et le comportement par défaut de `lm()` en R.

> [!note]- Choix de la catégorie de référence
> Le choix de la catégorie de référence est purement conventionnel — il ne change pas les prédictions, juste l'interprétation des coefficients. Bonne pratique : prendre la catégorie la plus fréquente ou la plus "naturelle" comme référence (par exemple "USA" sur Auto puisque c'est 245 voitures sur 392).

**(ii) Le piège à éviter — dummy variable trap**

> 💡 **Le dummy variable trap.** Si on incluait les $K$ indicatrices (USA, Europe, Japon) **sans en retirer une** ET avec un intercept, leur somme vaudrait toujours 1 — exactement la colonne d'intercept. C'est de la **multicolinéarité parfaite** (cf. H5), $X^T X$ devient singulière, OLS ne peut plus tourner. Pour s'en sortir, deux options : soit on retire une catégorie (treatment constraint), soit on retire l'intercept (cf. paramétrisations alternatives ci-dessous).

**(iii) Paramétrisations alternatives**

La treatment constraint n'est pas la seule façon de rendre le modèle identifiable. Deux autres existent, pratiques dans certains contextes.

> [!note]- Paramétrisation 1 — sans intercept (means encoding)
> Au lieu de retirer une catégorie, on retire **l'intercept**. Le modèle devient :
> 
> $$y_i = \tau_1 \mathbb{1}_{i \in \text{groupe 1}} + \tau_2 \mathbb{1}_{i \in \text{groupe 2}} + \cdots + \tau_K \mathbb{1}_{i \in \text{groupe K}} + \varepsilon_i$$
> 
> Lecture des coefficients : **$\tau_k$ = moyenne du groupe $k$** directement. Pas de groupe de référence, chaque coefficient se lit comme une moyenne.
> 
> *Quand l'utiliser ?* Quand tu veux extraire directement les moyennes par groupe sans les recalculer à partir de l'intercept. Typique en finance — dans un modèle factoriel cross-sectionnel, on veut souvent les rendements de chaque secteur directement, pas les différences.

> [!note]- Paramétrisation 2 — sum-to-zero constraint
> On impose $\sum_{j=1}^K \tau_j = 0$. Le dernier groupe devient l'opposé de la somme des précédents : $\tau_K = -(\tau_1 + \cdots + \tau_{K-1})$.
> 
> Lecture des coefficients :
> - **$\mu$** : **moyenne globale** (la moyenne des moyennes de groupes)
> - **$\tau_k$** : différence entre la moyenne du groupe $k$ et la **moyenne globale**
> 
> *Quand l'utiliser ?* Quand on veut une interprétation "écart à la moyenne globale" plutôt que "écart à un groupe de référence".

> [!example] La sum-to-zero en finance — Barra Industry Factor Model
> Le modèle Barra utilise **exactement** une sum-to-zero constraint, mais dans sa version **pondérée par capitalisation** :
> 
> $\sum_{k \in \text{secteurs}} w_k \cdot f_{k,t} = 0 \qquad \text{où } w_k = \text{poids de capitalisation du secteur } k$
> 
> **Pourquoi cette contrainte précise ?** Elle garantit que :
> - Le **rendement marché global** est porté par l'intercept (souvent appelé "Country" ou "World" factor)
> - Les rendements sectoriels $f_{k,t}$ représentent purement des **écarts au marché** : "Tech a outperformé de +X%", "Banque a underperformé de −Y%"
> - Un portefeuille market-cap-weighted sur les secteurs a un rendement nul par construction (les écarts se compensent)
> 
> **L'intuition économique vs treatment constraint.** Si on utilisait une treatment constraint (genre "Tech = référence"), Tech aurait toujours $f_{\text{Tech}} = 0$ par construction — ce qui n'a pas de sens économique (pourquoi Tech serait privilégié ?). Et les autres facteurs se liraient comme "Banque vs Tech" au lieu de "Banque vs marché" — interprétation tordue. La sum-to-zero pondérée met **tous les secteurs sur un pied d'égalité** et fournit la baseline naturelle (le marché) pour interpréter chaque $f_{k,t}$.
> 
> > 💡 **À retenir.** En quant equity, *la* paramétrisation des dummies sectorielles est sum-to-zero pondérée par cap, jamais treatment. C'est pour ça qu'on parle de **factor returns** (rendements de facteurs, écarts au marché) et pas de "coefficients" — l'interprétation est *intrinsèque* au facteur, pas relative à un groupe choisi.

> [!note]- Récapitulatif — trois lectures du même modèle
> | Paramétrisation | Contrainte | Interprétation des coefficients |
> |---|:---:|---|
> | **Treatment** (défaut) | $\tau_1 = 0$ | $\beta_0$ = moyenne du groupe référence ; $\beta_k$ = différence vs référence |
> | **Sans intercept** | (intercept retiré) | $\tau_k$ = moyenne du groupe $k$ |
> | **Sum-to-zero** | $\sum \tau_j = 0$ | $\mu$ = moyenne globale ; $\tau_k$ = différence vs moyenne globale |
> 
> Les **trois donnent les mêmes prédictions $\hat{y}$** — c'est juste l'étiquette posée sur les coefficients qui change. On choisit selon ce qu'on veut lire directement.

**(iv) Application — cross-section Barra**

> [!example] Modèle factoriel sectoriel (sans intercept)
> Dans un modèle Barra Industry Factor Model, on régresse les rendements des actions à un instant $t$ sur leurs **expositions sectorielles** (dummies one-hot) :
> 
> $$\underset{(N \times 1)}{\mathbf{R}_t} = \underset{(N \times K)}{\mathbf{B}}\, \underset{(K \times 1)}{\mathbf{f}_t} + \underset{(N \times 1)}{\boldsymbol{\varepsilon}_t}$$
> 
> où $\mathbf{B}$ est une matrice de dummies de secteur (chaque action appartient à un et un seul secteur). On utilise la paramétrisation **sans intercept** : chaque coefficient $f_{kt}$ se lit directement comme la **moyenne des rendements du secteur $k$ à l'instant $t$** :
> 
> $$\hat{f}_{kt,\, \text{OLS}} = \frac{1}{N_k} \sum_{i \in \text{secteur } k} R_{it}$$
> 
> C'est exactement le résultat naturel : le **factor return du secteur k** est la moyenne des rendements des actions de ce secteur. La régression sur dummies one-hot est l'outil qui formalise cette intuition et permet de la généraliser (ajout d'autres facteurs, contraintes, WLS pour pondérer par capitalisation, etc.).


#### Prédicteurs mixtes — continu + catégoriel (modèle ANCOVA)

En pratique, un modèle réaliste mélange **prédicteurs continus** et **prédicteurs catégoriels** dans la même équation. Le cas typique sur Auto :

$$\widehat{\text{mpg}} = \beta_0 + \underbrace{\beta_1 \cdot \text{hp}}_{\text{continu}} + \underbrace{\beta_2 \cdot \mathbb{1}_{\text{Europe}} + \beta_3 \cdot \mathbb{1}_{\text{Japon}}}_{\text{catégoriel (USA = référence)}}$$

Ce type de modèle s'appelle un **ANCOVA** (Analysis of Covariance) — il combine ANOVA (effet d'un facteur catégoriel) et régression (effet d'une covariable continue).

> 💡 **La règle d'interprétation est la même pour tous les coefficients.** $\beta_j$ = *"si $X_j$ augmente d'une unité, à autres prédicteurs fixés, alors $Y$ augmente en moyenne de $\beta_j$ unités"*. Ce qui change, c'est ce que veut dire concrètement "augmenter d'une unité" :
> - **Continu** (hp) : passer de 100 à 101 chevaux. $\beta_1$ = effet marginal direct, **pente** de la droite.
> - **Catégoriel** (dummy) : passer de 0 à 1 = **changer de groupe**. $\beta_k$ = écart de moyenne entre groupes, **décalage vertical** de la droite.

**(i) Lecture géométrique — trois droites parallèles**

L'écriture $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Europe}} + \beta_3 \mathbb{1}_{\text{Japon}}$ se réécrit naturellement comme **trois droites séparées**, une par groupe :

$$
\begin{aligned}
\text{USA :} \quad & \widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} \\
\text{Europe :} \quad & \widehat{\text{mpg}} = (\beta_0 + \beta_2) + \beta_1 \text{hp} \\
\text{Japon :} \quad & \widehat{\text{mpg}} = (\beta_0 + \beta_3) + \beta_1 \text{hp}
\end{aligned}
$$

Les **trois droites partagent la même pente $\beta_1$** (l'effet de hp est supposé identique pour tous les groupes) mais ont des **intercepts différents** : $\beta_0$ pour USA, $\beta_0 + \beta_2$ pour Europe, $\beta_0 + \beta_3$ pour Japon. Les dummies $\beta_2, \beta_3$ sont les **distances verticales** entre la droite de référence (USA) et les autres.

![[Pasted image 20260505215212.png]]

**Figure 7.** Modèle ANCOVA `mpg ~ horsepower + origin` ajusté sur Auto. Les trois nuages de points (USA en rouge, Europe en bleu, Japon en vert) sont fittés par **trois droites parallèles** de pente commune $\hat{\beta}_1 = -0.134$. Les flèches verticales noires matérialisent les coefficients des dummies : $\hat{\beta}_{\text{Europe}} = +2.43$ et $\hat{\beta}_{\text{Japon}} = +5.18$ — à puissance égale, une voiture européenne consomme en moyenne 2.43 mpg de plus qu'une USA, et une japonaise 5.18 mpg de plus.

> [!example] Lecture concrète des coefficients estimés
> Sur Auto, les coefficients estimés sont $\hat{\beta}_0 = 35.94$, $\hat{\beta}_1 = -0.134$, $\hat{\beta}_{\text{Europe}} = +2.43$, $\hat{\beta}_{\text{Japon}} = +5.18$
> 
> - **$\hat{\beta}_0 = 35.94$** : intercept de la droite USA (mpg prédit pour une voiture américaine à hp = 0). Pas d'interprétation physique directe puisque hp = 0 n'existe pas, mais c'est le **point d'ancrage** de la droite de référence.
> - **$\hat{\beta}_1 = -0.134$** : à origine fixée, **chaque cheval supplémentaire fait baisser la mpg de 0.134**. Cette pente est la même pour les trois origines.
> - **$\hat{\beta}_{\text{Europe}} = +2.43$** : à hp égale, une voiture européenne fait **2.43 miles de plus par gallon** qu'une américaine — elle est plus économe.
> - **$\hat{\beta}_{\text{Japon}} = +5.18$** : à hp égale, une japonaise est encore plus économe — **5.18 mpg de plus** qu'une USA.
> 
> *Lecture économique* : à puissance comparable, les voitures japonaises et européennes des années 1970-80 étaient plus économes que les américaines (technologie moteur, poids, aérodynamisme). Le modèle ANCOVA capture ce fait : l'effet "origine" est un **décalage** indépendant de la puissance.

**(ii) Limite du modèle ANCOVA — l'hypothèse de pentes parallèles**

> ⚠️ **Hypothèse implicite forte.** Le modèle suppose que **la pente $\beta_1$ est la même pour tous les groupes**. Autrement dit : un cheval supplémentaire a *exactement* le même impact sur la mpg qu'on soit en USA, Europe ou Japon. Cette hypothèse n'est pas toujours valide — il se peut très bien que l'effet de hp dépende lui-même de l'origine.

> [!note]- Et si les pentes diffèrent selon les groupes ? — termes d'interaction
> Pour autoriser des **pentes différentes par groupe**, on rajoute des **termes d'interaction** entre la dummy et le continu :
> 
> $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Eur}} + \beta_3 \mathbb{1}_{\text{Jap}} + \beta_4 (\text{hp} \cdot \mathbb{1}_{\text{Eur}}) + \beta_5 (\text{hp} \cdot \mathbb{1}_{\text{Jap}})$
> 
> Lecture :
> - $\beta_1$ : pente de hp **pour le groupe de référence** (USA)
> - $\beta_4$ : **différence de pente** entre Europe et USA — combien de mpg en plus/moins par cheval supplémentaire en Europe par rapport à USA
> - $\beta_5$ : différence de pente entre Japon et USA
> 
> Géométriquement : on n'a plus trois droites parallèles, mais trois droites avec des **pentes différentes**. Si $\beta_4 = \beta_5 = 0$, on retrouve le modèle ANCOVA. Tester $H_0 : \beta_4 = \beta_5 = 0$ via un test de Fisher (cf. III.E) revient à tester *"l'effet de hp est-il vraiment le même dans tous les groupes ?"*.

> 💡 **À retenir pour les entretiens.** Devant un modèle mixte, lire les coefficients dans l'ordre : **(1) intercept** = baseline du groupe de référence à covariables nulles, **(2) pentes des continus** = effet marginal à autres variables fixées, **(3) dummies** = écart de niveau entre groupes à covariables fixées. La géométrie sous-jacente est *toujours* un faisceau de droites — parallèles si pas d'interactions, divergentes si interactions.


![[Pasted image 20260505220100.png]]

**Figure 8.** Modèle avec interactions ```mpg ~ horsepower * origin```. Trois pentes différentes par groupe (au lieu d'une pente commune comme dans la Figure 7). Les coefficients $\beta_4, \beta_5$ mesurent les différences de pente par rapport au groupe de référence USA. Tester leur nullité jointe (test de Fisher) revient à tester l'hypothèse de pentes parallèles de l'ANCOVA.

### F. Standardisation des prédicteurs

Faut-il standardiser (z-score) les prédicteurs avant de fitter une régression ? La réponse dépend de ce qu'on veut faire — mais en pratique en quant, **standardiser les continus par défaut** est la règle qui évite tous les pièges.

**(i) OLS pur est invariant à la mise à l'échelle**

> [!warning] Invariance d'OLS
> Si on multiplie une colonne $X_j$ par une constante $c$, le coefficient $\hat{\beta}_j$ est divisé par $c$, et **les prédictions $\hat{y}$ restent identiques**. Le R², les résidus, et les p-values des tests Student individuels sont aussi invariants.

> [!note]- Démonstration
> Soit $\tilde{X} = XD$ avec $D$ matrice diagonale de mise à l'échelle. Alors :
> 
> $\hat{\beta}_{\tilde{X}} = (\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T y = (D X^T X D)^{-1} D X^T y = D^{-1} (X^T X)^{-1} X^T y = D^{-1} \hat{\beta}_X$
> 
> Donc $\tilde{X} \hat{\beta}_{\tilde{X}} = X D \cdot D^{-1} \hat{\beta}_X = X \hat{\beta}_X = \hat{y}$. Les prédictions sont strictement identiques.

Conséquence : pour OLS sans régularisation, **standardiser ne change rien au modèle** — on peut techniquement s'en passer. Mais ce n'est pas pour autant qu'il faut éviter de standardiser : dans les 4 cas suivants, c'est obligatoire ou très recommandé.

**(ii) Quand il faut standardiser**

> [!warning] Cas 1 — Régularisation (Ridge, Lasso, Elastic Net)
> Les pénalités $\lambda \sum \beta_j^2$ et $\lambda \sum |\beta_j|$ traitent **tous les coefficients sur un pied d'égalité**. Si `weight` est en kg (coefs $\sim 0.01$) et `revenue` en dollars (coefs $\sim 10^{-7}$), la pénalité écrase les petits coefs et ignore les gros — un résultat **entièrement déterminé par les unités arbitraires des features**, pas par leur importance réelle.
> 
> > ⚠️ **L'erreur classique** : utiliser `sklearn.linear_model.Ridge()` ou `Lasso()` sans `StandardScaler()` au préalable. Les résultats deviennent essentiellement absurdes. Toujours standardiser avant régularisation.

> [!warning] Cas 2 — Comparer l'importance des coefficients
> Pour dire *"quel prédicteur a l'effet le plus fort sur Y ?"*, comparer $\hat{\beta}_1 = 0.3$ et $\hat{\beta}_2 = 100$ ne veut rien dire si $X_1$ est en mètres et $X_2$ en millimètres. Après standardisation, on obtient les **coefficients standardisés** :
> 
> $\beta_j^{\text{std}} = \hat{\beta}_j \cdot \frac{\sigma_{X_j}}{\sigma_Y}$
> 
> Lecture sans dimension : *"une augmentation d'**un écart-type** de $X_j$ correspond à $\beta_j^{\text{std}}$ écarts-types de $Y$"*. C'est la version comparable entre features.

> [!warning] Cas 3 — Stabilité numérique
> Si on inclut des termes polynomiaux (`hp` et `hp²`), des interactions, ou des features sur des échelles très différentes, $X^T X$ devient mal conditionnée — **numériquement instable** même si mathématiquement inversible. Standardiser réduit le conditionnement et stabilise l'inversion.

> [!warning] Cas 4 — Optimisation par descente de gradient
> Pour les méthodes itératives (SGD, Adam, et toute la machinerie deep learning), des features sur des échelles différentes créent des **vallées étroites** dans la loss. La descente oscille au lieu d'aller droit au minimum. Standardiser rend la loss quasi-isotrope → convergence beaucoup plus rapide.

**(iii) La règle pratique**

> 💡 **Règle par défaut en quant.** Standardiser **toutes les features continues** (z-score : centré, divisé par l'écart-type), laisser **les dummies telles quelles**, l'intercept se gère tout seul. Pour le report final en unités d'origine, dé-standardiser le coef à la main : $\hat{\beta}_{\text{raw}} = \hat{\beta}_{\text{std}} \cdot \sigma_Y / \sigma_{X_j}$.

> [!note]- Pourquoi ne pas standardiser les dummies ?
> Standardiser une dummy 0/1 lui fait perdre son interprétation naturelle ($\beta_k$ = écart de moyenne entre groupes, $\beta_k$ = factor return en %, etc.). Après standardisation, le coefficient devient "l'effet d'une augmentation d'un écart-type de la dummy" — ce qui n'a aucun sens économique. **Les dummies se laissent toujours en 0/1.**

> [!example] En Barra — standardisation différenciée
> Le modèle Barra applique exactement cette règle :
> - **Style factors continus** (Value, Momentum, Size...) → **z-scorés cross-sectionnellement chaque jour** → distribution centrée réduite à chaque date $t$ sur l'estimation universe
> - **Industry/Country dummies** → **laissées en 0/1**
> 
> Conséquence pour l'interprétation des factor returns :
> - $f_{\text{Value},t}$ se lit comme *"rendement d'un portefeuille long les stocks à +1$\sigma$ de Value, short les stocks à $-1\sigma$"* — le z-scoring rend ce factor return comparable à ceux des autres styles (Momentum, Quality, etc.)
> - $f_{\text{Tech},t}$ se lit comme *"rendement du secteur Tech vs marché en %"* — unité absolue, pas en écarts-types, parce que l'appartenance à un secteur est binaire
> 
> > 💡 **À retenir.** Cette différence de traitement (z-score les continus, laisser les dummies) est la raison pour laquelle on peut **comparer entre eux les style factors** ("Momentum a fait +30 bps, Value a fait −20 bps") tout en gardant une **interprétation économique directe** des factor returns sectoriels ("Tech a outperformé de +2%").


---

## II. Diagnostic

On généralise ici au cas de la **régression linéaire multiple** : on observe $n$ couples $(x_i, y_i)$ avec $x_i \in \mathbb{R}^p$, et le modèle devient :

$$y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \varepsilon_i$$

La régression simple est le cas particulier $p = 1$. Tout ce qui suit s'applique aux deux.

### A. Les hypothèses de Gauss-Markov

> [!warning] Les cinq hypothèses de Gauss-Markov
> Pour que MCO soit un "bon" estimateur, on a besoin de cinq hypothèses sur les résidus $\varepsilon_i$ :
> 
> - **H1 — Linéarité.** La relation entre $Y$ et les $X_j$ est bien linéaire.
> - **H2 — Exogénéité.** $E(\varepsilon_i \mid X) = 0$ — les résidus ne sont pas corrélés avec les prédicteurs.
> - **H3 — Homoscédasticité.** $V(\varepsilon_i) = \sigma^2$ — tous les résidus ont la même variance.
> - **H4 — Indépendance.** Les $\varepsilon_i$ sont indépendants entre eux — pas d'autocorrélation.
> - **H5 — Absence de multicolinéarité parfaite.** Les colonnes de $X$ ne sont pas combinaisons linéaires les unes des autres.
> 
> On ajoutera plus loin une **6ème hypothèse** (H6, normalité des résidus) qui n'est pas nécessaire pour BLUE mais qui sert pour l'inférence exacte.

> 💡 **Hiérarchie des violations.** Les 5 hypothèses ne sont pas équivalentes en termes de gravité. **H1 et H2** affectent l'estimateur lui-même : si elles sont violées, $\hat{\beta}$ devient **biaisé** — il pointe au mauvais endroit. **H3, H4, H5** affectent seulement la précision et l'inférence : $\hat{\beta}$ reste sans biais, mais les écart-types calculés par OLS sont faux, donc les tests sont invalides. À retenir pour les entretiens : seules H1 et H2 biaisent l'estimateur.

> [!example] Fil rouge — dataset Auto
> Pour illustrer les hypothèses on utilise le dataset `Auto` (ISLR). Variable cible : `mpg` (miles per gallon, consommation inverse). Prédicteur principal : `horsepower` (puissance du moteur en chevaux). 392 observations sur des modèles de voitures 1970-1982.
> 
> ![[Pasted image 20260505183850.png|477]]
> 
> **Figure 9.** Scatter plot brut `mpg ~ horsepower`. La relation est clairement décroissante (plus puissant → plus gourmand → moins de mpg) mais visiblement courbée et la dispersion change avec le niveau.

#### H1 — Linéarité

> [!warning] Définition (Linéarité)
> La vraie relation entre $Y$ et les $X_j$ est linéaire :
> 
> $$Y = \beta_0 + \sum_{j=1}^p \beta_j X_j + \varepsilon$$
> 
> Aucune courbure, aucun terme d'interaction caché, aucune transformation nécessaire.

> 💡 **Conséquence d'une violation.** Si la vraie relation est non-linéaire, la droite OLS passe en moyenne mais rate la courbure. $\hat{\beta}$ est **biaisé** : on estime un effet linéaire qui n'existe pas vraiment. C'est l'une des deux pathologies (avec H2) qui faussent l'estimateur lui-même.

**Diagnostic — le residual plot.** Le test visuel de référence : on plot $\hat{y}_i$ en x-axis et les résidus $\varepsilon_i = y_i - \hat{y}_i$ en y-axis. Si le modèle est bien spécifié, les résidus doivent former un nuage aléatoire centré sur 0. **Toute structure visible révèle une violation.** En particulier, une **courbure en U** signe la non-linéarité.

> [!note]- Pourquoi le residual plot fonctionne — lien avec la vue $L^2$
> Sous les hypothèses, les résidus sont **non corrélés** à $\hat{y}$ et à toute fonction de $X$ (orthogonalité, cf. section I.B). Tout pattern visible — courbure, cône, tendance — révèle donc une violation. Le residual plot est l'outil graphique le plus universel parce qu'il marche en régression simple comme en régression multiple : $\hat{y}$ agrège l'effet de tous les prédicteurs.

> [!example] Auto — courbure en U claire
> Sur Auto avec un fit linéaire `mpg ~ horsepower`, le residual plot révèle une courbure en U marquée :
> 
> ![[Pasted image 20260505184015.png|492]]
> 
> **Figure 10.** Residual plot pour `mpg ~ horsepower`. Pour les valeurs prédites faibles ($\hat{y} \in [5, 12]$) et élevées ($\hat{y} \in [28, 35]$), les résidus sont systématiquement positifs ; pour les valeurs moyennes, ils sont négatifs. Cette **structure systématique** (et non aléatoire) signe la non-linéarité.
> 
> **Lecture économique.** La vraie relation entre `horsepower` et `mpg` est convexe (rendements marginaux décroissants : passer de 50 à 100 ch coûte beaucoup, passer de 200 à 250 ch coûte peu). Une droite ne peut pas capturer cette courbure.
> 
> **Remède.** Ajouter un terme polynomial : $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \text{hp}^2$. Le residual plot devient alors quasi plat (la courbure disparaît).

#### H2 — Exogénéité

> [!warning] Définition (Exogénéité)
> Les résidus sont en moyenne nuls conditionnellement aux prédicteurs :
> 
> $$E(\varepsilon_i \mid X) = 0$$
> 
> Cela implique deux choses :
> - $E(\varepsilon_i) = 0$ — les résidus sont centrés
> - $\text{cov}(\varepsilon_i, X_j) = 0$ pour tout $j$ — les résidus ne sont corrélés à aucun prédicteur

> 💡 **L'hypothèse la plus grave à violer.** Si exogénéité est violée, $\hat{\beta}$ est **biaisé** — pas juste les écart-types, le coefficient lui-même pointe au mauvais endroit. Pire, le biais ne disparaît pas avec $n \to \infty$ (l'estimateur est **inconsistant**). C'est la pathologie centrale de l'économétrie causale moderne.

> [!note]- Démo du biais
> La formule OLS donne (avec $\beta$ vrai et $\hat{\beta}$ estimé) :
> 
> $$\hat{\beta} = \beta + (X^T X)^{-1} X^T \varepsilon$$
> 
> En espérance :
> 
> $$E[\hat{\beta}] = \beta + (X^T X)^{-1} X^T E[\varepsilon \mid X]$$
> 
> Si $E[\varepsilon \mid X] = 0$ (exogénéité) → $E[\hat{\beta}] = \beta$, sans biais.
> Si $E[\varepsilon \mid X] \neq 0$ → terme de biais qui **ne disparaît jamais**, même avec $n \to \infty$.

**Trois sources classiques de violation** :

> [!example] Source 1 — Variable omise corrélée à $X$
> Le cas le plus fréquent. Tu omets une variable qui (1) influence $Y$ et (2) est corrélée à $X$. Son effet se cache dans $\varepsilon$, et $\varepsilon$ devient corrélé à $X$.
> 
> *Exemple canonique* : régression `salaire ~ éducation`. L'**intelligence** est corrélée à l'éducation (les gens intelligents font plus d'études) et influence le salaire (à éducation égale, ils gagnent plus). L'intelligence se retrouve donc dans $\varepsilon$, qui devient corrélé à éducation. Conséquence : $\hat{\beta}_1$ surestime le rendement de l'éducation — il capte aussi l'effet de l'intelligence.

> [!example] Source 2 — Causalité inverse / simultanéité
> $X$ et $Y$ se déterminent mutuellement. Tu régresses $Y$ sur $X$, mais $Y$ influence aussi $X$.
> 
> *Exemple* : régression `quantité ~ prix` en économie. Prix et quantité sont co-déterminés par l'équilibre offre-demande — on ne peut pas régresser naïvement l'un sur l'autre. Autre exemple : `performance ~ qualité du management`, mais une bonne performance attire de bons managers (boucle de rétroaction).

> [!example] Source 3 — Erreur de mesure sur $X$
> Si $X$ observé = $X$ vrai + bruit, alors le bruit de mesure se retrouve mécaniquement dans $\varepsilon$ et est corrélé à $X$ observé.
> 
> *Conséquence connue* : **biais d'atténuation** — $\hat{\beta}$ est tiré vers 0. Plus l'erreur de mesure est grande, plus le coefficient est sous-estimé.

**Diagnostic.** Pas de plot magique ici. Contrairement à H1, H3 ou H4, on ne peut pas détecter l'endogénéité visuellement — par construction, OLS impose $\text{cov}(\hat{\varepsilon}, X) = 0$ dans l'échantillon. Le diagnostic se fait par **raisonnement causal** : quelles variables que je n'ai pas mesurées pourraient influencer $Y$ et être corrélées à $X$ ? Y a-t-il une boucle $Y \to X$ ?

**Remèdes.** Tout un champ de l'économétrie moderne : variables instrumentales (IV), expériences naturelles, différence-en-différence (DiD), régression discontinue. Voir [[Inférence causale]].

#### H3 — Homoscédasticité

> [!warning] Définition (Homoscédasticité)
> Tous les résidus ont la même variance, indépendamment de $X$ :
> 
> $$V(\varepsilon_i) = \sigma^2 \quad \forall i$$
> 
> La violation s'appelle **hétéroscédasticité** : $V(\varepsilon_i) = \sigma_i^2$ dépend de $i$ (typiquement de $X$).

> 💡 **Conséquences.** $\hat{\beta}$ reste **sans biais** mais (1) il n'est plus efficace (WLS fait mieux, cf. section IV), (2) les écart-types calculés par OLS sont faux → tests invalides (typiquement on rejette $H_0$ trop facilement, faux positifs partout).

**Diagnostic — le cône.** Sur le residual plot, l'hétéroscédasticité se voit comme un **évasement** de la dispersion verticale des résidus quand $\hat{y}$ varie. Cas typique : cône qui s'ouvre vers la droite → variance qui croît avec $\hat{y}$.

**Remèdes** : (a) **WLS** avec poids $w_i \propto 1/\sigma_i^2$ (cf. section IV), (b) **écart-types robustes** (Huber-White), (c) **transformation de $Y$** (ex: $\log$).

> [!example] Auto — courbure ET cône simultanément
> Sur Auto, le residual plot de `mpg ~ horsepower` présente **deux pathologies en même temps** : la courbure en U (H1, déjà vue) et un cône (H3).
> 
> ![[Pasted image 20260505184700.png]]
> 
> **Figure 11.** Residual plot avec enveloppe ±2σ local pour `mpg ~ horsepower`. À gauche le modèle linéaire montre courbure + cône. À droite après ajout de $\text{horsepower}^2$ pour corriger H1 : la courbure disparaît mais **le cône reste**.
> 
> **Le point pédagogique fort.** H1 et H3 sont **deux pathologies indépendantes**. Corriger l'une ne corrige pas l'autre. Sur Auto, après avoir ajouté le terme quadratique, il faut encore traiter l'hétéroscédasticité (par exemple via WLS ou écart-types robustes) si on veut faire de l'inférence valide.

#### H4 — Indépendance

> [!warning] Définition (Indépendance)
> Les résidus sont indépendants entre eux :
> 
> $$\text{cov}(\varepsilon_i, \varepsilon_j) = 0 \quad \forall i \neq j$$
> 
> Connaître $\varepsilon_i$ ne donne aucune information sur $\varepsilon_j$.

L'hypothèse d'indépendance est violée dès qu'il y a **une structure** qui crée des corrélations entre observations. Quatre cas typiques :

- **Temporelle** (autocorrélation) — séries temporelles où $\varepsilon_t$ dépend de $\varepsilon_{t-1}$. Diagnostic : plot des résidus dans l'ordre chronologique, autocorrélogramme, test Durbin-Watson.
- **Clustering** (par groupe) — observations dans des groupes naturels (école, secteur, origine). Au sein de chaque groupe les résidus se ressemblent.
- **Spatiale** — observations géographiquement proches corrélées (climat, démographie locale).
- **Panel / longitudinale** — combinaison temporelle + clustering : on suit les mêmes individus dans le temps. Double dépendance (au sein d'un individu à travers le temps, et entre individus à un instant donné).

> 💡 **Conséquences.** Comme H3, $\hat{\beta}$ reste sans biais mais les écart-types sont faux — typiquement **sous-estimés**, ce qui mène à du faux significatif. Si tu as 1000 observations regroupées en 50 écoles, ta vraie taille effective est plus proche de 50 que 1000.

> [!note]- Distinction H3 vs H4 — piège classique
> Sur un boxplot des résidus par groupe, les deux pathologies se voient mais à **différents endroits** :
> 
> - **H3 (hétéroscédasticité)** = la **variance** des résidus change selon les groupes. Les boîtes ont des **tailles** différentes (largeur/hauteur). Mais elles peuvent rester centrées sur 0.
> - **H4 (clustering)** = la **moyenne** des résidus est différente selon les groupes. Les boîtes ont des **médianes** décalées (au-dessus ou en-dessous de 0). Mais elles peuvent avoir la même taille.
> 
> H3 = "tailles différentes". H4 = "médianes décalées". Les deux peuvent coexister.

**Remèdes** :
- Cas temporel : Newey-West (écart-types corrigés), modèles ARIMA/GLS (cf. section IV).
- Cas clustering : **inclure la variable de groupe dans le modèle** (effets fixes), ou **cluster-robust standard errors**.
- Cas spatial : modèles spatiaux (SAR, SEM).

> [!example] Auto — clustering par origine
> Sur Auto, il existe une variable `origin` (USA, Europe, Japon). Si on régresse `mpg ~ horsepower` **sans** inclure l'origine, on observe que les résidus sont systématiquement biaisés par groupe.
> 
> Moyennes des résidus par origine (devrait être ~0 si H4 OK) :
> 
> | Origine | Moyenne | Écart-type | Effectif |
> |---|:---:|:---:|:---:|
> | Europe | $+0.383$ | $4.997$ | $68$ |
> | Japon | $+3.116$ | $4.684$ | $79$ |
> | USA | $-1.111$ | $4.491$ | $245$ |
> 
> Les voitures japonaises ont des résidus systématiquement positifs (à puissance égale, elles consomment moins — meilleure technologie, plus légères) et les américaines des résidus négatifs.
> 
> ![[Pasted image 20260505190506.png]]
> 
> **Figure 12.** Boxplot des résidus par origine pour `mpg ~ horsepower`. **Avant** inclusion de l'origine, les médianes sont nettement décalées par groupe — signature classique du clustering.
> 
> **Remède : inclure `origin` dans le modèle.** On passe de $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp}$ à $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Europe}} + \beta_3 \mathbb{1}_{\text{Japon}}$ (USA = référence, cf. section I.E sur les dummies).
> 
> ![[Pasted image 20260505191012.png]]
> 
> **Figure 13.** Comparaison avant/après. **Gauche** : sans `origin`, médianes décalées. **Droite** : avec `origin` inclus, les trois boîtes sont recentrées sur 0 — la violation H4 a disparu.
> 
> > 💡 **À retenir.** L'effet "origine" est passé du résidu vers le modèle. La même information a juste migré : $\hat{\beta}_{\text{Japon}} \approx +2$ capture ce que les résidus japonais portaient avant. **La violation H4 par clustering n'est PAS une propriété intrinsèque des données, c'est une propriété du modèle qu'on a choisi.** En incluant la variable de groupe, on l'élimine à la racine.

> [!note]- Lien profond H4 ↔ H2
> Le clustering est en fait **une variable omise déguisée**. "Mes résidus sont clusterisés par groupe" = "j'ai omis une variable de groupe corrélée à $Y$". Inclure la variable revient à la transférer du résidu vers le modèle. Le seul cas où H4 est **structurellement** violée (non réductible à une variable omise) est l'autocorrélation temporelle pure : $\varepsilon_t = \phi \varepsilon_{t-1} + u_t$ ne peut pas être "inclus" dans le modèle puisque $\varepsilon_{t-1}$ est inobservable.

#### H5 — Multicolinéarité

> [!warning] Définition (Multicolinéarité)
> Les colonnes de $X$ ne sont pas combinaisons linéaires les unes des autres. Si une colonne s'écrit comme combinaison des autres (par exemple $X_3 = X_1 + X_2$), $X^\top X$ n'est pas inversible et la formule MCO ne marche plus — c'est la **multicolinéarité parfaite**.

**Le problème (cas non parfait).** Même sans être parfaite, une corrélation forte entre prédicteurs cause des problèmes. Si $X_1$ et $X_2$ pointent presque dans la même direction dans $L^2$, projeter $Y$ dessus devient ambigu — une infinité de combinaisons $(\beta_1, \beta_2)$ donnent à peu près la même projection.

> 💡 **À retenir.** Le modèle **sait prédire** mais il **ne sait pas répartir les coefficients**. Conséquence : les variances de $\hat{\beta}_j$ explosent, les coefficients deviennent instables et changent radicalement quand on ajoute/retire d'autres variables corrélées.

![Multicolinéarité](images/2-Statistiques/A_Frequentist/regression-lineaire/im3.png)

**Figure 14.** Multicolinéarité : quand $X_1$ et $X_2$ pointent dans la même direction, projeter $Y$ dessus devient ambigu — les coefficients $\beta_1$ et $\beta_2$ ne sont plus identifiables individuellement.

> [!warning] Diagnostic — le VIF (Variance Inflation Factor)
> $$\text{VIF}_j = \frac{1}{1 - R^2_j}$$
> 
> où $R^2_j$ est le $R^2$ de la régression de $X_j$ sur tous les autres prédicteurs. Si $X_j$ est parfaitement expliqué par les autres, $R^2_j \to 1$ et $\text{VIF}_j \to \infty$.
> 
> **Règle empirique** : $\text{VIF}_j > 10$ → signal d'alarme.

**Remèdes** : (a) supprimer une variable redondante, (b) **Ridge** (pénalité L2 qui stabilise les coefficients au prix d'un léger biais — voir section V), (c) **Lasso** (sélection automatique), (d) PCA.

> [!example] Auto — quatre variables qui mesurent la même chose
> Sur Auto, plusieurs prédicteurs mesurent essentiellement la même chose ("taille/puissance du moteur") : `horsepower`, `weight`, `displacement`, `cylinders`. Logique : un gros moteur = plus de chevaux + plus lourd + plus de cylindrée + plus de cylindres. Tout est la même variable latente "grosse caisse".
> 
> ![[Pasted image 20260505191355.png|438]]
> 
> **Figure 15.** Heatmap des corrélations entre prédicteurs sur Auto. Les 4 premières variables (`horsepower`, `weight`, `displacement`, `cylinders`) forment un bloc fortement corrélé (toutes les corrélations $> 0.84$).
> 
> **VIF associés** :
> 
> | Variable | VIF | Statut |
> |---|:---:|:---:|
> | horsepower | $8.92$ | OK (limite) |
> | weight | $10.43$ | ⚠️ alarme |
> | displacement | $19.54$ | ⚠️ alarme forte |
> | cylinders | $10.63$ | ⚠️ alarme |
> | acceleration | $2.61$ | OK |
> 
> **Instabilité du coefficient sur `horsepower`** selon le modèle :
> 
> | Modèle | $\hat{\beta}_{\text{horsepower}}$ |
> |---|:---:|
> | `mpg ~ hp` | $-0.158$ |
> | `mpg ~ hp + weight` | $-0.047$ |
> | `mpg ~ hp + weight + displacement + cylinders` | $-0.043$ |
> 
> Le coefficient passe de $-0.16$ à $-0.04$ quand on ajoute les variables corrélées. Avec d'autres datasets ou échantillons, il pourrait même **changer de signe** — ce qui est physiquement absurde (plus puissant ne peut pas faire baisser la consommation). Symptôme classique de multicolinéarité.
> 
> > 💡 **À retenir.** Le modèle 3 prédit `mpg` **aussi bien** que le modèle 1 (mêmes $\hat{y}$ à peu près). Mais les coefficients individuels deviennent ininterprétables. Le modèle prédit bien mais ne sait pas répartir l'effet "grosse caisse" entre les 4 variables corrélées. Si l'objectif est la prédiction, la multicolinéarité n'est pas dramatique. Si l'objectif est l'**interprétation des coefficients**, c'est un vrai problème.


#### H6 — Hypothèse de Normalité

Gauss-Markov (H1–H5) ne suppose **pas** la gaussianité — OLS est BLUE sans elle. Mais une 6ème hypothèse implicite circule dans toute la note : on l'a vue en I.C (vue probabiliste, OLS = MLE) et en III.A (distribution exacte de $\hat{\beta}$). Voilà son statut précis.

> [!warning] H6 — Normalité des résidus
> Les résidus suivent une loi normale : $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$
> 
> Cette hypothèse vient **en supplément** des cinq hypothèses de Gauss-Markov. Elle n'est pas requise pour BLUE, mais devient nécessaire pour avoir des **tests Student et Fisher exacts**.

> 💡 **Quel rôle joue-t-elle vraiment ?**
> - **Sans H6, avec H1–H5** : OLS reste BLUE (Gauss-Markov), mais les tests Student/Fisher deviennent seulement *asymptotiquement* valides — c'est le TCL qui sauve la mise quand $n$ est grand.
> - **Avec H6** : $\hat{\beta}$ est **exactement** gaussien à n'importe quelle taille d'échantillon, OLS coïncide avec MLE (cf. I.C), et les tests Student/Fisher sont **exacts** (pas seulement asymptotiques).

> 💡 **Conséquence pratique d'une violation.** Sur **grand échantillon** (n > 30–50 typiquement), H6 n'est pas critique — le TCL rend les tests robustes à la non-normalité. Sur **petit échantillon**, les p-values et IC peuvent être incorrects.

**(a) Diagnostic — quatre outils**

Contrairement aux autres hypothèses, la normalité a des outils dédiés très visuels.

**(i) Le QQ-plot.** On classe les résidus standardisés et on les compare aux quantiles théoriques d'une $\mathcal{N}(0, 1)$. Si les points suivent la **bissectrice**, les résidus sont gaussiens. Déviations classiques :
- Points qui s'écartent vers le haut à droite et vers le bas à gauche → **queues lourdes** (kurtosis > 3)
- Courbure en S → **asymétrie** (skewness $\neq$ 0)

**(ii) Le test de Jarque-Bera.** Combine asymétrie (skew) et aplatissement (kurtosis) en une statistique $JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4})$. Sous $H_0$ de normalité, $JB \sim \chi^2_2$. Apparaît dans le summary statsmodels sous `Jarque-Bera (JB)` et `Prob(JB)`.

**(iii) Le test omnibus de D'Agostino-Pearson.** Autre combinaison skew + kurtosis avec une normalisation différente. C'est l'`Omnibus` du summary statsmodels.

**(iv) Le test de Shapiro-Wilk.** Le plus puissant pour petit échantillon ($n < 50$). Pas dans le summary statsmodels par défaut, à appeler via `scipy.stats.shapiro`.

> [!note]- Skewness et kurtosis — les indicateurs bruts
> Le summary statsmodels affiche directement deux mesures de forme :
> - **Skewness** $S = E[(\frac{\varepsilon - \mu}{\sigma})^3]$ : asymétrie. $S = 0$ pour une gaussienne, $S > 0$ = queue à droite, $S < 0$ = queue à gauche.
> - **Kurtosis** $K = E[(\frac{\varepsilon - \mu}{\sigma})^4]$ : aplatissement. $K = 3$ pour une gaussienne, $K > 3$ = queues lourdes (leptokurtique), $K < 3$ = queues légères (platikurtique).
> 
> JB et Omnibus sont juste des combinaisons formelles de ces deux nombres en un seul test.

**(b) Application sur Auto**

Sur le modèle `mpg ~ horsepower`, le summary affiche :
- `Skew = 0.49` (devrait être ~0)
- `Kurtosis = 3.30` (devrait être ~3, OK ici)
- `Jarque-Bera (JB) = 17.3, Prob(JB) = 0.0002`
- `Omnibus = 16.4, Prob(Omnibus) < 0.001`

→ on rejette la normalité au seuil 5% sur les deux tests.

> ⚠️ **Causalité — la non-normalité est souvent un *symptôme*, pas le vrai problème.** Sur Auto, la non-normalité observée vient en réalité de la **mauvaise spécification** du modèle (courbure en U, H1 violée). Quand on corrige H1 en ajoutant `horsepower²`, les résidus deviennent beaucoup plus proches d'une gaussienne. **Avant de "corriger H6", toujours vérifier H1 et H2 d'abord.** La normalité est rarement le problème de fond.

**(c) Remèdes**

Si H6 est vraiment violée et qu'on tient à l'inférence exacte :
- **Grand échantillon** : ne rien faire, le TCL gère
- **Petit échantillon** : transformation de $Y$ (log, Box-Cox), bootstrap pour les IC, ou tests non-paramétriques
- **Avant tout** : vérifier H1 et H2 — la non-normalité est souvent un symptôme d'une autre violation

**(d) Récapitulatif H1–H6**

Voilà la hiérarchie complète des hypothèses :

| Hypothèse | Pour quoi est-elle nécessaire ? | Diagnostic |
|---|---|---|
| H1 (linéarité) | Estimateur sans biais | residual plot |
| H2 (exogénéité) | Estimateur sans biais | raisonnement causal |
| H3 (homoscédasticité) | Variance OLS correcte → tests valides | residual plot (cône) |
| H4 (indépendance) | Variance OLS correcte → tests valides | Durbin-Watson, boxplot par groupe |
| H5 (multicolinéarité) | $X^TX$ inversible, $\hat{\beta}$ stables | VIF |
| **H6 (normalité)** | **Tests Student/Fisher exacts (sinon asymptotiques)** | **QQ-plot, JB, Omnibus, Shapiro-Wilk** |

> 💡 **À retenir.** H1–H5 sont les hypothèses **du modèle** ; H6 est l'hypothèse **pour l'inférence exacte**. Sur grand échantillon, H6 devient optionnelle grâce au TCL.

### B. BLUE — ce que ça veut dire

> [!warning] Théorème de Gauss-Markov
> Sous H1–H5, l'estimateur MCO est **BLUE** : *Best Linear Unbiased Estimator*.
> 
> - **Linéaire** — $\hat{\beta}$ est une fonction linéaire des observations $y_i$
> - **Non biaisé** — $E[\hat{\beta}] = \beta$, i.e. $\text{bias}(\hat{\beta}) = 0$
> - **Best** — parmi tous les estimateurs linéaires non biaisés, MCO a la **variance minimale** : $V(\hat{\beta})$ est la plus petite possible

> 💡 **L'intuition.** On ne peut pas faire mieux que MCO sans soit introduire du biais, soit sortir de la classe des estimateurs linéaires. C'est ce qui justifie qu'OLS soit la méthode de référence — tant que les hypothèses tiennent, aucun autre estimateur linéaire non biaisé n'est plus précis.


### C. Observations influentes

Les hypothèses Gauss-Markov sont des hypothèses sur le **processus** générateur des données. Mais il existe une autre classe de problèmes, propre à l'**échantillon** : certaines observations individuelles peuvent disproportionnellement influencer $\hat{\beta}$. Ce n'est pas une violation d'hypothèse, c'est un problème de robustesse.

> [!warning] Trois objets distincts à ne pas confondre
> - **Outlier** — observation avec un **résidu** inhabituellement grand. Le point est loin de la droite ajustée. Détecté par : résidus standardisés $|r_i| > 2$ ou $3$.
> - **High leverage point** — observation avec une valeur **inhabituelle de $X$** (loin du centre du nuage). Aucun jugement sur le résidu. Détecté par le terme $h_{ii}$ de la **hat matrix** $H = X(X^T X)^{-1} X^T$.
> - **Observation influente** — observation qui change beaucoup $\hat{\beta}$ si on la retire. C'est la combinaison des deux.

> 💡 **Le seul cas vraiment dangereux : outlier ET high leverage simultanément.** Un outlier seul (X normal mais Y aberrant) a peu d'effet sur la pente, juste sur les écart-types. Un high leverage seul (X extrême mais sur la droite) ne pose pas de problème. Mais un point à la fois extrême en $X$ ET très éloigné de la tendance va **tirer la droite** vers lui — son grand bras de levier amplifie l'erreur.

> [!warning] Distance de Cook — la métrique synthétique
> Mesure de combien $\hat{\beta}$ changerait si on retirait l'observation $i$ :
> 
> $$D_i = \frac{r_i^2}{p+1} \cdot \frac{h_{ii}}{(1-h_{ii})^2}$$
> 
> où $r_i$ est le résidu standardisé et $h_{ii}$ le leverage. Combine les deux : grand résidu × grand leverage = grande distance de Cook.
> 
> **Règle empirique** : $D_i > 1$ ou $D_i > 4/n$ → observation à examiner.

**En pratique** :
1. Calculer $D_i$ pour toutes les observations
2. Identifier les points avec $D_i$ élevé
3. Pour chaque point suspect : (a) erreur de saisie ? → corriger, (b) donnée légitime mais inhabituelle ? → garder mais signaler la sensibilité, (c) vraiment aberrant ? → discuter le retrait

> [!note]- Pourquoi distinguer les hypothèses Gauss-Markov des observations influentes ?
> Gauss-Markov dit "si les hypothèses sont vraies sur le processus, OLS est BLUE". Le théorème ne dit rien sur la **stabilité** de l'estimateur face à des observations individuelles. Tu peux avoir un estimateur BLUE qui est très sensible à un seul point. C'est une question orthogonale aux hypothèses du modèle.

> [!example] Données factices — outlier vs high leverage
> On simule un dataset propre $y = 2 + 3x + \varepsilon$ avec $\varepsilon \sim \mathcal{N}(0, 2^2)$, puis on ajoute deux points pathologiques :
> - **Obs 50** — outlier seul : $X$ au milieu du nuage (X = 5), mais $Y$ aberrant ($Y = 35$ alors que la droite prédit $\sim 17$).
> - **Obs 51** — high leverage seul : $X$ extrême (X = 20, loin du nuage qui s'arrête à 10), mais $Y$ exactement sur la droite.
> 
> ![[Pasted image 20260505202556.png]]
> 
> **Figure 16.** **Gauche** : scatter brut. L'obs 50 (rouge) est un outlier visible au milieu du nuage en hauteur. L'obs 51 (orange) est extrême en X mais sur la droite OLS. **Droite** : plot diagnostic Leverage vs Studentized Residuals. L'obs 50 sort par l'**axe Y** (résidu standardisé > 3) → détecté comme outlier. L'obs 51 sort par l'**axe X** (leverage $h$ au-delà du seuil $3(p+1)/n$) → détecté comme high leverage. Aucun des deux n'est dans un coin (haut-droite ou bas-droite) → ni l'un ni l'autre n'est réellement *influent* sur $\hat{\beta}$.
> 
> > 💡 **Le plot de droite est universel.** En régression simple ($p=1$), tu peux voir un point extrême en $X$ directement sur le scatter. Mais en régression multiple ($p$ grand), le scatter 2D ne suffit plus — un point peut être extrême dans une **combinaison** des variables sans l'être sur aucune individuellement (genre 1m95 + 60kg). Le plot Leverage vs Residuals reste lisible à n'importe quelle dimension parce que $h_{ii}$ et $r^*_i$ sont des scalaires par observation. C'est le seul outil pour détecter ces points cachés en multi-D.

---

## III. Inférence & Tests

> Une fois le modèle estimé et diagnostiqué, trois questions se posent naturellement, dans cet ordre :
> 1. **Le modèle marche-t-il globalement ?** Est-ce qu'il vaut mieux que prédire la moyenne ?
> 2. **Quels coefficients comptent vraiment ?** Lesquels sont significatifs, individuellement ou jointement ?
> 3. **Avec quelle précision peut-on prédire ?** Pour une nouvelle observation, quel est l'intervalle d'incertitude ?
>
> Mais avant de pouvoir répondre, il faut un **socle théorique** : connaître la distribution de $\hat{\beta}$. C'est ce que fait la section A — sans elle, aucun test n'est possible. Les sections B (qualité globale), C (inférence sur les coefficients) et D (inférence sur les prédictions) s'enchaînent ensuite naturellement.

### A. Sous H6 — la distribution de $\hat{\beta}$

On se place ici sous l'hypothèse de normalité **H6** (cf. II.A.H6) : les résidus sont gaussiens, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$. Comme $\hat{\beta} = (X^T X)^{-1} X^T y$ est une **combinaison linéaire** de $y$ — donc des $\varepsilon_i$ — il est lui aussi gaussien.

> [!warning] Distribution de $\hat{\beta}$ sous H6
> $$\hat{\beta} \sim \mathcal{N}\!\left(\beta,\ \sigma^2 (X^T X)^{-1}\right)$$
> 
> $\hat{\beta} \in \mathbb{R}^{p+1}$ est un **vecteur** gaussien, et $\sigma^2 (X^T X)^{-1}$ est sa **matrice de covariance** $(p+1) \times (p+1)$. La variance du $j$-ème coefficient s'obtient sur la diagonale :
> 
> $$V(\hat{\beta}_j) = \sigma^2 \big[(X^T X)^{-1}\big]_{jj}$$

> 💡 **Pourquoi c'est le résultat fondamental.** Connaître la **loi exacte** de $\hat{\beta}$ — pas juste son espérance et sa variance — c'est ce qui débloque toute l'inférence. Une fois $\hat{\beta}$ gaussien, toute fonction de $\hat{\beta}$ a une loi calculable : ratios → Student, sommes de carrés → $\chi^2$, ratios de $\chi^2$ → Fisher. Tous les tests qui suivent (B.2, C.1, C.2) et les intervalles de confiance/prédiction (D) sont des conséquences directes de cette ligne.

> 💡 **Trois choses à lire dans cette distribution.**
> - **Centré sur $\beta$** : $E[\hat{\beta}] = \beta$, donc $\text{bias}(\hat{\beta}) = 0$ — $\hat{\beta}$ est sans biais (résultat déjà connu via Gauss-Markov, cf. II.B).
> - **La précision dépend de $X^T X$** : plus les colonnes de $X$ sont "grandes" et orthogonales entre elles, plus $(X^T X)^{-1}$ est petite, plus $\hat{\beta}$ est précis. Intuition : un $X$ bien dispersé porte plus d'information sur $\beta$.
> - **Variance $\to 0$ quand $n \to \infty$** : les éléments de $(X^T X)^{-1}$ décroissent en $1/n$ pour des données i.i.d. → $\hat{\beta}$ est convergent.

> [!note]- Cas particulier — régression simple
> En régression simple ($p = 1$, un seul prédicteur), la matrice $X^T X$ est $2 \times 2$ et la formule se réduit à :
> 
> $\hat{\beta}_1 \sim \mathcal{N}\!\left(\beta_1,\ \frac{\sigma^2}{S_{XX}}\right) \qquad \text{où } S_{XX} = \sum_i (x_i - \bar{x})^2$
> 
> $S_{XX}$ est la **dispersion empirique de $X$** (somme des écarts au carré, le numérateur de la variance empirique). Plus $X$ est dispersé, plus $S_{XX}$ est grand, plus $\hat{\beta}_1$ est précis. C'est le cas particulier $p = 1$ de la formule matricielle générale.

> [!note]- Lien avec multicolinéarité (H5)
> Si deux colonnes de $X$ sont presque colinéaires, $X^T X$ est presque singulière → $(X^T X)^{-1}$ a des entrées **énormes** → $V(\hat{\beta}_j)$ explose. C'est exactement le diagnostic via VIF (cf. II.A.H5) : 
> 
> $V(\hat{\beta}_j) = \frac{\sigma^2}{S_{X_j X_j}} \cdot \text{VIF}_j$
> 
> Le VIF mesure littéralement *de combien la variance de $\hat{\beta}_j$ est gonflée* par rapport au cas où $X_j$ serait orthogonal aux autres prédicteurs.

### B. Qualité globale du modèle

> 💡 **Première question naturelle.** Avant de regarder les coefficients un par un, on veut savoir si le modèle vaut quelque chose **dans son ensemble** : capture-t-il une part substantielle de la variance de $Y$ ? Fait-il mieux qu'un modèle constant qui prédirait simplement $\bar{y}$ ? Deux outils répondent à cette question — un descriptif ($R^2$) et un inférentiel (F-statistic global).

#### B.1 — Le $R^2$

> 💡 **L'intuition la plus directe.** Avant la décomposition formelle ci-dessous, voici la façon la plus simple de comprendre le $R^2$ : c'est une comparaison entre deux modèles.
>
> - **Modèle naïf** : je prédis toujours $\bar y$ (la moyenne), peu importe $x$. L'erreur résiduelle de ce modèle a une variance $\text{Var}(\text{mean})$ — c'est exactement $\text{SST}/n$.
> - **Modèle régression** : je prédis $\hat y = \hat\alpha + \hat\beta x$. L'erreur résiduelle a une variance $\text{Var}(\text{line})$ — c'est $\text{SSR}/n$.
>
> Comme la droite ne peut **jamais** faire pire que la moyenne (au sens des moindres carrés — la moyenne est le cas particulier $\beta=0$), on a toujours $\text{Var}(\text{line}) \le \text{Var}(\text{mean})$. Le $R^2$ mesure **de combien la droite réduit cette variance, en proportion** :
>
> $R^2 = \frac{\text{Var}(\text{mean}) - \text{Var}(\text{line})}{\text{Var}(\text{mean})}$
>
> Lecture : si la droite réduit la variance à 0 (prédiction parfaite), $R^2 = 1$. Si la droite n'apporte **rien** par rapport à la moyenne ($\hat\beta \approx 0$, $\text{Var}(\text{line}) \approx \text{Var}(\text{mean})$), $R^2 \approx 0$ — la régression ne sert à rien, autant prédire la moyenne. C'est **rigoureusement identique** à la définition SSE/SST ci-dessous : $\text{Var}(\text{mean}) = \text{SST}/n$ et $\text{Var}(\text{line}) = \text{SSR}/n$, et le $n$ s'annule dans le ratio.

**(i) Définition classique — la décomposition des sommes de carrés**

> [!warning] Décomposition fondamentale
> $$\underbrace{\sum_i (y_i - \bar{y})^2}_{\text{SST (variance totale de } Y\text{)}} = \underbrace{\sum_i (\hat{y}_i - \bar{y})^2}_{\text{SSE (variance expliquée)}} + \underbrace{\sum_i (y_i - \hat{y}_i)^2}_{\text{SSR (résidus)}}$$
> 
> - **SST** (Sum of Squares Total) — variance totale de $Y$ autour de sa moyenne
> - **SSE** (Sum of Squares Explained) — variance des prédictions autour de la moyenne, ce que le modèle capture
> - **SSR** (Sum of Squares Residual) — variance des résidus, ce que le modèle ne capture pas

> [!warning] Définition du $R^2$
> $$R^2 = \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\text{SSR}}{\text{SST}}$$
> 
> C'est la **fraction de la variance de $Y$ expliquée par le modèle**. Va de $0$ (le modèle n'explique rien, équivalent à prédire la moyenne) à $1$ (le modèle prédit parfaitement).


![[Pasted image 20260505213127.png|481]]

**Figure 17.** Visualisation du $R^2$. Le modèle constant (qui prédit toujours $\bar{y}$, en rouge) capture nulle variance. La droite de régression (en violet) capture une fraction $R^2$ de la variance totale de $Y$ — c'est l'écart entre les deux qu'on rapporte à la variance totale.

**(ii) Lecture géométrique — Pythagore dans $L^2$**

La décomposition SST = SSE + SSR n'est rien d'autre que **Pythagore** appliqué dans $L^2$ après centrage. On retire $\bar{y}$ partout et on obtient le triangle rectangle $\bar{y}, \hat{Y}, Y$.

> [!warning] $R^2$ géométrique
> $$R^2 = \cos^2 \theta$$
> 
> où $\theta$ est l'angle entre $Y - \bar{y}$ et le sous-espace $L^2_X$ (sous-espace des prédictions).

![Pythagore dans L² centré|500](images/2-Statistiques/A_Frequentist/regression-lineaire/im4.png)

**Figure 18.** Pythagore dans $L^2$ centré. La décomposition $\text{SST} = \text{SSE} + \text{SSR}$ est le théorème de Pythagore appliqué au triangle $\bar{y}, \hat{Y}, Y$. $R^2 = \cos^2\theta$.

> 💡 **Deux lectures du même objet.** $R^2 = \text{SSE}/\text{SST}$ (ratio de sommes de carrés) et $R^2 = \cos^2\theta$ (lecture géométrique) disent exactement la même chose. L'une est calculatoire et apparaît dans tous les outputs de logiciels ; l'autre est conceptuelle et permet de comprendre pourquoi $R^2 \in [0, 1]$ (un cosinus carré est toujours dans cet intervalle).

**(iii) $R^2$ ajusté**

> 💡 **Le défaut majeur du $R^2$.** Ajouter une variable au modèle agrandit mécaniquement le sous-espace $L^2_X$ — un espace plus grand capture toujours un peu mieux $Y$, même si la variable ajoutée est du **bruit pur**. Le $R^2$ augmente donc toujours quand on ajoute un prédicteur, **même inutile**. C'est pour ça qu'on ne peut pas comparer deux modèles avec des nombres de prédicteurs différents en regardant juste leur $R^2$ — il faut une version corrigée.

Le **$R^2$ ajusté** corrige ce défaut en pénalisant le nombre de paramètres via les degrés de liberté.

> [!warning] Définition du $R^2$ ajusté
> $$\bar{R}^2 = 1 - \frac{\text{SSR}/(n-p-1)}{\text{SST}/(n-1)}$$
> 
> où $p$ est le nombre de prédicteurs (hors intercept) et $n$ le nombre d'observations.

> 💡 **L'idée.** Au lieu de comparer SSR et SST directement (comme dans $R^2$), on les **divise chacune par leurs degrés de liberté**. Quand on ajoute une variable inutile, SSR baisse un peu (numérateur ↓) mais le diviseur $n-p-1$ baisse aussi (numérateur effectif ↑). Si la variable est vraiment du bruit, le second effet l'emporte et $\bar{R}^2$ **diminue**. C'est le mécanisme de pénalisation.

> [!note]- Quand utiliser $R^2$ vs $\bar{R}^2$ ?
> - **$R^2$** : pour décrire la qualité d'un modèle fixé. Lecture intuitive ("le modèle explique X% de la variance").
> - **$\bar{R}^2$** : pour **comparer des modèles** avec des nombres de prédicteurs différents. Comparer $R^2$ entre deux modèles favorise systématiquement le plus complexe — toujours utiliser $\bar{R}^2$ pour la sélection de modèle.

#### B.2 — Le F-statistic global

Le $R^2$ est descriptif — il dit *combien* le modèle explique, pas si cette explication est statistiquement significative. Pour le savoir, on a besoin d'un **test** : c'est le rôle du **F-statistic global**, qui apparaît dans tout summary OLS sous le nom `F-statistic` et `Prob (F-statistic)`.

> [!warning] Test de signification globale du modèle
> On compare le modèle complet (avec tous ses $p$ prédicteurs) au **modèle constant** ($\hat{y}_i = \bar{y}$, aucun prédicteur). L'hypothèse nulle :
> 
> $$H_0 : \beta_1 = \beta_2 = \cdots = \beta_p = 0$$
> 
> *("aucun prédicteur n'apporte rien")*. La statistique :
> 
> $$F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$$
> 
> Sous $H_0$, $F$ suit une loi de Fisher à $(p, n-p-1)$ degrés de liberté. Si $F$ est grand → on rejette $H_0$ → le modèle est globalement informatif.

> 💡 **Lecture en deux mots.** $F$ est un **ratio signal/bruit** au niveau global. Numérateur = part de variance expliquée par les $p$ prédicteurs (par degré de liberté). Dénominateur = variance résiduelle (par degré de liberté). Si signal > bruit → le modèle marche.

> [!note]- Cas particulier d'un test plus général
> Ce $F$ global n'est pas un objet à part — c'est le **cas particulier** du test de Fisher de modèles emboîtés (cf. C.2) où le modèle restreint est le modèle constant. La mécanique générale (test joint sur $q$ coefficients quelconques) est traitée en C.2.

> 💡 **Attention au piège.** Si le F global rejette $H_0$, ça veut dire **"au moins un prédicteur est utile"**, pas **"tous le sont"**. Pour savoir lesquels, il faut passer aux tests individuels (C.1) ou aux tests joints partiels (C.2). $R^2$ et F global donnent une vue d'ensemble — pas un diagnostic au niveau des coefficients.

### C. Inférence sur les coefficients

> 💡 **Deuxième question naturelle.** Le modèle marche globalement (B). Très bien. Mais **quels prédicteurs comptent vraiment** ? Est-ce que `horsepower` a un effet significatif sur `mpg` ? Et l'`origin` apporte-t-elle quelque chose **au-delà** de ce que `horsepower` capture déjà ? Deux tests pour deux granularités : Student pour un coefficient à la fois (C.1), Fisher pour plusieurs coefficients d'un coup (C.2). Et un récapitulatif pratique en C.3 : comment lire ces tests dans le summary OLS.

#### C.1 — Test de Student sur $\hat{(\beta_j)}$

**La question.** Est-ce que le prédicteur $X_j$ a vraiment un effet sur $Y$, ou est-ce que $\hat{\beta}_j \neq 0$ par chance ? On teste $H_0 : \beta_j = 0$ contre $H_1 : \beta_j \neq 0$, **un coefficient à la fois**.

> [!warning] Statistique de Student
> $\sigma^2$ est inconnue en pratique, on l'estime par $\hat{\sigma}^2 = \text{SSR}/(n - p - 1)$. L'**erreur-type** du coefficient $\hat{\beta}_j$ s'obtient à partir de la diagonale de $(X^T X)^{-1}$ :
> 
> $\text{se}(\hat{\beta}_j) = \hat{\sigma} \sqrt{\big[(X^T X)^{-1}\big]_{jj}}$
> 
> La statistique de Student :
> 
> $t_j = \frac{\hat{\beta}_j}{\text{se}(\hat{\beta}_j)}$
> 
> **Lecture.** $t_j$ se lit « à combien d'écarts-types de 0 se trouve mon estimation » (ratio **signal / bruit** : effet estimé ÷ son incertitude).
> - $t_j = 5$ → $\hat{\beta}_j$ est à 5 erreurs-types de 0 ; il ne retomberait quasi jamais à 0 en refaisant l'expérience → **effet réel**.
> - $t_j = 0.3$ → $\hat{\beta}_j$ est à 0.3 erreur-type de 0 ; le bruit seul explique la valeur observée → **effet indistinguable de zéro**.
> 
> Sous $H_0$, $t_j$ suit une loi de Student à $n - p - 1$ degrés de liberté. Pour $n$ grand, on retient la règle :
> 
> $|t_j| > 1.96 \implies \text{on rejette } H_0 \text{ au seuil } 5\%$

![Loi de la statistique de test sous H0 et p-value](images/2-Statistiques/A_Frequentist/regression-lineaire/pvalue.png)

**Figure — La p-value comme aire sous la loi de $t$ sous $H_0$.** La cloche est la distribution de $t_j$ si $\beta_j = 0$. Le $t_{\text{obs}}$ observé tombe dans la queue ; l'aire orange au-delà de $\pm|t_{\text{obs}}|$ (les deux queues) est la **p-value**. Les pointillés $\pm 1.96$ délimitent la zone de rejet à 5%. Rejeter $H_0$ ($|t| > 1.96$) équivaut exactement à « 0 hors de l'IC à 95% ».

> 💡 **La p-value.** C'est la probabilité d'observer un $|t_j|$ aussi grand si $H_0$ était vraie. Plus elle est petite, plus on est confiant que $\beta_j \neq 0$. Seuil classique : $p < 0.05$. C'est la colonne `P>|t|` du summary OLS (cf. C.3).

> [!note]- Rappel — p-value, et son lien avec la t-stat
> **Définition.** La p-value est la probabilité, **sous $H_0$** (ici $\beta_j = 0$), d'observer une statistique au moins aussi extrême que celle mesurée :
> 
> $p = P\big(|T| \ge |t_{\text{obs}}| \;\big|\; H_0\big), \quad T \sim t_{n-p-1}$
> 
> Petite p → ce qu'on observe serait très improbable si $\beta_j = 0$ → on rejette $H_0$.
> 
> **Lien avec la t-stat.** La p-value n'est que la t-stat **passée dans la CDF de Student** — une fonction décroissante de $|t|$. Asymptotiquement (grand $n$), $p \approx 2(1 - \Phi(|t|))$ :
> 
> | $|t|$ | p-value (bilatérale) |
> |:---:|:---:|
> | 1.96 | 0.05 |
> | 2.58 | 0.01 |
> | 3.0 | 0.0027 |
> 
> Les deux portent la même information ; en finance on cite souvent directement la **t-stat** (sans dimension, règle du pouce $|t| > 2$) plutôt que la p-value.
> 
> **Le débat des seuils (asset pricing).** Harvey, Liu & Zhu (2016) montrent qu'avec des centaines de facteurs testés (multiple testing), $|t| > 2$ est trop laxiste : ils recommandent $|t| > 3$ pour qu'un facteur soit crédible. D'où l'habitude de comparer des t-stats entre facteurs.
> 
> > ⚠️ **Ce que la p-value n'est PAS.** Ce n'est **pas** $P(H_0 \text{ vraie} \mid \text{données})$, ni « la probabilité que le résultat soit dû au hasard ». C'est une probabilité calculée *en supposant $H_0$ vraie*. Confondre les deux est l'erreur la plus classique en entretien.

> [!warning] Intervalle de confiance sur $\beta_j$ à 95%
> $\hat{\beta}_j \pm 1.96 \cdot \text{se}(\hat{\beta}_j)$
> 
> C'est l'ensemble des valeurs de $\beta_j$ qu'on ne rejetterait pas au seuil 5%. Plus $\text{se}(\hat{\beta}_j)$ est petite, plus l'intervalle est étroit — plus on est précis. C'est la colonne `[0.025, 0.975]` du summary.

![Distribution d'échantillonnage et IC répétés](images/2-Statistiques/A_Frequentist/regression-lineaire/ic_beta.png)

**Figure — Ce que « 95% » signifie.** *Gauche* : un dataset → la pente OLS $\hat{\beta}_1$ et son IC $[1.43, 2.17]$. *Droite* : 100 datasets simulés (même $\beta_1^\ast$, bruit re-tiré) → 100 IC, dont ~95 sur 100 contiennent $\beta_1^\ast$ (bleu), le reste le rate (rouge). La barre verte est **ton** IC réel — le seul que tu as en pratique. Illustre directement la loi $\hat{\beta}_1 \sim \mathcal{N}(\beta_1, \sigma^2/S_{XX})$ de III.A. Cependant tu n'as aucun moyen pour savoir si le IC réel contient vraiment $\beta_1^*$. 

> [!tip] Ce que « 95% » veut vraiment dire
> Le « 95% » porte sur la **procédure**, pas sur ton intervalle. Une fois les données collectées, ton IC est **figé** : il contient $\beta_1^\ast$ ou non, point — plus aucune probabilité sur *ce* cas précis. Ce que tu fais avec un IC, en pratique :
> - **$0 \in$ IC ?** → si oui, l'effet n'est pas distinguable de zéro (feature potentiellement inutile). C'est le test de Student ci-dessus, lu géométriquement.
> - **Largeur ?** → précision de l'estimation. $[1, 3]$ est utilisable, $[1, 10]$ ne dit presque rien.
> - **Niveau 95%** = convention : monter à 99% **élargit** l'IC (moins informatif). Ce n'est pas gratuit.
> 
> Pour une vraie probabilité sur *ton* intervalle, $P(\beta_1 \in [a,b] \mid \text{données})$, il faut passer bayésien (intervalle de crédibilité, au prix d'un prior).

> [!note]- Pipeline de simulation — comment les 100 IC sont générés
> Pour chaque dataset $k = 1, \dots, 100$ :
> 1. $x_i$ **fixes**, vrais $\beta_0, \beta_1$ **fixes** (connus — c'est une simulation) ;
> 2. tire un nouveau bruit $\varepsilon_i^{(k)} \sim \mathcal{N}(0, \sigma^2)$ ;
> 3. calcule $y_i^{(k)} = \beta_0 + \beta_1 x_i + \varepsilon_i^{(k)}$ ;
> 4. OLS sur $(x_i, y_i^{(k)})$ → $\hat{\beta}_1^{(k)}$ et $\text{se}^{(k)}$ ;
> 5. $\text{IC}^{(k)} = \hat{\beta}_1^{(k)} \pm t \cdot \text{se}^{(k)}$.
> 
> Seul le bruit (donc $y$) change entre les 100 datasets ; $x$ et le vrai $\beta_1$ ne bougent jamais. Sur de vraies données on ne peut pas faire ça (on ignore le vrai $\beta_1$, et re-tirer le bruit = collecter de nouvelles observations) — d'où le **bootstrap** comme substitut.

> [!note]- Cas particulier — régression simple
> En régression simple ($p = 1$), $\text{se}(\hat{\beta}_1) = \hat{\sigma}/\sqrt{S_{XX}}$ et la statistique se simplifie en :
> 
> $t = \frac{\hat{\beta}_1}{\hat{\sigma}/\sqrt{S_{XX}}} \sim t_{n-2}$
> 
> avec $\hat{\sigma}^2 = \text{SSR}/(n-2)$. C'est ce qu'on rencontre dans la majorité des manuels d'intro — c'est juste la formule générale appliquée au cas $p = 1$.

#### C.2 — Test de Fisher — test joint sur plusieurs coefficients

**La question.** Le test de Student de C.1 teste **un coefficient à la fois**. Mais souvent on veut tester **un groupe de coefficients ensemble**. Exemples typiques :

- *Mes 4 dummies de secteur (Tech, Oil, Banque, Real Estate vs référence) sont-elles **jointement** significatives ?* — un test pour les 4 d'un coup, pas 4 tests Student séparés.
- *Ajouter `weight + displacement + cylinders` au modèle `mpg ~ hp` apporte-t-il quelque chose ?* — comparaison entre un modèle restreint et un modèle complet.
- *Le terme polynomial $\text{hp}^2$ et le terme d'interaction $\text{hp} \times \text{origin}$ sont-ils utiles ensemble ?*

> 💡 **Pourquoi pas juste plusieurs tests Student ?** Faire 4 tests Student séparés au seuil 5% donne une probabilité de faux positif **bien supérieure à 5%** sur l'ensemble (problème de tests multiples). Et surtout, ça ne capture pas l'**effet conjoint** : deux coefficients individuellement insignificants peuvent être conjointement très significatifs s'ils sont corrélés. Le test de Fisher répond directement à la bonne question.

**(i) Cadre — modèle restreint vs modèle complet**

L'idée centrale : on compare deux modèles emboîtés.

> [!warning] Modèles emboîtés
> - **Modèle complet** ($M_1$) : utilise tous les $p$ prédicteurs, $\text{SSR}_1$ résiduel.
> - **Modèle restreint** ($M_0$) : on annule $q$ coefficients (par exemple $q=4$ dummies de secteur), il reste $p - q$ prédicteurs, $\text{SSR}_0$ résiduel.
> 
> **Hypothèse nulle** : $H_0 : \beta_{p-q+1} = \cdots = \beta_p = 0$ (les $q$ coefficients qu'on retire sont tous nuls).

Par construction, $\text{SSR}_0 \geq \text{SSR}_1$ : retirer des prédicteurs ne peut qu'augmenter les résidus. La question : **l'augmentation est-elle significative** ?

**(ii) Statistique de Fisher**

> [!warning] Statistique F (modèles emboîtés)
> 
> $$F = \frac{(\text{SSR}_0 - \text{SSR}_1)/q}{\text{SSR}_1/(n - p - 1)}$$
> 
> Sous $H_0$, $F$ suit une **loi de Fisher** à $(q, n-p-1)$ degrés de liberté.
> 
> - **Numérateur** : $(\text{SSR}_0 - \text{SSR}_1)/q$ = gain de variance expliqué par les $q$ coefficients ajoutés, par degré de liberté
> - **Dénominateur** : $\text{SSR}_1/(n - p - 1)$ = variance résiduelle du modèle complet, par degré de liberté
> 
> Si $F$ est grand → les coefficients ajoutés expliquent beaucoup plus de variance que ce qu'on attendrait par hasard → on rejette $H_0$.

> 💡 **Interprétation en deux mots.** $F$ est le **ratio signal/bruit**. Numérateur = ce que les $q$ coefficients supplémentaires apportent. Dénominateur = la variance résiduelle de référence. Si signal > bruit (typiquement $F > 4$ pour des seuils usuels), alors les coefficients sont jointement significatifs.

> [!note]- Lien avec l'ANOVA classique
> La formulation ci-dessus (modèles emboîtés via $\text{SSR}_0 - \text{SSR}_1$) est la version **régression** du test de Fisher. Elle est strictement équivalente à la formulation **ANOVA classique** par décomposition $\text{SC}_{\text{tot}} = \text{SC}_{\text{inter}} + \text{SC}_{\text{intra}}$ avec son tableau ANOVA (Source / ddl / SC / CM / F).
> 
> Pour la mécanique formelle de la décomposition de variance, le tableau ANOVA standardisé, et le lien Fisher↔Student, voir [[Tests d'hypothèses]] section IV.C. Ici on garde la formulation modèles emboîtés parce qu'elle est la plus naturelle en cadre régression — elle s'applique à n'importe quel test joint de coefficients, pas seulement à un facteur catégoriel.

**(iii) Cas particuliers utiles**

> [!note]- Cas 1 — Test de signification globale du modèle
> Quand $M_0$ est le modèle constant ($\hat{y}_i = \bar{y}$, aucun prédicteur) et $M_1$ est le modèle complet avec tous les $p$ prédicteurs, le test de Fisher devient :
> 
> $$F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$$
> 
> Ce $F$ apparaît dans tout output `summary()` (R) ou `.summary()` (statsmodels) : "F-statistic" et "Prob (F-statistic)". Il teste $H_0$ : *"aucun prédicteur n'a d'effet"*. Si on rejette, le modèle est globalement informatif.

> [!note]- Cas 2 — Lien avec le test de Student
> Quand on teste **un seul** coefficient ($q = 1$), Fisher et Student sont équivalents :
> 
> $$F_{(1, n-p-1)} = t_{(n-p-1)}^2$$
> 
> Le test de Student est donc le cas particulier du test de Fisher pour $q = 1$. Fisher est la généralisation à $q$ coefficients simultanés.

> [!example] Application — significativité jointe d'un facteur catégoriel
> Sur Auto, on veut savoir si la variable `origin` (USA/Europe/Japon) apporte quelque chose au modèle `mpg ~ hp`. C'est un test joint sur **2 dummies** ($\beta_{\text{Europe}}$ et $\beta_{\text{Japon}}$, USA = référence) :
> 
> - **Modèle restreint** : $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp}$ → $\text{SSR}_0$
> - **Modèle complet** : $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Europe}} + \beta_3 \mathbb{1}_{\text{Japon}}$ → $\text{SSR}_1$
> - **$H_0$** : $\beta_2 = \beta_3 = 0$ (l'origine n'apporte rien à hp égale)
> - **Statistique** : $F = \dfrac{(\text{SSR}_0 - \text{SSR}_1)/2}{\text{SSR}_1/(392 - 4)}$ avec ddl $(2, 388)$
> 
> Si $F$ est grand → l'origine apporte une information **au-delà** de ce que `hp` capture → on rejette $H_0$ et on garde les dummies. C'est exactement le diagnostic qu'on a fait en II.A.H4 (clustering par origine), formalisé statistiquement.

> [!note]- En pratique — `anova()` ou `f_test()`
> En R : `anova(model_restreint, model_complet)` calcule directement le F et la p-value. En Python avec statsmodels : `model.f_test("origin_Europe = 0, origin_Japon = 0")` ou comparaison de modèles via `anova_lm`. Sur sklearn pas d'équivalent natif — il faut passer par statsmodels pour l'inférence.

#### C.3 — Lecture d'un summary OLS sur Auto

Tout ce qu'on a vu jusqu'ici (R², F global, test de Student, test de Fisher) apparaît dans **un seul tableau standardisé** : le `summary()` produit par n'importe quel logiciel de stats (`statsmodels` en Python, `lm()` + `summary()` en R). Cette sous-section explique comment le lire, sur deux modèles concrets ajustés sur Auto.

##### C.3.a Modèle simple — `mpg ~ horsepower`

On régresse simplement `mpg` sur `horsepower`. Voici l'output complet de `model.summary()` :

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.606
Model:                            OLS   Adj. R-squared:                  0.605
Method:                 Least Squares   F-statistic:                     599.7
Date:                Wed, 06 May 2026   Prob (F-statistic):           7.03e-81
Time:                        21:58:50   Log-Likelihood:                -1178.7
No. Observations:                 392   AIC:                             2361.
Df Residuals:                     390   BIC:                             2369.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept     39.9359      0.717     55.660      0.000      38.525      41.347
horsepower    -0.1578      0.006    -24.489      0.000      -0.171      -0.145
==============================================================================
Omnibus:                       16.432   Durbin-Watson:                   0.920
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               17.305
Skew:                           0.492   Prob(JB):                     0.000175
Kurtosis:                       3.299   Cond. No.                         322.
==============================================================================
```

##### Bloc 1 — métadonnées et qualité globale du modèle

- **`R-squared = 0.606`** → le R² de la section B.1 : 60.6% de la variance de `mpg` est expliquée par `horsepower`. Le reste est dans les résidus.
- **`Adj. R-squared = 0.605`** → R² ajusté, pénalisé par le nombre de prédicteurs. Utile pour comparer entre modèles avec un nombre de prédicteurs différent.
- **`F-statistic = 599.7`, `Prob (F-statistic) = 7e-81`** → c'est le **F-statistic global** de la section B.2. Il teste $H_0$ : *aucun prédicteur n'a d'effet*. Ici on rejette écrasement — le modèle est globalement informatif.
- **`No. Observations = 392`** → taille de l'échantillon $n$.
- **`Df Residuals = 390`** → degrés de liberté résiduels $n - p - 1 = 392 - 1 - 1$.
- **`Df Model = 1`** → nombre de prédicteurs (hors intercept). Ici juste `horsepower`.
- **`AIC` / `BIC`** → critères d'information (Akaike / Bayesian) pour comparaison de modèles. Plus petit = meilleur. Pénalisent la complexité comme l'adjusted R².

##### Bloc 2 — tableau des coefficients (le cœur)

Lecture de la ligne `horsepower` :

- **`coef = -0.1578`** → c'est $\hat{\beta}_1$. Chaque cheval supplémentaire fait baisser la mpg de **0.158** mpg en moyenne.
- **`std err = 0.006`** → erreur-type de l'estimateur, $\hat{\sigma}/\sqrt{S_{XX}}$ (cf. A). Mesure l'incertitude sur $\hat{\beta}_1$.
- **`t = -24.489`** → c'est la **statistique de Student** (cf. C.1) : `coef / std err` = $-0.1578 / 0.006$. Teste $H_0 : \beta_1 = 0$.
- **`P>|t| = 0.000`** → **p-value bilatérale** du test de Student. Ici < 0.001 → on rejette $H_0$ très fortement — `horsepower` a un effet significatif sur `mpg`.
- **`[0.025, 0.975] = [-0.171, -0.145]`** → **intervalle de confiance à 95%** sur $\beta_1$ (cf. C.1). Si 0 n'est pas dedans → significatif au seuil 5%. Ici 0 est très loin de l'intervalle, cohérent avec la p-value écrasante.

##### Bloc 3 — diagnostics sur les résidus

Ces statistiques vérifient les **hypothèses Gauss-Markov** (cf. II.A) :

- **`Omnibus = 16.4`, `Prob(Omnibus) = 0.000`** + **`Jarque-Bera (JB) = 17.3`, `Prob(JB) = 0.0002`** → deux tests de **normalité des résidus**. Ici p < 0.001 dans les deux cas → résidus **non-gaussiens**. Suggestion : modèle mal spécifié (cf. courbure en U vue en II.A.H1).
- **`Skew = 0.49`** (asymétrie) et **`Kurtosis = 3.30`** (aplatissement) → métriques associées. Sous normalité parfaite : Skew = 0, Kurtosis = 3.
- **`Durbin-Watson = 0.92`** → test d'**autocorrélation des résidus** (H4). DW ≈ 2 = OK, DW < 1 = forte autocorrélation positive ⚠️. Ici 0.92 → lien avec le clustering par origine vu en II.A.H4.
- **`Cond. No. = 322`** → **conditionnement** de $X^TX$. > 30 = signal de multicolinéarité ou instabilité numérique (H5). Ici élevé — dû au fait que `horsepower` est centré autour de ~100 plutôt que 0.

##### C.3.b Modèle ANCOVA — `mpg ~ horsepower + origin`

On ajoute la variable catégorielle `origin` (USA, Europe, Japon) au modèle. USA est la référence (cf. I.E sur la treatment constraint). L'output devient :

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    mpg   R-squared:                       0.662
Model:                            OLS   Adj. R-squared:                  0.659
Method:                 Least Squares   F-statistic:                     253.4
Date:                Wed, 06 May 2026   Prob (F-statistic):           4.93e-91
Time:                        21:58:50   Log-Likelihood:                -1148.5
No. Observations:                 392   AIC:                             2305.
Df Residuals:                     388   BIC:                             2321.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=======================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------------
Intercept              35.9441      0.867     41.444      0.000      34.239      37.649
C(origin)[T.Europe]     2.4253      0.678      3.578      0.000       1.093       3.758
C(origin)[T.Japon]      5.1764      0.648      7.990      0.000       3.903       6.450
horsepower             -0.1336      0.007    -19.474      0.000      -0.147      -0.120
==============================================================================
Omnibus:                       23.486   Durbin-Watson:                   1.036
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               25.987
Skew:                           0.602   Prob(JB):                     2.27e-06
Kurtosis:                       3.376   Cond. No.                         510.
==============================================================================
```

##### Ce qui change avec une catégorielle

- **`Df Model = 3`** au lieu de 1 — maintenant 3 prédicteurs : `horsepower`, dummy Europe, dummy Japon
- **Une ligne de coefficient par modalité, sauf la référence** — USA est absorbé dans l'intercept
- **`R²` passe de 0.606 à 0.662** — ajouter `origin` apporte de la variance expliquée
- **`Adj. R²`** passe de 0.605 à 0.659 — le gain est réel même après pénalisation pour les paramètres ajoutés
- **`Durbin-Watson`** passe de 0.92 à 1.04 — inclure `origin` réduit le clustering résiduel (cf. II.A.H4)

##### Lecture des coefficients

- **`Intercept = 35.94`** → mpg prédit pour une voiture **USA** à `horsepower = 0`. Pas de sens physique (hp = 0 n'existe pas), mais c'est le **point d'ancrage** de la droite USA.
- **`C(origin)[T.Europe] = +2.43`** → à puissance égale, une voiture **européenne fait +2.43 mpg de plus qu'une américaine**. C'est le décalage vertical entre la droite Europe et la droite USA (cf. Figure 7 d'ANCOVA en I.E).
- **`C(origin)[T.Japon] = +5.18`** → à puissance égale, une **japonaise fait +5.18 mpg de plus qu'une américaine**.
- **`horsepower = -0.1336`** → pente commune aux trois groupes : 1 cheval supplémentaire = -0.134 mpg, indépendamment de l'origine.
- **Toutes les `P>|t|` sont à 0.000** → chaque effet est individuellement significatif vs référence USA.

##### C.3.c Le piège — p-values individuelles vs test F joint

Si on te demande *"l'origine est-elle significative dans le modèle ?"*, **regarder les p-values des dummies individuellement n'est pas suffisant**. Deux raisons :

1. **Choix arbitraire de la référence** : si on prend Japon comme référence à la place de USA, les coefficients et p-values des dummies changent. Mais l'effet "origine" en tant que variable est le même — ce n'est pas une propriété intrinsèque du facteur.
2. **Multiple testing** : 2 tests Student au seuil 5% chacun → vraie probabilité de faux positif > 5% sur l'ensemble.

**La bonne question** : *"$\beta_{\text{Europe}} = \beta_{\text{Japon}} = 0$ jointement ?"* — c'est exactement le **test F de modèles emboîtés** (cf. C.2). Il faut le faire **séparément**, il n'apparaît pas dans le summary :

```python
# Méthode 1 : f_test direct
model.f_test("C(origin)[T.Europe] = 0, C(origin)[T.Japon] = 0")
# Output : F=32.24, p=1.1e-13, df=(2, 388)

# Méthode 2 : comparaison modèles emboîtés
from statsmodels.stats.anova import anova_lm
anova_lm(model_sans_origin, model_avec_origin)
# Output :
#    df_resid     ssr    df_diff   ss_diff       F      Pr(>F)
# 0    390.0   9385.92    0.0       NaN        NaN       NaN
# 1    388.0   8048.39    2.0     1337.53    32.24    1.11e-13
```

Les deux méthodes donnent **le même F = 32.24, p = 1.1e-13**. Conclusion : `origin` apporte **vraiment** quelque chose au-delà de `horsepower`, indépendamment du choix de référence.

> 💡 **À retenir.** Pour un facteur catégoriel à $K$ modalités, le summary affiche $K-1$ tests de Student (un par dummy vs référence). Pour tester *"le facteur est-il globalement significatif ?"*, il faut un **test F joint sur les $K-1$ dummies en même temps** — à faire à la main via `f_test()` ou `anova_lm()`. C'est *exactement* le test F de modèles emboîtés de C.2.

##### C.3.d Workflow de lecture en 30 secondes

Devant un summary OLS quelconque, mon workflow mental :

1. **Le modèle marche-t-il globalement ?** → R², F-stat (`Prob (F-statistic)`)
2. **Quels prédicteurs sont significatifs individuellement ?** → colonne `P>|t|` du tableau de coefficients
3. **Quels sont les effets concrets ?** → colonne `coef` avec son unité métier
4. **Catégorielles** → si présentes, faire un test F joint **séparément** via `f_test()`
5. **Hypothèses Gauss-Markov** → bloc du bas : Durbin-Watson ≈ 2 ? Omnibus/JB > 0.05 ? Cond. No. < 30 ?

### D. Inférence sur les prédictions

> 💡 **Troisième question naturelle.** Le modèle marche globalement (B), on sait quels coefficients comptent (C). Reste la question pratique : **avec quelle précision peut-on prédire** ? Pour une nouvelle valeur $x^*$, $\hat{y}^* = \hat{\alpha} + \hat{\beta}x^*$ donne une prédiction ponctuelle — mais pas son incertitude. Et il y a un piège : *l'incertitude sur la moyenne de $Y$ en $x^*$* (D.1) et *l'incertitude sur une nouvelle observation individuelle en $x^*$* (D.2) sont deux choses différentes. La distinction est subtile mais cruciale.

#### D.1 — Intervalle de confiance vs Intervalle de prédiction

On dispose d'un modèle estimé $\hat{y} = \hat{\alpha} + \hat{\beta}x$. Pour une nouvelle valeur $x^*$, on veut quantifier l'incertitude autour de la prédiction $\hat{y}^* = \hat{\alpha} + \hat{\beta}x^*$.

> 💡 **Deux questions très différentes.**
> - **Où se situe la moyenne de $Y$ en $x^*$ ?** → Intervalle de **confiance** (IC).
> - **Où va tomber une nouvelle observation individuelle en $x^*$ ?** → Intervalle de **prédiction** (IP).
> 
> La distinction est subtile mais cruciale en pratique : un IC sur la moyenne est beaucoup plus étroit qu'un IP sur une observation individuelle.

> [!warning] Intervalle de confiance sur $E(Y \mid X = x^*)$ à 95%
> $$\hat{y}^* \pm 1.96 \cdot \hat{\sigma}\sqrt{\frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{XX}}}$$
> 
> Encadre la **vraie moyenne** $\alpha + \beta x^*$. L'incertitude vient uniquement de l'estimation de $\hat{\alpha}$ et $\hat{\beta}$.

> [!warning] Intervalle de prédiction sur $Y(x^*)$ à 95%
> 
> $$\hat{y}^* \pm 1.96 \cdot \hat{\sigma}\sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{XX}}}$$
> 
> Encadre une **nouvelle observation individuelle**. L'incertitude vient de **deux sources** : l'estimation de $\hat{\alpha}$ et $\hat{\beta}$ **plus** le bruit irréductible $\varepsilon^*$.

> 💡 **Le rôle du $+1$.** La seule différence entre les deux formules est le $+1$ sous la racine dans l'IP. Ce $1$ représente la variance du bruit $\varepsilon^*$ — irréductible peu importe la taille de l'échantillon. Même avec $n \to \infty$, on estimerait $\alpha + \beta x^*$ parfaitement, mais une observation individuelle resterait dispersée autour de cette moyenne. **On ne peut pas prédire le bruit.**

Les deux intervalles s'élargissent aussi quand $x^*$ s'éloigne de $\bar{x}$ — on extrapole loin des données, l'estimation devient moins fiable.

> [!danger] En pratique — ces formules sont rarement utilisées
> Les formules ci-dessus reposent sur **H1–H6** (linéarité, homoscédasticité, indépendance, normalité). Dès qu'une hypothèse pète — et en pratique au moins une pète toujours — **l'IP est mal calibré** : un IP "à 95%" qui couvre en réalité 80% des observations est pire que pas d'IP du tout.
> 
> Ce qu'on fait à la place :
> - **Bootstrap** — on ré-échantillonne 1000 fois et on lit les quantiles empiriques de $\hat{y}^*$. Robuste, distribution-free, marche même en non-linéaire.
> - **Conformal prediction** — approche moderne (ML) qui donne des **garanties de couverture** sans hypothèse sur la loi des résidus. Devenu standard en deep learning quand on veut quantifier l'incertitude prédictive.
> - **Quantile regression** — on régresse directement les quantiles 5% et 95% de $Y$ au lieu de la moyenne, sans supposer la symétrie ni la normalité.
> 
> Les formules classiques restent utiles pour : (1) reporting académique avec bandes grises sur un fit, (2) entretiens "théorie de la régression", (3) intuition du rôle du $+1$ et de l'extrapolation. Pour de la vraie prédiction sous incertitude en production — bootstrap ou conformal.

---

## IV. Variantes d'OLS

OLS est BLUE sous H1–H5, mais en pratique au moins une de ces hypothèses est souvent violée. Plutôt que d'abandonner la régression linéaire, on l'**adapte**. Chaque variante d'OLS répond à une violation spécifique des hypothèses Gauss-Markov.

### A. Quel modèle quand ? — table de décision

> [!warning] Choisir sa variante en fonction du symptôme
> 
> | Symptôme observé | Hypothèse violée | Variante à utiliser | Sous-section |
> |---|:---:|:---:|:---:|
> | Cône dans le residual plot | H3 (hétéroscédasticité) | **WLS** | B |
> | Cône + autocorrélation/clustering | H3 + H4 | **GLS** | C |
> | Contrainte structurelle à imposer (ex: somme nulle, neutralité) | — (info externe) | **RLS** | D |
> | Quelques outliers qui tirent la droite | — (robustesse) | **Robust LS** | E |
> | Multicolinéarité forte | H5 | **Ridge / Lasso** | section V |

> 💡 **L'idée commune.** Toutes ces variantes ont la **même structure** que OLS : on minimise une somme de carrés. Ce qui change c'est soit la **métrique** (poids différents par observation), soit la **contrainte** (imposer des relations sur $\beta$), soit la **fonction de perte** (autre que quadratique). On reste dans la famille "moindres carrés" mais on l'adapte au contexte.

### B. WLS — Weighted Least Squares

> [!warning] Quand utiliser WLS ?
> Quand H3 est violée (hétéroscédasticité) mais H4 reste vraie (résidus non corrélés entre eux). Symptôme typique : **cône** dans le residual plot, sans structure de dépendance entre observations.

> [!warning] Critère et estimateur WLS
> Au lieu de minimiser $\sum (y_i - x_i^T \beta)^2$ (toutes les observations sur un pied d'égalité), on **pondère** chaque résidu :
> 
> $$\hat{\beta}_{\text{WLS}} = \arg\min_\beta \sum_{i=1}^n w_i (y_i - x_i^T \beta)^2$$
> 
> Sous forme matricielle avec $W = \text{diag}(w_1, \ldots, w_n)$ :
> 
> $$\boxed{\hat{\beta}_{\text{WLS}} = (X^T W X)^{-1} X^T W y}$$
> 
> Le **choix optimal des poids** : $w_i = 1/\sigma_i^2$ où $\sigma_i^2 = V(\varepsilon_i)$. Sous ces poids, WLS est BLUE.

> 💡 **L'intuition.** Les observations à grande variance ($\sigma_i^2$ grand) sont **moins fiables** — elles devraient compter moins dans l'ajustement. Le poids $1/\sigma_i^2$ donne exactement cette pondération inverse. Une observation très bruitée pèse peu, une observation précise pèse beaucoup. C'est l'**inverse-variance weighting**, principe qu'on retrouve partout en stats (méta-analyses, fusion de capteurs, etc.).

> [!note]- Pourquoi WLS marche — l'astuce du blanchiment
> On peut réécrire le critère WLS comme un OLS classique sur des données transformées. En posant $\tilde{X} = \sqrt{W} X$ et $\tilde{y} = \sqrt{W} y$ :
> 
> $$\sum_i w_i (y_i - x_i^T \beta)^2 = \|\sqrt{W}(y - X\beta)\|^2 = \|\tilde{y} - \tilde{X}\beta\|^2$$
> 
> C'est juste OLS sur $(\tilde{X}, \tilde{y})$ ! Et sous les bons poids ($w_i = 1/\sigma_i^2$), les résidus transformés deviennent **homoscédastiques** : $V(\sqrt{w_i} \varepsilon_i) = w_i \sigma_i^2 = 1$ pour tout $i$. On a "blanchi" le bruit, et OLS sur les données blanchies retrouve son caractère BLUE.

> [!example] Lien avec ton expérience MS — Barra
> Le modèle factoriel **Barra** (Morgan Stanley RiskLab) utilise WLS pour estimer les rendements factoriels. Les actions ont des **variances idiosyncratiques** $\sigma_i^2$ très différentes : une grande capitalisation stable a un faible $\sigma_i^2$, une small cap volatile a un grand $\sigma_i^2$. WLS avec $w_i = 1/\sigma_i^2$ donne plus de poids aux stocks "bien comportés" et moins aux stocks bruités — c'est la version moderne de l'estimation cross-sectionnelle des facteurs.

### C. GLS — Generalized Least Squares

> [!warning] Quand utiliser GLS ?
> Quand H3 ET H4 sont violées simultanément : résidus à variances différentes **ET** corrélés entre eux. Cas typiques : autocorrélation temporelle, données panel, structure spatiale.

GLS généralise WLS : au lieu d'une matrice de poids **diagonale** $W$ (variances seulement), on utilise une matrice de covariance **complète** $\Omega$ qui capture aussi les corrélations entre observations.

> [!warning] Critère et estimateur GLS
> $$\hat{\beta}_{\text{GLS}} = \arg\min_\beta (y - X\beta)^T \Omega^{-1} (y - X\beta)$$
> 
> où $\Omega$ est la matrice de covariance des résidus : $V(\varepsilon \mid X) = \sigma^2 \Omega$. La solution :
> 
> $$\boxed{\hat{\beta}_{\text{GLS}} = (X^T \Omega^{-1} X)^{-1} X^T \Omega^{-1} y}$$
> 
> Cas particuliers :
> - **$\Omega = I$** → on retrouve OLS
> - **$\Omega$ diagonale** → on retrouve WLS

> 💡 **L'astuce universelle : GLS = OLS sur données blanchies.** Comme $\Omega$ est définie positive, elle admet une décomposition de Cholesky $\Omega^{-1} = L L^T$. En posant $\tilde{y} = L^T y$ et $\tilde{X} = L^T X$, le critère GLS devient $\|\tilde{y} - \tilde{X}\beta\|^2$ — un OLS classique. La transformation par $L^T$ "**blanchit**" les résidus : on passe de résidus corrélés et hétéroscédastiques à des résidus iid de variance $\sigma^2$. C'est exactement la même idée qu'en WLS, généralisée au cas non-diagonal.

> [!example] Cas typique — séries temporelles avec autocorrélation AR(1)
> Si $\varepsilon_t = \phi \varepsilon_{t-1} + u_t$ avec $u_t \sim \mathcal{N}(0, \sigma^2)$ iid, alors la matrice $\Omega$ a une structure de **Toeplitz** :
> 
> $$\Omega = \frac{1}{1-\phi^2} \begin{pmatrix} 1 & \phi & \phi^2 & \cdots \\ \phi & 1 & \phi & \cdots \\ \phi^2 & \phi & 1 & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$
> 
> En estimant $\phi$ (typiquement par les résidus OLS), on construit $\hat{\Omega}$ et on applique GLS. C'est ce que fait la procédure de **Cochrane-Orcutt** classique en économétrie.

> [!note]- Limite pratique de GLS
> En pratique, **on ne connaît jamais $\Omega$** — il faut l'estimer à partir des données, ce qui donne le **FGLS** (Feasible GLS). Le risque : si l'estimation de $\hat{\Omega}$ est mauvaise, FGLS peut être pire que OLS. Pour cette raison, on utilise souvent OLS avec des **écart-types robustes** (Newey-West, cluster-robust) plutôt que FGLS — moins efficient mais plus robuste à la mauvaise spécification de $\Omega$.

### D. RLS — Restricted Least Squares

> [!warning] Quand utiliser RLS ?
> Quand on veut **imposer des contraintes linéaires** sur les coefficients $\beta$ — non pas parce qu'une hypothèse Gauss-Markov est violée, mais parce qu'on a une **information externe** (théorique, structurelle) qu'on veut intégrer au modèle.

Toute contrainte linéaire sur $\beta$ peut s'écrire :

$$\beta = S\gamma + s$$

où $S$ et $s$ sont fixés et $\gamma$ est le nouveau vecteur de paramètres libres (dimension réduite). Une fois cette paramétrisation fixée, on substitue dans le critère OLS et on minimise sans contrainte.

> [!warning] Estimateur RLS
> $$\boxed{\hat{\beta}_R = S(S^T X^T X S)^{-1} S^T X^T (y - Xs) + s}$$
> 
> On retrouve OLS classique sur les données transformées $X_R = XS$, $y_R = y - Xs$, puis on remappe vers $\beta$ par $\beta_R = S\hat{\gamma} + s$.

> 💡 **L'idée géométrique.** OLS classique projette $Y$ sur tout le sous-espace engendré par les colonnes de $X$. RLS projette sur un **sous-espace plus petit** — celui qui satisfait la contrainte $\beta = S\gamma + s$. Géométriquement, on rajoute une contrainte qui restreint l'ensemble des $\beta$ admissibles, et on cherche le point de cet ensemble qui minimise la distance à $y$.

> [!example] Cas typique 1 — Exclusion (zero restrictions)
> Le cas le plus simple. On veut forcer certains coefficients à zéro. Avec $p = 5$ et la contrainte $\beta_2 = \beta_4 = 0$ :
> 
> $$\beta = \begin{pmatrix} \beta_1 \\ 0 \\ \beta_3 \\ 0 \\ \beta_5 \end{pmatrix} = \underbrace{\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}}_{S} \begin{pmatrix} \gamma_1 \\ \gamma_2 \\ \gamma_3 \end{pmatrix} + \underbrace{\begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}}_{s = 0}$$
> 
> Résultat : on retire simplement $X_2$ et $X_4$ du modèle et on régresse $Y$ sur $X_1, X_3, X_5$. RLS = sélection manuelle de variables.

> [!example] Cas typique 2 — Neutralité par secteur (ton terrain Barra)
> Dans un modèle factoriel équités, on impose souvent que les **rendements factoriels par secteur somment à zéro** :
> 
> $$\sum_{k \in \text{secteurs}} \beta_k = 0$$
> 
> C'est une seule contrainte linéaire qui peut s'écrire $R\beta = 0$ avec $R = (1, 1, \ldots, 1, 0, \ldots, 0)$ (1 sur les coefficients de secteur, 0 ailleurs). Cette contrainte garantit que le facteur "marché global" n'est pas double-compté — il est porté par l'intercept seul, et les secteurs ne capturent que la **dispersion relative** entre eux. C'est exactement ce type de contrainte que tu manipulais sur Barra.

> [!note]- Lien avec les multiplicateurs de Lagrange
> Une autre façon d'aborder RLS : Lagrangien. Pour la contrainte $R\beta = q$, on minimise $\|y - X\beta\|^2 + 2\lambda^T (R\beta - q)$. La solution est :
> 
> $$\hat{\beta}_R = \hat{\beta}_{\text{OLS}} - (X^T X)^{-1} R^T \big[R(X^T X)^{-1} R^T\big]^{-1} (R\hat{\beta}_{\text{OLS}} - q)$$
> 
> C'est OLS **corrigé** par un terme qui projette sur la contrainte. Forme alternative équivalente à celle via $S, s$, parfois plus pratique selon la contrainte.

### E. Robust LS — robustesse aux outliers

> [!warning] Quand utiliser Robust LS ?
> Quand quelques observations aberrantes (outliers, points influents — cf. II.C) peuvent **tirer la droite** hors de la tendance principale. OLS minimise une somme de **carrés** des résidus, ce qui pénalise très fort les grandes erreurs : un seul outlier peut dominer le critère.

> [!warning] Idée centrale
> On remplace la fonction de perte quadratique $\rho(r) = r^2$ par une fonction qui croît **moins vite** pour les grands résidus :
> 
> $$\hat{\beta}_{\text{Robust}} = \arg\min_\beta \sum_{i=1}^n \rho(y_i - x_i^T \beta)$$
> 
> Choix classiques :
> - **Huber** : quadratique près de 0, linéaire pour les grands résidus
> - **Tukey biweight** : annule complètement les contributions des résidus très grands
> - **L1 (LAD)** : $\rho(r) = |r|$ — médiane plutôt que moyenne

> 💡 **L'intuition.** OLS = $\rho(r) = r^2$ : un résidu de 10 pèse 100, un résidu de 100 pèse 10 000. Les outliers dominent. Avec Huber, au-delà d'un certain seuil $k$, $\rho(r) \approx k|r|$ — un résidu de 100 pèse seulement $\sim 100 k$. L'outlier ne peut plus tirer la droite. C'est un **trade-off entre efficacité et robustesse** : on perd un peu d'efficience sous les hypothèses Gauss-Markov pures, mais on gagne énormément si quelques points sont contaminés.

> [!example] Reprise de l'exemple obs 50/51 (II.C)
> Sur le dataset simulé avec $y = 2 + 3x + \varepsilon$ plus l'outlier (obs 50, $r^* > 3$) :
> 
> - **OLS** : la droite est légèrement déviée par l'outlier, $\hat{\beta} \approx 2.95$ au lieu du vrai $3$.
> - **Huber** : la droite reste presque parfaitement à $\hat{\beta} \approx 3.00$ — l'outlier est "écrasé" par la fonction de perte.
> 
> En pratique, sur des datasets réels avec ~5% de contamination (industries comme la finance ou l'imagerie médicale), Huber peut faire une différence très significative.

> [!note]- M-estimateurs et IRLS
> Robust LS s'inscrit dans la famille des **M-estimateurs** (Huber 1964). En pratique on les calcule par **IRLS** (Iteratively Reweighted Least Squares) : on fait un OLS, on calcule les résidus, on assigne des poids inverses (grands résidus → petits poids), on refait WLS avec ces poids, on itère jusqu'à convergence. Concrètement, c'est WLS où les poids sont **adaptatifs** au lieu d'être fixés a priori.

### F. Quantile Regression

> [!warning] Quand utiliser Quantile Regression ?
> Quand on s'intéresse à **autre chose que la moyenne** de $Y \mid X$ : les queues, la médiane, la dispersion conditionnelle, ou plus généralement la **distribution entière** de $Y$ sachant $X$. C'est aussi la réponse propre quand l'hétéroscédasticité est forte et que les hypothèses Gauss-Markov sont fragiles.

**(i) L'idée centrale — changer de cible**

OLS estime $E(Y \mid X)$ : la **moyenne conditionnelle**. Quantile Regression (QR) estime $Q_\tau(Y \mid X)$ : le **quantile $\tau$** de la distribution conditionnelle, pour n'importe quel $\tau \in (0, 1)$ choisi.

- $\tau = 0.5$ → médiane conditionnelle (= LAD, cas particulier de Robust LS, cf. IV.E)
- $\tau = 0.05$ → queue basse (la VaR à 5% en finance)
- $\tau = 0.95$ → queue haute
- En estimant plusieurs $\tau$ simultanément, on reconstitue la **distribution conditionnelle entière** de $Y \mid X$

**(ii) La pinball loss — l'astuce qui rend tout possible**

Le miracle de QR est qu'on peut estimer un quantile sans hypothèse sur la loi de $Y$, juste en changeant la fonction de perte.

> [!warning] Pinball loss (check function)
> $$\hat{\beta}_\tau = \arg\min_\beta \sum_{i=1}^n \rho_\tau(y_i - x_i^T \beta)$$
> 
> avec
> $$\rho_\tau(r) = \begin{cases} \tau \cdot r & \text{si } r \geq 0 \\ (\tau - 1) \cdot r & \text{si } r < 0 \end{cases} = r \cdot (\tau - \mathbb{1}_{r < 0})$$
> 
> Cette perte pénalise **asymétriquement** les résidus positifs et négatifs avec poids $\tau$ et $1-\tau$. Pour $\tau = 0.5$, la perte est symétrique → on retrouve $|r|/2$, soit la médiane (LAD).

> 💡 **Pourquoi ça marche.** Minimiser $E[\rho_\tau(Y - q)]$ par rapport à $q$ donne **exactement** $q = Q_\tau(Y)$. C'est la version asymétrique du résultat *"minimiser l'erreur quadratique donne la moyenne, minimiser l'erreur absolue donne la médiane"*. La pinball loss généralise à n'importe quel quantile.

> [!note]- Pas de solution fermée — résolution par programmation linéaire
> Contrairement à OLS, la pinball loss n'est pas différentiable en 0. Pas de formule type $(X^TX)^{-1}X^Ty$. Mais le problème se reformule comme un **programme linéaire** (Koenker & Bassett 1978) qu'on résout par simplexe ou méthode intérieure. En pratique : `statsmodels.regression.quantile_regression.QuantReg` en Python, package `quantreg` en R.

**(iii) Lien avec les autres variantes d'OLS**

QR n'est pas isolée dans cette section IV. Le tableau de IV.A se complète naturellement :

| Variante | Cible estimée | Fonction de perte |
|---|---|---|
| **OLS** | $E(Y \mid X)$ — moyenne | $\rho(r) = r^2$ |
| **Robust LS (Huber)** | intermédiaire moyenne/médiane | quadratique puis linéaire |
| **LAD** ($\tau = 0.5$) | $\text{Med}(Y \mid X)$ — médiane | $\rho(r) = \lvert r \rvert$ |
| **Quantile Regression** | $Q_\tau(Y \mid X)$ — quantile $\tau$ | $\rho_\tau(r) = r(\tau - \mathbb{1}_{r<0})$ |

QR est donc la **généralisation naturelle** : OLS, LAD, et Robust LS sont tous des cas particuliers (ou cousins) du même jeu — changer la fonction de perte change la cible estimée.

**(iv) Visualisation — OLS vs QR**

![[qr_motivation.png]]

**Figure 20.** OLS vs Quantile Regression sur des données simulées `salaire ~ éducation` où le rendement de l'éducation **dépend du quantile** (talent latent qui module la pente individuelle). **Gauche** : OLS estime une seule pente $\hat{\beta}_{\text{OLS}} = 3.0$ k€/an — *"un an d'études rapporte +3 k€ en moyenne"*. **Droite** : QR à trois quantiles révèle que la pente passe de **+1.5 k€/an** pour les bas salaires ($\tau = 0.10$) à **+4.6 k€/an** pour les hauts salaires ($\tau = 0.90$) — un **facteur 3×** d'écart entièrement masqué par la moyenne. La médiane verte ($\tau = 0.5$) coïncide avec OLS, ce qui est attendu : sur cette simulation symétrique, moyenne et médiane se rejoignent. L'**éventail des droites** est la signature visuelle de l'hétérogénéité de l'effet — là où OLS écrase tout en un seul nombre.

> 💡 **A retenir.** Le rendement de l'éducation **n'est pas une constante** — il dépend de où on est dans la distribution. OLS te donne *un* nombre. QR te donne *une fonction de $\tau$*. C'est strictement plus d'information, sans hypothèse supplémentaire.

**(v) Cas applicatifs**

> [!example] Trois usages typiques de la Quantile Regression
> 
> **1. Effets hétérogènes — quand l'effet de $X$ dépend du quantile**
> 
> *Exemple canonique* : rendement de l'éducation sur le salaire (figure ci-dessus). OLS te donne *"+X k€ par an d'études en moyenne"* ; QR révèle que les hauts salaires bénéficient bien plus d'une année supplémentaire que les bas salaires. Même logique pour : impact d'un traitement médical (effets différents sur les patients fragiles vs robustes), effet du temps de sommeil sur la performance cognitive, effet d'une politique économique (les pauvres et les riches ne réagissent pas pareil).
> 
> **2. Prédiction sous incertitude — vrais intervalles de prédiction sans hypothèse de loi**
> 
> Reprend le problème du callout `[!danger]` en III.D.1 : les IP classiques reposent sur H1–H6 et sont mal calibrés dès qu'une hypothèse pète. **QR donne directement un IP empirique** : on régresse $\tau = 0.025$ et $\tau = 0.975$ et on obtient un IP à 95% sans supposer la normalité ni l'homoscédasticité. C'est l'alternative propre, et c'est devenu standard en deep learning sous le nom de **conformal prediction** dans sa version moderne.
> 
> **3. Robustesse — cas particulier $\tau = 0.5$**
> 
> Quand $\tau = 0.5$, QR estime la médiane conditionnelle (= LAD, cf. IV.E). Insensible aux outliers, fonctionne sur des distributions à queues lourdes. C'est le pont naturel entre QR et Robust LS.
> 
> **4. Finance — modélisation directe de la queue**
> 
> En quant equity / risk management : la **VaR conditionnelle** à 5% c'est exactement $Q_{0.05}(R_t \mid \text{facteurs}_t)$. QR permet de modéliser comment des facteurs (VIX, momentum, taille) influencent **la queue gauche** des rendements — pas la moyenne. La littérature CAViaR (Engle & Manganelli 2004) en est l'application directe. Même logique pour le stress-testing : voir comment la dispersion conditionnelle s'étale en régime de stress.

> [!note]- Extension non-linéaire — Quantile Regression Forests
> Pour des relations non-linéaires entre $X$ et les quantiles de $Y$, l'analogue forestier existe : **Quantile Regression Forests** (Meinshausen 2006). On garde la structure des Random Forests mais on retient toute la distribution empirique de $Y$ dans chaque feuille au lieu de la moyenne. Très utilisé dans les contextes ML modernes pour la quantification d'incertitude.

---

## V. Régularisation (seulement pour la prédiction)

### A. Principe

La multicolinéarité fait exploser la variance de $\hat{\beta}$ — les coefficients sont instables. Plutôt que choisir entre modèle simple et modèle complexe, on prend un modèle flexible et on **pénalise la complexité** via un hyperparamètre $\lambda$. On part de la loss OLS :

$$\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

> 💡 **L'idée en une phrase.** On ne change pas la nature du modèle — on garde la régression linéaire. On ajoute juste une **pénalité** sur la taille des coefficients. Plus $\lambda$ est grand, plus on contraint les coefficients à être petits. Au prix d'un léger biais, on gagne énormément en variance.

> [!danger] Pourquoi "seulement pour la prédiction" ?
> La régularisation **casse l'interprétation des coefficients**. C'est un trade-off conscient : on accepte un biais pour gagner en variance prédictive. Quatre conséquences :
> 
> 1. **$\hat{\beta}_{\text{Ridge}}$ est biaisé** : $E[\hat{\beta}_{\text{Ridge}}] \neq \beta$. Le coefficient ne pointe plus vers le vrai effet causal — il est **shrinké vers 0** par construction.
> 2. **L'interprétation ceteris paribus disparait.** $\hat{\beta}_j = 0.3$ ne se lit plus *"+1 unité de $X_j$ → +0.3 unité de $Y$ à autres variables fixées"*, mais *"l'effet après contraction par la pénalité"* — sans lecture métier directe.
> 3. **Les tests Student/Fisher du Section III ne s'appliquent plus.** La distribution de $\hat{\beta}_{\text{Ridge}}$ n'est plus centrée sur $\beta$, donc les p-values du `summary()` n'ont plus de sens probabiliste valide.
> 4. **Les coefficients dépendent de $\lambda$.** Pas un coefficient unique mais une **trajectoire** $\hat{\beta}(\lambda)$. Lequel reporter ? À $\lambda$ optimal par CV ? À $\lambda = 0$ ? Pas de bonne réponse.
> 
> **Pour Lasso, c'est encore pire** : si Lasso met $\hat{\beta}_j = 0$, **ça ne veut pas dire que $\beta_j = 0$ dans la réalité**. Ça veut juste dire que dans cet échantillon, avec ce $\lambda$, Lasso a choisi de ne pas la garder. Avec un autre $\lambda$ ou un autre tirage, la sélection peut changer radicalement.
> 
> **Règle pratique :**
> - **Prédiction out-of-sample** → Ridge/Lasso, no problem. C'est leur terrain.
> - **Inférence / interprétation causale** → **OLS pur**, jamais régularisé. Si multicolinéarité bloque, on retire des variables ou on utilise des instruments (cf. II.A.H2).
> 
> > 💡 **Si tu veux vraiment les deux** — régulariser ET faire de l'inférence — il existe des cadres modernes : **post-selection inference** (Lee et al. 2016), **debiased Lasso** (van de Geer et al. 2014), **double machine learning** (Chernozhukov et al. 2018). Standard en économétrie causale moderne quand on a beaucoup de prédicteurs. Hors-scope pour cette note, mais bon à mentionner en entretien.

### B. Ridge — pénalité L2

> [!warning] Loss Ridge
> $$\mathcal{L}_{\text{Ridge}}(\theta) = \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j=1}^p \theta_j^2$$
> 
> Solution fermée (en dérivant et annulant) :
> 
> $$\hat{\theta}_{\text{Ridge}} = (X^TX + \lambda I)^{-1}X^Ty$$

### C. Lasso — pénalité L1

> [!warning] Loss Lasso
> $$\mathcal{L}_{\text{Lasso}}(\theta) = \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \lambda \sum_{j=1}^p |\theta_j|$$
> 
> Pas de solution fermée — résolution par algorithme itératif (coordinate descent, LARS).

> 💡 **L'effet de $\lambda$.**
> - $\lambda \to 0$ : on retrouve OLS (faible biais, forte variance).
> - $\lambda \to \infty$ : tous les $\theta_j \to 0$, le modèle prédit une constante (fort biais, variance nulle).
> - $\lambda$ intermédiaire : on contracte les coefficients, introduisant un peu de biais pour réduire la variance.
> 
> Le compromis biais-variance se règle via $\lambda$, choisi typiquement par validation croisée.

### D. Différence géométrique

La solution régularisée est le premier point de contact entre les ellipses de la loss OLS et la contrainte. La différence entre Ridge et Lasso est géométrique :

![Géométrie Ridge vs Lasso](images/2-Statistiques/A_Frequentist/regression-lineaire/im5.png)

**Figure 19.** Géométrie Ridge vs Lasso. À gauche, la contrainte sphérique de Ridge provoque une tangence hors des axes : les coefficients sont contractés mais non nuls. À droite, la contrainte losange de Lasso provoque une tangence sur un coin : certains coefficients sont exactement nuls.

- **Ridge** : tangence rarement sur un axe → $\theta_j \neq 0$, coefficients contractés vers zéro.
- **Lasso** : coins sur les axes → $\theta_j = 0$ exactement. **Lasso fait de la sélection de variables automatique.**

> 💡 **Quand utiliser quoi ?** Ridge est préférable quand **beaucoup de features contribuent un peu** (effet diffus). Lasso quand **seules quelques features comptent vraiment** (sparsité). En pratique, on essaie souvent **Elastic Net** qui combine les deux pénalités : $\lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2$.
