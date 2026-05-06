---
title: Régression linéaire
order: 1
---

# Régression linéaire

> Cette note couvre la régression linéaire : le modèle, les hypothèses de Gauss-Markov, l'inférence, les variantes d'OLS et la régularisation. Le fil rouge pour les diagnostics est le dataset `Auto` (ISLR), qui présente naturellement plusieurs des pathologies classiques (non-linéarité, hétéroscédasticité, clustering, multicolinéarité).

## I. Le modèle

### A. Vue classique

> [!warning] Modèle de régression linéaire simple
> On observe $n$ couples $(x_i, y_i)$. On suppose que $Y$ est une fonction linéaire de $X$ plus un bruit :
> 
> $$y_i = \alpha + \beta x_i + \varepsilon_i$$
> 
> Les $\varepsilon_i$ sont les **résidus** — tout ce que le modèle ne capture pas.

Géométriquement, chaque résidu est la distance verticale entre le point observé $y_i$ et la valeur prédite $\hat{y}_i = \alpha + \beta x_i$.

![Vue classique de la régression linéaire simple|458](images/2-Statistiques/Frequentist/regression-lineaire/im1.png)

**Figure 1.** Vue classique de la régression linéaire simple. Chaque résidu $\varepsilon_i$ est la distance verticale entre le point observé $y_i$ et la valeur prédite $\hat{y}_i = \alpha + \beta x_i$.

### B. Vue géométrique dans $L^2$

On travaille dans $L^2$, muni du produit scalaire $\langle X, Y \rangle = E(XY)$. Dans cet espace, **les vecteurs sont des variables aléatoires**.

> [!warning] Décomposition par projection orthogonale
> $E(Y|X)$ est la **projection orthogonale** de $Y$ sur le sous-espace $L^2_X$ des fonctions de $X$. Cela donne une décomposition naturelle de $Y$ en deux parties orthogonales :
> 
> $$Y = E(Y|X) + \varepsilon$$
> 
> Le résidu $\varepsilon = Y - E(Y|X)$ est **orthogonal à $L^2_X$** : $E[\varepsilon \cdot f(X)] = 0$ pour toute fonction $f$.

Deux conséquences immédiates :

- $E(\varepsilon) = 0$ — en prenant $f = 1$
- $\text{cov}(\varepsilon, X) = 0$ — en prenant $f = X$

> 💡 **L'idée en une phrase.** Ces propriétés ne sont pas des hypothèses qu'on impose : elles **découlent directement de la géométrie**. La régression "à la main" (chercher $\alpha, \beta$ qui minimisent l'erreur quadratique) revient *exactement* à projeter $Y$ orthogonalement sur le sous-espace des fonctions affines de $X$. On retrouvera cette idée partout — dans la dérivation MCO, dans le $R^2$ comme cosinus carré, dans le Pythagore SCT = SCE + SCR.

Quand on restreint $E(Y|X)$ aux fonctions **linéaires**, on obtient : $E(Y|X) = \alpha + \beta X$.

![Interprétation géométrique dans L²|407](images/2-Statistiques/Frequentist/regression-lineaire/im2.png)

**Figure 2.** Interprétation géométrique dans $L^2$. $E(Y|X)$ est la projection orthogonale de $Y$ sur $L^2_X$. Le résidu $\varepsilon = Y - E(Y|X)$ est orthogonal à tout le sous-espace $L^2_X$.

### Geometry

\textbf{(iii) Geometry.} Estimation of $y$ using OLS regression can be visualized as the orthogonal projection of the vector $y$ onto the column space of $X$. The estimated error term, epsilon, is the orthogonal distance between the projection and the true vector $y$. Figure 1 shows this projection for a $y$ that is regressed on two explanatory variables, $X_1$ and $X_2$.

donc en gros qd je pose que $\hat{\beta}=(X^TX)^{-1}X^T y$ - en gros $X^Ty$ est le vecteur des produits scalaires entre chaque colonne de X et $y$. Comment $y$ se projette sur chaque direction $X_j$ individuellement. 
Et le $(X^TX)^{-1}$ c'est la matrice de Gram des colonnes de $X$ en gros si les colonnes de $X$ sont orthogonales (corrélations nulles) alors $X^TX$ est diagonale sinon je sais pas quoi.

![[geo-1.png|430]]
### C. Vue probabiliste — OLS comme maximum de vraisemblance

> [!warning] Hypothèse probabiliste
> On suppose les résidus indépendants et gaussiens :
> 
> $$\varepsilon_i \sim \mathcal{N}(0, \sigma^2) \quad \text{i.i.d.}$$
> 
> Cela revient à modéliser $y_i \mid x_i \sim \mathcal{N}(x_i^T \beta,\ \sigma^2)$.

La log-vraisemblance des observations s'écrit :

$$\ell(\beta, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - x_i^T \beta)^2$$

Maximiser $\ell$ par rapport à $\beta$ revient à **minimiser** $\sum (y_i - x_i^T \beta)^2$ — c'est exactement le critère OLS.

> 💡 **OLS = MLE sous bruit gaussien.** Les moindres carrés ne sont pas un choix arbitraire : sous l'hypothèse de bruit gaussien iid, l'estimateur OLS coïncide avec l'estimateur du maximum de vraisemblance. C'est la justification probabiliste qui complète la justification géométrique de I.B. Trois lectures du même objet :
> - **Vue classique** : minimiser la somme des carrés des résidus
> - **Vue géométrique** : projeter $Y$ orthogonalement sur le sous-espace des fonctions linéaires
> - **Vue probabiliste** : maximiser la vraisemblance sous bruit gaussien

> [!note]- Lien avec Gauss-Markov
> Gauss-Markov donne le caractère BLUE de OLS **sans hypothèse gaussienne** — il faut juste H1–H5. L'hypothèse gaussienne supplémentaire apporte deux choses : (1) OLS devient MLE, donc *efficient* parmi tous les estimateurs (pas juste les linéaires), (2) la distribution de $\hat{\beta}$ devient exactement gaussienne, ce qui permet les tests de Student exacts (cf. III.A).

### D. Dérivation des estimateurs MCO

#### Dérivation par orthogonalité (régression simple)

Le résidu $\varepsilon = Y - \alpha - \beta X$ doit être orthogonal au sous-espace linéaire engendré par $1$ et $X$. **MCO** (Moindres Carrés Ordinaires, ou OLS en anglais) trouve $\hat{\alpha}$ et $\hat{\beta}$ en minimisant $\sum(y_i - \hat{y}_i)^2$ — ce qui revient exactement à cette condition d'orthogonalité.

> [!warning] Estimateurs MCO en régression simple
> $$\boxed{\alpha = E(Y) - \beta E(X)} \qquad \boxed{\beta = \frac{\text{cov}(X,Y)}{V(X)}}$$
> 
> La droite passe toujours par le point $\big(E(X), E(Y)\big)$ — $\alpha$ n'est qu'un ajustement de position. $\beta$ mesure combien de la variation de $X$ se transfère à $Y$.

> [!note]- Preuve — dérivation par orthogonalité
> **Condition 1 :** $\langle \varepsilon, 1 \rangle = 0$
> 
> $$E(Y - \alpha - \beta X) = 0 \implies \alpha = E(Y) - \beta E(X)$$
> 
> **Condition 2 :** $\langle \varepsilon, X \rangle = 0$
> 
> $$E\big[(Y - \alpha - \beta X)X\big] = 0 \implies E(YX) - \alpha E(X) - \beta E(X^2) = 0$$
> 
> On substitue $\alpha = E(Y) - \beta E(X)$ :
> 
> $$E(YX) - E(Y)E(X) = \beta\big(E(X^2) - E(X)^2\big)$$
> 
> Or $E(XY) - E(X)E(Y) = \text{cov}(X,Y)$ et $E(X^2) - E(X)^2 = V(X)$, donc :
> 
> $$\beta = \frac{\text{cov}(X,Y)}{V(X)}$$

#### Dérivation matricielle (régression multiple)

Avec $X \in \mathbb{R}^{n \times (p+1)}$ (la première colonne étant des 1 pour l'intercept), $\beta \in \mathbb{R}^{p+1}$ et $y \in \mathbb{R}^n$, le critère OLS s'écrit :

$$\hat{\beta} = \arg\min_\beta \|y - X\beta\|^2$$

> [!warning] Estimateur OLS sous forme matricielle
> $$\boxed{\hat{\beta} = (X^T X)^{-1} X^T y}$$
> 
> C'est *la* formule à connaître par cœur en entretien. Valable tant que $X^T X$ est inversible (i.e. pas de multicolinéarité parfaite, cf. H5).

> [!note]- Preuve — gradient nul
> On développe le critère :
> 
> $$\|y - X\beta\|^2 = (y - X\beta)^T(y - X\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta$$
> 
> Le gradient par rapport à $\beta$ :
> 
> $$\nabla_\beta \|y - X\beta\|^2 = -2 X^T y + 2 X^T X \beta$$
> 
> Annuler le gradient donne les **équations normales** :
> 
> $$X^T X \beta = X^T y$$
> 
> Si $X^T X$ est inversible : $\hat{\beta} = (X^T X)^{-1} X^T y$.

> 💡 **Lecture géométrique des équations normales.** $X^T X \hat{\beta} = X^T y$ signifie $X^T (y - X\hat{\beta}) = 0$, soit $X^T \hat{\varepsilon} = 0$ : les résidus sont **orthogonaux aux colonnes de $X$**. C'est exactement la condition d'orthogonalité de I.B, écrite en coordonnées. La géométrie et le calcul disent la même chose.

> [!note]- La hat matrix $H = X(X^T X)^{-1} X^T$
> En substituant $\hat{\beta}$, on obtient $\hat{y} = X \hat{\beta} = H y$ avec $H = X(X^T X)^{-1} X^T$. Cette matrice $H$ ("hat matrix" parce qu'elle "met le chapeau" sur $y$) est la **matrice de projection orthogonale** sur l'espace colonne de $X$. Propriétés : symétrique, idempotente ($H^2 = H$), $\text{tr}(H) = p+1$. Elle réapparaîtra dans le calcul du leverage (cf. II.C).

### E. Interprétation des coefficients

#### Prédicteurs continus

$\beta$ se lit ainsi : **si $X$ augmente d'une unité, $Y$ augmente en moyenne de $\beta$ unités**, à valeurs des autres prédicteurs constantes.

![[im1-2 (1).png|377]]

**Figure 3.** Lecture visuelle de $\beta$ : passer de $x$ à $x+1$ sur l'axe horizontal correspond à un changement vertical de $\beta$ unités sur la droite ajustée.

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

**Figure 4.** Régression sur dummy = comparaison de moyennes. Les deux nuages de points représentent les deux groupes ; les losanges noirs marquent les moyennes empiriques. La droite OLS passe **exactement** par les deux moyennes : $\beta_0$ est la hauteur de la moyenne du groupe A (à $x=0$), et $\beta_1$ est l'écart vertical entre les deux moyennes.

> 💡 **Régression sur dummy = t-test.** Tester $H_0 : \beta_1 = 0$ via la statistique de Student de la régression revient *exactement* à tester $H_0 : \overline{y}_A = \overline{y}_B$ via un t-test classique de comparaison de moyennes. Les deux approches donnent **la même p-value**. Ce n'est pas une coïncidence : la régression linéaire englobe les comparaisons de moyennes comme cas particulier. C'est le premier exemple d'un fait plus général : *régression sur dummies = ANOVA*.


pareil ici avoir une table de t test pour vraiment comprendre le rapport entre les deux. 

##### Généralisation à $K$ groupes — le modèle ANOVA

Quand le prédicteur catégoriel a $K$ modalités (par exemple 3 secteurs : Tech, Oil, Other), on généralise naturellement le modèle à deux groupes :

> [!warning] Modèle ANOVA à un facteur
> $$y_{ij} = \mu + \tau_j + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$
> 
> avec $j = 1, \ldots, K$ groupes (populations) et $i = 1, \ldots, n_j$ observations dans chaque groupe.
> 
> Chaque observation s'écrit comme la somme de **trois composantes** :
> - **$\mu$** — le niveau de base (la moyenne globale, ou le niveau de la référence selon la paramétrisation)
> - **$\tau_j$** — l'effet spécifique du groupe $j$ (écart par rapport au niveau de base)
> - **$\varepsilon_i$** — le résidu individuel, propre à l'observation

![Décomposition de la variance dans un cas à trois populations A, B et C fictives.|590](https://biodatascience-course.sciviews.org/sdd-umons-2018/10-Variance_files/figure-html/anova1-1.svg)

**Figure 5.** Décomposition de la variance dans le modèle ANOVA à trois groupes. La ligne pointillée noire est la **moyenne globale $\mu$**. Pour chaque observation, l'écart **total** à la moyenne globale se décompose en : (1) un écart **inter-groupes** $\tau_j$ qui mesure la différence entre la moyenne du groupe et la moyenne globale, et (2) un écart **intra-groupe** $\varepsilon_i$ qui mesure la variabilité individuelle au sein du groupe. C'est cette décomposition $\text{total} = \text{inter} + \text{intra}$ qui fonde le test de Fisher en ANOVA.

> 💡 **Le problème d'identification.** Le modèle a $K + 1$ paramètres ($\mu, \tau_1, \ldots, \tau_K$) mais on n'observe que $K$ moyennes de groupe. Il y a donc **un paramètre de trop** — on peut ajouter une constante $c$ à $\mu$ et la retirer à tous les $\tau_j$ sans changer les prédictions. Pour rendre le modèle identifiable, il faut imposer **une contrainte** sur les $\tau_j$. Selon la contrainte choisie, l'interprétation des coefficients change — mais les **prédictions sont toujours les mêmes**.

##### La paramétrisation par défaut — treatment constraint

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

##### Le piège à éviter — dummy variable trap

> 💡 **Le dummy variable trap.** Si on incluait les $K$ indicatrices (USA, Europe, Japon) **sans en retirer une** ET avec un intercept, leur somme vaudrait toujours 1 — exactement la colonne d'intercept. C'est de la **multicolinéarité parfaite** (cf. H5), $X^T X$ devient singulière, OLS ne peut plus tourner. Pour s'en sortir, deux options : soit on retire une catégorie (treatment constraint), soit on retire l'intercept (cf. paramétrisations alternatives ci-dessous).

##### Paramétrisations alternatives

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
> *Quand l'utiliser ?* Quand on veut une interprétation "écart à la moyenne globale" plutôt que "écart à un groupe de référence". C'est aussi la paramétrisation "naturelle" pour la décomposition de la variance illustrée Figure 5.

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
> > 💡 **Le punch.** En quant equity, *la* paramétrisation des dummies sectorielles est sum-to-zero pondérée par cap, jamais treatment. C'est pour ça qu'on parle de **factor returns** (rendements de facteurs, écarts au marché) et pas de "coefficients" — l'interprétation est *intrinsèque* au facteur, pas relative à un groupe choisi.

> [!note]- Récapitulatif — trois lectures du même modèle
> | Paramétrisation | Contrainte | Interprétation des coefficients |
> |---|:---:|---|
> | **Treatment** (défaut) | $\tau_1 = 0$ | $\beta_0$ = moyenne du groupe référence ; $\beta_k$ = différence vs référence |
> | **Sans intercept** | (intercept retiré) | $\tau_k$ = moyenne du groupe $k$ |
> | **Sum-to-zero** | $\sum \tau_j = 0$ | $\mu$ = moyenne globale ; $\tau_k$ = différence vs moyenne globale |
> 
> Les **trois donnent les mêmes prédictions $\hat{y}$** — c'est juste l'étiquette posée sur les coefficients qui change. On choisit selon ce qu'on veut lire directement.

##### Application — cross-section Barra

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

##### Lecture géométrique — trois droites parallèles

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

**Figure 6.** Modèle ANCOVA `mpg ~ horsepower + origin` ajusté sur Auto. Les trois nuages de points (USA en rouge, Europe en bleu, Japon en vert) sont fittés par **trois droites parallèles** de pente commune $\hat{\beta}_1 = -0.134$. Les flèches verticales noires matérialisent les coefficients des dummies : $\hat{\beta}_{\text{Europe}} = +2.43$ et $\hat{\beta}_{\text{Japon}} = +5.18$ — à puissance égale, une voiture européenne consomme en moyenne 2.43 mpg de plus qu'une USA, et une japonaise 5.18 mpg de plus.

> [!example] Lecture concrète des coefficients estimés
> Sur Auto, les coefficients estimés sont $\hat{\beta}_0 = 35.94$, $\hat{\beta}_1 = -0.134$, $\hat{\beta}_{\text{Europe}} = +2.43$, $\hat{\beta}_{\text{Japon}} = +5.18$
> 
> - **$\hat{\beta}_0 = 35.94$** : intercept de la droite USA (mpg prédit pour une voiture américaine à hp = 0). Pas d'interprétation physique directe puisque hp = 0 n'existe pas, mais c'est le **point d'ancrage** de la droite de référence.
> - **$\hat{\beta}_1 = -0.134$** : à origine fixée, **chaque cheval supplémentaire fait baisser la mpg de 0.134**. Cette pente est la même pour les trois origines.
> - **$\hat{\beta}_{\text{Europe}} = +2.43$** : à hp égale, une voiture européenne fait **2.43 miles de plus par gallon** qu'une américaine — elle est plus économe.
> - **$\hat{\beta}_{\text{Japon}} = +5.18$** : à hp égale, une japonaise est encore plus économe — **5.18 mpg de plus** qu'une USA.
> 
> *Lecture économique* : à puissance comparable, les voitures japonaises et européennes des années 1970-80 étaient plus économes que les américaines (technologie moteur, poids, aérodynamisme). Le modèle ANCOVA capture ce fait : l'effet "origine" est un **décalage** indépendant de la puissance.

##### Limite du modèle ANCOVA — l'hypothèse de pentes parallèles

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
Figure 7. Modèle avec interactions ```mpg ~ horsepower * origin```.Trois pentes différentes par groupe (au lieu d'une pente commune comme dans la Figure 6). Les coefficients $\beta_4, \beta_5$ mesurent les différences de pente par rapport au groupe de référence USA. Tester leur nullité jointe (test de Fisher) revient à tester l'hypothèse de pentes parallèles de l'ANCOVA.

### F. Standardisation des prédicteurs

Faut-il standardiser (z-score) les prédicteurs avant de fitter une régression ? La réponse dépend de ce qu'on veut faire — mais en pratique en quant, **standardiser les continus par défaut** est la règle qui évite tous les pièges.

#### OLS pur est invariant à la mise à l'échelle

> [!warning] Invariance d'OLS
> Si on multiplie une colonne $X_j$ par une constante $c$, le coefficient $\hat{\beta}_j$ est divisé par $c$, et **les prédictions $\hat{y}$ restent identiques**. Le R², les résidus, et les p-values des tests Student individuels sont aussi invariants.

> [!note]- Démonstration
> Soit $\tilde{X} = XD$ avec $D$ matrice diagonale de mise à l'échelle. Alors :
> 
> $\hat{\beta}_{\tilde{X}} = (\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T y = (D X^T X D)^{-1} D X^T y = D^{-1} (X^T X)^{-1} X^T y = D^{-1} \hat{\beta}_X$
> 
> Donc $\tilde{X} \hat{\beta}_{\tilde{X}} = X D \cdot D^{-1} \hat{\beta}_X = X \hat{\beta}_X = \hat{y}$. Les prédictions sont strictement identiques.

Conséquence : pour OLS sans régularisation, **standardiser ne change rien au modèle** — on peut techniquement s'en passer. Mais ce n'est pas pour autant qu'il faut éviter de standardiser : dans les 4 cas suivants, c'est obligatoire ou très recommandé.

#### Quand il faut standardiser

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

#### La règle pratique

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
> > 💡 **Le punch.** Cette différence de traitement (z-score les continus, laisser les dummies) est la raison pour laquelle on peut **comparer entre eux les style factors** ("Momentum a fait +30 bps, Value a fait −20 bps") tout en gardant une **interprétation économique directe** des factor returns sectoriels ("Tech a outperformé de +2%").


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

> 💡 **Hiérarchie des violations.** Les 5 hypothèses ne sont pas équivalentes en termes de gravité. **H1 et H2** affectent l'estimateur lui-même : si elles sont violées, $\hat{\beta}$ devient **biaisé** — il pointe au mauvais endroit. **H3, H4, H5** affectent seulement la précision et l'inférence : $\hat{\beta}$ reste sans biais, mais les écart-types calculés par OLS sont faux, donc les tests sont invalides. À retenir pour les entretiens : seules H1 et H2 biaisent l'estimateur.

> [!example] Fil rouge — dataset Auto
> Pour illustrer les hypothèses on utilise le dataset `Auto` (ISLR). Variable cible : `mpg` (miles per gallon, consommation inverse). Prédicteur principal : `horsepower` (puissance du moteur en chevaux). 392 observations sur des modèles de voitures 1970-1982.
> 
> ![[Pasted image 20260505183850.png|477]]
> 
> **Figure 4.** Scatter plot brut `mpg ~ horsepower`. La relation est clairement décroissante (plus puissant → plus gourmand → moins de mpg) mais visiblement courbée et la dispersion change avec le niveau.

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
> **Figure 5.** Residual plot pour `mpg ~ horsepower`. Pour les valeurs prédites faibles ($\hat{y} \in [5, 12]$) et élevées ($\hat{y} \in [28, 35]$), les résidus sont systématiquement positifs ; pour les valeurs moyennes, ils sont négatifs. Cette **structure systématique** (et non aléatoire) signe la non-linéarité.
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
> **Figure 6.** Residual plot avec enveloppe ±2σ local pour `mpg ~ horsepower`. À gauche le modèle linéaire montre courbure + cône. À droite après ajout de $\text{horsepower}^2$ pour corriger H1 : la courbure disparaît mais **le cône reste**.
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
> **Figure 7.** Boxplot des résidus par origine pour `mpg ~ horsepower`. **Avant** inclusion de l'origine, les médianes sont nettement décalées par groupe — signature classique du clustering.
> 
> **Remède : inclure `origin` dans le modèle.** On passe de $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp}$ à $\widehat{\text{mpg}} = \beta_0 + \beta_1 \text{hp} + \beta_2 \mathbb{1}_{\text{Europe}} + \beta_3 \mathbb{1}_{\text{Japon}}$ (USA = référence, cf. section I.E sur les dummies).
> 
> ![[Pasted image 20260505191012.png]]
> 
> **Figure 8.** Comparaison avant/après. **Gauche** : sans `origin`, médianes décalées. **Droite** : avec `origin` inclus, les trois boîtes sont recentrées sur 0 — la violation H4 a disparu.
> 
> > 💡 **Le punch.** L'effet "origine" est passé du résidu vers le modèle. La même information a juste migré : $\hat{\beta}_{\text{Japon}} \approx +2$ capture ce que les résidus japonais portaient avant. **La violation H4 par clustering n'est PAS une propriété intrinsèque des données, c'est une propriété du modèle qu'on a choisi.** En incluant la variable de groupe, on l'élimine à la racine.

> [!note]- Lien profond H4 ↔ H2
> Le clustering est en fait **une variable omise déguisée**. "Mes résidus sont clusterisés par groupe" = "j'ai omis une variable de groupe corrélée à $Y$". Inclure la variable revient à la transférer du résidu vers le modèle. Le seul cas où H4 est **structurellement** violée (non réductible à une variable omise) est l'autocorrélation temporelle pure : $\varepsilon_t = \phi \varepsilon_{t-1} + u_t$ ne peut pas être "inclus" dans le modèle puisque $\varepsilon_{t-1}$ est inobservable.

#### H5 — Multicolinéarité

> [!warning] Définition (Multicolinéarité)
> Les colonnes de $X$ ne sont pas combinaisons linéaires les unes des autres. Si une colonne s'écrit comme combinaison des autres (par exemple $X_3 = X_1 + X_2$), $X^\top X$ n'est pas inversible et la formule MCO ne marche plus — c'est la **multicolinéarité parfaite**.

**Le problème (cas non parfait).** Même sans être parfaite, une corrélation forte entre prédicteurs cause des problèmes. Si $X_1$ et $X_2$ pointent presque dans la même direction dans $L^2$, projeter $Y$ dessus devient ambigu — une infinité de combinaisons $(\beta_1, \beta_2)$ donnent à peu près la même projection.

> 💡 **Le punch.** Le modèle **sait prédire** mais il **ne sait pas répartir les coefficients**. Conséquence : les variances de $\hat{\beta}_j$ explosent, les coefficients deviennent instables et changent radicalement quand on ajoute/retire d'autres variables corrélées.

![Multicolinéarité](images/2-Statistiques/Frequentist/regression-lineaire/im3.png)

**Figure 9.** Multicolinéarité : quand $X_1$ et $X_2$ pointent dans la même direction, projeter $Y$ dessus devient ambigu — les coefficients $\beta_1$ et $\beta_2$ ne sont plus identifiables individuellement.

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
> **Figure 10.** Heatmap des corrélations entre prédicteurs sur Auto. Les 4 premières variables (`horsepower`, `weight`, `displacement`, `cylinders`) forment un bloc fortement corrélé (toutes les corrélations $> 0.84$).
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
> **Figure 11.** **Gauche** : scatter brut. L'obs 50 (rouge) est un outlier visible au milieu du nuage en hauteur. L'obs 51 (orange) est extrême en X mais sur la droite OLS. **Droite** : plot diagnostic Leverage vs Studentized Residuals. L'obs 50 sort par l'**axe Y** (résidu standardisé > 3) → détecté comme outlier. L'obs 51 sort par l'**axe X** (leverage $h$ au-delà du seuil $3(p+1)/n$) → détecté comme high leverage. Aucun des deux n'est dans un coin (haut-droite ou bas-droite) → ni l'un ni l'autre n'est réellement *influent* sur $\hat{\beta}$.
> 
> > 💡 **Le plot de droite est universel.** En régression simple ($p=1$), tu peux voir un point extrême en $X$ directement sur le scatter. Mais en régression multiple ($p$ grand), le scatter 2D ne suffit plus — un point peut être extrême dans une **combinaison** des variables sans l'être sur aucune individuellement (genre 1m95 + 60kg). Le plot Leverage vs Residuals reste lisible à n'importe quelle dimension parce que $h_{ii}$ et $r^*_i$ sont des scalaires par observation. C'est le seul outil pour détecter ces points cachés en multi-D.

---

## III. Inférence & Tests

### A. Distribution de $\hat{\beta}$

On ajoute une hypothèse aux cinq de Gauss-Markov : les erreurs sont gaussiennes, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Alors $\hat{\beta}$ est une combinaison linéaire de variables gaussiennes, donc lui-même gaussien.

> [!warning] Distribution de $\hat{\beta}$ sous bruit gaussien
> $$\hat{\beta} \sim \mathcal{N}\!\left(\beta,\ \frac{\sigma^2}{S_{XX}}\right)$$
> 
> où $S_{XX} = \sum(x_i - \bar{x})^2$ est la dispersion empirique de $X$.

> [!note]- Rappel — biais et variance d'un estimateur
> Le **biais** d'un estimateur $\hat{\theta}$ est l'écart entre son espérance et la vraie valeur :
> 
> $$\text{bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$
> 
> La **variance** d'un estimateur mesure sa dispersion autour de son espérance :
> 
> $$V(\hat{\theta}) = E\big[(\hat{\theta} - E[\hat{\theta}])^2\big]$$

> 💡 **Trois choses à lire dans cette distribution.**
> - **Centré sur $\beta$** : $E[\hat{\beta}] = \beta$, donc $\text{bias}(\hat{\beta}) = 0$ — $\hat{\beta}$ est sans biais.
> - **Variance $= \sigma^2/S_{XX}$** : plus $X$ est dispersé, plus $\hat{\beta}$ est précis. C'est intuitif : si tous tes $x_i$ sont concentrés au même endroit, tu n'as aucune information sur la pente.
> - **Variance $\to 0$ quand $n \to \infty$** : $\hat{\beta}$ est convergent.

### B. Le $R^2$

#### Définition classique — la décomposition des sommes de carrés

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
Figure Visuellement le $R^2$ va capturer la différence entre la moyenne (aka pas d'information en rouge) et la ligne de régression (en violet) 

#### Lecture géométrique — Pythagore dans $L^2$

La décomposition SST = SSE + SSR n'est rien d'autre que **Pythagore** appliqué dans $L^2$ après centrage. On retire $\bar{y}$ partout et on obtient le triangle rectangle $\bar{y}, \hat{Y}, Y$.

> [!warning] $R^2$ géométrique
> $$R^2 = \cos^2 \theta$$
> 
> où $\theta$ est l'angle entre $Y - \bar{y}$ et le sous-espace $L^2_X$ (sous-espace des prédictions).

![Pythagore dans L² centré|500](images/2-Statistiques/Frequentist/regression-lineaire/im4.png)

**Figure 12.** Pythagore dans $L^2$ centré. La décomposition $\text{SST} = \text{SSE} + \text{SSR}$ est le théorème de Pythagore appliqué au triangle $\bar{y}, \hat{Y}, Y$. $R^2 = \cos^2\theta$.

> 💡 **Deux lectures du même objet.** $R^2 = \text{SSE}/\text{SST}$ (ratio de sommes de carrés) et $R^2 = \cos^2\theta$ (lecture géométrique) disent exactement la même chose. L'une est calculatoire et apparaît dans tous les outputs de logiciels ; l'autre est conceptuelle et permet de comprendre pourquoi $R^2 \in [0, 1]$ (un cosinus carré est toujours dans cet intervalle).

#### $R^2$ ajusté

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

### C. Test de Student sur $\hat{\beta}$

**La question.** Est-ce que $X$ a vraiment un effet sur $Y$, ou est-ce que $\hat{\beta} \neq 0$ par chance ? On teste $H_0 : \beta = 0$ contre $H_1 : \beta \neq 0$.

> [!warning] Statistique de Student
> $\sigma^2$ est inconnue en pratique, on l'estime par $\hat{\sigma}^2 = \text{SSR}/(n-2)$. On pose :
> 
> $$t = \frac{\hat{\beta}}{\hat{\sigma}/\sqrt{S_{XX}}}$$
> 
> Sous $H_0$, $t$ suit une loi de Student à $n-2$ degrés de liberté. Pour $n$ grand, on retient la règle :
> 
> $$|t| > 1.96 \implies \text{on rejette } H_0 \text{ au seuil } 5\%$$

> 💡 **La p-value.** C'est la probabilité d'observer un $|t|$ aussi grand si $H_0$ était vraie. Plus elle est petite, plus on est confiant que $\beta \neq 0$. Seuil classique : $p < 0.05$.

> [!warning] Intervalle de confiance sur $\beta$ à 95%
> $$\hat{\beta} \pm 1.96 \cdot \frac{\hat{\sigma}}{\sqrt{S_{XX}}}$$
> 
> C'est l'ensemble des valeurs de $\beta$ qu'on ne rejetterait pas au seuil 5%. Plus $S_{XX}$ est grand ($X$ dispersé), plus l'intervalle est étroit — plus on est précis.

### D. Prédiction — Intervalle de confiance vs Intervalle de prédiction

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
> $$\hat{y}^* \pm 1.96 \cdot \hat{\sigma}\sqrt{1 + \frac{1}{n} + \frac{(x^* - \bar{x})^2}{S_{XX}}}$$
> 
> Encadre une **nouvelle observation individuelle**. L'incertitude vient de **deux sources** : l'estimation de $\hat{\alpha}$ et $\hat{\beta}$ **plus** le bruit irréductible $\varepsilon^*$.

> 💡 **Le rôle du $+1$.** La seule différence entre les deux formules est le $+1$ sous la racine dans l'IP. Ce $1$ représente la variance du bruit $\varepsilon^*$ — irréductible peu importe la taille de l'échantillon. Même avec $n \to \infty$, on estimerait $\alpha + \beta x^*$ parfaitement, mais une observation individuelle resterait dispersée autour de cette moyenne. **On ne peut pas prédire le bruit.**

Les deux intervalles s'élargissent aussi quand $x^*$ s'éloigne de $\bar{x}$ — on extrapole loin des données, l'estimation devient moins fiable.

### E. Test de Fisher — test joint sur plusieurs coefficients

**La question.** Le test de Student de III.C teste **un coefficient à la fois**. Mais souvent on veut tester **un groupe de coefficients ensemble**. Exemples typiques :

- *Mes 4 dummies de secteur (Tech, Oil, Banque, Real Estate vs référence) sont-elles **jointement** significatives ?* — un test pour les 4 d'un coup, pas 4 tests Student séparés.
- *Ajouter `weight + displacement + cylinders` au modèle `mpg ~ hp` apporte-t-il quelque chose ?* — comparaison entre un modèle restreint et un modèle complet.
- *Le terme polynomial $\text{hp}^2$ et le terme d'interaction $\text{hp} \times \text{origin}$ sont-ils utiles ensemble ?*

> 💡 **Pourquoi pas juste plusieurs tests Student ?** Faire 4 tests Student séparés au seuil 5% donne une probabilité de faux positif **bien supérieure à 5%** sur l'ensemble (problème de tests multiples). Et surtout, ça ne capture pas l'**effet conjoint** : deux coefficients individuellement insignificants peuvent être conjointement très significatifs s'ils sont corrélés. Le test de Fisher répond directement à la bonne question.

#### Cadre — modèle restreint vs modèle complet

L'idée centrale : on compare deux modèles emboîtés.

> [!warning] Modèles emboîtés
> - **Modèle complet** ($M_1$) : utilise tous les $p$ prédicteurs, $\text{SSR}_1$ résiduel.
> - **Modèle restreint** ($M_0$) : on annule $q$ coefficients (par exemple $q=4$ dummies de secteur), il reste $p - q$ prédicteurs, $\text{SSR}_0$ résiduel.
> 
> **Hypothèse nulle** : $H_0 : \beta_{p-q+1} = \cdots = \beta_p = 0$ (les $q$ coefficients qu'on retire sont tous nuls).

Par construction, $\text{SSR}_0 \geq \text{SSR}_1$ : retirer des prédicteurs ne peut qu'augmenter les résidus. La question : **l'augmentation est-elle significative** ?

#### Décomposition de la variance — le fondement géométrique

Le test de Fisher repose sur la **décomposition fondamentale** $\text{SST} = \text{SSE} + \text{SSR}$ (cf. III.B). Plus précisément, on décompose la variance expliquée par groupes :

> [!warning] Sommes de carrés inter-groupes / intra-groupes
> Pour un facteur catégoriel à $K$ groupes (cas ANOVA), avec $\bar{y}_j$ la moyenne du groupe $j$ et $\bar{y}$ la moyenne globale :
> 
> $$\underbrace{\text{SC}_{\text{inter}} = \sum_{j=1}^K n_j (\bar{y}_j - \bar{y})^2}_{\text{variance expliquée par les groupes}}$$
> 
> $$\underbrace{\text{SC}_{\text{intra}} = \sum_{j=1}^K \sum_{i=1}^{n_j} (y_{ij} - \bar{y}_j)^2}_{\text{variance résiduelle}}$$
> 
> Et $\text{SC}_{\text{inter}} + \text{SC}_{\text{intra}} = \text{SST}$ (Pythagore).

![Décomposition de la variance dans un cas à trois populations A, B et C fictives.|590](https://biodatascience-course.sciviews.org/sdd-umons-2018/10-Variance_files/figure-html/anova1-1.svg)

**Figure 14.** *(Reprise de la Figure 5.)* Décomposition $\text{total} = \text{inter} + \text{intra}$. Le test de Fisher quantifie statistiquement cette intuition visuelle : si la variance **inter-groupes** (écarts entre moyennes de groupes) domine la variance **intra-groupe** (dispersion individuelle), alors les groupes diffèrent vraiment.

#### Statistique de Fisher

> [!warning] Statistique F
> $$F = \frac{(\text{SSR}_0 - \text{SSR}_1)/q}{\text{SSR}_1/(n - p - 1)}$$
> 
> Sous $H_0$, $F$ suit une **loi de Fisher** à $(q, n-p-1)$ degrés de liberté.
> 
> - **Numérateur** : $(\text{SSR}_0 - \text{SSR}_1)/q$ = gain de variance expliqué par les $q$ coefficients ajoutés, par degré de liberté
> - **Dénominateur** : $\text{SSR}_1/(n - p - 1)$ = variance résiduelle du modèle complet, par degré de liberté
> 
> Si $F$ est grand → les coefficients ajoutés expliquent beaucoup plus de variance que ce qu'on attendrait par hasard → on rejette $H_0$.

> 💡 **Interprétation en deux mots.** $F$ est le **ratio signal/bruit**. Numérateur = ce que les $q$ coefficients supplémentaires apportent. Dénominateur = la variance résiduelle de référence. Si signal > bruit (typiquement $F > 4$ pour des seuils usuels), alors les coefficients sont jointement significatifs.

#### Tableau ANOVA — la présentation classique

Dans le cas particulier d'un seul facteur catégoriel à $K$ groupes (ANOVA pure : $H_0$ = "tous les groupes ont la même moyenne"), on présente les calculs sous forme de tableau :

| Source | Degrés de liberté | Somme des carrés | Carré moyen | Statistique F |
|---|:---:|:---:|:---:|:---:|
| **Inter-groupes** (facteur) | $K - 1$ | $\text{SC}_{\text{inter}}$ | $\text{CM}_{\text{inter}} = \dfrac{\text{SC}_{\text{inter}}}{K-1}$ | $F = \dfrac{\text{CM}_{\text{inter}}}{\text{CM}_{\text{intra}}}$ |
| **Intra-groupes** (résidus) | $n - K$ | $\text{SC}_{\text{intra}}$ | $\text{CM}_{\text{intra}} = \dfrac{\text{SC}_{\text{intra}}}{n-K}$ | |
| **Total** | $n - 1$ | $\text{SST}$ | | |

> 💡 **Le carré moyen est juste "variance par degré de liberté"**. On divise chaque somme de carrés par ses ddl pour obtenir des **estimateurs de variance** comparables. Si la vraie variance inter et intra sont égales (= $H_0$ vraie), alors $F \approx 1$. Dès que $F$ s'éloigne nettement de 1, on rejette $H_0$.

#### Cas particuliers utiles

> [!note]- Cas 1 — Test de signification globale du modèle
> Quand $M_0$ est le modèle constant ($\hat{y}_i = \bar{y}$, aucun prédicteur) et $M_1$ est le modèle complet avec tous les $p$ prédicteurs, le test de Fisher devient :
> 
> $F = \frac{R^2/p}{(1-R^2)/(n-p-1)}$
> 
> Ce $F$ apparaît dans tout output `summary()` (R) ou `.summary()` (statsmodels) : "F-statistic" et "Prob (F-statistic)". Il teste $H_0$ : *"aucun prédicteur n'a d'effet"*. Si on rejette, le modèle est globalement informatif.

> [!note]- Cas 2 — Lien avec le test de Student
> Quand on teste **un seul** coefficient ($q = 1$), Fisher et Student sont équivalents :
> 
> $F_{(1, n-p-1)} = t_{(n-p-1)}^2$
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

---

## V. Régularisation

### A. Principe

La multicolinéarité fait exploser la variance de $\hat{\beta}$ — les coefficients sont instables. Plutôt que choisir entre modèle simple et modèle complexe, on prend un modèle flexible et on **pénalise la complexité** via un hyperparamètre $\lambda$. On part de la loss OLS :

$$\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

> 💡 **L'idée en une phrase.** On ne change pas la nature du modèle — on garde la régression linéaire. On ajoute juste une **pénalité** sur la taille des coefficients. Plus $\lambda$ est grand, plus on contraint les coefficients à être petits. Au prix d'un léger biais, on gagne énormément en variance.

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

![Géométrie Ridge vs Lasso](images/2-Statistiques/Frequentist/regression-lineaire/im5.png)

**Figure 13.** Géométrie Ridge vs Lasso. À gauche, la contrainte sphérique de Ridge provoque une tangence hors des axes : les coefficients sont contractés mais non nuls. À droite, la contrainte losange de Lasso provoque une tangence sur un coin : certains coefficients sont exactement nuls.

- **Ridge** : tangence rarement sur un axe → $\theta_j \neq 0$, coefficients contractés vers zéro.
- **Lasso** : coins sur les axes → $\theta_j = 0$ exactement. **Lasso fait de la sélection de variables automatique.**

> 💡 **Quand utiliser quoi ?** Ridge est préférable quand **beaucoup de features contribuent un peu** (effet diffus). Lasso quand **seules quelques features comptent vraiment** (sparsité). En pratique, on essaie souvent **Elastic Net** qui combine les deux pénalités : $\lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2$.
