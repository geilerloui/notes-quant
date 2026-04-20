## Introduction

| Algorithme | Auteur | Année |
|---|---|---|
| Simplex (Dantzig) | G. Dantzig | 1947 |
| Revised Simplex | Dantzig & Orchard-Hays | ~1950s |
| Gomory cuts | R. Gomory | 1958 |
| Branch & Bound | Land & Doig | 1960 |
| Branch & Cut | Padberg & Rinaldi | 1991 |

Un programme linéaire c'est maximiser une fonction affine sous des contraintes affines. L'ensemble réalisable $\mathcal{P} = \{x \mid Ax \leq b,\ x \geq 0\}$ est un **polyèdre convexe** — une intersection de demi-espaces. La clé : si une solution optimale existe, elle se trouve nécessairement à un **sommet** du polyèdre.

$$
\begin{aligned}
&\max~ z= c^T x\\
&\text{s.t.} \quad Ax \leq b, \quad x \geq 0
\end{aligned}
$$

**Pourquoi pas le Lagrangien ?** En optimisation différentiable, les conditions KKT via le Lagrangien cherchent un point où $\nabla_x \mathcal{L} = 0$. Ici la fonction objectif $c^Tx$ est linéaire : son gradient $c$ est constant et ne s'annule jamais. L'optimum n'est donc jamais à l'intérieur du polytope — il est toujours sur la frontière, et même à un sommet. Le Lagrangien donnerait $c = A^T\lambda$, un système linéaire qu'on ne peut résoudre que si $A$ est carré et inversible — ce qui n'est pas le cas en général. On exploite donc directement la structure géométrique.

Trois approches pour trouver ce sommet optimal :

- **Vertex enumeration** — tester tous les $\binom{n}{m}$ sommets possibles. Correct mais combinatoire : inutilisable dès que $n$ et $m$ sont grands.
- **Solution graphique** — en 2D, on pousse les courbes de niveau de $z$ jusqu'au dernier sommet touché. Bonne intuition, pas généralisable.
- **Simplex (Dantzig, 1947)** — on se déplace de sommet en sommet le long des arêtes, en ne choisissant que les directions qui font monter $z$. Exponentiel au pire cas, mais $O(m)$ itérations en pratique.

Visuellement, $z = c^Tx$ est un hyperplan au-dessus du polyèdre — on cherche le sommet qui atteint la plus grande hauteur.

![[simplexe-3d-fonction-objectif.png|416]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 1. Fonction objectif $z = 3x_1 + 5x_2$ au-dessus du polyèdre réalisable. L'optimum est au sommet $(3,1)$.</p>

# 1. Programmation linéaire

## 1.1 Résolution — Méthode du Simplex *(Dantzig, 1947)*

On fabrique des **pantalons** ($x_1$) et des **vestes** ($x_2$), vendus respectivement à des prix $c_1 = 3$ et $c_2 = 5$. Le profit généré est $z = c_1 x_1 + c_2 x_2$ : vendre $x_1$ pantalons au prix $c_1$ rapporte $c_1 x_1$, idem pour les vestes. On cherche à **maximiser ce profit** sous deux contraintes de ressources : la quantité de tissu disponible ($b_1 = 4$ mètres) et les heures de main d'oeuvre ($b_2 = 6$ heures).

Les coefficients $a_{ij}$ mesurent la consommation de ressource $i$ par unité de produit $j$ — par exemple $a_{11} = 1$ signifie qu'un pantalon consomme 1m de tissu.

$$
\begin{aligned}
\max \quad & z = c_1 x_1 + c_2 x_2 \\
\text{s.t.} \quad & a_{11} x_1 + a_{12} x_2 \leq b_1 \\
& a_{21} x_1 + a_{22} x_2 \leq b_2 \\
& x_1, x_2 \geq 0
\end{aligned}
\qquad \Longleftrightarrow \qquad
\begin{aligned}
\max \quad & z = 3x_1 + 5x_2 \\
\text{s.t.} \quad & x_1 + x_2 \leq 4 \quad \text{(tissu)} \\
& x_1 + 3x_2 \leq 6 \quad \text{(main d'oeuvre)} \\
& x_1, x_2 \geq 0
\end{aligned}
$$

---

### Forme standard

On ajoute des **variables d'écart** $x_3, x_4 \geq 0$ pour transformer les inégalités en égalités :

$$
\begin{aligned}
\max \quad & z = 3x_1 + 5x_2 \\
\text{s.t.} \quad & x_1 + x_2 + x_3 = 4 \\
& x_1 + 3x_2 + x_4 = 6 \\
& x_1, x_2, x_3, x_4 \geq 0
\end{aligned}
$$

On peut alors écrire le système sous forme matricielle $Ax = b$, avec la décomposition base / hors-base :

$$
Bx_B + Nx_N = b
$$

où :

$$
c^T = \begin{pmatrix} 3 & 5 & 0 & 0 \end{pmatrix}, \qquad
A = \begin{pmatrix} 1 & 1 & 1 & 0 \\ 1 & 3 & 0 & 1 \end{pmatrix}, \qquad
b = \begin{pmatrix} 4 \\ 6 \end{pmatrix}
$$

> Au départ $z = 0$, et $N = \begin{pmatrix} 1 & 1 \\ 1 & 3 \end{pmatrix} = (A_1, A_2)$ (colonnes hors base)

---

#### Itération 1 — de $(0,0)$ à $(0,2)$

##### (a) Localiser le noeud courant

Visuellement : si on est au noeud $(0,2)$, alors les contraintes $x_2 = 0$ et $x_4 = 0$ sont actives — ce sont elles qui définissent le sommet.

![[simplexe-polytope-point-courant.png|353]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 1. Graphe du polytope avec les contraintes — point courant (0,0)</p>

On choisit $J_B^0 = \{x_3, x_4\}$ comme base initiale. Les slacks forment naturellement une identité :

$$
B = (A_3\ A_4) = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \qquad B^{-1} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

$$
x_B = B^{-1}b = \begin{pmatrix} 4 \\ 6 \end{pmatrix}
$$

$\Rightarrow x_3 = 4,\ x_4 = 6$ (variables de base). Les variables hors-base sont **fixées à zéro par définition** : $x_N = (x_1, x_2)^T = 0$. C'est précisément ce choix qui définit le sommet — poser $x_1 = x_2 = 0$ nous place à l'origine du polytope, sans calcul supplémentaire.

**On est au point $(x_1, x_2) = (0, 0)$, avec $z^{(0)} = 0$.**

##### (b) Coûts réduits — quelle variable entre ?

Pour chaque variable hors-base $x_j$, on calcule le **coût réduit** — la pente de $z$ si on active $x_j$ :

$$
\boxed{\bar{c}_j = c_j - c_B^T B^{-1} A_j}
$$

> [!note]- Construction de l'équation
> On substitue $x_B = B^{-1}b - B^{-1}Nx_N$ dans la fonction objectif :
>
> $z = c_B^T x_B + c_N^T x_N = c_B^T B^{-1}b + \underbrace{\left(c_N^T - c_B^T B^{-1}N\right)}_{\bar{c}_N^T} x_N$
>
> Dans notre exemple : $z = c_B^T B^{-1}b - c_B^T B^{-1}(A_1 x_1 + A_2 x_2) + (c_1 x_1 + c_2 x_2)$
>
> La dérivée partielle $\partial z / \partial x_j$ donne directement $\bar{c}_j$ — c'est le gain marginal si on "allume" $x_j$ d'une unité, en tenant compte de la réaction forcée des variables de base.

Ici $c_B = (0,\ 0)^T$ donc $c_B^T B^{-1} = (0,\ 0)$, ce qui donne directement :

$$
\bar{c}_1 = \frac{\partial z}{\partial x_1} = 3, \qquad \bar{c}_2 = \frac{\partial z}{\partial x_2} = 5
$$

$\bar{c}_2 = 5 > \bar{c}_1 = 3$ $\Rightarrow$ **$x_2$ entre dans la base** : c'est la direction de montée la plus raide.

##### (c) Vecteur directionnel et ratio test

On calcule le vecteur directionnel — il dit comment les variables de base réagissent quand $x_2$ monte :

$$
d_B = B^{-1} A_2 = \begin{pmatrix} 1 \\ 3 \end{pmatrix} \quad \leftrightarrow \quad \begin{pmatrix} x_3 \\ x_4 \end{pmatrix}
$$

Quand $x_2$ monte de $\alpha$, la nouvelle valeur des variables de base est $x_B(\alpha) = x_B - \alpha\, d_B$. On doit rester dans l'espace réalisable, donc $x_B(\alpha) \geq 0$ (eg on ne peut pas avoir une quantité négative de pantalon ie $\le 0$), ce qui impose $\alpha \leq x_B^{(i)} / d_B^{(i)}$ pour chaque composante. Géométriquement, $d_B$ pointe vers le prochain sommet, et $\alpha$ est la distance exacte pour l'atteindre sans sortir du polytope.

**Ratio test** — on cherche le $\alpha$ maximal tel que $x_B(\alpha) \geq 0$ :

$$
\alpha = \min\left\{ \frac{x_B^{(i)}}{d_B^{(i)}} \ \Big|\ d_B^{(i)} > 0 \right\} = \min\left\{ \frac{4}{1},\ \frac{6}{3} \right\} = 2
$$

$\Rightarrow$ **$x_4$ sort de la base** (contrainte la plus serrée).

![[simplexe-vecteur-directionnel-iter1.png|372]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 2. Vecteur directionnel d_B depuis (0,0) — direction vers (0,2)</p>

##### (d) Nouveau sommet et progression

$$
x_B(\alpha) = \begin{pmatrix} 4 \\ 6 \end{pmatrix} - 2\begin{pmatrix} 1 \\ 3 \end{pmatrix} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}
$$

$\Rightarrow x_4 = 0$ (maintenant actif), $x_3 = 2$, $x_2 = 2$.

$$
z^{(1)} = 3 \times 0 + 5 \times 2 = 10 \qquad \text{(progression : } z^{(0)} = 0 \to z^{(1)} = 10\text{)}
$$

---

#### Itération 2 — de $(0,2)$ à $(3,1)$

##### (a) Localiser le noeud courant

On repart du résultat de l'itération 1 : $x_4$ est sorti de la base, remplacé par $x_2$. La nouvelle base est $J_B^{(1)} = \{x_3, x_2\}$, ce qui correspond au sommet $(0, 2)$ — on est maintenant sur la frontière $x_4 = 0$.

On recalcule $B$, $B^{-1}$, et $x_B$ pour ce nouveau sommet :

$$
B = (A_3\ A_2) = \begin{pmatrix} 1 & 1 \\ 0 & 3 \end{pmatrix}, \qquad N = (A_1\ A_4) = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}, \qquad c_B^T = (0,\ 5)
$$

$$
x_B = B^{-1}b = \begin{pmatrix} x_3 \\ x_2 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix}
$$

Donc $x_3 = 2$, $x_2 = 2$, $x_1 = x_4 = 0$. **On est au point $(x_1, x_2) = (0, 2)$, avec $z^{(1)} = 10$.**

##### (b) Coûts réduits — quelle variable entre ?

On calcule les coûts réduits pour les deux variables hors-base $x_1$ et $x_4$. Ici $c_B = (0, 5)^T$ donc $c_B^T B^{-1}$ n'est plus trivial — le calcul donne :

$$
\bar{c}_1 = c_1 - c_B^T B^{-1} A_1 = \frac{4}{3} > 0, \qquad \bar{c}_4 = c_4 - c_B^T B^{-1} A_4 = -\frac{5}{3} \leq 0
$$

$\bar{c}_4 \leq 0$ signifie qu'activer $x_4$ ferait baisser $z$ — on ne le fait pas. $\bar{c}_1 > 0$ en revanche : activer $x_1$ fait monter $z$ de $4/3$ par unité. $\Rightarrow$ **$x_1$ entre dans la base**.

##### (c) Vecteur directionnel et ratio test

On calcule comment les variables de base réagissent quand $x_1$ monte :

$$
d_B = B^{-1} A_1 = \begin{pmatrix} 2/3 \\ 1/3 \end{pmatrix} \quad \leftrightarrow \quad \begin{pmatrix} x_3 \\ x_2 \end{pmatrix}
$$

Les deux composantes sont positives, donc les deux variables de base contraignent $\alpha$. On applique le ratio test :

$$
\alpha = \min\left\{ \frac{x_3}{2/3},\ \frac{x_2}{1/3} \right\} = \min\left\{ \frac{2}{2/3},\ \frac{2}{1/3} \right\} = \min\{3,\ 6\} = 3
$$

C'est $x_3$ qui atteint 0 en premier $\Rightarrow$ **$x_3$ sort de la base**.

![[simplexe-vecteur-directionnel-iter2.png|457]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 3. Vecteur directionnel d_B depuis (0,2) — direction vers (3,1)</p>

##### (d) Nouveau sommet et progression

$$
x_B(3) = \begin{pmatrix} 2 \\ 2 \end{pmatrix} - 3\begin{pmatrix} 2/3 \\ 1/3 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
$$

Donc $x_3 = 0$ (actif), $x_2 = 1$, $x_1 = 3$. On vérifie les coûts réduits restants : tous $\leq 0$. On ne peut plus améliorer $z$ dans aucune direction $\Rightarrow$ **on est à l'optimum**.

$$
z^{(2)} = 3 \times 3 + 5 \times 1 = 14 \qquad \text{(progression : } z^{(1)} = 10 \to z^{(2)} = 14\text{)}
$$

---

## 1.2 Dualité

### Dualité forte

Le problème **dual** associé au primal est :

$$
\begin{array}{l\qquad l}
\textbf{Primal} & \textbf{Dual} \\[9pt]
\begin{aligned}
\max \quad & z = 3x_1 + 5x_2 \\
\text{s.t.} \quad & x_1 + x_2 \leq 4 \\
& x_1 + 3x_2 \leq 6 \\
& x_1, x_2 \geq 0
\end{aligned}
&
\begin{aligned}
\min \quad & W = 4y_1 + 6y_2 \\
\text{s.t.} \quad & y_1 + y_2 \geq 3 \\
& y_1 + 3y_2 \geq 5 \\
& y_1, y_2 \geq 0
\end{aligned}
\end{array}
$$

Solution duale optimale : $y_1^* = 2,\ y_2^* = 1$.

**Théorème de dualité forte :**

$$
z_{\max} = \sum_j c_j x_j^* = \sum_i b_i y_i^* = W_{\min} = 14
$$

**Interprétation économique :**

$$
\begin{aligned}
&\frac{\partial z}{\partial b_1} = y_1^* \quad \Rightarrow \quad \text{1m de tissu supplémentaire augmente le profit de } y_1^* \\
&\frac{\partial z}{\partial c_1} = x_1^* = 3 \quad \text{(pantalons produits à l'optimal)}
\end{aligned}
$$

> $y_i^*$ = **prix fantôme** de la ressource $i$ : valeur marginale d'une unité supplémentaire de $b_i$.

---

### Analyse de sensibilité (Sensitivity Analysis)

L'analyse de sensibilité répond à la question : **jusqu'à quel point peut-on perturber les paramètres du problème sans changer la base optimale ?** On distingue deux types de perturbations : sur les coûts $c_j$ (variables de décision) et sur les ressources $b_i$ (contraintes).

À l'optimum : $x_1^* = 3,\ x_2^* = 1,\ z^* = 14$, prix fantômes $y_1^* = 2,\ y_2^* = 1$.

### Variables de décision — Allowable range sur $c_j$

La base optimale reste $\{x_1, x_2\}$ tant que tous les coûts réduits des variables hors-base restent $\leq 0$. À l'optimum, $\bar{c}_3 = -2$ et $\bar{c}_4 = -5/3$. Si on perturbe $c_1$ ou $c_2$, ces coûts réduits changent et peuvent devenir positifs — la base change.

- **$x_1$ ($c_1 = 3$)** : $\bar{c}_4 = -5/3 + \Delta c_1 \cdot (\ldots)$ — la base tient tant que $c_1 \geq 5/3$, soit une baisse de $4/3$ au maximum. Hausse : $+\infty$.
- **$x_2$ ($c_2 = 5$)** : la base tient tant que $c_2 \geq 10/3$, soit une baisse de $5/3$ au maximum. Hausse : $+\infty$.

### Contraintes — Shadow price et allowable range sur $b_i$

Le shadow price $y_i^*$ dit de combien augmente $z^*$ si $b_i$ augmente d'une unité. Il est valide seulement dans un certain intervalle autour de $b_i$ — au-delà, une autre contrainte devient active et la base change.

- **Contrainte tissu ($b_1 = 4$)** : si on a 1m de tissu supplémentaire, le profit augmente de $y_1^* = 2$. Valide pour $b_1 \in [2,\ 6]$.
- **Contrainte main d'œuvre ($b_2 = 6$)** : 1h supplémentaire augmente le profit de $y_2^* = 1$. Valide pour $b_2 \in [4,\ +\infty)$.

### Adjustable Cells (variables de décision)

Pour chaque variable, le **Reduced Cost** est le coût réduit $\bar{c}_j$ à l'optimum — il vaut 0 pour les variables dans la base. L'**Objective Coefficient** est la valeur actuelle de $c_j$. Les colonnes *Allowable* indiquent de combien $c_j$ peut varier sans changer la base optimale.

| Name | Final Value | Reduced Cost | Obj. Coefficient | Allowable Increase | Allowable Decrease |
|---|---|---|---|---|---|
| Pantalons ($x_1$) | 3 | 0 | 3 | $+\infty$ | $4/3$ |
| Vestes ($x_2$) | 1 | 0 | 5 | $+\infty$ | $5/3$ |

### Constraints

Le **Final Value** est la valeur de la contrainte à l'optimum (ex. $x_1+x_2 = 4$ pour la contrainte tissu). Le **Shadow Price** est $y_i^*$ — gain sur $z^*$ par unité supplémentaire de $b_i$. Les colonnes *Allowable* indiquent la plage de $b_i$ pour laquelle ce shadow price reste valide.

| Name | Final Value | Shadow Price | Constraint RHS | Allowable Increase | Allowable Decrease |
|---|---|---|---|---|---|
| Tissu ($b_1$) | 4 | 2 | 4 | 2 | 2 |
| Main d'œuvre ($b_2$) | 6 | 1 | 6 | $+\infty$ | 2 |

> **Lecture** : une contrainte avec Shadow Price = 0 n'est pas saturée à l'optimum — augmenter sa ressource ne change rien au profit. Une contrainte saturée (Final Value = RHS) a en général un shadow price $> 0$.

---

### Complementary slackness (Théorème des écarts complémentaires) le KKT dans le cas linéaire

STRONG DUALITY et y'a pas de lagrangien dedans si je ne m'abuse - ce que je trouve bizarre c qu'on n'a pas de lagrangien dans ce domaine mais on a quand même le lambda de la sensibilité qui est un truc de lagrange non ou j'ai pas compris ? 

## 2. Programmes en nombres entiers (ILP / MILP)

Un **programme linéaire en nombres entiers** (ILP) ou mixte (MILP) ajoute la contrainte $x_j \in \mathbb{Z}$ pour certaines variables. Ça suffit à rendre le problème **NP-difficile** : il n'existe pas d'algorithme polynomial connu. La stratégie générale repose sur des **bornes** : on maintient une borne supérieure $\bar{z}$ (meilleure solution entière trouvée) et une borne inférieure $\underline{z}$ (relaxation continue), et on s'arrête quand $\bar{z} - \underline{z} \leq \varepsilon$.

La **relaxation continue** (drop des contraintes entières) donne une borne inférieure via le simplexe — mais arrondir sa solution ne donne pas nécessairement une solution entière réalisable, et peut être loin de l'optimum entier.

| Algorithme | Date | Auteur |
|---|---|---|
| Branch & Bound | 1960 | Land & Doig |
| Gomory cuts | 1958 | Gomory |
| Branch & Cut | 1991 | Padberg & Rinaldi |
| Intégration dans les solveurs commerciaux | 1997 | Ceria (Columbia / CPLEX) |

### Branch & Bound *(Land & Doig, 1960)*

> **Intuition.** L'idée c'est simple : on résout d'abord le problème en oubliant que les variables doivent être entières — le simplexe nous donne une solution avec des valeurs continues, genre $x_1 = 2.33$. Ce point n'est pas réalisable pour notre problème entier, donc on "coupe" : on crée deux sous-problèmes, un où $x_1 \leq 2$ et un où $x_1 \geq 3$, et on résout chacun par le simplexe. On répète jusqu'à avoir des solutions entières. À la fin on garde celle qui maximise la fonction objectif — en éliminant au passage toutes les branches qui ne peuvent pas faire mieux que ce qu'on a déjà trouvé.

On résout la relaxation continue. Si la solution n'est pas entière, on **branche** sur une variable fractionnaire $x_j = v$ en créant deux sous-problèmes : $x_j \leq \lfloor v \rfloor$ et $x_j \geq \lceil v \rceil$. On explore l'arbre ainsi créé en élaguant les branches dont la borne inférieure dépasse la meilleure solution entière connue. Correct et complet, mais exponentiel au pire cas.

**Exemple.** Considérons le problème :

$$
\begin{aligned}
\max \quad & z = 5x_1 + 6x_2 \\
\text{s.t.} \quad & x_1 + x_2 \leq 5 \\
& 4x_1 + 7x_2 \leq 28 \\
& x_1, x_2 \geq 0, \quad x_1, x_2 \in \mathbb{N}
\end{aligned}
$$

**Étape 1 — Relaxation.** On retire la contrainte d'intégrité ($x_1, x_2 \in \mathbb{N}$ devient $x_1, x_2 \geq 0$ réels) et on résout par le simplexe. On obtient $(x_1, x_2) = (7/3, 8/3)$ avec $z = 27.67$ — solution fractionnaire. L'espace entier (points discrets) n'est plus convexe, ce point n'est pas réalisable.

![[bb-relaxation-vs-entier.png]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 2. Gauche : relaxation continue, optimum fractionnaire. Droite : espace entier discret, plus convexe.</p>

**Étape 2 — Branchement sur $x_2$.** On choisit $x_2 = 8/3$ car c'est la plus grande partie fractionnaire. On crée deux sous-problèmes et on résout chacun par le simplexe :

- **Branche $x_2 \leq 2$** — on résout :
$$
\begin{aligned}
\max \quad & z = 5x_1 + 6x_2 \\
\text{s.t.} \quad & x_1 + x_2 \leq 5,\ \ x_2 \leq 2,\ \ 4x_1 + 7x_2 \leq 28,\ \ x_1, x_2 \geq 0
\end{aligned}
$$
Simplexe $\Rightarrow$ $z = 27$, $x_1 = 3$, $x_2 = 2$ — solution entière. **Borne courante = 27.**

- **Branche $x_2 \geq 3$** — on résout :
$$
\begin{aligned}
\max \quad & z = 5x_1 + 6x_2 \\
\text{s.t.} \quad & x_1 + x_2 \leq 5,\ \ x_2 \geq 3,\ \ 4x_1 + 7x_2 \leq 28,\ \ x_1, x_2 \geq 0
\end{aligned}
$$
Simplexe $\Rightarrow$ $z = 26.75$, $x_1 = 1.75$, $x_2 = 3$ — fractionnaire, on rebranche sur $x_1$.
  - **Branche $x_1 \geq 2$** : infaisable (simplexe ne trouve pas de solution).
  - **Branche $x_1 \leq 1$** : simplexe $\Rightarrow$ $z = 23$, $x_1 = 1$, $x_2 = 3$ — entier mais $23 < 27$ → **fathomed** (cette branche ne peut pas améliorer la borne connue).

Conclusion : $z^* = 27$, $x_1 = 3$, $x_2 = 2$.

![[bb-arbre.png|697]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 3. Arbre Branch & Bound — à chaque noeud on branche et on borne.</p>

### Coupes de Gomory *(Gomory, 1958)*

> **Intuition.** Le même ensemble de points entiers peut être encadré par une infinité de polytopes différents. Sur l'image de gauche, le polytope de la relaxation a des sommets fractionnaires — le simplexe y atterrit et donne une solution non entière. Si on arrive à construire un polytope dont **tous les sommets sont des points entiers** (image de droite), alors résoudre la relaxation continue suffit — on tombe directement sur la solution entière optimale. Les coupes de Gomory construisent ce polytope itérativement : à chaque étape on ajoute une inégalité qui coupe le sommet fractionnaire courant sans supprimer aucun point entier, jusqu'à ce que tous les sommets soient entiers.

![[gomory-polytopes-entiers.png|455]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 4. Gauche : polytope avec sommets fractionnaires. Droite : polytope idéal dont tous les sommets sont entiers — la relaxation donne directement l'optimum entier.</p>

Formellement, pour la ligne $i$ du tableau du simplexe final, avec $x_i + \sum_j \alpha_{ij} x_j = b_i$ et $b_i \notin \mathbb{Z}$, la coupe de Gomory est :

$$
\sum_j \{\alpha_{ij}\}\, x_j \geq \{b_i\}
$$

où $\{\cdot\}$ désigne la partie fractionnaire. Cette inégalité est violée par la solution fractionnaire courante mais satisfaite par tous les points entiers réalisables. On l'ajoute comme contrainte et on relance le simplexe.

**Exemple.** On cherche $\max x_2$ sous $3x_1 + 2x_2 \leq 6$, $-3x_1 + 2x_2 \leq 0$, $x_1, x_2 \in \mathbb{Z}_+$. On introduit les slacks $x_3, x_4$ pour mettre en forme standard :

$$
\begin{aligned}
\min \quad & -x_2 \\
\text{s.t.} \quad & 3x_1 + 2x_2 + x_3 = 6 \\
& -3x_1 + 2x_2 + x_4 = 0 \\
& x_1, x_2, x_3, x_4 \geq 0 \text{ entiers}
\end{aligned}
$$

**Étape 1 — Relaxation continue.** On résout par le simplexe en ignorant la contrainte d'intégrité. Le tableau initial a $x_3$ et $x_4$ en base. Après pivotage on obtient $x_1 = 1,\ x_2 = 3/2$ avec $z = -3/2$. La solution est fractionnaire : $x_2 = 3/2 \notin \mathbb{Z}$.

![[gomory-tableau-iter1.png|470]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 4. Tableau du simplexe après la première itération et région réalisable de la relaxation (grisée).</p>

**Étape 2 — Première coupe de Gomory.** La ligne de $x_2$ dans le tableau donne $x_2 + \frac{1}{4}x_3 + \frac{1}{4}x_4 = \frac{3}{2}$. La partie fractionnaire de $3/2$ est $1/2$, celle de $1/4$ est $1/4$. La coupe de Gomory est donc :

$$
\frac{1}{4}x_3 + \frac{1}{4}x_4 \geq \frac{1}{2}
$$

On réexprime $x_3$ et $x_4$ en termes de $x_1, x_2$ : $x_3 = 6 - 3x_1 - 2x_2$ et $x_4 = 3x_1 - 2x_2$. En substituant :

$$
\frac{1}{4}(6 - 3x_1 - 2x_2) + \frac{1}{4}(3x_1 - 2x_2) \geq \frac{1}{2}
\quad \Longrightarrow \quad
x_2 \leq 1
$$

On ajoute la contrainte $x_2 \leq 1$ (variable de slack $s_1$) et on relance le simplexe. On obtient $x_1 = 1,\ x_2 = 3/2$ — encore fractionnaire.

![[gomory-tableau-iter2.png]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 5. Tableau après ajout de la première coupe et nouvelle région réalisable (la coupe rouge élimine le sommet fractionnaire).</p>

**Étape 3 — Deuxième coupe de Gomory.** La ligne de $x_1$ dans le nouveau tableau donne la coupe :

$$
\frac{2}{3}x_4 + \frac{2}{3}s_1 \geq \frac{2}{3}
$$

En réexprimant en termes de $x_1, x_2$ : $x_1 - x_2 \geq 0$. On ajoute cette contrainte et on relance le simplexe. On obtient cette fois $x_1 = 2,\ x_2 = 1$ — **solution entière optimale**.

![[gomory-tableau-iter3.png|545]]
<p style="text-align:center; font-style:italic; font-size:0.85em; color:gray">Figure 6. Après la deuxième coupe, les deux sommets restants du polytope sont entiers — la relaxation donne directement l'optimum.</p>


### Branch & Cut *(Padberg & Rinaldi, 1991)*

> **Intuition.** Branch & Bound c'est puissant mais ça crée beaucoup de sous-problèmes. Gomory c'est élégant mais lent seul. L'idée du Branch & Cut c'est de combiner les deux : à chaque noeud de l'arbre, avant de brancher, tu essaies d'abord d'ajouter des coupes pour resserrer la relaxation. Si les coupes suffisent à rendre la solution entière, pas besoin de brancher. Sinon tu branches. En pratique ça réduit drastiquement le nombre de branchements — c'est pour ça que tous les solveurs modernes font ça.

La combinaison des deux : à chaque noeud de l'arbre Branch & Bound, avant de brancher, on ajoute des coupes pour resserrer la relaxation. Cela réduit le nombre de branchements nécessaires. C'est l'approche utilisée par tous les solveurs modernes (Gurobi, CPLEX, HiGHS) depuis les années 1990.

$$
\text{B\&C} = \text{Branch \& Bound} + \text{Cutting planes à chaque noeud}
$$

Algorithme : (1) résoudre la relaxation, (2) si infaisable → stop, (3) si solution entière → stop, (4) sinon → **ajouter des coupes** et retourner en (1), ou **brancher** et résoudre récursivement.

## Faire une remarque sur la dualité ici

