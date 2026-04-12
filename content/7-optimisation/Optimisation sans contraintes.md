## 1. Objectif

On cherche à **identifier un minimum local** $x^*$ avec un ou des tests — on dit "local" car il peut en exister plusieurs. Deux grandes familles de méthodes existent : **line search** et **trust region**.

---
Autre trucs => chapitre convex set
Définition : ensemble convexe
Un ensemble $C \subset \mathbb{R}^d$ est dit convexe s'il vérifie la propriété :

$$
\forall x_0, x_1 \in C, \forall \alpha \in[0,1], \quad x_\alpha=\alpha x_1+(1-\alpha) x_0 \in C
$$

![[Pasted image 20260408173234.png|205]]

Eg. Une droite,un plan, un hyperplan, un carré, une boule, un cube etc

![[Pasted image 20260408173603.png|435]]

Définition : fonction convexe
$f: \mathbb{R}^d \rightarrow \mathbb{R}$ est convexe si elle vérifie $\forall x_0, x_1 \in \mathbb{R}^d, \forall \alpha \in [0,1], f(\underbrace{\alpha x_1+(1-\alpha)\left(x_0\right)}_{x_\alpha}) \leq \alpha f\left(x_1\right)+(1-\alpha) f\left(x_0\right)$

![[Pasted image 20260408173651.png|359]]

Interprétation: le segment joignant deux points de la courbe est au dessus de la courbe

Définition : strictement fonction convexe $f: \mathbb{R}^d \rightarrow \mathbb{R}$ est strictement convexe si elle vérifie $\forall x_0 \neq x_1 \in \left.\mathbb{R}^d, \forall \alpha \in\right] 0,1[, f(\underbrace{\alpha x_1+(1-\alpha)\left(x_0\right)}_{x_\alpha})<\alpha f\left(x_1\right)+(1-\alpha) f\left(x_0\right)$

![[Pasted image 20260408173735.png|382]]

Interprétation: le segment joignant deux points de la courbe est strictement au dessus de la courbe

\textbf{hyperplane:} set of the form $\left\{x \mid a^{T} x=b\right\}(a \neq 0)$
![[ex-1.png|221]]

Puis il manque rappel hyperplan, halfplace, polyhèdre etc

![[ex-2.png|208]]

\textcolor{cornellred}{\textbf{Definition (Polyhedron).}} A polyhedron is defined as the solution set of a finite number of linear equalities and inequalities:
$$
\mathcal{P}=\left\{x \mid a_{j}^{T} x \leq b_{j}, j=1, \ldots, m, c_{j}^{T} x=d_{j}, j=1, \ldots, p\right\}
$$
A polyhedron is thus the \underline{intersection} of a finite number of halfspaces and hyperplanes. Affine sets (e.g., subspaces, hyperplanes, lines), rays, line segments, and halfspaces are all polyhedra. It is easily shown that polyhedra are convex sets.
![[ex-4.png|181]]

It will be convenient to use the compact notation
$$
\mathcal{P}=\{x \mid A x \preceq b, C x=d\}
$$
for where
$$
A=\left[\begin{array}{c}
a_{1}^{T} \\
\vdots \\
a_{m}^{T}
\end{array}\right], \quad C=\left[\begin{array}{c}
c_{1}^{T} \\
\vdots \\
c_{p}^{T}
\end{array}\right]
$$
and the symbol $\preceq$ denotes vector inequality or componentwise inequality in $\mathbf{R}^{m}$ :
$u \prec v$ means $u_{i} \leq v_{i}$ for $i=1, \ldots,

---
=> chapitre convex function

----

## 2. Définitions : minimiseur

**Minimiseur global** : $x^*$ tel que $f(x^*) \leq f(x), \quad \forall x \in \text{dom}(f)$

**Minimiseur local** : $x^*$ tel que $f(x^*) \leq f(x), \quad \forall x \in \mathcal{N}(x^*)$

où $\mathcal{N}(x^*)$ désigne un voisinage ouvert de $x^*$.

On distingue aussi :
- **Weak minimizer** : inégalité large $f(x^*) \leq f(x)$
- **Strong minimizer** : inégalité stricte $f(x^*) < f(x)$

> 📌 **IMAGE À INSÉRER** — Illustration weak vs strong minimizer
![[Pasted image 20260408173130.png|505]]

> **Remarque :** En contexte **non-convexe**, le point de départ est crucial. En contexte **convexe**, tout minimum local est aussi un minimum global.

> 📌 **IMAGE À INSÉRER** — Comparaison convexe vs non-convexe
![[Pasted image 20260408173039.png|562]]
![[Pasted image 20260408172902.png|545]]

---

## 3. Conditions d'optimalité

### Condition du 1er ordre (nécessaire)

Si $x^*$ est un minimiseur local et $f$ est continûment différentiable dans un voisinage ouvert de $x^*$, alors :

$$\nabla f(x^*) = 0$$

Un point vérifiant cette condition est appelé **point stationnaire**.

---

### Condition du 2nd ordre

On suppose que $\nabla^2 f$ existe et est continue dans un voisinage ouvert de $x^*$.

**Condition nécessaire** — si $x^*$ est un minimiseur local :

1. $\nabla f(x^*) = 0$
2. $\nabla^2 f(x^*) \succeq 0$ (PSD) — weak minimizer ; $\nabla^2 f(x^*) \succ 0$ (PD) — strong minimizer

**Condition suffisante** — si les deux conditions suivantes sont vérifiées, alors $x^*$ est un minimiseur local :

1. $\nabla f(x^*) = 0$
2. $\nabla^2 f(x^*) \succ 0$ — ou $\succeq 0$ pour le cas weak

---

### Intuition géométrique de la PSD

En $1D$, on approche localement $f$ par une parabole :

$$f(x) \approx a + bx + cx^2$$

Pour que $x^*$ soit un minimum local, on doit avoir $c > 0$. En dimension $n$, le développement de Taylor au second ordre donne :

$$f(x^* + d) \approx f(x^*) + \nabla f(x^*)^\top d + \frac{1}{2} d^\top \nabla^2 f(x^*)\, d$$

Pour que $x^*$ soit un minimum dans **toute direction** $d \in \mathbb{R}^n$ :

$$d^\top \nabla^2 f(x^*)\, d \geq 0 \quad \forall d \in \mathbb{R}^n \iff \nabla^2 f(x^*) \succeq 0$$

> 📌 **IMAGE À INSÉRER** — Quadratic forms (PD, ND, PSD singulière, indéfinie)
![[Pasted image 20260408173947.png|480]]

---

## 4. Vue d'ensemble des algorithmes

Tous les algorithmes partagent la même structure itérative :

$$x_0 \longrightarrow \{x_k\} \longrightarrow x^*$$

À chaque itération, on calcule $x_{k+1}$ à partir de $x_k$. **C'est le passage de $x_k$ à $x_{k+1}$ qui différencie les algorithmes.**

> 📌 **IMAGE À INSÉRER** — Schéma overview algorithms (`algo_overview.svg`)

---

## 5. Choix de $x_0$

Plusieurs stratégies possibles :
- Tirage **aléatoire**
- Partir du **vecteur nul** $x_0 = 0$
- Résoudre un **problème plus simple** et utiliser sa solution comme point de départ
- Utiliser du **ML** pour prédire un bon $x_0$

---

## 6. Line Search

Depuis $x_k$, on choisit une **direction** $p_k$, puis on cherche le **pas** $\alpha_k > 0$ optimal le long de cette direction :

$$x_{k+1} = x_k + \alpha_k \cdot p_k$$

> 📌 **IMAGE À INSÉRER** — Schéma line search (`line_search.svg`)

---

### 6.1 Propriétés des directions de descente

On part du développement de Taylor au premier ordre, pour $\varepsilon > 0$ petit :

$$f(x_k + \varepsilon p_k) = f(x_k) + \varepsilon \, \nabla f(x_k)^\top p_k + o(\varepsilon^2)$$

On décompose le produit scalaire :

$$\nabla f(x_k)^\top p_k = \|\nabla f(x_k)\| \cdot \|p_k\| \cos(\theta_k)$$

Pour que $f_{k+1} < f_k$, il faut :

$$\nabla f_k^\top p_k < 0 \iff \cos(\theta_k) < 0 \iff \theta_k \in \left(\frac{\pi}{2}, \frac{3\pi}{2}\right)$$

La **décroissance maximale** est atteinte pour $\theta_k = \pi$, soit $\cos(\theta_k) = -1$ :

$$\Rightarrow \quad p_k = -\nabla f_k \qquad \text{(steepest descent)}$$

> **Remarque :** Valable uniquement si $\varepsilon$ est suffisamment petit — si $\varepsilon$ est trop grand, le terme $o(\varepsilon^2)$ peut dominer.

---

### 6.2 Choix de la direction $p_k$

**C'est le choix de $p_k$ qui définit l'algorithme.** Quatre grandes options :

#### (i) Steepest Descent (= Gradient Descent)

$$p_k = -\nabla f_k$$

Direction de plus grande pente. Simple mais **convergence lente** (taux linéaire).

**Pourquoi c'est la direction de descente maximale ?** On cherche la direction $p_k$ qui minimise $\nabla f_k^\top p_k$ sous contrainte $\|p_k\| = 1$ :

$$\nabla f_k^\top p_k = \|\nabla f_k\| \cos(\theta_k)$$

Minimal quand $\cos(\theta_k) = -1$, i.e. $p_k = -\nabla f_k$ (à normalisation près).

#### (ii) Méthode de Newton

On part du développement de Taylor au 2nd ordre et on annule le gradient du modèle :

$$\boxed{p_k^{\text{Newton}} = -(\nabla^2 f_k)^{-1} \nabla f_k}$$

Valide tant que $\nabla^2 f_k \succ 0$. C'est une direction de descente ssi $\nabla^2 f_k \succ 0$ :

$$p_k^\top \nabla f_k = -\nabla f_k^\top (\nabla^2 f_k)^{-1} \nabla f_k < 0 \iff \nabla^2 f_k \succ 0$$

#### (iii) Quasi-Newton

$$p_k = -B_k \, \nabla f_k$$

$B_k \approx (\nabla^2 f_k)^{-1}$, mis à jour itérativement pour éviter le coût $O(n^3)$ de l'inversion exacte.

#### (iv) Gradient Conjugué

$$p_k = -\nabla f_k + \beta_k \, p_{k-1}$$

Corrige la direction via l'historique $p_{k-1}$ — évite le zig-zag du steepest descent.

> ==**Note :** Les méthodes (i) à (iv) sont toutes généralisables à des problèmes **non-convexes**.==

---

### 6.3 Choix du pas $\alpha_k$

Le pas $\alpha_k$ est aussi appelé **step length** ou **learning rate**. On définit la fonction unidimensionnelle :

$$\varphi(\alpha) = f(x_k + \alpha p_k), \quad \alpha > 0$$

Puisque $p_k$ est une direction de descente, $\varphi'(0) = \nabla f_k^\top p_k < 0$. Un minimiseur exact $\varphi'(\alpha) = 0$ est coûteux à calculer $\Rightarrow$ on utilise une **inexact line search**.

#### 6.3.1 Conditions de Wolfe

Deux conditions pour accepter un $\alpha$ :

**(i) Sufficient decrease (Armijo) :**
$$f(x_k + \alpha p_k) \leq f(x_k) + c_1 \alpha \nabla f_k^\top p_k, \quad c_1 \in (0,1)$$

Garantit une décroissance suffisante — évite les pas trop petits qui ne progressent pas.

**(ii) Curvature condition :**
$$\nabla f(x_k + \alpha p_k)^\top p_k \geq c_2 \cdot \nabla f_k^\top p_k, \quad 0 < c_1 < c_2 < 1$$

Garantit qu'on n'est pas trop loin du minimiseur de $\varphi$ — évite les pas trop petits qui restent dans la pente.

> **Remarque :** Si $f$ est **convexe**, la condition (i) seule suffit.

#### 6.3.2 Backtracking Line Search

Algorithme pratique pour satisfaire Armijo :

1. Partir avec $\alpha$ grand
2. Si $f(x_k + \alpha p_k) > f(x_k) + c_1 \alpha \nabla f_k^\top p_k$ $\Rightarrow$ $\alpha \leftarrow \alpha \cdot \rho$, $\rho \in (0,1)$
3. Sinon : accepté, $x_{k+1} = x_k + \alpha p_k$

#### 6.3.3 Interpolation quadratique

Alternative plus précise : calculer $\varphi(0)$, $\varphi'(0)$ et $\varphi(\alpha_0)$ pour un $\alpha_0$ initial, ajuster une parabole et minimiser analytiquement.

#### 6.3.4 Learning rate en pratique ML

En ML/DL, faire une line search à chaque itération est hors de question : une évaluation de $f$ = un forward pass complet sur le dataset, ce qui est déjà coûteux. On ne fait **pas d'exact line search** — à la place, $\alpha_k$ est fixé à l'avance selon une règle simple.

**Approche d'Andrew Ng :** tester $\alpha$ sur une grille logarithmique et observer la courbe de loss :

$$\dots, \quad 0.001, \quad 0.01, \quad 0.1, \quad 1, \quad \dots$$

- $\alpha$ trop petit $\Rightarrow$ convergence lente
- $\alpha$ trop grand $\Rightarrow$ la loss diverge ou oscille sans converger

**Learning rate schedule :** plus généralement, on définit une règle de décroissance de $\alpha_k$ au fil de l'entraînement. Les schedules courants sont le step decay (réduire d'un facteur fixe tous les $n$ epochs), le cosine annealing (décroissance en cosinus sur toute la durée), et le warmup (commencer petit puis monter avant de redescendre — utile pour les transformers). L'idée commune : un grand $\alpha$ au début pour explorer rapidement, puis un $\alpha$ petit en fin d'entraînement pour affiner la convergence.

> En résumé : la line search (Wolfe, backtracking) s'applique aux méthodes de second ordre (L-BFGS, Newton) où $f$ est bon marché à évaluer. En ML stochastique on schedule $\alpha_k$ à l'avance, ou on délègue à un optimiseur adaptatif (Adam) qui gère le pas par coordonnée.

---

### 6.4 Analyse de convergence — Steepest Descent avec exact line search

> 📌 **IMAGE À INSÉRER** — Courbes de niveau avec trajectoire zig-zag

**Observations :**
1. **Trajectoire en zig-zag** — convergence lente
2. **Les step sizes décroissent** au fil des itérations
3. **Chaque pas est $\perp$ au pas précédent** — après une line search exacte, $\nabla f_{k+1} \perp p_k$, donc $p_{k+1} = -\nabla f_{k+1} \perp p_k$

> La convergence reste **linéaire** quelle que soit la stratégie de choix de $\alpha$ — pas mieux avec une line search exacte.

---

## 7. Trust Region

Plutôt que de choisir une direction puis un pas, la **trust region** construit un **modèle approché** $m_k$ de $f$ autour de $x_k$, puis minimise ce modèle dans un voisinage :

$$\min_p \; m_k(x_k + p) \quad \text{s.t.} \quad x_k + p \in \mathcal{TR}$$

Deux étapes :
1. **Construire** un modèle $m_k$ de $f$ au voisinage de $x_k$
2. **Chercher** le minimum de $m_k$ dans ce voisinage (la région de confiance)

Le modèle le plus courant est **quadratique** (Hessienne exacte ou approchée) :

$$m_k(x_k + p) = f_k + \nabla f_k^\top p + \frac{1}{2} p^\top B_k \, p$$

où $B_k \approx \nabla^2 f_k$.

> **Intuition :** On se donne des points autour de $x_k$, on ajuste une approximation quadratique dessus, et on minimise cette parabole — mais seulement dans la zone où on lui fait confiance.

---

## 8. Gradient descent appliqué à la régression linéaire

**Cadre :** $\theta = (\theta_0, \theta_1)$, $n$ individus, $h_\theta(x_i) = \theta_0 + \theta_1 x_i$. Chaque terme de la somme :

$$f_i(\theta) = \frac{1}{2}\left(y_i - (\theta_0 + \theta_1 x_i)\right)^2$$

Le gradient individuel :

$$g_i = \nabla f_i(\theta) = \begin{pmatrix} \partial f_i / \partial \theta_0 \\ \partial f_i / \partial \theta_1 \end{pmatrix} = \begin{pmatrix} -(y_i - \hat{y}_i) \\ -x_i(y_i - \hat{y}_i) \end{pmatrix}$$

Mise à jour gradient descent full-batch :

$$\theta^{(k+1)} = \theta^{(k)} - \alpha \cdot \frac{1}{n} \sum_{i=1}^n \nabla f_i(\theta^{(k)})$$

**Exemple numérique** ($n=2$) :

| $x$ | $y$ |
|-----|-----|
| $3$ | $10$ |
| $4$ | $8$ |
chaque observation génère sa propre fonction 
$$f_1(\theta) = \frac{1}{2}(10 - (\theta_0 + 3\theta_1))^2, \qquad f_2(\theta) = \frac{1}{2}(8 - (\theta_0 + 4\theta_1))^2$$
On va tirer un vecteur $\vec{\theta} \sim N(0,1)$ eg $\vec{\theta}^{(1)} = (1.;0.75)$ et pour chaque $f_i(\theta)$ on calcule sa pente selon $\theta_0$ et $\theta_1$ ce qui me donne le $g_i = \nabla f_i(\theta^{(1)})$ . Puis dans la régression linéaire classique on donne un poids **uniforme** à chaque échantillon et on va calculer la moyenne des dérivées par rapport à l'intercept et la pente. itération 2 on va avancer d'un pas vers le minimum dans l'espace $\vec{\theta}$. On calcule un nouveau $\vec{\theta}^{(2)}$ etc 
![[Pasted image 20260412201356.png]]
Add Figure on calcule la moyenne des dérivés au point courant theta^{(i)}

**Intuition géométrique :** pour chaque échantillon $i$, on dispose d'une surface de coût $f_i(\theta)$ dans l'espace des paramètres. Le gradient descent fait la **moyenne de ces pentes** sur tous les individus.

> *"C'est la direction qui en moyenne fait descendre l'altitude sur toutes les montagnes à la fois."*

---

### 8.1 Weighted linear regression

Au lieu de pondérer uniformément, on pose :

$$f(\theta) = \sum_{i=1}^n w_i \cdot f_i(\theta)$$

**Exemple Barra :** penny stock ($\text{MCap}_A = 10\text{M€}$) vs Apple ($\text{MCap}_B = 2990\text{M€}$). En normalisant par la capitalisation totale :

$$w_1 = \frac{10}{3000} \approx 0.0033, \qquad w_2 = \frac{2990}{3000} \approx 0.9967$$

Si le penny stock a une perte immense, elle est quasi effacée car $w_1 \approx 0.003$. Dans Barra, ce sont les plus grosses capitalisations qui pilotent les facteurs de retour — si Apple a un momentum très élevé et le penny stock un momentum très faible, ce sera un momentum factor return fort qui l'emporte.

![[Pasted image 20260412203127.png|625]]
Figure graphe construit avec les mêmes échantillons que l'exemple de régression linéaire mais il illustrer bien que le calcule des gradients et maintenant pondérés

---

## 9. Méthodes de réduction de variance

En ML, l'objectif prend souvent la forme d'une **somme finie** :

$$f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)$$

Calculer $\nabla f(x) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(x)$ est coûteux pour $n$ grand. Les méthodes stochastiques remplacent ce gradient exact par un estimateur bon marché.

---

### 9.1 SGD — Stochastic Gradient Descent

À chaque itération $k$, on tire un indice $i_k \sim \mathcal{U}\{1, \dots, n\}$ et on pose :

$$x^{(k)} = x^{(k-1)} - t_k \cdot \nabla f_{i_k}(x^{(k-1)})$$

**Non-biais de l'estimateur :**

$$\mathbb{E}[\nabla f_{i_k}(x)] = \sum_{j=1}^n \mathbb{P}(i_k = j) \cdot \nabla f_j(x) = \frac{1}{n} \sum_{j=1}^n \nabla f_j(x) = \nabla f(x) \checkmark$$

**Exemple — régression linéaire :**

$$\min_\theta J(\theta) = \min_\theta \frac{1}{n} \sum_{i=1}^n \left(y_i - (\theta_0 + \theta_1 x_i)\right)^2$$

En posant $h_\theta(x) = X\theta$, on obtient $J(\theta) = \frac{1}{2n} \|X\theta - y\|^2$ et le gradient full-batch :

$$\nabla_\theta J(\theta) = \frac{1}{n} X^\top (X\theta - y)$$

SGD tire $i \sim \mathcal{U}[n]$ à chaque pas :

$$\theta^{(k)} = \theta^{(k-1)} - t_k \cdot x_i^\top \left(x_i \theta^{(k-1)} - y_i\right)$$
![[Pasted image 20260412203253.png|392]]
Figure SGD va aléatoirement choisir de garder le gradient de soit l'observation 1 ou l'observation 2, il ne calcule pas en mode uniforme ou pondéré il en prends que un

> **Remarque :** L'estimateur est non-biaisé, mais la variance reste élevée — la trajectoire est bruyante et oscille autour de $x^*$ sans converger exactement.

![[Pasted image 20260412204511.png]]
Figure c'est ce qu'on voit ici

---

### 9.2 SAG — Stochastic Average Gradient *(Schmidt et al. 2013)*

**Idée :** tenir à jour un carnet de $n$ cases, une par individu. À chaque pas on tire $i$, on recalcule uniquement $\nabla f_i$, et on utilise la moyenne de tout le carnet.

**Algorithme :**

- Étape 0 : initialiser $\theta^{(0)} = (0,0)$, carnet $g_1^0 = \cdots = g_n^0 = 0$
- À chaque $t$ :
  1. Tirer $i \sim \mathcal{U}\{1, \dots, n\}$
  2. Mettre à jour la case $i$ : $g_i^t = \nabla f_i(\theta^{(t)})$, les autres cases restent inchangées
  3. Calculer $\bar{g} = \frac{1}{n} \sum_{j=1}^n g_j^t$
  4. $\theta^{(t+1)} = \theta^{(t)} - t_k \cdot \bar{g}$

**Évolution du carnet** ($n = 10$, exemple avec 3 colonnes affichées) :

| $t$ | $g_1$ | $g_2$ | $g_3$ | $\cdots$ | $g_{10}$ | $\theta$ |
|-----|-------|-------|-------|----------|----------|---------|
| $t=0$ | $g_1^0$ | $g_2^0$ | $g_3^0$ | $\cdots$ | $g_{10}^0$ | $\theta^{(0)} = (0,0)$ |
| $t=1$, tire $i=3$ | $g_1^0$ | $g_2^0$ | $\mathbf{g_3^1}$ | $\cdots$ | $g_{10}^0$ | $\theta^{(1)} = \theta^{(0)} - t\bar{g}$ |
| $t=2$, tire $i=10$ | $g_1^0$ | $g_2^0$ | $g_3^1$ | $\cdots$ | $\mathbf{g_{10}^2}$ | $\theta^{(2)} = \theta^{(1)} - t\bar{g}$ |
| $t=3$, tire $i=1$ | $\mathbf{g_1^3}$ | $g_2^0$ | $g_3^1$ | $\cdots$ | $g_{10}^2$ | $\theta^{(3)} = \theta^{(2)} - t\bar{g}$ |

*(gras = case mise à jour ce tour)*

> **Remarque :** Estimateur **biaisé** — les cases non mises à jour ne correspondent pas au $\theta$ courant. La variance est réduite mais le biais est non nul. Convergence linéaire (preuve : ~50 pages).

---

### 9.3 SAGA *(Defazio, Bach et al. 2014)*

**Objectif :** corriger le biais de SAG tout en gardant la réduction de variance. Même carnet, mais mise à jour modifiée :

$$x^{(k+1)} = x^{(k)} - \gamma \left[\underbrace{\nabla f_i(x^{(k)})}_{\text{frais}} - \underbrace{g_i^k}_{\text{vieux}} + \underbrace{\bar{g}}_{\text{moyenne}}\right]$$

**Évolution du carnet SAGA** ($n = 3$) :

| $t$ | $g_1$ | $g_2$ | $g_3$ | $\bar{g}$ utilisé | $\theta$ |
|-----|-------|-------|-------|-------------------|---------|
| $t=0$ | $g_1^0$ | $g_2^0$ | $g_3^0$ | $\frac{1}{3}(g_1^0+g_2^0+g_3^0)$ | $\theta^{(0)}$ |
| $t=1$, tire $i=2$ | $g_1^0$ | $\mathbf{g_2^1}$ | $g_3^0$ | $\frac{1}{3}(g_1^0+g_2^1+g_3^0)$ | $\theta^{(1)} = \theta^{(0)} - \gamma[\nabla f_2(\theta^{(0)}) - g_2^0 + \bar{g}^0]$ |
| $t=2$, tire $i=2$ | $g_1^0$ | $\mathbf{g_2^2}$ | $g_3^0$ | $\frac{1}{3}(g_1^0+g_2^2+g_3^0)$ | $\theta^{(2)} = \theta^{(1)} - \gamma[\nabla f_2(\theta^{(1)}) - g_2^1 + \bar{g}^1]$ |
| $t=3$, tire $i=1$ | $\mathbf{g_1^3}$ | $g_2^2$ | $g_3^0$ | $\frac{1}{3}(g_1^3+g_2^2+g_3^0)$ | $\theta^{(3)} = \theta^{(2)} - \gamma[\nabla f_1(\theta^{(2)}) - g_1^0 + \bar{g}^2]$ |

*(gras = case mise à jour ce tour)*

**Motivation du correcteur $(\nabla f_i(x^{(k)}) - g_i^k)$ :**

En SAG, l'info stale est noyée dans la moyenne — on compense en baissant le learning rate. SAGA injecte directement la variation (frais − vieux) dans le pas : cela agit comme un **accélérateur de mise à jour** qui annule le biais. On récupère ainsi la propriété sans biais tout en gardant une variance faible.

> $\checkmark$ Sans biais + variance $\to 0$ pour $t \to +\infty$

---

### 9.4 Mini-batch gradient descent

Compromis entre GD exact et SGD : on tire un mini-lot de taille $b$ à chaque pas.

$$g_{\text{MB}} = \frac{1}{b} \sum_{i \in \text{batch}} \nabla f_i(x^{(k)})$$

$$\text{Var}(g_{\text{MB}}) = \frac{\text{Var}(g_{\text{SGD}})}{b}$$

En général $b = 32$, donc on divise la variance par 32 — mais il reste toujours du bruit (contrairement au gradient descent exact qui a variance nulle).

> Si $b = n$ $\Rightarrow$ variance nulle = gradient descent exact.

---

## 10. Méthodes adaptatives

L'idée commune : au lieu d'un learning rate global $\alpha$, **adapter le pas par coordonnée** en fonction de l'historique des gradients.

---

### 10.1 AdaGrad *(Duchi, Hazan, Singer — JMLR 2011)*

**Problème de SGD avec pas décroissant :** les features peu informatives reçoivent de moins en moins de signal au fil du temps — leur gradient est petit et le pas diminue encore davantage.

**Solution :** diviser le pas de chaque coordonnée $j$ par la racine carrée de la somme cumulée des gradients au carré :

$$x_j^{(k)} = x_j^{(k-1)} - \frac{\alpha}{\sqrt{\displaystyle\sum_{l=1}^k \left(g_j^{(l)}\right)^2} + \varepsilon} \cdot g_j^{(k)}$$

où $g^{(k)} = \nabla f_{i_k}(x^{(k-1)})$ et $\varepsilon > 0$ pour la stabilité numérique.

**Avantages :** $\alpha$ devient un hyperparamètre fixe (plus besoin de scheduler). Très efficace sur les problèmes sparse — les features rares reçoivent un grand pas car leur historique $\sum_l (g_j^{(l)})^2$ est petit.

**Limitation :** le dénominateur accumule indéfiniment $\Rightarrow$ le pas tend vers 0, l'apprentissage s'arrête trop tôt.

---

### 10.2 Moyennes exponentiellement pondérées (EWA)

Brique de base pour Momentum, RMSprop et Adam. Plutôt que de moyenner tous les pas passés avec un poids uniforme, on pondère exponentiellement : les observations récentes comptent plus.

$$v_t = \beta \, v_{t-1} + (1 - \beta)\, \theta_t, \qquad v_0 = 0$$

On interprète $v_t$ comme une moyenne sur $\approx \dfrac{1}{1-\beta}$ observations passées :

- $\beta = 0.9$ $\Rightarrow$ $\approx 10$ dernières valeurs
- $\beta = 0.98$ $\Rightarrow$ $\approx 50$ dernières valeurs (plus lisse, mais courbe décalée)
- $\beta = 0.5$ $\Rightarrow$ $\approx 2$ dernières valeurs (très réactif, bruité)

**Pourquoi exponentiel ?** En déroulant la récurrence :

$$v_t = (1-\beta) \sum_{l=0}^{t-1} \beta^l \, \theta_{t-l}$$

Les coefficients $\beta^l$ décroissent exponentiellement avec l'âge. On montre que $(1-\varepsilon)^{1/\varepsilon} \approx 1/e$, donc après $\frac{1}{1-\beta}$ pas les poids ont perdu $\approx 63\%$ de leur valeur.

**Biais en début de série :** $v_0 = 0$ $\Rightarrow$ les premières itérations sous-estiment la vraie moyenne. Correction :

$$\hat{v}_t = \frac{v_t}{1 - \beta^t}$$

Pour $t$ grand, $\beta^t \approx 0$ donc la correction devient négligeable.


exemple avec on voit qu'on a une importance de 10% pour la valeur courante

$$
v_8 = 0.1\theta_8 + 0.09\theta_7 + 0.081\theta_6 + 0.0729\theta_5 + 0.0656\theta_4 + 0.0590\theta_3 + 0.0531\theta_2 + 0.0478\theta_1
$$
Si on fait la somme des coefficients on obtient $0.5695$ 

or si je rajoute un sans biais
$$
\hat{v}_8=\frac{v_8}{1-0.9^8}
$$
on obtient donc 
$$
\hat{v}_8 = 0.1756\theta_8 + 0.1580\theta_7 + 0.1422\theta_6 + 0.1280\theta_5 + 0.1152\theta_4 + 0.1036\theta_3 + 0.0932\theta_2 + 0.0839\theta_1
$$



![[Pasted image 20260412213721.png]]
Figure vec beta=0.9

---

### 10.3 Gradient Descent avec Momentum *(Polyak 1964 ; Rumelhart, Hinton, Williams 1986)*

**Problème du GD standard :** oscillations dans les directions de forte courbure, progression lente dans les directions de faible courbure.

**Idée :** appliquer une EWA sur les gradients successifs pour accumuler une "vitesse" qui lisse les oscillations.

**Algorithme (notation réseau de neurones) :** initialiser $v_{dW} = 0$. À chaque itération $t$ :

$$v_{dW} = \beta \, v_{dW} + (1 - \beta)\, dW$$

$$W := W - \alpha \, v_{dW}$$

où $W$ sont les poids du réseau et $dW = \frac{\partial \mathcal{L}}{\partial W}$ le gradient calculé par backpropagation.

**Intuition physique :** $dW$ joue le rôle d'une accélération, $v_{dW}$ d'une vitesse, $\beta$ d'un frottement. La balle roule de plus en plus vite dans la bonne direction et amortit les oscillations transverses.

Valeur typique : $\beta = 0.9$ (moyenne sur $\approx 10$ derniers gradients).

> **Remarque :** certains papiers écrivent $v_{dW} = \beta \, v_{dW} + dW$ sans le $(1-\beta)$ — revient à scaler $v_{dW}$ d'un facteur $1/(1-\beta)$.

**Dans nos notations (régression linéaire, $\theta = (\theta_0, \theta_1)$) :**

On introduit un vecteur vitesse $v^{(k)} \in \mathbb{R}^2$, initialisé à $v^{(0)} = (0, 0)$. À chaque itération $k$ :

$$v^{(k)} = \beta \, v^{(k-1)} + (1-\beta) \cdot \frac{1}{n}\sum_{i=1}^n \nabla f_i(\theta^{(k)})$$

$$\theta^{(k+1)} = \theta^{(k)} - \alpha \, v^{(k)}$$

Le terme $\frac{1}{n}\sum_i \nabla f_i(\theta^{(k)})$ est exactement le gradient full-batch qu'on calculait avant — la seule différence c'est qu'on ne l'utilise pas directement comme pas, on en fait une moyenne glissante pondérée exponentiellement.

![[Pasted image 20260412213250.png]]
Figure explication visuelle, si on se rappelle l'image de la régeression linéaire j'avais calculé à un instant t la moyenne des gradients, la je fais la même chose sauf que je somme de façon pondéré inversement exponentiel les gradients du passé et celui actuel. Donc mon vecteur directionel va dépendre de ce qui a été fait dans le passé 

**Pourquoi ça aide sur notre exemple à 2 points ?** La surface de loss $f(\theta)$ est une parabole allongée dans l'espace $(\theta_0, \theta_1)$ — forte courbure selon $\theta_0 + \theta_1 x$ et faible courbure perpendiculairement. Sans momentum, le GD zigzague : les gradients successifs oscillent en signe dans la direction de forte courbure et s'accumulent dans la direction vers le minimum. Avec momentum, les oscillations s'annulent dans $v^{(k)}$ (alternance de signe $\to$ moyenne $\approx 0$) et la composante utile s'accumule (même signe $\to$ vitesse croissante $\to$ convergence plus rapide).

![[sto-5.png]]

Motivation: perso je comprends pas la logique de l'algo car on va récuépérer des gradients passé de direciton qui me disait déjà d'aller vers le min donc c'est inutile, lui il dit utile en fait car:
1. la loss est allongée (anisotrope)
2. SGD pas GD on utilise SGD en général avec GD with momentum 
3. Saddle point: car on a des points selles avec les réseaux de neurones classiques 



---

### 10.4 RMSprop *(Hinton — Coursera 2012, non publié)*

**Idée :** adapter le learning rate par coordonnée en utilisant une EWA des gradients **au carré** — ce qui évite l'accumulation infinie d'AdaGrad.

**Algorithme (notation réseau de neurones) :** initialiser $s_{dW} = 0$. À chaque itération $t$ :

$$s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2)\, dW^2$$

$$W := W - \alpha \, \frac{dW}{\sqrt{s_{dW} + \varepsilon}}$$

**Intuition :** les coordonnées à grands gradients (forte oscillation) ont $s_{dW}$ grand $\Rightarrow$ pas réduit $\Rightarrow$ amortissement. Les coordonnées à petits gradients (direction lente) ont $s_{dW}$ petit $\Rightarrow$ pas augmenté $\Rightarrow$ accélération.

**Dans nos notations (régression linéaire, $\theta = (\theta_0, \theta_1)$) :**

On introduit un vecteur $s^{(k)} \in \mathbb{R}^2$ qui accumule les gradients **au carré**, coordonnée par coordonnée. Initialiser $s^{(0)} = (0, 0)$. À chaque itération $k$ :

$$s^{(k)} = \beta_2 \, s^{(k-1)} + (1 - \beta_2) \cdot \left(\frac{1}{n}\sum_{i=1}^n \nabla f_i(\theta^{(k)})\right)^{\odot 2}$$

$$\theta^{(k+1)} = \theta^{(k)} - \frac{\alpha}{\sqrt{s^{(k)} + \varepsilon}} \odot \frac{1}{n}\sum_{i=1}^n \nabla f_i(\theta^{(k)})$$

où $\odot$ désigne le produit élément par élément. Explicitement sur nos deux coordonnées :

$$\theta_0^{(k+1)} = \theta_0^{(k)} - \frac{\alpha}{\sqrt{s_{\theta_0}^{(k)} + \varepsilon}} \cdot \frac{\partial J}{\partial \theta_0}(\theta^{(k)})$$

$$\theta_1^{(k+1)} = \theta_1^{(k)} - \frac{\alpha}{\sqrt{s_{\theta_1}^{(k)} + \varepsilon}} \cdot \frac{\partial J}{\partial \theta_1}(\theta^{(k)})$$

**Pourquoi ça aide sur notre exemple à 2 points ?** La clé c'est que $\theta_0$ et $\theta_1$ n'ont pas la même sensibilité dans la loss — la courbure est différente selon les deux directions. RMSprop mesure cette courbure empiriquement via $s^{(k)}$ : si $\frac{\partial J}{\partial \theta_0}$ est historiquement grand (forte courbure, direction oscillante), $s_{\theta_0}^{(k)}$ est grand $\Rightarrow$ le pas selon $\theta_0$ est divisé par un grand nombre $\Rightarrow$ amortissement. Si $\frac{\partial J}{\partial \theta_1}$ est historiquement petit (faible courbure, direction lente), $s_{\theta_1}^{(k)}$ est petit $\Rightarrow$ le pas selon $\theta_1$ est peu divisé $\Rightarrow$ accélération. On corrige ainsi automatiquement l'anisotropie de la loss sans avoir besoin de connaître la Hessienne.

![[sto-6.png]]

---

### 10.5 Adam — Adaptive Moment Estimation *(Kingma, Ba — ICLR 2015)*

**Idée :** Momentum + RMSprop + correction de biais. Adam estime le **1er moment** (moyenne de $dW$, via $\beta_1$) et le **2ème moment** (variance de $dW$, via $\beta_2$).

**Algorithme :** initialiser $v_{dW} = 0$, $s_{dW} = 0$. À chaque itération $t$ :

**(1) Momentum :**
$$v_{dW} = \beta_1 \, v_{dW} + (1 - \beta_1)\, dW$$

**(2) RMSprop :**
$$s_{dW} = \beta_2 \, s_{dW} + (1 - \beta_2)\, dW^2$$

**(3) Correction de biais :**
$$\hat{v}_{dW} = \frac{v_{dW}}{1 - \beta_1^t}, \qquad \hat{s}_{dW} = \frac{s_{dW}}{1 - \beta_2^t}$$

**(4) Mise à jour :**
$$W := W - \alpha \, \frac{\hat{v}_{dW}}{\sqrt{\hat{s}_{dW}} + \varepsilon}$$

**Hyperparamètres standards :**

| Hyperparamètre | Valeur typique | Rôle |
|---|---|---|
| $\alpha$ | à tuner | learning rate global |
| $\beta_1$ | $0.9$ | decay du 1er moment (momentum) |
| $\beta_2$ | $0.999$ | decay du 2ème moment (RMSprop) |
| $\varepsilon$ | $10^{-8}$ | stabilité numérique |

> Adam est l'optimiseur par défaut en deep learning — robuste au choix de $\alpha$, efficace sur les problèmes sparse et non-stationnaires.

---

## 11. Méthodes de second ordre : Newton et Quasi-Newton

### 11.1 Motivation et taux de convergence

La méthode de Newton exploite la courbure locale via la Hessienne — ce qui se paie : on doit calculer le **gradient du gradient**.

Taux de convergence comparés :

$$\frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} \xrightarrow{k \to \infty} \begin{cases} 0 & \text{superlinéaire (SC)} \\ \mu \in (0,1) & \text{linéaire} \end{cases}$$

- **Newton** $\to$ convergence quadratique
- **Quasi-Newton** $\to$ convergence superlinéaire
- **Gradient Descent** $\to$ convergence linéaire

**Calcul du gradient par différences finies.** Évaluer $\nabla f$ directement nécessite au minimum $n+1$ évaluations de $f$ (différence unilatérale). La différence bilatérale est plus précise :

$$\frac{\partial f}{\partial x}(x,y) \approx \frac{f(x+h,y) - f(x-h,y)}{2h}$$

Elle requiert $2n$ évaluations mais a une erreur en $O(h^2)$ contre $O(h)$ pour la différence unilatérale (résulte d'un développement de Taylor). En pratique, pour $n$ grand (millions de paramètres), inutilisable en ML.

**Calcul de la Hessienne par différences finies.** La Hessienne $\nabla^2 f \in \mathbb{R}^{n \times n}$ s'estime par :

$$f_{xy} = \frac{\partial^2 f}{\partial x \partial y} \approx \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right) \quad \Rightarrow \text{ différences finies en } y, \text{ puis en } x$$

Erreur en $O(h^3)$. En pratique, $n$ de l'ordre du million $\Rightarrow$ inutilisable.

---

### 11.2 Méthode de Newton — algorithme complet

Newton est une méthode de **line search** :

$$x_{k+1} = x_k + \alpha_k p_k$$

avec direction de Newton $p_k = -(\nabla^2 f_k)^{-1} \nabla f_k$ et line search inexacte.

**Algorithme :**

1. Choisir $x_0$
2. Calculer $p_k = -B_k^{-1} \nabla f_k$ — vérifier que $B_k \succ 0$ (sinon voir §11.3)
3. Calculer $\alpha_k$ par backtracking line search (ou inexacte)
4. $x_{k+1} = x_k + \alpha_k p_k$
5. Vérifier $\|\nabla f_{k+1}\| \approx 0$ : si oui ✓, sinon retour à (2)

**Direction de Newton vs gradient descent :**

$$p_k = \begin{cases} -(\nabla^2 f_k)^{-1} \nabla f_k & \text{direction de Newton} \\ -\nabla f_k & \text{gradient descent (= } B_k = I\text{)} \end{cases}$$

**Direction de descente :** on vérifie $\cos\theta = -\dfrac{\nabla f_k^\top p_k}{\|\nabla f_k\|\|p_k\|} > 0$, soit :

$$\nabla f_k^\top (\nabla^2 f_k)^{-1} \nabla f_k > 0 \iff (\nabla^2 f_k)^{-1} \succ 0 \iff \nabla^2 f_k \succ 0$$

Donc $p_k^{\text{Newton}}$ est une direction de descente **si et seulement si** $\nabla^2 f_k \succ 0$.

---

### 11.3 Que faire si $B_k$ n'est pas définie positive ?

Si $B_k \not\succ 0$, la direction de Newton n'est plus une direction de descente. Deux options :

**(1) Retomber sur gradient descent** — simple mais perd la convergence quadratique.

**(2) Hessian modification** — perturber $B_k$ pour la rendre PD tout en restant proche.

**Vérifier si $B_k \succ 0$ :** plusieurs méthodes de coût croissant :

- (a) **Valeurs propres** — $O(n^3)$, infaisable en grande dimension
- (b) **Test $z^\top B_k z > 0$** — trop coûteux à vérifier exhaustivement
- (c) **Décomposition de Cholesky** — si $A$ est PD, $A = LL^\top$ avec $L$ triangulaire inférieure et diagonale $> 0$; réciproque vraie. Coût $O(n^3)$ mais plus rapide en pratique
- (d) **Disques de Gershgorin** *(méthode la moins chère)*

**Théorème de Gershgorin.** Toute valeur propre de $A$ est dans au moins un disque $D_i$ défini par :

$$D_i : \text{centre } A_{ii}, \quad \text{rayon } R_i = \sum_{j \neq i} |A_{ij}|$$

Coût : simple addition de valeurs absolues — $O(n^2)$, pas d'inversion. Si tous les disques sont à droite de 0, alors $\lambda_{\min} > 0$ et $A \succ 0$.

> **Remarque :** $B_k$ est réelle symétrique $\Rightarrow$ valeurs propres réelles $\Rightarrow$ les disques sont sur l'axe réel. S'il y a un disque qui va vers les négatifs, c'est mauvais signe.

---

### 11.4 Hessian modification

**Problème :** $A = Q\Lambda Q^\top$ avec $\lambda_{\min} < 0$.

**Solution :** ajouter $\Delta A = \tau I = \tau Q Q^\top$ pour déplacer toutes les valeurs propres de $+\tau$ :

$$A + \Delta A = Q[\Lambda + \tau I] Q^\top$$

Les nouvelles valeurs propres sont $\lambda_i + \tau$. En choisissant $\tau > |\lambda_{\min}|$, toutes deviennent positives.

**Choix de $\tau$ :** on veut $\|\Delta A\|$ minimal $\Rightarrow$ perturber le moins possible. Puisque $\|\Delta A\| = \tau$ (car $\|I\| = 1$), on prend :

$$\tau > -\lambda_{\min} \quad \text{(uniquement nécessaire si } \lambda_{\min} < 0\text{)}$$

En pratique $\tau \gtrsim -\lambda_{\min}$ pour rester robuste (éviter des valeurs propres à $10^{-15}$).

---

### 11.5 Quasi-Newton — motivation et modèle

**Motivation :** dans Newton, on minimise le modèle $m_k$ à chaque itération — ce qui requiert recalculer $B_k = \nabla^2 f_k$ (coûteux). L'idée Quasi-Newton : ne pas recalculer $B_k$ de zéro, mais le **mettre à jour itérativement** depuis $B_{k-1}$.

**Modèle quadratique :** on choisit $B_k$ symétrique PD :

$$m_k(p) = f_k + \nabla f_k^\top p + \frac{1}{2} p^\top B_k \, p$$

Propriétés clés du modèle :
- $m_k(0) = f_k$ ✓
- $\nabla m_k(0) = \nabla f_k$ ✓ (interpolation du gradient en $x_k$)

On veut aussi que $\nabla m_{k+1}(0) = \nabla f_{k+1}$ **et** que $m_{k+1}$ interpole le gradient en $x_k$ :

$$\nabla m_{k+1}(-\alpha_k p_k) = \nabla f_k \implies \nabla f_{k+1} - B_{k+1} \alpha_k p_k = \nabla f_k$$

En posant $s_k = x_{k+1} - x_k = \alpha_k p_k$ et $y_k = \nabla f_{k+1} - \nabla f_k$ :

$$\boxed{y_k = B_{k+1} s_k} \qquad \text{(équation sécante)}$$

---

### 11.6 BFGS

**Relation BFGS :** $B_{k+1} s_k = y_k$, soit $s_k = B_{k+1}^{-1} y_k = H_{k+1} y_k$.

En multipliant à gauche par $s_k^\top$ : $s_k^\top B_{k+1} s_k = s_k^\top y_k > 0$ pour que $B_{k+1} \succ 0$ — c'est la **condition de courbure positive**.

$B_{k+1}$ est symétrique $\Rightarrow \frac{n(n+1)}{2}$ inconnues, mais l'équation sécante ne donne que $n$ contraintes $\Rightarrow$ système sous-déterminé. BFGS ajoute le critère : choisir $B_{k+1}$ aussi proche que possible de $B_k$. La mise à jour itérative de $H_k = B_k^{-1}$ est :

$$H_{k+1} = \left(I - \rho_k s_k y_k^\top\right) H_k \left(I - \rho_k y_k s_k^\top\right) + \rho_k s_k s_k^\top$$

où $\rho_k = (y_k^\top s_k)^{-1}$.

> **Avantage clé :** on ne stocke et n'inverse jamais $B_k$ — on met à jour directement $H_k = B_k^{-1}$, ce qui évite le $O(n^3)$ de l'inversion. BFGS est conçu pour préserver la propriété PD, donc pas besoin de Gershgorin.

> **Remarque :** BFGS n'approche pas juste la Hessienne — il utilise l'**équation sécante** (différence de gradients entre deux étapes) pour mettre à jour $B_k$. C'est fondamentalement différent d'une approximation par différences finies.

---

### 11.7 L-BFGS — Limited-memory BFGS

**Problème de BFGS :** stocker $H_k \in \mathbb{R}^{n \times n}$. Pour $n = 10^6$, $H_k$ fait $10^{12}$ entrées $\approx$ 4 To de RAM. Impossible.

**Solution L-BFGS :** ne pas stocker la matrice. Au lieu de $H_k$, on garde en mémoire les $m$ derniers vecteurs de déplacement et de changement de gradient :

$$s_k = x_{k+1} - x_k, \qquad y_k = \nabla f_{k+1} - \nabla f_k$$

En général $m \in [5, 20]$. On utilise ces vecteurs pour **calculer à la volée** la direction de descente $H_k \nabla f_k$ via l'algorithme **"two-loop recursion"** — sans jamais former $H_k$ explicitement.

> L-BFGS est l'algorithme Quasi-Newton de référence pour les grands problèmes en ML et optimisation numérique.
