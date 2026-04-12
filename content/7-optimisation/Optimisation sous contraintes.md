
A convex optimization problem is one of the form
$$
\begin{equation}
\begin{array}{ll}
\operatorname{minimize} & f_{0}(x) \\
\text { subject to } & f_{i}(x) \leq 0, \quad i=1, \ldots, m \\
& a_{i}^{T} x=b_{i}, \quad i=1, \ldots, p
\end{array}
\end{equation}
$$
to describe the problem of finding an $x$ that minimizes $f_{0}(x)$ among all $x$ that satisfy the conditions $f_{i}(x) \leq 0, i=1, \ldots, m,$ and $h_{i}(x)=0, i=1, \ldots, p .$ We call $x \in \mathbf{R}^{n}$
the optimization variable and the function $f_{0}: \mathbf{R}^{n} \rightarrow \mathbf{R}$ the objective function or cost function. The inequalities $f_{i}(x) \leq 0$ are called inequality constraints, and the corresponding functions $f_{i}: \mathbf{R}^{n} \rightarrow \mathbf{R}$ are called the inequality constraint functions. The equations $h_{i}(x)=0$ are called the equality constraints, and the functions $h_{i}: \mathbf{R}^{n} \rightarrow \mathbf{R}$ are the equality constraint functions. If there are no constraints $(i . e .,$ $m=p=0$ ) we say the problem (4.1) is unconstrained.

![[prob-1.png|530]]


\textbf{Definition (Optimal and feasible).} A point $x \in \mathcal{D}$ is feasible if it satisfies the constraints $f_{i}(x) \leq 0, i=1, \ldots, m,$ and $h_{i}(x)=0, i=1, \ldots, p$ The problem (4.1) is said to be feasible if there exists at least one feasible point, and infeasible otherwise. The set of all feasible points is called the feasible set or the constraint set. The optimal value $p^{\star}$ of the problem (4.1) is defined as
$$
p^{\star}=\inf \left\{f_{0}(x) \mid f_{i}(x) \leq 0, i=1, \ldots, m, h_{i}(x)=0, i=1, \ldots, p\right\}
$$
We allow $p^{\star}$ to take on the extended values $\pm \infty .$ If the problem is infeasible, we have $p^{\star}=\infty$ (following the standard convention that the infimum of the empty set is $\infty$ ).

+mettre un truc sur solution réalisable , feasible, unfeasible etc


## La dualité

On s'intéresse aux conditions d'optimalité pour les problèmes sous contraintes. L'idée centrale est de relier la géométrie des courbes de niveau de $f$ à celle des contraintes, via les multiplicateurs de Lagrange.

### (i) Contrainte d'égalité — Motivation géométrique

On considère le problème :
$$
\min_x f(x) \quad \text{s.t.} \quad c_1(x) = 0
$$
![[Pasted image 20260409173205.png|389]]

**Exemple.** On prend $f(x) = x_1 + x_2$ et $c_1(x) = x_1^2 + x_2^2 - 2 = 0$.

$$
\nabla f(x) = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \qquad \nabla c_1(x) = 2\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

Le points A est la solution car c'est le point le plus bas sur la contrainte. Géométriquement, au point optimal $\bar{x}$ les courbes de niveau de $f$ sont tangentes au cercle, ce qui se traduit par :
$$
\nabla f(\bar{x}) = \lambda \, \nabla c_1(\bar{x})
$$

### Direction de déplacement faisable

Soit un point réalisable $x$ sur le cercle. Pour me déplacer vers $x + s$ en restant réalisable, on note déjà deux propriétés qu'il faut toujours rester précisément sur la contrainte car il y'a égalité ainsi on a que $c_1(x)=0$ et que $c_1(x+s)=0$ . Ainsi si on développe par Taylor :
$$
c_1(x + s) \approx c_1(x) + \nabla c_1(x)^T s = 0 \quad \Rightarrow \quad \nabla c_1(x)^T s = 0
$$

La direction $s$ doit être **orthogonale à $\nabla c_1$** : elle est sur la tangente au cercle pour ne pas quitter la contrainte on va se déplacer de façon infinitésimal. Pour que $x + s$ soit un meilleur point, on veut $f(x+s) < f(x)$, soit par Taylor :
$$
f(x+s) \approx f(x) + \nabla f(x)^T s 
$$
Or pour que $f(x+s)<f(x) ~~\Rightarrow ~~ \nabla f(x)^T s < 0$. Géométriquement, $\nabla f^T s < 0$ définit un demi-espace (half-plane). On cherche donc $s$ vérifiant simultanément les deux legs :

$$
\boxed{\nabla c_1(x)^T s = 0 \quad \text{(faisabilité)} \qquad \text{et} \qquad \nabla f(x)^T s < 0 \quad \text{(amélioration)}}
$$

Quand aucun tel $s$ n'existe, on est à un point stationnaire.

![[Pasted image 20260409175202.png|351]]
### Le lagrangien et la signature d'un point stationnaire

**Idée.** Au lieu de jongler avec $f$ et $c_1$ séparément, on cherche le point stationnaire du **lagrangien**, on repart de la formule $\nabla f(\bar{x}) - \lambda \, \nabla c_1(\bar{x}) = 0$ simplement avant l'optimal ce n'est pas égal à zéro mais c'est égal à  $\mathcal{L}(x, \lambda)$
$$
\mathcal{L}(x, \lambda) = f(x) - \lambda \, c_1(x)
$$

Les conditions de stationnarité sont :
$$
\nabla_x \mathcal{L}(\bar{x}, \lambda) = 0 \quad \Rightarrow \quad \nabla f(\bar{x}) = \lambda \, \nabla c_1(\bar{x})
$$
$$
\frac{\partial \mathcal{L}}{\partial \lambda} = 0 \quad \Rightarrow \quad c_1(\bar{x}) = 0
$$

On est ainsi passé d'un problème avec contrainte à un problème **sans contrainte** sur le lagrangien.

### (ii) Contrainte d'inégalité — Quand améliorer ?

On considère maintenant :
$$
\min_x f(x) = x_1 + x_2 \quad \text{s.t.} \quad c_2(x) = 2 - x_1^2 - x_2^2 \geq 0
$$
(rester à l'intérieur ou sur le bord du disque)

> 📌 **Image à rajouter** : disque avec points A (bord gauche $(-\sqrt{2},0)$), B (bord droit $(\sqrt{2},0)$), C (bord haut), lignes de niveau de $f$ — illustre les deux questions : comment améliorer depuis $x$, quand s'arrêter.
![[Pasted image 20260410152826.png|371]]

Deux questions : (i) depuis $x$, comment améliorer ? (ii) quand s'arrêter ?

Pour aller en $x + s$ avec $c_2(x+s) \geq 0$ :
$$
c_2(x+s) \approx c_2(x) + \nabla c_2(x)^T s \geq 0
$$

#### Cas 1 : Contrainte inactive — $c_2(x) > 0$ (intérieur)

La contrainte n'est pas active. N'importe quel petit $s$ reste faisable. On peut prendre $s = -\alpha \nabla f$ comme en optimisation non-contrainte.

**Quand s'arrêter ?** Quand $\nabla f(x) = 0$ — il n'y a plus de direction faisable de descente. Si $\nabla f(x) = 0$, avec $\lambda_2 = 0$ dans le lagrangien, on a bien $\nabla_x \mathcal{L} = 0$.

#### Cas 2 : Contrainte active — $c_2(x) = 0$ (frontière)

> 📌 **Image à rajouter** : disque avec $\nabla f$ et $\nabla c_2$ représentés à un point de la frontière, deux demi-espaces colorés (bleu pour $\nabla c_2^T s \geq 0$, rouge pour $\nabla f^T s \leq 0$) — illustre le cône d'intersection qui devient vide au point stationnaire.
![[Pasted image 20260410152550.png|235]]
$$
\nabla c_2(x) = -2\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

Soit un point réalisable $x$ sur le cercle. Pour me déplacer vers $x + s$ en restant réalisable, je peux aller soit dans le cercle (contrainte inactive) soit sur sa frontière (contrainte active). On peut donc écrire que $c_2(x)=0$ et que $c_2(x+s)\ge 0$ . Ainsi si on développe par Taylor :
$$
c_2(x + s) \approx c_2(x) + \nabla c_2(x)^T s \ge 0 \quad \Rightarrow \quad \nabla c_2(x)^T s \ge 0
$$

La direction $s$ doit être **orthogonale à $\nabla c_2$** : elle est sur la tangente. Pour que $x + s$ soit un meilleur point, on veut $f(x+s) < f(x)$, soit par Taylor :
$$
f(x+s) \approx f(x) + \nabla f(x)^T s 
$$
Or pour que $f(x+s)<f(x) ~~\Rightarrow ~~ \nabla f(x)^T s < 0$. Géométriquement, $\nabla f^T s < 0$ définit un demi-espace (half-plane). On a deux conditions simultanées :
$$
\nabla c_2(x)^T s \geq 0 \quad \text{(faisabilité)} \qquad \text{et} \qquad \nabla f(x)^T s < 0 \quad \text{(décroissance)}
$$

Géométriquement :
- $\nabla c_2^T s \geq 0$ : région bleue (angle $< 90°$, produit scalaire positif)
- $\nabla f^T s \leq 0$ : région rouge ($270°$)

**Aucune intersection** $\Leftrightarrow$ on ne peut plus améliorer $\Leftrightarrow$ point stationnaire. Cela se produit quand $\nabla f$ et $\nabla c_2$ sont anti-parallèles :
$$
\nabla f(x) = \lambda_2 \, \nabla c_2(x) \quad \text{avec} \quad \lambda_2 \geq 0
$$
Le signe $\lambda_2 \geq 0$ est crucial : il capture le point A (minimum) et pas le point C.
![[Pasted image 20260410154156.png|233]]
### Conditions KKT — Contrainte d'inégalité

Le lagrangien est :
$$
\mathcal{L}(x, \lambda_2) = f(x) - \lambda_2 \, c_2(x)
$$

$$
\boxed{
\nabla_x \mathcal{L}(x^*, \lambda_2^*) = 0 \quad \text{et} \quad \lambda_2^* \geq 0
}
$$

L'élégance vient de réunir les deux cas via la **condition de complémentarité** :
$$
\lambda_2 \, c_2(x^*) = 0
$$

Cela signifie :
- Soit $\lambda_2 = 0$ (cas 1 : intérieur, contrainte inactive)
- Soit $c_2(x^*) = 0$ (cas 2 : frontière, contrainte active)

C'est la condition centrale en optimisation sous contraintes.

### Cas général — Contraintes multiples

On considère :
$$
\min_x f(x)=x_1+x_2 \quad \text{s.t.} \quad \begin{cases} c_1(x) = 2 - x_1^2 - x_2^2 \geq 0 \\ c_2(x) = x_2 \geq 0 \end{cases}
$$

> 📌 **Image à rajouter** : demi-disque (intersection des deux contraintes, partie supérieure du disque), points $A = (-\sqrt{2}, 0)$ et $B = (\sqrt{2}, 0)$ sur la frontière, lignes de niveau de $f = x_1 + x_2$, gradients $\nabla c_1$, $\nabla c_2$ et $\nabla f$ — illustre la région faisable et pourquoi A est candidat et pas B.

![[Pasted image 20260410154253.png|279]]
![[Pasted image 20260410154414.png|219]]
**Mise en forme standard** : toutes les inégalités doivent être $\geq 0$. Le lagrangien devient :
$$
\mathcal{L}(x, \lambda) = f(x) - \lambda_1 c_1(x) - \lambda_2 c_2(x) = f(x) - \sum_{i} \lambda_i c_i(x)
$$

Pour une direction faisable $d$ depuis un point où les deux contraintes sont actives :
$$
\nabla f(x)^T d < 0 \quad \text{(décroissance)}, \qquad \nabla c_i(x)^T d \geq 0 \quad \forall i \quad \text{(faisabilité)}
$$
(1) $\nabla_x \mathcal{L}(x, \lambda)=0$ et $\lambda_1 \ge 0, \lambda_2 \ge 0$ 
(2) $\lambda_1 c_1(x)=0, \lambda_2 c_2(x)=0$ 
$\nabla_x \mathcal{L}(x, \lambda) = \begin{pmatrix}1 \\ 1\end{pmatrix}+\lambda_1 \begin{pmatrix}2 x_1 \\ 2 x_2\end{pmatrix} - \lambda_2 \begin{pmatrix}0 \\ 1\end{pmatrix}=\begin{pmatrix}1+2\lambda_1 x_1 \\ 1+2 \lambda_1 x_2 - \lambda_2\end{pmatrix}=0$

Au point A(-sqrt(2), 0) no intersection, no feasible direction on peut résoudre
$\nabla_x \mathcal{L}(x, \lambda) =\begin{pmatrix}1-2\sqrt{2}\lambda_1 \\ 1-\lambda_2\end{pmatrix}=\begin{pmatrix}0\\0\end{pmatrix} \iff \begin{cases} \lambda_1=1/(2\sqrt{2})\\\lambda_2=1\end{cases}$
good car $\lambda_i \ge 0$ + complimentary active constraint donc : on peut dire que le point A est stationary point ? En fait faudrait dire que A est un candidat pour a stationary point/minima point.
![[Pasted image 20260410154414.png|160]]
Au point B(sqrt2,0) intersection ok jaune on a un cone qui te donen la direction où aller 
$\nabla_x \mathcal{L}(x, \lambda) =\begin{pmatrix}1+2\sqrt{2}\lambda_1 \\ 1-\lambda_2\end{pmatrix}=\begin{pmatrix}0\\0\end{pmatrix} \iff \begin{cases} \lambda_1=-1/(2\sqrt{2})\\\lambda_2=1\end{cases}$
Fail B is not a stationary point
![[Pasted image 20260410155408.png|203]]


### Conditions KKT — test d'optimalité

On formalise maintenant. Le lagrangien général est :
$$
\mathcal{L}(x, \lambda) = f(x) - \sum_{i \in \mathcal{E} \cup \mathcal{I}} \lambda_i c_i(x)
$$
où $\mathcal{E}$ est l'ensemble des contraintes d'égalité et $\mathcal{I}$ celui des inégalités.

**Théorème (First Order Necessary Conditions — KKT).** Supposons que :
1. $x^*$ est une solution locale de $\min_{x \in \Omega} f(x)$,
2. $f$ et les $c_i$ sont continûment différentiables,
3. une condition de qualification des contraintes (CQC) est vérifiée en $x^*$.

Alors il existe des multiplicateurs de Lagrange $\lambda^*$ tels que les conditions suivantes sont satisfaites en $(x^*, \lambda^*)$ :

$$
\boxed{
\begin{aligned}
&(a) \quad \nabla_x \mathcal{L}(x^*, \lambda^*) = 0 \\
&(b) \quad c_i(x^*) = 0, \quad i \in \mathcal{E} \\
&(c) \quad c_i(x^*) \geq 0, \quad i \in \mathcal{I} \\
&(d) \quad \lambda_i^* \geq 0, \quad i \in \mathcal{I} \\
&(e) \quad \lambda_i^* \, c_i(x^*) = 0, \quad i \in \mathcal{E} \cup \mathcal{I} \quad \textbf{(complémentarité)}
\end{aligned}
}
$$

Il peut y avoir beaucoup de points où (a)-(e) sont satisfaites — ce sont les candidats. Si de plus la CQC ② est vérifiée, alors $\lambda^*$ est **unique**.

### Exemple numérique — Vérification KKT

> 📌 **Image à rajouter** : page de calcul avec $f(x) = x_1 + x_2$, contraintes $c_1(x) = x_1^2 + x_2^2 - 2 = 0$ (égalité) et $c_2(x) = x_2 \geq 0$, calcul des multiplicateurs aux points A et B, vérification des conditions KKT.


On reprend $f(x) = x_1 + x_2$ avec les deux contraintes. La condition $\nabla_x \mathcal{L} = 0$ donne :
$$
\begin{pmatrix} 1 \\ 1 \end{pmatrix} - \lambda_1 \cdot 2\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} - \lambda_2 \begin{pmatrix} 0 \\ 1 \end{pmatrix} = 0
\quad \Rightarrow \quad
\begin{cases} 1 + 2\lambda_1 x_1 = 0 \\ 1 + 2\lambda_1 x_2 - \lambda_2 = 0 \end{cases}
$$

**Au point A $= (-\sqrt{2}, 0)$** (les deux contraintes actives) :
$$
1 - 2\sqrt{2}\,\lambda_1 = 0 \quad \Rightarrow \quad \lambda_1 = \frac{1}{2\sqrt{2}} > 0 \checkmark
$$
$$
1 - \lambda_2 = 0 \quad \Rightarrow \quad \lambda_2 = 1 > 0 \checkmark
$$
Complémentarité : $\lambda_1 c_1 = 0$ ✓ et $\lambda_2 c_2 = 1 \cdot 0 = 0$ ✓. **A est un point stationnaire candidat** (candidat au minimum).

**Au point B $= (\sqrt{2}, 0)$** :
$$
1 + 2\sqrt{2}\,\lambda_1 = 0 \quad \Rightarrow \quad \lambda_1 = -\frac{1}{2\sqrt{2}} < 0 \times
$$
$\lambda_1 < 0$ viole la condition (d). **B n'est pas un point stationnaire.**

**Exemple KKT avec 4 contraintes.** On prend :
$$
f(x) = \left(x_1 - \tfrac{3}{2}\right)^2 + \left(x_2 - \tfrac{1}{2}\right)^4
$$
avec les contraintes :
$$
c_1 = 1 - x_1 - x_2 \geq 0, \quad c_2 = 1 - x_1 + x_2 \geq 0, \quad c_3 = 1 + x_1 - x_2 \geq 0, \quad c_4 = 1 + x_1 + x_2 \geq 0
$$
> 📌 **Image à rajouter** : carré de contraintes avec courbes de niveau de $f$ (ellipses centrées en $(3/2, 1/2)$), points $(-1, 0)$, $(1, 0)$, $(0, 1)$, $(0, -1)$ aux coins, optimum à $(1, 0)$.
![[Pasted image 20260410155552.png|227]]
On teste le point $(1, 0)$ : quelles contraintes sont actives ? $c_1 = 0$, $c_2 = 0$, mais $c_3, c_4 > 0$ donc inactives. Par complémentarité $\lambda_3 = 0 = \lambda_4$.

La condition $\nabla_x \mathcal{L} = 0$ donne $\nabla f = \lambda_1 \nabla c_1 + \lambda_2 \nabla c_2$, soit :
$$
\begin{pmatrix} 2(x_1 - 3/2) \\ 4(x_2 - 1/2)^3 \end{pmatrix} = \lambda_1 \begin{pmatrix} -1 \\ -1 \end{pmatrix} + \lambda_2 \begin{pmatrix} -1 \\ 1 \end{pmatrix} = \begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \end{pmatrix}
$$

En substituant $(1, 0)$ : $\nabla f = \begin{pmatrix} -1 \\ -1/2 \end{pmatrix}$. La matrice est inversible donc $\lambda^*$ est unique :
$$
\lambda_1 = \frac{3}{4}, \quad \lambda_2 = \frac{1}{4} > 0 \quad \Rightarrow \quad \textbf{point optimal}
$$

### KKT et dualité

Les KKT sont surtout utiles en **optimisation convexe**. L'approche est la suivante : on résout d'abord en mode "boîte noire" puis on résout via la dualité.

**Exemple.** $f(x) = \frac{2}{5}(x_1^2 + x_2^2)$, $\min_x f(x)$ s.t. $C(x) = x_1 + x_2 - 2 \geq 0$ (demi-espace).

> 📌 **Image à rajouter** : courbes de niveau de $f$ (cercles concentriques, comme un bol), demi-espace $x_1 + x_2 \geq 2$, point optimal $(\lambda_1, 1)$ sur la frontière, région faisable hachurée.

![[Pasted image 20260410160011.png|365]]


Le lagrangien : $\mathcal{L}(x, \lambda) = f(x) - \lambda C(x)$.

KKT : $\nabla_x f - \lambda \nabla C = 0$, $\lambda \geq 0$, et par complémentarité $\lambda C(x) = 0$.
$$
\frac{4}{5} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} - \lambda \begin{pmatrix} 1 \\ 1 \end{pmatrix} = 0 \quad \Rightarrow \quad x_1 = x_2 = \frac{5\lambda}{4}
$$

Si $\lambda = 1$ : $x^* = \begin{pmatrix} 5/4 \\ 5/4 \end{pmatrix}$ mais $C(x^*) = 5/2 - 2 \neq 0$, donc pas faisable. Si $\lambda = 4/5$ : $x^* = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $C(x^*) = 0$ ✓.

**Deux façons de résoudre (2 étapes) :**

**Étape 1 — Fonction duale de Lagrange** : $q(\lambda) = \min_x \mathcal{L}(x, \lambda)$.

En résolvant $\nabla_x \mathcal{L} = 0$ : $x_i^* = \frac{5\lambda}{4}$, valeur optimale :
$$
q(\lambda) = \frac{2}{5}\left(\frac{5\lambda}{4}\right)^2 \cdot 2 - \lambda\left(\frac{5\lambda}{2} - 2\right) = -\frac{5}{4}\lambda^2 + 2\lambda \quad \Rightarrow \text{ parabole descendante}
$$

**Étape 2 — Maximiser $q(\lambda)$ s.t. $\lambda \geq 0$** :
$$
q'(\lambda) = -\frac{10}{4}\lambda + 2 = 0 \quad \Rightarrow \quad \lambda^* = \frac{4}{5}
$$
$$
\Rightarrow \quad x_1^* = x_2^* = 1 \quad \text{(approche régulière = approche primale)}
$$

La **solution primale = solution duale** : c'est la **dualité forte**.

### Problème primal et dual

$$
\text{Primal} : \min_x f(x) \quad \text{s.t.} \quad C(x) \geq 0
$$
$$
\text{Dual} : \max_\lambda q(\lambda) \quad \text{s.t.} \quad \lambda \geq 0, \quad i \in \mathcal{I}
$$
$$
\text{où} \quad q(\lambda) = \min_x \mathcal{L}(x, \lambda)
$$

Quelle approche choisir ? Si le problème est **convexe**, les deux donnent le même résultat. Le dual peut être plus facile à résoudre que le primal. Pour un problème **non convexe**, il n'y a pas de dualité forte : primal $\neq$ dual, on parle de **duality gap**.

IL FAUT PARLER DE STRONG DUALITY ET AVEC DES EXEMPLES EN HAUT DIRE QUE LEURS FONCTION DE COUTS SONT EGALE 

### Intuition géométrique de la dualité

**Exemple.** $f(x) = \frac{2}{5}(x_1^2 + x_2^2)$, $C(x) = x_1 + x_2 - 2 \geq 0$.

On change d'espace : on va dans l'espace $(y, z)$ où $y = C(x)$ et $z = f(x)$.

> 📌 **Image à rajouter** : deux schémas côte à côte. Gauche : espace $(x_1, x_2)$ avec région faisable $x_1+x_2 \geq 2$ hachurée et cercles de niveau de $f$. Droite : espace $(y, z)$ avec l'image $G$ de la correspondance $(C(x), f(x))$, droites de la forme $z = \lambda y + d$ (tangentes à $G$), optimum au point $(-2, 0)$ sur l'axe $y=0$.

![[Pasted image 20260410160054.png]]


![[Pasted image 20260412161302.png|413]]

un peu le bordel mais y'a le graphe du lagrangien aussi ?

![[Pasted image 20260412161429.png|368]]



On a :
$$
\begin{pmatrix} y \\ z \end{pmatrix} = \begin{pmatrix} C(x) \\ f(x) \end{pmatrix} \quad \Rightarrow \quad y \geq 0 \text{ car } C(x) \geq 0
$$

En fixant $x_2 = a$ et en éliminant $x_1$ :
$$
y = x_1 + a - 2, \qquad z = \frac{2}{5}(x_1^2 + a^2) \quad \Rightarrow \quad z - \frac{2}{5}a^2 = \frac{2}{5}(y - a + 2)^2
$$

Pour $a = 0$ : parabole $z = \frac{2}{5}(y+2)^2$. Si $a > 0$ elle se décale vers le haut — ce sont les paraboles à dessiner.

**Problème primal** dans cet espace : $\min z$ s.t. $y \geq 0$, i.e. la plus petite valeur de $z$ en gardant $y \geq 0$ — visible graphiquement.

**Problème dual** : $\mathcal{L}(x, \lambda) = z - \lambda y = \alpha$, soit $z = \lambda y + \alpha$ (droite de pente $\lambda$, intercept $\alpha$). On cherche le plus petit $\alpha$ tel que la droite touche encore $G$ :
$$
q(\lambda) = \min_{y,z \in G} [\underbrace{z - \lambda y}_{=\alpha}] = \alpha^*
$$
![[Pasted image 20260410160150.png|450]]
**Étape 1** (pente fixée $\lambda$) : trouver la tangente à $G$ de pente $\lambda$ → donne $q(\lambda)$.
**Étape 2** : maximiser $q(\lambda)$ sur $\lambda \geq 0$ → trouver la pente optimale.
![[Pasted image 20260410160216.png|368]]
Cela se produit en $T_2$ : $T_2$ est la solution du problème dual. **Primal solution = Dual solution = strong duality**.


étape 1:
On reprend l'exemple. Étape 1 c'est calculer :

$$
q(\lambda)=\min _x \mathcal{L}(x, \lambda)=\min _x[f(x)-\lambda C(x)]
$$


Avec $f(x)=\frac{2}{5}\left(x_1^2+x_2^2\right)$ et $C(x)=x_1+x_2-2$ :

$$
q(\lambda)=\min _{x_1, x_2}\left[\frac{2}{5}\left(x_1^2+x_2^2\right)-\lambda\left(x_1+x_2-2\right)\right]
$$


C'est un problème sans contrainte - on dérive et on annule:

$$
\begin{aligned}
& \frac{\partial \mathcal{L}}{\partial x_1}=\frac{4}{5} x_1-\lambda=0 \quad \Rightarrow \quad x_1^*=\frac{5 \lambda}{4} \\
& \frac{\partial \mathcal{L}}{\partial x_2}=\frac{4}{5} x_2-\lambda=0 \quad \Rightarrow \quad x_2^*=\frac{5 \lambda}{4}
\end{aligned}
$$


On substitue dans $\mathcal{L}$ :

$$
q(\lambda)=\frac{2}{5} \cdot 2 \cdot \frac{25 \lambda^2}{16}-\lambda\left(\frac{5 \lambda}{2}-2\right)=\frac{5 \lambda^2}{4}-\frac{5 \lambda^2}{2}+2 \lambda=-\frac{5}{4} \lambda^2+2 \lambda
$$


C'est la parabole concave qu'on avait plottée. L'étape 1 c'est juste une dérivée simple — c'est pour ça que la dualité est utile, le min sans contrainte est trivial.
![[Pasted image 20260411131315.png|379]]

### Propriétés de la fonction duale $q$

**Mise à jour sur la définition.** Pour certains $\lambda$, $q(\lambda) \to -\infty$ (pas borné inférieurement). Le domaine de $q$ est :
$$
\mathcal{D} = \{\lambda \mid q(\lambda) > -\infty\}
$$

**Deux résultats fondamentaux** (valables pour **toute** fonction $f$, même non convexe) :

1. **$q$ est concave** (en tant que minimum de fonctions linéaires en $\lambda$).
2. **$\mathcal{D}$ est convexe** (domaine d'une fonction concave).

$\Rightarrow$ Le problème dual est toujours un problème de **maximisation d'une fonction concave** sur un domaine convexe — c'est un problème convexe, et on dispose de nombreux outils pour le résoudre.

- Si $f$ est convexe : dualité forte → primal = dual, pas de gap.
- Si $f$ non convexe : **duality gap** (primal $\neq$ dual en général).




## Les programmes quadratiques

Résolution: méthode barrière 

f(x)=x^2+4x+5 on peut avoir deux types de contraintes: equality constraint vs inequality constraint ie x=5 => x_opt =5 et x > 1  on sait que x_opt=-2 => f(-2)=1, or on voit que quand x>1 la solution est f(1)=10.
On introduit la méthode barrier method, pour la contrainte d'égalité on peut pensenr à une fonction f(x,y)=x^2+y^2+3x+y-2 on peut la plotter en 3D  et on a la contrainte d'égalité y=5 , on trace donc une cross section qui est similaire que le 2D plot de gauche , le but est de trouver le x_opt , on peut voir ça aussi comme f(x,5)=x^2+25+3x+5-2=x^2+3x+28 donc ça dépend plus de y on peut écrire que f(x)=x^2+3x+28 => c'est une parabole, en utilisant gradient descent ou newton on a une solution direct x_opt pour le y=5
<=> equality contrained probleme to unconstrained problem.

![[Pasted image 20260411132037.png]]

Barriere method est une way to transform un problem d(inequality en un probleme unconstrained avec lequel on peut utilsier gradient descent.

on a la fonction f(x)=x^2+4x+5 on a x>1 iff x-1 > 0 => on veut le transformer en unconstrained problem on a un nouveau terme
f(x)=x^2+4x+5-b \cdot log_e(x-1) il dit on utilise le faite que x-1>0  il dit qu'on peut faire que b est very small ça impliquera que f(x) blue approx f(x) black





## Les autres programmes SDPs et Coniques j'ai pas trop vu anyway
