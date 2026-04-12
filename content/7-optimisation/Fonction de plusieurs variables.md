---
title: Rappels de calcul multivariable
date: 2026-04-08
tags: [optimisation, calcul différentiel, gradient, hessien, taylor]
---

## 0. Pourquoi ces rappels ?

L'optimisation sans contrainte se résume à une idée simple : trouver les points où la fonction *ne peut plus descendre dans aucune direction*. En un tel point $\mathbf{x}^*$, la pente est nulle dans toutes les directions — ce qui se traduit par $\nabla f(\mathbf{x}^*) = 0$.

Mais savoir que le gradient est nul ne suffit pas. Un point peut être un minimum, un maximum, ou un point-selle. Pour distinguer ces cas, il faut regarder la courbure de la fonction — encodée dans la **matrice hessienne** $H_f$. Et pour relier tout ça à des inégalités utiles, il faut le **théorème de Taylor à l'ordre 2**.

Ce chapitre pose les fondations. Voilà le fil directeur :

$$\text{géométrie} \;\longrightarrow\; \text{dérivées partielles} \;\longrightarrow\; \text{gradient} \;\longrightarrow\; \text{plan tangent} \;\longrightarrow\; \text{hessien} \;\longrightarrow\; \text{Taylor}$$

---

## 1. Géométrie dans $\mathbb{R}^n$

Avant de parler de fonctions, on a besoin d'objets géométriques simples : droites et plans. Ils vont réapparaître plus loin comme *plans tangents* à une surface.

### Équation d'une droite en 2D

On se donne deux ingrédients :
- un vecteur $\vec{n} = \begin{pmatrix} a \\ b \end{pmatrix}$ — la **normale**, qui fixe l'orientation de la droite
- un point de référence $P_0(x_0, y_0)$ quelconque dans le plan

On définit la droite comme l'ensemble des points $P(x, y)$ tels que $\overrightarrow{P_0 P}$ est **perpendiculaire à $\vec{n}$**.

> [!note]- Construction de l'équation
> On écrit la condition de perpendicularité :
> 
> $$\vec{n} \cdot \overrightarrow{P_0 P} = 0$$
> 
> $$\begin{pmatrix} a \\ b \end{pmatrix} \cdot \begin{pmatrix} x - x_0 \\ y - y_0 \end{pmatrix} = 0 \;\Longrightarrow\; a(x - x_0) + b(y - y_0) = 0$$
> 
> En posant $c = ax_0 + by_0$, on obtient l'équation générale :
> 
> $$ax + by = c$$

*Exemple.* Si $P_0 = (0, 0)$, alors $c = a \cdot 0 + b \cdot 0 = 0$ et l'équation est $ax + by = 0$ — c'est la droite qui **passe par l'origine**, la droite du bas sur le schéma. Pour obtenir la droite du haut (décalée), il suffit de prendre un $P_0$ différent, ce qui change $c$ et déplace la droite parallèlement dans la direction de $\vec{n}$.

![[Pasted image 20260406162033.png|386]]

En forme standard $y = -\frac{a}{b} x + \frac{c}{b}$, on reconnaît la pente $-a/b$ et l'ordonnée à l'origine $c/b$.

*Remarque.* Sur le schéma, $\theta = -a/b$ désigne la **pente** de la droite, pas un angle au sens trigonométrique.

### Équation d'un plan en 3D

La construction est exactement la même qu'en 2D, mais dans $\mathbb{R}^3$. On se donne :
- un vecteur $\vec{n} = \begin{pmatrix} a \\ b \\ c \end{pmatrix}$ — la **normale**, qui fixe l'orientation du plan
- un point de référence $P_0(x_0, y_0, z_0)$ quelconque dans l'espace

On définit le plan comme l'ensemble des points $P(x, y, z)$ tels que $\overrightarrow{P_0 P}$ est **perpendiculaire à $\vec{n}$**.

> [!note]- Construction de l'équation
> On écrit la condition de perpendicularité :
> 
> $$\vec{n} \cdot \overrightarrow{P_0 P} = 0$$
> 
> $$\begin{pmatrix} a \\ b \\ c \end{pmatrix} \cdot \begin{pmatrix} x - x_0 \\ y - y_0 \\ z - z_0 \end{pmatrix} = 0 \;\Longrightarrow\; a(x - x_0) + b(y - y_0) + c(z - z_0) = 0$$
> 
> En posant $d = ax_0 + by_0 + cz_0$, on obtient l'équation générale :
> 
> $$ax + by + cz = d$$

*Exemple.* Plan passant par $P_0(1,2,3)$ perpendiculaire à $\vec{n} = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$ :

$$4(x-1) + 5(y-2) + 6(z-3) = 0 \;\Longrightarrow\; 4x + 5y + 6z = 32$$

![[Pasted image 20260406163224.png|315]]

![[Pasted image 20260406162116.png|304]]

### Forme implicite vs forme paramétrique

Il existe deux façons fondamentalement différentes de décrire un objet géométrique.

**Forme implicite** : on décrit l'objet comme l'ensemble des points qui *vérifient une condition*. C'est ce qu'on a fait jusqu'ici — $ax + by = c$ filtre tous les points qui satisfont la contrainte. On ne génère pas les points, on les sélectionne.

**Forme paramétrique** : on décrit l'objet en *générant* ses points à partir d'un paramètre $t$ qui varie. C'est l'idée d'une **courbe paramétrée** :

**Définition.** Une **courbe paramétrée** dans $\mathbb{R}^n$ est une fonction $\vec{r} : \mathbb{R} \to \mathbb{R}^n$ qui associe un point de l'espace à chaque valeur du paramètre $t$. En faisant varier $t$, on trace une courbe.

Le cas le plus simple est la **droite paramétrée** : on part d'un point $\vec{r}_0$ et on se déplace dans une direction $\vec{v}$ à vitesse constante :

$\vec{r}(t) = \vec{r}_0 + t\vec{v}$

En faisant varier $t \in \mathbb{R}$, on parcourt toute la droite. Les deux formes décrivent le même objet : en 2D, si on prend $\vec{v} = \langle 1, m \rangle$, on retrouve exactement $y - y_0 = m(x - x_0)$, la forme implicite classique.

Mais une courbe paramétrée peut décrire des objets bien plus généraux qu'une droite — un cercle, une spirale, ou une ligne de niveau quelconque. C'est précisément ce qu'on utilisera en §4 pour décrire les lignes de niveau de $f$.

*Pourquoi deux formes ?* En 3D, une droite ne peut pas s'écrire avec une seule équation implicite — une équation en 3D définit un plan, pas une droite. La forme paramétrique est alors bien plus naturelle pour décrire une droite en 3D avec une seule expression.

> [!example]- Exemple : droite en 3D comme intersection de deux plans
> Prends ces deux plans :
> $\begin{cases} x + y + z = 3 \\ x - y + z = 1 \end{cases}$
> On soustrait la deuxième de la première : $2y = 2$ donc $y = 1$. Et on a $x + z = 2$, soit $z = 2 - x$. Donc $x$ est libre — c'est notre paramètre. En posant $x = t$ :
> $\begin{cases} x = t \\ y = 1 \\ z = 2 - t \end{cases}$
> Ce qui s'écrit sous forme paramétrique :
> $\vec{r}(t) = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix} + t \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}$
> Le système implicite (deux équations) et la forme paramétrique (un point + une direction) décrivent exactement la même droite.

### Généralisation : l'hyperplan en $\mathbb{R}^n$

La droite en 2D et le plan en 3D sont en fait le même objet vu dans des dimensions différentes — on les appelle tous les deux un **hyperplan**.

**Définition.** Dans $\mathbb{R}^n$, un hyperplan est l'ensemble des points $\mathbf{x} = (x_1, \dots, x_n)$ vérifiant :

$$\vec{n} \cdot (\mathbf{x} - \mathbf{x}_0) = 0 \;\Longleftrightarrow\; \vec{n} \cdot \mathbf{x} = c$$

où $\vec{n} \in \mathbb{R}^n$ est le vecteur normal et $c = \vec{n} \cdot \mathbf{x}_0$. C'est exactement la même construction que précédemment — seule la dimension change.

Un hyperplan dans $\mathbb{R}^n$ a dimension $n-1$ : il enlève un degré de liberté à l'espace. C'est pour ça qu'une droite ($n-1 = 1$) dans $\mathbb{R}^2$ et un plan ($n-1 = 2$) dans $\mathbb{R}^3$ sont tous les deux des hyperplans.

| Espace | Hyperplan | Dimension de l'hyperplan | Équation |
|---|---|---|---|
| $\mathbb{R}^2$ | droite | 1 | $ax + by = c$ |
| $\mathbb{R}^3$ | plan | 2 | $ax + by + cz = d$ |
| $\mathbb{R}^n$ | hyperplan | $n-1$ | $\vec{n} \cdot \mathbf{x} = c$ |

C'est cette notion d'hyperplan qui réapparaîtra dans l'optimisation sous contrainte : une contrainte linéaire $\vec{n} \cdot \mathbf{x} = c$ définit un hyperplan sur lequel on cherche le minimum de $f$.

---

## 2. Fonctions de plusieurs variables — visualisation

On s'intéresse à une fonction $f : \mathbb{R}^2 \to \mathbb{R}$. Son graphe est une surface dans $\mathbb{R}^3$ : l'ensemble des points $(x, y, f(x,y))$.

### Lignes de niveau

**Définition.** Une **ligne de niveau** de $f$ pour la valeur $C$ est l'ensemble des points du domaine où $f$ vaut $C$ :

$$\{(x, y) \in \mathbb{R}^2 \mid f(x, y) = C\}$$

C'est une courbe dans le plan 2D — pas dans l'espace 3D. La définition vit entièrement dans le domaine de $f$.

Pour visualiser ce que ça représente, on peut imaginer couper le graphe de $f$ avec des plans horizontaux successifs $z = C$. L'intersection de la surface avec chacun de ces plans donne une courbe dans l'espace 3D, et sa projection sur le plan $xy$ est exactement la ligne de niveau.

**Étape 1.** On tranche la surface avec les plans $z = 3, 2, 1, 0, -1, -2$ — chaque plan coupe la surface selon une courbe.

![[Pasted image 20260406171806.png|237]]

**Étape 2.** On détache ces courbes de la surface pour les voir seules dans l'espace.

![[Pasted image 20260406171834.png|257]]

**Étape 3.** On projette toutes ces courbes sur le plan 2D.

![[Pasted image 20260406171855.png|269]]

**Étape 4.** On obtient le **contour plot** — la vue 2D complète des lignes de niveau.

![[Pasted image 20260406171916.png|346]]

**Étape 5.** On ajoute des couleurs pour représenter la hauteur — les zones claires sont hautes, les zones sombres sont basses.

![[Pasted image 20260406172001.png|407]]

Là où les lignes de niveau sont **serrées**, la surface monte ou descend vite — la pente est forte. Là où elles sont **espacées**, la surface est quasi plate. Cette représentation sera essentielle pour interpréter le gradient.

---

## 3. Dérivées partielles

### Définition

Pour une fonction $f(x, y)$, la **dérivée partielle par rapport à $x$** en $(x_0, y_0)$ mesure la pente de $f$ quand on bouge dans la direction $x$ uniquement, $y$ étant bloqué à $y_0$. Géométriquement, on *tranche* la surface par le plan $y = y_0$ et on regarde la tangente à la courbe obtenue.

L'image suivante montre les deux cas côte à côte — à gauche on fixe $y = y_0$ et on fait varier $x$, à droite on fixe $x = x_0$ et on fait varier $y$. La tangente rouge c'est la dérivée partielle, et les points $f(x_0, y_0)$ et $f(x, y_0)$ (ou $f(x_0, y)$) sont annotés sur la courbe bleue.

![[im3.png|518]]

> [!note]- À améliorer
> Annoter cette image avec $\Delta x$, $\Delta f$ style 3blue1brown pour montrer visuellement le rise/run → slope = $\Delta f / \Delta x$.

Formellement :

$$\left.\frac{\partial f}{\partial x}\right|_{(x_0, y_0)} = \lim_{h \to 0} \frac{f(x_0 + h, y_0) - f(x_0, y_0)}{h}$$

$$\left.\frac{\partial f}{\partial y}\right|_{(x_0, y_0)} = \lim_{h \to 0} \frac{f(x_0, y_0 + h) - f(x_0, y_0)}{h}$$

### Exemple concret

Sur $f(x,y) = x^2 y + \sin(y)$ au point $(-1, 1)$, on cherche $\frac{\partial f}{\partial x}(-1,1)$. On tranche par le plan $y = 1$ — la courbe rouge sur la surface ci-dessous — et on lit la pente de la tangente bleue en ce point.

![[Pasted image 20260406172450.png|443]]

On fixe $y = 1$ et on découpe selon $x$ :

![[Pasted image 20260406172425.png|490]]

$$\frac{\partial f}{\partial x} = 2xy \;\Longrightarrow\; \frac{\partial f}{\partial x}(-1,1) = 2(-1)(1) = -2$$

De même pour $\frac{\partial f}{\partial y}(-1,1)$, on fixe $x = -1$ et on découpe selon $y$ :

![[Pasted image 20260406172526.png|418]]

$$\frac{\partial f}{\partial y} = x^2 + \cos(y) \;\Longrightarrow\; \frac{\partial f}{\partial y}(-1,1) = (-1)^2 + \cos(1) = 1 + \cos(1) \approx 1.54$$

> [!note]- À améliorer
> Remplacer les images 3blue1brown ci-dessus par des versions annotées avec $f(x_0, y_0)$, $f(x, y_0)$, $\Delta x$ et $\Delta f$ visibles sur le graphe — sur le modèle de `im3.png`.

### Calcul pratique

En pratique, $\frac{\partial f}{\partial x}$ se calcule en dérivant $f$ par rapport à $x$ en traitant $y$ comme une constante. Par exemple, pour $f(x,y) = x^2 y + y^3$ :

$$\frac{\partial f}{\partial x} = 2xy \qquad \frac{\partial f}{\partial y} = x^2 + 3y^2$$

### Règle de la chaîne multivariable

Si $f(x,y)$ et que $x = x(t)$, $y = y(t)$ dépendent d'un paramètre $t$, alors :

$$\frac{d}{dt} f(x(t), y(t)) = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}$$

En notation compacte avec $\mathbf{x}(t) = (x(t), y(t))$ :

$$\frac{d}{dt} f(\mathbf{x}(t)) = \nabla f(\mathbf{x}(t)) \cdot \mathbf{x}'(t)$$

C'est la généralisation directe de la règle de la chaîne en 1D. Elle sera utilisée à plusieurs reprises dans la suite — notamment pour dériver la dérivée directionnelle et pour Taylor.

---

## 4. Dérivée directionnelle et gradient

### Dérivée directionnelle

Les dérivées partielles mesurent la variation selon $x$ ou $y$. Mais on peut se demander : *quelle est la pente dans une direction quelconque ?*

Soit $\vec{u} = \begin{pmatrix} u_1 \\ u_2 \end{pmatrix}$ un vecteur unitaire ($|\vec{u}| = 1$). On paramétrise le déplacement depuis $(x_0, y_0)$ dans la direction $\vec{u}$ :

$$x(s) = x_0 + s u_1, \qquad y(s) = y_0 + s u_2$$

La **dérivée directionnelle** de $f$ en $(x_0, y_0)$ dans la direction $\vec{u}$ est :

$$\boxed{D_{\vec{u}} f(x_0, y_0) = \lim_{s \to 0} \frac{f(x_0 + s u_1, y_0 + s u_2) - f(x_0, y_0)}{s}}$$

![[part1-3.png|293]]

### Lien avec le gradient

Par la règle de la chaîne appliquée à $g(s) = f(x(s), y(s))$ :

$$D_{\vec{u}} f(x_0, y_0) = \frac{d}{ds} f(x(s), y(s))\bigg|_{s=0} = \frac{\partial f}{\partial x} u_1 + \frac{\partial f}{\partial y} u_2$$

On introduit le **gradient** :

$$\nabla f = \left\langle \frac{\partial f}{\partial x},\, \frac{\partial f}{\partial y} \right\rangle$$

Et on obtient la formule de calcul :

$$\boxed{D_{\vec{u}} f(x_0, y_0) = \nabla f\big|_{(x_0,y_0)} \cdot \vec{u}}$$

*Exemple.* $f(x, y) = x^2 y$ en $(x_0, y_0) = (1, 2)$ dans la direction $\vec{u}=\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$ — c'est la fonction illustrée sur le schéma ci-dessus.

$\nabla f = \langle 2xy, x^2 \rangle \;\Longrightarrow\; \nabla f(1,2) = \langle 4, 1 \rangle$

$D_{\vec{u}} f(1,2) = \langle 4, 1 \rangle \cdot \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right) = \frac{5}{\sqrt{2}} \approx 3.54$
![[Pasted image 20260406173249.png|531]]
> [!note]- À améliorer
> Annoter l'image `173249` avec la fonction $f(x,y) = x^2 y$, le point $(1,2)$ et le vecteur $\vec{u}$ pour que le schéma corresponde à cet exemple.

### Interprétation géométrique du gradient

Avant d'interpréter le gradient, il faut savoir ce qu'est un **champ de vecteurs**. Une fonction scalaire $f : \mathbb{R}^2 \to \mathbb{R}$ associe un *nombre* à chaque point. Un champ de vecteurs $F : \mathbb{R}^2 \to \mathbb{R}^2$ associe un *vecteur* à chaque point — on peut l'imaginer comme une flèche attachée à chaque point du plan. En physique c'est le champ électrique ou le champ de vitesse d'un fluide. En ML tu le vois pour visualiser comment un algorithme d'optimisation se déplace dans l'espace des paramètres — chaque flèche dit "si tu es ici, pousse dans cette direction".

Le gradient $\nabla f$ est exactement un champ de vecteurs : en chaque point $(x, y)$ il produit le vecteur $\begin{pmatrix} \partial f/\partial x \\ \partial f/\partial y \end{pmatrix}$.

![[Pasted image 20260406172747.png|588]]

Visuellement, les flèches du gradient pointent toujours dans la direction de montée la plus rapide :

![[Pasted image 20260406172842.png|524]]

Le gradient admet deux interprétations fondamentales.

**1. Le gradient est orthogonal aux lignes de niveau.**

On utilise ici la forme paramétrique vue en §1 : une ligne de niveau peut se décrire comme une courbe $\vec{r}(t) = x(t)\hat{\imath} + y(t)\hat{\jmath}$ paramétrée par $t$, telle que $f(\vec{r}(t)) = C$ (on reste sur la même ligne de niveau). En dérivant par rapport à $t$ via la règle de la chaîne :

$$\frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} = 0 \;\Longrightarrow\; \nabla f \cdot \frac{d\vec{r}}{dt} = 0$$

Donc $\nabla f$ est perpendiculaire au vecteur tangent à la ligne de niveau.

![[im10.png|527]]

**2. Le gradient pointe dans la direction de montée maximale.**

En réécrivant la dérivée directionnelle :

$$D_{\vec{u}} f = |\nabla f| \cos\theta$$

- Maximum quand $\theta = 0$ : $\vec{u} \parallel \nabla f$ → montée la plus rapide.
- Minimum quand $\theta = \pi$ : descente la plus rapide, direction $-\nabla f$.
- Nul quand $\theta = \pi/2$ : on se déplace sur une ligne de niveau.

![[Pasted image 20260406164226.png|524]]

![[part1-4.png|201]]

---

## 5. Plan tangent et linéarisation locale

### Motivation

En 1D, la tangente à $y = f(x)$ en $x_0$ est la droite $y = f(x_0) + f'(x_0)(x - x_0)$. C'est la meilleure approximation linéaire de $f$ près de $x_0$.

![[Pasted image 20260406183940.png|421]]

En 2D, l'analogue est le **plan tangent** au graphe de $f$ au point $(x_0, y_0, f(x_0, y_0))$. On cherche une fonction $L_f(x,y)$ dont le graphe est un plan passant par ce point.

![[Pasted image 20260406184107.png|510]]

Plein de plans peuvent passer par ce point — il faut un critère supplémentaire. Si on fait l'intersection avec le plan $y = 2$ (plan gris clair), on obtient une droite :

![[Pasted image 20260406184343.png|205]]

On a plein de plans en parallèle :

![[Pasted image 20260406184418.png|213]]

### Construction du plan tangent

On écrit un plan générique passant par $(x_0, y_0, z_0)$ avec $z_0 = f(x_0, y_0)$ :

$L(x,y) = a(x - x_0) + b(y - y_0) + z_0$

Il y a une infinité de plans qui passent par ce point — on a trois degrés de liberté ($a$, $b$, $z_0$) et pour l'instant seul $z_0$ est fixé. Il faut donc trois contraintes pour fixer le plan uniquement.

**Fait 1 : $L(x_0, y_0) = z_0 = f(x_0, y_0)$**

Le plan doit passer par le point $(x_0, y_0, f(x_0, y_0))$ sur la surface. C'est déjà satisfait par notre écriture. Ça fixe $z_0$ mais laisse $a$ et $b$ libres — il reste encore une infinité de plans.

![[Pasted image 20260406184343.png|205]]

![[Pasted image 20260406184418.png|213]]

**Fait 2 : $L_x(x_0, y_0) = f_x(x_0, y_0)$**

Si on fixe $y = y_0$ et qu'on regarde la tranche du plan, on obtient la droite $L(x, y_0) = a(x - x_0) + z_0$. Cette droite doit être tangente à la courbe $f(x, y_0)$ en $x_0$, donc sa pente doit valoir la dérivée partielle : $a = f_x(x_0, y_0)$.

![[Pasted image 20260406184629.png|168]]

**Fait 3 : $L_y(x_0, y_0) = f_y(x_0, y_0)$**

De même en fixant $x = x_0$, on obtient $b = f_y(x_0, y_0)$.

Ces trois faits fixent le plan complètement — il n'en existe qu'un seul qui vérifie les trois conditions simultanément.

![[Pasted image 20260406185121.png|196]]

### Équation du plan tangent

En substituant $a = f_x(x_0, y_0)$ et $b = f_y(x_0, y_0)$ :

$\boxed{L_f(x,y) = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)}$

En notation vectorielle avec $\mathbf{x} = \begin{pmatrix} x \\ y \end{pmatrix}$ et $\mathbf{x}_0 = \begin{pmatrix} x_0 \\ y_0 \end{pmatrix}$ :

$L_f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0)$

*Remarque.* Le vecteur normal au plan tangent est $\vec{n} = \begin{pmatrix} f_x \\ f_y \\ -1 \end{pmatrix}$ — c'est pourquoi on choisit $c = -1$ dans la forme générale $a(x-x_0) + b(y-y_0) + c(z-z_0) = 0$.

*Remarque.* Le gradient $\nabla f$ est défini sur le contour plot au sol — c'est un vecteur 2D. Le vecteur normal $\vec{n}$ au plan tangent, lui, est 3D.

![[Pasted image 20260406162220.png|562]]

![[plane-deriv-1.png|313]]

### Exemple de calcul

Pour $f(x,y) = 3 - \frac{1}{3}x^2 - y^2$ au point $(1, -2)$ :

$$f(1,-2) = 3 - \frac{1}{3} - 4 = -\frac{4}{3}$$

On veut $a = f_x(1,-2)$ — visuellement c'est la tangente line dans la direction $x$ :

![[Pasted image 20260406190126.png|277]]

Et le plan tangent a la même slope dans cette direction :

![[Pasted image 20260406190109.png|239]]

$$f_x(1,-2) = -\frac{2}{3}, \qquad f_y(1,-2) = 4$$

![[Pasted image 20260406190245.png|255]]

La tangente dans la direction $y$ :

![[Pasted image 20260406190314.png|219]]

Le plan tangent final :

![[Pasted image 20260406190547.png|191]]

$$L_f(x,y) = -\frac{4}{3} - \frac{2}{3}(x-1) + 4(y+2)$$

*Exemple 2.* $f(x,y) = x^2 + y^2$ en $(x_0, y_0) = (1,1)$ :

$$f(1,1) = 2, \quad \nabla f(1,1) = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

Plan tangent : $z = 2 + [2, 2] \cdot [x-1, y-1] = 2x + 2y - 2$.

---

## 6. Le hessien

### 6.1 Motivation : maxima et minima en 2D

Pour qu'un point $(x_0, y_0)$ soit un maximum local, le plan tangent doit être complètement plat. Formellement, cela signifie :

$$f_x(x_0, y_0) = 0 \quad \text{et} \quad f_y(x_0, y_0) = 0 \quad \Longleftrightarrow \quad \nabla f(x_0, y_0) = 0$$

![[Pasted image 20260406173826.png|418]]

Le slope selon $y$ est également nul :

![[Pasted image 20260406173909.png|379]]

On peut avoir des minima locaux, des maxima locaux :

![[Pasted image 20260406174142.png|358]]

Et aussi des **saddle points** (points-selle) — un phénomène qui n'existe pas en 1D. Sur un saddle point, le plan tangent est flat mais on a un minimum dans une direction et un maximum dans l'autre :

![[Pasted image 20260406174453.png|347]]

![[Pasted image 20260406174526.png|258]]

![[Pasted image 20260406174618.png|336]]

Par exemple pour $f(x,y) = x^2 - y^2$ : $f_x = 2x = 0$ et $f_y = -2y = 0$ mais $x$ et $y$ sont en désaccord sur si c'est un max ou un min.

Pour trancher, il faut le **test de la dérivée seconde** — l'analogue 2D de la concavité en 1D.

### 6.2 Dérivées secondes directionnelles

On peut itérer la dérivée directionnelle. Si $g(s) = f(\mathbf{x}_0 + s\mathbf{u})$, alors $g'(0) = D_{\vec{u}}f(\mathbf{x}_0)$ est la pente, et $g''(0)$ mesure la **courbure** de la tranche dans la direction $\vec{u}$.

Par la règle de la chaîne appliquée deux fois :

$$g''(0) = u_1^2 \frac{\partial^2 f}{\partial x^2} + 2u_1 u_2 \frac{\partial^2 f}{\partial x \partial y} + u_2^2 \frac{\partial^2 f}{\partial y^2}$$

Ce qui motive naturellement la définition matricielle qui suit.

### 6.3 Définition de la matrice hessienne

**Définition.** Soit $f$ une fonction admettant des dérivées partielles secondes en $\mathbf{x}$. Le **hessien** de $f$ en $\mathbf{x}$ est la matrice $2 \times 2$ :

$$H_f(\mathbf{x}) = \begin{bmatrix} \dfrac{\partial^2 f}{\partial x^2} & \dfrac{\partial^2 f}{\partial x \partial y} \\[10pt] \dfrac{\partial^2 f}{\partial y \partial x} & \dfrac{\partial^2 f}{\partial y^2} \end{bmatrix}$$

Si $f$ est suffisamment régulière (dérivées secondes continues), le théorème de Schwarz donne $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$ et $H_f$ est **symétrique**. Le résultat clé est :

$$g''(0) = \mathbf{u}^T H_f(\mathbf{x}_0)\, \mathbf{u}$$

La dérivée seconde selon $x$ (diagonale de $H_f$) nous donne la concavité dans la direction $x$ :

![[Pasted image 20260406182515.png|166]]

Pareil pour $y$ :

![[Pasted image 20260406182552.png|184]]

Mais le terme croisé $f_{xy}$ peut imposer son style. Pour $f(x,y) = xy$ : $f_{xx} = 0$, $f_{yy} = 0$, $f_{xy} = 1$ — c'est le terme croisé qui domine et donne un saddle point :

![[Pasted image 20260406182900.png|178]]

### 6.4 Directions de courbure minimale et maximale

Puisque $H_f$ est symétrique, elle admet une base orthonormée de vecteurs propres $\mathbf{u}_1, \mathbf{u}_2$ associés aux valeurs propres réelles $\lambda_1 \leq \lambda_2$. Pour tout vecteur unitaire $\mathbf{u} = \alpha \mathbf{u}_1 + \beta \mathbf{u}_2$ (avec $\alpha^2 + \beta^2 = 1$) :

$$\mathbf{u}^T H_f \mathbf{u} = \alpha^2 \lambda_1 + \beta^2 \lambda_2$$

C'est une moyenne pondérée de $\lambda_1$ et $\lambda_2$, donc :

$$\lambda_1 \leq \mathbf{u}^T H_f \mathbf{u} \leq \lambda_2$$

**Théorème.** La courbure minimale vaut $\lambda_1$ et est atteinte dans la direction $\pm\mathbf{u}_1$. La courbure maximale vaut $\lambda_2$ et est atteinte dans la direction $\pm\mathbf{u}_2$.

### 6.5 Matrices définies positives et classification

Une matrice symétrique $A$ est dite :
- **définie positive** ($A \succ 0$) si $\mathbf{u}^T A \mathbf{u} > 0$ pour tout $\mathbf{u} \neq 0$, i.e. toutes ses valeurs propres sont $> 0$
- **définie négative** ($A \prec 0$) si $\mathbf{u}^T A \mathbf{u} < 0$ pour tout $\mathbf{u} \neq 0$, i.e. toutes ses valeurs propres sont $< 0$
- **indéfinie** si elle a des valeurs propres de signes opposés

Pour le hessien, cela se traduit directement :
- $H_f(\mathbf{x}_0) \succ 0$ : la surface courbe *vers le haut* dans toutes les directions → $\mathbf{x}_0$ est un **minimum local**
- $H_f(\mathbf{x}_0) \prec 0$ : la surface courbe *vers le bas* dans toutes les directions → $\mathbf{x}_0$ est un **maximum local**
- $H_f(\mathbf{x}_0)$ indéfinie : certaines directions montent, d'autres descendent → **point-selle**

Un cas limite : si $H_f$ est semi-définie positive ($\lambda_1 = 0$), on a une infinité de points minimum dans une direction — le test ne conclut pas :

![[Pasted image 20260406181633.png|231]]

### 6.6 Test du second ordre

En 1D, on cherche $f'(x) = 0$ puis on regarde $f''$ pour la concavité :
équation cest deux pentes d'affilé on prends leur distance proche 
f^{\prime \prime}(x)=\lim _{h \rightarrow 0} \frac{f^{\prime}(x+h)-f^{\prime}(x)}{h}
je sais pas j'ai ça comme image https://www.youtube.com/watch?v=RQcUy9IfoAg


![[Pasted image 20260406175142.png|312]]

En 2D, la même logique s'applique. **Étape 1 :** résoudre $\nabla f(\mathbf{x}) = 0$.

**Étape 2 :** En chaque point critique $\mathbf{x}_0$, calculer le discriminant :

$$\Delta = f_{xx}(x_0,y_0)\, f_{yy}(x_0,y_0) - f_{xy}(x_0,y_0)^2 = \det H_f(\mathbf{x}_0)$$

- Si $\Delta > 0$ et $f_{xx} > 0$ : minimum local
- Si $\Delta > 0$ et $f_{xx} < 0$ : maximum local
- Si $\Delta < 0$ : point-selle
- Si $\Delta = 0$ : test non concluant

*Pourquoi le terme croisé ?* Considérons $f(x,y) = x^2 + y^2 + 4xy$. On a $f_{xx} = f_{yy} = 2 > 0$, ce qui semble indiquer un minimum dans chaque direction axiale. Pourtant $\Delta = 4 - 16 = -12 < 0$ : c'est un point-selle.

![[Pasted image 20260406181102.png|211]]

*Exemple.* $f(x,y) = x^4 - 4x^2 + y^2$. Points critiques : $(0,0)$, $(\sqrt{2}, 0)$, $(-\sqrt{2}, 0)$.

![[Pasted image 20260406175449.png|244]]

Concavité selon $x$ pour chaque point critique :

![[Pasted image 20260406175636.png|226]]

Concavité selon $x$ :

![[Pasted image 20260406175930.png|236]]

Concavité selon $y$ — toujours positive :

![[Pasted image 20260406180022.png|262]]

Pour $(\sqrt{2}, 0)$ : $f_{xx} = 16 > 0$, $f_{yy} = 2$, $f_{xy} = 0$, $\Delta = 32 > 0$ → minimum local.

Pour $(0,0)$ : $f_{xx} = -8 < 0$, $f_{yy} = 2$, $f_{xy} = 0$, $\Delta = -16 < 0$ → point-selle.

---

## 7. Approximation de Taylor à l'ordre 2

### Rappel en 1D

Pour $g : \mathbb{R} \to \mathbb{R}$ deux fois continûment différentiable :

$$g(t) = g(t_0) + g'(t_0)(t - t_0) + \frac{1}{2} g''(c)(t - t_0)^2$$

pour un certain $c$ entre $t_0$ et $t$ (forme de Lagrange du reste).

### Théorème de Taylor en dimension $n$

**Théorème.** Soit $f$ une fonction à dérivées partielles secondes continues sur un ouvert convexe $U \subset \mathbb{R}^2$. Pour tout $\mathbf{x}, \mathbf{x}_0 \in U$ :

$$f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H_f(\mathbf{x}_0 + c(\mathbf{x} - \mathbf{x}_0))(\mathbf{x} - \mathbf{x}_0)$$

pour un certain $c \in [0,1]$.

*Preuve (idée).* Poser $g(t) = f(\mathbf{x}_0 + t\mathbf{v})$ avec $\mathbf{v} = \mathbf{x} - \mathbf{x}_0$, appliquer Taylor 1D à $g$, et utiliser :
- $g'(0) = \mathbf{v} \cdot \nabla f(\mathbf{x}_0)$ (dérivée directionnelle)
- $g''(c) = \mathbf{v}^T H_f(\mathbf{x}_0 + c\mathbf{v})\, \mathbf{v}$ (dérivée directionnelle seconde)

### Approximation quadratique

En supprimant le reste (approximation locale) :

$$f(\mathbf{x}) \approx \underbrace{f(\mathbf{x}_0)}_{\text{constante}} + \underbrace{\nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0)}_{\text{terme linéaire}} + \underbrace{\frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H_f(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)}_{\text{terme quadratique}}$$

*Exemple.* Approximation quadratique de $\cos(x)\sin(y)$ :

![[quad-1.png|401]]

### Lien avec l'optimisation

Au voisinage d'un point critique $\mathbf{x}^*$ (où $\nabla f(\mathbf{x}^*) = 0$), le terme linéaire disparaît et il reste :

$$f(\mathbf{x}) \approx f(\mathbf{x}^*) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^T H_f(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*)$$

La classification du point critique se réduit au signe de la forme quadratique $\mathbf{v}^T H_f(\mathbf{x}^*)\mathbf{v}$ — ce qui ramène exactement aux conditions de définie positive/négative de la section 6.5. Taylor justifie rigoureusement le test du second ordre.

*Note.* Il manque ici un développement sur ce qu'il se passe exactement avec l'erreur $\varepsilon$ entre le plan et la fonction dans une fenêtre suffisamment petite — c'est le lien avec l'écriture de Taylor $f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T(\mathbf{x}-\mathbf{x}_0) + o(\|\mathbf{x}-\mathbf{x}_0\|)$.

---

## 8. Calcul numérique du gradient et du hessien

En pratique, les fonctions de coût n'ont pas toujours une expression analytique simple. Il faut donc calculer $\nabla f$ et $H_f$ numériquement.

### Différences finies pour le gradient

L'approximation la plus simple est la **différence finie centrée** :

$$\frac{\partial f}{\partial x_i}(\mathbf{x}) \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$$

où $\mathbf{e}_i$ est le $i$-ème vecteur de la base standard et $h$ est un petit pas. L'erreur est en $O(h^2)$. Pour $n$ variables, calculer $\nabla f$ coûte $2n$ évaluations de $f$.

### Calcul numérique du hessien

Un terme diagonal :

$$\frac{\partial^2 f}{\partial x_i^2}(\mathbf{x}) \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - 2f(\mathbf{x}) + f(\mathbf{x} - h\mathbf{e}_i)}{h^2}$$

Un terme croisé :

$$\frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{x}) \approx \frac{f(\mathbf{x} + h\mathbf{e}_i + h\mathbf{e}_j) - f(\mathbf{x} + h\mathbf{e}_i - h\mathbf{e}_j) - f(\mathbf{x} - h\mathbf{e}_i + h\mathbf{e}_j) + f(\mathbf{x} - h\mathbf{e}_i - h\mathbf{e}_j)}{4h^2}$$

### Le problème de passage à l'échelle

La matrice hessienne a $\frac{n(n+1)}{2}$ entrées distinctes. En grande dimension, stocker et calculer $H_f$ devient prohibitif. C'est pourquoi les algorithmes d'optimisation modernes utilisent :
- des **approximations du hessien** (méthodes quasi-Newton comme BFGS)
- uniquement le gradient $\nabla f$ (descente de gradient et ses variantes)
- la **différentiation automatique** (autograd, JAX) qui calcule $\nabla f$ exactement au coût d'une seule passe forward

---

## 9. Convexité

La convexité est le concept le plus important de l'optimisation. Une fonction convexe c'est une fonction où tout **minimum local est automatiquement un minimum global** — on n'a pas à s'inquiéter de rester bloqué dans un mauvais creux.

### Définition géométrique

**Définition.** Un ensemble $C \subset \mathbb{R}^n$ est **convexe** si pour tous $\mathbf{x}, \mathbf{y} \in C$ et tout $t \in [0,1]$ :

$t\mathbf{x} + (1-t)\mathbf{y} \in C$

Autrement dit : le segment reliant deux points quelconques de $C$ reste entièrement dans $C$. Un disque est convexe, un croissant ne l'est pas.

**Définition.** Une fonction $f : C \to \mathbb{R}$ définie sur un ensemble convexe $C$ est **convexe** si pour tous $\mathbf{x}, \mathbf{y} \in C$ et tout $t \in [0,1]$ :

$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq t f(\mathbf{x}) + (1-t) f(\mathbf{y})$

Géométriquement : la corde reliant deux points du graphe est **au-dessus** de la courbe. La fonction « courbe vers le haut » partout.

Elle est **strictement convexe** si l'inégalité est stricte pour $\mathbf{x} \neq \mathbf{y}$ et $t \in (0,1)$.

### Lien avec le hessien

C'est ici que tout se connecte. Pour une fonction deux fois différentiable :

- $f$ est **convexe** sur $C$ $\Longleftrightarrow$ $H_f(\mathbf{x}) \succeq 0$ pour tout $\mathbf{x} \in C$ (hessien semi-défini positif *partout*)
- $f$ est **strictement convexe** sur $C$ si $H_f(\mathbf{x}) \succ 0$ pour tout $\mathbf{x} \in C$ (hessien défini positif *partout*)

Attention à la différence avec le test du second ordre en §6 : là on regardait $H_f$ en *un seul point* pour classifier un point critique. Ici on regarde $H_f$ sur *tout le domaine* pour avoir une propriété globale.

*Exemple.* $f(x) = x^2$ : $f'' = 2 > 0$ partout → convexe. $f(x) = x^4 - 4x^2$ : $f'' = 12x^2 - 8$ qui est négatif près de l'origine → pas convexe.

### Lien avec le plan tangent

Pour une fonction convexe différentiable, le plan tangent en tout point est **en dessous** du graphe :

$f(\mathbf{x}) \geq f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0) \cdot (\mathbf{x} - \mathbf{x}_0) \quad \forall \mathbf{x}, \mathbf{x}_0$

C'est une autre façon de définir la convexité — et c'est exactement ce qu'on voit dans Taylor : le terme quadratique $\frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^T H_f (\mathbf{x}-\mathbf{x}_0) \geq 0$ quand $H_f \succeq 0$, donc $f(\mathbf{x}) \geq L_f(\mathbf{x})$.

### Pourquoi c'est crucial en optimisation

Pour une fonction convexe, les conditions d'optimalité sont simples et complètes :

$\nabla f(\mathbf{x}^*) = 0 \quad \Longleftrightarrow \quad \mathbf{x}^* \text{ est un minimum global}$

Il n'y a pas de point-selle, pas de minimum local qui ne soit pas global. La descente de gradient est garantie de converger vers le minimum. C'est pour ça que les fonctions de coût des réseaux de neurones sont souvent étudiées sous l'angle de leur « quasi-convexité », et que la régression logistique ou les SVM ont des propriétés de convergence garanties — leurs fonctions objectif sont convexes.

Pour une fonction **strictement convexe**, le minimum global est de plus **unique**.
