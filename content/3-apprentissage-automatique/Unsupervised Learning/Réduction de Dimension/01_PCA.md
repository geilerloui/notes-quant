## 0. Le point de départ — on a déjà tout ce qu'il faut

L'**ACP** (Analyse en Composantes Principales, ou *PCA* en anglais) est l'outil de référence pour **réduire la dimension** d'un jeu de données. On a $n$ observations en dimension $d$ — typiquement beaucoup de variables corrélées entre elles — et on cherche une représentation plus économique en $k \ll d$ dimensions qui préserve l'essentiel de l'information.

Le truc : **toute la machinerie nécessaire est déjà dans `[[01_Algèbre Linéaire]]`**. La PCA n'est pas une "méthode" qu'il faut découvrir, c'est une **application directe** de deux résultats qu'on a déjà vus :

1. **Théorème spectral** : toute matrice symétrique $\Sigma$ se diagonalise dans une base orthonormale, $\Sigma = V \Lambda V^\top$.
2. **SVD** : toute matrice rectangulaire $X$ se factorise $X = U S V^\top$.

L'ACP, c'est appliquer ces deux décompositions à un dataset. Vraiment. Toute la "théorie" se résume à : *on diagonalise la matrice de covariance, et on regarde où sont les directions propres*. Pas de Lagrangien magique, pas d'optimisation à reformuler trois fois — juste de l'algèbre linéaire appliquée.

> [!warning] Convention de notation
> Dans cette note j'utilise la **convention SVD standard** : $X = U S V^\top$, avec $V$ qui contient les vecteurs propres dans l'espace des **variables** ($\mathbb{R}^d$) et $U$ dans l'espace des **individus** ($\mathbb{R}^n$). Attention : beaucoup de cours français d'ACP utilisent la convention inverse ($U$ pour les variables, $V$ pour les individus). C'est juste un choix de lettres — la math est identique. Je m'aligne sur la convention internationale et sur `[[01_Algèbre Linéaire]]`.

Le plan de cette note :

- **§1–2** : la PCA "côté individus" — on regarde le nuage de $n$ points dans $\mathbb{R}^d$ et on diagonalise $\Sigma = X^\top X / n$ pour trouver ses axes propres. C'est l'angle dominant.
- **§3** : la PCA "côté variables" — on retourne le problème et on regarde les $d$ variables comme $d$ points dans $\mathbb{R}^n$. On diagonalise $X X^\top / n$ et on obtient une vision duale.
- **§4** : la SVD unifie tout — les deux PCA précédentes ne sont que les deux faces d'une même décomposition $X = USV^\top$.
- **§5–7** : pratique (combien de composantes garder, pièges, whitening).
- **§8** : extensions (PPCA, kernel PCA, etc.).

---

## 1. Setup et notations

### (i) Le dataset

On dispose de $n$ observations en dimension $d$, rangées dans une matrice

$$
X = \begin{pmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{n,1} & x_{n,2} & \cdots & x_{n,d}
\end{pmatrix} \in \mathbb{R}^{n \times d}
$$

Chaque **ligne** $x_i^\top \in \mathbb{R}^d$ est une **observation** (un individu). Chaque **colonne** $f_j \in \mathbb{R}^n$ est une **variable** (une feature). Cette dualité ligne/colonne est centrale en ACP — on l'exploitera en §3.

> [!example] Exemple fil rouge — météo de 6 villes
> Pour rendre les choses concrètes, voici un mini-dataset avec $n = 6$ villes françaises et $d = 3$ variables météo : pluie annuelle (mm), température maximale moyenne (°C), température minimale moyenne (°C).
> 
> | Ville | pluie (mm) | $t_{\max}$ (°C) | $t_{\min}$ (°C) |
> |---|---|---|---|
> | Lille       | 750  | 14.5 | 6.5  |
> | Paris       | 640  | 15.5 | 7.0  |
> | Strasbourg  | 670  | 14.8 | 5.5  |
> | Brest       | 1200 | 14.8 | 7.8  |
> | Toulouse    | 660  | 18.0 | 8.0  |
> | Nice        | 770  | 19.5 | 11.5 |
> 
> On voit à l'œil nu deux *axes de variation* : un axe "chaud/froid" (Nice vs Lille/Strasbourg) et un axe "pluvieux/sec" (Brest vs Paris). La PCA va retrouver ces axes automatiquement.

### (ii) Centrer

Avant tout, on **centre** les données — on soustrait la moyenne de chaque colonne :

$$
\bar{x}_j = \frac{1}{n} \sum_{i=1}^n x_{i,j}, \qquad \tilde{x}_{i,j} = x_{i,j} - \bar{x}_j
$$

Cela ramène le nuage de points au voisinage de l'origine. Sans centrage, la PCA "voit" la moyenne comme une direction de variance énorme et la ramène artificiellement dans le premier axe — ce qui n'a aucun intérêt.

Dans la suite, on suppose **toujours** que $X$ est centrée : $\mathbf{1}^\top X = \mathbf{0}^\top$ (chaque colonne a une moyenne nulle).

### (iii) Standardiser — selon le cas

**Question.** Les colonnes ont-elles des unités comparables ?

- *Non* : pluie en mm (valeurs de 600 à 1200), température en °C (valeurs de 5 à 20). Les variances sont incomparables — la variance de la pluie domine de plusieurs ordres de grandeur. Si on diagonalise dans cet état, la pluie "écrase" toutes les autres variables. **Il faut standardiser** : diviser chaque colonne par son écart-type $\sigma_j$.
- *Oui* : tous les pixels d'une image sont dans $[0, 255]$. Pas de standardisation, on garde l'échelle commune.

La standardisation s'écrit

$$
X_{cr} = \begin{pmatrix}
\frac{x_{1,1} - \bar{x}_1}{\sigma_1} & \cdots & \frac{x_{1,d} - \bar{x}_d}{\sigma_d} \\
\vdots & \ddots & \vdots \\
\frac{x_{n,1} - \bar{x}_1}{\sigma_1} & \cdots & \frac{x_{n,d} - \bar{x}_d}{\sigma_d}
\end{pmatrix}
$$

> [!warning] Centrer / standardiser — deux opérations distinctes
> **Centrer** = soustraire la moyenne (mettre le nuage à l'origine). **Toujours obligatoire** en PCA.
> **Standardiser** = centrer + diviser par l'écart-type (mettre toutes les variances à $1$). **Recommandé quand les unités sont incomparables.**
>
> Sur des données centrées, on travaille avec la **matrice de covariance**. Sur des données standardisées, on travaille avec la **matrice de corrélation** (qui est juste la covariance des données standardisées). Les deux PCA sont valables, elles ne donnent simplement pas les mêmes axes.

Pour le dataset météo, on **standardise** (pluie en mm vs température en °C). Dans la suite, sauf mention contraire, on travaille sur des données centrées-réduites.

### (iv) La matrice de covariance empirique

Le pivot fondamental : la **matrice de covariance empirique**

$$
\boxed{\;\Sigma = \frac{1}{n} X^\top X \in \mathbb{R}^{d \times d}\;}
$$

(Variante : $\frac{1}{n-1}$ pour l'estimateur non biaisé. Choix purement esthétique, ça ne change ni les axes ni la PCA.)

**Trois propriétés héritées de `[[01_Algèbre Linéaire]]`** :

1. **Symétrique** : $\Sigma^\top = \Sigma$. Trivial : $(X^\top X)^\top = X^\top X$.
2. **PSD** : pour tout $w \in \mathbb{R}^d$,
$$
w^\top \Sigma w = \frac{1}{n} w^\top X^\top X w = \frac{1}{n} \|Xw\|^2 \geq 0.
$$
C'est une norme au carré. Donc $\Sigma$ est PSD, ses valeurs propres sont $\geq 0$.
3. **Diagonalisable orthogonalement** : par le théorème spectral, il existe $V \in \mathbb{R}^{d \times d}$ orthogonale et $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ telles que
$$
\Sigma = V \Lambda V^\top, \qquad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0.
$$

C'est tout ce dont on a besoin pour faire la PCA.

> [!note] Lecture des entrées de $\Sigma$
> Sur des données **centrées**, $\Sigma_{j,k} = \frac{1}{n} \sum_i x_{i,j} x_{i,k}$ est la **covariance empirique** entre les variables $j$ et $k$. Sur des données **standardisées**, $\Sigma_{j,k}$ est directement la **corrélation** entre $j$ et $k$ (la diagonale vaut 1). Pour le dataset météo standardisé, on aurait par exemple $\Sigma_{t_{\max}, t_{\min}} \approx 0.85$ (forte corrélation positive), $\Sigma_{\text{pluie}, t_{\max}} \approx -0.20$ (faible corrélation négative).

---

## 2. PCA des individus — la voie naturelle

### (i) Le nuage de points et ses axes propres

Géométriquement, le dataset $X$ est un **nuage de $n$ points** dans $\mathbb{R}^d$. Centrer ramène le centroïde à l'origine ; standardiser le rend "rond" en termes d'unités.

![[fig01_nuage_donnees.png|400]]
*Figure 1. Nuage de points 2D centré. Visuellement, on voit immédiatement une direction d'étirement principale (l'axe rouge) et une direction transverse (l'axe vert). La PCA va retrouver ces deux axes par diagonalisation de $\Sigma$.*

**L'intuition** : ce nuage a une **forme**. Il s'étire dans certaines directions et est compact dans d'autres. On veut une **description efficace** de cette forme — un repère adapté qui aligne les axes de coordonnées sur les directions principales d'étirement.

Question naturelle : quelles sont ces directions ? On va voir qu'**elles sont exactement les vecteurs propres de $\Sigma$**.

### (ii) Pourquoi les vecteurs propres de $\Sigma$ ?

On part de la décomposition spectrale $\Sigma = V \Lambda V^\top$ (théorème spectral). Les colonnes de $V$ sont les vecteurs propres orthonormaux $v_1, \ldots, v_d$ et $\Lambda$ contient les valeurs propres ordonnées $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$.

**Variance dans une direction.** Soit $v \in \mathbb{R}^d$ un vecteur unitaire ($\|v\| = 1$). Si on projette tous les points du nuage sur la droite engendrée par $v$, on obtient $n$ scalaires $\langle x_i, v \rangle = x_i^\top v$. Leur variance empirique vaut (les données sont centrées, donc la moyenne est nulle) :

$$
\text{Var}(X v) = \frac{1}{n} \sum_{i=1}^n (x_i^\top v)^2 = \frac{1}{n} \|Xv\|^2 = \frac{1}{n} v^\top X^\top X v = v^\top \Sigma v.
$$

> [!warning] Identité-clé
> Pour tout vecteur unitaire $v$, la variance des données projetées sur $v$ vaut $v^\top \Sigma v$.

Maintenant, on a une question concrète : **quelle direction $v$ maximise cette variance ?** C'est le **quotient de Rayleigh** sur $\Sigma$ — exactement le problème vu en construction de la SVD dans `[[01_Algèbre Linéaire]]`.

**Réponse en deux lignes**. Décomposons $v$ dans la base propre : $v = \sum_k \alpha_k v_k$ avec $\sum_k \alpha_k^2 = 1$ puisque $v$ est unitaire et les $v_k$ orthonormaux. Alors

$$
v^\top \Sigma v = \sum_{k=1}^d \alpha_k^2 \lambda_k.
$$

C'est une **moyenne pondérée** des valeurs propres $\lambda_k$ avec poids $\alpha_k^2$ qui somment à 1. Donc cette moyenne pondérée est **maximisée** quand toute la masse est sur la plus grande valeur propre, c'est-à-dire $\alpha_1 = \pm 1$ et tous les autres nuls. Conclusion :

$$
\boxed{\;\max_{\|v\| = 1}\, v^\top \Sigma v = \lambda_1, \quad \text{atteint en } v = v_1\;}
$$

La direction de **variance maximale** est le **vecteur propre de plus grande valeur propre**. Ce qu'on appelle la **première composante principale**.

> [!tip] Pas de Lagrangien dans cette histoire
> On peut faire la même chose avec un Lagrangien — c'est la voie historique (Pearson 1901) — et on retombe sur $\Sigma v = \lambda v$. Mais c'est inutilement compliqué : le résultat tombe naturellement de la diagonalisation. **La diagonalisation EST la PCA.** Le Lagrangien est juste une façon plus compliquée de redécouvrir le théorème spectral.

**Et les composantes suivantes ?** Une fois $v_1$ trouvé, la **deuxième composante principale** est la direction de variance maximale **orthogonale à $v_1$**. Par le même raisonnement, c'est le vecteur propre $v_2$ avec valeur propre $\lambda_2$. Et ainsi de suite :

- $v_k$ = direction de variance maximale orthogonale à $v_1, \ldots, v_{k-1}$.
- Variance le long de $v_k$ : $\lambda_k$.

Les **$d$ vecteurs propres** $v_1, \ldots, v_d$ forment une **base orthonormale** de $\mathbb{R}^d$, classée par variance décroissante.

### (iii) Le changement de base $Y = X V$ — rotation, pas projection

Une fois les axes propres identifiés, on **change de base**. On forme la matrice $V = [v_1 | v_2 | \cdots | v_d] \in \mathbb{R}^{d \times d}$ (orthogonale) et on calcule

$$
\boxed{\;Y = X V \in \mathbb{R}^{n \times d}\;}
$$

C'est juste une **rotation** (et éventuellement réflexion) du nuage de points dans $\mathbb{R}^d$. Aucun point n'est perdu, aucune information n'est jetée — on **regarde simplement le même nuage depuis un autre repère**.

> [!warning] $Y = XV$ n'est pas une projection à bas rang
> C'est une rotation *de* $\mathbb{R}^d$ vers $\mathbb{R}^d$ — donc la matrice $Y$ est encore de taille $n \times d$. La réduction de dimension intervient **ensuite**, quand on choisit de ne garder que les $k$ premières colonnes de $Y$ (les composantes de plus grande variance). Ne pas confondre les deux étapes.

Les colonnes de $Y$ sont les **scores** ou **composantes principales** :

$$
Y = \big[\; \underbrace{X v_1}_{\text{PC}_1} \;\big|\; \underbrace{X v_2}_{\text{PC}_2} \;\big|\; \cdots \;\big|\; \underbrace{X v_d}_{\text{PC}_d} \;\big].
$$

La colonne $\text{PC}_k = X v_k$ donne, pour chaque individu, sa **coordonnée le long de l'axe propre $v_k$**.

![[fig02_rotation.png|550]]
*Figure 2. À gauche : le nuage dans le repère original $(e_1, e_2)$, avec les vecteurs propres $v_1, v_2$ en rouge (obliques). À droite : le même nuage après changement de base $Y = XV$ — les axes propres deviennent les axes de coordonnées. Les données sont décorrélées et les axes sont classés par variance décroissante.*

### (iii bis) Une composante principale est une combinaison linéaire des features

Il faut s'arrêter une seconde sur ce que représente concrètement $\text{PC}_k = Xv_k$. En écrivant le produit matrice-vecteur composante par composante, pour l'individu $i$ :

$
\text{PC}_k[i] \;=\; (Xv_k)[i] \;=\; \sum_{j=1}^d v_{k,j} \cdot x_{i,j} \;=\; v_{k,1} x_{i,1} + v_{k,2} x_{i,2} + \cdots + v_{k,d} x_{i,d}.
$

> [!important] Chaque composante principale est une somme pondérée des features originales
> $\text{PC}_k$ n'est **pas** une nouvelle mesure indépendante — c'est une **combinaison linéaire** des $d$ variables de départ, avec les coefficients $v_{k,1}, \ldots, v_{k,d}$ donnés par le $k$-ième vecteur propre. Ces coefficients s'appellent les **loadings**. La PCA ne fabrique aucune information nouvelle : elle **recombine** linéairement les colonnes existantes de $X$ pour trouver les directions qui concentrent le plus de variance.

**Lecture concrète sur l'exemple météo.** On a trouvé en (v) ci-dessous $v_1 \approx (-0.06,\ 0.69,\ 0.72)$ sur (pluie, $t_{\max}$, $t_{\min}$). Donc, explicitement :

$
\text{PC}_1 \;\approx\; -0.06 \cdot \text{pluie}_{cr} \;+\; 0.69 \cdot t_{\max,cr} \;+\; 0.72 \cdot t_{\min,cr}
$

(indices $cr$ = variables centrées-réduites). PC1 n'est rien d'autre qu'une **moyenne pondérée** des deux températures, quasiment indifférente à la pluie — c'est exactement ce qui permet de l'**étiqueter** "axe température" a posteriori. C'est toujours ainsi qu'on interprète une composante principale en pratique : on regarde quelles variables ont les **plus gros loadings** (en valeur absolue) et on leur donne un nom métier.

> [!note]- Pourquoi c'est obligatoirement linéaire
> La contrainte vient directement du problème d'optimisation posé en (ii) : on cherche $v$ qui maximise $v^\top \Sigma v = \text{Var}(Xv)$ parmi les vecteurs **unitaires de $\mathbb{R}^d$**. La quantité $Xv$ est par construction une combinaison linéaire des colonnes de $X$ — il n'y a pas d'autre forme possible à l'intérieur de ce cadre. C'est précisément cette restriction à des combinaisons **linéaires** qui limite la PCA face à des structures non-linéaires (cf. §6.iv) — et qui motive des extensions comme le Kernel PCA, qui relâche cette contrainte en travaillant dans un espace transformé non-linéairement.

> [!tip] Le pendant pour les loadings (vue duale)
> Symétriquement, en §3 on verra que les **variables** elles-mêmes se lisent comme des combinaisons linéaires des **individus** ($\widetilde Y_k = X^\top u_k$). C'est la même idée appliquée du côté dual : la SVD ($\S$4) montre que scores et loadings sont juste les deux faces du même objet $X = USV^\top$.

### (iv) Variance le long de chaque axe propre

Dans le nouveau repère, la matrice de covariance est **diagonale** :

$$
\Sigma_Y = \frac{1}{n} Y^\top Y = \frac{1}{n} V^\top X^\top X V = V^\top \Sigma V = V^\top (V \Lambda V^\top) V = \Lambda.
$$

Conclusion :

$$
\boxed{\;\Sigma_Y = \Lambda = \begin{pmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_d \end{pmatrix}\;}
$$

Trois conséquences importantes :

- **La variance le long de l'axe $k$** vaut exactement $\lambda_k$.
- **Les composantes sont décorrélées** : $\text{Cov}(\text{PC}_j, \text{PC}_k) = 0$ pour $j \neq k$. Les coefficients hors-diagonaux sont nuls.
- **La variance totale est conservée** :
$$
\text{Var totale} = \text{tr}(\Sigma) = \text{tr}(V \Lambda V^\top) = \text{tr}(\Lambda V^\top V) = \text{tr}(\Lambda) = \sum_{k=1}^d \lambda_k.
$$

> [!tip] Décomposition de la variance
> La variance totale du dataset $\text{tr}(\Sigma) = \sum_k \lambda_k$ se **répartit exactement** entre les composantes principales. Chaque $\lambda_k$ représente la part de variance "captée" par l'axe propre $v_k$. C'est ce qui justifie de garder les $k$ premières composantes : elles concentrent l'essentiel de la variance.

### (v) Exemple à la main — météo

Reprenons les 6 villes. Après centrage-réduction des 3 variables, on calcule la matrice de corrélation :

$$
\Sigma \approx \begin{pmatrix}
1.00 & -0.18 & 0.14 \\
-0.18 & 1.00 & 0.87 \\
0.14 & 0.87 & 1.00
\end{pmatrix} \quad \text{(pluie, } t_{\max}, t_{\min}\text{)}
$$

On voit immédiatement la structure : $t_{\max}$ et $t_{\min}$ sont **très corrélés** ($r = 0.87$), la pluie est **quasi-orthogonale** aux températures ($r \approx -0.18$ et $0.14$). On s'attend donc à ce que la PCA produise :
- Un premier axe "température" qui mélange $t_{\max}$ et $t_{\min}$ (corrélés)
- Un deuxième axe "pluie" quasi-perpendiculaire

**Diagonalisation** : on calcule les valeurs propres et vecteurs propres de $\Sigma$. Numériquement :

$$
\lambda_1 \approx 1.95, \qquad \lambda_2 \approx 1.00, \qquad \lambda_3 \approx 0.05
$$

et les vecteurs propres correspondants (lus en colonnes de $V$) :

$$
V \approx \begin{pmatrix}
-0.06 & 0.99 & 0.10 \\
0.69 & 0.11 & -0.71 \\
0.72 & -0.02 & 0.70
\end{pmatrix}
$$

**Lecture de $v_1$** : ses composantes sur (pluie, $t_{\max}$, $t_{\min}$) sont $(-0.06, 0.69, 0.72)$. Donc $v_1$ pondère **fortement les deux températures** ($0.69$ et $0.72$, du même signe) et **quasi-pas la pluie** ($-0.06$). C'est bien l'axe "température".

**Lecture de $v_2$** : $(0.99, 0.11, -0.02)$. Pondère **uniquement la pluie**. C'est l'axe "pluie".

**Lecture de $v_3$** : $(0.10, -0.71, 0.70)$. C'est l'**opposition** $t_{\max}$ vs $t_{\min}$ — une mesure de l'**amplitude thermique** journalière. La valeur propre associée est très petite ($\lambda_3 \approx 0.05$), ce qui veut dire que cet axe ne capture quasiment pas de variance — les amplitudes thermiques varient peu entre les 6 villes.

> [!example] Lecture des composantes
> Les coordonnées de chaque ville sur les composantes principales s'obtiennent par $Y = X_{cr} V$. Par exemple Nice (chaude, modérément pluvieuse) aura une grande coordonnée positive sur PC1 (température), une coordonnée moyenne sur PC2 (pluie). Brest (modérément froide, très pluvieuse) aura une coordonnée légèrement négative sur PC1, grande positive sur PC2. Lille (froide, pluvieuse modérée) : négatif sur PC1, neutre sur PC2.

**Variance expliquée** : $\lambda_1 / 3 \approx 65\%$, $\lambda_2 / 3 \approx 33\%$, $\lambda_3 / 3 \approx 2\%$. Les deux premières composantes captent **98% de la variance**. On peut donc **représenter le dataset 3D en 2D quasi sans perte d'information** — c'est la promesse centrale de la PCA.

---

## 3. PCA des variables — l'éclairage dual

Jusqu'ici on a traité $X$ comme un nuage de **$n$ individus dans $\mathbb{R}^d$**. Mais $X$ peut aussi être vu comme un ensemble de **$d$ variables dans $\mathbb{R}^n$** — chaque colonne $f_j \in \mathbb{R}^n$ est un point dans l'espace des observations. C'est l'angle dual.

### (i) Pourquoi un autre point de vue ?

L'analyse des **individus** répond à : *quelles observations se ressemblent ?* On regarde la géométrie du nuage de $n$ points dans $\mathbb{R}^d$ et on trouve ses axes principaux.

L'analyse des **variables** répond à : *quelles features se comportent de façon similaire ?* On regarde la géométrie des $d$ vecteurs-features dans $\mathbb{R}^n$ et on identifie les groupes de variables corrélées.

Les deux analyses sont **complémentaires** — on verra plus bas qu'elles sont en fait **deux faces de la même SVD**.

### (ii) Les $d$ variables comme $d$ points dans $\mathbb{R}^n$

Sur des données **centrées-réduites**, chaque colonne $f_j \in \mathbb{R}^n$ est un vecteur de moyenne nulle et de variance unitaire — donc de norme $\|f_j\| = \sqrt{n}$. Les $d$ variables vivent toutes sur une **sphère de rayon $\sqrt{n}$** dans $\mathbb{R}^n$.

**Géométrie du produit scalaire entre deux variables**. Pour deux features $f_j$ et $f_k$ centrées-réduites :

$$
\langle f_j, f_k \rangle = \sum_{i=1}^n f_{i,j}\, f_{i,k} = n \cdot \text{corr}(f_j, f_k).
$$

Le **produit scalaire** entre deux vecteurs-variables est proportionnel à leur **corrélation**. Géométriquement,

$$
\cos(\theta_{jk}) = \frac{\langle f_j, f_k \rangle}{\|f_j\| \|f_k\|} = \frac{n \cdot \text{corr}(f_j, f_k)}{n} = \text{corr}(f_j, f_k).
$$

> [!warning] L'angle entre deux variables = leur corrélation
> Sur des données centrées-réduites, **l'angle entre deux vecteurs-features dans $\mathbb{R}^n$ a pour cosinus leur coefficient de corrélation**. Deux variables très corrélées ($r \approx 1$) ont un angle proche de $0$ (elles "pointent dans la même direction"). Deux variables décorrélées ($r \approx 0$) sont **orthogonales**. Deux variables anti-corrélées ($r \approx -1$) sont **opposées** (angle de $180°$).

C'est cette propriété qui rend l'analyse des variables géométriquement intéressante : la **proximité spatiale dans $\mathbb{R}^n$ traduit la corrélation statistique**.

### (iii) Diagonalisation de $X X^\top$

Pour trouver les "axes principaux" du nuage des $d$ variables dans $\mathbb{R}^n$, on applique le même raisonnement qu'en §2, mais sur l'autre matrice :

$$
\boxed{\;G = \frac{1}{n} X X^\top \in \mathbb{R}^{n \times n}\;}
$$

(Cette matrice porte parfois le nom de **matrice de Gram** ou de **noyau linéaire**.)

Comme $\Sigma = X^\top X / n$, $G$ est symétrique PSD et se diagonalise :

$$
G = U M U^\top, \qquad M = \text{diag}(\mu_1, \ldots, \mu_n), \qquad \mu_1 \geq \cdots \geq \mu_n \geq 0.
$$

Les colonnes de $U \in \mathbb{R}^{n \times n}$ sont les **vecteurs propres dans l'espace des individus** $\mathbb{R}^n$. Les coordonnées des variables sur ces axes s'obtiennent par

$$
\widetilde{Y} = X^\top U \in \mathbb{R}^{d \times n}.
$$

Chaque colonne $\widetilde{Y}_k = X^\top u_k$ donne, pour chaque variable, sa coordonnée sur le $k$-ième axe principal de $\mathbb{R}^n$.

### (iv) Le cercle des corrélations

Le diagnostic visuel principal pour l'analyse des variables est le **cercle des corrélations** — un graphique 2D où on représente chaque variable par un point, dont les coordonnées sont ses corrélations avec PC1 et PC2 (les deux premières composantes principales) :

$$
\text{Coord}(f_j) = \big(\text{corr}(f_j, \text{PC}_1),\; \text{corr}(f_j, \text{PC}_2)\big).
$$

Comme les corrélations sont entre $-1$ et $1$, tous les points tombent dans le **disque unité**. Plus une variable est proche du bord du cercle (norme $\approx 1$), mieux elle est représentée par les deux premières composantes.

![[fig03_cercle_correlations.png|400]]
*Figure 3. Cercle des corrélations pour le dataset météo. Les deux températures $t_{\max}$ et $t_{\min}$ sont fortement corrélées entre elles (vecteurs quasi-superposés) et alignées sur l'axe PC1. La pluie est presque orthogonale aux températures et alignée sur l'axe PC2. Toutes les variables sont proches du bord du cercle ⇒ bien représentées par le plan factoriel (PC1, PC2).*

**Lecture du cercle** :
- Variables proches l'une de l'autre ⇒ **corrélées positivement**.
- Variables diamétralement opposées ⇒ **anti-corrélées**.
- Variables orthogonales ⇒ **décorrélées**.
- Variable proche du bord ⇒ **bien représentée** par le plan factoriel.
- Variable proche du centre ⇒ **mal représentée**, son information est portée par des composantes d'ordre supérieur.

### (v) Le lien individus ↔ variables

Plutôt que de redériver tout, on remarque qu'**il existe une relation directe** entre les deux problèmes. Soit $u_k$ un vecteur propre de $G = X X^\top / n$ :

$$
\frac{1}{n} X X^\top u_k = \mu_k\, u_k.
$$

Multiplions à gauche par $X^\top$ :

$$
\frac{1}{n} X^\top X \, (X^\top u_k) = \mu_k\, (X^\top u_k) \quad \Longleftrightarrow \quad \Sigma\, (X^\top u_k) = \mu_k\, (X^\top u_k).
$$

Donc $X^\top u_k$ est un **vecteur propre de $\Sigma$** avec valeur propre $\mu_k$. C'est-à-dire :

$$
\boxed{\;X^\top u_k \;\propto\; v_k \qquad \text{et} \qquad \mu_k = \lambda_k\;}
$$

(à un facteur de normalisation près, qu'on précisera en §4 via la SVD). Pour normaliser proprement, on a en fait $X^\top u_k = \sqrt{n \lambda_k}\, v_k$, ce qui donne $\|X^\top u_k\| = \sqrt{n \lambda_k}$ — la composante variable $\widetilde{Y}_k$ a norme $\sqrt{n \lambda_k}$.

Réciproquement, par la même manip dans l'autre sens, $X v_k = \sqrt{n \lambda_k}\, u_k$. Ce qui donne $\text{PC}_k = X v_k$ de norme $\sqrt{n \lambda_k}$, cohérent avec $\text{Var}(\text{PC}_k) = \lambda_k$.

> [!important] Le résultat-clé
> Les **valeurs propres de $\Sigma = X^\top X / n$ et de $G = X X^\top / n$ sont identiques** (modulo des zéros si $n \neq d$). Les vecteurs propres sont reliés par $X v_k = \sqrt{n \lambda_k}\, u_k$ et $X^\top u_k = \sqrt{n \lambda_k}\, v_k$.
>
> **Conséquence concrète** : on ne fait qu'**une seule diagonalisation** (la plus économique : $\Sigma$ si $d < n$, $G$ si $n < d$), et on obtient l'autre par la formule de liaison. Dans le contexte image $n = 60000$, $d = 784$, on diagonalise $\Sigma$ (784×784) plutôt que $G$ (60000×60000) — gain énorme.

---

## 4. SVD unifie tout

### (i) $X = U S V^\top$

La diagonalisation de $\Sigma$ et celle de $G$ ne sont pas deux problèmes distincts : ce sont **deux facettes d'une même SVD** de $X$. C'est exactement le lien vu dans `[[01_Algèbre Linéaire]]` :

$$
\boxed{\;X = U S V^\top\;}
$$

avec $U \in \mathbb{R}^{n \times n}$ orthogonale (vecteurs singuliers gauches), $V \in \mathbb{R}^{d \times d}$ orthogonale (vecteurs singuliers droits), et $S \in \mathbb{R}^{n \times d}$ "diagonale rectangulaire" avec $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ sur la diagonale.

**Liens avec ce qui précède** :

$$
\frac{1}{n} X^\top X = \frac{1}{n} V S^\top S V^\top = V \cdot \frac{S^\top S}{n} \cdot V^\top \qquad \Rightarrow \qquad \lambda_k = \frac{\sigma_k^2}{n}.
$$

De même $G = U \frac{S S^\top}{n} U^\top$ donne $\mu_k = \sigma_k^2/n = \lambda_k$.

**À retenir** :

$$
\boxed{\;\sigma_k = \sqrt{n \lambda_k} \qquad \text{(valeur singulière de } X \text{ = } \sqrt{n} \times \text{écart-type le long de PC}_k)\;}
$$

### (ii) Scores et loadings

Les deux objets clés de la PCA s'expriment **directement** depuis la SVD, sans rien recalculer :

- **Scores** (composantes des individus) :
$$
\text{PC} = X V = U S V^\top V = U S \in \mathbb{R}^{n \times d}.
$$
La $k$-ième colonne $\text{PC}_k = \sigma_k u_k$ donne les coordonnées des $n$ individus le long de l'axe $v_k$.

- **Loadings** (composantes des variables) :
$$
\widetilde{Y} = X^\top U = V S^\top U^\top U = V S^\top \in \mathbb{R}^{d \times n}.
$$
La $k$-ième colonne $\widetilde{Y}_k = \sigma_k v_k$ donne les coordonnées des $d$ variables le long de l'axe $u_k$.

> [!tip] Symétrie scores/loadings
> Scores = $US$ (matrice $n \times d$, une ligne par individu). Loadings = $VS$ (matrice $d \times n$, une ligne par variable). Les deux PCA (individus et variables) ne sont littéralement que **les deux côtés** de $X = USV^\top$ — gauche pour les individus, droit pour les variables. **La SVD unifie tout.**

### (iii) PCA tronquée = Eckart-Young

La PCA "réduit la dimension" en gardant seulement les $k$ premières composantes principales. Concrètement, on tronque la SVD :

$$
X \;\approx\; X_k = \sum_{i=1}^k \sigma_i\, u_i v_i^\top = U_k S_k V_k^\top,
$$

où $U_k$, $S_k$, $V_k$ ne contiennent que les $k$ premières colonnes/diagonales. **Le théorème d'Eckart-Young** (vu dans `[[01_Algèbre Linéaire]]`) garantit que $X_k$ est la **meilleure approximation de rang $k$** de $X$ au sens Frobenius :

$$
X_k = \arg\min_{\text{rang}(Y) \leq k}\, \|X - Y\|_F^2, \qquad \|X - X_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2.
$$

> [!important] Le sens profond de la PCA
> Garder les $k$ premières composantes principales n'est pas un choix arbitraire — c'est **mathématiquement la meilleure compression linéaire à $k$ dimensions** au sens de l'erreur quadratique. Eckart-Young garantit qu'on ne peut pas faire mieux avec une transformation linéaire de rang $\leq k$.
>
> C'est pour ça que la PCA est partout : compression d'images, débruitage, systèmes de recommandation à bas rang, factor models — tout repose sur la propriété d'approximation optimale d'Eckart-Young.

### (iv) Algorithme — la recette concrète

En pratique, voici ce qu'on fait :

1. **Centrer** (et standardiser si les unités sont incomparables) : $X \leftarrow \tilde{X}$ ou $X_{cr}$.
2. **SVD de $X$** : $X = U S V^\top$ via `numpy.linalg.svd(X, full_matrices=False)`.
3. **Scores** : $\text{PC} = U S$ (chaque ligne = un individu, chaque colonne = une composante).
4. **Variance expliquée** : $\lambda_k = \sigma_k^2 / n$, proportion expliquée par PC$_k$ : $\sigma_k^2 / \sum_j \sigma_j^2$.
5. **Choisir $k$** : voir §5.
6. **Réduire** : ne garder que les $k$ premières colonnes de $\text{PC}$ ⇒ $\text{PC}_{1:k} \in \mathbb{R}^{n \times k}$.

> [!warning] SVD directement sur $X$, pas via $\Sigma$
> En théorie, diagonaliser $\Sigma = X^\top X / n$ et faire la SVD de $X$ donnent le même résultat. **En arithmétique flottante, c'est faux** — former $X^\top X$ **carre le conditionnement** et dégrade la précision des petites valeurs propres (cf. `[[01_Algèbre Linéaire]]`, figure SVD-4). Utiliser toujours `numpy.linalg.svd(X, full_matrices=False)`, pas `numpy.linalg.eigh(X.T @ X)`.

---

## 5. Combien de composantes garder ?

Aucune réponse universelle — ça dépend du contexte. Mais trois outils standards.

### (i) Variance expliquée cumulée

Pour chaque $k$, on calcule

$$
\text{var\_expliquée}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2}.
$$

C'est une fonction croissante de $k$, allant de $0$ à $1$.

### (ii) Règle des 80% (ou 90%, ou 95%)

Le seuil le plus commun : on choisit $k$ tel que la variance expliquée cumulée atteigne **80%** (ou 90%, ou 95% — dépend de la tolérance à la perte d'information). Pour le dataset météo, $\lambda_1 / 3 + \lambda_2 / 3 \approx 65\% + 33\% = 98\%$ ⇒ **$k = 2$ suffit** largement.

### (iii) Scree plot et règle du coude

Tracer les valeurs propres $\lambda_k$ en fonction de $k$. Souvent on observe une **chute brutale** entre quelques grandes valeurs propres et le reste — c'est le **"coude"** du graphique. On garde les composantes **avant** le coude.

![[fig04_scree_plot.png|450]]
*Figure 4. Scree plot pour un dataset à 10 dimensions. Les 3 premières valeurs propres sont nettement plus grandes que les 7 suivantes — coude marqué à $k = 3$. Choix raisonnable : garder $k = 3$ composantes.*

### (iv) Limites de ces règles

- **Aucune des règles n'est rigoureuse**. Le coude peut être ambigu, le seuil 80% est arbitraire.
- **En présence de bruit**, les valeurs propres du bruit ressemblent à un plateau — pas toujours évident de distinguer "signal" de "bruit statistique". C'est précisément la question que résout la théorie des matrices aléatoires (notes RMT).
- **En supervisé**, ce qui compte n'est pas la variance expliquée mais la **performance sur la tâche** ⇒ on cross-valide $k$.

---

## 6. Pièges et bonnes pratiques

### (i) Centrer est obligatoire

Sans centrage, le premier vecteur propre pointe vers le **centroïde** du nuage — il "voit" la moyenne comme une direction de variance immense et la met dans PC1, ce qui n'a aucun intérêt analytique.

### (ii) Standardiser dépend des unités

Voir §1.iii. À retenir : si les variables ont des unités différentes (mm vs °C vs %), **standardiser**. Si les variables sont dans la même unité (intensités de pixels, log-rendements), **ne pas standardiser** — la variance brute porte de l'information.

### (iii) PCA est très sensible aux outliers

La covariance empirique est dominée par les **carrés** des écarts à la moyenne. Un seul outlier extrême peut **complètement biaiser** les axes principaux — il "tire" PC1 dans sa direction. Solutions :

- Détecter et retirer les outliers avant PCA.
- Utiliser une PCA **robuste** (Robust PCA, basée sur des estimateurs M ou sur la décomposition en composantes basse-rang + sparse).

### (iv) Linéarité — PCA ne capture que les structures linéaires

Si les données vivent sur un **manifold non-linéaire** (genre une spirale ou une variété courbée), la PCA **ne le voit pas** — elle projette linéairement et écrase les structures non-linéaires. Pour ce genre de cas, on passe à :

- **Kernel PCA** : appliquer PCA sur une feature map non-linéaire de $X$, via l'astuce du noyau (kernel trick).
- **t-SNE** ou **UMAP** : méthodes non-linéaires basées sur des préservations locales de voisinage.
- **Autoencodeurs** : version "deep" — les encodeur/décodeur peuvent apprendre des manifolds courbes.

![[fig05_pca_vs_nonlineaire.png|500]]
*Figure 5. Limite de la PCA. À gauche : un swiss roll en 3D. La PCA (au centre) écrase brutalement la spirale en projetant linéairement, perdant la structure de manifold. UMAP (à droite) préserve la connectivité locale et déroule proprement la spirale en 2D.*

### (v) Le piège du signe

Les vecteurs propres sont définis **à un signe près** : si $v$ est un vecteur propre, alors $-v$ l'est aussi avec la même valeur propre. Différentes implémentations (numpy, scikit-learn, R…) peuvent renvoyer des signes différents pour le **même vecteur propre**, ce qui change le **signe des scores** mais **pas leur interprétation géométrique**.

> [!warning] Conséquence pratique
> Si tu refais une PCA après mise à jour de scikit-learn et que les signes de tes composantes ont inversé, **ce n'est pas un bug** — c'est juste le choix d'algorithme de SVD qui a changé. La variance expliquée et les distances entre individus sont identiques.

### (vi) PCA ≠ feature selection

PCA crée de **nouvelles variables** (combinaisons linéaires des originales) — elle ne **sélectionne pas** les variables existantes. Si l'interprétabilité métier est critique (par exemple : *"je veux savoir lesquels de mes 100 indicateurs sont importants"*), **PCA n'est pas l'outil**. Utiliser plutôt LASSO, recursive feature elimination, ou des méthodes d'attribution.

---

## 7. Whitening — pour aller un cran plus loin

Une fois la PCA effectuée, on peut pousser la transformation un cran plus loin : le **whitening** (ou "blanchiment").

### (i) Pourquoi blanchir ?

Après PCA, les composantes principales sont **décorrélées** mais elles ont des **variances différentes** ($\lambda_1, \lambda_2, \ldots$). Pour certaines applications (réseaux de neurones avant la mode du batch norm, ICA, modèles génératifs basés sur du bruit gaussien isotrope), on veut des features non seulement **décorrélées**, mais aussi de **variance unitaire** dans toutes les directions.

### (ii) Comment

Après PCA, les scores $\text{PC} = X V$ ont covariance $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$. Pour rendre toutes les variances égales à $1$, on **rescale** chaque composante par $1/\sqrt{\lambda_k}$ :

$$
\boxed{\;Z = \Lambda^{-1/2} V^\top (X - \bar{X})\;}
$$

Vérification :

$$
\frac{1}{n} Z^\top Z = \Lambda^{-1/2} V^\top \Sigma V \Lambda^{-1/2} = \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I.
$$

La covariance des données blanchies est l'**identité** — toutes les directions sont équivalentes statistiquement.

![[fig06_whitening.png|550]]
*Figure 6. Trois nuages 2D : à gauche les données brutes (axes obliques, variances inégales). Au milieu : après PCA (axes alignés, variances toujours inégales). À droite : après whitening (nuage isotrope, toutes les directions équivalentes).*

### (iii) Quand l'utiliser ?

- **Préprocessing** avant des algorithmes sensibles à l'échelle (vieux réseaux de neurones sans batch norm).
- **ICA** (Independent Component Analysis) : étape de pré-traitement standard.
- **Modèles génératifs** où on veut un latent espace gaussien isotrope.
- **Pas** quand on veut conserver l'information de variance (PC1 vs PC2 — qui est plus important ?). Le whitening **détruit** cette information.

---

## 8. Pour aller plus loin

Quelques directions naturelles à explorer après cette note :

- **PPCA** — la version probabiliste de la PCA, qui ajoute un modèle génératif latent gaussien. Permet le maximum de vraisemblance, l'imputation de valeurs manquantes, et fait le pont vers les VAE. Voir `[[PPCA]]`.
- **Kernel PCA** — PCA appliquée dans un espace de features non-linéaire via l'astuce du noyau. Permet de capturer des structures non-linéaires (cf. §6.iv).
- **Sparse PCA** — variantes où les composantes principales sont contraintes à n'avoir que quelques entrées non-nulles, pour l'interprétabilité. Lien avec LASSO.
- **Robust PCA** — décomposition $X = L + S$ avec $L$ basse-rang et $S$ sparse. Robuste aux outliers et corruptions, beaucoup utilisée en computer vision (séparation background/foreground).
- **Autoencodeurs linéaires** — un autoencoder linéaire (sans non-linéarité, loss MSE) entraîné optimalement **converge vers la PCA**. Avec des non-linéarités, on obtient une généralisation non-linéaire.
- **Factor models** — en finance et économétrie, la PCA est utilisée pour extraire des **facteurs de risque** orthogonaux à partir de matrices de covariance de rendements. Voir les notes sur les modèles de facteurs.

Tous ces objets s'appuient sur la même machinerie de base : décomposition spectrale, SVD, théorème d'Eckart-Young. Si ces fondations sont solides (cf. `[[01_Algèbre Linéaire]]`), comprendre ces extensions devient un jeu d'enfant.
