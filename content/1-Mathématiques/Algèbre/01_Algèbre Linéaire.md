
## 0. Rappels — spectre et invariance orthogonale

Avant d'attaquer, deux objets d'algèbre linéaire qu'on va utiliser sans arrêt.

### (i) Spectre d'une matrice

Le **spectre** d'une matrice carrée $M$, c'est juste l'**ensemble de ses valeurs propres**. Rien de plus.

Rappel : $\lambda \in \mathbb{R}$ est valeur propre de $M$ s'il existe un vecteur non nul $v$ tel que $Mv = \lambda v$. Géométriquement : il existe une direction $v$ dans laquelle $M$ agit comme une simple multiplication par le scalaire $\lambda$ — pas de rotation, juste un étirement (ou compression) d'un facteur $\lambda$. Une matrice $N \times N$ a $N$ valeurs propres (comptées avec multiplicité).

Quand on parlera de "loi du spectre" en RMT, on parlera de **l'histogramme de ces $N$ valeurs propres**. On tire une matrice aléatoire $N \times N$, on calcule ses $N$ valeurs propres, on les met dans un histogramme. La question centrale : à quoi ressemble cet histogramme quand $N$ devient grand ?

> [!example] Exemple à la main — spectre d'une matrice 2×2
> Soit
> $$M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}$$
> Cette matrice est symétrique, avec deux valeurs propres bien contrastées :
> $$\lambda_1 = 3.0, \quad \lambda_2 = 0.4$$
> associées aux vecteurs propres
> $$v_1 = \begin{pmatrix} 0.89 \\ 0.45 \end{pmatrix}, \quad v_2 = \begin{pmatrix} -0.45 \\ 0.89 \end{pmatrix}$$
> Ces deux vecteurs sont orthogonaux ($v_1^\top v_2 = 0$) et de norme 1 — ils forment une **base orthonormale** de $\mathbb{R}^2$.
>
> Cela signifie que n'importe quel vecteur $w \in \mathbb{R}^2$ s'écrit comme combinaison linéaire de $v_1$ et $v_2$ :
> $$w = \alpha_1 v_1 + \alpha_2 v_2, \quad \text{avec} \quad \alpha_i = v_i^\top w$$
> Par exemple, pour $w = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ : $\alpha_1 = 0.89$, $\alpha_2 = -0.45$. Le vecteur $w$ se décompose en $0.89$ parts de direction $v_1$ et $-0.45$ parts de direction $v_2$. Ce sont exactement les coordonnées de $w$ dans le repère des vecteurs propres — c'est-à-dire $Q^\top w$ avec $Q = [v_1 | v_2]$.

> [!warning] Théorème spectral
> Toute matrice **symétrique réelle** $M \in \mathbb{R}^{N \times N}$ admet $N$ valeurs propres réelles et $N$ vecteurs propres **orthogonaux entre eux**. On peut donc écrire
> $$M = Q \Lambda Q^\top$$
> avec $Q = [v_1 | \cdots | v_N]$ orthogonale ($Q^\top Q = I$) et $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$. Les vecteurs propres forment une base orthonormale de $\mathbb{R}^N$ — tout vecteur $w$ se décompose uniquement comme $w = \sum_i (v_i^\top w)\, v_i$.

![[images/1-Mathématiques/Random Matrix Theory/figA_vecteurs_propres.png]]
*Figure A. Chaque vecteur bleu (sur le cercle unité) est transformé en un vecteur rouge par $M$ — en général il change de direction. Seuls les deux vecteurs propres (vert foncé) restent sur leur droite : $\lambda_1 = 3.0$ étire, $\lambda_2 = 0.4$ compresse.*

### (ii) Invariance orthogonale, sans algèbre

Une matrice orthogonale $O$ représente une **rotation** de l'espace (éventuellement combinée à un miroir). C'est un changement de repère qui préserve les longueurs et les angles. L'opération $M \mapsto O M O^\top$ revient à **regarder $M$ dans un autre repère** — c'est la même matrice, exprimée dans d'autres coordonnées.

> [!warning] L'invariance orthogonale en une phrase
> Dire que "la loi de $M$ est invariante par $M \mapsto OMO^\top$" signifie : **la distribution de $M$ est la même dans tous les repères**. Aucune direction de l'espace n'est privilégiée.

Conséquence concrète qu'on utilisera : les **vecteurs propres** d'une telle matrice pointent dans des directions complètement uniformes sur la sphère. La matrice est "isotrope" — elle ne sait rien des axes de coordonnées.

> [!example] Exemple à la main — rotation préserve le spectre
> Reprenons la matrice $M = \begin{pmatrix} 2.48 & 1.04 \\ 1.04 & 0.92 \end{pmatrix}$. Appliquons une rotation d'angle $\theta = \pi/4$ : $M' = O M O^\top$ avec $O = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$. Les entrées de $M'$ sont différentes — par exemple $M'_{11} \approx 1.78$ au lieu de $2.48$.
>
> Mais ses **valeurs propres sont exactement les mêmes** : $\{0.4, 3.0\}$. Le spectre est invariant par changement de repère. C'est une propriété générale, pas un hasard.
>
> Pour une matrice GOE, ce n'est pas seulement le spectre mais aussi **toute la distribution** de la matrice qui est invariante par $M \mapsto OMO^\top$ : si on rejoue le tirage de $M$ après rotation, on retombe sur la même loi.

![[images/1-Mathématiques/Random Matrix Theory/figB_ellipse.png|289]]
*Figure B. Le cercle unité est transformé en ellipse par $M$. Les axes de l'ellipse coïncident exactement avec les vecteurs propres, et leur longueur est proportionnelle à la valeur propre correspondante : la valeur propre mesure l'étirement dans la direction propre.*

### (iii) Diagonalisation et changement de base

Pour une matrice **symétrique** $\Sigma$, les vecteurs propres sont toujours orthogonaux entre eux — c'est le théorème spectral. On peut donc former la matrice $P = [v_1 | v_2 | \cdots | v_N]$ dont les colonnes sont les vecteurs propres normalisés : $P$ est une matrice **orthogonale** ($P^{-1} = P^\top$). On obtient alors la décomposition

$$\Sigma = P \Lambda P^\top$$

avec $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$ la matrice diagonale des valeurs propres. C'est la **diagonalisation** de $\Sigma$.

La décomposition $\Sigma = P \Lambda P^\top$ est une tautologie algébrique — on n'a "rien fait", c'est la même matrice écrite autrement. L'intérêt est qu'elle **sépare deux types d'information** mélangés dans $\Sigma$ : $P$ dit *où* (les directions), $\Lambda$ dit *combien* (les amplitudes). Cette séparation rend trois opérations triviales qui seraient autrement coûteuses :

$$\Sigma^{-1} = P \Lambda^{-1} P^\top \quad \text{(inverser = prendre } 1/\lambda_i \text{)}, \qquad \Sigma^{1/2} = P \Lambda^{1/2} P^\top, \qquad \Sigma \approx \sum_{i=1}^k \lambda_i v_i v_i^\top \text{ (troncature rang } k\text{)}$$

Mais l'usage principal est ailleurs. La vraie séquence en pratique :

1. Estimer $\Sigma$ sur les données
2. Décomposer pour **obtenir $P$** — c'est l'objectif, pas un résultat intermédiaire
3. Calculer $P^\top X$ — **pivoter tous les points de données** dans le repère propre
4. Dans ce repère, les colonnes de $P^\top X$ sont décorrélées et classées par variance décroissante $\lambda_1 \geq \lambda_2 \geq \dots$

$P^\top x$ c'est une **rotation pure** ($P$ est orthogonale, elle préserve distances et angles). Les points ne bougent pas dans l'absolu — on les regarde depuis un nouveau repère, celui où les axes correspondent aux directions de variance maximale.

> [!warning] Pourquoi c'est central en RMT
> Les matrices de covariance $\Sigma$ sont toujours symétriques. Leur diagonalisation $\Sigma = P\Lambda P^\top$ décompose le risque d'un portefeuille en directions indépendantes — les vecteurs propres sont les **facteurs de risque**, les valeurs propres leur **variance**. La question de RMT est précisément : quand $\Sigma$ est estimée sur des données finies, quels $\lambda_i$ et $v_i$ sont du signal, et lesquels sont du bruit statistique ?

![[images/1-Mathématiques/Random Matrix Theory/figC_diagonalisation.png]]
*Figure C. À gauche : l'ellipse de $\Sigma$ dans le repère original — les axes propres sont obliques. À droite : après le changement de base $P^\top x$, l'ellipse est alignée sur les axes de coordonnées. Les directions sont décorrélées, chaque axe porte une variance $\lambda_i$.*

![[images/1-Mathématiques/Random Matrix Theory/figD_pca_nuage.png]]
*Figure D. À gauche : nuage de données corrélées dans le repère original, avec les vecteurs propres en rouge. À droite : le même nuage après projection $P^\top x$ — les données sont décorrélées et les axes de variance maximale sont alignés sur les axes de coordonnées. C'est exactement ce que fait la PCA.*

### (iv) Matrices PSD et matrices de projection

#### Définie positive, semi-définie positive — les 4 catégories

Une matrice symétrique $A$ est caractérisée par le signe de la **forme quadratique** $x^\top A x$. Ce scalaire mesure si $A x$ pointe dans la même direction que $x$ (produit scalaire positif) ou en sens opposé.

> [!warning] Les 4 catégories
> Soit $A$ une matrice symétrique réelle. Pour tout vecteur non nul $x$ :
>
> **Définie positive (PD)** : $x^\top A x > 0$ — toutes les valeurs propres $> 0$. $Ax$ pointe toujours dans la même demi-sphère que $x$.
>
> **Semi-définie positive (PSD)** : $x^\top A x \geq 0$ — valeurs propres $\geq 0$, au moins une nulle. Il existe des directions dans lesquelles $A$ écrase tout à zéro.
>
> **Définie négative (ND)** : $x^\top A x < 0$ — toutes les valeurs propres $< 0$. $Ax$ pointe toujours en sens opposé à $x$.
>
> **Indéfinie** : le signe de $x^\top Ax$ dépend de $x$ — des valeurs propres positives et négatives coexistent.

![[images/1-Mathématiques/Random Matrix Theory/figPSD1_transformation.png]]
*Figure PSD-1. Les 4 catégories vues comme transformations : en PD, $Ax$ reste dans la même demi-sphère que $x$ (angle aigu). En PSD, certaines directions sont écrasées. En ND, $Ax$ retourne systématiquement $x$. En indéfinie, le signe dépend de la direction.*

La forme quadratique $f(x) = x^\top A x$ peut être tracée en 3D en faisant varier $x$ — elle révèle immédiatement la catégorie. La connexion avec la vision "transformation" : $x^\top (Ax)$ est le produit scalaire entre $x$ et son image $Ax$, donc son signe dit si l'angle entre les deux est aigu ou obtus.

![[images/1-Mathématiques/Random Matrix Theory/figPSD2_quadratique.png]]
*Figure PSD-2. La forme quadratique $f(x) = x^\top A x$ tracée en 3D pour les 4 catégories. PD → bol convexe avec un unique minimum global. PSD → gouttière, minimum sur toute une droite (direction nulle). ND → bol concave, unique maximum. Indéfinie → point de selle, ni minimum ni maximum.*

#### Pourquoi les matrices de covariance sont toujours PSD

Une matrice de covariance empirique s'écrit $\Sigma = \frac{1}{T} X X^\top$. Pour tout vecteur $w$ :

$$w^\top \Sigma w = \frac{1}{T} w^\top X X^\top w = \frac{1}{T} \|X^\top w\|^2 \geq 0$$

C'est une norme au carré — toujours positive ou nulle. Donc $\Sigma$ est automatiquement PSD, sans hypothèse sur les données. Si $X$ est plein rang ($T > N$), alors $\Sigma$ est même PD et inversible. Si $T < N$ (moins d'observations que de dimensions), $\Sigma$ est singulière — certaines valeurs propres sont nulles et $\Sigma^{-1}$ n'existe pas. C'est un problème concret en finance quand on a 200 actifs et 150 jours de données.

#### Matrice de projection

La décomposition spectrale $\Sigma = \sum_i \lambda_i v_i v_i^\top$ fait apparaître les matrices $P_i = v_i v_i^\top$. Chacune est une **matrice de projection** : elle projette orthogonalement n'importe quel vecteur sur la droite engendrée par $v_i$.

$$P_i w = v_i v_i^\top w = (v_i^\top w) v_i$$

Le scalaire $v_i^\top w$ est la coordonnée de $w$ dans la direction $v_i$, et $P_i w$ est le vecteur obtenu en gardant uniquement cette composante. On vérifie que $P_i^2 = P_i$ (projeter deux fois = projeter une fois) et $P_i P_j = 0$ pour $i \neq j$ (les projections sont orthogonales entre elles).

La décomposition spectrale s'écrit alors :

$$\Sigma = \sum_{i=1}^N \lambda_i P_i = \sum_{i=1}^N \lambda_i v_i v_i^\top$$

$\Sigma$ est une **somme pondérée de projections** — chaque direction propre contribue à hauteur de sa variance $\lambda_i$. La troncature rang $k$ garde les $k$ termes dominants et écarte les $N-k$ directions de faible variance (ou de bruit).

![[images/1-Mathématiques/Random Matrix Theory/figProjection.png]]
*Figure E. Décomposition de $w$ via les matrices de projection. Gauche : $P_1 w = (v_1^\top w) v_1$ — projection orthogonale sur $v_1$. Milieu : $P_2 w = (v_2^\top w) v_2$ — projection sur $v_2$. Droite : $w = P_1 w + P_2 w = \alpha_1 v_1 + \alpha_2 v_2$ — reconstruction exacte par somme des deux projections.*

### (v) Déterminant

Le **déterminant** de $A$ mesure le **facteur de volume** de la transformation $x \mapsto Ax$ : si tu prends un cube unité dans $\mathbb{R}^N$ et que tu lui appliques $A$, son volume est multiplié par $|\det(A)|$.

$$\det(A) = \prod_{i=1}^N \lambda_i$$

Cette identité est fondamentale. Elle relie directement déterminant et spectre.

> [!warning] Trois cas clés
> **$\det(A) > 0$** : la transformation préserve l'orientation et étire le volume. Toutes les valeurs propres ont le même signe (toutes positives pour PD).
>
> **$\det(A) = 0$** : au moins une valeur propre est nulle. La transformation écrase l'espace sur un sous-espace de dimension inférieure — la matrice est **singulière**, non inversible. En finance : $\Sigma$ non-inversible si $T < N$.
>
> **$\det(A) < 0$** : la transformation retourne l'orientation (comme un miroir). Au moins une valeur propre est négative — la matrice est indéfinie.

![[images/1-Mathématiques/Random Matrix Theory/figDet.png]]
*Figure F. Le déterminant comme facteur de volume. Gauche : $\det > 0$, le carré unité est étiré en parallélogramme, volume augmenté. Milieu : $\det = 0$, le carré est écrasé sur une droite — la matrice est singulière. Droite : $\det < 0$, l'orientation est retournée (miroir).*

En pratique, $\det(\Sigma)$ mesure le "volume total de risque" d'un portefeuille — le produit de toutes les variances directionnelles. Si une direction de risque est nulle ($\lambda_i = 0$), le déterminant s'annule et $\Sigma$ n'est plus inversible, ce qui bloque l'optimisation de Markowitz.

### (vi) Deux visions d'une matrice

Un piège récurrent en algèbre linéaire appliquée : confondre **matrice comme fonction** et **matrice comme données**. Ce sont deux objets conceptuellement différents, même si la notation est identique.

**Matrice comme fonction** : $A \in \mathbb{R}^{N \times P}$ est une transformation linéaire qui envoie des vecteurs de $\mathbb{R}^P$ vers $\mathbb{R}^N$. Les colonnes de $A$ décrivent où vont les vecteurs de base standard. Tout ce qu'on a vu jusqu'ici — valeurs propres, diagonalisation, PSD — concerne cette vision : on étudie comment $A$ déforme l'espace.

**Matrice comme données** : $X \in \mathbb{R}^{T \times N}$ est un tableau où chaque ligne est une observation ($T$ jours de données) et chaque colonne est une variable ($N$ actifs). Ici $X$ ne "transforme" rien — c'est un stockage d'information. Les colonnes de $X$ sont des séries temporelles, pas des directions de transformation.

La confusion surgit parce que $\Sigma = \frac{1}{T} X^\top X$ utilise $X$ comme données pour construire une matrice-fonction $\Sigma$. $X$ est du côté données, $\Sigma$ est du côté fonction.

> [!example] En finance
> $X \in \mathbb{R}^{T \times N}$ : $T = 500$ jours, $N = 200$ actifs. Chaque ligne de $X$ est un vecteur de rendements journaliers — c'est une **donnée**. $\Sigma = \frac{1}{T} X^\top X \in \mathbb{R}^{N \times N}$ est la covariance empirique — c'est une **fonction** qui transforme des portefeuilles (vecteurs de poids $w \in \mathbb{R}^N$) en vecteurs de risque $\Sigma w$. Appliquer $P^\top$ à $X$ (changer de base) opère sur les données : chaque observation est réexprimée dans le repère propre.

![[images/1-Mathématiques/Random Matrix Theory/figDeuxVisions.png]]
*Figure G. Gauche : matrice comme fonction — des vecteurs $x$ sont transformés en $Ax$, les colonnes de $A$ définissent où vont les vecteurs de base. Droite : matrice comme données — chaque point est une observation, les axes sont des features. Même notation $A$, deux objets conceptuellement différents.*

### (vii) Décomposition en valeurs singulières (SVD)

Tout ce qu'on a vu jusqu'ici — diagonalisation $M = Q\Lambda Q^\top$, théorème spectral, vecteurs propres orthogonaux — ne marche que pour les matrices **carrées symétriques**. C'est restrictif : les vraies données arrivent en matrices $X \in \mathbb{R}^{T \times N}$ avec $T \neq N$ (500 jours, 200 actifs ; 60000 images, 784 pixels), et $X$ n'a aucune raison d'être symétrique. Comment décomposer $X$ pour en extraire ses directions principales ?

La **SVD** (Singular Value Decomposition, "décomposition en valeurs singulières") généralise la diagonalisation à toutes les matrices, carrées ou non, symétriques ou non. C'est l'outil le plus important d'algèbre linéaire appliquée.

> [!warning] Théorème SVD
> Toute matrice $X \in \mathbb{R}^{T \times N}$ se factorise sous la forme
> $$X = U \Sigma V^\top$$
> avec :
> - $U \in \mathbb{R}^{T \times T}$ orthogonale ($U^\top U = I_T$),
> - $\Sigma \in \mathbb{R}^{T \times N}$ "diagonale rectangulaire" : entrées $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r > 0$ sur la diagonale (avec $r = \mathrm{rang}(X)$), zéros partout ailleurs,
> - $V \in \mathbb{R}^{N \times N}$ orthogonale ($V^\top V = I_N$).
>
> Les $\sigma_i$ sont les **valeurs singulières**, les colonnes de $U$ et $V$ sont les **vecteurs singuliers gauches** et **droits**. La décomposition existe pour toute matrice réelle.

#### Géométrie : rotation → étirement → rotation

L'interprétation géométrique est limpide. Appliquer $X = U\Sigma V^\top$ à un vecteur $x$, c'est faire trois choses dans l'ordre :

1. $V^\top x$ : on **tourne** $x$ pour qu'il s'exprime dans une base privilégiée (celle des vecteurs singuliers droits $v_i$),
2. $\Sigma (V^\top x)$ : on **étire** axe par axe selon les facteurs $\sigma_1, \dots, \sigma_r$,
3. $U \Sigma V^\top x$ : on **tourne à nouveau** pour s'aligner dans le repère d'arrivée (celui des vecteurs singuliers gauches $u_i$).

Bref : *toute* application linéaire est une rotation, suivie d'un étirement axe par axe, suivi d'une autre rotation. C'est universel.

![[images/1-Mathématiques/Algèbre/figSVD1_trois_etapes.png]]
*Figure SVD-1. Les trois étapes de la SVD sur le cercle unité (cas carré 2×2). Le cercle reste un cercle après $V^\top$ (rotation pure), devient une ellipse alignée sur les axes après $\Sigma$ (étirement de facteur $\sigma_1$ et $\sigma_2$), puis tourne sous l'effet de $U$. Les vecteurs $e_1, e_2$ suivent la même séquence.*

#### Cas rectangulaire : sphère $\to$ ellipsoïde dans un sous-espace

Le vrai pouvoir de la SVD apparaît quand $X$ est rectangulaire. Prenons $X : \mathbb{R}^2 \to \mathbb{R}^3$ (matrice $3 \times 2$). La sphère unité de $\mathbb{R}^2$ — un cercle — est envoyée dans $\mathbb{R}^3$, mais comme l'image de $X$ est au plus de dimension 2, on obtient une **ellipse plate vivant dans un plan oblique** de $\mathbb{R}^3$. Les axes de cette ellipse sont $\sigma_1 u_1$ et $\sigma_2 u_2$. La troisième direction $u_3$ est perpendiculaire à ce plan — c'est une direction de l'espace d'arrivée que $X$ **n'atteint jamais**.

![[images/1-Mathématiques/Algèbre/figSVD2_rectangulaire.png]]
*Figure SVD-2. Cas rectangulaire $X : \mathbb{R}^2 \to \mathbb{R}^3$. À gauche : le cercle unité dans $\mathbb{R}^2$, avec les directions singulières droites $v_1, v_2$ (pointillés). À droite : l'image — une ellipse plate dans $\mathbb{R}^3$, vivant dans le plan engendré par $u_1, u_2$. Les longueurs des demi-axes valent $\sigma_1$ et $\sigma_2$. Le vecteur $u_3$ est hors de l'image : $X$ ne peut produire aucun vecteur dans cette direction.*

Plus généralement : le **rang** de $X$ vaut $r = $ nombre de $\sigma_i > 0$. Les $u_1, \dots, u_r$ engendrent l'image de $X$, les $u_{r+1}, \dots, u_T$ engendrent son complément orthogonal (les directions "inaccessibles"). Les $v_1, \dots, v_r$ engendrent l'orthogonal du noyau, les $v_{r+1}, \dots, v_N$ engendrent le noyau (les directions que $X$ écrase à zéro). La SVD décrit complètement la structure d'une application linéaire.

#### Full SVD vs thin SVD : les dimensions en pratique

Pour $X \in \mathbb{R}^{T \times N}$ avec $T > N$ (cas typique : beaucoup d'observations, peu de variables), la **full SVD** produit $U$ de taille $T \times T$, mais $T - N$ colonnes de $U$ ne servent à rien — elles multiplient les zéros de $\Sigma$. Ces colonnes sont une base du complément orthogonal de l'image, utile en théorie, inutile en pratique. La **thin SVD** (ou SVD économique) les jette :

$$X \;=\; \underbrace{U_r}_{T \times r}\; \underbrace{\Sigma_r}_{r \times r}\; \underbrace{V_r^\top}_{r \times N}, \qquad r = \min(T, N)$$

Le résultat sur $X$ est strictement identique, mais on stocke et calcule $O(rTN)$ au lieu de $O(T^2)$ — gain énorme dès que $T \gg N$.

![[images/1-Mathématiques/Algèbre/figSVD3_dimensions.png]]
*Figure SVD-3. Diagramme des dimensions. Gauche : full SVD — $U$ est $T \times T$, $\Sigma$ a un bloc diagonal $N \times N$ en haut et des zéros en bas, $V^\top$ est $N \times N$. Droite : thin SVD — on garde seulement les $r = N$ premières colonnes de $U$. La reconstruction de $X$ est exacte dans les deux cas.*

> [!warning] Convention en pratique
> Quand on dit "calculer la SVD", on entend presque toujours la **thin SVD**. C'est ce que renvoie `numpy.linalg.svd(X, full_matrices=False)` et `scipy.linalg.svd(X, full_matrices=False)`. Le mode `full_matrices=True` (par défaut !) est surtout utile pour les démonstrations théoriques.

#### Esquisse de construction : Rayleigh sur $\|Xv\|$

D'où viennent les $u_i, \sigma_i, v_i$ ? L'idée centrale : on cherche la direction $v \in \mathbb{R}^N$ unitaire pour laquelle $\|Xv\|$ est maximal — c'est la direction où $X$ "étire le plus".

Or $\|Xv\|^2 = v^\top X^\top X v$, et $X^\top X$ est symétrique PSD. Maximiser $v^\top X^\top X v$ sur la sphère unité est un problème classique de **quotient de Rayleigh** : la solution est $v_1 = $ vecteur propre principal de $X^\top X$, avec valeur $v_1^\top X^\top X v_1 = \lambda_1 = \sigma_1^2$. On pose alors $u_1 = X v_1 / \sigma_1$, et on recommence sur l'orthogonal de $v_1$ pour obtenir $v_2, u_2, \sigma_2$, et ainsi de suite. Au bout de $r$ étapes, on a épuisé l'image de $X$.

C'est cette construction qui justifie que la SVD existe pour toute matrice : la symétrie PSD de $X^\top X$ garantit le théorème spectral, et on en déduit la SVD par cette récurrence.

#### Lien avec la diagonalisation

La construction précédente révèle un lien direct :

$$X^\top X = V \Sigma^\top \Sigma V^\top = V \Sigma^2_{N} V^\top \qquad \text{et} \qquad X X^\top = U \Sigma \Sigma^\top U^\top = U \Sigma^2_{T} U^\top$$

où $\Sigma_N^2 = \mathrm{diag}(\sigma_1^2, \dots, \sigma_N^2)$ (et idem pour $\Sigma_T^2$, en complétant éventuellement de zéros).

> [!warning] À retenir
> **Les valeurs singulières de $X$ sont les racines carrées des valeurs propres de $X^\top X$** (ou de $XX^\top$).
> **Les vecteurs singuliers droits $v_i$ sont les vecteurs propres de $X^\top X$.**
> **Les vecteurs singuliers gauches $u_i$ sont les vecteurs propres de $X X^\top$.**
>
> En particulier : si $\Sigma_{\text{cov}} = \frac{1}{T} X^\top X$ est la covariance empirique, alors $\lambda_i^{\text{cov}} = \sigma_i^2 / T$ et les axes principaux sont $v_i$.

Ce lien donne **deux routes** pour obtenir les axes principaux à partir de $X$ : (1) former $X^\top X$ et diagonaliser, (2) faire directement la SVD de $X$. Les deux donnent le même résultat en arithmétique exacte, **mais pas en arithmétique flottante** — voir plus bas.

![[images/1-Mathématiques/Algèbre/figSVD5_conditionnement.png]]
*Figure SVD-4. Gauche : les deux routes pour obtenir les axes principaux. Droite : démonstration numérique. Quand la plus petite valeur singulière de $X$ vaut $\sigma_{\min} = 10^{-8}$, la diagonalisation de $X^\top X$ donne une erreur relative de ~25% sur $\sigma_{\min}$, alors que la SVD directe donne $\sim 10^{-10}$. L'écart entre les deux courbes croît systématiquement à mesure que $\sigma_{\min}$ diminue : c'est le **carré du conditionnement**. Former $X^\top X$ élève les rapports $\sigma_i / \sigma_j$ au carré, ce qui dégrade la précision relative des petites valeurs singulières.*

En pratique : utiliser la SVD directement sur $X$ (`numpy.linalg.svd`), pas diagonaliser $X^\top X$. Coût comparable, précision incomparable.

#### Pseudo-inverse de Moore-Penrose

Quand $X$ n'est pas carrée ou pas inversible, on ne peut pas écrire $X^{-1}$. Mais la SVD donne un **substitut** universel, le pseudo-inverse :

$$X^+ \;=\; V\, \Sigma^+\, U^\top$$

où $\Sigma^+$ est obtenue en transposant $\Sigma$ et en remplaçant chaque $\sigma_i > 0$ par $1/\sigma_i$ (et en gardant les zéros).

L'usage principal : la solution des **moindres carrés** $\min_w \|X w - y\|^2$ s'écrit directement $w^\star = X^+ y$, valable que $X$ soit inversible ou non. C'est aussi la base de la régression ridge, des projecteurs sur l'image/le noyau, et de la résolution numérique des systèmes singuliers.

#### Théorème d'Eckart-Young : la meilleure approximation de rang $k$

La décomposition SVD permet d'écrire $X$ comme une somme de matrices de rang 1 :

$$X \;=\; \sum_{i=1}^r \sigma_i\, u_i v_i^\top$$

Chacune $\sigma_i u_i v_i^\top$ est une matrice de **rang 1** (extérieurement, un produit colonne × ligne). Plus $\sigma_i$ est grand, plus le terme contribue. La question naturelle : si on garde seulement les $k$ premiers termes, qu'obtient-on ?

> [!warning] Théorème d'Eckart-Young
> Soit $X_k = \sum_{i=1}^k \sigma_i\, u_i v_i^\top$ la troncature de la SVD aux $k$ plus grandes valeurs singulières. Alors **$X_k$ est la meilleure approximation de rang $k$ de $X$** au sens de la norme de Frobenius (et aussi de la norme opérateur) :
> $$X_k \;=\; \arg\min_{\mathrm{rang}(Y) \leq k} \|X - Y\|_F^2$$
> et l'erreur d'approximation vaut exactement
> $$\|X - X_k\|_F^2 \;=\; \sum_{i=k+1}^r \sigma_i^2$$

Ce théorème est **central**. Il transforme la SVD d'un objet algébrique abstrait en outil de compression optimal : pour compresser $X$ en rang $k$, on ne peut pas faire mieux que de garder les $k$ premières composantes singulières. C'est ce qui justifie en aval la PCA, la compression d'images, le débruitage, les systèmes de recommandation à bas rang.

![[images/1-Mathématiques/Algèbre/figSVD4_compression.png]]
*Figure SVD-5. Théorème d'Eckart-Young appliqué à la compression d'image. Une image $400 \times 400$ est reconstruite par troncature SVD à $k = 1, 5, 20, 50$ composantes. À $k=1$ on n'a qu'un dégradé grossier (rang 1 = produit extérieur d'un vecteur ligne et d'un vecteur colonne). À $k=5$ les formes principales (cercle, rectangle) apparaissent. À $k=50$ l'image est quasi indistinguable de l'originale ($k=400$), tout en occupant 4× moins de mémoire. Le spectre des $\sigma_i$ en bas montre la décroissance rapide qui rend la compression efficace.*

> [!example] Une image = une matrice
> Une image en niveaux de gris $400 \times 400$ est exactement une matrice $X \in \mathbb{R}^{400 \times 400}$ où chaque entrée est l'intensité d'un pixel. Stocker l'image complète coûte $400^2 = 160\,000$ valeurs. La stocker en rang $k$ via SVD coûte $k(400 + 400 + 1) = 801k$ valeurs. Avec $k = 50$, on tombe à $40\,050$ valeurs — 4× moins — pour une qualité visuelle quasi identique. C'est exactement l'idée derrière les codecs basés sur des décompositions à bas rang (JPEG utilise une variante par blocs avec la DCT, pas la SVD pleine, mais l'esprit est le même).

> [!warning] Pourquoi c'est central en PCA et en RMT
> Eckart-Young justifie que **garder les $k$ plus grandes composantes principales = meilleure compression linéaire à $k$ dimensions**. En RMT, ça pose la question inverse : si on garde les $k$ plus grandes $\sigma_i$ d'une matrice de données bruitée, est-ce qu'on capte du **signal** ou simplement les fluctuations statistiques les plus grandes du bruit ? C'est exactement la question que Marchenko-Pastur va trancher.

#### Récapitulatif

La SVD est l'outil le plus général d'algèbre linéaire appliquée :
- Elle existe pour **toute** matrice (carrée, rectangulaire, singulière, complexe).
- Elle **généralise la diagonalisation** : pour une matrice symétrique PSD, SVD et diagonalisation coïncident exactement.
- Elle décrit la **géométrie complète** d'une application linéaire : rotation, étirement axe par axe, rotation.
- Elle donne le **pseudo-inverse** (moindres carrés universels) et la **meilleure approximation à bas rang** (Eckart-Young).
- Elle est **numériquement stable**, contrairement au calcul de $X^\top X$ qui carre le conditionnement.

Tout ce qui suit dans les notes de RMT et de PCA s'appuie sur cette décomposition.
