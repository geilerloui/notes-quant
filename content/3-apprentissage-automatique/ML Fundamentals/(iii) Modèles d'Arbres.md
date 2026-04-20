# I - Arbre de décision

## A. Classification
### Rappel : deux familles de modèles

En classification supervisée, on distingue deux approches :

**Modèles discriminatifs** : on modélise directement $p(y = 1 \mid x \in R_j)$ — étant donné qu'un patient tombe dans la feuille $R_j$ (i.e. présente certains symptômes), quelle est la probabilité qu'il soit malade ?

**Modèles génératifs** : on modélise $p(x \mid c)$ — sachant qu'un patient est malade, quelle est la probabilité qu'il présente tel symptôme ? On s'intéresse à la vraisemblance des features conditionnellement à la classe.

Les arbres de décision sont des modèles **discriminatifs**.

---

### Impureté de Gini

#### Origine

L'indice de Gini est introduit en économie par Corrado Gini en 1912 pour mesurer les inégalités de revenus. Breiman le réutilise en 1984 pour construire un critère de split dans CART (Classification and Regression Trees) : l'**impureté de Gini**.

après y'a eu divers modifications ID3 entropie classificatgion Quinlan en 1986 et C4.5 extension industrielle de ID3 par Quinlan en 1993


#### Construction

Considérons une feuille $R_j$ contenant des patients malades ($y = 1$) ou sains ($y = 0$). On note :

$$p_k(R_j) = \frac{1}{|R_j|} \sum_{i \in R_j} \mathbf{1}[y_i = k]$$

la proportion de patients de classe $k$ dans $R_j$. Sur notre exemple : $p_0(R_j) = 0.7$, $p_1(R_j) = 0.3$.

On tire deux patients $A$ et $B$ de façon aléatoire et **indépendante** dans $R_j$. On se demande : **quelle est la probabilité qu'ils soient de classes différentes ?** On énumère tous les événements possibles : $A$ est malade et $B$ ne l'est pas, ou $A$ ne l'est pas et $B$ l'est :

$$p(A \neq B) = p\big((A=1 \cap B=0) \cup (A=0 \cap B=1)\big)$$

Ces deux événements étant disjoints :

$$= p(A=1 \cap B=0) + p(A=0 \cap B=1)$$

$$= p_1(R_j)(1-p_1(R_j)) + (1-p_1(R_j))p_1(R_j)$$

On peut réécrire cette expression plus élégamment en passant par le complémentaire. On se demande : quelle est la probabilité que $A$ et $B$ soient de la **même classe** ? $A$ est malade et $B$ l'est aussi, ou $A$ ne l'est pas et $B$ non plus :

$$p(A = B) = \sum_{k=0}^{1} p(A=k \cap B=k)$$

Et donc :

$$p(A \neq B) = 1 - p(A = B) = 1 - \sum_{k=0}^{1} p(A=k \cap B=k)$$

Et par indépendance de $A$ et $B$, $p(A=k \cap B=k) = p_k^2$, ce qui donne :

$$\boxed{G(R_j) = 1 - \sum_{k=1}^{K} p_k(R_j)^2}$$

ce qui se généralise naturellement à $K$ classes.

#### Interprétation

Si $G(R_j)$ est élevé, la feuille est **impure** : deux patients tirés au hasard ont de grandes chances d'être de classes différentes — on n'a aucune information discriminante. À l'inverse, $G(R_j) = 0$ signifie que la feuille est **pure** : tous les patients appartiennent à la même classe.

C'est la même intuition que l'entropie $H(R_j) = -\sum_k p_k \log p_k$ : les deux sont minimaux en $0$ et maximaux sous la loi uniforme.

📌 *[Graphe à ajouter : comparaison Gini vs entropie en fonction de $p_1$ pour $K=2$]*
![[Pasted image 20260415193533.png|427]]


---

### Assignation de classe à une feuille

À chaque feuille $R_j$ on associe la classe majoritaire :

$$c_j = \underset{k \in \{1, \ldots, K\}}{\arg\max} \; p_k(R_j)$$

Sur notre exemple : $c_j = \arg\max\{p_0(R_j), p_1(R_j)\} = \arg\max\{0.7, 0.3\} = 0$ (classe saine majoritaire).

📌 *[Image à ajouter : feuille $R_j$ avec des $\times$ et $\circ$ représentant les deux classes]*


----

![[Pasted image 20260415185926.png|261]]
Table 1. Dataset

METTRE ICI EN MODE CLIQUABLE LA FORMULE DE GINI CLASSIQUE ET GINI PONDEREE je parle de formule générale avec n_1 N ... 

The CART algorithm begins by considering splitting on variable $j$ and split point $s$, and defines the regions
$$
R_{1}=\left\{X \in \mathbb{R}^{p}: X_{j} \leq s\right\}, \quad R_{2}=\left\{X \in \mathbb{R}^{p}: X_{j}>s\right\}
$$

$GI_{left} = 1 - p(C=0 \mid x \in R_1)^2 - p(C=1 \mid x \in R_1)^2$

$GI_{right} = 1 - p(C=0 \mid x \in R_2)^2 - p(C=1 \mid x \in R_2)^2$

$GI_{chest~pain} = \frac{n_1}{N} GI_{left}+ \frac{n_2}{N} GI_{right}$


**Étape 1 — Calcul du coefficient de Gini** : On calcule le Gini, sans split, sur la colonne cible : "Grippe"
$$Gini_{\text {initial }}=1-\left((6 / 10)^2+(4 / 10)^2\right)=1-(0.36+0.16)=\mathbf{0 . 4 8}$$
**Étape 2 — Choix du 1er split** : 
(i) On va calculer le Gini pondéré pour chaque feature on commence par la feature "Fatigue ?"
* feuille des oui: $\text { Gini }_{\text {Oui }}=1-\left((3 / 4)^2+(1 / 4)^2\right)=1-(0.5625+0.0625)=\mathbf{0 .37 5}$
* feuille des non: $\text { Gini }_{N o n}=1-\left((1 / 6)^2+(5 / 6)^2\right)=1-(0.0278+0.6944)=\mathbf{0 . 2 7 7}$
puis un gini pondéré en fonction du nombre d'échantillons par feuille
$$\begin{gathered}G i n i_{\text {Stump }}=\left(\frac{4}{10} \times 0.375\right)+\left(\frac{6}{10} \times 0.2778\right) \\ G i n i_{\text {Stump }}=0.15+0.1666=\mathbf{0 . 3 1 6 6}\end{gathered}$$
On répète la même opération pour "Fièvre ?" et "Toux ?"
(ii) On choisit celui qui minimise le Gini ie celui qui réduit la "surprise" => C'est la feature "Fatigue ?"
(iii) On peut calculer le gain feature importance 
$$Gain = Gini_{vide} - Gini_{fatigue} = 0.48-0.32=\textbf{0.16}$$


![[Pasted image 20260415185817.png|583]]
Figure X. Comparaison des trois stumps

**Étape 3 — Choix du 2nd split:**
(i) On va calculer le Gini pondéré pour {Fièvre ?, Toux ?}
(ii) On choisit celui qui minimise le Gini ie celui qui réduit la "surprise" => C'est la feature "Fièvre ?"
(iv) On peut calculer le gain feature importance:
$$Gain = Gini_{fatigue} - Gini_{fièvre} =  0.32-0.25=0.07$$

![[Pasted image 20260415191756.png|354]]
Figure X. 2nd split

**Étape 4 — Choix du 3e et 4e split:**



![[Pasted image 20260418102957.png|482]]
Figure X. 3e et 4e splits

**Étape 5 — Critère d'arrêts:**

Préciser qu'on a du rajouter un critère d'arrêt aussi pour arrêter bon ici c'est que toutes les feuilles sont pures 


**Étape 6 — L'élagage de l'arbre:**

LAST STEP:

Having defined the splitting criterion, we will get a big tree $T_0$. Its leaves define regions $R_1, ..., R_m$. We then \textcolor{cornellred}{prune} this tree, meaning that we collapse some of its leaves into the parent nodes.\\

For any tree $T,$ let $|T|$ denote its number of leaves aka terminal nodes. We define
$$
C_{\alpha}(T)=\sum_{j=1}^{|T|}\left[1-\hat{p}_{c_{j}}\left(R_{j}\right)\right]+\alpha|T|
$$

We seek the tree $T \subseteq T_{0}$ that minimizes $C_{\alpha}(T) .$ It turns out that this can be done by pruning the weakest leaf one at a time. Note that $\alpha$ is a \textcolor{cornellred}{tuning parameter}, and a larger $\alpha$ yields a smaller tree. CART picks $\alpha$ by 5 - or 10 -fold cross-validation

=> c'est la ou faut expliquer l'histire du alpha où on va pruner l'arbre avec |T| = nombre de feuilles

**Étape 7 — Prédiction:**

À chaque feuille $R_j$ on associe la classe majoritaire :

$$c_j = \underset{k \in \{1, \ldots, K\}}{\arg\max} \; p_k(R_j)$$

Sur notre exemple : $c_j = \arg\max\{p_0(R_j), p_1(R_j)\} = \arg\max\{0.7, 0.3\} = 0$ (classe saine majoritaire).

+ rajouter un plan !!!! avec les points points coloriés en fct de leur classe 

+VISUALISATION DES PLANS 

![[Pasted image 20260418133309.png|309]]




![[Pasted image 20260418131332.png|338]]



**Étape 8 — Feature importance:**

eg
* gain du split fatigue, gini parent = 0.48 puis gini apres split 0.32 donc Gain = 0.48-0.32=0.16
* gain du split fievre : Gain_fiever=0.375-0.25=0.125 !!!!! c'est bien 0.375 et pas 0.32 car on prend le Gini_oui de la branche => 0.125*0.4=0.05
* Gain_toux_droite=0.277-0=0.277 =>0.277*0.6=0.166 
* Gain_toux_gauche=0.50-0=0.50 => donc au global toux c'est 0.50*2 pondéré y'a 2

donc résumé: dans une table variable vs somme des gains 'importance'
* toux = 0.166+0.10=0.266
* fatigue=0.160
* fièvre=0.050

avec ça on a pu plotter un feature importance plot

![[Pasted image 20260415193238.png|525]]
Figure X. Feature importance










## B. Régression

**Étape 1 — Initialisation** : On fait ça

**Étape 2 — Transformation** : On fait ça


![[Pasted image 20260415195718.png|446]]


Au début on calcule juste la moyenne et on calcule le SSR sur toute la feuille
$$
\begin{gathered}
S S R=(10-51,67)^2+(50-51,67)^2+(100-51,67)^2+(90-51,67)^2+ \\
S S R \approx 1736+2,8+2336+1469+2,8+1736=\mathbf{7 2 8 2}, \mathbf{6}
\end{gathered}
$$

ok
$\mathrm{SSR}=1156+81+1296+1849+1+1369=\mathbf{5 7 5 2}$

ok

![[Pasted image 20260415200316.png]]



Au premier noeud on test différent split on prend le split qui minimise le SSR 

![[Pasted image 20260415200754.png]]

1. Test sur le Dosage ( $x_1$ )

Tentons un split "médian" pour séparer la montée de la descente, par exemple à $x_1=8.5$ (entre les patients 2 et 3 ).
- Groupe Gauche $\left(x_1<8.5\right)$ : Patients 1, 2 .
- Valeurs $y$ : $\{15,40\}$. Moyenne $\bar{y}_G=27.5$.
- $S S R_G=(15-27.5)^2+(40-27.5)^2=156.25+156.25=\mathbf{3 1 2 . 5}$.
- Groupe Droite $\left(x_1>8.5\right)$ : Patients 3, 4, 5, 6 .
- Valeurs $y$ : $\{85,92,50,12\}$. Moyenne $\bar{y}_D=59.75$.
- $S S R_D=(85-59.75)^2+(92-59.75)^2+(50-59.75)^2+(12-59.75)^2 \approx 637+ 1040+95+2280=\mathbf{4 0 5 2}$.
- SSR Total après split $x_1=312.5+4052=\mathbf{4 3 6 4 . 5}$.
- Gain : $5752-4364.5=1387.5$.
2. Test sur le Score Santé ( $x_2$ )

Tentons de séparer les patients par score santé, par exemple à $x_2=40$.
- Groupe Gauche $\left(x_2<40\right)$ : Patients 1, 3, 5 .
- Valeurs $y$ : $\{15,85,50\}$. Moyenne $\bar{y}_G=50$.
- $S S R_G=(15-50)^2+(85-50)^2+(50-50)^2=1225+1225+0=\mathbf{2 4 5 0}$.
- Groupe Droite $\left(x_2>40\right)$ : Patients 2, 4, 6 .
- Valeurs $y$ : $\{40,92,12\}$. Moyenne $\bar{y}_D=48$.
- $S S R_D=(40-48)^2+(92-48)^2+(12-48)^2=64+1936+1296=\mathbf{3 2 9 6}$.
- SSR Total après split $x_2=2450+3296=\mathbf{5 7 4 6}$.
- Gain : $5752-5746=\mathbf{6}$. (C'est presque inutile !).





![[Pasted image 20260415201016.png|407]]



![[Pasted image 20260415201127.png|441]]
L'algorithme va chercher à séparer ces 4 points. Le split le plus logique pour capturer la "chute" de la cloche est entre le Patient 4 et le Patient 5.
Seuil : $(15.5+22.0) / 2=\mathbf{1 8 . 7 5}$
- Sous-groupe Droite-Gauche $\left(8.5 \leq x_1<18.75\right)$ : Patients 3,4 .
- $y=\{85,92\}$
- Moyenne $\bar{y}_{D G}=(85+92) / 2=\mathbf{8 8 . 5}$
- $S S R_{D G}=(85-88.5)^2+(92-88.5)^2=12.25+12.25=\mathbf{2 4 . 5}$
- Sous-groupe Droite-Droite $\left(x_1 \geq 18.75\right)$ : Patients 5, 6 .
- $y=\{50,12\}$
- Moyenne $\bar{y}_{D D}=(50+12) / 2=\mathbf{3 1}$
- $S S R_{D D}=(50-31)^2+(12-31)^2=361+361=722$

SSR total de l'arbre : 312.5 (branche de gauche inchangée) $+24.5+722=\mathbf{1 0 5 9}$
On est passé d'un SSR de 5752 (départ) à 1059. L'arbre devient très précis !

![[Pasted image 20260415201153.png|320]]




graphe finale 


![[Pasted image 20260415201352.png|437]]
critère d'arrêt:

la on l'a fait sur notre training set mais en fait faut un validation set qui va calculer son propre SSR => d'où sort le alpha a faire aussi 

![[Pasted image 20260415204452.png]]


ok on peut ploter en 3D 


![[Pasted image 20260415202158.png|466]]


on affiche aussi le plan de partition:

![[Pasted image 20260415203227.png|474]]



1. Le calcul des gains pour ton arbre

Reprenons les étapes de construction pour calculer l'importance cumulée :
1. DOSAGE $\left(X_1\right)$ :
- Split $1\left(t_1=8.5\right)$ : Réduction de 5485.5.
- Split $2\left(t_2=18.75\right)$ : Réduction de 313.3.
- Total Dosage : $5485.5+313.3=\mathbf{5 7 9 8 . 8}$.
2. SANTÉ ( $X_2$ ) :
- Split $3\left(t_3=45.5\right)$ : Réduction de 722.0.
- Total Santé : 722.0.




![[Pasted image 20260415203850.png]]

hyperparamètres: => ça c'est ceux du random forest mais probablmenet mettre ceux du decision tree
* n_estimators: Le nombre d'arbres plus il y'en a mieux cest jusqu'à un certain plateau de stabilité
* max_depth: la profondeur des arbres
* min_samples_leaf: le nombre minimum d'échantillon pour créer une feuille 




## L'histoire du biais variance et 

et expliquer pk random foreest a été inventé juste apres


# II - Méthode du bagging

La base c'est un base learner puis on aggrège, on pourrait même se taper un délire c le graphe en 3D que j'ai fais pour la régression du decision tree ben c'est comme si je générai m graphes en parallèle et chacun donnée leurs propres valeur et on fait une prédiction.

## A. Random forest Brieman 2001



utilise CART comme base learner on parle de base learner 


on fait du bootstraping 
![[Pasted image 20260415223833.png|556]]

puis on prends le premier in bag train on va construire un arbre avec sqrt(d) features a chaque fois à la première itération on choisit uniquement Fatigue et Toucher ? 


![[Pasted image 20260415224923.png|483]]

2e itération on tire aléatoirement avec remise deux nouvelles features 


![[Pasted image 20260415224939.png|509]]
Et on itère comme ça jusqu'à la création de l'arbre

Enfin on aura une forêt d'arbre 

![[Pasted image 20260415225510.png|474]]


Une prédiction:
![[Pasted image 20260416094358.png]]

out of bag errors:
- **Collecte des votes OOB** : Tu identifies tous les arbres où l'Observation 2 était dans le tableau **Out-of-Bag** (par exemple, les arbres 1, 4 et 5).
- **Passage dans les arbres** : Tu fais "descendre" les caractéristiques de l'obs 2 (Fièvre: Oui, Fatigue: Oui, Toux: Non) dans chacun de ces arbres spécifiques.
- **Résumé local** : Tu obtiens une petite table de vote juste pour cette observation (ex: 2 votes "Grippe" / 1 vote "Sain").
- **Prédiction finale OOB** : La majorité l'emporte. Si "Grippe" gagne, ta prédiction OOB pour l'obs 2 est **YES**.
- **Comparaison** : Tu compares ce **YES** à la vraie valeur dans ton tableau d'origine (la colonne "Gri.").
	- Si c'est identique : l'observation est bien classée.
    - Si c'est différent : c'est une erreur.



Erreur $\mathrm{OOB}=\frac{\text { Nombre d'observations mal classées }}{\text { Nombre total d'observations }(\mathrm{N})}$


Feature importance:
* méthode du gini importance MDI Mean Decrease Impurity -> par défaut
* Permutation importance: plus recommandé et utilise le out of bag (OOB) l'idée est que si une variable est importante "casser" ses données devrait faire chuter la précision du modèle.



hyperparamètres:
* n_estimators: Le nombre d'arbres plus il y'en a mieux cest jusqu'à un certain plateau de stabilité
* max_depth: la profondeur des arbres
* min_samples_leaf: le nombre minimum d'échantillon pour créer une feuille 


# III - Boosting

Principe du weak learner

## A. AdaBoost — Freund & Schapire, 1997

### Le cadre général : stagewise additive modeling

Avant d'entrer dans l'algorithme, il faut comprendre le cadre dans lequel AdaBoost s'inscrit. L'idée du boosting c'est de construire $f$ comme une **somme de fonctions simples** (les stumps) :

$$f(x) = \sum_{t=1}^T \alpha_t h_t(x)$$

En théorie, on pourrait optimiser tous les $\alpha_t$ et $h_t$ simultanément — mais c'est computationnellement cauchemardesque. Le *stagewise additive modeling* propose à la place de construire $f$ **terme par terme, de façon greedy** : à chaque étape $t$, on cherche le meilleur $(\alpha_t, h_t)$ à ajouter à $f_{t-1}$ déjà fixé :

$$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i L\!\left(y_i,\ f_{t-1}(x_i) + \alpha\, h(x_i)\right)$$

puis on met à jour $f_t = f_{t-1} + \alpha_t h_t$, sans jamais retoucher les termes précédents.

L'analogie : imagine que tu veux approximer une courbe mystère ($y = 30 + x + \sin(x)$). Au lieu de trouver la formule d'un coup, tu procèdes par étapes — d'abord une constante $f_1 = 30$ qui capture la moyenne, puis tu ajoutes une droite pour capter la tendance linéaire, puis un sinus pour les ondulations. Chaque terme corrige ce que le précédent a raté. C'est exactement ce que fait AdaBoost, avec des stumps à la place.
![[Pasted image 20260416162642.png|209]]
qu'on approxime avec 
![[Pasted image 20260416162659.png]]



Il reste à choisir la **loss $L$**. AdaBoost utilise la **loss exponentielle** :

$$L(y, f(x)) = e^{-y f(x)}$$

Le produit $y \cdot f(x)$ mesure si la prédiction est correcte : s'il est positif (bonne prédiction), la loss est $< 1$ et faible ; s'il est négatif (erreur), la loss explose exponentiellement. C'est ce choix précis qui va faire tomber toutes les formules d'AdaBoost.

---

L'idée centrale d'AdaBoost est de construire itérativement une forêt de stumps où **chaque stump se concentre sur les erreurs du précédent**. Pour forcer cela, on associe à chaque observation un **poids $w_i^{(t)}$** qui mesure à quel point elle est difficile à classer au tour $t$ — plus une observation a été mal classée aux tours précédents, plus son poids est élevé, et plus le prochain stump sera forcé de la prendre en compte.

### Dérivation des formules

> [!note]- Preuve complète
> **Initialisation.** Le cadre stagewise définit $w_i^{(t)} = e^{-y_i f_{t-1}(x_i)}$. À $t=1$, $f_0 = 0$ donc $w_i^{(1)} = 1$ pour tout $i$. On normalise par $n$ : $w_{i,1} = 1/n$.
>
> **Calcul de $\alpha_t$ et $\varepsilon_t$.** À l'étape $t$, on cherche $(\alpha_t, h_t)$ qui minimisent :
> $( \alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i w_i^{(t)} \cdot e^{-\alpha y_i h(x_i)}$
> Comme $y_i, h(x_i) \in \{-1, +1\}$, on a $y_i h(x_i) = 1 - 2\cdot\mathbf{1}[y_i \neq h(x_i)]$, donc :
> $= e^{-\alpha} \sum_i w_i^{(t)} \cdot e^{2\alpha \cdot \mathbf{1}[y_i \neq h(x_i)]}$
> $= e^{-\alpha} \left[ \sum_{y_i = h(x_i)} w_i^{(t)} + e^{2\alpha} \sum_{y_i \neq h(x_i)} w_i^{(t)} \right]$
> On pose :
> $\boxed{\varepsilon_t = \sum_{y_i \neq h(x_i)} w_i^{(t)}}$
> ce qui donne :
> $= e^{-\alpha}\left[(1 - \varepsilon_t) + e^{2\alpha} \varepsilon_t\right]$
> On dérive par rapport à $\alpha$ et on annule :
> $-e^{-\alpha}(1 - \varepsilon_t) + e^{\alpha}\varepsilon_t = 0 \implies e^{2\alpha} = \frac{1-\varepsilon_t}{\varepsilon_t} \implies \boxed{\alpha_t = \frac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}}$
>
> **Mise à jour des poids.** On pose $f_t = f_{t-1} + \alpha_t h_t$ et on développe $w_i^{(t+1)} = e^{-y_i f_t(x_i)}$ :
> $w_i^{(t+1)} = e^{-y_i(f_{t-1}(x_i) + \alpha_t h_t(x_i))} = \underbrace{e^{-y_i f_{t-1}(x_i)}}_{w_i^{(t)}} \cdot e^{-y_i \alpha_t h_t(x_i)}$
> $\boxed{w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}}$


---

### Tour 1 — Illustration à $t=1$

**Étape 1 — Initialisation** : $w_{i,1} = \dfrac{1}{n}$ pour tout $i$.

![[Pasted image 20260417171440.png|305]]
Table. XXXX

**Étape 2 — Choix du stump optimal** : on choisit $h_t$ qui minimise l'erreur pondérée : $\varepsilon_t = \sum_{i=1}^n w_{i,t} \cdot \mathbf{1}\left[h_t(x_i) \neq y_i\right]$

On construit trois stumps (tronc d'arbre) et au lieu de choisir celui qui minimise l'Index de Gini on va prendre celui qui minimise l'erreur pondérée. Ainsi avec $n = 8$, $w_{i,1} = 1/8$ pour tout $i$. En testant les trois features :
- Chest Pain : 3 erreurs $\Rightarrow \varepsilon_1 = 3/8$
- Blocked Arteries : 4 erreurs $\Rightarrow \varepsilon_1 = 4/8 = 1/2$
- Weight $> 176$ : 1 erreur $\Rightarrow \varepsilon_1 = 1/8$

On retient le stump "**Weight $> 176$**"
![[Pasted image 20260416105200.png]]
**Étape 3 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$

On va ensuite assigner un poids à ce "stump" : $\alpha_1 = \frac{1}{2}\ln(\frac{1-0.125}{0.125}) =\frac{1}{2}\ln(7) \approx 0.97$.

A CACHER: L'interprétation de $\alpha_t$ est naturelle : un stump parfait ($\varepsilon_t \to 0$) obtient un $\alpha_t$ très grand ; un stump aléatoire ($\varepsilon_t = 1/2$) obtient $\alpha_t = 0$ ; un stump qui se trompe systématiquement obtient un $\alpha_t$ négatif — son vote est inversé.

![[Pasted image 20260416110650.png|240]]

**Étape 4 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$

Mise à jour des poids :

$\text{mal classé :} \quad \tfrac{1}{8} \cdot e^{+0.97} \approx 0.33 \qquad \text{bien classé :} \quad \tfrac{1}{8} \cdot e^{-0.97} \approx 0.05$

A CACHER: Ces deux figures nous disent que 

![[Pasted image 20260416111212.png|466]]

On va ensuite modifier les poids dans la table qu'on normalise ensuite

![[Pasted image 20260416150127.png|467]]
Table X

---

### Tour 2 — Illustration à $t=2$

**Étape 1 — Choix du stump optimal** : on choisit $h_t$ qui minimise l'erreur pondérée : $\varepsilon_t = \sum_{i=1}^n w_{i,t} \cdot \mathbf{1}\left[h_t(x_i) \neq y_i\right]$

Après mise à jour des poids, on recommence. En recalculant $\varepsilon_t$ pour chaque feature :

- Weight $> 176$ (gagnant du tour 1) : 1 erreur à poids $0.49$ $\Rightarrow \varepsilon_2 = 0.49$
- Chest Pain : 3 erreurs à poids $0.07$ chacune $\Rightarrow \varepsilon_2 = 0.21$
- Blocked Arteries : 4 erreurs à poids $0.07$ $\Rightarrow \varepsilon_2 = 0.28$

**Étape 2 — Poids du stump** : $\alpha_t = \frac{1}{2}\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$

On retient **Chest Pain**. Son poids : $\alpha_2 = \frac{1}{2}\ln(\frac{1-0.21}{0.21}) =\frac{1}{2}\ln(3.76) \approx 0.66$.

A CACHER: L'interprétation de $\alpha_t$ est naturelle : un stump parfait ($\varepsilon_t \to 0$) obtient un $\alpha_t$ très grand ; un stump aléatoire ($\varepsilon_t = 1/2$) obtient $\alpha_t = 0$ ; un stump qui se trompe systématiquement obtient un $\alpha_t$ négatif — son vote est inversé.

![[Pasted image 20260416110650.png|240]]

**Étape 3 — Mise à jour des poids** : $w_i^{(t+1)} = w_i^{(t)} \cdot e^{-y_i \alpha_t h_t(x_i)}$

La mise à jour des poids au tour 2 : l'individu $n°4$ (poids $0.49$, bien classé cette fois) passe à $\approx 0.25$ ; les 3 erreurs de Chest Pain (poids $0.07$) passent à $\approx 0.13$ ; les autres tombent à $\approx 0.035$.

A CACHER: Ces deux figures nous disent que 

![[Pasted image 20260416111212.png|475]]

On modifie la table en conséquence

![[Pasted image 20260416151745.png|382]]
Table X.

---

### Prédiction finale

Après $T$ itérations :

$$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

Pour un patient avec Weight $= 205$ et Chest Pain $=$ Yes :

$$H(x) = \text{sign}\left[0.97 \times (+1) + 0.66 \times (+1)\right] = \text{sign}(1.63) = +1 \implies \text{Yes Heart Disease}$$

![[Pasted image 20260416153745.png]]


## B. Gradient Boosting — Friedman, 2001

**(a) Adaboost:** Pour rappel, AdaBoost construisait son modèle selon la récurrence :

$F_t(x) = F_{t-1}(x) + \alpha_t h_t(x)$

où à chaque étape on résolvait :

$(\alpha_t, h_t) = \arg\min_{\alpha, h} \sum_i L\!\left(y_i,\ F_{t-1}(x_i) + \alpha\, h(x_i)\right)$

avec la **loss exponentielle**. La limite : cette loss est très sensible aux outliers, et on ne peut pas en changer.

**(b) Gradient boosting:** Gradient Boosting garde exactement la même structure :

$F_t(x) = F_{t-1}(x) + \nu \cdot \gamma_t \cdot h_t(x)$

où $h_t$ est un arbre CART, $\gamma_t$ le pas optimal calculé à cette étape, et $\nu \in (0,1]$ un learning rate qui contrôle la contribution de chaque arbre. À chaque étape on résout :

$(\gamma_t, h_t) = \arg\min_{\gamma, h} \sum_i L\!\left(y_i,\ F_{t-1}(x_i) + \nu \cdot \gamma\, h(x_i)\right)$

La différence n'est pas dans la forme — elle est dans **ce sur quoi on entraîne $h_t$** : plutôt que de résoudre ce problème directement (difficile pour une loss quelconque), on l'approche en entraînant $h_t$ sur le **gradient négatif de la loss**, ce qui permet d'utiliser n'importe quelle loss différentiable.

L'algorithme se déroule en 3 étapes : initialisation de $F_0$, puis $T$ itérations de correction, puis prédiction finale $F_T(x)$.

---

### Étape 1 — Initialisation : $F_0(x)$

À $t=0$, il n'y a pas encore de modèle ($F_{t-1} = 0$) ni d'arbre à entraîner ($h = 1$, constante). La forme générale se réduit donc à chercher simplement la meilleure constante $\gamma$ qui minimise la loss sur tout le dataset :

$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$

Ici $\gamma$ joue le rôle de $F(x)$ — c'est littéralement $F_0(x) = \gamma$ pour tout $x$, donc on substitue directement dans la loss. Avec le MSE $L(y, F) = \frac{1}{n}\sum_i (y_i - F(x_i))^2$, on dérive par rapport à $\gamma$ et on annule :

$\frac{\partial}{\partial \gamma} \sum_{i=1}^n (y_i - \gamma)^2 = \sum_{i=1}^n -2(y_i - \gamma) = 0 \implies \boxed{\gamma = \frac{1}{n}\sum_{i=1}^n y_i}$

$F_0$ est donc simplement la **moyenne des $y_i$** — une feuille unique qui prédit la même valeur pour tout le monde.

Sur notre exemple (Height, Favorite Color, Gender $\to$ Weight) :

$F_0(x) = \frac{88 + 76 + 56}{3} = 73.3 \text{ kg}$

📌 *[Image : feuille unique avec valeur 73.3]*

---

### Étape 2 — Itérations : construire les arbres correctifs

À chaque itération $t = 1, \ldots, T$, on procède en quatre sous-étapes.

**A — Calcul des pseudo-résidus**

Rappelons la descente de gradient classique dans l'espace des **paramètres** :

$$\theta_t = \theta_{t-1} - \eta \cdot \nabla_\theta \mathcal{L}(\theta_{t-1})$$

À chaque étape, on corrige $\theta$ en allant dans la direction opposée au gradient de la loss. GBM fait exactement la même chose, mais dans l'espace des **fonctions** : au lieu de mettre à jour un vecteur $\theta$, on met à jour une fonction $F$. Le "gradient" devient alors le gradient de la loss par rapport aux valeurs prédites $F(x_i)$, et on l'annote $r_{i,t}$ :

$r_{i,t} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}$

La mise à jour idéale serait 

$$
F_t(x_i) =  F_{t-1}(x_i) + \nu \cdot \Big(-\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}\Big)=F_{t-1}(x_i) + \nu \cdot r_{i,t}
$$
— exactement le pendant fonctionnel de la descente de gradient. Le problème : $r_{i,t}$ n'est défini que sur les $n$ points d'entraînement, pas sur tout l'espace des $x$. C'est pour ça qu'on **entraîne un arbre $h_t$ pour approximer ces pseudo-résidus** — l'arbre généralise la direction de descente à tout $x$.

Avec le MSE $L = \frac{1}{2}(y_i - F(x_i))^2$ :

$r_{i,t} = y_i - F_{t-1}(x_i)$

MAIS DU COUP VU QUE CES DES ARBRES DE REGRESION CEST DES INDICATRICES LES F_t-1 ?

Ce sont les résidus classiques — c'est une coïncidence du MSE. Avec une autre loss les pseudo-résidus seraient différents, d'où le terme "pseudo".

Sur notre exemple, à $t=1$ avec $F_0 = 73.3$ :

$r_{1,1} = 88 - 73.3 = 14.7 \qquad r_{2,1} = 76 - 73.3 = 2.7 \qquad r_{3,1} = 56 - 73.3 = -17.3$

📌 *[Image : tableau avec colonne $r_{i,1}$]*

**B — Entraînement d'un arbre sur les pseudo-résidus**

On entraîne un arbre CART sur les $r_{i,t}$ — on cherche à **prédire les résidus**, pas les $y_i$ directement. L'arbre crée $J_t$ feuilles (régions terminales $R_{j,t}$). Sur notre exemple on obtient un stump avec $J_1 = 2$ feuilles :

- $R_{1,1}$ : Height $< 1.55$ $\to$ résidu $-17.3$
- $R_{2,1}$ : Height $\geq 1.55$ $\to$ résidus $14.7, 2.7$

📌 *[Image : stump]*

**C — Calcul du pas optimal $\gamma_{j,t}$ par feuille**

Pour chaque feuille $R_{j,t}$, on cherche le $\gamma$ qui minimise la loss sur les observations qui tombent dans cette feuille :

$\gamma_{j,t} = \arg\min_\gamma \sum_{x_i \in R_{j,t}} L\!\left(y_i,\ F_{t-1}(x_i) + \nu \cdot \gamma\right)$

$$
\begin{aligned}
\gamma_{jm} &= \underset{\gamma}{argmin} \sum_{x_i \in R_{ij}} L(y_i, F_{m-1}(x_i)+\gamma) \\
&= \partial_{\gamma} \sum_{x_i \in R_{ij}} (y_i - (F_{m-1}(x_i) + \gamma)^2) \\
&= \sum_{x_i \in R_{ij}} y_i - F_{m-1}(x_i) - \gamma = 0 
\end{aligned}
$$

Avec le MSE, cela revient à calculer la **moyenne des résidus** dans la feuille :

$\gamma_{j,t} = \frac{1}{|R_{j,t}|} \sum_{x_i \in R_{j,t}} r_{i,t-1}$

Sur notre exemple :

$\gamma_{1,1} = -17.3 \qquad \gamma_{2,1} = \frac{14.7 + 2.7}{2} = 8.7$

**D — Mise à jour du modèle**

On met à jour $F_t$ en ajoutant la contribution de l'arbre :

$F_t(x) = F_{t-1}(x) + \nu \sum_{j=1}^{J_t} \gamma_{j,t} \cdot \mathbf{1}(x \in R_{j,t})$

Avec $\nu = 0.1$ sur notre exemple :

$F_1(x_1) = 73.3 + 0.1 \times 8.7 = 74.2$
$F_1(x_2) = 73.3 + 0.1 \times 8.7 = 74.2$
$F_1(x_3) = 73.3 + 0.1 \times (-17.3) = 71.6$

📌 *[Image : $F_1(x)$]*


## C. XGBoost Chen & Guestrin 2016


## D. LightGBM - Ke et al 2017 Microsoft


## E. CatBoost Prokorenkova 2018 Yandex
