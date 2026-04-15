
## Classification par arbre de décision

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

Dataset
![[Pasted image 20260415185926.png|261]]

étape 1: on va calculer le gini initial sur notre jeu de donnée donc sans split 

Gini $_{\text {initial }}=1-\left((6 / 10)^2+(4 / 10)^2\right)=1-(0.36+0.16)=\mathbf{0 . 4 8}$

étape 2: on va split le jeu de donnée par features eg pour Fatigue on a la feuille :
* feuille des oui: $\text { Gini }_{\text {Oui }}=1-\left((3 / 4)^2+(1 / 4)^2\right)=1-(0.5625+0.0625)=\mathbf{0 .37 5}$
* feuille des non: $\text { Gini }_{N o n}=1-\left((1 / 6)^2+(5 / 6)^2\right)=1-(0.0278+0.6944)=\mathbf{0 . 2 7 7}$
puis un gini pondéré en fonction du nombre d'échantillons par feuille
$\begin{gathered}G i n i_{\text {Stump }}=\left(\frac{4}{10} \times 0.375\right)+\left(\frac{6}{10} \times 0.2778\right) \\ G i n i_{\text {Stump }}=0.15+0.1666=\mathbf{0 . 3 1 6 6}\end{gathered}$

![[Pasted image 20260415185817.png]]
on répète la même opération pour Fièvre et pour Toux => on chosit Fatigue qui a le plus bas Gini index. 
On peut donc calculer également la notion de gain qui sera utilisé plus tard pour le feature importance Gain = Gini_vide - Gini_fatigue = 0.48-0.32=0.16=16%

étape 3: 

Gain gini 0.32-0.25=0.07=7%

![[Pasted image 20260415191756.png|447]]

étape 4: 


gain = gini parent - gini pondéré enfants

eg
* gain du split fatigue, gini parent = 0.48 puis gini apres split 0.32 donc Gain = 0.48-0.32=0.16
* gain du split fievre : Gain_fiever=0.375-0.25=0.125 !!!!! c'est bien 0.375 et pas 0.32 car on prend le Gini_oui de la branche => 0.125*0.4=0.05
* Gain_toux_droite=0.277-0=0.277 =>0.277*0.6=0.166 
* Gain_toux_gauche=0.50-0=0.50 => donc au global toux c'est 0.50*2 pondéré y'a 2

donc résumé: dans une table variable vs somme des gains 'importance'
* toux = 0.166+0.10=0.266
* fatigue=0.160
* fièvre=0.050




![[Pasted image 20260415192034.png]]


avec ça on a pu plotter un feature importance plot

![[Pasted image 20260415193238.png|525]]


Préciser qu'on a du rajouter un critère d'arrêt aussi pour arrêter bon ici c'est que toutes les feuilles sont pures 

LAST STEP:

Having defined the splitting criterion, we will get a big tree $T_0$. Its leaves define regions $R_1, ..., R_m$. We then \textcolor{cornellred}{prune} this tree, meaning that we collapse some of its leaves into the parent nodes.\\

For any tree $T,$ let $|T|$ denote its number of leaves aka terminal nodes. We define
$$
C_{\alpha}(T)=\sum_{j=1}^{|T|}\left[1-\hat{p}_{c_{j}}\left(R_{j}\right)\right]+\alpha|T|
$$

We seek the tree $T \subseteq T_{0}$ that minimizes $C_{\alpha}(T) .$ It turns out that this can be done by pruning the weakest leaf one at a time. Note that $\alpha$ is a \textcolor{cornellred}{tuning parameter}, and a larger $\alpha$ yields a smaller tree. CART picks $\alpha$ by 5 - or 10 -fold cross-validation

=> c'est la ou faut expliquer l'histire du alpha où on va pruner l'arbre avec |T| = nombre de feuilles


+ rajouter un plan !!!! avec les points points coloriés en fct de leur classe 



## Regression


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





## L'histoire du biais variance et 

et expliquer pk random foreest a été inventé juste apres



# Random forest Brieman 2001


utilise CART comme base learner

# AdaBoost 1997 Freund & Schapire





# Gradient Boosting Friedman 2001


il utilise CART comme weak learner


# XGBoost Chen & Guestrin 2016


# LightGBM - Ke et al 2017 Microsoft


# CatBoost Prokorenkova 2018 Yandex


