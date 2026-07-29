---
title: a-Markowitz
order: 1
---
# Théorie moderne du portefeuille — Markowitz (1952)

> Avant Markowitz, la recherche financière répondait à *"quel titre acheter ?"*. Markowitz déplace la question vers *"quel portefeuille construire ?"* — il est le premier à formaliser la diversification avec des métriques précises, et à montrer qu'on peut réduire le risque sans sacrifier le rendement en combinant intelligemment des actifs. Ces notes suivent la chronologie : on part de la théorie de l'utilité (vNM, 1947), on construit le critère espérance-variance (Markowitz, 1952), puis on dérive la frontière efficiente avec et sans actif sans risque, jusqu'au théorème des deux fonds.

## I. Théorie de l'utilité de Von Neumann-Morgenstern

**Le besoin.** Pour comparer des choix risqués (un actif sûr à 500 € vs un pari 50/50 entre 100 € et 900 €), on a besoin d'une fonction qui transforme une situation incertaine en un nombre comparable. C'est ce que fait la **fonction d'utilité** $u(W)$ : elle mesure la satisfaction que l'agent retire d'une richesse $W$. Si l'agent préfère $A$ à $B$, alors $u(A) > u(B)$.

> [!warning] Aversion au risque et concavité
> Une fonction d'utilité est toujours croissante ($u' > 0$). C'est le signe de $u''$ qui détermine l'attitude face au risque :
> 
> $$u''(W) < 0 \;\Rightarrow\; \text{concave} \;\Rightarrow\; \textbf{averse au risque}$$
> 
> $$u''(W) > 0 \;\Rightarrow\; \text{convexe} \;\Rightarrow\; \textbf{amateur de risque}$$
> 
> L'exemple canonique pour un agent averse au risque est $u(W) = \ln(W)$, avec $u' = 1/W > 0$ et $u'' = -1/W^2 < 0$.

![[images/5-Finance/A_markowitz/im1.png]]

**Figure 1.** Fonction $u(W) = \ln(W)$ concave. Pour un pari 50/50 entre 100 € et 900 €, l'utilité espérée $E[u(W)] \approx 5.70$ est inférieure à $u(500\text{ €}) \approx 6.21$ : l'agent préfère 500 € certains au pari, bien que les deux aient la même espérance.

**Prime de risque et équivalent certain.** Face à un pari risqué d'espérance $E(W)$, un agent averse au risque accepterait une somme certaine $C < E(W)$ procurant la même utilité. On appelle $C$ l'**équivalent certain** et la différence

$$\pi = E(W) - C \;\geq\; 0$$

la **prime de risque** : ce que l'agent sacrifie en espérance pour avoir la certitude. Par définition $u(C) = E[u(W)]$.

![[images/5-Finance/A_markowitz/im2.png]]

**Figure 2.** La droite rouge relie $u(W_1)$ à $u(W_2)$ — son milieu donne $E[u(W)]$. La courbe bleue étant au-dessus, $u(E(W)) > E[u(W)]$ : l'agent préfère le certain. $C$ est l'équivalent certain tel que $u(C) = E[u(W)]$.

> [!warning] Coefficient d'Arrow-Pratt
> La mesure formelle de l'aversion au risque locale est
> 
> $$\theta(W) = -\frac{u''(W)}{u'(W)}.$$
> 
> Plus $\theta$ est grand, plus l'agent est averse au risque. C'est la version formelle du paramètre $\lambda$ qui apparaîtra dans la formule de Markowitz.

**Théorème de Von Neumann-Morgenstern (1944).** Tout agent rationnel face à des choix risqués agit *comme s'il maximisait* $E[u(W)]$ — l'espérance de sa fonction d'utilité appliquée à sa richesse finale. C'est le socle théorique sur lequel Markowitz construit tout : il a le droit de raisonner en termes d'$E[u(W^x)]$ parce que vNM lui assure que c'est *le* critère de décision rationnel.

## II. Le modèle de Markowitz

### A. Du critère vNM au critère espérance-variance

**Richesse finale et rendement du portefeuille.** On note $x = (w_1, \ldots, w_n)$ le **vecteur de composition** du portefeuille — les poids de chaque actif, avec $\sum_i w_i = 1$. La richesse finale vaut

$$W^x = W_0\,(1 + R(x))$$

où $W_0$ est la richesse initiale et $R(x)$ le rendement réalisé. On en déduit

$$E[W^x] = W_0\,(1 + E[R(x)]), \qquad \text{Var}(W^x) = W_0^2\,\text{Var}(R(x)).$$

$W_0$ étant une constante, le passage de $W^x$ à $R(x)$ est **transparent pour l'optimisation** : maximiser $f(E[W^x], \text{Var}(W^x))$ par rapport à $x$ est équivalent à maximiser $f(E[R(x)], \text{Var}(R(x)))$. On travaille donc directement sur les rendements.

**Hypothèse quadratique.** Travailler avec $E[u(W^x)]$ en toute généralité est trop complexe — il faudrait connaître toute la distribution de $W^x$. Markowitz suppose que $u$ est **quadratique** :

$$u(W) = W - c\,W^2.$$
![[images/5-Finance/A_markowitz/im3.png]]


**Figure 3.** La parabole $u(W) = W - cW^2$ croît jusqu'au sommet $W^* = 1/(2c)$ puis décroît — zone économiquement absurde (plus de richesse rendrait moins heureux). C'est le prix à payer pour la tractabilité : sous cette hypothèse, $E[u(W^x)]$ ne dépend que de l'espérance et de la variance du rendement.

> [!note]- Preuve : hypothèse quadratique → critère espérance-variance
> $$E[u(W^x)] = E[W^x - c(W^x)^2] = E[W^x] - c\,E[(W^x)^2].$$
> 
> Or $\text{Var}(W^x) = E[(W^x)^2] - (E[W^x])^2$, donc $E[(W^x)^2] = \text{Var}(W^x) + (E[W^x])^2$. En substituant :
> 
> $$E[u(W^x)] = E[W^x] - c\,\text{Var}(W^x) - c\,(E[W^x])^2.$$
> 
> Le dernier terme est une constante quand on compare des portefeuilles à même espérance de rendement. On obtient la forme de Markowitz :
> 
> $$E[u(W^x)] \;\propto\; E[R(x)] - \lambda\,\text{Var}(R(x)).$$

> [!warning] Score d'utilité d'un portefeuille
> $$U(x) = E[R(x)] - \lambda\,\text{Var}(R(x))$$
> 
> où $\lambda \geq 0$ est le **coefficient d'aversion au risque** de l'investisseur (lié à $\theta$ d'Arrow-Pratt). C'est une fonction du vecteur de composition $x$ : on fait varier les poids, on recalcule $U$, on cherche le maximum.

> 💡 **Lecture du critère.** $U(x)$ est un compromis : on aime l'espérance de rendement (terme $+E[R(x)]$), on n'aime pas la variance (terme $-\lambda\text{Var}(R(x))$), et $\lambda$ règle l'arbitrage. $\lambda$ grand → agent prudent qui pénalise fortement le risque ; $\lambda$ petit → agent agressif qui privilégie le rendement. Cet arbitrage scalaire est le *cœur* de toute la suite : la frontière efficiente, la CML, le portefeuille tangent — tout en découle.

### B. Pourquoi la diversification fonctionne

Le rendement espéré du portefeuille est **linéaire** en les poids :

$$E[R(x)] = \sum_i w_i\,E[R_i].$$

La variance, elle, fait intervenir toutes les **covariances** :

$$\text{Var}(R(x)) = \sum_i \sum_j w_i\,w_j\,\sigma_{ij}, \qquad \sigma_{ij} = \sigma_i\,\sigma_j\,\rho_{ij}.$$

C'est l'asymétrie qui rend la diversification possible. Pour deux actifs $A$ et $B$ :

$$\text{Var}(R) = w_A^2\sigma_A^2 + w_B^2\sigma_B^2 + 2\,w_A w_B\,\sigma_{AB}.$$

Le terme croisé $2\,w_A w_B\,\sigma_{AB}$ est la clé : si $\rho_{AB} < 0$ (actifs anti-corrélés), ce terme **réduit** la variance totale sans toucher au rendement espéré.

> [!example] Diversification avec deux actifs
> $w_A = w_B = 50\%$, $\sigma_A = \sigma_B = 20\%$, $\rho_{AB} = -0.5$ (donc $\sigma_{AB} = -0.02$).
> 
> $$\text{Var}(R) = 0.25 \times 0.04 + 0.25 \times 0.04 + 2 \times 0.5 \times 0.5 \times (-0.02) = 0.01.$$
> 
> Soit $\sigma = 10\%$. Chaque actif seul avait $\sigma = 20\%$ : la diversification a divisé le risque par deux, sans rien sacrifier en rendement espéré.

> 💡 **Le message conceptuel.** Diversifier n'est pas simplement "mettre ses œufs dans plusieurs paniers". C'est exploiter le fait que la variance d'une combinaison n'est *pas* la combinaison des variances — il y a un terme croisé qui peut être négatif. Plus les actifs sont décorrélés, plus ce terme tire la variance vers le bas. C'est mathématiquement *gratuit* : on baisse le risque sans toucher à l'espérance.

## III. Frontière efficiente sans actif sans risque

**Allocation optimale.** L'**allocation optimale** $x^*$ est le vecteur de poids qui maximise l'utilité de l'investisseur :

$$x^* = \underset{x}{\arg\max}\; U(x) = \underset{x}{\arg\max}\; E[R(x)] - \lambda\,\text{Var}(R(x)).$$

Concrètement : quelle pondération des actifs (30 % Apple, 50 % Total, 20 % cash...) donne le meilleur score $U$ ?

**Frontière efficiente.** Chaque vecteur $x$ donne un point $(\sigma(x),\, E[R(x)])$ dans le graphe risque/rendement. Les points **non dominés** — ceux pour lesquels on ne peut pas faire mieux en rendement sans prendre plus de risque — forment la **frontière efficiente**. C'est l'ensemble des allocations optimales possibles, paramétré par $\lambda$.

> [!warning] Deux formulations équivalentes du programme
> $$\text{(1)}\quad \sup_x\,E[R(x)] \quad\text{s.c.}\quad \text{Var}(R(x)) = \sigma^2$$
> 
> $$\text{(2)}\quad \min_x\,\text{Var}(R(x)) \quad\text{s.c.}\quad E[R(x)] = m$$
> 
> Pour chaque $\sigma$ (resp. $m$), la solution $x^*$ est un portefeuille efficient. L'ensemble de ces solutions trace la frontière. Les deux formulations donnent la même frontière : (1) maximise le rendement à risque fixé, (2) minimise le risque à rendement fixé.

![[images/5-Finance/A_markowitz/im4.png]]

**Figure 4.** Chaque point gris est un portefeuille possible. La courbe bleue pleine est la frontière efficiente (partie haute) ; la partie en pointillés est non efficiente (à $\sigma$ donné, on peut faire mieux en rendement). Le point $\lambda$ grand correspond à un investisseur très averse au risque (à gauche), $\lambda$ petit à un investisseur moins averse (à droite).

**Coordonnées du portefeuille de variance minimale.** On définit deux scalaires à partir de la matrice de covariance $\Sigma$ et du vecteur $\mathbf{1} = (1,\ldots,1)^\top$ :

$$a = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \qquad b = E\mathbf{R}^\top \Sigma^{-1} \mathbf{1}.$$

Le **portefeuille de variance minimale** — point le plus à gauche de la frontière — a pour coordonnées exactes :

$$\sigma_{\min} = \frac{1}{\sqrt{a}}, \qquad \mu_{\min} = \frac{b}{a}.$$

Il sépare deux zones : **au-dessus**, la frontière efficiente ; **en dessous**, les portefeuilles dominés.

> [!note]- Preuve (Cauchy-Schwarz)
> On cherche $\min_x \text{Var}(R(x)) = x^\top \Sigma x$ sous contrainte $\mathbf{1}^\top x = 1$. Par Cauchy-Schwarz dans l'espace muni du produit scalaire $\langle u,v\rangle_{\Sigma^{-1}} = u^\top \Sigma^{-1} v$ :
> 
> $$|\langle u, v\rangle_{\Sigma^{-1}}|^2 \leq \|u\|^2_{\Sigma^{-1}}\,\|v\|^2_{\Sigma^{-1}}$$
> 
> avec égalité si et seulement si $u$ et $v$ sont colinéaires. La minimisation donne directement $\sigma_{\min} = 1/\sqrt{a}$.

## IV. Frontière efficiente avec actif sans risque

### A. Capital Market Line et portefeuille tangent

**Introduction d'un actif sans risque.** On introduit un actif sans risque $S^0$ de rendement $r_f$ et $\sigma = 0$. Il se place en $(0, r_f)$ dans le graphe risque/rendement. Combiner $S^0$ et un portefeuille risqué $x$ trace une **droite** depuis $(0, r_f)$ : c'est l'effet du fait que $\sigma$ et $E[R]$ sont tous deux linéaires en la proportion investie dans $S^0$.

> [!warning] Capital Market Line (CML)
> La nouvelle frontière efficiente est la droite tangente à la frontière courbe :
> 
> $$m(\sigma) = r_f + \frac{E[R(\xi)] - r_f}{\sigma(\xi)}\cdot\sigma$$
> 
> où $\xi$ est le **portefeuille tangent** — le seul point où la droite touche la frontière courbe.

![[images/5-Finance/A_markowitz/im5.png]]

**Figure 5.** La droite verte (CML) part de $r_f$ et est tangente à la frontière bleue en $\xi$. Elle domine la frontière courbe pour tout $\sigma$ : c'est l'avantage d'avoir accès à un actif sans risque — n'importe quel investisseur préfère un point sur la CML à un point de la frontière courbe à $\sigma$ égal.

**Ratio de Sharpe.** Le **ratio de Sharpe** d'un portefeuille $x$ mesure le rendement excédentaire par unité de risque :

$$S(x) = \frac{E[R(x)] - r_f}{\sigma(x)}.$$

Géométriquement, c'est la **pente de la droite** depuis $(0, r_f)$ vers le point $(\sigma(x), E[R(x)])$.

![[images/5-Finance/A_markowitz/im6.png]]

**Figure 6.** Parmi toutes les droites possibles depuis $r_f$, celle de pente maximale est la CML. Elle touche la frontière exactement en $\xi$ — donc le portefeuille tangent est le **portefeuille de Sharpe maximal**.

> 💡 **Le portefeuille tangent est doublement remarquable.** $\xi$ appartient à la fois à la frontière sans actif sans risque (la courbe) et à la frontière avec (la droite). C'est le seul portefeuille 100 % risqué qu'un agent rationnel voudra détenir. Il maximise le Sharpe — autrement dit, c'est la meilleure récompense par unité de risque que l'on puisse obtenir avec les actifs risqués disponibles.

### B. Théorème des deux fonds

> [!warning] Théorème des deux fonds (two-fund separation)
> Tout agent rationnel qui suit un portefeuille efficient partage son investissement entre exactement deux fonds :
> 
> $$\text{Portefeuille optimal} = x_0 \cdot S^0 \;+\; (1-x_0)\cdot \xi$$
> 
> où $x_0 \in [0,1]$ dépend de l'aversion au risque $\lambda$ :
> - $\lambda$ grand $\Rightarrow$ $x_0$ grand : beaucoup d'actif sans risque, peu de $\xi$.
> - $\lambda$ petit $\Rightarrow$ $x_0$ petit : beaucoup de $\xi$, peu d'actif sans risque.

> 💡 **Conséquence pratique.** Peu importe le profil de risque de l'investisseur, le portefeuille risqué optimal est **toujours** $\xi$ — seule la dose change. Il n'est pas nécessaire de construire un portefeuille risqué sur mesure pour chaque client : il suffit de construire $\xi$ une fois pour toutes, puis de le mélanger avec $S^0$ dans la proportion adaptée à chaque client. C'est la fondation théorique de la gestion indicielle et des fonds "balanced" qui mélangent un fonds actions de référence et du cash.

## V. Limites du modèle

Le modèle de Markowitz est élégant, mais il repose sur des hypothèses fortes qui produisent des effets indésirables en pratique. La limite la plus citée — celle qu'on retrouve systématiquement en entretien — concerne le choix de la **variance** comme mesure de risque.

### A. La variance pénalise hausses et baisses symétriquement

La variance est par définition une mesure **symétrique** :

$$\text{Var}(R) = \mathbb{E}\big[(R - \mathbb{E}[R])^2\big].$$

C'est le carré qui pose problème : un écart de $+10\%$ par rapport à la moyenne contribue $0.1^2 = 0.01$ à la variance, exactement comme un écart de $-10\%$. **Une hausse exceptionnelle est pénalisée autant qu'une perte exceptionnelle.**

> 💡 **Le bon sens financier.** Aucun investisseur réel ne considère un gros gain comme du "risque". Le risque, c'est le *downside* — la perte. Pénaliser les hausses revient à punir un portefeuille pour avoir trop bien performé, ce qui est absurde. Markowitz hérite de cette symétrie parce qu'il vient de la statistique gaussienne, où elle est naturelle ; mais en finance, elle ne l'est pas.

### B. Semi-variance : pénaliser uniquement le downside

La parade standard est la **semi-variance**, qui ne compte que les écarts négatifs :

$$\text{semivar}(R) = \mathbb{E}\Big[\big((R - \mathbb{E}[R])^-\big)^2\Big]$$

où $(y)^- = \max(-y, 0) = -\min(y, 0)$ ne retient que la partie négative. Concrètement :

- Un rendement supérieur à la moyenne contribue $0$ à la semi-variance.
- Un rendement inférieur à la moyenne est pénalisé par son écart au carré, comme dans la variance classique.

> [!example] Variance vs semi-variance
> Soit deux écarts symétriques par rapport à la moyenne : $+10\%$ et $-10\%$.
> 
> | | Variance | Semi-variance |
> |---|:---:|:---:|
> | Écart de $+10\%$ | $0.10^2 = 0.01$ | $0$ (ignoré) |
> | Écart de $-10\%$ | $0.10^2 = 0.01$ | $0.10^2 = 0.01$ |
> 
> La variance pénalise les deux ; la semi-variance ne pénalise que la perte.

Remplacer $\text{Var}$ par $\text{semivar}$ dans le critère de Markowitz donne le programme de **mean-semivariance optimization**, plus aligné avec la perception intuitive du risque. Le coût est calculatoire : la semi-variance n'a pas de forme matricielle aussi simple que $x^\top \Sigma x$, et l'optimisation devient plus lourde.

### C. Autres limites brièvement

La symétrie de la variance n'est qu'une des limites de Markowitz. D'autres reproches classiques :

- **Sensibilité aux paramètres.** $E[R]$ et $\Sigma$ sont estimés sur données historiques avec une grande incertitude. Une petite erreur d'estimation produit des poids optimaux très instables — on parle d'**error maximization** : l'optimiseur a tendance à concentrer les poids sur les actifs dont $E[R]$ est *surestimée*. C'est ce qui motive les approches Black-Litterman, robust optimization, ou Bayesian.
- **Hypothèse gaussienne implicite.** Le critère espérance-variance est exact si les rendements sont gaussiens, ou si l'utilité est quadratique. En réalité, les rendements ont des **queues épaisses** (fat tails) et des **asymétries** (skewness) que la variance ignore complètement.
- **Horizon unique.** Le modèle est statique (une seule période). Les approches multi-périodes (Merton, programmation dynamique) sont nécessaires pour le rebalancement.

Ces critiques motivent les extensions modernes : mesures de risque cohérentes (CVaR, voir Annexe A.4), modèles factoriels pour stabiliser $\Sigma$, et approches bayésiennes pour intégrer la vue de l'investisseur.

## Annexe A — Markowitz en pratique : contraintes du programme

En théorie, le programme de Markowitz est posé sur $x \in \mathbb{R}^n$ avec une seule contrainte : $\mathbf{1}^\top x = 1$. En pratique, le gérant fait face à une longue liste de contraintes additionnelles, imposées par la réglementation, le mandat client, ou la praticabilité de l'exécution. Cette annexe les passe en revue, des plus simples (poids) aux plus structurées (cardinalité, turnover, risque, concentration).

> [!warning] Forme générale du programme contraint
> $$\max_x\; E[R]^\top x - \lambda\, x^\top \Sigma x \qquad \text{s.c.}\quad x \in \mathcal{X}$$
> 
> où $\mathcal{X}$ est l'ensemble admissible défini par les contraintes. Selon la nature des contraintes ajoutées, le problème reste un **QP** (Quadratic Program) standard, devient un **QCQP** (avec contrainte quadratique), ou un **MIP/MIQP** (Mixed Integer (Quadratic) Program) quand on introduit des variables binaires.

### A.1 Contraintes de poids

Les contraintes les plus élémentaires portent directement sur le vecteur $x$ :

- **Budget** : $\mathbf{1}^\top x = 1$. Toute la richesse est investie.
- **Long-only** : $x_i \geq 0$ pour tout $i$. Interdit la vente à découvert (short-selling).
- **Bornes par actif** (upper/lower bounds) : $\ell_i \leq x_i \leq u_i$. Empêche un actif d'être trop gros ou impose une exposition minimale. Cas typique : $u_i = 5\%$ pour qu'aucun actif ne dépasse 5% du portefeuille.

Ces trois contraintes sont linéaires en $x$ : ajoutées à un QP, elles laissent le problème dans la classe QP, donc résolvable en temps polynomial par des solveurs standards (CVXPY, MOSEK, Gurobi).

### A.2 Contrainte de cardinalité (L0-norm)

**Le besoin.** Souvent, on veut limiter le nombre d'actifs détenus — pour réduire les coûts de suivi, simplifier la mise en œuvre, ou répondre à un mandat (ex. *"un fonds concentré sur 30 actions maximum"*). On exprime ça par la pseudo-norme $\ell_0$ :

$$\|x\|_0 \leq K, \qquad K < N$$

où $\|x\|_0$ compte le nombre d'éléments non nuls de $x$, et $N$ est la taille de l'univers d'investissement.

> [!warning] Pourquoi ça pose problème
> La pseudo-norme $\ell_0$ est **non convexe et discontinue** : passer de $x_i = 0$ à $x_i = \varepsilon > 0$ fait sauter $\|x\|_0$ d'une unité, peu importe la taille de $\varepsilon$. Le problème devient combinatoire — il faut explorer les sous-ensembles d'actifs possibles, soit $\binom{N}{K}$ combinaisons.

**Solution standard : reformulation MIP.** On introduit une **variable binaire** $z_i \in \{0, 1\}$ pour chaque actif :

- $z_i = 1$ si l'actif $i$ est détenu,
- $z_i = 0$ sinon.

Et deux contraintes :

$$\sum_{i=1}^n z_i \leq K \qquad \text{et} \qquad x_i \leq M \cdot z_i \quad \forall i$$

où $M$ est une constante "big-M" suffisamment grande (généralement le poids maximal autorisé, ex. $M = u_i$).

> 💡 **Le tour de force du big-M.** La deuxième contrainte couple $x_i$ et $z_i$ :
> - Si $z_i = 0$ → $x_i \leq 0$, donc avec long-only ($x_i \geq 0$) on force $x_i = 0$.
> - Si $z_i = 1$ → $x_i \leq M$, contrainte non bloquante : $x_i$ reste libre dans $[0, M]$.
> 
> La variable binaire pilote donc un *interrupteur* : $z_i$ décide si l'actif est dans le portefeuille, $x_i$ décide combien. La somme $\sum z_i \leq K$ limite le nombre d'interrupteurs allumés.

> [!example] 3 actions, $K = 1$ (un seul actif autorisé)
> Soit $A = \{x_A, x_B, x_C\}$ et $z = \{z_A, z_B, z_C\} \in \{0,1\}^3$. Avec $M = 1$ :
> 
> **(i)** La contrainte $\sum z_i \leq 1$ force le programme à choisir l'un des trois vecteurs :
> 
> $$z = (1, 0, 0), \quad z = (0, 1, 0), \quad \text{ou} \quad z = (0, 0, 1).$$
> 
> **(ii)** Si le programme choisit $z = (1, 0, 0)$ (actif A) :
> 
> | Actif | Contrainte | Conséquence |
> |---|---|---|
> | A | $x_A \leq 1 \cdot 1 = 1$ | $x_A \in [0, 1]$, libre |
> | B | $x_B \leq 1 \cdot 0 = 0$ | $x_B = 0$, stocké out |
> | C | $x_C \leq 1 \cdot 0 = 0$ | $x_C = 0$, stocké out |
> 
> **(iii)** En pratique, le solveur regarde la fonction objectif et identifie l'actif qui maximise l'utilité — disons B. Il choisit alors $z_B = 1$ et le programme se réduit à un QP standard sur l'actif B uniquement.

> [!note]- Combinatoire et scaling
> $\|x\|_0 \leq K$ revient à choisir le meilleur sous-ensemble de $K$ actifs parmi $N$, soit $\binom{N}{K}$ possibilités. Pour $N = 100$ et $K = 30$, c'est environ $3 \cdot 10^{25}$ combinaisons — infaisable en énumération directe. Les solveurs MIP (Gurobi, CPLEX) utilisent du Branch-and-Bound pour explorer cet espace efficacement, mais le problème reste **NP-hard** en général.

**Alternatives heuristiques.** Le MIP est exact mais coûteux. Deux approches plus rapides en pratique :

- **Relaxation Lasso** ($\ell_1$). On remplace $\|x\|_0 \leq K$ par une pénalité $\|x\|_1 = \sum_i |x_i|$ dans l'objectif. C'est convexe, donc résolvable en QP. Le Lasso produit naturellement des solutions *sparses* (beaucoup de poids exactement nuls), mais on ne contrôle pas directement le nombre de non-nuls — il faut calibrer le coefficient de pénalité.
- **Heuristique de seuil**. On résout le QP sans contrainte de cardinalité, puis on supprime les $N - K$ poids les plus faibles et on renormalise. Rapide mais pas optimal : la solution renormalisée peut être loin de l'optimum sous contrainte.

### A.3 Contrainte de turnover

**Le besoin.** Quand on rebalance un portefeuille existant $x_{\text{prev}}$ vers $x$, chaque mouvement génère des coûts de transaction. On veut limiter le **turnover** — la quantité totale échangée :

$$\sum_{i=1}^n |x_i - x_{i,\text{prev}}| \leq T$$

où $T$ est le seuil maximal autorisé (ex. $T = 0.20$ pour *"je ne veux pas bouger plus de 20% du portefeuille"*).

> [!example] Deux actions, $x_{\text{prev}} = (0.5, 0.5)$, $T = 0.10$
> 
> **Scénario A :** le programme propose $(0.7, 0.3)$.
> 
> $$|0.7 - 0.5| + |0.3 - 0.5| = 0.2 + 0.2 = 0.4 = 40\% \;\Rightarrow\; \text{refusé}.$$
> 
> **Scénario B :** le programme propose $(0.55, 0.45)$.
> 
> $$|0.55 - 0.5| + |0.45 - 0.5| = 0.05 + 0.05 = 0.1 = 10\% \;\Rightarrow\; \text{accepté}.$$

**Reformulation pour solveurs.** La valeur absolue n'est pas différentiable, et certains solveurs n'aiment pas les $|\cdot|$ directement. On les linéarise en introduisant deux variables auxiliaires positives par actif :

- $u_i \geq 0$ : ce qu'on **achète** de l'actif $i$,
- $v_i \geq 0$ : ce qu'on **vend** de l'actif $i$.

Avec la décomposition

$$x_i - x_{i,\text{prev}} = u_i - v_i$$

et la contrainte de turnover devient

$$\sum_{i=1}^n (u_i + v_i) \leq T.$$

Le problème reste un QP linéaire-quadratique standard, sans valeur absolue.

> 💡 **Pourquoi ça marche.** Pour tout réel $y$, on peut écrire $y = u - v$ avec $u, v \geq 0$ et $|y| = u + v$ (à condition que la solution optimale ne soit pas dégénérée — ce qui est le cas ici parce qu'on minimise les coûts, donc le solveur n'a aucun intérêt à *à la fois* acheter et vendre le même actif). Cette astuce est universelle pour gérer les normes $\ell_1$ dans des programmes linéaires/quadratiques.

### A.4 Contraintes de risque

Au-delà du contrôle implicite via le terme $-\lambda\, x^\top \Sigma x$ de l'objectif, on peut imposer des contraintes **explicites** sur le risque.

**Variance cible.** On borne directement la variance du portefeuille :

$$x^\top \Sigma x \leq \sigma_{\text{target}}^2.$$

C'est une contrainte quadratique convexe (puisque $\Sigma$ est positive semi-définie). Le programme devient un **QCQP**, toujours résolvable efficacement par des solveurs standards.

**Tracking error.** Quand on gère par rapport à un benchmark $x_b$ (un indice comme le S&P 500), ce qui compte n'est pas la variance absolue mais l'écart à l'indice. On définit l'**active weight** $x - x_b$ — l'écart de poids actif par rapport à la composition du benchmark — et on borne la variance de cet écart :

$$(x - x_b)^\top \Sigma (x - x_b) \leq \text{TE}_{\text{limit}}^2$$

où TE est la **tracking error** (écart-type de l'active return, en %).

> [!note]- Cas extrêmes : ETF et fonds actif
> - Si $x = x_b$ (réplication parfaite du benchmark), alors $x - x_b = (0, \ldots, 0)$ et TE $= 0$. C'est le cas d'un **ETF** indiciel : on ne fait que reproduire l'indice.
> - Si $x \neq x_b$, on prend des **paris actifs** par rapport au benchmark, ce qui augmente la TE.

**Interprétation gaussienne du tracking error.** Sous l'hypothèse que le rendement actif $R_a = R_p - R_b$ suit une loi normale,

$$R_p - R_b \sim \mathcal{N}(\alpha,\, \text{TE}^2),$$

où $\alpha$ est l'espérance du rendement actif (l'*alpha* du gérant) et TE l'écart-type. La règle empirique des intervalles de confiance gaussiens donne alors :

> [!example] TE = 2% : que ça veut dire concrètement ?
> Avec une TE de 2% (et $\alpha \approx 0$ pour simplifier), à la fin de l'année :
> 
> - **68% de chance** que ton rendement soit dans $[\text{benchmark} - 2\%,\ \text{benchmark} + 2\%]$ (zone à $\pm 1\sigma$).
> - **95% de chance** d'être dans $[\text{benchmark} - 4\%,\ \text{benchmark} + 4\%]$ (zone à $\pm 2\sigma$).
> 
> C'est ainsi qu'on traduit une contrainte technique (TE limit) en langage commercial pour le client.

**Contrainte CVaR.** On peut imposer que les pertes attendues dans le pire scénario restent bornées :

$$\text{CVaR}_\alpha(x) \leq c.$$

Le CVaR (Conditional Value-at-Risk, alias Expected Shortfall) est l'espérance des pertes au-delà du quantile $\alpha$. Voir [[01_Mesures de risque (VaR, CVaR, ES)]] pour la définition complète, le lien avec la VaR, et les trois formulations possibles (min CVaR, max espérance pénalisée par CVaR, contrainte CVaR ≤ seuil). Contrairement à la VaR, le CVaR est **convexe**, donc compatible avec un programme d'optimisation standard.

### A.5 Contrainte de concentration

**Le besoin.** Sans contrainte de concentration, l'optimiseur peut produire des solutions très peu diversifiées — par exemple mettre 90% du portefeuille sur 2 actifs alors qu'on en a 50 dans l'univers. Pour empêcher ça, on borne l'**indice de Herfindahl-Hirschman** (HHI) :

$$\sum_{i=1}^n x_i^2 \leq H^*$$

où le carré pénalise plus lourdement les gros poids que les petits, et $H^*$ est un seuil à calibrer.

> [!warning] Lecture du HHI
> $$\sum_i x_i^2 = \begin{cases} 1 & \text{si } x = (1, 0, \ldots, 0) : \text{tout sur une seule action (concentration max)} \\ \dfrac{1}{n} & \text{si } x = (1/n, \ldots, 1/n) : \text{equally-weighted (concentration min)} \end{cases}$$
> 
> Plus $H^*$ est petit, plus on force la diversification. Choisir $H^* = 1/n$ revient à imposer un portefeuille parfaitement equally-weighted ; $H^* = 1$ n'impose rien.

> 💡 **Origine économique du HHI.** Le HHI vient de l'économie industrielle, où il sert à mesurer la concentration d'un marché : un secteur où une entreprise détient 100% des parts a un HHI de $1$ (monopole) ; un secteur parfaitement concurrentiel avec $n$ entreprises de poids $1/n$ a un HHI de $1/n$. On transpose la mesure aux portefeuilles : un portefeuille concentré sur peu de lignes est l'analogue d'un marché monopolistique, alors qu'un portefeuille bien diversifié est un marché concurrentiel.

La contrainte $\sum_i x_i^2 \leq H^*$ est convexe (forme quadratique avec matrice identité, qui est positive définie), donc le programme reste un QCQP.

## Annexe B — Questions d'entretien

Cette annexe rassemble les questions classiques posées en entretien quant (Squarepoint, QRT, CFM…) sur la formulation pratique du programme de Markowitz. Les réponses sont volontairement courtes : en entretien, ce qui compte est de **comprendre la mécanique** plus que de réciter une formule.

### B.1 Quelle est l'interprétation financière de $\mathbf{1}^\top x = 1$ ? Que se passe-t-il si on l'enlève ?

**La contrainte est la contrainte de budget.** Elle dit que la somme des poids vaut 1, autrement dit que **l'intégralité de la richesse disponible est investie**. Avec $x = (0.6, 0.4)$, je sais que 60% du capital va sur l'actif 1 et 40% sur l'actif 2 — la lecture en pourcentages du portefeuille n'est valide que parce que la somme vaut 1.

**Sans cette contrainte**, plusieurs choses changent simultanément :

- **L'unité change.** Les composantes de $x$ ne sont plus des pourcentages du portefeuille mais des montants ou des expositions absolues. Dire "70% sur l'actif 1" ne veut plus rien dire si la somme totale vaut 5.
- **On peut avoir du leverage et des shorts.** Un vecteur comme $x = (-2, 7)$ devient admissible : on shorte 2 unités de l'actif 1 (vente à découvert) et on prend une exposition longue de 7 unités sur l'actif 2. La somme vaut $5$, ce qui correspond à un portefeuille **leveragé** (on a investi 5× son capital nominal, en utilisant les proceeds du short comme financement).
- **Le programme devient mal posé.** Sans aucune contrainte de taille, le rendement espéré $E[R]^\top x$ peut diverger en augmentant arbitrairement les poids — il n'y a plus de "meilleur portefeuille" bien défini.

> 💡 **Conclusion.** $\mathbf{1}^\top x = 1$ joue deux rôles : elle **normalise** les poids (pour qu'ils s'interprètent en pourcentages) et elle **borne** le problème (sans elle, l'optimum pourrait être à l'infini). En pratique on garde toujours cette contrainte ; c'est sur **d'autres** contraintes (long-only, leverage) qu'on joue pour autoriser ou non les positions courtes.

### B.2 Pourquoi le long-short rend-il le problème encore plus instable ?

Markowitz souffre déjà de la sensibilité aux paramètres ($\mu$ et $\Sigma$ mal estimés → poids instables, voir V.C). Quand on autorise les positions courtes ($x_i$ négatif), ce problème **explose**.

**Le mécanisme.** Sans contrainte long-only, les poids ne sont plus bornés inférieurement : le solveur peut prendre des positions courtes arbitrairement grandes. Une petite erreur d'estimation sur $\mu$ ou $\Sigma$ peut alors basculer le solveur vers une solution complètement différente, du type :

$$x = (-50,\ 2,\ 49)$$

— shorter massivement l'actif 1, prendre une position longue énorme sur l'actif 3, et tout ça est cohérent budgétairement ($-50 + 2 + 49 = 1$). Mais financièrement c'est un **portefeuille hyper leveragé**, instable, et inexploitable en pratique.

> 💡 **L'intuition.** Le long-only impose une borne géométrique : $x \in [0, 1]^n$ avec $\sum x_i = 1$, c'est un simplexe — un ensemble compact petit. Une petite erreur sur $\mu$ déplace l'optimum à l'intérieur de cet ensemble borné, l'effet est limité. Sans long-only, $x$ vit dans $\mathbb{R}^n$ avec une seule contrainte d'égalité — une variété affine non bornée. Une petite erreur sur $\mu$ peut maintenant déplacer l'optimum à l'autre bout de l'espace. C'est le **levier amplificateur** des positions courtes.

**Solutions standards.** Deux contraintes additionnelles permettent de garder un programme long-short tout en bornant l'instabilité :

- **Bornes par actif.** $\ell_i \leq x_i \leq u_i$ (ex. $-5\% \leq x_i \leq 5\%$). Borne directement chaque position individuelle.
- **Contrainte de leverage.** $\|x\|_1 = \sum_i |x_i| \leq L$. Borne la *somme des expositions absolues*, soit longues, soit courtes. C'est la mesure naturelle du levier d'un portefeuille long-short.

> [!warning] Lecture de la contrainte de leverage
> $$\sum_i |x_i| \leq L \;\;:\;\; \begin{cases} L = 1 & \text{long-only pur (toutes positions positives, somme = 1)} \\ L > 1 & \text{long-short autorisé, levier total = } L \end{cases}$$
> 
> En pratique, un fonds de gestion long-short typique aura $L = 1.4$ (fonds **130/30** : 130% long et 30% short, somme nette 100%), $L = 1.5$, ou jusqu'à $L = 2$ pour les hedge funds plus agressifs.

> [!example] Décomposition d'un portefeuille 130/30
> Avec $L = 1.4$, un vecteur admissible serait $x = (+1.2, -0.2, 0)$ :
> 
> - **Position longue totale :** $1.2$ (120% du capital sur l'actif 1).
> - **Position courte totale :** $0.2$ (20% short sur l'actif 2).
> - **Exposition nette :** $1.2 - 0.2 + 0 = 1.0$ (la contrainte budgétaire $\mathbf{1}^\top x = 1$ est satisfaite).
> - **Levier total :** $|1.2| + |0.2| + |0| = 1.4 = L$.
> 
> On respecte $\mathbf{1}^\top x = 1$ (totalement investi) **et** $\|x\|_1 \leq 1.4$ (levier borné).

> 💡 **Reformulation pour solveurs.** Comme pour le turnover (A.3), la valeur absolue dans $\|x\|_1$ peut être linéarisée : on pose $x_i = u_i - v_i$ avec $u_i, v_i \geq 0$ (la partie longue et la partie courte de chaque position), et la contrainte devient $\sum_i (u_i + v_i) \leq L$. Encore une fois, le solveur n'a aucun intérêt à prendre $u_i > 0$ et $v_i > 0$ simultanément, donc à l'optimum $|x_i| = u_i + v_i$ exactement.

---

## Résumé — fil logique

| Étape | Contenu |
|---|---|
| **I** | $u(W)$ concave → aversion au risque → prime de risque $\pi = E(W) - C$ ; vNM : agent rationnel maximise $E[u(W)]$ |
| **II.A** | Hypothèse quadratique → $E[u(W^x)] \propto E[R(x)] - \lambda\,\text{Var}(R(x))$ |
| **II.B** | Diversification : terme croisé $\sigma_{ij}$ négatif réduit la variance sans toucher l'espérance |
| **III** | Frontière efficiente sans actif sans risque ; min variance en $(1/\sqrt{a},\, b/a)$ |
| **IV.A** | Avec actif sans risque : frontière = CML, tangente en $\xi$, portefeuille de Sharpe maximal |
| **IV.B** | Théorème des deux fonds : tout agent mélange $r_f$ et $\xi$, seul $x_0$ varie selon $\lambda$ |
| **V** | Limites : variance symétrique → semi-variance ; sensibilité aux paramètres ; hypothèse gaussienne |
| **A** | Contraintes pratiques : poids, cardinalité (MIP), turnover ($\ell_1$), risque (var/TE/CVaR), concentration (HHI) |
| **B** | Questions d'entretien : interprétation $\mathbf{1}^\top x = 1$ ; instabilité long-short → bornes + contrainte leverage $\|x\|_1 \leq L$ |
