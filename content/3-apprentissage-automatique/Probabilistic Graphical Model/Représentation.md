---
title: Représentation - PGM
---
# Représentation

> Cette note couvre la première des trois parties des PGM : **comment représenter** une distribution de probabilité avec un graphe. On verra deux familles complémentaires : les **Bayesian Networks** (graphes orientés) et les **Markov Random Fields** (graphes non orientés), avec dans chaque cas la dualité fondamentale entre **factorisation** de la distribution et **indépendances** lisibles dans la structure du graphe.

---

## I. Bayesian Networks

### A. L'exemple de l'étudiant

> [!example] Fil rouge — l'exemple de l'étudiant
> Cinq variables aléatoires :
> - **G**rade (note)
> - Course **D**ifficulty (difficulté du cours)
> - Student **I**ntelligence (intelligence de l'étudiant)
> - Student **S**AT (score SAT)
> - Reference **L**etter (lettre de recommandation)

![[im9 (1).png]]
**Figure 1.** Bayesian Network de l'exemple de l'étudiant, avec les CPD (conditional probability distributions) attachées à chaque nœud.

Pour obtenir la **factorisation** du graphe, on applique d'abord la chain rule :

$$P(D, I, G, S, L) = P(D) \, P(I \mid D) \, P(G \mid D, I) \, P(S \mid D, I, G) \, P(L \mid D, I, G, S).$$

Puis on applique les **indépendances conditionnelles** lisibles dans le graphe pour simplifier :

$$P(D, I, G, S, L) = P(D) \, P(I) \, P(G \mid I, D) \, P(S \mid I) \, P(L \mid G).$$

> 💡 C'est exactement un **produit de facteurs**. Chaque CPD est un facteur, et le produit complet donne la jointe.

> [!example] Calcul d'une probabilité jointe
> $$p(d^0, i^1, g^3, s^1, l^1) = 0.6 \times 0.3 \times 0.02 \times 0.01 \times 0.8.$$

### B. Définition formelle

> [!warning] Définition (Bayesian Network)
> Un **Bayesian Network** est un Directed Graphical Model (DGM) $G = (V, E)$ dont les nœuds représentent les variables aléatoires $X_1, \ldots, X_n$. Pour chaque nœud $X_i$, on définit une CPD $p(X_i \mid \text{pa}_G(X_i))$. Le BN représente une distribution jointe via la **chain rule for Bayesian Networks** :
>
> $$p(X_1, X_2, \ldots, X_D) = \prod_{i=1}^D p(X_i \mid \text{pa}(X_i)).$$

> [!note]- Preuve que la distribution est légale (somme à 1)
> On part de la chain rule du BN et on utilise l'astuce que chaque facteur ne dépend que de variables précises, donc on peut faire entrer les sommes :
>
> $$\begin{aligned}
> \sum_{D, I, G, S, L} P(D, I, G, S, L) &= \sum_{D, I, G, S} P(D) P(I) P(G \mid I, D) P(S \mid I) \underbrace{\sum_L P(L \mid G)}_{=1} \\
> &= \sum_{D, I, G} P(D) P(I) P(G \mid I, D) \underbrace{\sum_S P(S \mid I)}_{=1} \\
> &= \sum_{D, I} P(D) P(I) \underbrace{\sum_G P(G \mid I, D)}_{=1} \\
> &= \sum_D P(D) \sum_I P(I) = 1.
> \end{aligned}$$

> [!warning] Définition (Factorisation)
> On dit qu'une probabilité $p$ **factorise** sur un DAG $G$ si elle peut être décomposée en un produit de facteurs, par la chain rule, comme spécifié par $G$ :
>
> $$p(X_1, X_2, \ldots, X_D) = \prod_{i=1}^D p(X_i \mid \text{pa}(X_i)).$$

### C. Patterns de raisonnement

**Causal** signifie qu'on va du haut vers le bas dans le graphe.

**(a) Raisonnement causal** (causal reasoning).

![[im10 (2).png|426]]
**Figure 2.** Raisonnement causal — on propage de la cause vers l'effet.

**(b) Raisonnement évidentiel** (evidential reasoning) — on remonte du bas vers le haut, on demande comment varient les ancêtres de la note.

![[im11 (1).png|431]]
**Figure 3.** Raisonnement évidentiel — observer l'effet change la croyance sur la cause.

**(c) Raisonnement intercausal** (intercausal reasoning) — l'information entre deux causes ayant un effet commun. L'étudiant a un C, et on apprend que ce cours est en fait difficile : la probabilité d'être intelligent **augmente** :

$$p(i^1 \mid g^3, d^1) \approx 0.11.$$

![[im12 (2).png]]
**Figure 4.** Raisonnement intercausal — observer une cause "explique" l'effet et change la croyance sur l'autre cause.

### D. Flot d'influence probabiliste

**Motivation.** On vient de voir des patterns où, intuitivement, l'influence probabiliste **part d'un nœud et flue à travers le graphe** vers un autre. C'est exactement ce qui se passe dans un BN. On va rendre cet argument rigoureux et identifier quelles régions du graphe ne s'influencent pas.

**Question : quand $X$ peut-il influencer $Y$ ?**

- $X \rightarrow Y$ : $X$ est parent de $Y$, donc $X$ influence $Y$.
- $X \leftarrow Y$ : observer $X$ change la distribution de $Y$ (c'est le raisonnement évidentiel). Conditionner sur $X$ change la croyance sur $Y$.
- $X \rightarrow W \rightarrow Y$ : raisonnement causal classique (de la difficulté à la lettre).
- $X \leftarrow W \leftarrow Y$ : symétrique au précédent, donc également valide.
- $X \leftarrow W \rightarrow Y$ : cause commune $W$. Si on observe le SAT, ça change la croyance sur l'intelligence et donc sur la note.
- $X \rightarrow W \leftarrow Y$ : la **V-structure**. C'est la **seule exception** où $X$ n'influence pas $Y$.

![[images/3-Apprentissage automatique/PGM/Représentation/im15.png]]
**Figure 5.** Les six configurations possibles entre $X$, $W$ et $Y$. Seule la V-structure (dernière) bloque par défaut le flot d'influence.

> [!warning] Définition (Active trail)
> Un **trail** est une séquence de nœuds $X_1 - \ldots - X_k$ où $-$ signifie qu'il y a une arête (orientée à gauche ou à droite). Un trail est **actif** si :
>
> - Pour chaque V-structure $X_{i-1} \rightarrow X_i \leftarrow X_{i+1}$ sur le trail, $X_i$ ou un de ses descendants appartient à $Z$ (l'observation **active** la V-structure).
> - Aucun autre $X_i$ n'est dans $Z$ (l'observation **bloque** les autres types de configurations).

> [!example] Exemple d'active trail
> Sur le graphe de l'étudiant, si on conditionne sur Grade, on a l'active trail : SAT → Intelligence → Grade ← Difficulty. Mais on ne peut pas atteindre Letter.

---

## II. Indépendance dans les Bayesian Networks

### A. Préliminaires

> [!warning] Définition (Indépendance)
> Pour des variables aléatoires $X, Y$, on dit que $P$ satisfait $X \perp Y$ si :
> - $P(X, Y) = P(X) P(Y)$
> - $P(X \mid Y) = P(X)$
> - $P(Y \mid X) = P(Y)$

![[images/3-Apprentissage automatique/PGM/Représentation/im13.png]]
**Figure 6.** L'indépendance pure correspond à un graphe sans aucune connexion — cas très rare.

L'indépendance pure se produit très rarement. On va définir une notion bien plus utile : l'**indépendance conditionnelle**.

> [!warning] Définition (Indépendance conditionnelle)
> Pour des (ensembles de) variables aléatoires $X, Y, Z$, on dit que $P$ satisfait $(X \perp Y \mid Z)$ si :
>
> $$\begin{aligned}
> p(X, Y \mid Z) &= p(X \mid Z) \, p(Y \mid Z), \\
> p(X \mid Y, Z) &= p(X \mid Z), \\
> p(Y \mid X, Z) &= p(Y \mid Z), \\
> \textcolor{blue}{p(X, Y, Z)} &\textcolor{blue}{\propto \phi_1(X, Z) \, \phi_2(Y, Z)}.
> \end{aligned}$$

> [!example] Indépendance conditionnelle — exemple des deux pièces
> On a deux pièces : une équilibrée, l'autre biaisée renvoyant pile 90% du temps. *Coin* est la pièce qu'on choisit, et $X_1, X_2$ sont les deux lancers. Sachant la pièce, les deux lancers sont indépendants — sans cette information, ils ne le sont pas.

![[images/3-Apprentissage automatique/PGM/Représentation/im14.png]]
**Figure 7.** Indépendance conditionnelle dans le BN de l'étudiant.

> 💡 **Important.** On a montré que $P(S, G \mid i^0) = P(S \mid i^0) \, P(G \mid i^0)$. Pour vérifier que $S$ et $G$ sont conditionnellement indépendants **sachant $I$**, il faut aussi vérifier l'égalité pour $i^1$. L'énoncé est que $S \perp G \mid I$, pas que $S \perp G \mid i^0$ uniquement — l'indépendance doit tenir **pour toutes les valeurs** de $I$.

### B. Indépendances depuis la structure du graphe

![[bayes1.png]]
**Figure 8.** Trois configurations canoniques sur trois nœuds.

> 💡 **Convention graphique.** Si on conditionne sur un nœud, on le grise. Sur les graphes (i) et (ii) ci-dessus, le nœud B serait colorié.

Les trois cas portent des noms standards :

1. **Head-Tail** ou **Cascade** : $A \to B \to C$
2. **Tail-Tail** ou **Common parent** : $A \leftarrow B \to C$
3. **Head-Head** ou **V-structure** : $A \to B \leftarrow C$

> [!note]- Preuve — Graphe 1 (Cascade)
> Joint :
>
> $$p(A, B, C) = p(A) \cdot p(B \mid A) \cdot p(C \mid B).$$
>
> On réécrit $p(A) \cdot p(B \mid A) = p(A, B) = p(A \mid B) \cdot p(B)$, d'où :
>
> $$p(A, B, C) = p(A \mid B) \cdot p(B) \cdot p(C \mid B).$$
>
> Par Bayes :
>
> $$p(A, C \mid B) = \frac{p(A, B, C)}{p(B)} = p(A \mid B) \cdot p(C \mid B).$$
>
> Donc $\boxed{A \perp C \mid B}$.

> [!note]- Preuve — Graphe 2 (Common parent)
> Joint :
>
> $$p(A, B, C) = p(A \mid C) \cdot p(B \mid C) \cdot p(C).$$
>
> Par Bayes :
>
> $$p(A, C \mid B) = \frac{p(A, B, C)}{p(B)} = p(A \mid B) \cdot p(C \mid B).$$
>
> Donc $\boxed{A \perp C \mid B}$.

> [!note]- Preuve — Graphe 3 (V-structure)
> Joint :
>
> $$p(A, B, C) = p(A) \cdot p(C) \cdot p(B \mid A, C).$$
>
> On voudrait avoir $p(A, C \mid B) = p(A \mid B) \, p(C \mid B)$, mais ce n'est **pas** vrai en général. Donc :
>
> $$\boxed{A \not\perp C \mid B}.$$
>
> Conditionner sur $B$ rend $A$ et $C$ **dépendants** — c'est l'effet *explaining away*.

### C. Le pont factorisation ↔ indépendance

**Motivation.** Une des plus belles propriétés des modèles graphiques est la **connexion intime entre la factorisation** de la distribution comme produit de facteurs et **les indépendances** qu'elle satisfait.

Rappel des deux liens :

$$\begin{aligned}
P(X, Y) &= P(X) P(Y) \quad \Leftrightarrow \quad X \perp Y, \\
P(X, Y, Z) &\propto \phi_1(X, Z) \, \phi_2(Y, Z) \quad \Leftrightarrow \quad X \perp Y \mid Z.
\end{aligned}$$

La factorisation d'une distribution $P$ implique les indépendances qui tiennent dans $P$.

> 💡 **La grande question.** Si $P$ factorise sur $G$, peut-on **lire les indépendances** directement depuis la structure de $G$ ? La réponse est oui, via la d-séparation.

> [!warning] Définition (d-séparation)
> $X$ et $Y$ sont **d-séparés** dans $G$ sachant $Z$ s'il n'y a aucun active trail dans $G$ entre $X$ et $Y$ sachant $Z$. On note :
>
> $$\text{d-sep}_G(X, Y \mid Z).$$

![[bayes3.png]]
**Figure 9.** Illustration de la d-séparation.

> [!example] Calcul de d-séparation
> On fixe $C = \{3\}$ et on cherche tous les $X_i \perp X_j \mid X_3$ depuis le graphe :
>
> ![[bayes2.png|208]]
>
> | $i$ | $j$ | d-sep | $i$ | $j$ | d-sep |
> | :---: | :---: | :---: | :---: | :---: | :---: |
> | 1 | 4 | oui | 4 | 6 | non |
> | 1 | 2 | non | 2 | 7 | oui |
> | 4 | 5 | oui | 2 | 5 | oui |
> | 4 | 7 | non |   |   |   |

> [!warning] Théorème (Factorisation ⟹ Indépendance)
> Si $P$ factorise sur $G$, et si $\text{d-sep}_G(X, Y \mid Z)$, alors $P$ satisfait $(X \perp Y \mid Z)$.

> [!note]- Preuve par l'exemple
> Sur le graphe de l'étudiant ([[#A. L'exemple de l'étudiant|Figure 1]]) :
>
> $$P(D, I, G, S, L) = P(D) P(I) P(G \mid D, I) P(S \mid I) P(L \mid G).$$
>
> On veut montrer que $P(D, S)$ se factorise en $\phi_1(D) \times \phi_2(S)$ (donc que $D \perp S$) :
>
> $$\begin{aligned}
> P(D, S) &= \sum_{G, L, I} P(D) P(I) P(G \mid D, I) P(S \mid I) P(L \mid G) \\
> &= \sum_I P(D) P(I) P(S \mid I) \sum_G \left( P(G \mid D, I) \underbrace{\sum_L P(L \mid G)}_{=1} \right) \\
> &= P(D) \sum_I P(I) P(S \mid I) \underbrace{\sum_G P(G \mid D, I)}_{=1} \\
> &= P(D) \cdot \underbrace{\sum_I P(I) P(S \mid I)}_{= P(S)} \\
> &= \phi_1(D) \times \phi_2(S).
> \end{aligned}$$
>
> *Note.* $\sum_L P(L \mid G) = 1$ et $\sum_G P(G \mid D, I) = 1$ par définition d'une CPD. $\sum_I P(I) P(S \mid I) = P(S)$ par marginalisation.

### D. I-maps

> [!warning] Définition (I-map)
> Si $P$ satisfait $I(G)$, on dit que $G$ est un **I-map** (independency map) de $P$, où :
>
> $$I(G) = \{(X \perp Y \mid Z) : \text{d-sep}_G(X, Y \mid Z)\}.$$

> [!example] Deux graphes, deux distributions
> On regarde deux graphes possibles : $I(G_1) = \{D \perp I\}$ et $I(G_2) = \emptyset$.
>
> - Si $P_1$ satisfait $D \perp I$, alors $G_1$ est un I-map pour $P_1$ et $G_2$ aussi (vacuously).
> - Si $P_2$ ne satisfait **pas** $D \perp I$, alors $G_1$ n'est **pas** un I-map pour $P_2$. $G_2$ reste un I-map (vacuously).

![[im16.png]]
**Figure 10.** Deux graphes candidats $G_1$ et $G_2$, et deux distributions $P_1, P_2$.

> [!warning] Théorème (Factorisation ⟹ I-map)
> Si $P$ factorise sur $G$, alors $G$ est un I-map pour $P$. Autrement dit, on peut **lire dans $G$** les indépendances qui tiennent dans $P$, **indépendamment des paramètres**.

> [!warning] Théorème (I-map ⟹ Factorisation)
> Si $G$ est un I-map pour $P$, alors $P$ factorise sur $G$.

> [!note]- Preuve par l'exemple
> Chain rule de probabilité, puis on simplifie selon les indépendances du graphe :
>
> $$\begin{aligned}
> P(D, I, G, S, L) &= P(D) P(I \mid D) P(G \mid D, I) P(S \mid D, I, G) P(L \mid D, I, G, S) \\
> &= P(D) P(I) P(G \mid D, I) P(S \mid I) P(L \mid G).
> \end{aligned}$$

> 💡 **Bilan.** Deux vues équivalentes de la structure du graphe :
> - **Factorisation** : $G$ permet à $P$ d'être représentée.
> - **I-map** : les indépendances encodées par $G$ tiennent dans $P$.
>
> Si $P$ factorise sur un graphe $G$, on peut lire dans le graphe des indépendances qui doivent tenir dans $P$ (un independency map).

---

## III. Naive Bayes

Une sous-classe particulière des Bayesian Networks est le **Naive Bayes**.

![[bayes5.png]]
**Figure 11.** Structure d'un Naive Bayes : une classe $C$ et des features $X_i$ observées, conditionnellement indépendantes sachant $C$.

L'objectif est d'inférer à quelle classe une instance particulière appartient. Les features $X_i$ sont observées, et on suppose :

$$(X_i \perp X_j \mid C) \quad \text{pour tous } X_i, X_j.$$

La chain rule dans ce contexte donne :

$$P(C, X_1, \ldots, X_n) = P(C) \prod_{i=1}^n P(X_i \mid C).$$

Pour comprendre ce modèle, on regarde le **ratio** entre deux classes pour une assignation donnée :

$$\frac{P(C = c^1 \mid x_1, \ldots, x_n)}{P(C = c^2 \mid x_1, \ldots, x_n)} = \frac{P(C = c^1)}{P(C = c^2)} \prod_{i=1}^n \frac{P(x_i \mid C = c^1)}{P(x_i \mid C = c^2)}.$$

> 💡 **Lecture.** Deux termes :
> - Le **ratio des priors** des deux classes.
> - Le **produit des odd ratios** : la probabilité de voir l'observation $x_i$ dans une classe vs l'autre.

---

## IV. Markov Random Fields (MRF)

### A. Rappels de théorie des graphes

> [!warning] Définition (Clique)
> Un ensemble $C$ est une **clique** du graphe $G$ ssi $C \subseteq V(G)$ et $\forall u, v \in C$ avec $u \neq v$, on a $uv \in E(G)$ (toutes les paires sont connectées).

> [!example] Cliques sur un graphe
> ![[graph1.png]]
>
> Les ensembles $C_1 = \{a, b, c, f\}$ et $C_2 = \{b, c, f\}$ sont des cliques. Mais $S = \{c, d, f, e\}$ n'en est pas une.

> [!warning] Définition (Clique maximale)
> Une **clique maximale** est une clique qu'on ne peut plus étendre.

> [!example] Clique maximale
> ![[graph2.png]]
>
> $\{a, b, c\}$ est une clique mais pas maximale : on peut ajouter $f$ adjacent à tous les autres et garder une clique. La clique maximale est $\{a, b, c, f\}$.

### B. Pairwise Markov Networks

> [!warning] Définition (Pairwise Markov Network)
> Un **pairwise Markov Network** est un graphe non orienté dont les nœuds sont $X_1, \ldots, X_n$ et où chaque arête $X_i - X_j$ est associée à un facteur (potentiel) $\phi_{ij}(X_i, X_j)$.

> [!example] Fil rouge — l'exemple de la "Misconception"
> Quatre étudiants travaillent en binômes. Charles et Alice ne se parlent pas. La variable aléatoire est : *l'étudiant a-t-il une mauvaise compréhension d'un point du cours ?*
>
> ![[markov5.png]]

**Comment paramétrer ?** Comme on n'a pas de conditionnement, on utilise la notion générale de facteur — aussi appelé *affinity function*, *compatibility function* ou *soft constraints*. La probabilité jointe est :

$$P(a, b, c, d) = \frac{1}{Z} \phi_1(a, b) \cdot \phi_2(b, c) \cdot \phi_3(c, d) \cdot \phi_4(d, a),$$

avec la **fonction de partition** :

$$Z = \sum_{a, b, c, d} \phi_1(a, b) \cdot \phi_2(b, c) \cdot \phi_3(c, d) \cdot \phi_4(d, a).$$

![[markov7.png]]
**Figure 12.** Les quatre tables de facteurs pour l'exemple Misconception.

On note la probabilité $p_\Phi(A, B)$ avec $\Phi = \{\phi_1, \phi_2, \phi_3, \phi_4\}$.

> 💡 **Important : pas de mapping naturel facteur-probabilité.** Si on regarde $\phi_1$, les valeurs les plus élevées ne correspondent pas au couple où la probabilité jointe est maximale. Visuellement, $(B, C)$ et $(A, D)$ "aiment être d'accord", mais $(C, D)$ pas. On ne peut donc pas avoir tout le monde d'accord en cycle — il faut casser quelque part. **Contrairement aux BN, il n'existe pas de mapping naturel entre la distribution et les facteurs du graphe.**

### C. Distribution de Gibbs (cas général)

**Motivation.** Le pairwise est limité — peut-il représenter n'importe quelle distribution ? Non : $O(d^n) > O(n^2 d^2)$, le pairwise n'est pas assez expressif. D'où la **distribution de Gibbs**.

> [!warning] Définition (Distribution de Gibbs)
> Une distribution $P_\Phi$ est une **distribution de Gibbs** paramétrée par un ensemble de facteurs $\Phi = \{\phi_1(D_1), \ldots, \phi_K(D_K)\}$ si :
>
> $$P_\Phi(X_1, \ldots, X_n) = \frac{1}{Z} \tilde P_\Phi(X_1, \ldots, X_n),$$
>
> avec
>
> $$\tilde P_\Phi(X_1, \ldots, X_n) = \phi_1(D_1) \times \phi_2(D_2) \times \cdots \times \phi_m(D_m)$$
>
> la mesure non normalisée (les $D_i$ peuvent être des paires, triplets, quadruplets — en fait des cliques) et
>
> $$Z = \sum_{X_1, \ldots, X_n} \tilde P_\Phi(X_1, \ldots, X_n)$$
>
> la **fonction de partition** (constante de normalisation).

> [!example] Markov Network induit par une distribution de Gibbs
> Considérons deux facteurs $\textcolor{blue}{\phi_1(A, B, C)}$ et $\textcolor{red}{\phi_2(B, C, D)}$. Quelles arêtes le Markov Network doit-il avoir ? Intuitivement, des arêtes entre toutes les variables d'un même facteur.
>
> ![[markov8.png]]

Le Markov Network induit, noté $H_\Phi$, a une arête $X_i - X_j$ chaque fois qu'il existe un facteur $\phi \in \Phi$ tel que $X_i, X_j \in \text{Scope}(\phi)$.

> [!warning] Définition (Factorisation sur un MRF)
> On dit qu'une distribution $P_\Phi$ avec $\Phi = \{\phi_1(D_1), \ldots, \phi_K(D_K)\}$ **factorise** sur un Markov Network $\mathcal{H}$ si chaque $D_k$ est un **sous-graphe complet** de $\mathcal{H}$ (i.e. une clique).

![[markov9.png]]
**Figure 13.** Plusieurs distributions de Gibbs peuvent induire le même graphe $\mathcal{H}$.

Sur le graphe ci-dessus, plusieurs distributions de Gibbs induisent toutes ce $\mathcal{H}$ :

- $\phi_1(A, B, D), \phi_2(B, C, D)$
- $\phi_1(A, B), \phi_2(B, C), \phi_3(C, D), \phi_4(A, D), \phi_5(B, D)$
- $\phi_1(A, B, D), \phi_2(B, C), \phi_3(C, D)$

> 💡 **Important.** Cela nous dit qu'on **ne peut pas lire la factorisation** depuis le graphe d'un MRF. Plusieurs paramétrisations différentes peuvent toutes correspondre au même graphe.

**Flot d'influence.** Bien que les paramétrisations soient différentes, **les trails dans le graphe sont les mêmes** quelle que soit la factorisation choisie.

![[markov10.png]]
**Figure 14.** Le flot d'influence est identique pour différentes factorisations.

> [!warning] Définition (Active trail dans un MRF)
> Un trail $X_1 - \ldots - X_n$ est **actif sachant $Z$** si aucun $X_i$ n'est dans $Z$.

> 💡 **Différence Bayesian vs Markov.** La principale différence concerne la **V-structure** : elle serait **bloquée** dans un MRF (parce qu'on n'a pas de notion de descendance), mais **activée par observation** dans un BN.

### D. Conditional Random Fields (CRF)

**Motivation.** Variant des MRF particulièrement utile pour la **prédiction task-specific** : on a un ensemble de variables d'**entrée** $X$ et un ensemble de variables **cibles** $Y$.

Cas d'usage :
- **Image** : $X$ = pixels et features traitées, $Y$ = classe pour chaque pixel (herbe, vache, eau...).
- **Texte** : $X$ = mots dans une phrase, $Y$ = labels (personne, lieu, organisation...).

**Le problème des features corrélées.** Imaginons qu'on veuille prédire $C_i$ = label d'un super-pixel. Les features sont très corrélées entre elles (histogrammes de texture, etc.). Avec un Naive Bayes, on a cinq features très corrélées qu'il faudrait modéliser ensemble.

![[markov11.png]]
**Figure 15.** Naive Bayes appliqué à la prédiction de label de pixel — les features sont corrélées.

Une solution serait d'ajouter beaucoup d'arêtes pour capturer ces corrélations, ce qui mène à des modèles très denses et difficiles à spécifier.

> 💡 **Solution radicalement différente.** On ne se soucie **pas** de la probabilité de voir un pixel vert à côté d'un autre pixel vert. Tout ce qui nous intéresse, c'est la distribution sur $Y$. Au lieu de modéliser $P(X, Y)$, on modélise directement $P(Y \mid X)$ — sans essayer de capturer la distribution sur $X$.

> [!warning] Définition (Conditional Random Field)
> Un **CRF** est un graphe non orienté $\mathcal{H}$ dont les nœuds correspondent à $X \cup Y$. Le réseau est annoté avec un ensemble de facteurs $\phi_1(D_1), \ldots, \phi_m(D_m)$ tels que chaque $D_i \nsubseteq X$ (chaque facteur touche au moins une variable cible). Le réseau encode une distribution **conditionnelle** :
>
> $$\begin{aligned}
> P(Y \mid X) &= \frac{1}{Z(X)} \tilde P(Y, X), \\
> \tilde P(Y, X) &= \prod_{i=1}^m \phi_i(D_i), \\
> Z(X) &= \sum_Y \tilde P(Y, X).
> \end{aligned}$$
>
> Deux variables sont connectées par une arête (non orientée) chaque fois qu'elles apparaissent ensemble dans le scope d'un facteur.

#### D.1 CRF et régression logistique

> 💡 **Connexion fondamentale.** La régression logistique **est** un cas particulier de CRF avec une cible binaire $Y$ et des features $X_i$ binaires.

Cas binaire avec :

$$\phi_i(X_i, Y) = \exp\big\{w_i \cdot \mathbf{1}\{X_i = 1, Y = 1\}\big\}.$$

On a :

- $\phi_i(X_i, Y = 1) = \exp\{w_i X_i\}$
- $\phi_i(X_i, Y = 0) = 1$

D'où la mesure non normalisée :

- $\tilde P_\Phi(X, Y = 1) = \exp\!\left\{\sum_i w_i X_i\right\}$
- $\tilde P_\Phi(X, Y = 0) = 1$

Et la probabilité normalisée :

$$P_\Phi(Y = 1 \mid X) = \frac{\exp\!\left\{\sum_i w_i X_i\right\}}{1 + \exp\!\left\{\sum_i w_i X_i\right\}}.$$

C'est exactement la **fonction sigmoïde** de la régression logistique.

![[markov12.png]]
**Figure 16.** Régression logistique vue comme un CRF avec cible $Y$ et features $X_1, \ldots, X_n$.

> 💡 **Important.** Comme on modélise une distribution **conditionnelle**, on n'aboutit pas à un Bayesian Network. Le CRF **a supprimé toutes les corrélations entre les $X_i$** — c'est précisément ce qui le distingue d'un modèle génératif.

---

## V. Indépendance dans les Markov Networks

### A. Séparation et factorisation

On définit une notion analogue à la d-séparation, mais pour les MRF.

> [!warning] Définition (Séparation)
> $X$ et $Y$ sont **séparés** dans $\mathcal{H}$ sachant $Z$ s'il n'y a aucun active trail dans $\mathcal{H}$ entre $X$ et $Y$ sachant $Z$.

> [!example] Exemple de séparation
> ![[markov13.png]]
>
> Sur ce graphe, $A$ et $E$ sont séparés sachant $\{B, D\}$. Aussi sachant $D$. Aussi sachant $B$.

> [!warning] Théorème (Factorisation ⟹ Indépendance)
> Si $P$ factorise sur $\mathcal{H}$, et si $\text{sep}_\mathcal{H}(X, Y \mid Z)$, alors $P$ satisfait $(X \perp Y \mid Z)$.

On définit alors :

$$I(\mathcal{H}) = \{(X \perp Y \mid Z) : \text{sep}_\mathcal{H}(X, Y \mid Z)\}.$$

Si $P$ satisfait $I(\mathcal{H})$, on dit que $\mathcal{H}$ est un **I-map** de $P$.

> [!warning] Théorème (Factorisation ⟹ I-map)
> Si $P$ factorise sur $\mathcal{H}$, alors $\mathcal{H}$ est un I-map de $P$.

> [!warning] Théorème (Hammersley-Clifford)
> Pour une distribution **strictement positive** $P$, si $\mathcal{H}$ est un I-map de $P$, alors $P$ factorise sur $\mathcal{H}$.

### B. I-maps minimaux et perfect maps

On a montré que la structure du graphe encode un ensemble d'indépendances qui tiennent nécessairement dans toute distribution représentable. Question inverse : prendre une distribution avec un certain ensemble d'indépendances et l'encoder dans un graphe — à quel point ce graphe peut-il **capturer** ces indépendances ?

On définit :

$$\boxed{I(P) = \{(X \perp Y \mid Z) : P \text{ satisfait } (X \perp Y \mid Z)\}.}$$

Toute indépendance qui tient dans $G$ (par d-séparation) tient aussi dans $P$ :

$$\boxed{I(G) \subseteq I(P).}$$

> [!warning] Définition (I-map minimal)
> Un I-map **sans arêtes redondantes**.

> [!example] I-map minimal — exemple simple
> Si on a un graphe $X \to Y$ où $P(Y \mid x^0) = P(Y \mid x^1)$, on peut retirer cette arête et avoir encore un I-map. Donc ce n'est **pas** un I-map minimal.

> [!example] I-map minimal sur l'exemple de l'étudiant
> ![[markov14.png]]
>
> **(i) Graphe de gauche.** $D, I, G$ — c'est un I-map minimal pour le cas où $D \perp I$.
>
> **(ii) Graphe de droite.** Vérifions qu'on ne peut retirer aucune arête :
> - Si on retire $D-G$, ça impliquerait $D \perp G$ — faux.
> - Si on retire $G-I$, ça impliquerait $G \perp I \mid D$ — faux.
> - Si on retire $D-I$, ça impliquerait $D \perp I \mid G$ — faux.
>
> Donc le graphe de droite est minimal.

> [!warning] Définition (Perfect map)
> Un graphe $G$ est un **perfect map** de $P$ si :
>
> $$I(G) = I(P).$$
>
> Autrement dit, $G$ capture **exactement** les indépendances de $P$.

> [!example] Distribution sans Bayesian perfect map
> Soit $P$ représentée par le pairwise Markov Network ci-dessous. On sait que $P$ satisfait $A \perp C \mid B, D$ et $B \perp D \mid A, C$.
>
> ![[markov15.png]]
>
> Essayons de l'encoder par un BN :
> - **(i)** Le BN à gauche fait l'hypothèse $B \perp D \mid A$, qui n'est pas dans $P$ — donc ce n'est même pas un I-map.
> - **(ii)** Le BN du milieu donne $B \perp D \mid A$ mais aussi $A \perp C$, qui n'est pas vraie dans $P$.
> - **(iii)** Le BN de droite est un I-map mais ne capture que $A \perp C \mid B, D$ — donc $I(G) \subset I(P)$, ce n'est pas un perfect map.
>
> **Conclusion** : pas de perfect map BN pour cette distribution.

> [!example] Distribution sans Markov perfect map
> Inversement, certaines distributions ont un BN comme perfect map mais pas de MRF.
>
> ![[markov16.png]]
>
> Pour avoir un I-map MRF, il faut ajouter $D-G$ et $I-G$, mais alors le candidat impliquerait $D \perp I \mid G$ — qui n'est pas vrai. Le seul I-map MRF est le triangle $D, I, G$ complet, mais on perd alors **toutes** les indépendances.

### C. I-équivalence

**Unicité du perfect map.** Sur le graphe ci-dessous, $G_1$ n'a aucune indépendance encodée, et $G_2$ a les arêtes inversées. Ce sont deux graphes distincts avec exactement les mêmes propriétés d'indépendance — ils peuvent représenter exactement le même ensemble de distributions.

![[markov17.png]]
**Figure 17.** Deux graphes structurellement différents avec $I(G_1) = I(G_2)$.

> [!warning] Définition (I-équivalence)
> Deux graphes $G_1$ et $G_2$ sur les mêmes variables $X_1, \ldots, X_n$ sont **I-équivalents** si $I(G_1) = I(G_2)$.

> [!example] I-équivalence sur trois variables
> Sur trois variables, parmi les graphes possibles, **toutes les configurations sauf la V-structure** sont I-équivalentes (cascade dans les deux sens et common parent ont les mêmes indépendances : $A \perp C \mid B$).
>
> ![[markov18.png]]

> 💡 **Pourquoi c'est important.** L'I-équivalence signifie que **certains aspects des modèles graphiques sont non identifiables**. Plusieurs graphes peuvent être équivalents si on n'a pas de connaissance a priori sur le sens des arêtes — c'est crucial en *causal discovery* (cf. [[SCM Pearl]] et le PC algorithm).

---

## Annexe — Bilan des théorèmes

| Type de graphe | Théorème | Sens |
| :--- | :--- | :--- |
| BN | Factorisation ⟹ I-map | Si $P$ factorise sur $G$, on lit les indépendances dans $G$ |
| BN | I-map ⟹ Factorisation | Si les indépendances de $G$ tiennent dans $P$, alors $P$ factorise |
| MRF | Factorisation ⟹ Séparation ⟹ Indépendance | Idem côté MRF |
| MRF | Hammersley-Clifford : I-map ⟹ Factorisation | Sous l'hypothèse de positivité stricte de $P$ |

> 💡 **Le résumé en une phrase.** Dans les deux familles (BN et MRF), il y a une **dualité fondamentale** entre la factorisation de la distribution et les indépendances qu'on peut lire dans le graphe — c'est ce qui rend les modèles graphiques aussi puissants pour modéliser des distributions complexes de façon tractable.
