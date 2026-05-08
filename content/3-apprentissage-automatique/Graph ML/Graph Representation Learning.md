---
title: Graph Representation Learning
---
# Graph Representation Learning

> Cette note couvre l'**apprentissage de représentations de nœuds et de graphes** : passer d'un graphe brut à des **embeddings vectoriels** qu'on peut ensuite utiliser dans n'importe quel pipeline ML standard (classification, clustering, link prediction). On voit le framework général **encoder/decoder**, les méthodes par **random walks** (DeepWalk, node2vec), une application aux **knowledge graphs** (TransE), et enfin l'embedding de **graphes entiers**.

> 💡 **Position dans le cours.** Suite logique de [[Message Passing]] : là on faisait de la classification de nœuds *à la main* (relational, iterative, BP). Ici on **apprend une représentation** une fois pour toutes, indépendamment d'une tâche. Les embeddings résultants peuvent ensuite servir pour **n'importe quelle** tâche downstream.

---

## I. Introduction

### A. Pourquoi des embeddings de graphes ?

> [!warning] Définition (Graph Representation Learning)
> On veut **apprendre automatiquement les features** plutôt que les construire à la main comme en ML basique. À partir d'un nœud, on génère un embedding via une transformation $f$.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im1.png]]
**Figure 1.** Du graphe à l'embedding : une transformation $f$ produit un vecteur dense par nœud.

**Le pipeline.** On part d'un grand graphe, on construit la **matrice d'adjacence** — qui est très **creuse et grande**, donc coûteuse à manipuler directement. On la mappe vers un **espace d'embedding** où chaque colonne capture un aspect du graphe (degré, motifs...). Comme c'est une dimension latente, on ne peut pas dire exactement quelle colonne correspond à quelle information.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im2.png]]
**Figure 2.** Pipeline : graphe → matrice d'adjacence (creuse) → embedding (dense, dimension réduite).

> [!example] Zachary's Karate Club
> Embeddings 2D des nœuds du fameux *Karate Club* de Zachary. Les nœuds similaires sont proches dans l'espace d'embedding.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im3 (1).png]]

### B. Pourquoi le ML classique ne marche pas sur les graphes

Pour une **image**, on pourrait construire un graphe à partir du dataset MNIST, mais ce graphe aurait beaucoup trop de **régularité** pour être représentatif d'un vrai graphe. Pour un **RNN**, la représentation textuelle est une chaîne, qui ne capture pas le même genre d'info qu'un graphe arbitraire.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im4.png]]
![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im5.png]]
**Figure 3.** Image et texte ont des structures **régulières** (grille / chaîne) — pas représentatives des graphes réels.

**Cas CNN sur graphe.** Un nœud noir = présent, blanc = absent. Quatre fenêtres convolutionnelles sur ce treillis prennent en compte les nœuds manquants — mais ce n'est pas comme ça qu'un graphe fonctionne. **On ne peut pas fixer un ordre.**

> 💡 **Le problème d'isomorphisme.** On peut prendre un graphe, permuter ses lignes et colonnes dans la matrice d'adjacence, et la **structure topologique reste la même** — ce qui n'est absolument pas le cas pour du texte ou des images. Encore pire : un graphe peut être **dynamique** (nœuds/arêtes qui apparaissent et disparaissent), avec des features multimodales, etc. D'où le besoin d'un nouveau modèle.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im6.png]]

---

## II. Node Embeddings — Encoder et Decoder

### A. Framework général

**Setup.** Soit un graphe $G = (V, E)$ avec ensemble de sommets $V$, ensemble d'arêtes $E$, et une matrice d'adjacence $A$ (binaire pour simplifier — pas de features de nœuds ni d'info supplémentaire).

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/1.Introduction/im1.png]]

**Objectif :**

$$\text{similarity}(u, v) \approx \mathbf{z}_v^T \mathbf{z}_u.$$

La **similarité** est dans le graphe original, le **produit scalaire** est dans l'espace d'embedding.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im7.png]]
**Figure 4.** La similarité dans le graphe doit être préservée par le produit scalaire des embeddings.

> [!warning] Pseudo-code (Node Embeddings)
> 1. **Encoder** $\text{ENC}(v) = \mathbf{z}_v$ : mappe les nœuds vers les embeddings ($v$ nœud d'entrée, $\mathbf{z}_v$ embedding $d$-dim). L'approche la plus simple : un **embedding lookup** :
>
> $$\text{ENC}(v) = \mathbf{z}_v = \mathbf{Z} \cdot v$$
>
> où $\mathbf{Z} \in \mathbb{R}^{d \times |V|}$ est une matrice (chaque colonne = embedding d'un nœud, **paramètres à apprendre**) et $v \in \mathbb{I}^{|V|}$ est le vecteur indicateur du nœud (one-hot).
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im8.png]]
>
> 2. **Similarity function** : spécifie comment les relations dans l'espace vectoriel correspondent aux relations dans le graphe :
>
> $$\text{similarity}(u, v) \approx \mathbf{z}_v^T \mathbf{z}_u.$$
>
> 3. **Decoder** $\text{DEC}$ : mappe les embeddings vers le score de similarité.
> 4. **Optimisation** : optimiser les paramètres de l'encoder pour que la similarité dans le graphe et dans l'espace d'embedding coïncident.

### B. Remarques importantes

> 💡 **Choix clé : la fonction de similarité.** Deux nœuds doivent-ils avoir des embeddings similaires si :
> - ils sont **connectés** ?
> - ils partagent des **voisins** ?
> - ils ont des **rôles structurels** similaires ?
>
> On va voir une définition de similarité basée sur les **random walks** et comment optimiser pour cette mesure.

> 💡 **C'est de l'apprentissage non supervisé / self-supervised.**
> - On n'utilise **pas** les labels des nœuds.
> - On n'utilise **pas** les features des nœuds.
> - L'objectif est d'estimer directement les coordonnées (l'embedding) d'un nœud de sorte que **un aspect de la structure du graphe** (capturé par DEC) soit préservé.
>
> Conséquence : ces embeddings sont **task-independent** — ils ne sont pas entraînés pour une tâche spécifique, mais peuvent servir pour n'importe quelle tâche.

---

## III. Approches par Random Walks

### A. DeepWalk

> 💡 **Référence.** [DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf) (Bryan et al., juin 2014).

**Notation.**
- $\mathbf{z}_u$ : l'embedding du nœud $u$ (objet à trouver).
- $P(v \mid \mathbf{z}_u)$ : probabilité (prédite) de visiter le nœud $v$ lors d'un random walk démarré en $u$.

> [!warning] Définition (Random Walk)
> Étant donné un graphe et un point de départ, on sélectionne un voisin au hasard, on s'y déplace, puis on recommence. La séquence de points visités ainsi est un **random walk** sur le graphe.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im1.png]]

> [!warning] Définition (Similarity Function pour Random-Walk Embedding)
> La similarité est définie comme la **probabilité que $u$ et $v$ co-occurrent sur un random walk** :
>
> $$\text{similarity}(u, v) \approx \mathbf{z}_v^T \mathbf{z}_u.$$

#### A.1 Pseudo-code

> [!note]- Algorithme général
> 1. Estimer la probabilité de visiter $v$ depuis $u$ via une stratégie de random walk $R$.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im2.png]]
>
> 2. Optimiser les embeddings pour encoder ces statistiques de random walks : la similarité dans l'espace d'embedding (produit scalaire = $\cos \theta$) encode la similarité par random walk.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im3 (1).png]]

> [!warning] Avantages des Random-Walk Embeddings
> - **Expressivité** : définition stochastique flexible de la similarité de nœuds qui incorpore l'info de **voisinage local et d'ordre supérieur**. Si un random walk depuis $u$ visite $v$ avec haute probabilité, $u$ et $v$ sont similaires (info multi-hop).
> - **Efficacité** : pas besoin de considérer toutes les paires de nœuds — seulement celles qui co-occurrent sur des random walks.

#### A.2 Pseudo-code détaillé

Étant donné $G = (V, E)$, l'objectif est d'apprendre une fonction $f : u \to \mathbb{R}^d$, $f(u) = \mathbf{z}_u$.

**(i) Random Walks.** Lancer des random walks courts de longueur fixe depuis chaque nœud $u$ avec une stratégie $R$.

**(ii) Collecter le multiset.** Pour chaque nœud $u$, collecter $N_R(u)$ — le **multiset** des nœuds visités lors des random walks démarrés en $u$ (les nœuds peuvent y apparaître plusieurs fois).

**(iii) Optimisation.** Étant donné $u$, prédire ses voisins $N_R(u)$. L'objectif log-vraisemblance :

$$\max_f \sum_{u \in V} \log P(N_R(u) \mid \mathbf{z}_u).$$

Forme équivalente :

$$\mathcal{L} = \sum_{u \in V} \sum_{v \in N_R(u)} -\log P(v \mid \mathbf{z}_u).$$

On paramétrise $P(v \mid \mathbf{z}_u)$ par un softmax :

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im4.png]]

> 💡 **Problème.** La double somme sur les nœuds donne une complexité $O(|V|^2)$. Le coupable : le **terme de normalisation** du softmax. Peut-on l'approcher ?

#### A.3 Negative Sampling (cf. [[(i) Word Embeddings]])

Au lieu de normaliser sur tous les nœuds, on normalise contre $k$ "**negative samples**" $n_i$ tirés au hasard :

$$\log\left(\frac{\exp(\mathbf{z}_u^T \mathbf{z}_v)}{\sum_{n \in V} \exp(\mathbf{z}_u^T \mathbf{z}_n)}\right) \approx \log \sigma(\mathbf{z}_u^T \mathbf{z}_v) - \sum_{i=1}^k \log \sigma(\mathbf{z}_u^T \mathbf{z}_{n_i}), \quad n_i \sim P_V.$$

où $P_V$ est une distribution sur les nœuds.

**Choix de $k$.** Échantillonner $k$ nœuds négatifs avec proba proportionnelle au degré. Deux considérations :
- **Plus $k$ est grand**, plus l'estimation est robuste.
- **Plus $k$ est grand**, plus le biais sur les events négatifs augmente.
- En pratique : $k = 5$ à $20$.

> 💡 **Lien direct avec Word2Vec.** C'est exactement le même negative sampling que dans [[(i) Word Embeddings]] section IV.B — le graphe joue le rôle du corpus, les random walks le rôle des phrases, les nœuds le rôle des mots.

#### A.4 Stochastic Gradient Descent

> [!note]- Algorithme SGD
> 1. Initialiser $z_i$ aléatoirement pour tout $i$.
> 2. Itérer jusqu'à convergence avec $\mathcal{L}^{(u)} = \sum_{v \in N_R(u)} -\log P(v \mid \mathbf{z}_u)$ :
>    - Échantillonner un nœud $i$, calculer $\partial \mathcal{L}^{(i)} / \partial z_j$ pour tout $j$.
>    - Update : $z_j \leftarrow z_j - \eta \cdot \partial \mathcal{L}^{(i)} / \partial z_j$.

#### A.5 L'algorithme DeepWalk en pratique

DeepWalk utilise **Skip-Gram** pour créer les embeddings. Première étape : **générer un corpus** depuis le graphe.

**Comment ?** Faire plusieurs random walks de longueur fixe depuis chaque nœud. Par exemple, voici le résultat de 1 random walk de longueur 20 sur chaque nœud d'un graphe :

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im9.png|411]]

La $i$-ème ligne = random walk démarré en nœud $i$. Par exemple, la 8e ligne :

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im10 (1).png]]

Lecture : le walk a démarré en 8, est allé en 9, puis 10, est revenu en 9, est retourné en 10, puis en 11, etc. jusqu'à 9.

> 💡 **L'analogie NLP.** On a 13 "phrases" de 20 "mots" chacune. Vocabulaire de taille 13. On peut appliquer **Skip-Gram exactement comme en NLP** : fenêtre autour d'un nœud (= ses voisins dans le walk) → embeddings.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im6.png]]
**Figure 5.** Skip-Gram appliqué sur les random walks d'un graphe.

**Visualisation.** Embeddings projetés en 2D via PCA. On voit clairement la **structure communautaire** du graphe — deux groupes émergent.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im7.png|462]]

### B. node2vec

> 💡 **Référence.** [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) (Grover & Leskovec, août 2016). Très similaire à DeepWalk — la différence principale est dans le **random walk biaisé** qu'ils proposent (*Biased Walks*).

#### B.1 Biased Walks — flexibilité local/global

> [!warning] Définition (node2vec — Biased Walks)
> Random walks **biaisés flexibles** qui font le compromis entre vue **locale** et vue **globale** du réseau. On définit le voisinage $N_R(u)$ avec deux stratégies classiques : **BFS** et **DFS**.

> [!example] Walk de longueur 3
> - $N_{BFS}(u) = \{s_1, s_2, s_3\}$ — vue locale microscopique.
> - $N_{DFS}(u) = \{s_4, s_5, s_6\}$ — vue globale macroscopique.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im12.png]]

#### B.2 Random walk de premier ordre

Walker en $v$ avec voisins $u_1, u_2, u_3$ (avec poids différents). Probabilités de transition :

$$p(u \mid v) = \frac{w(u, v)}{\sum_{u' \in \mathcal{N}_v} w(u', v)} = \frac{w(u, v)}{d(v)}$$

où $\mathcal{N}_v$ est l'ensemble des voisins de $v$ et $d(v)$ son degré.

> [!example] Calcul concret
> $p(u_3 \mid v) = 0.2 / 3 \approx 0.07$.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/3.Node2vec/im1.png]]

#### B.3 Random walk de second ordre (biaisé)

> [!warning] Définition (Second-order biased walk)
> Au lieu de regarder seulement les voisins directs de l'état courant, on applique un **facteur de biais $\alpha$** qui repondère les arêtes selon l'état précédent :
>
> $$\alpha_{pq}(t, x) = \begin{cases} \frac{1}{p} & \text{si } d_{tx} = 0 \\ 1 & \text{si } d_{tx} = 1 \\ \frac{1}{q} & \text{si } d_{tx} = 2 \end{cases}$$

**Lecture.** En partant de $s_1$, on calcule la distance aux autres nœuds. Distance 1 ($s_1$ à $s_2$) → proba 1. Retour à $s_1$ depuis $w$ → toujours $1/p$. Distance > 1 → proba $1/q$. Le paramètre $q$ aide à choisir entre **DFS** et **BFS**.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im5.png]]

Les probabilités de transition de second ordre :

$$p(u \mid v, t) = \frac{\alpha_{pq}(t, u) \, w(u, v)}{\sum_{u' \in \mathcal{N}_v} \alpha_{pq}(t, u') \, w(u', v)}.$$

Mêmes formules que pour le premier ordre, avec en plus le terme de biais $\alpha$.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/3.Node2vec/im2.png]]

> [!example] Illustration
> Le random walk vient de traverser l'arête $(s_1, w)$ et est en $w$. Le biased walk étant **second-order**, il se souvient toujours d'où il vient (= $s_1$).
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/2.Deep_walk/im5.png]]
>
> Où aller ensuite ? Trade-off entre :
> - **$p$** — *return parameter*.
> - **$q$** — *walk-away parameter*.
>
> Walk type **BFS** → $p$ faible. Walk type **DFS** → $q$ faible.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im13.png]]

#### B.4 Algorithme node2vec

> [!note]- Pseudo-code
> 1. Calculer les probabilités de random walk.
> 2. Simuler $r$ random walks de longueur $l$ depuis chaque nœud $u$.
> 3. Optimiser l'objectif node2vec via **SGD avec Skip-Gram** (exactement comme DeepWalk).

> 💡 **Complexité linéaire** et **les trois étapes sont individuellement parallélisables**.

### C. Récapitulatif

> [!warning] Définitions de similarité de nœuds
> - **Naïf** : similaire si 2 nœuds sont connectés.
> - **Neighborhood overlap** (jaccard, Adamic-Adar...).
> - **Random walk approaches** (DeepWalk, node2vec).

---

## IV. Comment utiliser les embeddings

Trois types d'applications principales (+ une).

> [!warning] Clustering / Community detection
> On clusterise les points $z_i$.

> [!warning] Node classification
> Prédire le label $f(z_i)$ d'un nœud à partir de son embedding $z_i$. $f$ peut être un arbre de décision ou n'importe quel algo supervisé entraîné sur le training set, puis appliqué sur les nœuds inconnus.

> [!warning] Link prediction
> Prédire l'arête $(i, j)$ à partir d'une fonction $f(z_i, z_j)$ — on peut concaténer, faire un produit, prendre une distance :
> - **Concatenate** : $f(z_i, z_j) = g([z_i, z_j])$
> - **Hadamard** (produit coordonnée par coordonnée) : $f(z_i, z_j) = g(z_i \odot z_j)$
> - **Sum/Avg** : $f(z_i, z_j) = g(z_i + z_j)$
> - **Distance** : $f(z_i, z_j) = g(\|z_i - z_j\|_2)$

> [!example] Link prediction
> On veut savoir si les arêtes marquées "?" devraient exister. L'algo répond oui/non pour chacune.
>
> ![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im14.png]]

> [!warning] Graph classification
> Embedding du graphe entier $z_G$ via agrégation d'embeddings de nœuds ou via *anonymous random walks*. Prédire un label sur la base de $z_G$.

---

## V. Application aux Knowledge Graphs — TransE

> 💡 **Référence.** [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) (Bordes et al., 2013).

### A. Vocabulaire spécifique aux KG

Dans un **Knowledge Graph (KG)**, les arêtes sont de **types différents** — appelées **relations**. Les nœuds sont appelés **entities**. Exemples d'arêtes : *genre dans un livre*, *relation entre deux auteurs*, etc.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im15.png]]

### B. Link prediction dans les KG (KG Completion)

Construire un KG est laborieux : si une arête manque, le résultat de recherche est impacté. On veut un modèle de link prediction qui apprend depuis les **patterns de connectivité locale et globale**, en tenant compte des entités et relations de différents types **simultanément**.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im16.png]]

### C. Translating Embeddings (TransE)

Les relations entre entités sont représentées comme des **triplets** :

$$(h, l, t) \;:\; h \text{ (head entity)}, \; l \text{ (relation)}, \; t \text{ (tail entity)}.$$

Les relations sont représentées comme des **translations** dans l'espace d'embedding :

$$h + l \approx t \quad \text{si la relation tient}.$$

> 💡 **Intuition.** Une fois le mapping appris, on translate d'un nœud à un autre par la relation. Si la relation tient, on doit retrouver la *tail entity*. C'est le genre de fonction objectif qu'on cherche.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im17.png]]

### D. Pseudo-code TransE

**Pas de random walk** ici — sampling uniforme de $l$ qu'on normalise.

> [!note]- Algorithme
> - On travaille avec des **mini-batches** de triplets.
> - Pour chaque triplet, on échantillonne un **triplet corrompu** (negative sampling) — on remplace la *head* ou la *tail* par une entité aléatoire.
> - On update les embeddings :
>   - **Réduire** la distance $\|h + l - t\|$ pour les vrais triplets.
>   - **Augmenter** la distance pour les négatifs.
>
> C'est une **comparative loss** : pull pour les bons, push pour les mauvais. L'intérêt majeur : on trouve des embeddings **sans random walks**.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im18.png]]

---

## VI. Embedding de graphes entiers

**Objectif.** Embedding d'un sous-graphe ou d'un graphe complet $G$ : $z_G$.

**Exemples d'applications.**
- Classifier des molécules toxiques vs non-toxiques.
- Identifier des graphes anomaux.

### A. Approche 1 — moyenne / somme des embeddings de nœuds

L'idée la plus simple : on lance une technique d'embedding standard sur le (sous)graphe $G$, puis on **somme ou moyenne** les embeddings de nœuds :

$$z_G = \sum_{v \in G} z_v.$$

> 💡 **Application.** Utilisée par [Duvenaud et al. 2016](https://arxiv.org/abs/1509.09292) pour classifier des molécules sur la base de leur structure de graphe. Une matrice d'embedding par graphe ; on somme les eigenvectors → $z_{G_1}, z_{G_2}, \ldots, z_{G_n}$ pour $n$ graphes, qu'on peut ensuite plotter et comparer.

### B. Approche 2 — Virtual Node

Introduire un **nœud virtuel** pour représenter le (sous)graphe et lancer une technique d'embedding standard. Proposé dans [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) (Li et al. 2017) comme technique générale pour subgraph embedding.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im19.png]]

### C. Approche 3 — Anonymous Walk Embeddings

> 💡 **Référence.** [Anonymous Walk Embeddings](https://arxiv.org/pdf/1805.11921.pdf) (Ivanov et al., 2018).

#### C.1 Idée

Dans un **anonymous walk**, les états correspondent à l'**index de la première visite** du nœud dans le random walk. Au lieu de représenter un random walk comme une séquence de nœuds, on le représente comme une séquence de **temps de première visite**.

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/im20.png]]

> [!example] Trois random walks
> - **Walk 1.** Index 1 (A visité au step 1), 2 (B visité au step 2), 3 (C au step 3), puis B (déjà visité, garde son index 2), puis C (garde son 3) → $1, 2, 3, 2, 3$.
> - **Walk 2.** Même séquence : $1, 2, 3, 2, 3$ même si les nœuds réels sont C, D, B, D, B.
> - **Walk 3.** Différente : nœuds visités dans un ordre différent.

> 💡 **Anonymes vis-à-vis de l'identité des nœuds.** Deux random walks qui visitent **différents nœuds mais dans un ordre similaire** auront la même représentation anonyme.

#### C.2 Nombre de walks

Le nombre d'anonymous walks **croît exponentiellement**. Il y a 5 walks de longueur 3 :

$$w_1 = 111, \; w_2 = 112, \; w_3 = 121, \; w_4 = 122, \; w_5 = 123.$$

![[images/3-Apprentissage automatique/Graph ML/Representation Learning/4.Graph_embed/im1.png]]

#### C.3 Embedding du graphe

> [!warning] L'idée
> Simuler des anonymous walks $w_i$ de longueur $l$ et compter leurs occurrences. Représenter le graphe comme une **distribution de probabilité** sur ces walks :
>
> $$Z_G[i] = \text{probabilité de l'anonymous walk } w_i \text{ dans } G.$$

> [!example] Avec $l = 3$
> Le graphe est représenté par un vecteur de dimension 5 (les 5 walks $111, 112, 121, 122, 123$). Pour augmenter la dimension, augmenter $l$.

#### C.4 Combien de random walks faut-il ?

Pour avoir une distribution avec une erreur de plus de $\varepsilon$ avec probabilité moins de $\delta$ :

$$m = \left\lceil \frac{2}{\varepsilon^2} \big( \log(2^\eta - 2) - \log \delta \big) \right\rceil$$

où $\eta$ est le nombre total d'anonymous walks de longueur $l$.

> [!example] Cas concret
> $\eta = 877$ anonymous walks de longueur $l = 7$. Si $\varepsilon = 0.1$ et $\delta = 0.01$, on a besoin de $m = 122\,500$ random walks.

#### C.5 Apprendre les embeddings des walks

Plutôt que de représenter chaque walk par sa fréquence, on **apprend l'embedding $z_i$** de chaque anonymous walk $w_i$. On apprend un graph embedding $Z_G$ **conjointement** avec tous les anonymous walk embeddings $Z = \{z_i : i = 1, \ldots, \eta\}$.

> 💡 **L'objectif.** Embedder les walks de sorte que **le walk suivant puisse être prédit** — comme dans Skip-Gram. On apprend un vecteur de taille $\eta + 1$ (les walks + l'embedding du graphe).
