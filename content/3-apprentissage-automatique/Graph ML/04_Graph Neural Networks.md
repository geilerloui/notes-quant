---
title: Graph Neural Networks
---
# Graph Neural Networks

> Cette note couvre les **Graph Neural Networks (GNN)** : comment généraliser les CNN à des graphes arbitraires. On part du framework de base (locality + aggregation + composition), on voit les architectures principales (vanilla GCN, GraphSAGE, GAT, GIN), les **limitations** des GNN classiques (test d'isomorphisme WL, robustesse adversarial), et trois **applications industrielles** (PinSAGE pour la reco, DECAGON pour les graphes hétérogènes, génération goal-directed de molécules).

> 💡 **Position dans le cours.** Suite logique de [[03_Graph Representation Learning]] : là on apprenait des embeddings *task-independent* via random walks ; ici on apprend des représentations **end-to-end** intégrées dans une architecture deep learning, qu'on entraîne directement pour la tâche cible.

> ⚠️ **Note sur la qualité.** La première moitié de cette note (Introduction, Basics, GraphSAGE, GAT, Limitations + WL) est mise au propre depuis mes notes du cours Stanford CS224W (Leskovec). La seconde moitié (PinSAGE, DECAGON, génération) provient de notes de slides bien plus brutes — j'ai tenté de les structurer du mieux possible mais les images manquent et certains points restent télégraphiques.

---

## I. Introduction

### A. Le CNN comme inspiration

On peut voir une **image** comme une grille 2D de nœuds interconnectés. Le CNN se décompose en trois opérations :

> [!warning] Les trois opérations du CNN
> - **Locality** — quand on place un masque (vert) sur l'image, on regarde les voisins autour du nœud central (rouge). Les 8 voisins contribuent à la nouvelle valeur du nœud central.
> - **Aggregation** — on multiplie les coefficients du masque par les pixels présents et on somme : c'est la nouvelle valeur du pixel central.
> - **Composition (function of a function)** — le masque vert se déplace jusqu'à couvrir toute l'image, et le résultat repasse dans la couche suivante du réseau.

> 💡 **Le cœur du CNN = Locality + Aggregation.** C'est ces deux étapes qu'on va vouloir généraliser aux graphes.

![[images/3-Apprentissage automatique/Graph ML/GNN/1.intro/im1.png]]
![[images/3-Apprentissage automatique/Graph ML/GNN/1.intro/im2.png]]

### B. Pourquoi le CNN ne s'applique pas directement aux graphes

> [!warning] Deux problèmes structurels
> - **Pas de grille fixe.** Les images et le texte sont des grilles régulières. Les graphes n'ont **pas** de pattern régulier — un nœud peut avoir 2 voisins ou 50.
> - **Pas d'ordre spatial.** Topologie complexe, espace non-euclidien, **pas d'ordre fixe des nœuds**. Si on permute les nœuds, le graphe est le même mais les "lignes" de la matrice d'adjacence sont dans un ordre différent.

![[images/3-Apprentissage automatique/Graph ML/GNN/1.intro/im3 (1).png]]
**Figure 1.** Image et texte = structures de grille fixe. Graphe = aucune régularité.

### C. L'approche naïve qui ne marche pas

On pourrait représenter le graphe par sa matrice d'adjacence, ajouter des features (bag-of-words...), et passer ligne par ligne dans un MLP.

> [!warning] Pourquoi ça ne marche pas
> - **Pas scalable.** Pour un graphe de $n$ nœuds, on a $n$ data points et $O(N)$ paramètres.
> - **Pas de réutilisation.** Pour un graphe de taille différente, il faut tout réapprendre.
> - **Sensible à l'ordre des nœuds.** Si on relabelle les nœuds, on doit tout réapprendre.
>
> Or les graphes sont **invariants par permutation des nœuds** — propriété qu'un MLP naïf ne respecte pas.

![[images/3-Apprentissage automatique/Graph ML/GNN/im2.png]]

### D. L'objectif des GNN — node embedding

On mappe les nœuds vers un espace d'embedding $d$-dimensionnel où les nœuds voisins sont proches. Pour des nœuds $a$ et $b$ on veut :

$$\text{sim}(a, b) = (z^a)^T z^b = (x^a)^T x^b.$$

**La question centrale : comment construire l'encoder ?** Il doit faire les trois opérations dans un graphe :

**(i) Locality** — par construction d'un **graphe computationnel** : pour chaque nœud, on regarde ses voisins, puis les voisins de ses voisins, etc.

**(ii) Aggregation** — on agrège l'information des nœuds dans une boîte qui doit être **invariante par permutation** :

$$f(a + b) \equiv f(b + a).$$

![[images/3-Apprentissage automatique/Graph ML/GNN/im4.png|555]]
**Figure 2.** Construction du graphe computationnel : à gauche le graphe, à droite l'arbre de calcul autour d'un nœud cible.

---

## II. Basics du deep learning sur graphes

> 💡 **Référence.** [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf) (Kipf & Welling, ICLR 2017). L'idée centrale : **le voisinage d'un nœud définit son graphe computationnel**.

![[images/3-Apprentissage automatique/Graph ML/GNN/im3 (1).png]]

### A. Aggregate Neighbors

#### A.1 Construction du réseau

On part d'un nœud cible $A$. Ses voisins (B, C, D) forment la **layer-1**. On répète itérativement jusqu'à la profondeur désirée.

> 💡 **Pas trop profond en pratique.** Six degrés de séparation suffisent à connecter deux personnes au monde — les réseaux réels ont un diamètre étonnamment petit. Aller à grande profondeur est coûteux et n'apporte rien.

**Embeddings par layer.**
- **Layer-0** : embedding initial = features d'entrée $x_u$.
- **Layer-$k$** : reçoit l'info des nœuds à $k$ hops.

**Propriété d'invariance.** Si $A$ et $C$ sont permutés en layer-0, le résultat ne doit pas changer.

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im1.png]]

#### A.2 Que met-on dans les boîtes ?

Typiquement un **petit réseau de neurones**, avec une **moyenne** des messages des voisins (ça garde les ordres de magnitude similaires). On peut aussi montrer théoriquement que la **somme** est optimale.

> 💡 **Sum, mean, max sont permutation-invariant.** C'est crucial : on ne veut pas que le résultat dépende de l'ordre des nœuds.

#### A.3 Formule mathématique

$$h_v^{(0)} = x_v$$

$$h_v^{(l+1)} = \sigma\!\left( W_l \sum_{u \in N(v)} \frac{h_u^{(l)}}{|N(v)|} + B_l \, h_v^{(l)} \right)$$

> [!warning] Lecture
> - **Indice $v$** : le nœud courant. **Exposant $l$** : la couche.
> - **Premier terme (rouge)** : on parcourt les voisins, on prend leur représentation au layer précédent, on **moyenne**. C'est la partie **aggregation**.
> - **Deuxième terme (bleu)** : on combine avec le message du nœud lui-même au layer précédent.
> - **$W_l$** et **$B_l$** : matrices de transformation **partagées par tous les nœuds** au layer $l$ — c'est ça qu'on apprend.

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im2.png]]

### B. Entraînement du modèle

On a besoin d'une **loss sur les embeddings**. On peut feeder les embeddings dans n'importe quelle loss et faire du SGD.

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im3 (1).png]]

> [!warning] Notation finale
> - $h_v^l$ : représentation cachée de $v$ au layer $l$.
> - $W_k$ : matrice de poids pour l'agrégation du voisinage.
> - $B_k$ : matrice de poids pour la transformation du nœud lui-même.

### C. Formulation matricielle

> [!note]- Setup
> - $X \in \mathbb{R}^{N \times M}$ : feature matrix d'entrée.
> - $A \in \mathbb{R}^{N \times N}$ : matrice d'adjacence.
> - Layer caché : $H^{(l+1)} = f(H^{(l)}, A)$, avec $H^{(0)} = X$ et $H^{(L)} = Z$ (sortie).
>
> En général :
>
> $$H^{(l)} = [h_1^{(l)}, \ldots, h_{|V|}^{(l)}]^T.$$

À chaque stage on stocke une matrice cachée où chaque ligne est un nœud.

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im11.png|407]]

**Update equation.**

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im5.png]]

> [!warning] Deux ajustements importants
> - **Self-loop.** On modifie $A$ en $A + I$ pour que chaque nœud "se voie lui-même".
> - **Normalisation.** $A$ non normalisée → explosion du gradient et sensibilité au scaling.

> [!note]- Dérivation matricielle de l'agrégation
> Aggregation des voisins :
>
> $$\sum_{u \in N(v)} h_u^{(l)} = A_{v, :} H^{(l)}.$$
>
> Soit $D$ la matrice diagonale de degrés : $D_{v, v} = |N(v)|$. Alors $D^{-1}$ est diagonale avec $1/|N(v)|$, et :
>
> $$\sum_{u \in N(v)} \frac{h_u^{(l)}}{|N(v)|} \;\Rightarrow\; H^{(l+1)} = D^{-1} A H^{(l)}.$$

**Forme matricielle finale (GCN de Kipf-Welling) :**

$$\boxed{H^{(l+1)} = \sigma\!\left( \textcolor{red}{\tilde A H^{(l)} W_l^T} + \textcolor{blue}{H^{(l)} B_l^T} \right)}$$

avec $\tilde A = D^{-1/2} A D^{-1/2}$ (normalisation symétrique). En **rouge** : agrégation du voisinage. En **bleu** : transformation du nœud lui-même.

![[images/3-Apprentissage automatique/Graph ML/GNN/im6.png|312]]

> 💡 **En pratique, $\tilde A$ est très sparse**, donc on peut utiliser de la multiplication matricielle creuse efficace. (Note : tous les GNN ne se mettent pas en forme matricielle — ça dépend de la complexité de l'agrégation.)

### D. Modes d'entraînement

> [!warning] (a) Unsupervised — "nœuds similaires ont des embeddings similaires"
> $$\mathcal{L} = \sum_{z_u, z_v} \text{CE}(y_{u,v}, \text{DEC}(z_u, z_v))$$
>
> où $y_{u,v} = 1$ si $u$ et $v$ sont similaires, CE = cross entropy, DEC = decoder (ex: produit scalaire). La similarité peut venir de [[03_Graph Representation Learning|random walks]] (node2vec, DeepWalk), de la factorisation matricielle, ou de proximité dans le graphe.

> [!warning] (b) Supervised — entraîner directement pour une tâche
> Par exemple node classification.
>
> ![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im4.png]]

> [!example] Application — drug-drug interaction
> Détecter si une drogue est toxique ou sûre dans un graphe d'interactions médicamenteuses.
>
> ![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im5.png]]

### E. Vue d'ensemble du design

![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im6.png]]
![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im7.png]]
![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im8.png]]

### F. Capacité inductive

> [!warning] Generalize to new graphs
> Une fois entraîné, on peut appliquer le modèle à un graphe **inconnu** par simple forward pass. C'est possible grâce au **parameter sharing** ($W_l$ et $B_l$ sont partagées entre tous les nœuds).
>
> Exemple : entraîner sur un graphe d'interactions protéiques de l'organisme A, et générer des embeddings pour de nouvelles données de l'organisme B.
>
> ![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im9.png]]
> ![[images/3-Apprentissage automatique/Graph ML/GNN/2.Basic/im10 (1).png]]

> [!warning] Generalize to new nodes
> On peut générer des embeddings **online** à mesure que de nouveaux nœuds arrivent (Reddit, YouTube, Google Scholar). L'embedding s'adapte dynamiquement.
>
> ![[images/3-Apprentissage automatique/Graph ML/GNN/im10 (1).png]]

---

## III. GraphSAGE

> 💡 **Référence.** [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf) (Hamilton et al., NIPS 2018).

### A. L'idée centrale

Jusqu'ici on agrégeait les messages des voisins par moyenne (pondérée). Peut-on faire mieux ?

**Formule simple précédente :**

$$h_v^k = \sigma\!\left( W_k \sum_{u \in N(v)} \frac{h_u^{k-1}}{|N(v)|} + B_k h_v^{k-1} \right).$$

**Modification GraphSAGE :**

$$\boxed{h_v^k = \sigma\!\left( \big[\, W_k \cdot \text{AGG}\big(\{h_u^{k-1}, \forall u \in N(v)\}\big), \; B_k h_v^{k-1} \,\big] \right)}$$

> [!warning] Deux différences avec GCN
> - **AGG généralisée** au lieu d'une simple moyenne.
> - **Concaténation** $[\cdot, \cdot]$ entre embedding du voisinage et embedding du nœud, au lieu de les additionner.

### B. Trois choix d'AGG

> [!warning] Mean
> Moyenne pondérée des voisins :
>
> $$\text{AGG} = \sum_{u \in N(v)} \frac{h_u^{k-1}}{|N(v)|}.$$

> [!warning] Pool
> Transformer chaque voisin par MLP, puis appliquer une fonction symétrique :
>
> $$\text{AGG} = \gamma\!\left(\big\{\text{MLP}(h_u^{(l)}), \forall u \in N(v)\big\}\right).$$

> [!warning] LSTM
> $$\text{AGG} = \text{LSTM}\big([h_u^{k-1}, \forall u \in \pi(N(v))]\big).$$
>
> 💡 **LSTM n'est pas order-invariant** par défaut. On essaie plusieurs permutations $\pi$ pour qu'il "apprenne" l'invariance. Un papier a montré qu'avec assez de permutations aléatoires, le LSTM devient effectivement invariant.

---

## IV. Graph Attention Networks (GAT)

> 💡 **Références.** Velickovic et al. ICLR 2018 + Vaswani et al. NIPS 2017 (l'attention originale des Transformers).

### A. Motivation

**Recap GCN :**

$$h_v^k = \sigma\!\left( W_k \sum_{u \in N(v)} \frac{h_u^{k-1}}{|N(v)|} + B_k h_v^{k-1} \right).$$

> 💡 **Ce qui ne va pas.** Le coefficient $\alpha_{vu} = 1/|N(v)|$ traite **tous les voisins comme également importants**. Or en pratique, certains voisins comptent plus que d'autres ("mes amis proches m'influencent plus que des connaissances").

### B. Mécanisme d'attention

On laisse l'algorithme apprendre des poids différents pour chaque voisin.

**Coefficients d'attention** :

$$e_{vu} = a(W_k h_u^{k-1}, W_k h_v^{k-1}).$$

$e_{vu}$ indique l'importance du message de $u$ pour $v$. On normalise par softmax :

$$\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in N(v)} \exp(e_{vk})},$$

$$\boxed{h_v^k = \sigma\!\left( \sum_{u \in N(v)} \alpha_{vu} W_k h_u^{k-1} \right)}$$

Les paramètres de $a$ sont appris **conjointement** avec les autres poids du réseau, end-to-end.

![[images/3-Apprentissage automatique/Graph ML/GNN/4.Attention/im1.png]]
**Figure 3.** Visualisation : les arêtes les plus importantes sont mises en valeur. Le softmax assure que les poids restent comparables entre voisinages.

![[images/3-Apprentissage automatique/Graph ML/GNN/4.Attention/im2.png]]
**Figure 4.** Synthèse du processus d'attention.

### C. Multi-head attention

Comme dans les Transformers, on **réplique** l'opération d'attention $R$ fois (chaque réplica avec ses propres paramètres) puis on agrège (concat ou somme). Stabilise l'apprentissage.

### D. Propriétés

> [!warning] Avantages
> - **Computationnellement efficace** — coefficients parallélisables sur les arêtes, agrégation parallélisable sur les nœuds.
> - **Storage efficient** — opérations sparse en $O(V + E)$, nombre de paramètres fixe (indépendant de la taille du graphe).
> - **Localisé** — n'attend que sur le voisinage local.
> - **Inductif** — mécanisme partagé sur les arêtes, ne dépend pas de la structure globale.

### E. Exemple — Cora Citation Net

Nœuds = papiers, arêtes = citations, classes = domaines scientifiques.

![[images/3-Apprentissage automatique/Graph ML/GNN/im11.png]]
**Figure 5.** Sur Cora, DeepWalk obtient 67.2% de classification accuracy ; GAT monte à **83.8%** — saut substantiel.

> 💡 Le t-SNE des embeddings GAT colore les 7 classes de publication, et l'épaisseur des arêtes est proportionnelle à $\sum_k (\alpha_{ij}^k + \alpha_{ji}^k)$ (somme sur 8 attention heads).

---

## V. Limitations des GNN classiques

### A. Le problème central

> [!warning] Deux limitations majeures
> - **(i)** Certaines structures de graphe simples **ne peuvent pas être distinguées** par les GNN conventionnels — quand les features de nœuds sont uniformes (tous jaunes par ex.), GCN et GraphSAGE n'arrivent pas à séparer deux graphes pourtant différents.
> - **(ii)** Les GNN ne sont **pas robustes au bruit** dans les données de graphe (attaques adversariales).

![[im2_1.png]]
![[im2_2.png]]

### B. Capturer la structure du graphe — test d'isomorphisme

**La question centrale :** étant donnés deux graphes différents, les GNN peuvent-ils les mapper vers des représentations différentes ?

> 💡 **C'est le problème de l'isomorphisme de graphes.** Pas d'algo polynomial connu dans le cas général. Donc les GNN ne distinguent pas parfaitement n'importe quels graphes.

#### B.1 Repenser le GNN

Le **graphe computationnel** d'un nœud est une **structure d'arbre enraciné** de profondeur arbitraire. Si tous les nœuds ont les mêmes features et que les voisinages sont structurellement similaires, les arbres computationnels seront identiques → on n'arrivera pas à les distinguer.

**Cas idéal.** Pour deux graphes différents, on veut une **fonction injective** : nœuds différents → représentations différentes.

![[im2_3.png]]

> [!warning] Rappel — injectivité
> Une fonction est **injective** si elle mappe des éléments différents vers des sorties différentes.
>
> ![[im2_4.png]]

#### B.2 Multi-set comme abstraction

L'agrégation de voisinage est une **fonction sur multi-sets** :
- Plusieurs nœuds peuvent avoir les mêmes features.
- L'ordre n'a pas d'importance (invariance recherchée).

![[im2_5.png]]

**Stratégie.** Plutôt que de raisonner directement sur les GNN, raisonnons sur les multi-sets : si on garantit l'injectivité au niveau du multi-set, on règle le problème côté GNN.

#### B.3 Case study 1 — GCN (mean pooling) : pas injectif

GCN utilise le **mean pooling**. On peut le **fooler** en dupliquant un nœud : deux multi-sets différents (un contient une copie en plus) mappent vers la même valeur après moyenne.

![[im2_6.png]]

#### B.4 Case study 2 — GraphSAGE-maxpool : pas injectif non plus

GraphSAGE avec max pooling échoue à distinguer des multi-sets ayant les **mêmes éléments distincts** (le max ne respecte pas la multiplicité).

![[im2_7.png]]

### C. Solution — fonction multi-set injective via réseaux de neurones

> [!warning] Théorème
> Toute fonction multi-set injective peut s'exprimer comme :
>
> $$\Phi\!\left(\sum_{x \in S} f(x)\right)$$
>
> avec $\Phi$ et $f$ des fonctions non linéaires.

![[im2_8.png]]

> 💡 **Le MLP est un universal approximator.** On peut donc paramétrer $\Phi$ et $f$ par des MLP — ils approchent n'importe quelle fonction.
>
> ![[im2_9.png]]

### D. Graph Isomorphism Network (GIN)

> 💡 **Référence.** Xu et al., ICLR 2019.

**Différence clé avec GCN/GraphSAGE :** au lieu de **mean** ou **max** pooling, GIN utilise du **sum pooling** + un MLP. Le sum pooling préserve la multiplicité — donc deux multi-sets différents mappent vers des sorties différentes.

![[im2_10.png]]
![[im2_11.png]]

> 💡 **Intuition.** Sum pooling + MLP donne une **agrégation injective** sur les graphes.

### E. Lien avec le test de Weisfeiler-Lehman (WL)

GIN est étroitement lié au **Weisfeiler-Lehman Graph Isomorphism Test (1968)**. Le test WL est connu pour distinguer la plupart des graphes du monde réel, et sa complexité est **linéaire en nombre d'arêtes** — c'est cheap.

> [!note]- L'algorithme WL en bref
> - Compte les nœuds du graphe (à chaque graphe la même couleur initiale).
> - **Niveau 1.** Pour chaque nœud, agrège les couleurs des voisins → nouvelle couleur.
> - **Niveau 2, 3...** On continue jusqu'à stabilité.
> - **Pooling.** On compte combien de nœuds ont chaque couleur. Si les counts diffèrent entre les deux graphes, ils sont distincts.

> [!example] WL en action
> (i) WL mappe différents arbres enracinés vers différentes couleurs.
> (ii) Compte les couleurs.
> (iii) Compare les counts entre graphes.
>
> ![[im2_12.png]]

**Limites du WL.** Il existe des cas pathologiques où WL échoue — par exemple les graphes "G-skip" (nombre de skips variable). Des travaux récents étendent WL mais deviennent exponentiels.

![[im2_13.png]]

### F. Performance expérimentale de GIN

**Datasets.** Réseaux sociaux et graphes bio-chimiques.

> [!warning] Résultats
> - **Training accuracy** : GIN aussi puissant que WL, atteint 100% sur certains datasets.
> - **NCI1** (chimie) : presque parfait.
> - **REDDIT** (social network) : binary task, GraphSAGE était au niveau d'un classifieur aléatoire (50%).

![[im2_14.png]]

**Test accuracy sans node features** (uniquement la topologie) : GCN et GraphSAGE échouent quasiment, GIN reste robuste. Avec features de nœud, GIN gagne aussi — la topologie + features se combinent bien.

![[im2_15.png]]

> 💡 **Take-away.** Les node features jouent un grand rôle pour la node classification. Le gap GCN vs GIN vient de la **combinaison** features + structure.

---

## VI. Vulnérabilité aux attaques adversariales

### A. Le problème

Les attaques adversariales sont un risque réel sur les GNN. Exemples concrets :
- **Web graph** — manipuler les liens pour booster artificiellement la pertinence d'un site (SEO black hat).
- **Friend recommendation** — créer des faux comptes recommandés à beaucoup d'utilisateurs (propagation de fake news).

![[im2_16.png]]

### B. Setup formel — semi-supervised node classification

On a une matrice d'adjacence $A$, une matrice de features $X$, et $\hat A$ la version renormalisée. On fait deux passes GCN. Entraînement = minimiser la cross entropy sur les nœuds labellisés.

![[im2_17.png]]

### C. Possibilités d'attaque

**Setup.**
- **Target node** = le nœud qu'on cherche à manipuler (ex : un compte twitter de fake news qu'on veut booster).
- **Attacker node** = un nœud qu'on contrôle.

**Trois types d'attaque possible :**
- **Modifier les features du target** (changer le contenu du compte fake news pour éviter les filtres).
- **Ajouter des connexions** (link farm : ajouter des arêtes vers d'autres comptes complices).
- **Supprimer des connexions**.

![[im2_18.png]]
![[im2_19.png]]

**Objectif.** Faire baisser la probabilité de la classe orange (vraie classe) et monter celle de la verte (la classe désirée).

### D. Formulation mathématique

On modifie la matrice d'adjacence (quelques edges en plus / en moins) en gardant l'objectif :
- $c_{\text{new}} \neq c_{\text{old}}$ (on change la prédiction).
- Le GCN reste celui qui sera retrained sur le graphe modifié.

> [!warning] Difficulté
> La modification d'un graphe est **discrète** (ajouter/retirer une arête) — pas de gradient lisse à exploiter.

![[im2_20.png]]

### E. Expériences (Nettack)

Sur la **classification margin** (différence entre la plus haute proba de classe et la 2e plus haute) — on veut une marge élevée pour que l'attaque soit fiable :

![[im2_21.png]]

**5 manipulations seulement** sur un graphe de 2k nœuds / 5k edges suffisent à casser la prédiction.

![[im2_22.png]]

### F. Questions ouvertes

> [!warning] Pretraining GNN
> Comme avec les Transformers en NLP : on pretrain sur de gros datasets de molécules connues (PubChem...), puis on **fine-tune** sur la tâche downstream. Vise à transférer la "connaissance chimique" générique.

> [!warning] GNN robustes
> Comment se défendre contre les attaques ? Trade-off **accuracy / robustness** à trouver.

![[im2_23.png]]
![[im2_24.png]]
![[im2_25.png]]

---

## VII. Application — PinSAGE (recommandation Pinterest)

> ⚠️ **Note.** Cette section et les suivantes proviennent de notes de slides brutes — j'ai structuré et clarifié ce qui était lisible, mais les détails sont moins fournis que les sections précédentes.

### A. Le problème de la recommandation

**Setup.** Items que les users consomment (achats, musique...). On veut personnaliser la reco depuis l'historique.

**Pipeline.** À partir d'un set de **query items** (basé sur le passé), on requête une base d'items et on retourne des recommandations.

> [!example] Pinterest
> Si un utilisateur a query "strawberry shake", on veut recommander des items liés (chocolate strawberry, etc.). Si le user a interagi avec **plusieurs items**, le système doit formuler une query qui combine ces multiples signaux.

### B. Définir la similarité

Deux approches classiques :

> [!warning] Content-based
> Similarité entre items basée sur leur contenu (image, texte, features).

> [!warning] Collaborative filtering
> User-item bipartite graph. Pour un user $X$, trouver les autres users qui interagissent avec les mêmes items que $X$. **Pas besoin de regarder le contenu** — il suffit de savoir qui aime quoi.

### C. Les questions à résoudre

- **Signal.** Comment savoir quels items sont liés ? Quels songs un user aime / n'aime pas ?
- **Extrapolation.** Comment extrapoler la similarité à un set inconnu d'items ?
- **Évaluation.** Comment mesurer le succès de la méthode ?

### D. Le graphe Pinterest

- **Pin** : une image + texte + saved on a board.
- **Board** : une collection de pins (4 milliards de pins au total).
- Le graphe est **dynamique** — les users sauvent en continu.

**Objectif.** Pour un pin donné, trouver les pins les plus similaires.

### E. La méthode

#### E.1 Embedding initialization

Initialiser à partir des **features visuelles et textuelles** de chaque pin. Puis on **agrège** dans le GNN pour obtenir l'embedding final du pin — qui exploite les voisins du graphe pour s'enrichir.

> 💡 **Pourquoi mieux qu'embedding direct.** Les voisins corrigent les erreurs purement visuelles. Si on ne regarde que l'image, on confond *brown rice* et *terre*, *carpet on the floor* et *decorative wall carpet*. Les voisins du graphe lèvent l'ambiguïté.

#### E.2 Locality sensitive hashing pour la recherche

Une fois les embeddings appris, pour requêter en haute dimension, on utilise du **LSH** — sinon le calcul de distance est intractable.

#### E.3 Génération du training data

> [!warning] Paires positives
> Deux pins sont **liés** s'ils sont **consécutifs sur le même board dans un court intervalle de temps**. Exemple : un user qui crée un board "cute dog" et y ajoute plusieurs pins en peu de temps.

> [!warning] Paires négatives
> Set aléatoire de pins.

#### E.4 Loss

$$\mathcal{L} = \sum_{(z_q, z_+, z_-)} \max\!\big( 0, \; z_q^T z_- - z_q^T z_+ + \Delta \big).$$

> 💡 **Max-margin loss.** Le terme $\Delta$ contrôle de combien la paire positive doit être plus similaire que la négative. On veut $z_q \cdot z_+$ grand et $z_q \cdot z_-$ petit.

#### E.5 Importance Pooling avec PageRank

> 💡 **Le problème du Lady Gaga.** Si on agrège tous les voisins d'un pin super populaire (ex : un pin Lady Gaga avec des millions de followers), c'est intractable. Solution : sélectionner les **top-K voisins les plus importants** par PageRank.

**Procédure.**
1. Démarrer un random walk depuis le query pin.
2. Compter les visites des autres pins.
3. Garder le top-$k$ (typiquement $k = 50$).
4. Les visit counts deviennent les **poids d'importance** dans l'agrégation.

#### E.6 Hard negative mining

À mesure que l'entraînement avance, on utilise des négatifs de plus en plus durs : pas des pins random, mais des pins **moyennement similaires** (rang 1000-5000 d'un random walk depuis le query). C'est une forme de **curriculum learning** : on commence facile, on durcit.

### F. Évaluation

**Métriques.**
- **MRR (Mean Reciprocal Rank)** : rang de l'item attendu dans le top.
- **Hit-rate** : fréquence à laquelle l'item est dans le top-$k$.

**Comparaison sur Pinterest.**
- **Visual only** : 0.23
- **Text only** : 0.19
- **Visual + text combined** : 0.37
- **Graph-based (PinSAGE)** : **0.59** — quasi 50% mieux que le combiné visuel/textuel.

> 💡 **Take-away.** Le graphe apporte un signal **complémentaire** au contenu pur. PinSAGE corrige typiquement les cas où visual seul confond *bibimbap* (bowl coréen) avec un autre bowl, ou où text seul donne des résultats trop génériques.

---

## VIII. Application — DECAGON (graphes hétérogènes)

### A. Le problème — polypharmacy side effects

> 💡 **Le contexte.** Les **clinical trials** testent généralement les médicaments **un par un**. Mais en pratique, **la moitié des patients de plus de 70 ans prennent au moins 2 médicaments**. Les **side effects de combinaisons** ne sont pas dans les trials — ils émergent après mise sur le marché, quand assez de patients se plaignent à la FDA.

**Le but.** Étant donnés deux médicaments $C$ et $D$ pris ensemble, prédire **quels side effects** peuvent apparaître.

### B. Le graphe hétérogène

> [!warning] Deux types de nœuds
> - **Drug nodes** — les médicaments. Chaque drug a un feature vector.
> - **Protein nodes** — les protéines (~20 000 dans le corps). Chacune a un feature vector décrivant son rôle.

> [!warning] Plusieurs types d'arêtes
> - **Protein-protein** : interactions physiques (testées expérimentalement par les biologistes).
> - **Drug-protein** : quel médicament agit sur quelle protéine.
> - **Drug-drug** : edges typés par **type de side effect** (~1000 types possibles).

### C. La task — multi-relational link prediction

Pour une paire de médicaments, prédire **quel type d'edge** peut exister (donc quel side effect peut apparaître). C'est un problème de **link prediction multi-typé**.

### D. Architecture

**Encoder GNN à 2 layers** (deux hops). Pour chaque type d'edge $r$, on a une **architecture d'agrégation séparée** :
- Aggregate l'info des proteins voisines.
- Aggregate l'info des autres drugs voisins, par type de relation $r$.

**Decoder** : pour chaque paire de drugs $(c, s)$, on prédit la probabilité de chaque side effect type — 1000 prédictions binaires en parallèle.

### E. Résultats

**Métrique** : Area Under the Precision-Recall Curve.

**Validation expérimentale.** Top 10 prédictions du modèle : **5 sur 10** ont été **vérifiées dans la littérature postérieure** au papier (publié en 2016, vérifications sur publications plus récentes). Les 5 autres ne sont pas forcément fausses — possiblement des hypothèses **non encore vérifiées**.

> 💡 **Limite.** C'est un **modèle de population** — il ne tient pas compte de la génétique individuelle. La FDA ne permet pas (encore) ce type d'usage personnalisé.

---

## IX. Goal-directed graph generation

### A. Le problème

Peut-on **générer des graphes** de manière ciblée — qui satisfont une **fonction objectif** ?

**Cas d'usage typique : drug discovery.**
- Générer des graphes qui **ressemblent à des molécules valides**.
- **Optimiser** des propriétés : drug-likeness, faible toxicité, affinité pour une protéine cible...

### B. L'approche

> [!warning] Génération conditionnelle
> On peut entraîner un générateur de graphes tel qu'il optimise une **fonction objectif** (drug-likeness, toxicité...). À partir d'une molécule **partiellement construite**, on peut la **compléter** en améliorant le critère choisi.

> 💡 **L'esprit.** C'est l'analogue graphe du *conditional generation* des images (text-to-image, etc.) — sauf que la sortie est un graphe moléculaire, et la condition est un score chimique.

---

## Annexe — Récapitulatif des architectures GNN

| Modèle | Agrégation | Forme | Spécificité |
| :--- | :--- | :--- | :--- |
| **GCN** (Kipf-Welling) | Mean (normalisée par degré) | $\sigma(\tilde A H W^T)$ | Forme matricielle élégante |
| **GraphSAGE** | Mean / Pool / LSTM | $\sigma([W \cdot \text{AGG}(N(v)), B h_v])$ | Concat self/neighbor, plus inductif |
| **GAT** | Attention | $\sigma(\sum \alpha_{vu} W h_u)$ | Poids appris par voisin |
| **GIN** | Sum + MLP | $\text{MLP}((1+\epsilon) h_v + \sum h_u)$ | Le seul prouvablement aussi puissant que WL |

> 💡 **Le résumé.** Plus l'agrégation est **expressive et injective**, plus le GNN distingue de structures différentes. GIN est l'aboutissement théorique ; GAT est le choix par défaut quand on veut de la flexibilité ; GCN reste le baseline solide.
