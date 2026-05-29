---
title: Message Passing - Graph ML
---
# Message Passing et Node Classification

> Cette note couvre le **message passing** pour la classification semi-supervisée de nœuds dans un graphe. On part du problème (semi-supervised node classification : étiqueter des nœuds non labellisés à partir de quelques labels connus), on voit les **trois algorithmes classiques** (Relational classification, Iterative classification, Loopy Belief Propagation), et on illustre avec deux applications réelles (détection de faux avis, fraude aux enchères en ligne).

> 💡 **Pré-requis utiles.** Cette note partage des concepts avec [[02_Représentation]] (notion de message passing dans Belief Propagation, V-structure) et [[03_Inférence#III. Belief Propagation]]. Le BP des PGM et le BP des graphes ML sont **la même idée mathématique**, juste appliquée à des graphes différents (PGM = graphe de variables aléatoires, ici = graphe de données).

---

## I. Introduction

> [!warning] Définition (Semi-supervised node classification)
> Étant donnés les labels de quelques nœuds, on veut **prédire les labels des nœuds non labellisés**.

> [!example] Exemple — détection de fraudeurs
> Dans un réseau, certains nœuds sont des fraudeurs et d'autres sont totalement fiables. Comment trouver les autres fraudeurs et personnes de confiance ?
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/1.introduction/im1.png]]

### A. L'intuition — corrélations dans les graphes

L'intuition centrale : **il existe des corrélations dans les réseaux**, c'est-à-dire que les nœuds similaires ont tendance à être connectés. La **collective classification** résout ce problème en assignant des labels à tous les nœuds simultanément, où les nœuds proches ont la même couleur.

![[images/3-Apprentissage automatique/08_Graph ML/Message passing/1.introduction/im2.png]]
**Figure 1.** L'idée de la collective classification : étiqueter tous les nœuds ensemble en exploitant les corrélations.

> [!warning] Trois techniques principales
> 1. **Relational classification**
> 2. **Iterative classification**
> 3. **Belief propagation**

### B. Sources de corrélation

Deux types principaux de dépendances qui mènent à des corrélations :

> [!warning] Homophilie (Homophily)
> La tendance des individus à s'associer et créer des liens avec des **autres similaires**.
>
> - *Exemple 1.* Des chercheurs travaillant dans le même domaine sont plus susceptibles d'établir des connexions (rencontres en conférences, échanges en séminaires...).
> - *Exemple 2.* Dans un réseau social en ligne, les nœuds sont des personnes, les arêtes des amitiés, les couleurs des centres d'intérêt (sports, arts...). Les personnes partageant des intérêts communs sont plus densément connectées par homophilie.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/im3 (1).png]]

> [!warning] Influence
> Les **connexions sociales peuvent influencer** les caractéristiques individuelles.
>
> - *Exemple.* Je recommande mes goûts musicaux à mes amis, jusqu'à ce que l'un d'eux finisse par aimer mon genre préféré.

### C. Hypothèses du SSL sur graphes

> [!warning] Définition (SSL sur graphes)
> Suivant la définition initiale, on suppose qu'**il y a de l'homophilie dans le réseau**. Le label d'un nœud $v$ peut alors dépendre de :
> - Ses propres **features**.
> - Les **labels** des nœuds dans le voisinage $N_v$ de $v$.
> - Les **features** des nœuds dans $N_v$.

> [!warning] Définition (Collective Classification)
> Classification **simultanée** de nœuds interconnectés en exploitant les corrélations. On utilise l'**hypothèse de Markov** : le label $Y_v$ d'un nœud $v$ ne dépend que des labels de ses voisins $N_v$ :
>
> $$P(Y_v) = P(Y_v \mid N_v).$$

L'algorithme se déroule en trois étapes :

1. **Local Classifier** — assignation initiale des labels. Prédit le label à partir des features du nœud avec un classifieur standard.
2. **Relational Classifier** — capture les corrélations. Apprend à étiqueter un nœud à partir des labels et/ou attributs de ses voisins. C'est ici qu'on utilise l'information du réseau.
3. **Collective Inference** — propage les corrélations en appliquant le relational classifier à chaque nœud **itérativement**. On itère jusqu'à minimiser l'inconsistance entre labels voisins.

> 💡 **Objectif graphique.** On cherche $P(Y_v)$ étant donnés toutes les features et la structure du réseau.

![[images/3-Apprentissage automatique/08_Graph ML/Message passing/1.introduction/im2.png]]

---

## II. Relational Classification

> [!warning] Définition
> La **probabilité de classe** $Y_v$ du nœud $v$ est une **moyenne pondérée** des probabilités de classe de ses voisins.

**Algorithme.**

1. Pour les nœuds labellisés $v$, initialiser $Y_v$ avec le ground-truth $Y_v^*$. Pour les nœuds non labellisés, initialiser $Y_v = 0.5$.
2. Mettre à jour tous les nœuds dans un ordre aléatoire jusqu'à convergence (ou max iterations). Pour chaque nœud $v$ et label $c$ :
>
> $$P(Y_v = c) = \frac{1}{\sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} A_{v, u} \cdot P(Y_u = c)$$
>
> où $A_{v, u}$ est le poids de l'arête entre $v$ et $u$ (si le graphe est pondéré), et $P(Y_v = c)$ la probabilité que $v$ ait le label $c$.

> 💡 **Limites.** Le relational classifier **n'utilise pas les features des nœuds** — uniquement les labels du voisinage. C'est ce qu'on corrige avec l'iterative classification (III).

> [!example] Exemple complet
>
> **(i) Initialisation.** Pour les nœuds labellisés, on utilise les ground-truth $Y$. Pour les non-labellisés, on initialise uniformément.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im1.png]]
>
> **(ii) Update — première itération.** Pour le nœud 3, $N_3 = \{1, 2, 4\}$ :
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im2.png]]
>
> Pour le nœud 4, $N_4 = \{1, 3, 5, 6\}$ :
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im3 (1).png]]
>
> Pour le nœud 5, $N_5 = \{4, 6, 7, 8\}$ :
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im4.png]]
>
> Après une itération complète :
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im5.png]]
>
> **(iii) Après itération 2.**
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im6.png]]
>
> **(iv) Après itération 3.**
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im7.png]]
>
> **(v) Après itération 4.** Tous les scores se stabilisent.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im8.png]]
>
> **Conclusion.**
> - Nœuds 4, 5, 8, 9 → classe 1 ($P_{Y_v} > 0.5$).
> - Nœud 3 → classe 0 ($P_{Y_v} < 0.5$).
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/2.Relational/im9.png]]

---

## III. Iterative Classification

### A. Framework

**Motivation.** Le relational classifier vu ci-dessus **n'utilise pas les features** des nœuds. L'iterative classification corrige ça.

> [!warning] Définition
> L'iterative classification utilise :
> - $f_v$ : vecteur de **features** du nœud $v$.
> - $Y_v$ : label du nœud $v$.
> - $z_v$ : **résumé des labels** des voisins $N_v$ de $v$ — par exemple, un vecteur du nombre/fraction de chaque label dans $N_v$, ou le label le plus commun, ou le nombre de labels distincts...

**Pseudo-code (Iterative Classification).**

> [!note]- Algorithme en deux phases
> **Phase 1 — entraînement.** Sur un training set, entraîner deux classifieurs (logistic regression, etc.) :
> - $\phi_1(f_v)$ : prédit $Y_v$ à partir de $f_v$ seul.
> - $\phi_2(f_v, z_v)$ : prédit $Y_v$ à partir de $f_v$ **et** du résumé $z_v$ des labels des voisins.
>
> **Phase 2 — inference itérative.** Sur le test set :
> 1. Initialiser les labels $Y_v$ à partir de $\phi_1$.
> 2. Calculer $z_v$ et raffiner les prédictions avec $\phi_2$.
> 3. Répéter pour chaque nœud $v$ :
>    - Mettre à jour $z_v$ à partir des $Y_u$ pour $u \in N_v$.
>    - Mettre à jour $Y_v = \phi_2(f_v, z_v)$.
>
> Itérer jusqu'à stabilisation des labels ou max iterations atteint.
>
> **Convergence non garantie.**

> [!example] Exemple — classification de pages web
>
> **Données.** Graphe de pages web ; nœuds = pages, arêtes = hyperliens (orientés). Features = description binaire de la page (2 features pour simplifier). Tâche = prédire le topic de la page.
>
> **(i) Baseline — features seules.** Classifieur linéaire sur les attributs binaires.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im1.png]]
>
> **(ii) Baseline avec features de voisinage.** Chaque nœud maintient des vecteurs $z_v$ de labels du voisinage :
> - $I$ = vecteur des labels des voisins **entrants**.
> - $O$ = vecteur des labels des voisins **sortants**.
>
> $I_0 = 1$ si au moins un voisin entrant a le label 0. Définitions similaires pour $I_1$, $O_0$, $O_1$.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im2.png]]
>
> **(iii) Étape 1 — entraînement des classifieurs.** Sur un training set séparé, entraîner :
> - $\phi_1$ : sur les features de nœud uniquement (cercles verts).
> - $\phi_2$ : sur features + vecteurs de liens (cercles rouges).
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im3 (1).png]]
>
> **(iv) Étape 2 — application sur le test.** Utiliser $\phi_1$ pour fixer les $Y_v$ initiaux.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im4.png]]
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im5.png]]
>
> **Étape 3.1 — Update $z_v$ pour tous les nœuds.**
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im6.png]]
>
> **Étape 3.2 — Re-classifier tous les nœuds avec $\phi_2$.**
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im7.png]]
>
> **Itération.** Continuer jusqu'à convergence : update $z_v$, update $Y_v = \phi_2(f_v, z_v)$.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im8.png]]
>
> **Prédiction finale.** Stop après convergence ou max iterations.
>
> ![[images/3-Apprentissage automatique/08_Graph ML/Message passing/3.Iterative/im9.png]]

### B. Application — détection de faux avis (REV2)

> 💡 **Référence.** *REV2: Fraudulent User Predictions in Rating Platforms*, Kumar et al. 2018.

**Le problème.** Les sites d'avis sont une cible attractive pour le spam : une augmentation d'1 étoile sur la note moyenne fait grimper le revenu de **5-9%**. Malheureusement, certains scores sont artificiellement gonflés par des spammeurs payés qu'on cherche à détecter.

**Approche naïve — ne marche pas bien.**
- **Behavioral Analysis** : features individuelles, géolocalisation, heures de connexion, historique de session.
- **Language Analysis** : usage de superlatifs, auto-références fréquentes, taux de fautes d'orthographe, mots d'accord exagérés...

> 💡 **Limite.** Ces deux méthodes sont **faciles à contourner** par des fraudeurs un peu sophistiqués.

**Approche graphe.** On capture les **relations** entre reviewers, reviews et magasins. L'idée centrale : utilisateurs, produits et notes ont des **scores de qualité intrinsèques** qu'on apprend par iterative classification.

**Setup.** Le graphe est **biparti pondéré et signé** :
- **Nœuds** = utilisateurs et produits.
- **Arêtes** = scores de notation entre $-1$ (rouge) et $+1$ (vert).

**Sortie** = ensemble d'utilisateurs donnant des fausses notes.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im18.png]]

**Trois quantités à apprendre simultanément.**

> [!warning] Fairness score $F(u)$ d'un utilisateur
> En fixant Goodness et Reliability, on calcule la fairness comme la moyenne de la reliability des reviews de l'utilisateur :
>
> $$F(u) = \frac{\sum_{(u, p) \in \text{Out}(u)} R(u, p)}{|\text{Out}(u)|}$$

> [!warning] Goodness $G(p)$ d'un produit
> La goodness d'un produit pondère le score de chaque review par sa reliability — les reviews peu fiables n'impactent pas le score final :
>
> $$G(p) = \frac{\sum_{(u, p) \in \text{In}(p)} R(u, p) \cdot \text{score}(u, p)}{|\text{In}(p)|}$$

> [!warning] Reliability $R(u, p)$ d'une note
> En fixant Fairness et Goodness, on mesure à quel point la note diverge de l'opinion commune sur ce produit.
>
> ![[images/3-Apprentissage automatique/05_Generative Models/score based/im19.png]]

**Itérations.**
1. Initialiser $F(u)$, $R(u, p)$, $G(p)$ au max ($= 1$).
2. Appliquer la formule de $G(p)$.
3. Appliquer celle de $R(u, p)$.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im20.png]]

4. Appliquer celle de $F(u)$.
5. Répéter jusqu'à convergence.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im21.png]]

> 💡 **Propriétés.** L'algorithme **converge garanti**, le nombre d'itérations est **borné**, et la complexité est **linéaire** en le nombre d'arêtes du graphe.

**Performance.** Dataset Flipkart (Inde). Sur le top 80, **précision = 100%**. **127 des 150 utilisateurs avec la fairness la plus basse** étaient effectivement des fraudeurs réels.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im22.png]]

---

## IV. Loopy Belief Propagation

### A. Algorithme

**Vue d'ensemble.** Loopy Belief Propagation est une approche par **programmation dynamique** pour répondre à des conditional probability queries dans un modèle graphique. C'est un processus **itératif** où les nœuds voisins "se parlent" en se passant des messages, jusqu'à atteindre un consensus.

> 💡 **Lien avec PGM.** C'est exactement le même algorithme que dans [[03_Inférence#III. Belief Propagation]] côté PGM. La différence : ici on l'applique à des graphes de données réelles (réseau social, bipartite reviewer/produit, etc.) plutôt qu'à un graphe de variables aléatoires.

#### A.1 Message Passing — exemple intuitif

> [!example] Tâche — compter les nœuds dans un graphe
> Chaque nœud ne peut interagir qu'avec ses voisins directs.
>
> **Cas linéaire.**
>
> ![[images/3-Apprentissage automatique/05_Generative Models/score based/im23.png]]
>
> **Solution :** chaque nœud écoute le message de son voisin, l'incrémente, et le passe au suivant.
>
> ![[images/3-Apprentissage automatique/05_Generative Models/score based/im24.png]]
>
> Zoom sur un nœud :
>
> ![[images/3-Apprentissage automatique/05_Generative Models/score based/im25.png]]
>
> **Cas arborescent.** Chaque nœud reçoit des reports de toutes les branches de l'arbre.
>
> ![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im30.png]]

> 💡 **En pratique.** Dans cet exemple les nœuds renvoient des **valeurs exactes**, mais en cas réel ce sont des **probabilités** sur de très grands graphes — par exemple, "il y a environ un million de nœuds dans cette branche".

#### A.2 Algorithme Loopy BP

**Initialisation.** Tous les messages initialisés à 1.

**Itération.** Répéter pour chaque label $Y_j \in \mathcal{L}$, pour tous les nœuds :

![[images/3-Apprentissage automatique/05_Generative Models/score based/im26.png]]

> [!warning] Trois types de paramètres
> ![[images/3-Apprentissage automatique/05_Generative Models/score based/im28.png]]
>
> - **Label-Label potential matrix** $\psi$ : dépendance entre un nœud et son voisin. $\psi(Y_i, Y_j)$ = probabilité qu'un nœud $j$ soit dans l'état $Y_j$ sachant qu'il a un voisin $i$ dans l'état $Y_i$.
> - **Prior belief** $\phi$ : probabilité $\phi_i(Y_i)$ que le nœud $i$ soit dans l'état $Y_i$.
> - **Message** $m_{i \to j}(Y_j)$ : estimation par $i$ de la probabilité que $j$ soit dans l'état $Y_j$.
> - $\mathcal{L}$ : ensemble de tous les états.
> - $\alpha$ : joue le rôle d'un learning rate.

**Convergence.** $b_i(Y_i)$ = belief finale de $i$ d'être dans l'état $Y_i$.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im27.png]]

#### A.3 Que peut-il mal se passer ?

> [!warning] Le problème des cycles
> ![[images/3-Apprentissage automatique/01_Supervised-Learning/00_Neural_nets_MLP/im29.png]]
>
> Si le graphe a des **cycles**, les messages venant de différents sous-graphes ne sont **plus indépendants**. Mais on peut quand même faire tourner BP : c'est un algorithme local qui ne "voit" pas les cycles.
>
> Les messages tournent en boucle : 2, 4, 8, 16, 32... Le modèle devient de plus en plus convaincu qu'une variable est dans l'état $T$. **BP traite incorrectement ces messages comme des évidences indépendantes**, alors qu'ils viennent en fait de la même information qui a circulé.

> 💡 **En pratique ça marche quand même souvent.** C'est un cas extrême — souvent dans la pratique les influences cycliques sont **faibles** parce que les cycles sont longs ou contiennent au moins une corrélation faible.

#### A.4 Avantages et limites

> [!warning] Avantages
> - **Facile à programmer et paralléliser**.
> - Applicable à **n'importe quel modèle graphique**, quelle que soit la forme des potentiels (pas seulement pairwise).

> [!warning] Limites
> - **Convergence non garantie** (quand s'arrêter ?), surtout si beaucoup de cycles fermés.
> - Les **fonctions de potentiel** (paramètres $\psi$) requièrent un entraînement par optimisation par gradient — avec ses propres problèmes de convergence.

### B. Application — fraude aux enchères en ligne (NetProbe)

> 💡 **Référence.** *Netprobe: A Fast and Scalable System for Fraud Detection in Online Auction Networks*, Pandit et al. 2007.

**Le contexte.** Les sites d'enchères sont une cible majeure de fraude : **63%** des plaintes au FBI Internet Crime Complaint Center en 2006, perte moyenne de **\$385** par incident. La fraude la plus courante est la **non-delivery fraud** : le buyer paie mais ne reçoit jamais l'item.

**Approche par features individuelles.** Échoue souvent (mêmes raisons que pour les fake reviews — facile à contourner).

**Approche par structure du graphe.** Les utilisateurs se donnent mutuellement un **reputation score** (plus haut = mieux). Les fraudeurs ont **deux stratégies** :

1. **Boost mutuel** (clique de fraudeurs). Inefficace — si on en détecte deux dans la clique, on trouve tout le monde.
2. **Near-bipartite cores** avec deux rôles :
   - **Accomplices** (jaune) : transactionnent avec des honnêtes, semblent légitimes.
   - **Fraudsters** (rouge) : transactionnent avec les accomplices, fraudent les honnêtes.

![[im31.png]]
**Figure 2.** Structure bipartite typique des fraudeurs en enchères en ligne.

**Objectif.** Trouver les rôles (honest, accomplice, fraudster) via Loopy BP. On définit les paramètres $\psi$ comme suit :

![[images/3-Apprentissage automatique/05_Generative Models/score based/im32.png]]
**Figure 3.** Label-label potential matrix encodant la structure bipartite fraudeur/accomplice.

**Résultat après plusieurs itérations.**

![[images/3-Apprentissage automatique/05_Generative Models/score based/im33.png]]
**Figure 4.** Sortie de l'algorithme — identification des rôles dans le graphe.

> 💡 **L'idée à retenir.** Quand la fraude se cache derrière une **structure relationnelle** plutôt que dans des features individuelles (cas typique de la fraude organisée), les algorithmes de message passing sur graphe sont l'outil naturel — et historiquement très utilisés en détection de fraude bancaire, e-commerce et antiblanchiment.
