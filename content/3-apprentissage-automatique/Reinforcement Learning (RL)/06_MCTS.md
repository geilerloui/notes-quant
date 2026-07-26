---
title: MCTS — Monte Carlo Tree Search
---
# MCTS — Monte Carlo Tree Search

> Quand l'espace d'états est trop grand pour être énuméré (Go a $10^{172}$ états — plus que d'atomes dans l'univers), on ne peut pas construire l'arbre de jeu complet. Le **Monte Carlo Tree Search** contourne ce problème en construisant un arbre **partiel** qu'on étend itérativement par simulations stochastiques, en équilibrant exploration et exploitation via UCB1. C'est la brique algorithmique derrière AlphaGo / AlphaZero.

> Pré-requis : [[01_RL Tabulaire]] (MDP, Bellman), [[04_Exploration (Bandits)]] (UCB1), [[02_Function Approximation (Deep RL)]] (réseaux profonds pour le couplage MCTS + NN).

> 💡 **Trois grandes parties** :
> - **I. Models and Planning** — distinction model-based vs model-free.
> - **II. Vanilla MCTS** — les 4 phases (Selection, Expansion, Simulation, Backpropagation), UCB1, déroulé pas-à-pas tic-tac-toe.
> - **III. MCTS + Neural Networks** — la formule PUCT, AlphaGo Zero, génération de données d'entraînement.

---

## I. Planning and Learning with Tabular Methods

### A. Vue unifiée model-based / model-free

> [!warning] Distinction conceptuelle
> - **Model-based** (DP, heuristic search) — on dispose d'un modèle de l'environnement $(P, R)$, et on **planifie** (look-ahead).
> - **Model-free** (Monte Carlo, TD) — pas de modèle, on **apprend** depuis l'expérience.

> 💡 **Le cœur commun.** Les deux familles calculent des **value functions** par look-ahead → backed-up value → update target. Cette note explore comment elles se mélangent en pratique.

### B. Models and Planning

*À développer.* (Mini-orientation : la planification utilise un modèle pour générer des "trajectoires simulées" à la place de trajectoires réelles. Si le modèle est exact, planifier ≡ Bellman backup. Si le modèle est appris, on a Dyna et ses variantes — voir section finale.)

---

## II. Monte Carlo Tree Search — Vanilla

### A. Motivation — Game Trees

Un **game tree** : nœuds = états du jeu, arêtes = actions possibles. Avec un arbre **complet**, calculer la meilleure action devient trivial.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im3 (1).png]]

> 💡 **Le problème.** L'arbre complet est faisable seulement avec un faible *branching factor*. Pour Go ($10^{172}$ états), aucun ordinateur ne peut tout simuler. **MCTS construit un arbre partiel** par simulations.

### B. Setup — exemple HAL et tic-tac-toe

> [!example] HAL joue au tic-tac-toe
> Notre AI (HAL) réfléchit à son premier coup. Son arbre actuel :
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im4.png]]
>
> Chaque nœud porte deux valeurs :
> - $\boldsymbol n$ — nombre de fois où l'état a été considéré (visit count).
> - $\boldsymbol w$ — nombre de victoires depuis cet état pour le joueur qui vient de jouer.
>
> Les nœuds foncés concernent X, les clairs O.

### C. Les quatre phases

> [!warning] Les 4 phases de MCTS (cycle répété)
> 1. **Selection** (tree traversal) — descendre dans l'arbre jusqu'à un nœud feuille.
> 2. **Expansion** — créer un ou plusieurs enfants.
> 3. **Simulation** (rollout) — jouer aléatoirement jusqu'à fin de partie.
> 4. **Backpropagation** — remonter le résultat dans l'arbre.

#### C.1 Phase 1 — Selection (UCB1)

> [!warning] Formule UCB1
> Pour balancer exploitation et exploration :
>
> $$\boxed{UCB_1(s_i) = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}}$$
>
> - $w_i$ — wins comptés pour ce nœud.
> - $n_i$ — visit count du nœud.
> - $N$ — somme des visit counts des frères atteignables.
> - $c$ — hyperparamètre. Plus haut → plus d'exploration. Typiquement proche de 1.

> 💡 **Lien avec [[04_Exploration (Bandits)]].** UCB1 vient directement de la théorie des bandits — chaque enfant d'un nœud est traité comme un bras d'un bandit avec sa propre estimation et sa propre incertitude.

#### C.2 Phase 2 — Node Expansion

Quand HAL atteint un nœud feuille avec assez d'info, on **étend** l'arbre. On choisit une action :
- **Light rollout** — action aléatoire (utilisé en vanilla MCTS).
- **Heavy rollout** — heuristique métier ou NN qui évalue la position. Convergence plus rapide, mais coûteux.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im5.png]]

#### C.3 Phase 3 — Simulation (rollout)

On **simule** la fin de la partie depuis le nœud courant, actions aléatoires, jusqu'à fin de partie.

> 💡 **Important.** Les états visités pendant la simulation ne sont **pas ajoutés** à l'arbre. La simulation sert juste à évaluer la position du nœud courant.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im6.png]]

#### C.4 Phase 4 — Backpropagation

On remonte le résultat le long du chemin parcouru :
- Incrémenter $n$ pour chaque nœud du chemin.
- Mettre à jour $w$ uniquement pour les nœuds correspondant au joueur ayant gagné.

### D. Pseudo-code

> [!note]- Algorithme principal (selection + expansion)
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im1.png]]

> [!note]- Algorithme du rollout
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im2.png]]

### E. Faire un coup en pratique

Après quelques secondes, HAL a itéré ces 4 phases des **milliers de fois**. Chaque nœud a vu son $n$ et $w$ mis à jour de nombreuses fois. Voici les premiers nœuds d'un MCTS réel sur tic-tac-toe :

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im7.png]]

### F. Exemple complet pas-à-pas

> [!example] Étape 1 — Initialisation
> État initial $s_0$ avec $t_0 = 0$ et $n_0 = 0$.
>
> ![[example-1.png]]

> [!example] Étape 2 — Première expansion
> Pas de leaf node initialement → on étend l'arbre et on initialise les enfants.
>
> ![[example-2.png]]

> [!example] Étape 3 — Premier rollout
> UCB des deux enfants = $\infty$ (jamais visités), donc on choisit aléatoirement $a_1$. Rollout → terminal node $v = 20$, on backtrack.
>
> ![[example-3.png]]

> [!example] Étape 4 — Deuxième visite à la racine
> On recalcule les UCB depuis $s_0$ :
>
> $$UCB_1(s_1) = 20 + 2\sqrt{\frac{\ln 1}{1}} = 20, \quad UCB_1(s_2) = \infty.$$
>
> On choisit $a_2$, simulation, backup.
>
> ![[example-4.png]]

> [!example] Étape 5 — Troisième visite à la racine
> $$UCB_1(s_1) = 20 + 2\sqrt{\frac{\ln 2}{1}} = 21.67, \quad UCB_1(s_2) = 10 + 2\sqrt{\frac{\ln 2}{1}} = 11.67.$$
>
> On va à gauche ($s_1$), expansion des enfants.
>
> ![[example-5.png]]

> [!example] Étape 6 — Rollout depuis le nouvel enfant
> Les deux nœuds en bas à gauche ont UCB $= \infty$, on choisit $a_3$, rollout, backtrack.
>
> ![[example-6.png]]

> [!example] Étape 7 — Quatrième visite à la racine
> $$UCB_1(s_1) = 10 + 2\sqrt{\frac{\ln 3}{2}} = 11.48, \quad UCB_1(s_2) = 10 + 2\sqrt{\frac{\ln 3}{1}} = 12.10.$$
>
> Cette fois on va à droite ($s_2$), expansion.
>
> ![[example-7.png]]

> [!example] Étape 8 — Simulation et backtrack final
> ![[example-8.png]]

> 💡 **Ce qu'on observe.** Les valeurs UCB s'équilibrent automatiquement. Au début un nœud non visité est priorisé ($\infty$). Une fois visité, son score reflète à la fois sa valeur moyenne et sa fréquence de visite. Plus un nœud est visité, plus son terme d'exploration $\sqrt{\ln N / n_i}$ diminue → on bascule progressivement vers l'exploitation.

---

## III. Integrating MCTS and Neural Networks

> 💡 **Référence.** AlphaGo (Silver et al. 2016) puis AlphaGo Zero (Silver et al. 2017). C'est le mariage MCTS + Deep NN qui a battu les meilleurs humains au Go.

### A. Le shift conceptuel — les arêtes deviennent porteuses

En vanilla MCTS, les **nœuds** stockaient $n$ et $w$. Avec NN, on déplace l'info sur les **arêtes** :
- $N(s, a)$ — visit count de l'arête.
- $P(s, a)$ — prior probability donnée par le NN.
- $W(s, a)$ — valeur intermédiaire accumulée.
- $Q(s, a)$ — action-value (analogue à $w/n$ en vanilla).

Les nœuds stockent juste un "state value".

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im8.png]]

### B. Phase 1 — Selection avec PUCT

> [!warning] Formule PUCT (Polynomial UCT)
> $$\boxed{\arg\max_a \left[ Q(s, a) + c_{puct} \, P(s, a) \, \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right]}$$
>
> Très similaire à UCB1 :
> - $Q(s, a)$ = exploitation (analogue à $w/n$).
> - Le 2e terme = exploration **pondérée par le prior** $P(s, a)$ du NN.
> - $c_{puct}$ = paramètre d'exploration.

> 💡 **L'innovation clé.** Le NN guide l'exploration via $P(s, a)$ — on n'explore plus uniformément, mais préférentiellement vers les actions que le NN juge prometteuses.

> 💡 **Dirichlet noise.** Pour éviter que des priors trop forts écrasent toute randomness, on ajoute du bruit Dirichlet **uniquement à la racine** (AlphaGo). Ça garantit qu'on explore quand même un peu, sans casser les priors profonds.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im9.png]]

### C. Phase 2 — Expansion / Evaluation

Quand on atteint une feuille, on l'évalue avec le NN :
- Output 1 : **vecteur de probabilités** $p_a$ pour chaque action.
- Output 2 : **valeur scalaire** $v$ de l'état (estimation de qui va gagner).

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im10 (1).png]]

> [!example] Tic-tac-toe
> NN prédit state value $v = 0.6$ et 9 logits pour les 9 cases possibles. Ces logits deviennent les priors $P(s, a)$ des nouvelles arêtes.

> 💡 **Validité des actions.** Le NN sort toujours 9 valeurs, mais certaines actions sont invalides (case déjà jouée). Le masquage des actions invalides est fait par l'implémentation MCTS.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im11.png]]

### D. Phase 3 — Update

Pour chaque arête $e_t$ traversée jusqu'à la feuille :

> [!warning] Trois updates par arête
> 1. **Visit count** : $N(s, a) \leftarrow N(s, a) + 1$.
> 2. **Cumul** : si le joueur owner de $e_t$ est le même que celui de la feuille, $W \leftarrow W + v$. Sinon $W \leftarrow W - v$.
> 3. **Action-value** : $Q(s, a) \leftarrow W / N$.

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im12.png]]

### E. Itération complète — un deuxième tour

> [!note]- Cycle complet
> **(1) Select with PUCT until a leaf.**
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im13.png]]
>
> **(2) Expand and evaluate the leaf.**
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im14.png]]
>
> **(3) Update all traversed edges.**
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im15.png]]

### F. Choix final de l'action

Après plusieurs milliers d'itérations, on choisit l'action selon :

$$\boxed{\pi_t(a) = \frac{N(s, a)^{1/\tau}}{\sum_b N(s, b)^{1/\tau}}}$$

Le paramètre **temperature** $\tau$ contrôle le compromis :
- $\tau \to 0$ : choix **greedy** par rapport au visit count.
- $\tau = 1$ : exploration plus permissive des actions moins visitées.

> 💡 **En pratique AlphaGo Zero.** Au début de la partie, $\tau = 1$ (exploration). Vers la fin, $\tau \to 0$ (exploitation pure).

### G. Utiliser MCTS comme données d'entraînement

> 💡 **L'idée révolutionnaire d'AlphaGo Zero.** MCTS lui-même génère les données pour entraîner le NN — pas besoin de jeux humains.

Après chaque partie complète (self-play) :

![[images/3-Apprentissage automatique/07_Reinforcement learning/MCTS/im16.png]]

> [!warning] Datapoints générés par coup
> Pour chaque coup joué :
> - **Input state** $s$.
> - **Target value** = qui a finalement gagné (+1 / -1).
> - **Predicted value** $v$ — sortie du NN à ce coup.
> - **Predicted policy** = priors $P(s, a)$ produits par le NN.
> - **Target policy** = $\pi_t$ calculé depuis MCTS (les *search probabilities*).

Le NN est entraîné à approcher conjointement target value et target policy.

### H. Deux réseaux — data generation vs training

> [!warning] Pourquoi deux réseaux
> On **n'entraîne pas** le NN qui guide MCTS en temps réel. On a :
> - **Data generation network** — utilisé dans MCTS pour générer les données.
> - **Training network** — entraîné sur ces données.
>
> Tous les $T$ steps (1000 dans AlphaGo Zero), un **tournament MCTS** oppose les deux réseaux. Si le data generation network perd, il est remplacé par le training network.

> 💡 **Pourquoi cette séparation.** Évite que le NN ne dérive pendant qu'il génère des données — instabilité catastrophique typique du RL on-policy. Le training network "challenger" doit prouver sa supériorité avant de devenir le champion.

---

## IV. Integrated Architecture — Dyna

*À développer.*

> 💡 **Mini-orientation Dyna (Sutton 1990).** Combine planning, acting, learning dans une seule architecture :
> - **Acting** — agent interagit avec l'environnement réel, génère des transitions.
> - **Learning** — modèle de l'environnement appris depuis ces transitions ($P, R$ estimés).
> - **Planning** — Q-learning supplémentaire sur des transitions **simulées** générées par le modèle appris.
>
> Le planning utilise les "vraies" transitions ET des transitions imaginées → bien plus data-efficient. Variantes : Dyna-Q, Dyna-Q+ (avec exploration bonus), Prioritized Sweeping.

---

## Annexe — récapitulatif

| Variante | Selection | Priors | Évaluation feuille | Cas d'usage |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla MCTS** | UCB1 | Aucun | Random rollout | Petits jeux, baseline |
| **MCTS + heavy rollout** | UCB1 | Aucun | Heuristique métier | Jeux moyens |
| **AlphaGo (2016)** | PUCT | NN priors | NN value + rollout | Go pro level |
| **AlphaGo Zero (2017)** | PUCT | NN priors | NN value (no rollout) | Go superhumain |
| **AlphaZero (2018)** | PUCT | NN priors | NN value | Go, échecs, shogi |
| **MuZero (2019)** | PUCT | NN priors | NN value (modèle appris) | Atari + jeux de plateau, sans règles |

> 💡 **Le résumé en une phrase.** MCTS est l'algorithme clé qui permet à un agent de **planifier dans des espaces gigantesques** en construisant un arbre **partiel et focalisé**, et le couplage avec un NN (PUCT + value/policy network) est la recette qui a permis à DeepMind de dépasser le niveau humain sur Go puis tous les jeux à information parfaite.

> 💡 **Lien avec la finance quantitative.** Les variantes MCTS sont parfois utilisées pour le **portfolio rebalancing en présence de coûts de transaction** ou pour des **stratégies de market-making en RL** (espace d'action continu, planification multi-step). Pas un standard industriel, mais ça apparaît dans la recherche.
