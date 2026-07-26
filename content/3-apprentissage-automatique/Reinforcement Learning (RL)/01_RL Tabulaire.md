---
title: Méthodes tabulaires - RL
---
# Méthodes tabulaires

> Cette première partie couvre le RL dit *tabulaire* (tabular RL) : on suppose que l'espace d'états et l'espace d'actions sont suffisamment petits pour qu'on puisse représenter explicitement les fonctions de valeur dans un tableau, sans approximation. C'est le cadre de Sutton & Barto Partie I, et le socle conceptuel sur lequel s'appuieront ensuite les méthodes par approximation de fonction (Deep RL).

## I. Cadre formel

### A. Introduction

**Cadre général.** Le RL formalise le problème de la **prise de décision séquentielle** (sequential decision making) : un agent interagit avec un environnement en choisissant à chaque pas de temps $t$ une action $a_t$, observe $o_t$ et reçoit une récompense scalaire $r_t$. L'historique (history) à l'instant $t$ est la séquence

$$h_t = (o_0, a_0, r_1, o_1, a_1, r_2, \ldots, o_t)$$

et la politique (policy) de l'agent peut être vue comme une fonction $a_{t} = \pi(h_{t-1})$ qui décide de l'action suivante à partir du passé.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/im1.png|391]]
**Figure 1.** Vue d'ensemble de l'interaction agent–environnement.

Ce cadre se distingue de l'apprentissage supervisé sur deux points essentiels :

- la cible n'est pas une étiquette correcte mais une récompense, potentiellement **différée** dans le temps — d'où le problème d'**attribution du crédit** (credit assignment) : à quelle action passée doit-on attribuer la récompense reçue maintenant ?
- l'agent génère lui-même ses données par ses actions, ce qui crée le compromis **exploration/exploitation** (exploration-exploitation tradeoff) : essayer des actions nouvelles pour mieux connaître l'environnement, ou exploiter ce qu'on sait déjà.

**Observabilité (observability).** Soit $S$ l'ensemble des états possibles du monde et $\{s_t\}$ la trajectoire d'états. Deux régimes :

- **Cas totalement observable** (fully observable) : $o_t = s_t$, l'agent voit l'état réel du monde. C'est le cadre du **MDP** (Markov Decision Process), qui sera l'hypothèse par défaut dans toute cette partie.
- **Cas partiellement observable** (partially observable) : $o_t \neq s_t$, l'agent ne voit qu'une observation bruitée ou incomplète de l'état réel. Pour décider, il maintient alors une distribution de probabilité sur l'état réel, appelée **état de croyance** (belief state). Ce cadre est modélisé par un **POMDP** (Partially Observable MDP), traité plus loin.

**Pourquoi le RL plutôt que le *search* ?** (*search* = recherche algorithmique dans un arbre/graphe, comme en IA classique — à ne pas confondre avec "recherche scientifique"). Quand le modèle de l'environnement (la dynamique et les récompenses) est entièrement connu et que l'espace d'états est petit et déterministe, des méthodes de *search* classiques en IA comme $A^*$ ou minimax suffisent à trouver une séquence d'actions optimale — en explorant l'arbre des possibilités, sans rien apprendre. Le RL devient nécessaire dès qu'**au moins une** des trois difficultés suivantes apparaît :

1. **Espace d'états trop grand** (jeux Atari, Go, problèmes de contrôle continus) — la recherche exhaustive est infaisable, il faut généraliser via une approximation de fonction (function approximation).
2. **Stochasticité** de la dynamique — il n'existe alors plus de "meilleure séquence d'actions", mais une politique optimale qui mappe états vers distributions sur actions.
3. **Modèle inconnu** — on ne connaît ni les probabilités de transition $p(s' \mid s, a)$ ni la fonction de récompense $r(s, a)$, et il faut les apprendre par interaction. C'est la motivation centrale du RL.

### B. Processus de décision markovien fini

On construit le MDP (Markov Decision Process) en trois temps : on part d'un processus de Markov (juste de la dynamique), on lui ajoute une récompense pour obtenir un MRP (Markov Reward Process), puis on ajoute des actions pour obtenir le MDP. Cette progression rend chaque ingrédient explicite et permet de comprendre où chaque hypothèse intervient.

> [!example] Fil rouge : FrozenLake
> Pour ancrer chaque définition, on utilisera tout au long de cette note l'environnement classique **FrozenLake** (Sutton & Barto / Gymnasium) : une grille $4\times4$ où l'agent part de la case **S** (départ) et doit atteindre la case **G** (objectif) sans tomber dans une case **H** (trou, fin d'épisode, récompense nulle). Les autres cases **F** sont de la glace praticable. Complication : la glace est **glissante** — quand l'agent choisit une direction, il n'y a que $1/3$ de chance qu'il parte réellement dans cette direction ; avec probabilité $1/3$ chacune, il part dans l'une des deux directions perpendiculaires à la place. C'est donc un environnement **stochastique**, à $16$ états et $4$ actions (Gauche, Bas, Droite, Haut), qui illustre bien mieux l'incertitude du RL qu'un exemple déterministe.
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_grid.png|280]]

#### Processus de Markov

> [!warning] Propriété de Markov
> Un processus stochastique $(s_0, s_1, s_2, \ldots)$ à valeurs dans un espace d'états $S$ vérifie la **propriété de Markov** (Markov property) si
> 
> $$P(s_{t+1} \mid s_t, s_{t-1}, \ldots, s_0) = P(s_{t+1} \mid s_t)$$
> 
> autrement dit : *l'état présent contient toute l'information utile pour prédire le futur*. Une fois qu'on connaît $s_t$, le passé n'apporte rien de plus. C'est ce qu'on résume en disant qu'un processus de Markov est **sans mémoire** (memoryless).

> [!note] À ce stade, pas encore d'action
> FrozenLake est nativement un MDP (il faut choisir une direction pour bouger). Pour illustrer un **pur** processus de Markov (sans action, cf. la progression Markov → MRP → MDP annoncée plus haut), on fixe artificiellement un comportement de référence — ici l'agent tire une direction **uniformément au hasard** parmi les 4 à chaque case. Ce choix fixé "absorbe" l'action dans la dynamique (exactement le mécanisme du callout *MDP + politique fixée = MRP* qu'on formalisera plus loin) et donne un pur $P(s' \mid s)$, sans action visible.

**Figure 2.** Étant donné $s_t = 0$ (la case départ, en vert), la distribution du prochain état $s_{t+1}$ est entièrement spécifiée par les probabilités de transition (glace glissante + direction aléatoire) — les états passés (colonne $t-1$, grisée) n'ajoutent aucune information supplémentaire.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_markov_property_v2.png|600]]

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_transition.png|350]]
*(Même transition, vue sur la grille plutôt qu'en graphe abstrait — utile pour garder le repère spatial.)*

**Figure 2bis.** La même transition, sous la forme "graphe de Markov classique" (rond = état, flèche = probabilité de transition) — plus abstraite que la vue grille ci-dessus, mais c'est la représentation standard des chaînes de Markov dans la littérature. Avec $16$ états on ne peut pas tout dessiner ; les "…" indiquent que le graphe continue de la même façon depuis $s_1$ et $s_4$.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_graphe_transition.png|400]]

**Hypothèses additionnelles.** Pour le cadre du RL tabulaire, on ajoute deux hypothèses standards :

- **Espace d'états fini** (finite state space) : $|S| < \infty$.
- **Transitions stationnaires** (stationary transitions) : les probabilités de transition ne dépendent pas du temps,

$$P(s_i = s' \mid s_{i-1} = s) = P(s_j = s' \mid s_{j-1} = s) \quad \forall s, s' \in S, \; \forall i, j.$$

> [!note]- Stationnarité des transitions ≠ processus stationnaire
> Dire que les transitions sont stationnaires signifie seulement que la *règle* de transition ne change pas avec le temps. Cela n'implique **pas** que la distribution de $s_t$ soit invariante en $t$ — celle-ci dépend en général de l'état initial et n'atteint la distribution stationnaire (si elle existe) qu'à la limite.

> [!warning] Matrice de transition
> Sous ces hypothèses, la dynamique est entièrement caractérisée par une matrice $\mathbf{P} \in \mathbb{R}^{|S| \times |S|}$ dont l'entrée $(i, j)$ est
> 
> $$P_{ij} = P(s_{t+1} = j \mid s_t = i),$$
> 
> c'est-à-dire la probabilité de passer en un pas de $i$ à $j$. Sous forme développée :
> 
> $$
> \mathbf{P} = \begin{pmatrix}
> P(s_1 \mid s_1) & P(s_2 \mid s_1) & \cdots & P(s_N \mid s_1) \\
> P(s_1 \mid s_2) & P(s_2 \mid s_2) & \cdots & P(s_N \mid s_2) \\
> \vdots & \vdots & \ddots & \vdots \\
> P(s_1 \mid s_N) & P(s_2 \mid s_N) & \cdots & P(s_N \mid s_N)
> \end{pmatrix}
> $$
> 
> C'est une matrice **stochastique en lignes** (row-stochastic) : ses entrées sont positives et chaque ligne somme à 1. Un processus de Markov est ainsi entièrement défini par le couple $(S, \mathbf{P})$.

**Lien définition–matrice.** Les deux objets précédents (propriété de Markov et matrice $\mathbf{P}$) décrivent la même chose sous deux angles. La propriété de Markov dit *qualitativement* qu'il suffit de connaître $s_t$ pour prédire $s_{t+1}$ ; la matrice $\mathbf{P}$ *stocke quantitativement* les probabilités conditionnelles à un pas $P(s_{t+1} \mid s_t)$ pour tous les couples $(s_t, s_{t+1}) \in S \times S$. Connaître $\mathbf{P}$, c'est connaître entièrement la dynamique du processus.

> [!example] FrozenLake — la matrice $\mathbf{P}$
> Avec $16$ états, $\mathbf{P}$ est $16\times16$ — trop grande pour être écrite intégralement. On se contente d'une ligne représentative : l'état $s=0$ (case départ), sous la politique de référence "direction uniformément aléatoire". Chaque action a $1/3$ de chance d'aboutir dans la direction voulue et $1/3$ chacune dans les deux directions perpendiculaires (heurter un mur laisse l'agent sur place) :
> 
> $$P(s' \mid s=0) = \begin{cases} 0.500 & s'=0 \text{ (reste sur place, mur à gauche/en haut)} \\ 0.333 & s'=4 \text{ (descend)} \\ 0.167 & s'=1 \text{ (droite)} \end{cases}$$
> 
> (Cette ligne mélange les 4 actions à parts égales ; le détail action par action est repris en I.B-MDP.) Comme pour toute chaîne de Markov, chaque ligne de $\mathbf{P}$ somme à $1$ — c'est la définition d'une matrice stochastique en lignes.

**Trajectoires (sample paths).** Une *réalisation* du processus est une suite d'états tirée selon $\mathbf{P}$ depuis un état initial. Tel quel, le processus de Markov ne fait que décrire une dynamique aléatoire — on n'a encore ni notion de "bien" ou "mal", ni de levier d'action. C'est ce qu'on ajoute dans les deux étapes suivantes.

> [!note]- Comment simule-t-on une trajectoire ?
> Deux choses différentes à ne pas confondre :
> 
> - **Propager une distribution.** Si $\mu^0$ est une distribution sur les états (par exemple $\mu^0 = [1, 0, \ldots, 0]$, un vecteur "one-hot" sur l'état $16$-dimensionnel pour "on part de la case $0$ avec certitude"), alors $\mu^k = \mu^0 \mathbf{P}^k$ donne la distribution à l'instant $k$. Cela décrit où l'on est *en moyenne sur tous les futurs possibles*. C'est l'angle utilisé par la programmation dynamique.
> 
> - **Échantillonner une trajectoire.** À chaque pas, on tire un état au hasard selon la ligne courante de $\mathbf{P}$ : sachant qu'on est en $s_t$, on échantillonne $s_{t+1} \sim \mathbf{P}[s_t, \cdot]$. C'est ainsi qu'on produit *une* réalisation concrète, comme la figure ci-dessous. C'est l'angle utilisé par les méthodes Monte Carlo.
> 
> **Lien entre les deux.** Si l'on simule des milliers de trajectoires et que l'on compte les fréquences des états à l'instant $k$, on retombe sur $\mu^k$ par la loi des grands nombres. Une trajectoire = un échantillon du processus ; la distribution = ce que voient toutes les trajectoires en moyenne. Cette tension entre *moyenner sur des échantillons* et *raisonner sur la distribution* est exactement la différence Monte Carlo / DP qu'on retrouvera plus loin.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_trajectoire.png|300]]

**Figure 4a.** Une trajectoire courte de 7 transitions, vue spatiale sur la grille, démarrant en $s=0$ : $0\to4\to4\to8\to9\to13\to14\to15$ — on voit le caractère glissant (le pas $0\to4\to4$ reste bloqué un tour, la glace ayant renvoyé l'agent sur sa case) et l'arrivée en $15$ (objectif). Cette trajectoire précise nous resservira telle quelle en III et IV (Monte Carlo, TD Learning) pour garder un exemple cohérent d'un bout à l'autre de la note.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_trajectoire_longue.png|700]]

**Figure 4b.** Vue "série temporelle" sur 120 pas (état $s_t$ en fonction du pas $t$, en escalier — le format qu'on utilisera pour lire les tables de valeurs $V_k$ plus loin). Contrairement à FrozenLake tout seul (qui est épisodique : l'épisode s'arrête net dans un trou ou à l'objectif), on **relance** un nouvel épisode en $s=0$ à chaque fin d'épisode pour obtenir un flux continu sur 120 pas, comme dans l'exemple du régime de marché original. On voit l'agent osciller autour du départ, retomber régulièrement dans un trou (rouge, reset immédiat), et de temps en temps atteindre l'objectif (vert, reset aussi) — sous la politique aléatoire de référence, les trous sont bien plus fréquents que l'objectif.

#### Markov Reward Process (MRP)

**Définition.** Un **MRP** (Markov Reward Process) est un processus de Markov enrichi d'une fonction de récompense et d'un facteur d'actualisation. Il est représenté par le quadruplet $(S, \mathbf{P}, R, \gamma)$ :

- $S$ : espace d'états fini.
- $\mathbf{P}$ : matrice de transition, $P(s' \mid s)$.
- $R$ : fonction de récompense, $R : S \to \mathbb{R}$.
- $\gamma \in [0, 1]$ : facteur d'actualisation (discount factor).

> 💡 **L'idée en une phrase.** Un MRP, c'est juste un processus de Markov auquel on ajoute un système de score : on gagne (ou on perd) des points en étant dans tel ou tel état. La dynamique reste la même, mais maintenant on peut poser la question "si je démarre ici, combien je vais gagner en moyenne sur le long terme ?". C'est ce que va capturer la fonction de valeur $V(s)$.

À chaque transition $s_t \to s_{t+1}$ est associée une récompense $r_t$, possiblement aléatoire. Un épisode (episode) du MRP s'écrit $(s_0, r_0, s_1, r_1, s_2, r_2, \ldots)$. La fonction $R(s)$ désigne par convention l'espérance de cette récompense sachant l'état :

$$R(s) = \mathbb{E}[r_t \mid s_t = s].$$

La récompense réelle $r_t$ peut donc être stochastique ; $R(s)$ en est l'espérance conditionnelle.

> [!note]- Convention d'indexation des récompenses
> Deux conventions coexistent dans la littérature :
> 
> - **Ici (et chez Silver, Stanford CS234)** : $r_t$ est la récompense observée à la transition issue de $s_t$. L'épisode s'écrit $(s_0, r_0, s_1, r_1, \ldots)$.
> - **Sutton & Barto** : la récompense reçue à la transition $s_t \to s_{t+1}$ est notée $R_{t+1}$. L'épisode s'écrit $(S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$.
> 
> Les deux sont équivalentes au décalage d'indice près. À retenir pour la lecture du livre de référence et de la majorité des papiers récents.

> [!example] FrozenLake — récompenses
> La récompense de FrozenLake est volontairement minimale : $+1$ pour la transition qui atteint la case $G$ (objectif), $0$ pour tout le reste — y compris tomber dans un trou. (On l'associe ici à l'état d'arrivée $s'$ plutôt qu'à l'état de départ $s$ ; cf. la note "trois définitions de la récompense" plus bas, qui couvre exactement ce choix de convention.)
> 
> $$R(s') = \begin{cases} +1 & \text{si } s' = 15 \text{ (objectif)} \\ 0 & \text{sinon} \end{cases}$$
> 
> C'est un exemple classique de **récompense éparse** (sparse reward) : sur les 16 états, un seul rapporte quelque chose, et il faut une longue séquence d'actions correctes pour l'atteindre. C'est justement ce qui rend FrozenLake pédagogiquement intéressant — contrairement à un signal dense (comme le rendement à chaque pas dans un exemple financier), l'agent doit apprendre à propager la valeur de $G$ vers les états lointains via l'équation de Bellman, sans retour d'information à chaque pas.

**Horizon (horizon).** L'horizon $H$ est le nombre de pas de temps d'un épisode. Il peut être fini ou infini. Le cas $H < \infty$ correspond aux MRP **finis** ; le cas $H = \infty$ aux MRP à horizon infini.

**Retour (return).** Le **retour** $G_t$ à partir de l'instant $t$ est la somme actualisée des récompenses futures :

$$G_t = \sum_{i=t}^{H-1} \gamma^{i-t} r_i \qquad \text{(horizon fini)}$$

$$G_t = \sum_{i=t}^{\infty} \gamma^{i-t} r_i \qquad \text{(horizon infini)}$$

C'est l'objet central de tout ce qui suit : la fonction de valeur sera définie comme l'espérance du retour, et les algorithmes chercheront soit à l'estimer, soit à le maximiser.

**Facteur d'actualisation $\gamma$.** Le rôle de $\gamma$ se justifie sur quatre plans, qu'il vaut la peine de garder en tête :

1. **Mathématique.** En horizon infini avec $\gamma = 1$, le retour peut diverger même si les récompenses sont bornées. Avec $\gamma < 1$ et $|r_t| \le R_{\max}$, on a la borne géométrique
$$|G_t| \le \frac{R_{\max}}{1 - \gamma}.$$
2. **Modélisation.** $\gamma$ encode une préférence pour l'immédiat. C'est exactement l'analogue d'un taux d'actualisation en finance (un euro aujourd'hui vaut plus qu'un euro demain).
3. **Incertitude future.** $\gamma$ peut s'interpréter comme la probabilité que le processus survive d'une étape à la suivante. Un $\gamma$ faible signifie qu'on doute du futur lointain.
4. **Algorithmique.** Avec $\gamma < 1$, l'opérateur de Bellman devient une **contraction** sur l'espace des fonctions de valeur. C'est ce qui garantira la convergence des algorithmes de programmation dynamique (Policy Iteration, Value Iteration). Ce point reviendra en force plus loin.

#### Markov Decision Process (MDP)

**Définition.** Un **MDP** (Markov Decision Process) est un MRP enrichi d'un espace d'actions. Il est représenté par le quintuplet $(S, A, \mathbf{P}, R, \gamma)$ :

- $S$ : espace d'états fini.
- $A$ : espace d'actions fini, disponible depuis chaque état.
- $\mathbf{P}$ : modèle de transition, $P(s' \mid s, a)$.
- $R$ : fonction de récompense, $R : S \times A \to \mathbb{R}$.
- $\gamma \in [0, 1]$ : facteur d'actualisation.

À chaque pas de temps $t$, l'agent observe $s_t$, choisit une action $a_t \in A$, reçoit une récompense $r_t$ et transite vers $s_{t+1}$. Un épisode s'écrit

$$(s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots).$$

La différence centrale avec le MRP est que la transition dépend désormais aussi de l'action :

$$P(s_{t+1} = s' \mid s_t = s, a_t = a).$$

L'hypothèse de stationnarité reste, étendue aux actions : la règle de transition ne dépend pas de $t$. La récompense espérée est définie par

$$R(s, a) = \mathbb{E}[r_t \mid s_t = s, a_t = a].$$

> [!note]- Trois définitions de la récompense coexistent
> Selon les sources, la fonction de récompense peut être notée $R(s)$, $R(s, a)$ ou $R(s, a, s')$ (la plus générale, où la récompense dépend aussi de l'état d'arrivée). Les trois sont équivalentes par espérance et le choix est souvent une question de commodité de notation. Ici on garde $R(s, a)$ comme convention par défaut.

Les notions d'horizon, de retour et de facteur d'actualisation introduites pour le MRP se reportent telles quelles dans le MDP.

> [!example] FrozenLake — actions et récompenses
> Les 4 actions sont les 4 directions : $A = \{\text{Gauche}, \text{Bas}, \text{Droite}, \text{Haut}\}$. Contrairement au MRP précédent, l'action change bel et bien la dynamique ($P(s' \mid s, a) \ne P(s' \mid s)$) : c'est elle qui détermine la direction *voulue*, même si la glace glissante en dévie parfois le résultat.
> 
> Prenons l'état $s=14$ (juste en-dessous de l'objectif $G=15$). L'espérance de récompense $R(s,a) = \sum_{s'} P(s'\mid s,a)\, r(s')$ pour chaque action :
> 
> | Action | Issues possibles ($1/3$ chacune) | $R(14, a)$ |
> |---|---|:---:|
> | Gauche | $10$, $13$, $14$ | $0.000$ |
> | Bas | $13$, $14$, $\mathbf{15}$ | $0.333$ |
> | Droite | $14$, $\mathbf{15}$, $10$ | $0.333$ |
> | Haut | $\mathbf{15}$, $10$, $13$ | $0.333$ |
> 
> Lecture : à cause de la glace glissante, Bas, Droite et Haut ont chacune *exactement une* de leurs 3 issues possibles qui tombe sur $G$ (en gras) — d'où la même espérance $1/3$ pour les trois, alors qu'aller à Gauche s'en éloigne et ne touche jamais $G$ depuis cet état. Intuitivement, la politique optimale ici est n'importe laquelle de {Bas, Droite, Haut}, mais sûrement pas Gauche — on le vérifiera formellement avec Policy/Value Iteration.

**Politique (policy).** Une **politique** $\pi$ est une règle de choix d'action. On distingue :

- **Politique déterministe** : $\pi : S \to A$, qui à chaque état associe une action $\pi(s)$.
- **Politique stochastique** : $\pi(a \mid s)$, distribution de probabilité sur $A$ conditionnée à $s$.

La politique stochastique englobe la politique déterministe (cas où la distribution est concentrée sur une seule action). On verra que dans un MDP fini avec $\gamma < 1$, il existe toujours une politique optimale **déterministe** — c'est un résultat fort qui sera réutilisé en programmation dynamique.

> [!example] Représentation d'une politique sur FrozenLake
> Une politique stochastique complète serait un tableau $16 \times 4$ (une ligne par état, une colonne par action) — trop grand pour tenir dans le texte. On en montre une ligne, à l'état $s=14$ : un agent encore peu sûr de lui pourrait jouer $\pi(\cdot \mid 14) = (0.05, 0.30, 0.30, 0.35)$ pour (Gauche, Bas, Droite, Haut) — mise principalement sur les trois bonnes directions identifiées plus haut, un peu d'exploration résiduelle sur Gauche.
> 
> Pour une politique **déterministe**, la représentation la plus naturelle sur une grille n'est pas un tableau mais une **carte de flèches** — un net avantage pédagogique de FrozenLake sur un exemple à 3 états abstraits :
> 
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_policy_grid.png|260]]
> 
> Chaque flèche est $\pi_*(s) = \arg\max_a(\ldots)$ pour cet état (calculée par Value Iteration, cf. II.D) ; les cases $H$ et $G$ sont terminales, sans action à choisir. On vérifiera formellement en Section II que c'est bien la politique optimale.
> 
> **Attention à ne pas confondre $\pi$ avec la matrice de transition $\mathbf{P}$ :** les deux sont "row-stochastic" (chaque ligne somme à 1), mais $\mathbf{P}(s'\mid s,a)$ est la physique de la glace — l'agent la subit, il ne la contrôle pas — alors que $\pi(a\mid s)$ est sa stratégie, ce qu'on optimise.

> [!important] MDP + politique fixée = MRP
> Si on fixe une politique $\pi$ dans un MDP, on "absorbe" l'action dans la dynamique : le système devient un MRP avec
> 
> $$P^\pi(s' \mid s) = \sum_a \pi(a \mid s) \, P(s' \mid s, a), \qquad R^\pi(s) = \sum_a \pi(a \mid s) \, R(s, a).$$
> 
> C'est cette réduction qui justifie tout ce qu'on va faire en **Policy Evaluation** : évaluer une politique dans un MDP, c'est analyser le MRP qu'elle induit. La distinction "évaluation = problème de prédiction" vs "amélioration = problème de contrôle" repose entièrement sur cette équivalence.

### C. Fonctions de valeur

**Le besoin.** On dispose maintenant d'un MDP et d'une politique $\pi$. La question naturelle est : *si l'agent suit cette politique, qu'est-ce que ça lui rapporte en moyenne sur le long terme ?* C'est ce qu'on va formaliser avec les **fonctions de valeur** (value functions). Elles sont l'objet central du RL : tous les algorithmes qui suivent (DP, Monte Carlo, TD-learning, Q-learning) ne font qu'une seule chose — estimer ou maximiser une fonction de valeur.

**Deux objets, deux questions.** On définit en fait *deux* fonctions de valeur, parce qu'il y a deux questions différentes à se poser :

> [!warning] Fonction de valeur d'état $v_\pi(s)$
> $v_\pi(s)$ est l'espérance du retour $G_t$ quand on démarre dans l'état $s$ et qu'on suit la politique $\pi$ ensuite :
> 
> $$v_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\Big|\; s_t = s\right].$$
> 
> Elle répond à la question : ***"combien vaut le fait d'être dans l'état $s$ si je suis $\pi$ ?"***. C'est l'outil d'**évaluation**.

> [!warning] Fonction de valeur d'action $q_\pi(s, a)$
> $q_\pi(s, a)$ est l'espérance du retour quand on démarre en $s$, qu'on prend l'action $a$ *cette fois-ci*, puis qu'on suit $\pi$ ensuite :
> 
> $$q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a].$$
> 
> Elle répond à la question : ***"combien vaut le fait de prendre l'action $a$ depuis l'état $s$ si je suis $\pi$ ensuite ?"***. C'est l'outil de **décision** : à état $s$ fixé, comparer $q_\pi(s, a_1)$ et $q_\pi(s, a_2)$ permet de dire quelle action est meilleure. On l'appelle aussi **Q-fonction** (Q-function).

> 💡 **Pourquoi deux fonctions ?** $v_\pi$ et $q_\pi$ portent la même information sous deux angles. $v_\pi$ moyenne sur les actions selon $\pi$ ; $q_\pi$ fixe l'action et regarde ce qui se passe ensuite. Pour *évaluer* une politique, $v_\pi$ suffit. Pour *décider quelle action prendre*, on a besoin de $q_\pi$ — sinon on ne peut pas comparer les actions à état fixé. C'est cette distinction qu'exploitent les algos comme Q-learning, qui apprennent directement $q$ pour pouvoir décider sans avoir besoin du modèle.

**Lien $v_\pi \leftrightarrow q_\pi$.** Les deux fonctions sont reliées par deux identités fondamentales :

$$v_\pi(s) = \sum_a \pi(a \mid s) \, q_\pi(s, a),$$

$$q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_\pi(s').$$

La première dit : la valeur d'un état = la moyenne des Q-valeurs des actions disponibles, pondérées par la politique. La seconde dit : la Q-valeur d'une action = la récompense immédiate + l'espérance de la valeur de l'état suivant, actualisée. Ces deux relations sont les briques élémentaires de tous les algorithmes de programmation dynamique.

> [!note]- Preuve — d'où sort $q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s, a)\, v_\pi(s')$ ?
> On part de la définition de $q_\pi$ : la valeur du couple $(s,a)$, c'est l'espérance du retour si on est en $s$, qu'on **choisit** l'action $a$ (donc plus besoin de $\pi$ pour ce premier pas), puis qu'on suit $\pi$ ensuite.
> 
> $$q_\pi(s,a) = \mathbb{E}\big[G_t \mid s_t = s,\, a_t = a\big].$$
> 
> On décompose le retour comme avant, $G_t = r_t + \gamma G_{t+1}$ :
> 
> $$q_\pi(s,a) = \mathbb{E}[r_t \mid s_t=s, a_t=a] + \gamma\, \mathbb{E}[G_{t+1} \mid s_t=s, a_t=a].$$
> 
> Le premier terme est direct : $s$ et $a$ sont fixés, donc $\mathbb{E}[r_t \mid s_t=s,a_t=a] = R(s,a)$ par définition de $R$.
> 
> **Le second terme est le seul point délicat**, et c'est là qu'intervient la **loi de l'espérance totale** : pour une variable aléatoire $X$ et une v.a. discrète $Y$,
> 
> $$\mathbb{E}[X] = \sum_y P(Y=y)\, \mathbb{E}[X \mid Y=y].$$
> 
> Ici $X = G_{t+1}$ et $Y = s_{t+1}$ (l'état où on atterrit). On conditionne donc sur *quel* état suivant $s'$ est réalisé, avec probabilité $P(s' \mid s,a)$ — c'est exactement la définition de la dynamique du MDP :
> 
> $$\mathbb{E}[G_{t+1} \mid s_t=s, a_t=a] = \sum_{s'} P(s' \mid s,a)\, \mathbb{E}\big[G_{t+1} \mid s_t=s, a_t=a, s_{t+1}=s'\big].$$
> 
> Reste à simplifier ce dernier terme conditionnel. C'est ici qu'intervient la **propriété de Markov** : une fois qu'on connaît $s_{t+1}=s'$, l'avenir ($G_{t+1}$, qui ne dépend que de la trajectoire à partir de $t+1$) est **indépendant** de comment on est arrivé là (peu importe $s_t$ et $a_t$) — la mémoire du chemin passé n'apporte rien de plus. Donc conditionner en plus sur $s_t=s, a_t=a$ est redondant :
> 
> $$\mathbb{E}\big[G_{t+1} \mid s_t=s, a_t=a, s_{t+1}=s'\big] = \mathbb{E}\big[G_{t+1} \mid s_{t+1}=s'\big] = v_\pi(s').$$
> 
> (La dernière égalité, c'est juste la définition de $v_\pi$ appliquée à l'instant $t+1$ au lieu de $t$ — le retour futur ne dépend pas de "quand" on est, seulement d'où on part.)
> 
> En recollant tous les morceaux :
> 
> $$q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, v_\pi(s').$$

**Équation de Bellman.** Calculer $v_\pi$ par sa définition (somme infinie d'espérances sur tous les futurs possibles) est infaisable. Heureusement, $v_\pi$ vérifie une **équation récursive** qui transforme cette somme infinie en système linéaire fini :

> [!warning] Équation de Bellman pour $v_\pi$
> Pour toute politique $\pi$ et tout état $s$ :
> 
> $$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma \, v_\pi(s') \big].$$
> 
> En substance : *la valeur de l'état présent = la récompense que je vais toucher maintenant + la valeur actualisée de l'état où je vais arriver, le tout moyenné sur la stochasticité de $\pi$ et de $\mathbf{P}$*.

> [!note]- Preuve express (à partir de $q_\pi$, déjà prouvé plus haut)
> On combine simplement les deux identités déjà établies : $v_\pi(s) = \sum_a \pi(a\mid s)\, q_\pi(s,a)$ et $q_\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\, v_\pi(s')$. En substituant la seconde dans la première :
> 
> $$v_\pi(s) = \sum_a \pi(a\mid s) \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)\, v_\pi(s') \Big].$$
> 
> Reste à faire rentrer $R(s,a)$ dans la somme sur $s'$, pour obtenir la forme "propre" de l'équation. C'est licite : $\sum_{s'} P(s' \mid s,a) = 1$ (c'est une distribution de probabilité sur $s'$), donc $R(s,a) = \sum_{s'} P(s'\mid s,a)\, R(s,a)$ — on multiplie par $1$ écrit sous cette forme. D'où :
> 
> $$v_\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s' \mid s,a) \big[ R(s,a) + \gamma\, v_\pi(s') \big],$$
> 
> ce qui est bien l'équation de Bellman encadrée ci-dessus.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_bellman_backup.png|650]]

**Figure 5.** Lecture visuelle de l'équation de Bellman sur $s=14$ (juste sous l'objectif), avec une politique arbitraire $\pi(\text{Bas}\mid14)=\pi(\text{Droite}\mid14)=0.5$ (Gauche et Haut omises pour la lisibilité, comme l'ancien exemple omettait Flat). Trois niveaux : (1) l'état présent $s=14$ (rond blanc) ; (2) une action choisie selon $\pi(a\mid s)$ (point noir) ; (3) transition vers un état suivant $s'$ selon $P(s'\mid s,a)$ (rond bleu/vert), avec sous chaque feuille la récompense $r$ et la valeur future actualisée $\gamma v_*(s')$ (valeurs numériques reprises du calcul de $V_*$ en I.C). La somme sur les actions correspond à $\sum_a \pi(a\mid s)$, la somme sur les états suivants à $\sum_{s'} P(s'\mid s,a)$.

> [!note]- Preuve (courte)
> On part de la définition $v_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ et on décompose le retour : $G_t = r_t + \gamma G_{t+1}$.
> 
> $$v_\pi(s) = \mathbb{E}_\pi[r_t \mid s_t = s] + \gamma \, \mathbb{E}_\pi[G_{t+1} \mid s_t = s].$$
> 
> Pour le premier terme, on conditionne sur l'action selon $\pi$ : $\mathbb{E}_\pi[r_t \mid s_t = s] = \sum_a \pi(a \mid s) R(s, a)$.
> 
> Pour le second, on conditionne sur l'action et l'état suivant. Par la propriété de Markov, sachant $s_{t+1}$, le retour $G_{t+1}$ ne dépend plus de ce qui s'est passé avant — son espérance vaut $v_\pi(s_{t+1})$. Donc
> 
> $$\mathbb{E}_\pi[G_{t+1} \mid s_t = s] = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \, v_\pi(s').$$
> 
> En recollant, on obtient l'équation de Bellman.

**Forme matricielle (cas MRP).** Si la politique est fixée, le MDP se réduit à un MRP (cf. section précédente) et l'équation de Bellman devient simplement

$$V^\pi = R^\pi + \gamma \, \mathbf{P}^\pi V^\pi$$

soit, en isolant $V^\pi$,

$$\boxed{V^\pi = (I - \gamma \mathbf{P}^\pi)^{-1} R^\pi.}$$

C'est exactement la formule close annoncée plus haut : **évaluer une politique = inverser une matrice**. La matrice $I - \gamma \mathbf{P}^\pi$ est inversible dès que $\gamma < 1$ (parce que les valeurs propres de $\gamma \mathbf{P}^\pi$ sont de module $< 1$). En pratique, quand $|S|$ est grand, on remplace cette inversion par des itérations — c'est tout l'objet de la section sur Policy Evaluation.

**Politique optimale.** On dit qu'une politique $\pi$ est **meilleure ou égale** à une autre politique $\pi'$ si $v_\pi(s) \ge v_{\pi'}(s)$ pour *tout* état $s$. On note $\pi \ge \pi'$. C'est un ordre **partiel** : deux politiques peuvent être incomparables (l'une meilleure dans certains états, l'autre dans les autres).

Le résultat fondamental, qu'on admettra ici, est qu'il existe toujours **au moins une politique $\pi_*$ qui domine toutes les autres** (au sens de cet ordre). Et plus fort encore : on peut toujours en choisir une qui est **déterministe**. Toutes les politiques optimales partagent la même fonction de valeur $v_*$ et la même Q-fonction $q_*$.

> [!warning] Fonctions de valeur optimales
> $$v_*(s) = \max_\pi v_\pi(s), \qquad q_*(s, a) = \max_\pi q_\pi(s, a).$$
> 
> Ce sont les fonctions de valeur de n'importe quelle politique optimale.

**Équations de Bellman d'optimalité.** $v_*$ et $q_*$ vérifient elles aussi des équations récursives, mais avec un $\max$ à la place de la moyenne sur $\pi$ :

> [!warning] Bellman optimality
> $$v_*(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma \, v_*(s') \big],$$
> 
> $$q_*(s, a) = \sum_{s'} P(s' \mid s, a) \Big[ R(s, a) + \gamma \max_{a'} q_*(s', a') \Big].$$
> 
> Le $\max$ remplace l'espérance sur $\pi$ : la politique optimale ne pondère pas les actions, elle prend toujours la meilleure.

**Le résultat charnière : greedy par rapport à $q_*$ = optimal.** C'est *le* résultat qui justifie tout le reste du cours. Il dit que **si l'on connaît $q_*$, on connaît la stratégie optimale gratuitement** : il suffit de prendre, à chaque état, l'action qui maximise $q_*(s, a)$.

$$\pi_*(s) = \arg\max_{a} \, q_*(s, a).$$

> 💡 **Pourquoi c'est si important.** Tout le problème du RL se réduit à *estimer $q_*$* (ou de manière équivalente $v_*$ et le modèle, mais $q_*$ est plus direct). Une fois qu'on a $q_*$, l'optimisation est triviale : on agit gloutonnement. C'est exactement ce que font Q-learning, SARSA, DQN — ils essaient tous d'estimer $q_*$, parce qu'ils savent qu'ensuite agir devient gratuit.

**Lien $v_* \leftrightarrow q_*$.** Comme pour le cas non optimal, les deux fonctions sont liées :

$$v_*(s) = \max_{a} q_*(s, a), \qquad q_*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_*(s').$$

La première dit que la valeur optimale d'un état = la valeur de la meilleure action depuis cet état. La seconde permet de reconstruire $q_*$ à partir de $v_*$ et du modèle.

> [!example] FrozenLake — calcul des fonctions de valeur
> On fixe $\gamma = 0.9$ et on compare $V^\pi(s)$ (fonction de valeur d'**état** — pas de politique ici, juste un nombre par état) pour deux politiques, via la formule close $V^\pi = (I - \gamma \mathbf{P}^\pi)^{-1} R^\pi$ — ici $\mathbf{P}^\pi$ est $16\times16$, donc l'inversion se fait par ordinateur plutôt qu'à la main, mais la formule et son sens restent identiques à l'exemple à 3 états. Comme les états sont disposés sur la grille $4\times4$, on affiche $V(s)$ directement dessus plutôt qu'en liste : chaque case affiche $V$ de l'état physique correspondant.
> 
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_V_random.png|280]]
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_V_star.png|280]]
> 
> Valeurs minuscules partout à gauche (politique aléatoire) : en tirant une direction au hasard, la probabilité d'atteindre $G$ avant de tomber dans un trou est très faible depuis la case de départ. À droite (politique quasi-optimale $\pi_*$, celle de la carte de flèches plus haut), énorme différence relative (jusqu'à $\times 15$ à la case départ). Mais même optimale, la glace glissante empêche de s'approcher de la certitude : $V_*(\text{départ}) \approx 0.069$, donc l'agent optimal n'atteint le but qu'environ $7\%$ du temps en partant de $0$ — la stochasticité de l'environnement plafonne la performance, un phénomène que l'ancien exemple (où l'action ne changeait pas la dynamique) ne pouvait pas illustrer.
> 
> **Conjecture.** $\pi_*$ ci-dessus est très probablement la politique optimale. On le démontrera formellement avec Policy Iteration et Value Iteration dans la prochaine section.

## II. Programmation dynamique (model-based)

**Le contexte.** On dispose d'un MDP entièrement connu : on a la matrice de transition $\mathbf{P}$ et la fonction de récompense $R$. La question : **comment calculer $v_\pi$ (évaluation) et $\pi_*$ (optimisation) en pratique ?**

> [!important] C'est quoi, "programmation dynamique", au juste ?
> Erreur fréquente : penser que la programmation dynamique (DP), c'est juste une fonction qui s'appelle elle-même, $f(f(\ldots))$ — c'est-à-dire de la simple **récursion**. Ce n'est pas ça. La différence tient en un mot : le **stockage**.
>
> Prenons l'exemple classique de Fibonacci, $f(n) = f(n-1) + f(n-2)$. En récursion naïve, pour calculer $f(5)$, on rappelle $f(4)$ et $f(3)$ ; pour calculer $f(4)$, on rappelle à nouveau $f(3)$ et $f(2)$ ; etc. **Le même sous-problème ($f(3)$, $f(2)$...) est recalculé de nombreuses fois**, et le nombre total d'appels explose exponentiellement avec $n$.
>
> La programmation dynamique résout **exactement la même récurrence**, mais avec une règle en plus : **on stocke le résultat de chaque sous-problème la première fois qu'on le calcule**, dans un tableau (ou un dictionnaire). Si on doit recalculer $f(3)$ plus tard, on va juste le lire dans le tableau au lieu de le recalculer. Chaque sous-problème n'est résolu **qu'une seule fois** — on passe d'un temps exponentiel à un temps linéaire (ou polynomial).
>
> **Le lien avec le RL tabulaire.** C'est très exactement ce que font Policy Evaluation, Policy Iteration et Value Iteration : à chaque itération $k$, on calcule $V_{k+1}(s)$ pour **tous les états**, et on **stocke** le résultat dans un tableau $V$ — c'est littéralement ça, le "tabulaire" du titre de la note. L'itération suivante relit ce tableau au lieu de tout recalculer depuis zéro. La "fonction qui s'appelle elle-même" que tu avais en tête, c'est bien la récursivité de l'équation de Bellman ($v_\pi(s)$ dépend de $v_\pi(s')$) — mais sans le stockage à chaque étape, ce ne serait qu'une récursion naïve, pas de la programmation dynamique.

**Rappel : deux façons de raisonner sur un processus stochastique.** On a déjà rencontré cette dichotomie dans la note "Comment simule-t-on une trajectoire ?" de la section sur les processus de Markov :

- **Propager la distribution.** Si on connaît $\mathbf{P}$, on peut calculer où l'on est *en moyenne sur tous les futurs possibles* via $\mu^{k+1} = \mu^k \mathbf{P}$. Cela calcule l'espérance exactement, sans tirage aléatoire.
- **Échantillonner.** Si on n'a pas $\mathbf{P}$, on simule des trajectoires concrètes une à une, et on estime les espérances par moyenne empirique.

> 💡 **DP est tout entier du côté "propager la distribution".** L'équation de Bellman fait précisément ça : elle exprime $v_\pi(s)$ comme une espérance calculée *exactement* à partir de $\mathbf{P}$ et $R$. Tous les algos de cette section (Policy Evaluation, Policy Iteration, Value Iteration) partagent ce même squelette : appliquer Bellman comme une **mise à jour itérative** sur la fonction de valeur, jusqu'à convergence. C'est uniquement parce qu'on a le modèle qu'on peut calculer le terme de droite de Bellman directement.
> 
> **Monte Carlo (III) et TD Learning (IV)** abandonneront cette hypothèse : on ne connaîtra plus $\mathbf{P}$, on devra estimer les espérances en échantillonnant des trajectoires. Mais l'objectif (résoudre Bellman) restera le même.

**L'idée unique.** L'équation de Bellman donne une *consigne de cohérence* sur $v_\pi$. On ne peut pas l'imposer d'un coup parce que $v_\pi$ apparaît des deux côtés. La solution : on l'impose **itérativement**. On part d'une estimation $V_0$ quelconque, on applique Bellman comme une mise à jour, et la suite $V_0, V_1, V_2, \ldots$ converge vers $v_\pi$. À partir de là, les quatre algos qui suivent ne sont que des variations sur ce même thème :

- **Policy Evaluation** (A) — évaluer une politique fixée en itérant Bellman.
- **Policy Improvement** (B) — partir d'une évaluation, en déduire une meilleure politique.
- **Policy Iteration** (C) — alterner les deux jusqu'à atteindre $\pi_*$.
- **Value Iteration** (D) — fusionner évaluation et amélioration en une seule boucle (Bellman optimality).

### A. Policy Evaluation (Policy Prediction)

**Le problème.** *Policy Evaluation* (aussi appelé **prediction problem**) consiste à calculer la fonction de valeur $v_\pi$ d'une politique **fixée** $\pi$. On ne cherche pas la meilleure politique ici — on évalue celle qu'on a sous la main.

**Approche 1 : résolution exacte.** Déjà vue en I.C : une fois la politique fixée, le MDP devient un MRP et l'équation de Bellman est linéaire,

$$V^\pi = (I - \gamma \mathbf{P}^\pi)^{-1} R^\pi.$$

Correct mathématiquement, mais l'inversion de matrice coûte $O(|S|^3)$. Dès que $|S|$ dépasse quelques milliers, c'est infaisable. En pratique on préfère une approche itérative.

**Approche 2 : Iterative Policy Evaluation.** L'idée est de transformer l'équation de Bellman en **règle de mise à jour**. On part d'une estimation arbitraire $V_0$ (typiquement $V_0(s) = 0$ pour tout $s$), puis on applique itérativement :

> [!warning] Règle de mise à jour
> $$V_{k+1}(s) \;=\; \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma \, V_k(s') \big]$$
> 
> autrement dit : $V_{k+1}$ = on applique Bellman à droite avec la valeur actuelle $V_k$. La suite $V_0, V_1, V_2, \ldots$ converge vers $v_\pi$ quand $\gamma < 1$.

C'est le même contenu que l'équation de Bellman, lue comme un programme : *"prends ta meilleure estimation $V_k$ et raffine-la d'un cran en appliquant Bellman une fois"*.

**Critère d'arrêt.** En pratique on s'arrête quand les valeurs ne bougent plus assez :

$$\Delta = \max_s \big| V_{k+1}(s) - V_k(s) \big| < \theta$$

où $\theta$ est un seuil de tolérance fixé à l'avance (typiquement $10^{-6}$).

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/algo-1.png|349]]


> [!example] FrozenLake — le modèle $(P, R)$, fixe pendant tout Policy Iteration
> Avant de lancer quoi que ce soit : $\mathbf{P}$ et $R$ sont connus et ne changent **jamais** pendant Policy Evaluation/Improvement/Iteration — seule la politique $\pi$ (et donc $V$, $Q$) évolue. Les voici, sous forme de heatmaps : $R(s,a)$ (une seule case non nulle : atteindre $G$ depuis $s=14$), puis les 4 matrices $P(s'\mid s, a)$, une par action — chaque case colorée montre vers quel $s'$ on peut atterrir. Sur FrozenLake *slippery*, $P(s'\mid s,a)$ ne prend que 4 valeurs possibles ; légende : noir = $0$, bleu = $1/3$, orange = $2/3$, rouge = $1$ (états terminaux $H$/$G$, qui se bouclent sur eux-mêmes avec probabilité $1$).
> 
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_model_R.png|280]]
![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_model_P_2x2.png|700]]

> [!example] FrozenLake — Iterative Policy Evaluation à la main
> On évalue la politique initiale $\pi_0$ (*toujours Bas*) avec $\gamma = 0.9$. La politique étant déterministe, le $\sum_a$ s'effondre et la règle de mise à jour devient $V_{k+1}(s) = R(s,\text{Bas}) + 0.9\sum_{s'} P(s'\mid s,\text{Bas})\,V_k(s')$. Trace sur 3 états représentatifs (départ à $V_0=0$ partout) :
> 
> | $k$ | $V_k(9)$ | $V_k(13)$ | $V_k(14)$ |
> |---|:---:|:---:|:---:|
> | 0 | 0.000 | 0.000 | 0.000 |
> | 1 | 0.000 | 0.000 | 0.333 |
> | 2 | 0.000 | 0.100 | 0.433 |
> | 3 | 0.060 | 0.160 | 0.493 |
> | 4 | 0.087 | 0.196 | 0.529 |
> | 5 | 0.114 | 0.218 | 0.551 |
> | $\infty$ | **0.163** | **0.250** | **0.583** |
> 
> **Détail du calcul de $V_2(14)$** (les autres suivent le même schéma) : depuis $s=14$, l'action Bas mène à $\{13, 14, 15\}$ avec $1/3$ chacun,
> 
> $$V_2(14) = \underbrace{\tfrac13\cdot0 + \tfrac13\cdot1 + \tfrac13\cdot0}_{R(14,\text{Bas})=0.333} + 0.9\Big[\tfrac13 V_1(13) + \tfrac13 V_1(14) + \tfrac13 V_1(15)\Big] = 0.333 + 0.9\times\tfrac13(0+0.333+0) = 0.433.$$
> 
> Deux observations :
> - **Convergence géométrique**, comme dans l'exemple à 3 états — l'écart à la valeur finale se réduit d'un facteur proche de $\gamma=0.9$ à chaque itération.
> - **Propagation depuis $G$** : $V(14)$ (voisin direct de l'objectif) décolle dès $k=1$, alors que $V(9)$ (deux cases plus loin) reste à $0$ jusqu'à $k=3$ — la valeur de l'objectif met plusieurs itérations à "remonter" jusqu'aux états lointains. C'est la conséquence directe de la récompense éparse notée en I.B.
> 
> La grille complète de $V^{\pi_0}$ (tous les états) est le panneau "iter 1" de la Figure ci-dessous (II.C) — on y voit que $\pi_0$ ("toujours Bas") est déjà loin d'être stupide (elle traverse la grille verticalement) mais reste sous-optimale sur la première ligne, où Bas mène tout droit dans les trous $s=5$ ou $s=7$.

### B. Policy Improvement

**L'idée.** Policy Evaluation nous donne $v_\pi$ pour une politique fixée. C'est un outil de *prédiction*. Mais notre vrai objectif, c'est le **contrôle** : trouver $\pi_*$. Comment passer de "j'évalue $\pi$" à "j'améliore $\pi$" ?

**Le test ponctuel.** Plaçons-nous dans un état $s$ et demandons-nous : *"que se passerait-il si, juste cette fois, je prenais une action $a \neq \pi(s)$, puis que je reprenais à suivre $\pi$ ensuite ?"* La valeur de cette stratégie hybride est exactement

$$q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_\pi(s').$$

> 💡 **Intuition.** Si $q_\pi(s, a) > v_\pi(s)$, ça veut dire que prendre $a$ une seule fois en $s$ est strictement meilleur que de suivre $\pi$. Or si c'est vrai *une fois*, ça reste vrai à *chaque visite* de $s$ — d'où l'idée : changer définitivement la politique en $s$ vers $a$ doit donner une amélioration globale. C'est cette intuition que formalise le théorème suivant.

> [!warning] Théorème de Policy Improvement
> Soient $\pi$ et $\pi'$ deux politiques déterministes. Si pour tout $s \in S$,
> 
> $$q_\pi(s, \pi'(s)) \;\ge\; v_\pi(s),$$
> 
> alors $\pi'$ est au moins aussi bonne que $\pi$ partout :
> 
> $$v_{\pi'}(s) \;\ge\; v_\pi(s) \qquad \forall s \in S.$$
> 
> Et si l'inégalité est stricte en au moins un état, alors $\pi'$ est strictement meilleure.

> [!note]- Preuve
> On part de l'hypothèse $v_\pi(s) \le q_\pi(s, \pi'(s))$ et on **déroule** Bellman en remplaçant à chaque étape $v_\pi$ par $q_\pi(\cdot, \pi'(\cdot))$ :
> 
> $$\begin{aligned}
> v_\pi(s) &\le q_\pi(s, \pi'(s)) \\
> &= \mathbb{E}_{\pi'}\big[ r_t + \gamma \, v_\pi(s_{t+1}) \mid s_t = s \big] \\
> &\le \mathbb{E}_{\pi'}\big[ r_t + \gamma \, q_\pi(s_{t+1}, \pi'(s_{t+1})) \mid s_t = s \big] \\
> &= \mathbb{E}_{\pi'}\big[ r_t + \gamma \, r_{t+1} + \gamma^2 v_\pi(s_{t+2}) \mid s_t = s \big] \\
> &\le \mathbb{E}_{\pi'}\big[ r_t + \gamma \, r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 v_\pi(s_{t+3}) \mid s_t = s \big] \\
> &\;\;\vdots \\
> &\le \mathbb{E}_{\pi'}\big[ r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t = s \big] \\
> &= v_{\pi'}(s).
> \end{aligned}$$
> 
> À chaque ligne on applique l'hypothèse une fois de plus, ce qui transfère un pas de temps de $\pi$ vers $\pi'$. À la limite, l'espérance est entièrement sous $\pi'$ — c'est $v_{\pi'}(s)$.

**La politique gloutonne.** Comment construire $\pi'$ qui satisfait l'hypothèse du théorème ? Le plus simple : prendre, en chaque état, l'action qui maximise $q_\pi(s, a)$ :

> [!warning] Politique gloutonne par rapport à $v_\pi$
> $$\pi'(s) \;=\; \arg\max_{a} \, q_\pi(s, a) \;=\; \arg\max_{a} \, \Big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_\pi(s') \Big].$$
> 
> Par construction, $q_\pi(s, \pi'(s)) = \max_a q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)$. L'hypothèse du théorème est donc automatiquement satisfaite : **$\pi'$ est nécessairement au moins aussi bonne que $\pi$**.

> [!note]- Que faire en cas d'égalité (ties) ?
> Si plusieurs actions atteignent le maximum en un même état, on en choisit une arbitrairement (ou on construit une politique stochastique uniforme sur les actions optimales). Le théorème reste valide dans tous les cas — n'importe quel choix donne une politique au moins aussi bonne que $\pi$.

**Le résultat charnière.** Que se passe-t-il si la politique gloutonne $\pi'$ n'est **pas strictement** meilleure que $\pi$, c'est-à-dire $v_{\pi'} = v_\pi$ ? Alors pour tout $s$,

$$v_\pi(s) = v_{\pi'}(s) = \max_a \, q_\pi(s, a) = \max_a \Big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, v_\pi(s') \Big].$$

Mais cette dernière égalité, c'est **exactement l'équation de Bellman d'optimalité** vue en I.C. Donc $v_\pi = v_*$, et $\pi$ est déjà optimale.

algo

![[algo-3.png|313]]

> 💡 **Conclusion.** Tant que la politique gloutonne donne une amélioration stricte, on peut continuer à améliorer. Le seul moment où l'on ne peut plus améliorer, c'est quand on est *déjà* à l'optimum. C'est exactement le mécanisme de bouclage qui justifiera **Policy Iteration** dans la sous-section suivante : alterner évaluation et amélioration jusqu'à ce que ça ne bouge plus, et on a $\pi_*$.

> [!example] FrozenLake — une étape d'amélioration
> On part de $\pi_0$ (*toujours Bas*), qu'on a évaluée en II.A ($V^{\pi_0}$ ci-dessus). On calcule $q_{\pi_0}(s,a) = R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)\,V^{\pi_0}(s')$ pour tous les couples $(s,a)$ — la grille complète est le panneau "Q(s,a) — iter 1" de la Figure en II.C. Deux états représentatifs :
> 
> | | Gauche | Bas | Droite | Haut | argmax |
> |---|:---:|:---:|:---:|:---:|:---:|
> | $q_{\pi_0}(0, a)$ | $\mathbf{0.020}$ | 0.019 | 0.019 | 0.016 | **Gauche** |
> | $q_{\pi_0}(14, a)$ | 0.317 | $\mathbf{0.583}$ | 0.576 | 0.476 | **Bas** |
> 
> **État $s=0$ (départ) : changement.** $\pi_0(0)$ = Bas, mais $\arg\max_a q_{\pi_0}(0,a)$ = Gauche (de justesse : $0.020$ contre $0.019$). Contre-intuitif — Gauche mène droit dans un mur ! Mais justement : heurter le mur laisse l'agent sur place ($1/3$ de chance), ce qui est légèrement moins risqué ici que Bas (qui a $1/3$ de chance de finir en $s=4$ puis, plus tard, dans le trou $s=5$). Avec une valeur $V^{\pi_0}$ encore très fraîche (2 itérations à peine), ces écarts sont ténus — Policy Iteration va les affiner.
> 
> **État $s=14$ (voisin de l'objectif) : inchangé.** $\pi_0(14)=$ Bas était déjà l'action gloutonne ($q=0.583$, la plus grande) — pas de changement ici.
> 
> **Politique améliorée $\pi_1$.** En répétant ce calcul sur les 16 états (résultat complet : panneau "pi(a\|s) — iter 2" de la Figure ci-dessous), on obtient une nouvelle politique déterministe, différente de $\pi_0$ sur plusieurs états. Le théorème de Policy Improvement garantit $V^{\pi_1}(s) \ge V^{\pi_0}(s)$ partout — et sur $s=0$, on passe de $V^{\pi_0}(0)=0.019$ à $V^{\pi_1}(0)=0.069$, soit $\times 3.6$.
> 
> C'est précisément ce mécanisme qu'on va automatiser en C avec **Policy Iteration** : évaluer, améliorer, ré-évaluer, ré-améliorer... jusqu'à ce que la politique cesse de changer.

### C. Policy Iteration

**L'idée.** A et B sont les deux moitiés naturelles d'une boucle. **Policy Iteration** consiste à les enchaîner :

$$\pi_0 \xrightarrow{\text{A}} V^{\pi_0} \xrightarrow{\text{B}} \pi_1 \xrightarrow{\text{A}} V^{\pi_1} \xrightarrow{\text{B}} \pi_2 \xrightarrow{\text{A}} \cdots \xrightarrow{} \pi_*$$

À chaque tour : on évalue la politique courante (A), puis on construit la politique gloutonne par rapport à la valeur obtenue (B). On s'arrête dès que la politique ne change plus.

**Pourquoi ça converge — et même en un nombre fini d'étapes.** Trois ingrédients, déjà tous établis :

1. **Chaque tour produit une politique au moins aussi bonne** (théorème de Policy Improvement, II.B).
2. **Si la politique ne change pas après une étape de B, c'est qu'on est à l'optimum** (résultat charnière, II.B).
3. **Le nombre de politiques déterministes est fini** : $|A|^{|S|}$ politiques en tout. Comme on ne peut pas revenir en arrière (chaque amélioration stricte est strictement meilleure, donc la valeur est monotone croissante), on ne peut pas boucler indéfiniment.

Donc Policy Iteration termine en **au plus $|A|^{|S|}$ tours**. En pratique, c'est généralement bien plus rapide.

![[algo-4.png|360]]
Caption. Pseudo code

> [!example] FrozenLake — Policy Iteration boucle en 2 tours
> On part de $\pi_0$ (*toujours Bas*). La figure ci-dessous montre l'évolution complète de $V$, $Q$ et $\pi$ sur les 2 itérations nécessaires à la convergence — nettement plus rapide que les 6 itérations qu'il aurait fallu en partant d'une politique arbitraire moins bien choisie (ex. *toujours Gauche*, qui reste bloquée sur place partout et ne "voit" jamais l'objectif au premier passage).
> 
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/fl_policy_iteration_evolution.png]]
> 
> **Tour 1.** Évaluation de $\pi_0$ (bloc II.A) → $V^{\pi_0}$ (colonne de gauche, "iter 1"). Amélioration (bloc II.B) → $\pi_1$, différente de $\pi_0$ sur plusieurs états (colonne de droite, "iter 1" montre encore $\pi_0$, la politique *avant* la mise à jour — comparer avec "iter 2" pour voir le changement).
> 
> **Tour 2.** Évaluation de $\pi_1$ → $V^{\pi_1}$ ("iter 2", identique à $V_*$ calculé en I.C). Amélioration : pour chaque $s$, on vérifie que $\arg\max_a Q(s,a)$ retombe sur $\pi_1(s)$ déjà en place — plus aucun changement, donc **politique stable → on s'arrête**.
> 
> **Conclusion.** $\pi_1 = \pi_*$ est optimale, avec $V_*$ identique à celui calculé en I.C (par exemple $V_*(0) \approx 0.069$). Ce qui confirme, cette fois avec une preuve algorithmique complète, la politique qu'on avait seulement conjecturée par calcul direct en I.C.

### D. Value Iteration

**Motivation.** Policy Iteration est correct mais cher : à chaque tour externe, on doit attendre que **toute** la boucle interne de Policy Evaluation converge — potentiellement des centaines d'itérations de Bellman, juste pour ensuite faire *une* étape d'amélioration. C'est du gâchis.

**L'idée.** On peut tronquer l'évaluation après *une seule* itération sans perdre la convergence vers $\pi_*$. Et quand on tronque à 1, l'évaluation et l'amélioration peuvent **fusionner** en une seule mise à jour avec un $\max$ :

> [!warning] Règle de mise à jour Value Iteration
> $$V_{k+1}(s) \;=\; \max_{a} \Big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V_k(s') \Big].$$
> 
> C'est exactement Bellman optimality vu en I.C, lu comme une règle de mise à jour. La suite $V_0, V_1, V_2, \ldots$ converge vers $V_*$ pour $\gamma < 1$, indépendamment de l'initialisation.

Une fois $V_*$ obtenue, on récupère $\pi_*$ par une étape gloutonne finale :

$$\pi_*(s) = \arg\max_a \Big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V_*(s') \Big].$$

> 💡 **L'image à retenir.** Policy Iteration alterne deux phases distinctes (évaluer / améliorer). Value Iteration **fusionne les deux** en injectant le $\max$ directement dans la mise à jour. C'est plus direct, plus économe — et ça converge vers la même chose.

> [!note]- Comment Policy Iteration se réduit à Value Iteration
> On part de la boucle Policy Iteration et on la simplifie en quatre temps.
> 
> **(i) Les deux étapes écrites explicitement.** Pour chaque $s$ :
> - **Évaluation** : $V(s) \leftarrow \sum_{s'} P(s' \mid s, \pi(s)) \big[ R(s, \pi(s)) + \gamma V(s') \big]$.
> - **Amélioration** : $\pi'(s) \leftarrow \arg\max_a \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma V(s') \big]$.
> 
> **(ii) On tronque l'évaluation à une seule itération.** Au lieu d'attendre la convergence complète de l'évaluation, on fait *une* mise à jour de $V$, puis on enchaîne immédiatement sur l'amélioration. C'est un raccourci légitime — la convergence vers $V_*$ tient toujours.
> 
> **(iii) On remplace $\pi$ par sa version améliorée et on collapse.** Après l'amélioration, on a $\pi \leftarrow \pi'$. La prochaine évaluation utilisera donc $\pi(s) = \arg\max_a [\ldots]$. On peut substituer directement.
> 
> **(iv) On observe que les deux étapes ont la même structure.** Évaluation et amélioration parcourent toutes deux *tous les états*. La seule différence est que l'évaluation utilise $\pi(s)$ tandis que l'amélioration utilise $\arg\max_a$. Si on injecte directement le $\max$ dans la mise à jour de $V$, les deux étapes fusionnent en :
> 
> $$V(s) \leftarrow \max_{a} \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma V(s') \big].$$
> 
> C'est la règle Value Iteration. La politique optimale est récupérée à la fin par un argmax final.

![[algo-5.png|324]]

> [!example] FrozenLake — Value Iteration à la main
> On lance VI à partir de $V_0=0$ partout, $\gamma=0.9$. À chaque itération, pour chaque $s$ : $V_{k+1}(s) = \max_a \sum_{s'} P(s'\mid s,a)[r + \gamma V_k(s')]$ — pas de politique fixée, on prend le max sur les 4 actions à chaque case. Trace sur les 3 mêmes états qu'en II.A :
> 
> | $k$ | $V_k(9)$ | $V_k(13)$ | $V_k(14)$ |
> |---|:---:|:---:|:---:|
> | 0 | 0.000 | 0.000 | 0.000 |
> | 1 | 0.000 | 0.000 | 0.333 |
> | 2 | 0.000 | 0.100 | 0.433 |
> | 3 | 0.060 | 0.160 | 0.493 |
> | 4 | 0.087 | 0.214 | 0.529 |
> | 5 | 0.122 | 0.249 | 0.556 |
> | 8 | 0.178 | 0.316 | 0.600 |
> | $\infty$ (144 itér.) | **0.247** | **0.380** | **0.639** |
> 
> **Trois observations :**
> - **Identique à $V_*$** calculé en I.C — normal, c'est la même équation de Bellman d'optimalité, juste résolue par itération plutôt que par l'algorithme Value Iteration "générique" appelé différemment.
> - **Convergence beaucoup plus lente à converger complètement** ($144$ itérations pour $10^{-10}$ près) que le nombre de *tours* de Policy Iteration ($2$) — mais chaque itération de VI est nettement moins chère qu'un tour de PI, qui contient lui-même toute une boucle de Policy Evaluation interne. C'est tout le compromis évoqué en introduction de II.D : VI fait moins de travail par itération, PI fait moins d'itérations externes.
> - **Politique optimale extraite à la fin** : $\pi_*(s) = \arg\max_a[\ldots]$ — c'est exactement la carte de flèches obtenue en I.C et confirmée par Policy Iteration en II.C. Les trois méthodes (formule close, Policy Iteration, Value Iteration) convergent vers la même réponse, par des chemins de calcul différents.

> 💡 **Bilan DP.** On a vu trois algorithmes (Policy Evaluation, Policy Iteration, Value Iteration) qui sont en réalité **trois lectures de la même équation de Bellman**. Évaluer = itérer Bellman avec une politique fixée. Optimiser = itérer Bellman avec un $\max$. Tous convergent en $\gamma^k$ grâce à la propriété de contraction de l'opérateur de Bellman (résultat admis ici). Reste un seul problème : tout ça suppose qu'on **connaît le modèle** $\mathbf{P}$ et $R$. C'est ce qu'on lâche maintenant en passant à Monte Carlo et TD Learning.

## III. Monte Carlo (model-free)

**Le contexte.** Toute la section II reposait sur une hypothèse forte : on connaît la matrice de transition $\mathbf{P}$ et la fonction de récompense $R$. C'est rarement le cas en pratique. Quand un agent joue à un jeu vidéo, conduit une voiture ou trade un portefeuille, il **n'a pas** de modèle explicite de l'environnement — il observe seulement des séquences d'états, d'actions et de récompenses.

Les méthodes **Monte Carlo** (MC) abandonnent l'hypothèse de modèle connu. Elles n'ont besoin que d'**expérience** : des trajectoires complètes échantillonnées dans l'environnement.

**L'idée.** La fonction de valeur $v_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ est une **espérance conditionnelle**. En DP, on la calculait *exactement*, parce qu'on connaissait $\mathbf{P}$ et $R$ (on pouvait sommer sur tous les futurs possibles). Sans modèle, cette somme exacte est hors de portée. L'idée de Monte Carlo, c'est justement le nom : **remplacer une espérance qu'on ne peut pas calculer par une moyenne empirique sur des échantillons tirés de cette même distribution** —

$$v_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] \;\approx\; \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)},$$

où chaque $G_t^{(i)}$ est un retour *réellement observé* en jouant $\pi$ (un échantillon de la variable aléatoire $G_t$), et $N(s)$ le nombre de tels échantillons collectés pour l'état $s$. C'est tout le principe : on génère plein d'épisodes en suivant $\pi$, on récupère les retours observés depuis chaque visite à $s$, et on en prend la moyenne. Par la loi des grands nombres, cette moyenne empirique converge vers l'espérance $v_\pi(s)$ quand $N(s) \to \infty$.

> 💡 **MC est tout entier du côté "échantillonner".** C'est l'autre versant de la dichotomie qu'on avait posée en I.B (rappelée en intro de II) : DP propage la distribution exacte (modèle connu), MC moyenne sur des trajectoires concrètes (modèle inconnu). Les deux cherchent à résoudre Bellman, mais avec deux outils statistiques différents.

> [!important] Trois propriétés à retenir avant d'attaquer les algos
> - **Tâches épisodiques uniquement.** Le retour $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ doit être un nombre bien défini, donc fini. MC suppose donc que les épisodes terminent toujours (état terminal, ou troncature après $T$ pas).
> - **Pas de bootstrap.** *Bootstrap* = utiliser une estimation de $V(s')$ pour mettre à jour $V(s)$. C'est ce que fait DP (et plus tard TD). MC, lui, calcule chaque retour **à partir des récompenses réelles** observées jusqu'à la fin de l'épisode — sans jamais utiliser d'autres $V$. Conséquence : les estimations de chaque état sont **indépendantes**. C'est plus simple à analyser, mais on ne profite pas de la structure récursive de Bellman.
> - **Mise à jour en fin d'épisode.** Pour calculer $G_t$, il faut connaître toutes les récompenses jusqu'à la fin. Donc MC ne peut pas mettre à jour à chaque pas — il attend la fin de chaque épisode. Pas d'apprentissage en ligne, contrairement à TD (IV).

### A. MC Prediction (MC Evaluation)

**Le problème.** *Prediction* signifie : on a une politique $\pi$ fixée, et on veut estimer $v_\pi$. Mêmes objectifs que Policy Evaluation (II.A), mais sans le modèle.

**Le matériau de travail.** On génère un certain nombre d'épisodes en suivant $\pi$ :

$$\text{Épisode } i : \quad s_0^{(i)}, \, a_0^{(i)}, \, r_0^{(i)}, \, s_1^{(i)}, \, a_1^{(i)}, \, r_1^{(i)}, \, \ldots, \, s_{T_i}^{(i)}.$$

Pour chaque épisode et chaque pas $t$, on calcule le **retour observé**

$$G_t^{(i)} = r_t^{(i)} + \gamma \, r_{t+1}^{(i)} + \gamma^2 \, r_{t+2}^{(i)} + \cdots + \gamma^{T_i - t - 1} \, r_{T_i - 1}^{(i)}.$$

C'est un échantillon (bruité) de $G_t \mid s_t = s_t^{(i)}$. On va simplement **moyenner** ces échantillons par état pour estimer $v_\pi$.

Reste une subtilité : un état peut être visité plusieurs fois dans un même épisode. Compte-t-on chaque visite, ou seulement la première ? Deux conventions, selon la réponse.

#### First-Visit MC

> [!warning] First-Visit MC
> Pour chaque épisode $i$ et chaque état $s$, on ne considère que la **première occurrence** de $s$ dans cet épisode. Si elle se produit au pas $t^*$, on enregistre le retour $G_{t^*}^{(i)}$.
> 
> L'estimateur est la moyenne de tous ces retours collectés sur l'ensemble des épisodes :
> 
> $$\hat V(s) \;=\; \frac{1}{N(s)} \sum_{\text{épisodes où } s \text{ apparaît}} G_{\text{première visite}}.$$
> 
> Quand $N(s) \to \infty$, $\hat V(s) \to v_\pi(s)$ par la loi des grands nombres. Variance en $1/\sqrt{N(s)}$.

Le grand intérêt de First-Visit : les retours collectés à travers les épisodes sont **indépendants et identiquement distribués** (i.i.d.), parce que chaque épisode est tiré indépendamment et qu'on ne prend qu'**un** retour par épisode. C'est ce qui rend l'analyse statistique propre.

> [!note]- Pseudo-code (First-Visit MC)
> ```
> Entrée : politique pi, facteur d'actualisation gamma
> Sortie : V ≈ v_pi
> 
> Initialiser V(s) = 0, N(s) = 0 pour tout s in S
> Boucle (sur les épisodes) :
>     Générer un épisode complet (s_0, a_0, r_0, s_1, ..., s_T) suivant pi
>     Pour chaque état s visité dans l'épisode :
>         Soit t* l'instant de la PREMIÈRE visite à s
>         G ← r_{t*} + gamma * r_{t*+1} + ... + gamma^(T-t*-1) * r_{T-1}
>         N(s) ← N(s) + 1
>         V(s) ← V(s) + (G - V(s)) / N(s)    # moyenne incrémentale, voir plus bas
> Retourner V
> ```

**Exemple :** Pour estimer $v_{\pi}(X)$ le calcul du gain $(G_t)$ ne commence qu'à partir de la première apparition de l'état $X$ dans chaque épisode. Eg pour l'épisode 1 on a $G_t=R_0+R_1+R_2$ avec $\gamma=1$, le gain est donc juste une somme simple des récompenses jusqu'à l'état terminal $Z$. 

![[Pasted image 20260502111915.png]]
Figure. First-Vist MC Example

#### Every-Visit MC

> [!warning] Every-Visit MC
> Variante : on enregistre le retour à **chaque** visite de $s$ dans chaque épisode, pas seulement la première.
> 
> L'estimateur est la moyenne sur toutes ces visites :
> 
> $$\hat V(s) \;=\; \frac{1}{N(s)} \sum_{\text{toutes visites à } s} G_t.$$
> 
> Converge également vers $v_\pi(s)$, mais l'analyse théorique est plus délicate parce que les retours collectés dans un même épisode sont **corrélés** (ils partagent une portion de futur).

> [!note]- Pseudo-code (Every-Visit MC)
> ```
> Entrée : politique pi, facteur d'actualisation gamma
> Sortie : V ≈ v_pi
> 
> Initialiser V(s) = 0, N(s) = 0 pour tout s in S
> Boucle (sur les épisodes) :
>     Générer un épisode complet (s_0, a_0, r_0, s_1, ..., s_T) suivant pi
>     Pour chaque pas t = 0, 1, ..., T-1 de l'épisode :
>         s ← s_t
>         G ← r_t + gamma * r_{t+1} + ... + gamma^(T-t-1) * r_{T-1}
>         N(s) ← N(s) + 1
>         V(s) ← V(s) + (G - V(s)) / N(s)
> Retourner V
> ```

> 💡 **First-Visit ou Every-Visit ?** Les deux convergent vers $v_\pi$. First-Visit a une analyse plus simple (échantillons i.i.d., biais nul), Every-Visit utilise plus d'échantillons par épisode (donc moins d'épisodes nécessaires) au prix d'une corrélation entre échantillons. En pratique, la différence est mineure ; First-Visit reste le choix par défaut dans la littérature pédagogique.

**Exemple :** Pour estimer $v_{\pi}(X)$ le calcul du gain $(G_t)$ ne commence qu'à partir de la première apparition de l'état $X$ dans chaque épisode. Eg pour l'épisode 1 on a $G_t=R_0+R_1+R_2$ avec $\gamma=1$, le gain est donc juste une somme simple des récompenses jusqu'à l'état terminal $Z$. Sauf que, vu que c'est "every-visit" dés que l'état $X$ apparait une seconde fois on refait le processus.

![[Pasted image 20260502112036.png|459]]
Figure. Every-visit MC

#### Mise à jour incrémentale

Calculer la moyenne $\hat V(s) = \frac{1}{N} \sum_k G_k$ naïvement nécessite de stocker tous les retours observés. Or on peut faire mieux : il existe une formule récursive qui met à jour la moyenne après chaque nouveau retour, sans rien stocker.

> [!warning] Moyenne incrémentale
> Si $\mu_n$ désigne la moyenne des $n$ premiers retours $x_1, \ldots, x_n$, alors
> 
> $$\mu_n \;=\; \mu_{n-1} + \frac{1}{n}\big( x_n - \mu_{n-1} \big).$$
> 
> *On part de l'ancienne estimation, et on la corrige d'une fraction $1/n$ de l'erreur entre la nouvelle observation et l'ancienne moyenne.*

> [!note]- Dérivation
> Par définition $\mu_n = \frac{1}{n} \sum_{k=1}^n x_k$. On extrait le dernier terme :
> 
> $$\mu_n = \frac{1}{n}\Big( x_n + \sum_{k=1}^{n-1} x_k \Big) = \frac{1}{n}\big( x_n + (n-1)\mu_{n-1} \big) = \mu_{n-1} + \frac{1}{n}(x_n - \mu_{n-1}).$$

Appliquée à MC, la mise à jour devient :

$$V(s) \;\leftarrow\; V(s) + \frac{1}{N(s)}\big( G - V(s) \big).$$

C'est la forme qui apparaît dans les pseudo-codes ci-dessus.

> 💡 **La forme universelle du RL.** Cette structure
> 
> $$\text{nouvelle estimation} \;\leftarrow\; \text{ancienne estimation} + \alpha \times \big(\text{cible} - \text{ancienne estimation}\big)$$
> 
> revient **partout** dans le RL — TD, SARSA, Q-learning, deep RL... C'est tellement central que beaucoup d'algorithmes se résument à *"prends cette forme et change la cible"*. En MC, la cible est le retour $G_t$ ; en TD on verra que la cible devient $r_t + \gamma V(s_{t+1})$. Le **pas d'apprentissage** $\alpha$ vaut $1/N(s)$ pour MC pur (vraie moyenne), mais on le remplace souvent par une constante (par exemple $\alpha = 0.1$) pour donner plus de poids aux observations récentes — utile quand l'environnement n'est pas parfaitement stationnaire ou quand on alterne avec des étapes d'amélioration.

### B. Policy Control

**L'idée.** En III.A on évaluait $v_\pi$ à partir d'épisodes — c'était le pendant MC de Policy Evaluation. Maintenant on veut le pendant MC de **Policy Iteration** : alterner évaluation et amélioration jusqu'à converger vers $\pi_*$, mais sans modèle. Deux problèmes nouveaux apparaissent par rapport à II.C.

#### Problème 1 : on a besoin de $Q$, pas de $V$

En DP (II.B), la politique gloutonne s'écrivait

$$\pi'(s) = \arg\max_a \Big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V(s') \Big].$$

Le $\max_a$ utilise explicitement la dynamique $\mathbf{P}$ et la récompense $R$. **Sans modèle, on ne peut pas calculer ce max.**

La parade : apprendre directement la **Q-fonction** $Q(s, a)$. Une fois qu'on a $Q$, l'amélioration gloutonne devient triviale et **purement model-free** :

$$\pi'(s) = \arg\max_a Q(s, a).$$

> 💡 **Pourquoi $Q$ et pas $V$.** C'est un point qui revient souvent : tous les algos de contrôle model-free (MC, SARSA, Q-learning, DQN…) apprennent $Q$ et non $V$. La raison est exactement celle-ci — sans $\mathbf{P}$, $V$ ne suffit pas à choisir l'action. C'est le cas typique en pratique, et c'est pour ça que la Q-fonction est l'objet central du RL appliqué.
> 
> Le pendant pratique : on doit visiter chaque paire $(s, a)$ assez souvent pour que $Q(s, a)$ converge. Cela contraste avec la prédiction où on ne visitait que les états sous $\pi$.

L'algorithme MC Prediction pour $Q$ est identique à III.A, en remplaçant simplement les compteurs sur les états par des compteurs sur les couples état-action :

$$Q(s, a) \;\leftarrow\; Q(s, a) + \frac{1}{N(s, a)}\big( G_t - Q(s, a) \big),$$

appliqué à chaque visite (ou première visite) du couple $(s_t = s, a_t = a)$ dans un épisode.

#### Problème 2 : exploration

Si on est gloutonne par rapport à notre $Q$ courante, on choisit toujours $\arg\max_a Q(s, a)$. Mais $Q$ est une **estimation** — pour les actions qu'on n'a jamais essayées, $Q$ est arbitraire (souvent initialisé à $0$). Si l'action vraiment optimale a une mauvaise estimation initiale, on ne la choisira jamais, donc on ne mettra jamais à jour son estimation, donc elle restera mauvaise. **L'agent reste prisonnier d'une politique sous-optimale.**

> [!example] Door A vs Door B
> Imaginons un agent qui ouvre des portes pour gagner des récompenses. Il essaie d'abord la Porte B et reçoit $0$. Puis la Porte A et reçoit $1$. Avec une politique purement gloutonne, à chaque visite suivante il prendra Porte A (parce que $Q(\text{A}) = 1 > Q(\text{B}) = 0$) — il y gagne $3$, puis $1$, puis $2$… et ne réessaiera **jamais** la Porte B. Or peut-être qu'en y retournant il aurait découvert $Q(\text{B}) = 100$.
> 
> L'estimation initiale de $Q(\text{B})$ après une seule visite ($0$) est **bruitée** : avec une seule observation, on n'a aucune idée fiable de la vraie valeur. Une politique gloutonne agit comme si chaque estimation était certaine — ce qu'elle n'est pas. D'où la nécessité d'**explorer**.

![[Pasted image 20260502112120.png]]

La solution standard : la politique **ε-greedy**, qui prend la meilleure action *la plupart du temps* mais teste de temps en temps une autre action au hasard.

> [!warning] Politique ε-greedy
> Soit $\varepsilon \in [0, 1]$ et $|A|$ le nombre d'actions. La politique ε-greedy par rapport à $Q$ est définie par
> 
> $$\pi(a \mid s) \;=\; \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|A|} & \text{si } a = \arg\max_{a'} Q(s, a'), \\[6pt] \dfrac{\varepsilon}{|A|} & \text{sinon}. \end{cases}$$
> 
> Lecture : avec probabilité $1 - \varepsilon$ on choisit l'action gloutonne (exploitation) ; avec probabilité $\varepsilon$ on tire une action au hasard uniformément (exploration). La probabilité totale de l'action gloutonne combine donc les deux cas — d'où le terme additionnel $\varepsilon / |A|$.

> [!note]- Vérification de la formule
> Avec $\varepsilon = 0.1$ et $|A| = 4$ actions, et en supposant que l'action gloutonne est $a_3$ : 
> 
> - Probabilité de $a_3$ : $1 - 0.1 + 0.1 / 4 = 0.925$.
> - Probabilité de chacune des trois autres : $0.1 / 4 = 0.025$.
> 
> Total : $0.925 + 3 \times 0.025 = 1$ ✓.

> 💡 **Lien avec le théorème de Policy Improvement (II.B).** Une politique ε-greedy par rapport à $Q^\pi$ est garantie d'être au moins aussi bonne que $\pi$, comme dans le cas DP. Ça reste vrai même avec exploration : la perte due aux $\varepsilon / |A|$ d'exploration aléatoire est plus que compensée par le passage à la nouvelle action gloutonne (quand celle-ci change). C'est ce qui justifie qu'on puisse boucler l'évaluation et l'amélioration ε-greedy.

#### GLIE : les deux conditions de convergence

L'ε-greedy permet d'explorer, mais avec un $\varepsilon$ constant on ne convergera jamais vers $\pi_*$ : on continuera à prendre des actions aléatoires une fraction $\varepsilon$ du temps. Pour atteindre l'optimum, il faut **que $\varepsilon$ décroisse vers 0 — mais pas trop vite**.

> [!warning] Conditions GLIE (Greedy in the Limit with Infinite Exploration)
> Une suite de politiques $\{\pi_k\}$ est dite **GLIE** si elle satisfait simultanément :
> 
> 1. **Infinite Exploration** : chaque paire $(s, a)$ est visitée un nombre infini de fois,
> 
> $$N_k(s, a) \to \infty \quad \text{quand } k \to \infty, \qquad \forall (s, a).$$
> 
> 2. **Greedy in the Limit** : la politique converge vers la politique gloutonne par rapport à la limite des $Q$,
> 
> $$\pi_k(a \mid s) \;\xrightarrow[k \to \infty]{} \;\mathbb{1}\{ a = \arg\max_{a'} Q_*(s, a') \}.$$
> 
> Sous ces deux conditions, MC Control avec ε-greedy converge vers $\pi_*$ et $Q_*$.

Une recette simple pour satisfaire GLIE : faire **décroître $\varepsilon$ comme $1/k$** où $k$ est le numéro d'épisode. Avec $\varepsilon_k = 1/k$ :

- L'exploration ne s'arrête jamais ($\varepsilon_k > 0$ pour tout $k$ fini), donc la condition 1 est satisfaite.
- Mais $\varepsilon_k \to 0$, donc à long terme la politique devient gloutonne — condition 2 satisfaite.

#### Algorithme MC Control complet

> [!warning] MC Control avec ε-greedy
> 1. Initialiser $Q(s, a) = 0$ et $N(s, a) = 0$ pour tout $(s, a)$.
> 2. Pour chaque épisode $k = 1, 2, 3, \ldots$ :
>    - Choisir $\varepsilon_k$ (par exemple $\varepsilon_k = 1/k$).
>    - Générer un épisode complet $(s_0, a_0, r_0, \ldots, s_T)$ en suivant la politique ε-greedy par rapport à $Q$.
>    - Pour chaque pas $t$ visité dans l'épisode :
>      - $G \leftarrow r_t + \gamma r_{t+1} + \ldots + \gamma^{T-t-1} r_{T-1}$
>      - $N(s_t, a_t) \leftarrow N(s_t, a_t) + 1$
>      - $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \dfrac{1}{N(s_t, a_t)} \big( G - Q(s_t, a_t) \big)$
> 3. Retourner $\pi_*(s) = \arg\max_a Q(s, a)$.


![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/im5.png|558]]
Caption. Pseudo-code MC Control

#### Pas constant α et politique non-stationnaire

Il y a une raison pratique de remplacer $1/N(s, a)$ par un pas d'apprentissage constant $\alpha$ (par exemple $\alpha = 0.1$) :

$$Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) + \alpha\big( G_t - Q(s_t, a_t) \big).$$

Le facteur $1/N(s, a)$ produit la **vraie moyenne** de tous les retours observés depuis le début. C'est correct quand la cible est stationnaire (cas Prediction). Mais en Control, **la politique change à chaque épisode** : les retours observés au début (sous une politique mauvaise) ne sont plus représentatifs des retours sous la politique courante. On veut donner plus de poids aux retours récents et **oublier** progressivement les anciens.

C'est exactement ce que fait un pas constant : la mise à jour est une moyenne mobile exponentielle, où l'influence des anciens retours décroît en $(1-\alpha)^k$.

> 💡 **Boucle sur la forme universelle.** On retrouve ici la structure annoncée en III.A :
> 
> $$\text{nouvelle estimation} \;\leftarrow\; \text{ancienne estimation} + \alpha \big(\text{cible} - \text{ancienne estimation}\big),$$
> 
> avec cible $= G_t$ pour MC, et le rôle de $\alpha$ est de régler l'inertie. En TD (IV) on changera la cible — mais le squelette reste identique.

## IV. TD Learning

**L'idée centrale.** S'il fallait identifier *une* idée centrale et originale au RL, ce serait le **TD Learning** (Temporal-Difference Learning). C'est une combinaison directe des deux approches qu'on a vues jusqu'ici :

- **Comme MC** : TD apprend par expérience pure, sans modèle de l'environnement. On observe des transitions $(s_t, r_t, s_{t+1})$ et on s'en sert pour mettre à jour les estimations.
- **Comme DP** : TD fait du **bootstrap**, c'est-à-dire qu'il utilise sa propre estimation $V(s_{t+1})$ pour mettre à jour $V(s_t)$. Pas besoin d'attendre la fin de l'épisode pour avoir un retour réel — on bricole un retour à partir de l'estimation courante.

> 💡 **L'image à retenir.** TD = MC + DP. On garde le côté model-free de MC (on échantillonne) et le côté incrémental de DP (on bootstrap). Le résultat : un algorithme qui apprend en ligne, à chaque pas de temps, sans modèle, et qui fonctionne même sur des tâches non-épisodiques.

**Generalized Policy Iteration (GPI).** Pour le contrôle (trouver $\pi_*$), DP, MC et TD utilisent tous le même squelette : alterner **évaluation** (estimer $v_\pi$ ou $q_\pi$) et **amélioration** (politique gloutonne par rapport à l'estimation). Ce squelette est appelé **Generalized Policy Iteration**. Les trois familles d'algorithmes ne diffèrent que par leur méthode d'évaluation. C'est pourquoi on commence à chaque fois par le problème de prédiction.

### A. TD Prediction

**Le problème.** Comme en III.A : on a une politique $\pi$ fixée, on veut estimer $v_\pi$. Sans modèle. Mais on veut faire mieux que MC — apprendre en ligne, sans attendre la fin de l'épisode.

**Rappel : la mise à jour MC.** En III.A on utilisait

$V(s_t) \;\leftarrow\; V(s_t) + \alpha\big( G_t - V(s_t) \big),$

où $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ est le retour observé jusqu'à la fin de l'épisode. Le problème : pour calculer $G_t$, il faut **attendre la fin de l'épisode**. Pas d'apprentissage en ligne.

**Le coup de génie.** On revient à l'équation de Bellman pour $v_\pi$ :

$v_\pi(s) = \mathbb{E}_\pi\big[ G_t \mid s_t = s \big] = \mathbb{E}_\pi\big[ r_t + \gamma \, v_\pi(s_{t+1}) \mid s_t = s \big].$

Ce que ça nous dit : $r_t + \gamma \, v_\pi(s_{t+1})$ est un **échantillon non-biaisé** du retour $G_t$ — il a la même espérance. On peut donc l'utiliser comme cible à la place de $G_t$ ! Sauf qu'on ne connaît pas $v_\pi$ (c'est ce qu'on cherche), alors on triche : on utilise notre **estimation courante** $V(s_{t+1})$ à la place.

C'est exactement ça, le **bootstrap** : utiliser une estimation pour en améliorer une autre.

> [!warning] One-step TD, ou TD(0)
> $V(s_t) \;\leftarrow\; V(s_t) + \alpha\Big[ r_t + \gamma \, V(s_{t+1}) - V(s_t) \Big]$
> 
> où $\alpha \in (0, 1]$ est le pas d'apprentissage. À chaque transition observée $(s_t, r_t, s_{t+1})$, on met à jour $V(s_t)$ — pas besoin d'attendre la fin de l'épisode. La suite des estimations converge vers $v_\pi$ pour $\alpha$ assez petit.

**Vocabulaire essentiel.**

> 💡 **TD target et TD error.** Dans la mise à jour TD(0), on distingue trois objets :
> 
> - **TD target** : $r_t + \gamma \, V(s_{t+1})$. C'est la cible bootstrap-ée — l'estimation actuelle de ce que devrait valoir $V(s_t)$.
> - **TD error** : $\delta_t = r_t + \gamma \, V(s_{t+1}) - V(s_t)$. L'écart entre la cible et l'estimation actuelle. C'est le signal d'apprentissage.
> - **Lecture comme combinaison convexe** : on peut réécrire la mise à jour
> 
> $V(s_t) \;\leftarrow\; (1 - \alpha) \, V(s_t) + \alpha \big[ r_t + \gamma \, V(s_{t+1}) \big].$
> 
> $V(s_t)$ devient un mélange entre l'ancienne valeur (poids $1 - \alpha$) et la TD target (poids $\alpha$). Plus $\alpha$ est grand, plus on fait confiance à la nouvelle observation au détriment de l'ancienne estimation.

On retrouve une fois de plus la **forme universelle** annoncée en III.A : $\text{nouvelle} \leftarrow \text{ancienne} + \alpha (\text{cible} - \text{ancienne})$. Seul le choix de la cible distingue MC et TD.

#### Tableau comparatif DP / MC / TD

| | **DP** | **MC** | **TD** |
|---|:---:|:---:|:---:|
| Modèle $\mathbf{P}$, $R$ requis ? | Oui | Non | Non |
| Bootstrap (utilise $V(s')$) ? | Oui | Non | Oui |
| Échantillonnage de trajectoires ? | Non (espérance exacte) | Oui | Oui |
| Cible de la mise à jour | $\mathbb{E}[r + \gamma V(s')]$ | $G_t$ (retour complet) | $r_t + \gamma V(s_{t+1})$ |
| Apprentissage en ligne ? | — | Non (fin d'épisode) | Oui (chaque pas) |
| Tâches non-épisodiques ? | Oui | Non | Oui |

> 💡 **Lecture du tableau.** TD est la seule méthode qui combine *bootstrap* (du côté DP) et *échantillonnage* (du côté MC). C'est ce qui lui donne ses deux avantages-clés : pas besoin de modèle, et apprentissage en ligne.

#### TD vs MC : biais et variance

Les deux méthodes estiment la même quantité $v_\pi$, mais avec un compromis biais/variance différent.

- **MC est non biaisé** mais a une **variance élevée**. Le retour $G_t$ est une somme de récompenses bruitées sur tout un épisode — chaque trajectoire diffère beaucoup d'une autre.
- **TD est biaisé** mais a **moins de variance**. La TD target $r_t + \gamma V(s_{t+1})$ utilise une seule récompense (faible bruit) mais s'appuie sur l'estimation $V(s_{t+1})$ qui n'est pas la vraie valeur — d'où le biais.

En pratique, le compromis penche presque toujours en faveur de TD : la réduction de variance accélère drastiquement la convergence. C'est pour ça que TD est devenu la base de tous les algos modernes (SARSA, Q-learning, DQN, A2C…).

#### Avantages pratiques de TD

- **Apprentissage en ligne.** On met à jour à chaque pas, sans attendre la fin de l'épisode.
- **Tâches non-épisodiques.** TD fonctionne sur des flux d'expérience continus (pas besoin d'état terminal).
- **Convergence plus rapide en moyenne** que MC sur la plupart des tâches, grâce à la variance réduite.

> [!note]- Pseudo-code TD(0)
> ```
> Entrée : politique pi, pas alpha, facteur d'actualisation gamma
> Sortie : V ≈ v_pi
> 
> Initialiser V(s) = 0 pour tout s in S
> Observer s_0
> Boucle (sur les pas de temps t = 0, 1, 2, ...) :
>     Choisir a_t ~ pi(. | s_t)
>     Exécuter a_t, observer r_t et s_{t+1}
>     V(s_t) ← V(s_t) + alpha * [r_t + gamma * V(s_{t+1}) - V(s_t)]
>     s_t ← s_{t+1}
> ```
> 
> Note : pas de boucle externe sur les épisodes — TD est en ligne. Si l'environnement est épisodique, on relance une trajectoire à chaque épisode terminé, mais la mise à jour reste la même à chaque pas.

### B. TD Control

**L'idée.** En IV.A on faisait de la prédiction TD : politique fixée, on estime $v_\pi$. Maintenant on veut le **contrôle** : trouver $\pi_*$. Comme en III.B (MC Control), on suit le squelette **GPI** — alterner évaluation et amélioration — mais en utilisant TD pour l'évaluation au lieu de MC.

Les deux problèmes identifiés en III.B se reposent à l'identique :

- **On a besoin de $Q$, pas de $V$.** Sans modèle, $V$ ne suffit pas à choisir l'action. On apprend donc directement la Q-fonction.
- **Il faut explorer.** On utilise une politique $\varepsilon$-greedy par rapport à $Q$, avec décroissance GLIE de $\varepsilon$.

Ce qui change par rapport à MC Control : la mise à jour de $Q$ se fait à chaque pas (bootstrap) au lieu d'à la fin de l'épisode. Et il y a deux façons naturelles de bootstrap-er sur $Q$ — d'où deux algorithmes : **SARSA** et **Q-learning**.

#### SARSA(0) - On policy

**Le nom.** SARSA vient des cinq objets utilisés dans la mise à jour : **S**tate $s_t$, **A**ction $a_t$, **R**eward $r_t$, next **S**tate $s_{t+1}$, next **A**ction $a_{t+1}$. À chaque pas, on a besoin de ce quintuplet pour mettre à jour.

**Construction.** On part de l'équation de Bellman pour $q_\pi$ :

$$q_\pi(s, a) = \mathbb{E}_\pi\big[ r_t + \gamma \, q_\pi(s_{t+1}, a_{t+1}) \mid s_t = s,\, a_t = a \big].$$

Même coup de génie qu'en TD(0) : on remplace la vraie $q_\pi$ par notre estimation $Q$, et on observe une réalisation au lieu de calculer l'espérance.

> [!warning] Mise à jour SARSA(0)
> $Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) + \alpha\Big[ r_t + \gamma \, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \Big]$
> 
> où $a_{t+1}$ est l'action effectivement choisie selon la politique $\varepsilon$-greedy courante en $s_{t+1}$.

**Le point clé : $a_{t+1}$ est l'action que l'agent va *réellement* prendre.** On l'tire selon $\pi$ au pas $t+1$, on l'utilise dans la mise à jour, *puis* on l'exécute. La mise à jour évalue donc la politique que l'agent suit réellement, $\varepsilon$-greedy comprise.

> 💡 **SARSA est on-policy.** *On-policy* signifie que la politique évaluée par les mises à jour est la même que celle que l'agent suit pour explorer. SARSA évalue $q_{\pi_\varepsilon}$ où $\pi_\varepsilon$ est la politique $\varepsilon$-greedy courante — il "prend en compte" le coût de l'exploration. Si l'exploration peut faire prendre une mauvaise action $a_{t+1}$, $Q(s_t, a_t)$ baisse en conséquence.


![[im1 1.png|497]]

**Lecture du schéma.** La ligne du bas est la trajectoire de l'agent, pas après pas : $S_0, A_0, R_1, S_1, A_1 \mid R_2, S_2, A_2 \mid \ldots$. Chaque groupe séparé par une barre correspond à un pas de temps. Dès qu'on a le quintuplet $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ — d'où le nom SARSA — on met à jour $Q(S_t, A_t)$ (flèche pointillée)

$$
Q\left(s_t, a_t\right) \leftarrow Q\left(s_t, a_t\right)+\alpha\left[r_t+\gamma Q\left(s_{t+1}, a_{t+1}\right)-Q\left(s_t, a_t\right)\right]
$$

puis on rafraîchit $\pi$ en $\varepsilon$-greedy par rapport au $Q$ tout juste mis à jour,

$$
\pi(a \mid s)= \begin{cases}1-\varepsilon+\frac{\varepsilon}{|A|} & \text { si } a=\arg \max _{a^{\prime}} Q\left(s, a^{\prime}\right), \\ \frac{\varepsilon}{|A|} & \text { sinon. }\end{cases}
$$

*avant* de repartir sur le pas suivant. C'est cette ré-évaluation continue de $\pi$ à partir d'un $Q$ qui bouge à chaque pas qui rend SARSA on-policy : l'action $A_{t+1}$ utilisée dans la mise à jour est celle que l'agent va réellement jouer ensuite.

#### SARSAMAX (ou Q-Learning) - Off policy

**L'astuce.** SARSA utilise $Q(s_{t+1}, a_{t+1})$ — l'action que l'agent va réellement prendre. Q-learning utilise $\max_{a'} Q(s_{t+1}, a')$ — la **meilleure** action possible en $s_{t+1}$, indépendamment de ce que l'agent va vraiment faire.

Pourquoi ce choix ? Parce qu'on revient à l'équation de Bellman *d'optimalité* (vue en I.C) :

$$q_*(s, a) = \mathbb{E}\Big[ r_t + \gamma \max_{a'} q_*(s_{t+1}, a') \mid s_t = s,\, a_t = a \Big].$$

Le $\max$ remplace l'espérance sur $\pi$. Q-learning échantillonne directement cette équation, sans même attendre que l'agent prenne $a_{t+1}$.

> [!warning] Mise à jour Q-learning (SARSAMAX)
> $Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) + \alpha\Big[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]$
> 
> On utilise le maximum sur les actions à l'état suivant, sans tenir compte de l'action que l'agent va effectivement choisir.

> 💡 **Q-learning est off-policy.** *Off-policy* signifie que la politique évaluée (la politique gloutonne, via le $\max$) est *différente* de la politique suivie par l'agent (l'$\varepsilon$-greedy, nécessaire pour explorer). Q-learning estime directement $q_*$ — la Q-fonction *optimale* — même si l'agent agit de manière sous-optimale pour explorer. C'est ce qui en fait le premier algorithme RL réellement "en boucle fermée" sur l'optimalité.

![[images/3-Apprentissage automatique/07_Reinforcement learning/RL Tabulaire/im3.png]]

**Lecture du schéma.** Même trajectoire que pour SARSA, $S_0, A_0, R_1, S_1 \mid A_1, R_2, S_2 \mid \ldots$, mais les groupes ne se coupent pas au même endroit : ici le pas se referme dès qu'on observe $S_{t+1}$, *avant* même de choisir $A_{t+1}$. C'est la différence visuelle qui traduit l'astuce du $\max$ : pour mettre à jour $Q(S_0, A_0)$, Q-learning n'a besoin ni de connaître ni d'attendre l'action que l'agent va réellement jouer en $S_1$ — il regarde directement la meilleure Q-valeur disponible, $\max_{a} Q(S_1, a)$. La flèche pointillée revient donc de $S_1$ (pas de $A_1$) vers la mise à jour de $Q(S_0, A_0)$, et de même de $S_2$ vers $Q(S_1, A_1)$.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Big(R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t)\Big).$$

Le $\pi \leftarrow \varepsilon\text{-greedy}(Q)$ est quand même rafraîchi après chaque mise à jour — il faut bien une politique pour que l'agent *agisse* et explore — mais cette politique n'intervient **nulle part** dans la cible de la mise à jour. C'est exactement ce découplage (la politique suivie sert à explorer, la politique évaluée dans la cible est la gloutonne pure via le $\max$) qui rend Q-learning off-policy, contrairement à SARSA où $A_{t+1}$ tiré par $\pi_\varepsilon$ apparaissait directement dans la cible.

#### SARSA vs Q-learning : on-policy vs off-policy

La différence entre les deux algorithmes tient à *un seul caractère* dans la mise à jour, mais elle change la nature de ce qu'on apprend.

| | **SARSA** | **Q-learning** |
|---|:---:|:---:|
| Cible | $r_t + \gamma Q(s_{t+1}, a_{t+1})$ | $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ |
| Type | **on-policy** | **off-policy** |
| Politique évaluée | $\pi_\varepsilon$ (celle suivie) | $\pi_*$ (gloutonne) |
| Tient compte de l'exploration ? | Oui | Non |
| Converge vers | $q_{\pi_\varepsilon}$ → $q_*$ si $\varepsilon \to 0$ | $q_*$ directement |

> 💡 **Pourquoi cette distinction est-elle importante ?** Le cas classique pour comprendre est le **cliff walking** de Sutton & Barto : un grid-world où marcher près d'une falaise rapporte $-100$ si on tombe. Q-learning apprend la politique optimale "longer la falaise au plus court". SARSA apprend une politique plus prudente "s'éloigner de la falaise" — parce qu'avec l'exploration $\varepsilon$-greedy, longer la falaise risque de tomber, donc SARSA en tient compte. Q-learning ignore ce risque parce qu'il évalue la politique purement gloutonne.
> 
> **En résumé** : Q-learning est plus agressif et apprend l'optimum théorique ; SARSA est plus conservateur et apprend l'optimum *sous l'exploration que l'on impose*. Si $\varepsilon \to 0$, les deux convergent vers la même politique $\pi_*$.


> 💡 **Bilan TD Control.** SARSA et Q-learning sont les deux algorithmes fondamentaux du contrôle TD. Q-learning est devenu dominant dans la littérature moderne parce qu'il apprend $q_*$ directement — c'est l'ancêtre de **DQN** (Deep Q-Network), qui remplace la table $Q$ par un réseau de neurones. SARSA reste pertinent quand le coût de l'exploration est réel (robotique, systèmes physiques) : on préfère une politique qui tient compte du fait qu'on explore.

## V. POMDP

*À venir.*
