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

![[images/3-Apprentissage automatique/Reinforcement learning/im1.png|391]]
**Figure 1.** Vue d'ensemble de l'interaction agent–environnement.

Ce cadre se distingue de l'apprentissage supervisé sur deux points essentiels :

- la cible n'est pas une étiquette correcte mais une récompense, potentiellement **différée** dans le temps — d'où le problème d'**attribution du crédit** (credit assignment) : à quelle action passée doit-on attribuer la récompense reçue maintenant ?
- l'agent génère lui-même ses données par ses actions, ce qui crée le compromis **exploration/exploitation** (exploration-exploitation tradeoff) : essayer des actions nouvelles pour mieux connaître l'environnement, ou exploiter ce qu'on sait déjà.

**Observabilité (observability).** Soit $S$ l'ensemble des états possibles du monde et $\{s_t\}$ la trajectoire d'états. Deux régimes :

- **Cas totalement observable** (fully observable) : $o_t = s_t$, l'agent voit l'état réel du monde. C'est le cadre du **MDP** (Markov Decision Process), qui sera l'hypothèse par défaut dans toute cette partie.
- **Cas partiellement observable** (partially observable) : $o_t \neq s_t$, l'agent ne voit qu'une observation bruitée ou incomplète de l'état réel. Pour décider, il maintient alors une distribution de probabilité sur l'état réel, appelée **état de croyance** (belief state). Ce cadre est modélisé par un **POMDP** (Partially Observable MDP), traité plus loin.

**Pourquoi le RL plutôt que la recherche ?** Quand le modèle de l'environnement (la dynamique et les récompenses) est entièrement connu et que l'espace d'états est petit et déterministe, des méthodes de recherche classiques en IA comme $A^*$ ou minimax suffisent à trouver une séquence d'actions optimale. Le RL devient nécessaire dès qu'**au moins une** des trois difficultés suivantes apparaît :

1. **Espace d'états trop grand** (jeux Atari, Go, problèmes de contrôle continus) — la recherche exhaustive est infaisable, il faut généraliser via une approximation de fonction (function approximation).
2. **Stochasticité** de la dynamique — il n'existe alors plus de "meilleure séquence d'actions", mais une politique optimale qui mappe états vers distributions sur actions.
3. **Modèle inconnu** — on ne connaît ni les probabilités de transition $p(s' \mid s, a)$ ni la fonction de récompense $r(s, a)$, et il faut les apprendre par interaction. C'est la motivation centrale du RL.

### B. Processus de décision markovien fini

On construit le MDP (Markov Decision Process) en trois temps : on part d'un processus de Markov (juste de la dynamique), on lui ajoute une récompense pour obtenir un MRP (Markov Reward Process), puis on ajoute des actions pour obtenir le MDP. Cette progression rend chaque ingrédient explicite et permet de comprendre où chaque hypothèse intervient.

> [!example] Fil rouge : régime de marché
> Pour ancrer chaque définition, on utilisera tout au long de cette section un exemple de **régime de marché** (market regime) à trois états : Bull (haussier), Sideways (latéral), Bear (baissier). Cet exemple suffit à illustrer toute la mécanique tabulaire et nous accompagnera ensuite jusqu'aux algorithmes de programmation dynamique. Les chiffres sont volontairement simples, choisis pour produire des résultats lisibles, pas calibrés sur des données réelles.

#### Processus de Markov

> [!warning] Propriété de Markov
> Un processus stochastique $(s_0, s_1, s_2, \ldots)$ à valeurs dans un espace d'états $S$ vérifie la **propriété de Markov** (Markov property) si
> 
> $$P(s_{t+1} \mid s_t, s_{t-1}, \ldots, s_0) = P(s_{t+1} \mid s_t)$$
> 
> autrement dit : *l'état présent contient toute l'information utile pour prédire le futur*. Une fois qu'on connaît $s_t$, le passé n'apporte rien de plus. C'est ce qu'on résume en disant qu'un processus de Markov est **sans mémoire** (memoryless).

![[rl_mp_markov_property.png]]

**Figure 2.** Propriété de Markov. Étant donné $s_t =$ Bull, la distribution du prochain état $s_{t+1}$ est entièrement spécifiée par les probabilités de transition. Les états passés (à gauche) n'ajoutent aucune information.

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

> [!example] Régime de marché — la matrice $\mathbf{P}$
> Avec $S = \{\text{Bull}, \text{Sideways}, \text{Bear}\}$, on choisit
> 
> $$
> \mathbf{P} \;=\;
> \begin{array}{r@{\hskip 8pt}c}
> & \begin{array}{ccc} \text{Bull} & \text{Sideways} & \text{Bear} \end{array} \\
> \begin{array}{r} \text{Bull} \\ \text{Sideways} \\ \text{Bear} \end{array} &
> \left(\begin{array}{ccc}
> 0.7 & 0.2 & 0.1 \\
> 0.3 & 0.4 & 0.3 \\
> 0.1 & 0.2 & 0.7
> \end{array}\right)
> \end{array}
> $$
> 
> Lecture : depuis Bull, on reste en Bull avec probabilité $0.7$, on passe en Sideways avec $0.2$, et on bascule directement en Bear avec seulement $0.1$. L'idée modélisée est que les régimes sont **persistants** (diagonale dominante) et qu'on transite rarement directement d'un extrême à l'autre sans passer par le régime latéral.

![[rl_mp_transition_graph.png|406]]

**Figure 3.** Diagramme de transition du régime de marché. Chaque arête porte la probabilité de la transition correspondante ; chaque ligne du diagramme somme à 1.

**Trajectoires (sample paths).** Une *réalisation* du processus est une suite d'états tirée selon $\mathbf{P}$ depuis un état initial. Tel quel, le processus de Markov ne fait que décrire une dynamique aléatoire — on n'a encore ni notion de "bien" ou "mal", ni de levier d'action. C'est ce qu'on ajoute dans les deux étapes suivantes.

> [!note]- Comment simule-t-on une trajectoire ?
> Deux choses différentes à ne pas confondre :
> 
> - **Propager une distribution.** Si $\mu^0$ est une distribution sur les états (par exemple $\mu^0 = [1, 0, 0]$ pour "on part de Bull avec certitude"), alors $\mu^k = \mu^0 \mathbf{P}^k$ donne la distribution à l'instant $k$. Cela décrit où l'on est *en moyenne sur tous les futurs possibles*. C'est l'angle utilisé par la programmation dynamique.
> 
> - **Échantillonner une trajectoire.** À chaque pas, on tire un état au hasard selon la ligne courante de $\mathbf{P}$ : sachant qu'on est en $s_t$, on échantillonne $s_{t+1} \sim \mathbf{P}[s_t, \cdot]$. C'est ainsi qu'on produit *une* réalisation concrète, comme la figure ci-dessous. C'est l'angle utilisé par les méthodes Monte Carlo.
> 
> **Lien entre les deux.** Si l'on simule des milliers de trajectoires et que l'on compte les fréquences des états à l'instant $k$, on retombe sur $\mu^k$ par la loi des grands nombres. Une trajectoire = un échantillon du processus ; la distribution = ce que voient toutes les trajectoires en moyenne. Cette tension entre *moyenner sur des échantillons* et *raisonner sur la distribution* est exactement la différence Monte Carlo / DP qu'on retrouvera plus loin.

![[rl_mp_trajectoire.png]]

**Figure 4.** Une trajectoire de 120 pas simulée à partir de la matrice $\mathbf{P}$ ci-dessus, démarrant en Bull. On voit la persistance des régimes (plages de plusieurs pas dans le même état) et les transitions stochastiques entre eux.

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

> [!example] Régime de marché — récompenses
> On prend pour récompense le rendement moyen mensuel (en %) du régime, l'agent étant supposé "passivement long" sur un indice :
> 
> $$R(\text{Bull}) = +2, \quad R(\text{Sideways}) = 0, \quad R(\text{Bear}) = -2.$$
> 
> À ce stade l'agent ne fait rien : il subit le régime. La récompense mesure simplement la qualité moyenne d'être dans chaque état.

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

> [!example] Régime de marché — actions et récompenses
> On donne à l'agent trois actions correspondant à la position prise sur l'indice :
> 
> $$A = \{\text{Long}, \text{Flat}, \text{Short}\}, \quad \text{codées } a \in \{+1, 0, -1\}.$$
> 
> La récompense devient le PnL réalisé sur une période, soit la position multipliée par le rendement du régime :
> 
> $$R(s, a) = a \cdot \mu(s), \qquad \mu(\text{Bull}) = +2, \; \mu(\text{Sideways}) = 0, \; \mu(\text{Bear}) = -2.$$
> 
> | $R(s, a)$       | Long ($+1$) | Flat ($0$) | Short ($-1$) |
> | --------------- | :---------: | :--------: | :----------: |
> | **Bull** ($+2$)    | $+2$        | $0$        | $-2$         |
> | **Sideways** ($0$) | $0$         | $0$        | $0$          |
> | **Bear** ($-2$)    | $-2$        | $0$        | $+2$         |
> 
> Intuitivement, la politique optimale est évidente : **Long en Bull, Short en Bear, indifférent en Sideways**. Une politique aléatoire ferait beaucoup moins bien. On vérifiera ce résultat formellement quand on disposera des outils (Policy Iteration, Value Iteration).
> 
> **Hypothèse simplificatrice.** Pour cet exemple, l'action *ne change pas* la dynamique des régimes : $P(s' \mid s, a) = P(s' \mid s)$. C'est réaliste — un trader ne déplace pas le régime de marché par sa position — et garde la matrice $\mathbf{P}$ inchangée par rapport au MRP. L'action n'agit que sur la récompense. Dans un cadre plus riche on pourrait imaginer une action "rebalancer" qui augmente la probabilité d'aller vers Sideways, mais on n'en a pas besoin ici.

**Politique (policy).** Une **politique** $\pi$ est une règle de choix d'action. On distingue :

- **Politique déterministe** : $\pi : S \to A$, qui à chaque état associe une action $\pi(s)$.
- **Politique stochastique** : $\pi(a \mid s)$, distribution de probabilité sur $A$ conditionnée à $s$.

La politique stochastique englobe la politique déterministe (cas où la distribution est concentrée sur une seule action). On verra que dans un MDP fini avec $\gamma < 1$, il existe toujours une politique optimale **déterministe** — c'est un résultat fort qui sera réutilisé en programmation dynamique.

> [!example] Représentation matricielle d'une politique
> Une politique stochastique sur notre exemple est un tableau $|S| \times |A|$ où chaque ligne est une distribution sur les actions (somme = 1). Par exemple, un agent prudent en marché baissier pourrait choisir :
> 
> | $\pi(a \mid s)$ | Long | Flat | Short |
> |---|:---:|:---:|:---:|
> | **Bull**     | 0.8 | 0.15 | 0.05 |
> | **Sideways** | 0.5 | 0.4 | 0.1 |
> | **Bear**     | 0.0 | 0.3 | 0.7 |
> 
> Lecture : « en Bull, je suis Long 80% du temps, Flat 15%, Short 5% ». Les chiffres ici sont arbitraires — c'est l'agent qui les choisit.
> 
> La politique déterministe $\pi^* = (\text{Long}, \text{Flat}, \text{Short})$ est le cas particulier où chaque ligne contient un seul $1$ :
> 
> | $\pi(a \mid s)$ | Long | Flat | Short |
> |---|:---:|:---:|:---:|
> | **Bull**     | **1** | 0 | 0 |
> | **Sideways** | 0 | **1** | 0 |
> | **Bear**     | 0 | 0 | **1** |
> 
> **Attention à ne pas confondre $\pi$ avec la matrice de transition $\mathbf{P}$ :** les deux sont des tableaux row-stochastic (lignes positives qui somment à 1), mais ils décrivent des choses totalement différentes. $\mathbf{P}$ est la dynamique du marché — l'agent la subit, il ne la contrôle pas. $\pi$ est la stratégie de l'agent — c'est ce qu'on optimise. Les colonnes ne sont même pas du même type : pour $\mathbf{P}$ ce sont des états (où l'on arrive), pour $\pi$ ce sont des actions (ce qu'on décide).

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

**Équation de Bellman.** Calculer $v_\pi$ par sa définition (somme infinie d'espérances sur tous les futurs possibles) est infaisable. Heureusement, $v_\pi$ vérifie une **équation récursive** qui transforme cette somme infinie en système linéaire fini :

> [!warning] Équation de Bellman pour $v_\pi$
> Pour toute politique $\pi$ et tout état $s$ :
> 
> $$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \big[ R(s, a) + \gamma \, v_\pi(s') \big].$$
> 
> En substance : *la valeur de l'état présent = la récompense que je vais toucher maintenant + la valeur actualisée de l'état où je vais arriver, le tout moyenné sur la stochasticité de $\pi$ et de $\mathbf{P}$*.

![[Pasted image 20260501145219.png]]
**Figure 5.** Lecture visuelle de l'équation de Bellman, partant de l'état Bull. Trois niveaux : (1) on se trouve dans un état $s$ ; (2) on choisit une action $a$ selon $\pi(a \mid s)$, ce qui déclenche la récompense immédiate $R(s, a)$ ; (3) on transite vers un état suivant $s'$ selon $P(s' \mid s, a)$, et la valeur future $v_\pi(s')$ est actualisée par $\gamma$. La somme sur les actions correspond au $\sum_a \pi(a \mid s)$, la somme sur les états suivants au $\sum_{s'} P(s' \mid s, a)$. L'action Flat est omise pour la lisibilité.

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

> [!example] Régime de marché — calcul des fonctions de valeur
> On fixe $\gamma = 0.9$ et on évalue deux politiques sur notre exemple, en utilisant la formule close $V^\pi = (I - \gamma \mathbf{P})^{-1} R^\pi$.
> 
> **Politique passive "toujours Long".** $R^\pi = (R(\text{Bull}, \text{Long}), R(\text{Sideways}, \text{Long}), R(\text{Bear}, \text{Long})) = (+2, 0, -2)$. On obtient :
> 
> $$V^{\text{Long}}(\text{Bull}) \approx 4.35, \quad V^{\text{Long}}(\text{Sideways}) = 0, \quad V^{\text{Long}}(\text{Bear}) \approx -4.35.$$
> 
> Lecture : être en Bull aujourd'hui en suivant cette politique vaut $4.35$ — beaucoup plus que la récompense immédiate de $+2$, parce que la persistance des régimes ($P(\text{Bull} \mid \text{Bull}) = 0.7$) fait qu'on touchera encore $+2$ pendant plusieurs pas en moyenne. Inversement, Bear est punitif sur le long terme. Sideways vaut exactement $0$ par symétrie de la matrice.
> 
> **Politique candidate $\pi^* = (\text{Long}, \text{Flat}, \text{Short})$.** Ici $R^{\pi^*} = (+2, 0, +2)$ (Short en Bear rapporte $-1 \cdot (-2) = +2$). On obtient :
> 
> $$V^{\pi^*}(\text{Bull}) \approx 15.61, \quad V^{\pi^*}(\text{Sideways}) \approx 13.17, \quad V^{\pi^*}(\text{Bear}) \approx 15.61.$$
> 
> Énorme différence : en s'adaptant au régime, on convertit chaque état (sauf Sideways) en gain. Sideways vaut un peu moins que Bull/Bear parce que c'est un régime moins persistant ($0.4$ contre $0.7$) — on y stagne moins longtemps avant de basculer vers un régime rentable.
> 
> **Conjecture.** $\pi^*$ est très probablement la politique optimale. On le démontrera formellement avec Policy Iteration et Value Iteration dans la prochaine section.

## II. Programmation dynamique

**Le contexte.** On dispose d'un MDP entièrement connu : on a la matrice de transition $\mathbf{P}$ et la fonction de récompense $R$. La question : **comment calculer $v_\pi$ (évaluation) et $\pi_*$ (optimisation) en pratique ?**

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

> [!note]- Pseudo-code (Iterative Policy Evaluation)
> ```
> Entrée : MDP (S, A, P, R, gamma), politique pi, tolérance theta > 0
> Sortie : V ≈ v_pi
> 
> Initialiser V(s) = 0 pour tout s in S
> Répéter :
>     Delta ← 0
>     Pour chaque s in S :
>         v_old ← V(s)
>         V(s) ← sum_a pi(a|s) sum_s' P(s'|s,a) [R(s,a) + gamma * V(s')]
>         Delta ← max(Delta, |v_old - V(s)|)
> Jusqu'à Delta < theta
> Retourner V
> ```

> [!example] Régime de marché — Iterative Policy Evaluation à la main
> On évalue la politique passive *toujours Long* avec $\gamma = 0.9$. La politique étant déterministe ($\pi(\text{Long} \mid s) = 1$), le $\sum_a$ s'effondre et la règle de mise à jour devient simplement :
> 
> $$V_{k+1}(s) = R(s, \text{Long}) + 0.9 \sum_{s'} P(s' \mid s) \, V_k(s').$$
> 
> Avec $R^\pi = (+2, 0, -2)$ et la matrice $\mathbf{P}$ du régime de marché, en partant de $V_0 = (0, 0, 0)$ :
> 
> | $k$ | $V_k(\text{Bull})$ | $V_k(\text{Sideways})$ | $V_k(\text{Bear})$ |
> |---|:---:|:---:|:---:|
> | 0 | 0.000 | 0.000 | 0.000 |
> | 1 | 2.000 | 0.000 | −2.000 |
> | 2 | 3.080 | 0.000 | −3.080 |
> | 3 | 3.663 | 0.000 | −3.663 |
> | 4 | 3.974 | 0.000 | −3.974 |
> | 5 | 4.143 | 0.000 | −4.143 |
> | $\infty$ | **4.348** | **0.000** | **−4.348** |
> 
> **Détail du calcul de $V_2(\text{Bull})$** (les autres lignes suivent le même schéma) :
> 
> $$V_2(\text{Bull}) = +2 + 0.9 \times \big[ 0.7 \cdot 2 + 0.2 \cdot 0 + 0.1 \cdot (-2) \big] = 2 + 0.9 \times 1.2 = 3.080.$$
> 
> Trois observations :
> - **Sideways reste à 0** à chaque itération. C'est dû à la symétrie : $R(\text{Sideways}, \text{Long}) = 0$ et la transition depuis Sideways est symétrique entre Bull (+) et Bear (−).
> - **Symétrie Bull/Bear** : $V_k(\text{Bull}) = -V_k(\text{Bear})$ à chaque étape. Même raison.
> - **Convergence géométrique** : l'écart à la valeur finale est divisé par environ $\gamma = 0.9$ à chaque itération. Après 50 itérations on est à $10^{-3}$ près ; après 100 à $10^{-5}$ près.
> 
> On retombe sur $V^\pi = (4.348, 0, -4.348)$ — c'est exactement la même valeur que celle calculée par la formule close $V = (I - \gamma \mathbf{P})^{-1} R$ en I.C, comme attendu. **Les deux approches résolvent la même équation, l'une exactement et l'autre par approximation itérative.**

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

> 💡 **Conclusion.** Tant que la politique gloutonne donne une amélioration stricte, on peut continuer à améliorer. Le seul moment où l'on ne peut plus améliorer, c'est quand on est *déjà* à l'optimum. C'est exactement le mécanisme de bouclage qui justifiera **Policy Iteration** dans la sous-section suivante : alterner évaluation et amélioration jusqu'à ce que ça ne bouge plus, et on a $\pi_*$.

> [!example] Régime de marché — une étape d'amélioration
> On part de la politique passive $\pi=$ *toujours Long*, qu'on a évaluée en II.A : $V^\pi = (4.348,\, 0,\, -4.348)$.
> 
> **Étape 1 : calculer $q_\pi(s, a)$ pour tous les couples.** On utilise
> 
> $$q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s) \, V^\pi(s').$$
> 
> Comme l'action ne change pas la dynamique ($P(s' \mid s, a) = P(s' \mid s)$), le terme $\gamma \sum_{s'} P(s' \mid s) V^\pi(s')$ ne dépend que de $s$. On le calcule une fois par état :
> 
> | $s$ | $\sum_{s'} P(s' \mid s) V^\pi(s')$ | $\gamma \times \text{(...)}$ |
> | --- | :---: | :---: |
> | Bull     | $0.7 \cdot 4.348 + 0.2 \cdot 0 + 0.1 \cdot (-4.348) = 2.609$ | $+2.348$ |
> | Sideways | $0.3 \cdot 4.348 + 0.4 \cdot 0 + 0.3 \cdot (-4.348) = 0$     | $\phantom{+}0.000$ |
> | Bear     | $0.1 \cdot 4.348 + 0.2 \cdot 0 + 0.7 \cdot (-4.348) = -2.609$ | $-2.348$ |
> 
> En ajoutant $R(s, a)$, on obtient le tableau des Q-valeurs :
> 
> | $q_\pi(s, a)$ | Long ($a=+1$) | Flat ($a=0$) | Short ($a=-1$) |
> |---|:---:|:---:|:---:|
> | **Bull**     | $+2 + 2.348 = \mathbf{+4.348}$ | $\phantom{+}0 + 2.348 = +2.348$ | $-2 + 2.348 = +0.348$ |
> | **Sideways** | $\phantom{+}0 + 0 = \mathbf{0}$ | $\phantom{+}0 + 0 = \mathbf{0}$ | $\phantom{+}0 + 0 = \mathbf{0}$ |
> | **Bear**     | $-2 - 2.348 = -4.348$ | $\phantom{+}0 - 2.348 = -2.348$ | $+2 - 2.348 = \mathbf{-0.348}$ |
> 
> (en gras, l'argmax de chaque ligne)
> 
> **Étape 2 : politique gloutonne.** $\pi'(s) = \arg\max_a q_\pi(s, a)$ donne :
> 
> - **Bull** : Long (inchangé).
> - **Sideways** : tie à 3 — n'importe quelle action convient. On choisit Flat par convention.
> - **Bear** : Short. **Changement** par rapport à $\pi$ qui jouait Long.
> 
> Donc $\pi' = (\text{Long}, \text{Flat}, \text{Short})$ !! La politique stochastique est devenu déterministe !!
> 
> **Le résultat fascinant.** $\pi'$ est exactement la politique candidate $\pi^*$ que nous avions conjecturée en I.C ! On a donc *récupéré la politique optimale en une seule étape* de Policy Improvement, en partant d'une politique passive. Et la valeur a fait un bond énorme :
> 
> $$V^{\pi} \approx (4.35,\, 0,\, -4.35) \quad \xrightarrow{\text{1 étape}} \quad V^{\pi'} \approx (15.61,\, 13.17,\, 15.61).$$
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

> [!warning] Algorithme Policy Iteration
> 1. Initialiser une politique $\pi_0$ arbitraire.
> 2. **Boucle** : pour $k = 0, 1, 2, \ldots$ :
>    - **Évaluation** : calculer $V^{\pi_k}$ par Iterative Policy Evaluation (II.A).
>    - **Amélioration** : poser $\pi_{k+1}(s) = \arg\max_a \big[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^{\pi_k}(s') \big]$.
>    - Si $\pi_{k+1} = \pi_k$, sortir de la boucle.
> 3. Retourner $(\pi_*, V^{\pi_*}) = (\pi_k, V^{\pi_k})$.

> [!note]- Pseudo-code détaillé
> ```
> Entrée : MDP (S, A, P, R, gamma), tolérance theta > 0
> Sortie : politique optimale pi, valeur optimale V
> 
> Initialiser pi(s) arbitrairement pour tout s in S
> Répéter :
>     # Étape A : Policy Evaluation
>     V ← policy_evaluation(pi, theta)
>     
>     # Étape B : Policy Improvement
>     stable ← True
>     Pour chaque s in S :
>         a_old ← pi(s)
>         pi(s) ← argmax_a sum_s' P(s'|s,a) [R(s,a) + gamma * V(s')]
>         Si a_old ≠ pi(s) :
>             stable ← False
> Jusqu'à stable
> Retourner (pi, V)
> ```

> [!example] Régime de marché — Policy Iteration boucle en 2 tours
> On part de la politique passive $\pi_0=$ *toujours Long*.
> 
> **Tour 1.**
> - **A** (déjà fait en II.A) : $V^{\pi_0} = (4.348,\, 0,\, -4.348)$.
> - **B** (déjà fait en II.B) : $\pi_1 = (\text{Long}, \text{Flat}, \text{Short})$. Politique modifiée → on continue.
> 
> **Tour 2.**
> - **A** : $V^{\pi_1} = (15.61,\, 13.17,\, 15.61)$ (formule close).
> - **B** : on calcule $q_{\pi_1}(s, a)$ pour tous les couples. Pour chaque $s$, l'argmax doit retomber sur $\pi_1(s)$ — sinon on aurait une amélioration. Vérifions sur Bull :
>   
>   $$q_{\pi_1}(\text{Bull}, a) = R(\text{Bull}, a) + 0.9 \cdot (0.7 \cdot 15.61 + 0.2 \cdot 13.17 + 0.1 \cdot 15.61) = R(\text{Bull}, a) + 13.61.$$
>   
>   Donc $q_{\pi_1}(\text{Bull}, \cdot) = (15.61, 13.61, 11.61)$ pour (Long, Flat, Short). Argmax : **Long** ✓.
>   
>   Calcul similaire pour Sideways et Bear : argmax = (Long, Flat, Short) = $\pi_1$. **Politique inchangée → on s'arrête**.
> 
> **Conclusion.** $\pi_* = (\text{Long}, \text{Flat}, \text{Short})$ est optimale, avec $V_* = (15.61,\, 13.17,\, 15.61)$. Ce qui confirme la conjecture posée en I.C, cette fois avec une preuve algorithmique.

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

> [!note]- Pseudo-code (Value Iteration)
> ```
> Entrée : MDP (S, A, P, R, gamma), tolérance theta > 0
> Sortie : politique optimale pi, valeur optimale V
> 
> Initialiser V(s) = 0 pour tout s in S
> Répéter :
>     Delta ← 0
>     Pour chaque s in S :
>         v_old ← V(s)
>         V(s) ← max_a sum_s' P(s'|s,a) [R(s,a) + gamma * V(s')]
>         Delta ← max(Delta, |v_old - V(s)|)
> Jusqu'à Delta < theta
> 
> # Extraction de la politique optimale
> Pour chaque s in S :
>     pi(s) ← argmax_a sum_s' P(s'|s,a) [R(s,a) + gamma * V(s')]
> Retourner (pi, V)
> ```

> [!example] Régime de marché — Value Iteration à la main
> On lance VI à partir de $V_0 = (0, 0, 0)$ avec $\gamma = 0.9$. À chaque itération, pour chaque $s$ :
> 
> $$V_{k+1}(s) = \max_a \big[ a \cdot \mu(s) + 0.9 \cdot E_k(s) \big], \quad \text{où } E_k(s) = \sum_{s'} P(s' \mid s) V_k(s').$$
> 
> Comme l'action ne modifie pas la dynamique, le $\max_a$ se réduit à $|\mu(s)|$ : on prend Long si $\mu(s) > 0$, Short si $\mu(s) < 0$, n'importe quoi si $\mu(s) = 0$. On a donc $\max_a a \cdot \mu(s) = (+2, 0, +2)$ pour (Bull, Sideways, Bear).
> 
> | $k$ | $V_k(\text{Bull})$ | $V_k(\text{Sideways})$ | $V_k(\text{Bear})$ |
> |---|:---:|:---:|:---:|
> | 0 | 0.000 | 0.000 | 0.000 |
> | 1 | 2.000 | 0.000 | 2.000 |
> | 2 | 3.440 | 1.080 | 3.440 |
> | 3 | 4.671 | 2.246 | 4.671 |
> | 4 | 5.768 | 3.330 | 5.768 |
> | 5 | 6.753 | 4.313 | 6.753 |
> | $\infty$ | **15.610** | **13.170** | **15.610** |
> 
> **Détail du calcul de $V_2(\text{Bull})$** :
> 
> $$E_1(\text{Bull}) = 0.7 \cdot 2 + 0.2 \cdot 0 + 0.1 \cdot 2 = 1.6, \qquad V_2(\text{Bull}) = 2 + 0.9 \cdot 1.6 = 3.440.$$
> 
> **Trois observations :**
> - **Symétrie Bull/Bear** : ici $V_k(\text{Bull}) = V_k(\text{Bear})$ (et plus l'opposé comme en II.A). C'est parce que VI prend la *meilleure* action partout : Long en Bull, Short en Bear, qui rapportent toutes deux $+2$. Bear devient un état rentable.
> - **Convergence plus lente** que Policy Evaluation pour la politique passive : on part de $0$ et il faut atteindre $\sim 15$ au lieu de $\sim 4$. La progression géométrique en $\gamma^k = 0.9^k$ reste la même, mais l'écart total est plus grand.
> - **Politique optimale extraite à la fin** : $\pi_*(s) = \arg\max_a (a \cdot \mu(s)) = (\text{Long}, \text{tie}, \text{Short})$. En cassant le tie en faveur de Flat, on retombe sur $\pi_* = (\text{Long}, \text{Flat}, \text{Short})$. Le même résultat que Policy Iteration.

> 💡 **Bilan DP.** On a vu trois algorithmes (Policy Evaluation, Policy Iteration, Value Iteration) qui sont en réalité **trois lectures de la même équation de Bellman**. Évaluer = itérer Bellman avec une politique fixée. Optimiser = itérer Bellman avec un $\max$. Tous convergent en $\gamma^k$ grâce à la propriété de contraction de l'opérateur de Bellman (résultat admis ici). Reste un seul problème : tout ça suppose qu'on **connaît le modèle** $\mathbf{P}$ et $R$. C'est ce qu'on lâche maintenant en passant à Monte Carlo et TD Learning.

## III. Monte Carlo

**Le contexte.** Toute la section II reposait sur une hypothèse forte : on connaît la matrice de transition $\mathbf{P}$ et la fonction de récompense $R$. C'est rarement le cas en pratique. Quand un agent joue à un jeu vidéo, conduit une voiture ou trade un portefeuille, il **n'a pas** de modèle explicite de l'environnement — il observe seulement des séquences d'états, d'actions et de récompenses.

Les méthodes **Monte Carlo** (MC) abandonnent l'hypothèse de modèle connu. Elles n'ont besoin que d'**expérience** : des trajectoires complètes échantillonnées dans l'environnement.

**L'idée.** La fonction de valeur $v_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$ est une **espérance**. Si on ne peut plus la calculer exactement (DP), on peut l'**estimer par moyenne empirique** : on génère plein d'épisodes en suivant $\pi$, on récupère les retours observés depuis chaque visite à $s$, et on en prend la moyenne. Loi des grands nombres → ça converge vers $v_\pi(s)$.

> 💡 **MC est tout entier du côté "échantillonner".** C'est l'autre versant de la dichotomie qu'on avait posée en I.B (rappelée en intro de II) : DP propage la distribution exacte (modèle connu), MC moyenne sur des trajectoires concrètes (modèle inconnu). Les deux cherchent à résoudre Bellman, mais avec deux outils statistiques différents.

**Trois propriétés à retenir avant d'attaquer les algos.**

- **Tâches épisodiques uniquement.** Le retour $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ doit être un nombre bien défini, donc fini. MC suppose donc que les épisodes terminent toujours (état terminal, ou troncature après $T$ pas).
- **Pas de bootstrap.** *Bootstrap* = utiliser une estimation de $V(s')$ pour mettre à jour $V(s)$. C'est ce que fait DP (et plus tard TD). MC, lui, calcule chaque retour **à partir des récompenses réelles** observées jusqu'à la fin de l'épisode — sans jamais utiliser d'autres $V$. Conséquence : les estimations de chaque état sont **indépendantes**. C'est plus simple à analyser, mais on ne profite pas de la structure récursive de Bellman.
- **Mise à jour en fin d'épisode.** Pour calculer $G_t$, il faut connaître toutes les récompenses jusqu'à la fin. Donc MC ne peut pas mettre à jour à chaque pas — il attend la fin de chaque épisode. Pas d'apprentissage en ligne, contrairement à TD (IV).

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

> [!example] Régime de marché — First-Visit MC à la main
> 
> **Setup.** On évalue la politique passive *toujours Long* avec $\gamma = 0.9$. Notre régime de marché n'est *a priori* pas épisodique, donc on triche pédagogiquement : on tronque chaque épisode à $T = 4$ transitions (4 récompenses). Plus l'épisode est court, plus l'estimateur est biaisé vers le bas (on coupe le futur lointain), mais la mécanique reste identique.
> 
> **Trois épisodes simulés**, chacun démarré en Bull, où les transitions ont été tirées (à la main, de manière représentative) selon la matrice $\mathbf{P}$ :
> 
> | Épisode | Trajectoire d'états (5 états, 4 transitions) | Récompenses observées $r_0, r_1, r_2, r_3$ |
> |:---:|:---|:---|
> | 1 | Bull → Bull → Bull → Sideways → Bull | $+2,\, +2,\, +2,\, 0$ |
> | 2 | Bull → Sideways → Bear → Bear → Bear | $+2,\, 0,\, -2,\, -2$ |
> | 3 | Bull → Bull → Sideways → Sideways → Bull | $+2,\, +2,\, 0,\, 0$ |
> 
> *(Note : la récompense $r_t$ est observée en quittant l'état $s_t$ ; elle vaut $R(s_t, \text{Long}) = \mu(s_t)$.)*
> 
> **Calcul des retours pour l'épisode 1** (les autres se font pareil) :
> 
> $$\begin{aligned}
> G_0 &= 2 + 0.9 \cdot 2 + 0.81 \cdot 2 + 0.729 \cdot 0 = 5.42 \quad (\text{état Bull}) \\
> G_1 &= 2 + 0.9 \cdot 2 + 0.81 \cdot 0 = 3.80 \quad (\text{état Bull}) \\
> G_2 &= 2 + 0.9 \cdot 0 = 2.00 \quad (\text{état Bull}) \\
> G_3 &= 0 \quad (\text{état Sideways})
> \end{aligned}$$
> 
> **First-Visit MC pour Bull** : on collecte le retour de la **première** visite à Bull dans chaque épisode.
> 
> | Épisode | Première visite à Bull à $t = ?$ | Retour collecté |
> |:---:|:---:|:---:|
> | 1 | 0 | $5.42$ |
> | 2 | 0 | $-1.078$ |
> | 3 | 0 | $3.80$ |
> 
> $$\hat V_{\text{FV}}(\text{Bull}) \;=\; \frac{5.42 + (-1.078) + 3.80}{3} \;=\; \frac{8.142}{3} \;\approx\; 2.71.$$
> 
> La vraie valeur (calculée en I.C par formule close) est $V^\pi(\text{Bull}) \approx 4.35$. L'estimation MC est **biaisée vers le bas** (à cause de la troncature à $T=4$ qui ignore les récompenses lointaines) **et bruitée** (3 épisodes c'est très peu). Avec des centaines d'épisodes longs, on convergerait vers $4.35$.
> 
> **Every-Visit MC pour Bull** : on collecte le retour à **chaque** visite de Bull.
> 
> | Épisode | Visites à Bull | Retours |
> |:---:|:---:|:---:|
> | 1 | $t = 0, 1, 2$ | $5.42,\, 3.80,\, 2.00$ |
> | 2 | $t = 0$ | $-1.078$ |
> | 3 | $t = 0, 1$ | $3.80,\, 2.00$ |
> 
> Total : 6 retours, somme $= 15.94$, moyenne $= 2.66$. Très proche de l'estimation First-Visit (la différence n'est significative qu'avec beaucoup d'épisodes).
> 
> **Mise à jour incrémentale** (sur First-Visit pour Bull) :
> 
> | Épisode | Retour $G$ | $N(\text{Bull})$ | Mise à jour : $V \leftarrow V + \frac{1}{N}(G - V)$ | $V$ après |
> |:---:|:---:|:---:|:---|:---:|
> | 1 | $5.42$ | $1$ | $0 + (5.42 - 0)/1$ | $5.420$ |
> | 2 | $-1.078$ | $2$ | $5.42 + (-1.078 - 5.42)/2$ | $2.171$ |
> | 3 | $3.80$ | $3$ | $2.171 + (3.80 - 2.171)/3$ | $2.714$ |
> 
> On retombe bien sur $\hat V_{\text{FV}}(\text{Bull}) = 2.714$ — la mise à jour incrémentale calcule la même moyenne, sans stocker les retours.
> 
> **Trois choses importantes à retenir de cet exemple :**
> 
> - **Convergence lente.** 3 épisodes courts donnent une estimation très imprécise. MC est notoirement gourmand en données — c'est un défaut inhérent à toute méthode Monte Carlo. C'est aussi ce qui motive TD (IV), qui apprend plus vite grâce au bootstrap.
> - **MC fonctionne sans connaître $\mathbf{P}$.** À aucun moment on n'a utilisé la matrice de transition pour calculer les retours. On a juste observé des trajectoires et moyenné. C'est *toute* la puissance de la méthode.
> - **Pas de bootstrap.** Chaque retour $G_t$ est calculé à partir de récompenses **réelles** observées, pas d'autres estimations $V(s')$. Cela contraste avec DP (qui utilise $V(s')$ dans la mise à jour) et avec TD (qui combinera les deux).

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

> [!note]- Pseudo-code MC Control
> ```
> Entrée : MDP (sans modèle), gamma, schedule epsilon_k
> Sortie : politique optimale pi (et Q-fonction associée)
> 
> Initialiser Q(s, a) = 0, N(s, a) = 0 pour tout (s, a)
> 
> Pour chaque épisode k = 1, 2, 3, ... :
>     epsilon ← epsilon_k        # par exemple 1/k pour satisfaire GLIE
>     pi ← politique epsilon-greedy par rapport à Q
>     
>     Générer un épisode (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T) suivant pi
>     
>     Pour chaque pas t = 0, 1, ..., T-1 :
>         G ← r_t + gamma * r_{t+1} + ... + gamma^(T-t-1) * r_{T-1}
>         N(s_t, a_t) ← N(s_t, a_t) + 1
>         Q(s_t, a_t) ← Q(s_t, a_t) + (G - Q(s_t, a_t)) / N(s_t, a_t)
> 
> Retourner pi(s) = argmax_a Q(s, a)
> ```

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

> [!example] Régime de marché — discussion qualitative de MC Control
> 
> Faire tourner MC Control numériquement à la main demanderait des dizaines voire des centaines d'épisodes (le temps que $Q$ se stabilise et que $\varepsilon$ décroisse), ce n'est pas pertinent ici. À la place, on raisonne **qualitativement** sur ce que l'algorithme fait sur notre exemple.
> 
> **Conditions de départ.** Politique initiale ε-greedy avec $Q(s, a) = 0$ partout : équivalent à uniformément aléatoire. L'agent va donc faire toutes les actions à peu près également au début, observer plein de retours, et raffiner $Q$.
> 
> **Ce que l'algorithme va découvrir.** Asymptotiquement, $Q$ converge vers $q_*$. Or on a déjà calculé $q_*$ implicitement en II.D : avec $V_* = (15.61, 13.17, 15.61)$, on a
> 
> $$q_*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s) V_*(s').$$
> 
> Le terme d'espérance $\gamma \sum_{s'} P(s' \mid s) V_*(s')$ ne dépend que de $s$ ; on le calcule pour les trois états (rappel : la dynamique ne dépend pas de l'action) :
> 
> | $s$ | $\gamma \cdot \mathbb{E}[V_* \mid s]$ |
> | :---: | :---: |
> | Bull | $0.9 \cdot (0.7 \cdot 15.61 + 0.2 \cdot 13.17 + 0.1 \cdot 15.61) = 13.61$ |
> | Sideways | $0.9 \cdot (0.3 \cdot 15.61 + 0.4 \cdot 13.17 + 0.3 \cdot 15.61) = 13.17$ |
> | Bear | $0.9 \cdot (0.1 \cdot 15.61 + 0.2 \cdot 13.17 + 0.7 \cdot 15.61) = 13.61$ |
> 
> En ajoutant $R(s, a) = a \cdot \mu(s)$, on obtient le tableau cible que MC Control va apprendre :
> 
> | $q_*(s, a)$ | Long | Flat | Short |
> |---|:---:|:---:|:---:|
> | **Bull** | $\mathbf{15.61}$ | $13.61$ | $11.61$ |
> | **Sideways** | $13.17$ | $\mathbf{13.17}$ | $13.17$ |
> | **Bear** | $11.61$ | $13.61$ | $\mathbf{15.61}$ |
> 
> Argmax par ligne (en gras) : $\pi_* = (\text{Long}, \text{Flat ou autre}, \text{Short})$ — exactement la politique optimale qu'on avait identifiée en II.
> 
> **Le rôle crucial de l'exploration.** Sans ε-greedy, l'agent partant en Bear avec $Q = 0$ partout pourrait choisir Long arbitrairement (cas de tie au tout début), recevoir $-2$, et estimer $Q(\text{Bear}, \text{Long}) = -2 < 0 = Q(\text{Bear}, \text{Short})$. Il prendrait alors Short. Mais imaginons à l'inverse qu'il parte avec Short en Bear et reçoive un retour bruité légèrement négatif (parce que la trajectoire est passée par Sideways et Bull avec des récompenses faibles) : il pourrait à tort estimer $Q(\text{Bear}, \text{Short}) < Q(\text{Bear}, \text{Long}) = 0$ et se mettre à toujours jouer Long en Bear — c'est le piège Door A/Door B. **L'exploration ε-greedy garantit qu'il continuera à essayer Short, et que sur le long terme l'estimation se stabilisera autour de la vraie valeur $+15.61$.**
> 
> **Trois leçons à retenir.**
> 
> - **Sans modèle, on apprend $Q$.** Argument structurant : tous les algos model-free de la suite (SARSA, Q-learning, DQN) reposent sur cette même idée.
> - **L'exploration n'est pas optionnelle.** ε-greedy + GLIE assurent que toutes les paires $(s, a)$ continuent à être visitées, donc que toutes les estimations $Q(s, a)$ peuvent être corrigées.
> - **Convergence asymptotique mais lente.** En pratique, MC Control demande beaucoup d'épisodes, surtout quand $\varepsilon$ décroît lentement. C'est l'inconvénient classique de Monte Carlo, qu'on adressera avec TD en IV — qui apprend en ligne, sans attendre la fin de chaque épisode.

### C. Off-policy Prediction

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

> [!example] Régime de marché — TD(0) à la main
> 
> **Setup.** On évalue la politique passive *toujours Long* avec $\gamma = 0.9$ et $\alpha = 0.1$. Initialisation $V_0 = (0, 0, 0)$ pour (Bull, Sideways, Bear). Cible : $V^\pi = (4.348,\, 0,\, -4.348)$ (calculée en I.C).
> 
> **L'avantage de TD ici.** Notre régime de marché n'est *pas* épisodique — c'est un flux infini de transitions. En III.A on avait dû tronquer artificiellement à $T = 4$ pour faire tourner MC. **TD n'a pas besoin de ça** : une seule longue trajectoire suffit. C'est précisément l'argument pratique de TD.
> 
> **Trajectoire de 8 transitions** (tirée à la main, représentative selon $\mathbf{P}$) :
> 
> $\text{Bull} \to \text{Bull} \to \text{Sideways} \to \text{Bear} \to \text{Bear} \to \text{Sideways} \to \text{Bull} \to \text{Bull} \to \text{Sideways}$
> 
> Récompenses (l'agent est Long, donc $r_t = \mu(s_t)$) :
> 
> $r_0 = +2,\ r_1 = +2,\ r_2 = 0,\ r_3 = -2,\ r_4 = -2,\ r_5 = 0,\ r_6 = +2,\ r_7 = +2.$
> 
> **Mise à jour pas à pas** : $V(s_t) \leftarrow V(s_t) + 0.1 \cdot [r_t + 0.9 \cdot V(s_{t+1}) - V(s_t)]$.
> 
> | Pas | $s_t \to s_{t+1}$ | $r_t$ | TD target $r_t + \gamma V(s_{t+1})$ | TD error $\delta_t$ | $V$ après mise à jour |
> |:---:|:---:|:---:|:---:|:---:|:---|
> | 0 | Bull → Bull | $+2$ | $2 + 0.9 \cdot 0 = 2.000$ | $+2.000$ | $V(\text{Bull}) = 0.200$ |
> | 1 | Bull → Sideways | $+2$ | $2 + 0.9 \cdot 0 = 2.000$ | $+1.800$ | $V(\text{Bull}) = 0.380$ |
> | 2 | Sideways → Bear | $0$ | $0 + 0.9 \cdot 0 = 0.000$ | $0.000$ | $V(\text{Sideways}) = 0.000$ |
> | 3 | Bear → Bear | $-2$ | $-2 + 0.9 \cdot 0 = -2.000$ | $-2.000$ | $V(\text{Bear}) = -0.200$ |
> | 4 | Bear → Sideways | $-2$ | $-2 + 0.9 \cdot 0 = -2.000$ | $-1.800$ | $V(\text{Bear}) = -0.380$ |
> | 5 | Sideways → Bull | $0$ | $0 + 0.9 \cdot 0.380 = 0.342$ | $+0.342$ | $V(\text{Sideways}) = 0.034$ |
> | 6 | Bull → Bull | $+2$ | $2 + 0.9 \cdot 0.380 = 2.342$ | $+1.962$ | $V(\text{Bull}) = 0.576$ |
> | 7 | Bull → Sideways | $+2$ | $2 + 0.9 \cdot 0.034 = 2.031$ | $+1.455$ | $V(\text{Bull}) = 0.722$ |
> 
> **État final** : $V \approx (0.722,\, 0.034,\, -0.380)$. Bull est devenu positif, Bear négatif, Sideways quasi nul — tout bouge dans la bonne direction vers $(4.348, 0, -4.348)$.
> 
> **Le bootstrap en action — pas 5.** À ce pas, la transition est Sideways → Bull. La TD target vaut $0 + 0.9 \cdot V(\text{Bull}) = 0.342$, et $V(\text{Sideways})$ se met à jour vers une valeur **positive** alors qu'on n'a observé qu'une récompense nulle ($r_5 = 0$). D'où vient cette positivité ? **De $V(\text{Bull})$ qui était déjà positif.** L'information "Bull rapporte" qu'on avait apprise aux pas 0-1 s'est propagée vers Sideways via la transition. C'est exactement ça, le bootstrap : l'estimation d'un état nourrit l'estimation des autres.
> 
> **Ce que MC ne peut pas faire.** Avec MC, $V(\text{Sideways})$ resterait à 0 jusqu'à ce qu'on observe un retour total non-nul depuis Sideways — ce qui demande d'attendre une trajectoire complète. TD propage l'information beaucoup plus vite parce qu'il bootstrap.
> 
> **Trois choses à retenir.**
> 
> - **Pas d'épisode requis.** Une seule longue trajectoire suffit. C'est *plus* naturel ici que MC.
> - **Apprentissage en ligne.** À chaque transition, on met à jour. Pas besoin d'attendre.
> - **Bootstrap = propagation rapide.** L'information se propage entre états voisins via la TD target, sans attendre des récompenses réelles partout.

### B. TD Control

**L'idée.** En IV.A on faisait de la prédiction TD : politique fixée, on estime $v_\pi$. Maintenant on veut le **contrôle** : trouver $\pi_*$. Comme en III.B (MC Control), on suit le squelette **GPI** — alterner évaluation et amélioration — mais en utilisant TD pour l'évaluation au lieu de MC.

Les deux problèmes identifiés en III.B se reposent à l'identique :

- **On a besoin de $Q$, pas de $V$.** Sans modèle, $V$ ne suffit pas à choisir l'action. On apprend donc directement la Q-fonction.
- **Il faut explorer.** On utilise une politique $\varepsilon$-greedy par rapport à $Q$, avec décroissance GLIE de $\varepsilon$.

Ce qui change par rapport à MC Control : la mise à jour de $Q$ se fait à chaque pas (bootstrap) au lieu d'à la fin de l'épisode. Et il y a deux façons naturelles de bootstrap-er sur $Q$ — d'où deux algorithmes : **SARSA** et **Q-learning**.

#### SARSA(0)

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

> [!note]- Pseudo-code SARSA(0)
> ```
> Entrée : MDP (sans modèle), gamma, alpha, schedule epsilon_k
> Sortie : politique pi (et Q-fonction associée)
> 
> Initialiser Q(s, a) = 0 pour tout (s, a)
> Pour chaque épisode k = 1, 2, 3, ... :
>     epsilon ← epsilon_k
>     Observer s_0
>     Choisir a_0 ~ politique epsilon-greedy par rapport à Q
>     Boucle (sur les pas t = 0, 1, 2, ...) :
>         Exécuter a_t, observer r_t et s_{t+1}
>         Choisir a_{t+1} ~ politique epsilon-greedy par rapport à Q
>         Q(s_t, a_t) ← Q(s_t, a_t) + alpha * [r_t + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]
>         s_t ← s_{t+1}, a_t ← a_{t+1}
>         Si s_t terminal : sortir
> Retourner pi(s) = argmax_a Q(s, a)
> ```

#### SARSAMAX (ou Q-Learning)

**L'astuce.** SARSA utilise $Q(s_{t+1}, a_{t+1})$ — l'action que l'agent va réellement prendre. Q-learning utilise $\max_{a'} Q(s_{t+1}, a')$ — la **meilleure** action possible en $s_{t+1}$, indépendamment de ce que l'agent va vraiment faire.

Pourquoi ce choix ? Parce qu'on revient à l'équation de Bellman *d'optimalité* (vue en I.C) :

$$q_*(s, a) = \mathbb{E}\Big[ r_t + \gamma \max_{a'} q_*(s_{t+1}, a') \mid s_t = s,\, a_t = a \Big].$$

Le $\max$ remplace l'espérance sur $\pi$. Q-learning échantillonne directement cette équation, sans même attendre que l'agent prenne $a_{t+1}$.

> [!warning] Mise à jour Q-learning (SARSAMAX)
> $Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) + \alpha\Big[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big]$
> 
> On utilise le maximum sur les actions à l'état suivant, sans tenir compte de l'action que l'agent va effectivement choisir.

> 💡 **Q-learning est off-policy.** *Off-policy* signifie que la politique évaluée (la politique gloutonne, via le $\max$) est *différente* de la politique suivie par l'agent (l'$\varepsilon$-greedy, nécessaire pour explorer). Q-learning estime directement $q_*$ — la Q-fonction *optimale* — même si l'agent agit de manière sous-optimale pour explorer. C'est ce qui en fait le premier algorithme RL réellement "en boucle fermée" sur l'optimalité.

> [!note]- Pseudo-code Q-learning
> ```
> Entrée : MDP (sans modèle), gamma, alpha, schedule epsilon_k
> Sortie : politique pi (et Q-fonction associée)
> 
> Initialiser Q(s, a) = 0 pour tout (s, a)
> Pour chaque épisode k = 1, 2, 3, ... :
>     epsilon ← epsilon_k
>     Observer s_0
>     Boucle (sur les pas t = 0, 1, 2, ...) :
>         Choisir a_t ~ politique epsilon-greedy par rapport à Q
>         Exécuter a_t, observer r_t et s_{t+1}
>         Q(s_t, a_t) ← Q(s_t, a_t) + alpha * [r_t + gamma * max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]
>         s_t ← s_{t+1}
>         Si s_t terminal : sortir
> Retourner pi(s) = argmax_a Q(s, a)
> ```
> 
> Note : par rapport à SARSA, on n'a plus besoin de tirer $a_{t+1}$ avant la mise à jour. La cible bootstrap $\max_{a'} Q(s_{t+1}, a')$ est calculée directement à partir de $Q$.

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

> [!example] Régime de marché — SARSA vs Q-learning, un pas de mise à jour
> 
> **Setup.** Politique $\varepsilon$-greedy avec $\varepsilon = 0.2$, $\gamma = 0.9$, $\alpha = 0.1$. On suppose que l'agent a déjà été entraîné et que sa Q-fonction courante en Bear vaut
> 
> $Q(\text{Bear}, \text{Long}) = 11,\quad Q(\text{Bear}, \text{Flat}) = 13,\quad Q(\text{Bear}, \text{Short}) = 15.$
> 
> L'action gloutonne en Bear est donc Short.
> 
> **Transition observée** : l'agent est en $s_t = \text{Sideways}$, prend $a_t = \text{Long}$ (au hasard via exploration), reçoit $r_t = 0$, et arrive en $s_{t+1} = \text{Bear}$. Initialement $Q(\text{Sideways}, \text{Long}) = 12$ (par hypothèse).
> 
> **Cas 1 : SARSA.** L'agent tire $a_{t+1}$ selon sa politique $\varepsilon$-greedy en Bear. Avec $\varepsilon = 0.2$ et 3 actions, la probabilité de choisir l'action gloutonne (Short) est $0.8 + 0.2/3 \approx 0.867$. Imaginons qu'**il explore** et tire $a_{t+1} = \text{Long}$ (probabilité $0.2/3 \approx 0.067$).
> 
> $Q(\text{Sideways}, \text{Long}) \leftarrow 12 + 0.1 \cdot [0 + 0.9 \cdot Q(\text{Bear}, \text{Long}) - 12]$
> $= 12 + 0.1 \cdot [0 + 0.9 \cdot 11 - 12] = 12 + 0.1 \cdot (-2.1) = 11.79.$
> 
> SARSA "voit" que l'exploration peut amener à prendre Long en Bear (mauvais), et réduit l'estimation en conséquence.
> 
> **Cas 2 : Q-learning.** Pas besoin de tirer $a_{t+1}$. On utilise directement $\max_{a'} Q(\text{Bear}, a') = 15$ (Short).
> 
> $Q(\text{Sideways}, \text{Long}) \leftarrow 12 + 0.1 \cdot [0 + 0.9 \cdot 15 - 12] = 12 + 0.1 \cdot 1.5 = 12.15.$
> 
> Q-learning estime que depuis Bear on prendra la meilleure action (Short), peu importe ce que l'agent fait réellement. La mise à jour est plus optimiste.
> 
> **Écart entre les deux** : $11.79$ vs $12.15$, une différence de $0.36$ sur cette seule transition. Sur des milliers de pas, cet écart systématique change la politique apprise : Q-learning apprend $q_*$ pur, SARSA apprend une version "dégradée" qui intègre le bruit d'exploration.
> 
> **Trois choses à retenir.**
> 
> - **La différence tient à un mot** : $a_{t+1}$ tiré (SARSA) vs $\max_{a'}$ (Q-learning).
> - **On-policy vs off-policy** : SARSA évalue la politique réellement suivie, Q-learning évalue la politique optimale.
> - **Convergence** : les deux convergent vers $\pi_*$ si $\varepsilon \to 0$ (GLIE), mais leur comportement *en cours d'apprentissage* diffère — Q-learning plus agressif, SARSA plus conservateur.

> 💡 **Bilan TD Control.** SARSA et Q-learning sont les deux algorithmes fondamentaux du contrôle TD. Q-learning est devenu dominant dans la littérature moderne parce qu'il apprend $q_*$ directement — c'est l'ancêtre de **DQN** (Deep Q-Network), qui remplace la table $Q$ par un réseau de neurones. SARSA reste pertinent quand le coût de l'exploration est réel (robotique, systèmes physiques) : on préfère une politique qui tient compte du fait qu'on explore.

## V. POMDP

*À venir.*
