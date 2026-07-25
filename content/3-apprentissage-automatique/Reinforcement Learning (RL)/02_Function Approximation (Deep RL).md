---
title: Function Approximation (Deep RL)
---
# Function Approximation (Deep RL)

> Quand l'espace d'états devient trop grand pour être stocké en table de lookup (jeux Atari avec image en pixels, robotique avec capteurs continus), on remplace les fonctions $V(s)$ et $Q(s, a)$ par des **approximateurs paramétriques** — typiquement des réseaux de neurones. C'est le passage du **RL tabulaire** au **Deep RL**.

> Pré-requis : [[01_RL Tabulaire]] (MDP, Bellman, MC, TD(0), Q-learning, SARSA).

> 💡 **Trois familles d'approches** suivies dans cette note :
> - **I. Value-Based (VFA)** — on apprend $\hat V$ ou $\hat Q$ paramétrés, on extrait la policy.
> - **II. Policy-Based (Policy Gradient)** — on apprend directement $\pi_\theta(a \mid s)$.
> - **III. Actor-Critic (Hybride)** — on apprend les deux conjointement.

---

## I. Value-Based (VFA)

### A. Introduction

Jusqu'ici on représentait $V(s)$ ou $Q(s, a)$ par une table de lookup. Cette approche ne **généralise pas** aux problèmes avec très grands espaces d'états/actions, et empêche un apprentissage rapide d'approximations utiles.

> [!warning] Idée centrale
> Approximer les value functions :
>
> $$v_\pi(s) \approx \hat v(s; \mathbf w) \quad \text{ou} \quad q_\pi(s, a) \approx \hat q(s, a; \mathbf w),$$
>
> où $\mathbf w$ est le **vecteur de paramètres** (poids) de l'approximateur.
>
> **Le but : remplacer une table par une fonction.** Au lieu de stocker une valeur par état (ou par couple état-action) dans un tableau, on la calcule à partir d'un petit nombre de paramètres $\mathbf w$, partagés entre tous les états. On peut ainsi représenter — et mettre à jour — la valeur d'un état jamais visité, à partir de ce qu'on a appris sur des états similaires.

Le problème est apparu concrètement avec **Barto, Sutton et Anderson (1983)**, sur la tâche du **pole balancing** (cart-pole) : un pendule à équilibrer sur un chariot. Sur un petit grid world, le tabulaire fonctionne très bien — chaque état (une case) a sa propre ligne dans la table $V(s)$ ou $Q(s, a)$.

Le problème : le cart-pole (comme `CartPole` sous Gymnasium aujourd'hui) est décrit par un état à **4 nombres réels continus** — 
```(position du chariot, vitesse, angle du pendule, vitesse angulaire).``` Il y a donc une **infinité d'états possibles** : impossible de leur allouer une ligne de table chacun. C'est exactement cette impasse qui a motivé le passage à des approximateurs paramétriques comme le VFA linéaire.

![[Pasted image 20260723170632.png]]
*Figure. L'environnement CartPole : l'état est un vecteur continu (position, vitesse, angle, vitesse angulaire), ce qui exclut toute représentation tabulaire.*

**Choix d'approximateurs courants** : combinaisons linéaires de features, réseaux de neurones, decision trees, k-NN, bases Fourier/wavelet.

![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Value based/im1.png|329]]

> 💡 **On va se concentrer sur deux familles différentiables** : représentations linéaires de features et réseaux de neurones. La différentiabilité est requise pour faire de la descente de gradient.

#### A.1 Pourquoi la généralisation matter

Le RL repose sur quatre composantes :

| Composante | Détail |
| :--- | :--- |
| **Optimization** | Étant donné un problème de décision séquentielle, trouver la politique $\pi$ qui maximise le retour espéré $\mathbb{E}[G_t]$. C'est tout le passage qu'on a fait sur MDP → Bellman → DP → MC → TD/SARSA/Q-learning ([[01_RL Tabulaire]]). |
| **Delayed consequences** | La difficulté que l'optimisation doit gérer — une action à $t$ affecte des récompenses bien après $t$ (d'où le discount $\gamma$, le retour $G_t$, l'équation de Bellman qui propage les valeurs futures). |
| **Exploration** | Une difficulté propre au fait qu'on n'a pas de modèle et qu'on doit générer ses propres données pour optimiser (d'où ε-greedy, GLIE). |
| **Généralisation** | Une difficulté propre au fait que l'espace d'états est trop grand pour optimiser état par état (d'où le VFA de cette note) — c'est cette dernière qu'on adresse ici. On veut généraliser à des états-actions jamais vus (par exemple un nouveau pixel dans Atari). |

> [!warning] Trois bénéfices clés
> - **Mémoire** : moins de stockage pour $(P, R) / V / Q / \pi$.
> - **Calcul** : moins de calcul pour évaluer/mettre à jour ces objets.
> - **Expérience** : moins de samples nécessaires pour apprendre une bonne policy.

### B. VFA Linéaire — Prédiction

> [!warning] Objectif
> Trouver le vecteur $\mathbf w$ qui minimise la perte entre la vraie value function $V^\pi(s)$ et son approximation $\hat V(s; \mathbf w)$ :
>
> $$J(\mathbf w) = \mathbb{E}_\pi\big[(V^\pi(s) - \hat V(s; \mathbf w))^2\big].$$

Descente de gradient :

$$\Delta \mathbf w = -\frac{1}{2} \alpha \nabla_\mathbf{w} J(\mathbf w).$$

**SGD** échantillonne le gradient :

$$\Delta \mathbf w = \alpha (V^\pi(s) - \hat V(s; \mathbf w)) \nabla_\mathbf w \hat V(s).$$

> 💡 **Drawback fondamental.** On n'a pas accès à un oracle qui donne $V^\pi(s)$ pour tout $s$ — c'est précisément ce qu'on veut estimer. La solution : remplacer $V^\pi(s)$ par un **target estimé** (MC ou TD).

#### B.1 Feature vector

Avant deep learning, l'ingénierie des features était l'essentiel du job. Maintenant, les NN font ça automatiquement.

$$\mathbf x(s) = \begin{pmatrix} x_1(s) \\ x_2(s) \\ \vdots \\ x_n(s) \end{pmatrix} = \begin{pmatrix} \text{dist at 1} \\ \text{dist at 2} \\ \vdots \\ \text{dist at 180} \end{pmatrix}.$$

> [!example]- Deux exemples de feature vector
> **Cas particulier important : $\mathbf x(s) = s$.** Quand l'état est déjà un vecteur de nombres directement exploitable, aucune transformation n'est nécessaire — la fonction "feature" est l'identité, on donne l'état brut tel quel à l'approximateur.
>
> - **Capteurs de distance** (ci-dessus) : $n = 180$, un capteur de distance par degré. Ici $\mathbf x(s) \neq s$ — c'est un cas où les features doivent être construites à la main à partir d'un état plus abstrait (position/orientation du robot).
> - **CartPole** (section G) : $\mathbf x(s) = s$ exactement — le vecteur à **4 dimensions** de l'état brut (position du chariot, vitesse, angle du pendule, vitesse angulaire) sert directement de feature vector, sans aucune ingénierie de features.

Pour une approximation linéaire :

$$\hat V(s; \mathbf w) = \sum_{j=1}^n x_j(s) w_j = \mathbf x(s)^T \mathbf w.$$

Update :

$$\Delta \mathbf w = -\frac{1}{2} \alpha \cdot 2 (V^\pi(s) - \hat V(s; \mathbf w)) \, \mathbf x(s).$$

#### B.2 Monte Carlo VFA

> [!warning] Idée
> Le retour $G_t$ est un échantillon **non biaisé mais bruité** du vrai retour espéré $V^\pi(s_t)$. On le substitue dans la loss :
>
> $$\boxed{J(\mathbf w) = \mathbb{E}_\pi[(G_t - \hat V(s; \mathbf w))^2]}$$

On réduit le problème à un **supervised learning** sur des paires $(s_i, G_i)$. Update :

$$\Delta \mathbf w = \alpha (G_t - \hat V(s_t; \mathbf w)) \nabla_\mathbf w \hat V(s_t; \mathbf w) = \alpha (G_t - \mathbf x(s_t)^T \mathbf w) \mathbf x(s_t).$$

> [!note]- Pseudo-code MC Linear VFA (first-visit)
> 1. Initialiser $\mathbf w = 0$, $k = 1$.
> 2. Pour chaque épisode $k$ généré par $\pi$ :
>    - Pour $t = 1, \ldots, L_k$ :
>      - Si first visit à $s$ dans l'épisode $k$ :
>        - $G_t(s) = \sum_{j=t}^{L_k} \gamma^{j-t} r_{k,j}$.
>        - $\mathbf w \leftarrow \mathbf w + \alpha (G_t(s) - \hat V(s; \mathbf w)) \mathbf x(s)$.
> 3. $k \leftarrow k + 1$.

**Variante batch.**

> 💡 **Motivation.** Dans les Recommender Systems on n'a pas tout le data en une fois — on le voit par batch. La forme batch :
>
> $$\arg\min_\mathbf w \sum_{i=1}^N (G(s_i) - \mathbf x(s_i)^T \mathbf w)^2.$$
>
> Solution analytique :
>
> $$\mathbf w = (X^T X)^{-1} X^T \mathbf G.$$

> [!example]- CartPole — MC VFA comme une régression linéaire
> **L'idée : construire une table comme un dataset supervisé.** On reprend les trois transitions de l'exemple de C.1/C.2 et leurs retours déjà calculés ($G_0 = 2.71$, $G_1 = 1.9$, $G_2 = 1$). On range chaque $(s_t, G_t)$ comme une ligne : les colonnes de features à gauche, la cible $G_t$ à droite — exactement comme un jeu de données de régression linéaire classique.
>
> | $i$ | pos | vel | angle | angvel | bias | **target $G$** |
> |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
> | 1 | 0.00 | 0.02 | 0.01 | −0.01 | 1 | **2.71** |
> | 2 | 0.00 | 0.04 | 0.00 | −0.02 | 1 | **1.90** |
> | 3 | 0.01 | 0.03 | 0.02 | 0.05 | 1 | **1.00** |
>
> Les 5 premières colonnes forment la matrice de design $X \in \mathbb{R}^{3 \times 5}$ (une ligne par état visité, une colonne par feature), la dernière colonne forme le vecteur cible $\mathbf G = (2.71,\, 1.90,\, 1.00)^T$. Résoudre $\mathbf w = (X^T X)^{-1} X^T \mathbf G$, c'est **exactement** une régression linéaire ordinaire (MCO) de $G$ sur les features — celle-là même que tu as vue en 2-statistiques/regression-lineaire.md, sauf que la variable expliquée n'est pas mesurée une fois pour toutes mais **change à chaque fois qu'on rejoue des épisodes** (les $G_t$ dépendent de la politique $\pi$ suivie).
>
> 💡 **Nuance importante.** Ici $N = 3$ observations pour $n = 5$ features (avec le biais) : le système est **sous-déterminé**, $X^T X$ n'est pas inversible avec seulement 3 lignes. En pratique il faut $N \gg n$ — beaucoup plus d'épisodes (donc de lignes) que de features — pour que la régression soit bien posée, exactement comme en régression linéaire classique où il faut plus d'observations que de variables explicatives. C'est aussi pour ça qu'en pratique on préfère la version incrémentale (le pseudo-code ci-dessus) : elle met à jour $\mathbf w$ au fur et à mesure, sans jamais avoir besoin d'inverser une matrice sur tout le dataset accumulé.

#### B.3 TD(0) VFA

> [!warning] Idée
> On remplace l'oracle par le target $TD(0)$ :
>
> $$\boxed{J(\mathbf w) = \mathbb{E}_\pi\big[(r_j + \gamma \hat V^\pi(s_{j+1}; \mathbf w) - \hat V(s_j; \mathbf w))^2\big]}$$

Update :

$$\Delta \mathbf w = \alpha (r + \gamma \mathbf x(s')^T \mathbf w - \mathbf x(s)^T \mathbf w) \mathbf x(s).$$

> [!note]- Pseudo-code TD(0) VFA
> 1. Initialiser $\mathbf w = 0$.
> 2. Boucle :
>    - Échantillonner un tuple $(s_k, a_k, r_k, s_{k+1})$ via $\pi$.
>    - $\mathbf w \leftarrow \mathbf w + \alpha (r + \gamma \mathbf x(s')^T \mathbf w - \mathbf x(s)^T \mathbf w) \mathbf x(s)$.

#### B.4 Garanties de convergence

On définit la **mean squared value error** (MSVE) :

$$\text{MSVE}(\mathbf w) = \sum_{s \in S} d(s) (v^\pi(s) - \hat v^\pi(s; \mathbf w))^2,$$

où $d(s)$ est la distribution stationnaire des états sous $\pi$.

> [!warning] Théorème 8.1 (MC convergence)
> MC policy evaluation avec VFA linéaire converge vers les poids $\mathbf w_{MC}$ avec la **MSVE minimum** :
>
> $$\text{MSVE}(\mathbf w_{MC}) = \min_\mathbf w \sum_{s \in S} d(s) (v^\pi(s) - \hat v^\pi(s; \mathbf w))^2.$$

> [!warning] Théorème 8.2 (TD(0) convergence)
> TD(0) avec VFA converge vers les poids $\mathbf w_{TD}$ qui sont **dans un facteur constant** du minimum :
>
> $$\text{MSVE}(\mathbf w_{TD}) = \frac{1}{1 - \gamma} \min_\mathbf w \sum_{s \in S} d(s) (v^\pi(s) - \hat v^\pi(s; \mathbf w))^2.$$

> [!warning] Convergence des méthodes de contrôle
> | Algorithme | Tabulaire | Linear VFA | Nonlinear VFA |
> | :--- | :--- | :--- | :--- |
> | **Monte Carlo Control** | Oui | (Oui) | Non |
> | **SARSA** | Oui | (Oui) | Non |
> | **Q-Learning** | Oui | Non | Non |
>
> *(Oui)* = oscille autour de la valeur optimale.

**Pourquoi ce tableau a cette forme précise — le Deadly Triad.** La divergence apparaît quand **trois ingrédients** sont réunis en même temps :

1. **Function approximation** (au lieu du tabulaire) — la mise à jour d'un état affecte aussi les états voisins, via les poids partagés $\mathbf w$.
2. **Bootstrapping** (TD/Q-learning, qui utilisent leur propre estimation $\hat V$/$\hat Q$ comme cible) — contrairement à MC, qui utilise le retour réel $G_t$.
3. **Off-policy** (la politique évaluée diffère de la politique suivie) — c'est le cas de Q-learning ($\max_{a'}$, politique gloutonne) mais pas de SARSA (on-policy).

![[Pasted image 20260724104251.png|313]]
*Figure. Le Deadly Triad — un triangle dont les trois sommets sont function approximation, bootstrapping et off-policy learning ; la divergence apparaît quand les trois sont réunis simultanément.*

> 💡 **Ce que ça explique dans le tableau ci-dessus.** MC Control n'a pas de bootstrap : il échappe à un des trois ingrédients, d'où le "(Oui)" qui oscille plutôt que diverge franchement. SARSA est on-policy : il échappe à un ingrédient différent, même résultat. Q-learning coche les **trois cases à la fois** (function approximation + bootstrap + off-policy) : c'est le seul cas où le "Non" est total, y compris en VFA linéaire. C'est aussi ce même triad qui motive **Experience Replay** et **Fixed Q-Targets** dans DQN (section E.5) — les deux remèdes qui permettent à Q-learning de fonctionner malgré tout avec un réseau de neurones.

### C. VFA pour le contrôle (action-value)

Similaire à VFA pour value function : on approxime $\hat q(s, a; \mathbf w) \approx q_\pi(s, a)$. On alterne **policy evaluation** (avec $\hat q$) et **policy improvement** ($\epsilon$-greedy).

> [!warning] Loss
> $$J(\mathbf w) = \mathbb{E}_\pi[(q_\pi(s, a) - \hat q^\pi(s, a; \mathbf w))^2].$$

Comme on n'a pas $q_\pi$, on substitue un target :

> [!warning] Trois variantes selon le target
> - **Monte Carlo** : $G_t$ comme target
>
> $$\Delta \mathbf w = \alpha (G_t - \hat Q(s_t, a_t; \mathbf w)) \nabla_\mathbf w \hat Q(s_t, a_t; \mathbf w).$$
>
> - **SARSA** : TD target $r + \gamma \hat Q(s', a'; \mathbf w)$
>
> $$\Delta \mathbf w = \alpha (r + \gamma \hat Q(s', a'; \mathbf w) - \hat Q(s, a; \mathbf w)) \nabla_\mathbf w \hat Q(s, a; \mathbf w).$$
>
> - **Q-learning** : TD target $r + \gamma \max_{a'} \hat Q(s', a'; \mathbf w)$
>
> $$\Delta \mathbf w = \alpha (r + \gamma \max_{a'} \hat Q(s', a'; \mathbf w) - \hat Q(s, a; \mathbf w)) \nabla_\mathbf w \hat Q(s, a; \mathbf w).$$

#### C.1 Construire $\mathbf x(s, a)$ — block feature stacking

**Le problème.** Toutes les formules ci-dessus supposent qu'on sache calculer $\hat Q(s, a; \mathbf w) = \mathbf x(s, a)^T \mathbf w$. Mais jusqu'ici, on n'a construit des feature vectors que pour un **état** ($\mathbf x(s)$, sections B.1 et D) — jamais pour un **couple état-action**. Il faut injecter l'action dans le vecteur.

> [!warning] Idée — un bloc par action
> Avec $|A|$ actions discrètes et un feature vector d'état $\mathbf x(s) \in \mathbb{R}^n$, on construit $\mathbf x(s, a) \in \mathbb{R}^{n \times |A|}$ en **empilant** $|A|$ blocs de taille $n$ : seul le bloc correspondant à l'action $a$ contient $\mathbf x(s)$, tous les autres blocs sont à zéro.
>
> $$\mathbf w^T \mathbf x(s, a) \;=\; \mathbf w_a^T \mathbf x(s),$$
>
> où $\mathbf w_a$ est le sous-vecteur de $\mathbf w$ correspondant au bloc actif. C'est équivalent à apprendre **un vecteur de poids séparé par action**, mais exprimé comme un seul produit scalaire — donc compatible avec toutes les formules d'update déjà écrites (MC, SARSA, Q-learning).

> [!example] CartPole — $\mathbf x(s, a)$ concrètement
> CartPole a $|A| = 2$ actions (gauche, droite) et $\mathbf x(s) = (\text{pos}, \text{vel}, \text{angle}, \text{angvel}, 1) \in \mathbb{R}^5$ (4 coordonnées de l'état + biais, cf. B.1). Le vecteur $\mathbf x(s, a)$ empile deux blocs de taille 5, soit $10$ dimensions au total :
>
> $$\mathbf x(s, \text{gauche}) = (\underbrace{\text{pos}, \text{vel}, \text{angle}, \text{angvel}, 1}_{\text{bloc gauche}},\; \underbrace{0, 0, 0, 0, 0}_{\text{bloc droite}})$$
>
> $$\mathbf x(s, \text{droite}) = (\underbrace{0, 0, 0, 0, 0}_{\text{bloc gauche}},\; \underbrace{\text{pos}, \text{vel}, \text{angle}, \text{angvel}, 1}_{\text{bloc droite}})$$
>
> $\mathbf w$ a lui aussi 10 dimensions : les 5 premières forment $\mathbf w_{\text{gauche}}$, les 5 dernières $\mathbf w_{\text{droite}}$. Comme le bloc inactif de $\mathbf x(s,a)$ est nul, une mise à jour sur $(s, \text{gauche})$ ne touche que $\mathbf w_{\text{gauche}}$ — les poids de l'action droite restent intacts.

**Algorithme complet (MC, CartPole).** Concrètement, avec la cible Monte Carlo, l'entraînement se déroule ainsi sur un épisode :

> [!warning] Entraînement — MC Control avec block feature
> 1. Générer un épisode complet en suivant la politique $\varepsilon$-greedy courante par rapport à $\hat Q(\cdot, \cdot; \mathbf w)$ :
>    $$(s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1}, s_T \;(\text{chute})).$$
> 2. **Attendre la fin de l'épisode** (la chute du pendule), puis remonter la trajectoire **à l'envers**, de $t = T-1$ jusqu'à $t = 0$ :
>    - Calculer le retour au pas $t$ par récursion arrière : $G_{T-1} = r_{T-1}$, puis $G_t = r_t + \gamma \, G_{t+1}$.
>    - Construire $\mathbf x(s_t, a_t)$ par block stacking (bloc gauche ou droite selon $a_t$, comme ci-dessus).
>    - Mettre à jour : $\mathbf w \leftarrow \mathbf w + \alpha \big(G_t - \hat Q(s_t, a_t; \mathbf w)\big) \, \mathbf x(s_t, a_t)$.
>    - Reculer d'un pas ($t \leftarrow t - 1$) et recommencer.
> 3. Passer à l'épisode suivant, avec un $\varepsilon$ qui a décru (schedule GLIE).
>
> **Concrètement** : on part de la dernière transition avant la chute, on regarde quelle action $a_{T-1}$ a été prise à cet instant, on construit le bloc $\mathbf x(s_{T-1}, a_{T-1})$ (le bloc gauche ou droite rempli avec l'état, l'autre à zéro), on met à jour $\mathbf w$ avec $G_{T-1} = r_{T-1}$, puis on recule d'une case dans la séquence, on refait exactement la même chose avec $G_{T-2} = r_{T-2} + \gamma G_{T-1}$, et ainsi de suite jusqu'à revenir à $s_0$.

> 💡 **Pourquoi remonter à l'envers.** $G_t$ se calcule récursivement à partir de $G_{t+1}$ ($G_t = r_t + \gamma G_{t+1}$) — c'est beaucoup plus simple à programmer en partant de la fin (où $G_{T-1} = r_{T-1}$ est immédiat) plutôt qu'en resommant toutes les récompenses futures à chaque pas. L'ordre dans lequel on applique les mises à jour de $\mathbf w$ n'a pas d'importance pour la convergence — seul compte le fait que chaque $(s_t, a_t, G_t)$ soit traité une fois.

#### C.2 Phase d'inférence

**Le contexte.** C.1 décrit la **phase d'entraînement** : $\mathbf w$ bouge à chaque mise à jour, et l'agent explore ($\varepsilon > 0$) pour découvrir de meilleures actions. Une fois l'entraînement terminé (ou jugé suffisant), on passe à la **phase d'inférence** (aussi dite d'évaluation ou de déploiement) : $\mathbf w$ est figé, et on utilise l'agent pour décider en conditions réelles.

> [!warning] Inférence — utiliser $\mathbf w$ sans l'entraîner
> 1. **Geler $\mathbf w$** : plus aucune mise à jour, plus de $G_t$ ni de gradient à calculer.
> 2. À chaque nouvel état $s_t$ rencontré :
>    - Construire $\mathbf x(s_t, \text{gauche})$ et $\mathbf x(s_t, \text{droite})$ par block stacking (C.1).
>    - Calculer $\hat Q(s_t, \text{gauche}; \mathbf w) = \mathbf w^T \mathbf x(s_t, \text{gauche})$ et $\hat Q(s_t, \text{droite}; \mathbf w) = \mathbf w^T \mathbf x(s_t, \text{droite})$ — deux produits scalaires, quasi instantanés.
>    - Choisir $a_t^* = \arg\max_a \hat Q(s_t, a; \mathbf w)$ (glouton pur, ou $\varepsilon$ résiduel très faible, ex. $0.05$).
>    - Exécuter $a_t^*$ dans l'environnement.
> 3. Répéter à chaque pas, jusqu'à la fin de l'épisode (ou indéfiniment en usage réel).

> 💡 **Ce qui change par rapport à l'entraînement.** Pas d'attente de fin d'épisode, pas de calcul de $G_t$, pas de mise à jour de $\mathbf w$ : l'inférence est purement un enchaînement de "observer $\to$ calculer deux produits scalaires $\to$ agir". C'est exactement ce qu'on ferait avec un DQN entraîné (section E) — sauf qu'ici $\hat Q$ est un produit scalaire linéaire plutôt qu'un forward pass dans un réseau de neurones.

> [!example]- CartPole — un épisode complet à la main
> **Setup.** $\gamma = 0.9$, $\alpha = 0.1$, $\mathbf w$ initialisé à $\mathbf 0$ (10 dimensions : 5 pour le bloc gauche, 5 pour le bloc droite). Épisode de 3 pas, terminé par une chute juste après le pas $t=2$. Les $s_t$ ci-dessous sont juste des repères de temps dans **cette** trajectoire précise — pas des labels réutilisables comme "Bull"/"Bear" en 01.
>
> | $t$ | $s_t$ = (pos, vel, angle, angvel) | $a_t$ | $r_t$ |
> |:---:|---|:---:|:---:|
> | 0 | $(0.00,\, 0.02,\, 0.01,\, -0.01)$ | Gauche | $+1$ |
> | 1 | $(0.00,\, 0.04,\, 0.00,\, -0.02)$ | Droite | $+1$ |
> | 2 | $(0.01,\, 0.03,\, 0.02,\, 0.05)$ | Droite | $+1$ (puis chute) |
>
> **Étape 1 — calculer les retours en remontant depuis la fin** (C.1) :
>
> $$G_2 = 1, \qquad G_1 = 1 + 0.9 \times 1 = 1.9, \qquad G_0 = 1 + 0.9 \times 1.9 = 2.71.$$
>
> **Étape 2 — mettre à jour $\mathbf w$ en remontant $t = 2 \to t = 0$**, en partant de $\mathbf w = \mathbf 0$.
>
> *Pas $t=2$ (Droite).* $\mathbf x(s_2, \text{droite}) = (0,0,0,0,0,\; 0.01, 0.03, 0.02, 0.05, 1)$. Comme $\mathbf w = \mathbf 0$, $\hat Q(s_2, \text{droite}; \mathbf w) = 0$.
>
> $$\mathbf w \leftarrow \mathbf 0 + 0.1 \times (1 - 0) \times \mathbf x(s_2, \text{droite}) = (0,0,0,0,0,\; 0.001,\, 0.003,\, 0.002,\, 0.005,\, 0.100).$$
>
> *Pas $t=1$ (Droite).* $\mathbf x(s_1, \text{droite}) = (0,0,0,0,0,\; 0.00, 0.04, 0.00, -0.02, 1)$. Avec le $\mathbf w$ obtenu au pas précédent, $\hat Q(s_1, \text{droite}; \mathbf w) \approx 0.100$.
>
> $$\mathbf w \leftarrow \mathbf w + 0.1 \times (1.9 - 0.100) \times \mathbf x(s_1, \text{droite}) \approx (0,0,0,0,0,\; 0.001,\, 0.010,\, 0.002,\, 0.001,\, 0.280).$$
>
> *Pas $t=0$ (Gauche).* $\mathbf x(s_0, \text{gauche}) = (0.00, 0.02, 0.01, -0.01, 1,\; 0,0,0,0,0)$. Le bloc gauche de $\mathbf w$ est encore à $\mathbf 0$ (jamais touché), donc $\hat Q(s_0, \text{gauche}; \mathbf w) = 0$.
>
> $$\mathbf w \leftarrow \mathbf w + 0.1 \times (2.71 - 0) \times \mathbf x(s_0, \text{gauche}) \approx (0,\, 0.005,\, 0.003,\, -0.003,\, 0.271,\; 0.001,\, 0.010,\, 0.002,\, 0.001,\, 0.280).$$
>
> **Résultat final.**
>
> $$\mathbf w_{\text{gauche}} \approx (0,\, 0.005,\, 0.003,\, -0.003,\, 0.271), \qquad \mathbf w_{\text{droite}} \approx (0.001,\, 0.010,\, 0.002,\, 0.001,\, 0.280).$$
>
> Trois observations :
> - **Les deux blocs évoluent indépendamment.** Le bloc gauche n'a été mis à jour qu'une fois (au pas $t=0$, seule fois où Gauche a été choisie) ; le bloc droite deux fois ($t=1$ et $t=2$). C'est exactement la propriété annoncée en C.1 : une mise à jour sur une action ne touche jamais les poids de l'autre action.
> - **Le biais domine** dans les deux blocs : logique, puisque les autres coordonnées de l'état sont proches de zéro dans cet exemple — c'est le biais qui capte l'essentiel de "combien vaut le fait d'être en vie encore quelques pas".
> - **Avec plus d'épisodes**, ce même mécanisme (remonter, calculer $G_t$, mettre à jour le bloc actif) se répète et affine $\mathbf w$, jusqu'à ce que $\hat Q$ distingue correctement les états stables des états sur le point de tomber.

### D. Exemple — Grid World

> [!example] Robot dans une grille
> Grille avec stations de charge (vert, $+1$), escaliers (rouge, $-1$), neutre ailleurs. 4 actions (forward, backward, left, right) avec proba 0.2 d'erreur.
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Value based/im2.png]]
>
> **Initialisation.**
>
> $$V_\pi(s) = \begin{pmatrix} v_1^\pi & v_2^\pi & \ldots & v_{25}^\pi \end{pmatrix}, \quad Q_\pi(s, a) \in \mathbb{R}^{25 \times 4}.$$
>
> **Feature vector** par état :
>
> $$\mathbf x(s) = \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}.$$
>
> Position $(x, y)$ + bias.
>
> **TD(0) VFA target** :
>
> $$U^\sim(s; \mathbf w) = \mathbf x(s_{t+1})^T \mathbf w.$$
>
> **Update rule** :
>
> $$\mathbf w_{t+1} = \mathbf w_t + \alpha [r_{t+1} + \mathbf x(s_{t+1})^T \mathbf w - \mathbf x(s)^T \mathbf w] \mathbf x(s).$$

> [!note]- Calcul concret pour un transition $(4,3) \to (4,4)$
> Avec $\mathbf w = (1.2, 1.2, -0.5)^T$ et $\alpha = 0.6$ :
>
> $$\mathbf w_{t+1} = \begin{pmatrix} 1.2 \\ 1.2 \\ -0.5 \end{pmatrix} + 0.6 \cdot \left[ +1 + \begin{pmatrix} 4 \\ 4 \\ 1 \end{pmatrix}^T \mathbf w - \begin{pmatrix} 4 \\ 3 \\ 1 \end{pmatrix}^T \mathbf w \right] \begin{pmatrix} 4 \\ 3 \\ 1 \end{pmatrix}.$$

**Résultat après convergence.** Avec $\mathbf w = (0.126, 0.120, -0.718)^T$ :

$$U^\sim(s_0) = \mathbf w^T (0, 0, 1) = -0.72.$$

![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Value based/im3 (1).png]]

> [!note]- Implémentation Python (extrait)
> ```python
> def update(w, x, x_t1, reward, alpha, gamma, done):
>     if done:
>         w_t1 = w + alpha * ((reward - np.dot(x, w)) * x)
>     else:
>         w_t1 = w + alpha * ((reward + gamma * np.dot(x_t1, w) - np.dot(x, w)) * x)
>     return w_t1
> 
> for step in range(1000):
>     action = np.random.randint(0, 4)
>     new_obs, reward, done = env.step(action)
>     x_t1 = np.array(new_obs + [1]) if use_bias else np.array(new_obs)
>     w = update(w, x, x_t1, reward, alpha, gamma, done)
>     x = x_t1
>     if done: break
> ```

### E. Deep Q-Learning (DQN)

#### E.1 Contexte historique

> 💡 **Historique en bref.**
> - **1994** : TD-Backgammon (Tesauro), premier succès neural net en RL.
> - **1995-1998** : focus sur function approximation off-policy + bootstrapping (TD). **Beaucoup de divergence** → la communauté devient prudente.
> - **Mid-2000s** : renaissance avec Deep NN.
> - **2014** : DeepMind sort **DQN sur Atari** — révolution.

![[neural-1.png]]

#### E.2 Pourquoi NN — limites du linéaire

Comme dans le **XOR-world**, certaines value functions ne se modélisent pas linéairement. On peut ajouter des termes d'interaction :

$$\hat U(s; \mathbf w) = x_1 w_1 + x_2 w_2 + x_1 x_2 w_3 + w_4.$$

![[neural-2.png]]

Mais ça reste manuel. Mieux : un **NN profond** qui apprend les features automatiquement :

![[neural-3.png]]

> [!warning] Trois familles d'approximateurs
> - **(i) Linéaire** — combinaison pondérée de features. Marche bien si les features sont bien choisies, mais demande du feature engineering.
> - **(ii) Kernel-based** — plus riche, sans features explicites. Convergence sous certaines conditions, mais pas scalable.
> - **(iii) Deep NN** — universal approximator, distributed representations, exponentiellement moins de paramètres qu'un shallow net pour la même fonction. SGD pour apprendre.

#### E.3 Architecture DQN

> 💡 **Référence.** Mnih et al. 2015 (Nature). Réseau qui prend en entrée des images Atari préprocessées et sort un vecteur de Q-values, une par action.

**Préprocessing des frames Atari.**

> [!note]- Pipeline de préprocessing
> 1. **Single frame encoding** — max pooling pixel-wise sur 2 frames consécutives (gère les artefacts du jeu).
> 2. **Dimensionality reduction** — extraire le canal Y (luminance), redimensionner à $84 \times 84 \times 1$.
> 3. **Stack 4 frames récentes** → input $84 \times 84 \times 4$ (pour avoir une approximation Markovienne).

**Architecture du réseau.**

> [!note]- Détails couches
> - Input : $84 \times 84 \times 4$.
> - Conv1 : 32 filtres $8 \times 8$, stride 4 + ReLU.
> - Conv2 : 64 filtres $4 \times 4$, stride 2 + ReLU.
> - Conv3 : 64 filtres $3 \times 3$, stride 1 + ReLU.
> - FC : 512 unités + ReLU.
> - Output : linéaire, une unité par action.

#### E.4 Loss et target

> [!warning] Loss DQN
> $$J(\mathbf w) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1})}\big[(y_t^{DQN} - \hat q(s_t, a_t; \mathbf w))^2\big],$$
>
> avec target one-step :
>
> $$y_t^{DQN} = r_t + \gamma \max_{a'} \hat q(s_{t+1}, a'; \mathbf w^-),$$
>
> où $\mathbf w^-$ sont les paramètres du **target network** (fixés temporairement), et le target $y_t$ est traité comme **fixe** lors du SGD sur $\mathbf w$.

#### E.5 Les deux innovations clés

> 💡 **Pourquoi Q-learning avec VFA peut diverger.**
> - **Corrélation entre samples** : transitions consécutives très corrélées → pas i.i.d.
> - **Targets non-stationnaires** : on modifie la policy continuellement.
>
> DQN adresse ces deux problèmes par **Experience Replay** et **Fixed Q-Targets**.

> [!warning] (1) Experience Replay
> Les transitions $e_t = (s_t, a_t, r_t, s_{t+1})$ sont stockées dans un **replay buffer** $D$ de taille fixe (1M dans le papier original). Le réseau est mis à jour par SGD sur des **mini-batches uniformément échantillonnés** depuis $D$.
>
> ![[dqn-3.png]]

> [!note]- Avantages de l'experience replay
> - **Greater data efficiency** — chaque transition peut servir à plusieurs updates.
> - **Décorrélation des samples** — randomization brise la corrélation, réduit la variance, stabilise.
> - **Évite oscillations/divergence** — la distribution comportementale est moyennée sur plusieurs anciens états.
>
> *Note : nécessite du off-policy (Q-learning) parce que les params actuels sont différents de ceux qui ont généré les samples.*

> 💡 **Limite et extension.** Le buffer de base traite toutes les transitions équitablement. **Prioritized Replay** (Schaul et al.) rejoue plus souvent les transitions importantes (TD error élevée) — agent apprend plus efficacement.

> [!warning] (2) Fixed Q-Targets
> Pour stabiliser, on utilise un **target network séparé** $\hat q(s, a; \mathbf w^-)$ pour générer les $y_j$. Tous les $C$ updates (typiquement $C = 10000$), on copie $\mathbf w^- \leftarrow \mathbf w$. Entre temps, le target reste **fixe**.

#### E.6 Pseudo-code DQN

> [!note]- Algorithme complet
> 1. Initialiser le replay memory $D$ avec capacité fixe.
> 2. Initialiser $\hat q$ avec poids aléatoires $\mathbf w$.
> 3. Initialiser le target network $\hat q$ avec $\mathbf w^- = \mathbf w$.
> 4. Pour chaque épisode $m = 1, \ldots, M$ :
>    - Observer la frame initiale $x_1$, préprocesser pour obtenir $s_1$.
>    - Pour chaque time step $t = 1, \ldots, T$ :
>      - Sélectionner $a_t = \begin{cases} \text{action aléatoire} & \text{avec proba } \epsilon \\ \arg\max_a \hat q(s_t, a; \mathbf w) & \text{sinon} \end{cases}$
>      - Exécuter $a_t$, observer $r_t$ et $x_{t+1}$.
>      - Préprocesser pour obtenir $s_{t+1}$, stocker $(s_t, a_t, r_t, s_{t+1})$ dans $D$.
>      - Échantillonner uniformément un mini-batch de $N$ transitions de $D$.
>      - $y_j = r_j$ si épisode termine à $j+1$, sinon $y_j = r_j + \gamma \max_{a'} \hat q(s_{j+1}, a'; \mathbf w^-)$.
>      - SGD sur $J(\mathbf w) = \frac{1}{N} \sum_j (y_j - \hat q(s_j, a_j; \mathbf w))^2$.
>      - Tous les $C$ steps : $\mathbf w^- \leftarrow \mathbf w$.

#### E.7 Détails d'entraînement (papier original)

> 💡 **Hyperparamètres clés.**
> - **Reward clipping** à $[-1, +1]$ (permet d'utiliser le même learning rate sur tous les jeux).
> - **Frame skipping** (action repeat) — l'agent agit toutes les 4 frames, l'action est répétée. Réduit la fréquence de décision sans perdre en perf.
> - **RMSProp** + mini-batch 32.
> - $\epsilon$-greedy : $\epsilon$ décroît linéairement de 1.0 à 0.1 sur le premier million de steps, puis fixé à 0.1.
> - À l'évaluation : $\epsilon = 0.05$.

#### E.8 Application Atari

![[dqn-5.png]]
**Figure.** L'état est l'image complète (avec stack de frames pour la dynamique).

![[dqn-4.png]]
**Figure.** Pong demande plus que l'image courante (vélocité, position de la balle) — d'où le stack de 4 frames.

![[dqn-1.png]]
**Figure.** Architecture DQN appliquée aux jeux Atari : 84×84×4 → 3 conv layers → 2 FC → output par action.

![[dqn-2.png]]
**Figure.** Résultat — niveau humain sur la majorité des jeux Atari (Mnih et al. 2015).

> [!warning] Quelles features étaient critiques au succès ?
> | Game | Linear | Deep Net | DQN w/ fixed Q | DQN w/ replay | DQN w/ replay + fixed Q |
> | :--- | :---: | :---: | :---: | :---: | :---: |
> | Breakout | 3 | 3 | 10 | 241 | **317** |
> | Enduro | 62 | 29 | 141 | 831 | **1006** |
> | River Raid | 2345 | 1453 | 2868 | 4102 | **7447** |
> | Seaquest | 656 | 275 | 1003 | 823 | **2894** |
> | Space Invaders | 301 | 302 | 373 | 826 | **1089** |
>
> 💡 **Replay** apporte le plus gros gain sur la plupart des jeux. **Fixed Q** stabilise davantage. La combinaison est meilleure.

> 💡 **Extensions notables (au-delà du papier original).**
> - **Double DQN** (Van Hasselt et al. AAAI 2016) — réduit l'overestimation.
> - **Prioritized Replay** (Schaul et al. ICLR 2016).
> - **Dueling DQN** (Wang et al. ICML 2016, best paper).

### F. Double DQN

> 💡 **Motivation.** L'opérateur $\max$ dans DQN utilise le **même réseau** pour sélectionner ET évaluer une action — biais d'**overestimation**. Solution : découpler.

> [!warning] Target Double DQN
> $$y_t^{\text{DoubleDQN}} = r_t + \gamma \, \hat q\big(s_{t+1}, \, \arg\max_{a'} \hat q(s_{t+1}, a'; \mathbf w); \, \mathbf w^-\big).$$
>
> L'**arg max** est calculé avec le réseau online ($\mathbf w$), la valeur est évaluée avec le target ($\mathbf w^-$).

Le reste de DQN (replay, target network update périodique) reste identique.

> [!note]- Pseudo-code Double Q-learning (version tabulaire originale, Hasselt)
> 1. Initialiser $Q_1, Q_2$.
> 2. Boucle :
>    - $\pi(s) = \arg\max_a Q_1(s, a) + Q_2(s, a)$, sample $a_t$ par $\epsilon$-greedy.
>    - Observer $(r_t, s_{t+1})$.
>    - Avec proba 0.5 :
>      - $Q_1(s_t, a_t) \leftarrow Q_1(s_t, a_t) + \alpha (r_t + Q_1(s_{t+1}, \arg\max_{a'} Q_2(s_{t+1}, a')) - Q_1(s_t, a_t))$.
>    - Sinon : symétrique en swappant $Q_1, Q_2$.

### G. Dueling DQN

> [!warning] Définition (Advantage function)
> $$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).$$
>
> Mesure relative de l'importance de chaque action par rapport à la valeur de l'état. Comme $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$, on a $\mathbb{E}_{a \sim \pi}[A^\pi(s, a)] = 0$.

#### G.1 Architecture

L'architecture **dueling** sépare le réseau en **deux flux** après les couches conv :
- Un flux pour estimer $\hat v(s; \mathbf w, \mathbf w_v)$ (state value).
- Un flux pour estimer $A(s, a; \mathbf w, \mathbf w_A)$ (advantage par action).
- Module d'agrégation pour produire $\hat q(s, a)$.

![[ddqn-1.png]]

> 💡 **Intuitions des auteurs.**
> - Pour beaucoup d'états, l'action choisie n'a pas d'importance — seul $V$ matter pour le bootstrap.
> - Les features pour estimer $V$ peuvent être différentes de celles pour estimer les avantages d'action.

#### G.2 Module d'agrégation

**Naïf** : $\hat q = \hat v + A$. **Problème** : non-identifiable (on peut ajouter une constante à $\hat v$ et la soustraire de $A$ sans changer $\hat q$). Mauvaise perf en pratique.

**Correction (max)** :

$$\hat q = \hat v + \big(A(s, a) - \max_{a'} A(s, a')\big).$$

**Variante (mean)** — choisie en pratique pour la stabilité :

$$\boxed{\hat q = \hat v + \big(A(s, a) - \tfrac{1}{|A|} \sum_{a'} A(s, a')\big).}$$

> 💡 **Pourquoi mean.** Les avantages doivent juste évoluer aussi vite que la moyenne, plutôt que compenser tout changement de l'avantage de l'action optimale. Plus stable. Perd un peu la sémantique pure mais gagne en convergence.

> 💡 **Quand le dueling brille.** Particulièrement utile quand le **nombre d'actions est grand**. State-of-the-art Atari en 2016.

### H. Imitation Learning

> 💡 **Motivation.** Apprendre depuis des **rewards sparse** est lent et risqué (autonomous driving, médical). Une alternative : apprendre par **imitation** d'un expert.

#### H.1 Setup

> [!warning] Learning from Demonstration
> On dispose de :
> - State space, action space.
> - Transition model $P(s' \mid s, a)$.
> - **Pas de reward function** $R$.
> - Set de trajectoires expertes $(s_0, a_0, s_1, a_1, \ldots)$ avec actions de la policy experte $\pi^*$.

#### H.2 Behavioral Cloning

> 💡 **Idée la plus simple.** Apprendre $\pi$ par **supervised learning** sur les paires $\{(s_i, a_i)\}$. Exemple historique : ALVINN (Pomerleau 1989, sensors → steering angle).

> [!warning] Limite — compounding errors
> Les data ne sont **pas i.i.d.** dans l'espace d'états : elles sont concentrées autour des trajectoires expertes. Si l'agent fait une erreur et atterrit dans un état non visité par l'expert, il n'a aucune donnée pour apprendre une recovery policy.
>
> **L'erreur scale en $T^2$** sur la longueur de l'épisode (vs linéaire en RL standard).

#### H.3 DAGGER (Dataset Aggregation)

> 💡 **Idée.** Itérativement, on génère des données dans les états que la policy actuelle visite, et on demande à l'expert de **labelliser ces nouveaux états**. Ça mitige les compounding errors.

> [!note]- Pseudo-code DAGGER
> 1. $\mathcal D \leftarrow \emptyset$, $\hat \pi_1$ arbitraire.
> 2. Pour $i = 1$ à $N$ :
>    - $\pi_i = \beta_i \pi^* + (1 - \beta_i) \hat \pi_i$ (mélange).
>    - Sample $T$-step trajectories avec $\pi_i$.
>    - Récupérer $\mathcal D_i = \{(s, \pi^*(s))\}$ — actions de l'expert sur les états visités par $\pi_i$.
>    - Aggregate : $\mathcal D \leftarrow \mathcal D \cup \mathcal D_i$.
>    - Train classifier $\hat \pi_{i+1}$ sur $\mathcal D$.
> 3. Retourner le meilleur $\hat \pi_i$ sur validation.

> 💡 **Limite.** L'expert doit être disponible pour labelliser **en temps réel** — pas toujours faisable.

#### H.4 Inverse Reinforcement Learning (IRL)

> 💡 **L'idée centrale.** Récupérer **la reward function** depuis les démonstrations expertes. Sans hypothèse d'optimalité, le problème est mal posé (n'importe quelle reward peut générer ces trajectoires).

**Linear Feature Reward IRL.** Reward représentée comme combinaison linéaire de features :

$$R(s) = w^T x(s).$$

La value function devient :

$$V^\pi(s) = w^T \mu(\pi),$$

où $\mu(\pi) \in \mathbb{R}^n$ est la **fréquence pondérée actualisée** des features sous $\pi$.

> [!warning] Critère de Ng-Russell
> Si l'expert est optimal,
>
> $$\mathbb{E}_{\pi^*}\!\left[\sum_t \gamma^t R^*(s_t) \mid s_0\right] \geq \mathbb{E}_\pi\!\left[\sum_t \gamma^t R^*(s_t) \mid s_0\right] \quad \forall \pi.$$
>
> Donc on cherche $w^*$ tel que :
>
> $$w^{*T} \mu(\pi^* \mid s_0) \geq w^{*T} \mu(\pi \mid s_0) \quad \forall \pi, \forall s.$$
>
> "Trouver une paramétrisation où la policy experte surpasse les autres."

#### H.5 Apprenticeship Learning via IRL

> 💡 **L'idée.** Si on peut **matcher les feature expectations** de l'expert ($\|\mu(\pi) - \mu(\pi^*)\|_1 \leq \epsilon$), alors par Cauchy-Schwarz :
>
> $$|w^T \mu(\pi) - w^T \mu(\pi^*)| \leq \epsilon \quad \forall w \text{ avec } \|w\|_\infty \leq 1.$$

> [!note]- Pseudo-code Apprenticeship Learning
> 1. Initialiser $\pi_0$.
> 2. Pour $i = 1, 2, \ldots$ :
>    - Trouver $w$ tel que l'expert outperform les controllers précédents max :
>
>      $$\arg\max_w \max_\gamma \gamma$$
>
>      $$\text{s.t. } w^T \mu(\pi^*) \geq w^T \mu(\pi) + \gamma, \forall \pi \in \{\pi_0, \ldots, \pi_{i-1}\}.$$
>
>    - Trouver $\pi_i$ optimal pour $w$.
>    - Si $\gamma \leq \epsilon/2$ : retourner $\pi_i$.

> 💡 **Limites pratiques.**
> - Si l'expert est sous-optimal, la policy résultante est un **mélange arbitraire**.
> - Demande de calculer une optimal policy à chaque itération — coûteux.
> - **Infinité de reward functions** ont la même optimal policy.

#### H.6 Maximum Entropy IRL

> 💡 **Adresse l'ambiguïté** des IRL classiques. Ziebart et al. introduit le principe d'entropie maximale : parmi toutes les policies qui matchent les feature expectations, choisir celle de plus grande entropie.

Distribution sur les trajectoires :

$$P(\tau_j \mid w) = \frac{1}{Z(w)} \exp(w^T \mu_{\tau_j}).$$

Likelihood des données observées :

$$L(w) = \sum_{\text{examples}} \log P(\tau \mid w).$$

Gradient :

$$\nabla L(w) = \tilde \mu - \sum_\tau P(\tau \mid w) \mu_\tau = \tilde \mu - \sum_{s_i} D(s_i) x(s_i),$$

où $D(s_i)$ est la **state visitation frequency**.

> 💡 **Très influent.** Sélection principielle parmi les multiples reward functions possibles. Mais demande la connaissance du transition model.

> [!note]- Pseudo-code MaxEnt IRL (résumé)
> **Backward pass** : calcul itératif des $Z_{a_{i,j}}$ et $Z_{s_i}$ (partition functions).
>
> **Local action probability** : $P(a_{i,j} \mid s_i) = Z_{a_{i,j}} / Z_{s_i}$.
>
> **Forward pass** : calcul des state visitation frequencies $D_{s_i, t}$ récursivement.
>
> **Sum frequencies** : $D_{s_i} = \sum_t D_{s_i, t}$.

### I. To do

> [!note] Points à approfondir plus tard
> - **Graphes de convergence de $\mathbf w$.** Générer un vrai graphique (code, pas un placeholder) qui trace l'évolution de $\mathbf w$ (ou d'une composante, ou de la performance/durée d'épisode) au fil des épisodes d'entraînement, pour visualiser concrètement la convergence.
> - **Comparer MC, SARSA et Q-learning (SARSAMAX) sur CartPole.** Un même graphe (ou une petite série de graphes) qui compare les trois méthodes côte à côte sur le même environnement — par exemple durée d'épisode moyenne en fonction du nombre d'épisodes d'entraînement, pour voir laquelle converge le plus vite et le plus stablement.
> - **Prudence de SARSA vs. agressivité de Q-learning.** Illustrer concrètement sur CartPole (pas seulement l'exemple cliff walking de 01) que Q-learning, en évaluant la politique gloutonne plutôt que la politique réellement suivie, peut apprendre une politique plus "risquée" qui se comporte mal pendant l'exploration — alors que SARSA, en tenant compte du coût de l'exploration ($\varepsilon$-greedy), apprend une politique plus prudente. Point déjà discuté en 01 (SARSA vs Q-learning, section IV.B) mais jamais illustré numériquement sur CartPole.

---

## II. Policy-Based (Policy Gradient)

### A. Introduction au policy search

> [!warning] Le shift conceptuel
> Au lieu d'apprendre $V$ ou $Q$ et d'extraire la policy ($\epsilon$-greedy, etc.), on **paramétrise directement la policy** :
>
> $$\pi_\theta(a \mid s) = \mathbb{P}[a \mid s; \theta].$$
>
> Objectif : trouver $\theta$ qui maximise $V^\pi$ directement.

> [!warning] Avantages du policy-based
> - **Meilleures propriétés de convergence** (cf. Sutton & Barto Ch. 13.3).
> - Efficace en **action spaces continus ou hauts-dim** (robotique).
> - **Stochastic policies** apprenables — pas possible avec les value-based purs.

> [!warning] Inconvénients
> - Convergence vers des **optima locaux** (gradient descent).
> - **Data-inefficient** et **high variance**.

### B. Pourquoi parfois une stochastic policy

#### B.1 Rock-paper-scissors

Toute policy non-uniforme est exploitable. La policy optimale Nash est uniformément aléatoire :

$$P(\text{rock}) = P(\text{paper}) = P(\text{scissors}) = \tfrac{1}{3}.$$

#### B.2 Aliased gridworld
		
> [!example] Environnement partiellement observable
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im1.png]]
>
> Agent senses uniquement les murs autour. Les **deux cases grises sont indistinguables**. Domaine non-Markovien.
>
> Une policy déterministe doit choisir "toujours gauche" ou "toujours droite" dans les cases grises → l'agent peut rester coincé :
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im2.png]]
>
> Une **stochastic policy** (E ou W avec proba 1/2 dans les cases grises) atteint le but avec haute proba :
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im3 (1).png]]

> 💡 **Conclusion.** Stochastic policies utiles dans des domaines :
> - **Adversariaux** ou non-stationnaires.
> - **Non-Markoviens** (état non pleinement observable).

### C. Policy Gradient — REINFORCE

#### C.1 Objectif et gradient

> [!warning] Objectif
> $$V(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau; \theta) R(\tau)$$
>
> où $\tau = (s_0, a_0, r_0, \ldots, s_T)$ est une trajectoire.

> [!note]- Dérivation du gradient (likelihood ratio trick)
> $$\begin{aligned}
> \nabla_\theta V(\theta) &= \sum_\tau \nabla_\theta P(\tau; \theta) R(\tau) \\
> &= \sum_\tau P(\tau; \theta) R(\tau) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} \\
> &= \sum_\tau P(\tau; \theta) R(\tau) \nabla_\theta \log P(\tau; \theta) \\
> &= \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \nabla_\theta \log P(\tau; \theta)]
> \end{aligned}$$
>
> L'expression $\nabla_\theta P / P$ est le **likelihood ratio**.

> 💡 **Pourquoi cette astuce.** Permet d'estimer le gradient par **échantillonnage de trajectoires** :
>
> $$\hat g = \frac{1}{m} \sum_{i=1}^m R(\tau^{(i)}) \nabla_\theta \log P(\tau^{(i)}; \theta).$$

> [!note]- Décomposition de $\nabla_\theta \log P(\tau; \theta)$
> $$\begin{aligned}
> \nabla_\theta \log P(\tau; \theta) &= \nabla_\theta \log\!\left[ \mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t) \right] \\
> &= \nabla_\theta\!\left[ \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \right] \\
> &= \sum_{t=0}^{T-1} \underbrace{\nabla_\theta \log \pi_\theta(a_t \mid s_t)}_{\text{score function}}.
> \end{aligned}$$

> 💡 **Le gros résultat.** Le **dynamics model et la distribution initiale disparaissent** ! On n'a besoin que de $\pi_\theta$.

> [!warning] Forme finale
> $$\boxed{\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(\tau^{(i)}) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})}$$

#### C.2 Exploiter la structure temporelle

L'action à $t$ ne peut affecter que les **rewards à $t' \geq t$**, pas avant. En décomposant $R(\tau) = \sum_t r_t$ :

$$\nabla_\theta V(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T-1} r_{t'}\right] = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right],$$

où $G_t = \sum_{t' \geq t} r_{t'}$ est le **retour à partir de $t$**.

> [!warning] Forme REINFORCE finale
> $$\boxed{\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^{T-1} G_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})}$$

#### C.3 Pseudo-code REINFORCE

> [!note]- REINFORCE (Monte Carlo policy gradient)
> 1. Initialiser $\theta$ arbitrairement.
> 2. Pour chaque épisode $\{s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$ :
>    - Pour $t = 1$ à $T - 1$ :
>      - $\theta \leftarrow \theta + \alpha \cdot G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
> 3. Retourner $\theta$.

### D. Classes de policies différentiables

#### D.1 Action space discret — softmax policy

$$\pi_\theta(a \mid s) = \frac{e^{\phi(s, a)^T \theta}}{\sum_{a'} e^{\phi(s, a')^T \theta}}.$$

> [!note]- Score function
> $$\nabla_\theta \log \pi_\theta(a \mid s) = \phi(s, a) - \mathbb{E}_{a' \sim \pi_\theta}[\phi(s, a')].$$

#### D.2 Action space continu — Gaussian policy

$$a \sim \mathcal{N}(\mu(s), \sigma^2), \quad \mu(s) = \phi(s)^T \theta.$$

Score function :

$$\nabla_\theta \log \pi_\theta(a \mid s) = \frac{(a - \mu(s)) \phi(s)}{\sigma^2}.$$

### E. Vanilla Policy Gradient avec baseline

> 💡 **Le problème de REINFORCE.** Les retours $G_t^{(i)}$ ont une **variance énorme** entre épisodes.
>
> **Solution** : soustraire une **baseline** $b(s)$ à $G_t$. La baseline ne doit pas dépendre de $a$.

$$\nabla_\theta V(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t (G_t - b(s_t)) \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right].$$

> [!warning] Pourquoi ça marche
> - **Réduction de variance** : on update proportionnellement à *combien on a fait mieux que prévu*, pas au $G_t$ brut.
> - **Pas de biais introduit** : $\mathbb{E}_\tau[b(s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t)] = 0$.

> [!note]- Preuve : pas de biais
> $$\begin{aligned}
> & \mathbb{E}_{\tau \sim \pi_\theta}[b(s_t) \nabla_\theta \log \pi_\theta(a_t \mid s_t)] \\
> &= \mathbb{E}_{s_t}[b(s_t) \mathbb{E}_{a_t}[\nabla_\theta \log \pi_\theta(a_t \mid s_t)]] \\
> &= \mathbb{E}_{s_t}\!\left[b(s_t) \sum_{a_t} \pi_\theta(a_t \mid s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}\right] \\
> &= \mathbb{E}_{s_t}[b(s_t) \nabla_\theta \sum_{a_t} \pi_\theta(a_t \mid s_t)] \\
> &= \mathbb{E}_{s_t}[b(s_t) \nabla_\theta 1] = 0.
> \end{aligned}$$

> [!warning] Définition (Advantage)
> $$\hat A_t = G_t^{(i)} - b(s_t).$$
>
> Choix naturel : $b(s_t) = V(s_t)$, donc $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$.

> [!note]- Pseudo-code Vanilla Policy Gradient
> 1. Initialiser $\theta$ et baseline $b(s)$ pour tout $s$.
> 2. Pour chaque itération :
>    - Collecter $m$ trajectoires sous $\pi_\theta$.
>    - Pour chaque $t$ de chaque $\tau^{(i)}$ :
>      - $G_t^{(i)} = \sum_{t'} r_{t'}$.
>      - $\hat A_t^{(i)} = G_t^{(i)} - b(s_t)$.
>    - Re-fitter la baseline en minimisant $\sum_i \sum_t \|b(s_t) - G_t^{(i)}\|^2$.
>    - Update :
>
>      $$\hat g = \sum_i \sum_t \hat A_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)}).$$
>
>      $\theta \leftarrow \theta + \alpha \hat g$ (ou Adam).

> 💡 **En pratique** on n'utilise pas $\hat g$ direct — on définit une loss combinée :
>
> $$L(\theta, \mathbf w) = \sum_t \big( \hat A_t \log \pi_\theta(a_t \mid s_t) - \|b(s_t) - G_t\|^2 \big).$$
>
> Et on lance Adam sur $\nabla_\theta L$ et $\nabla_\mathbf{w} L$.

### F. N-step estimators et bias-variance trade-off

> 💡 **Idée.** On peut blender MC et TD pour le calcul du target :
>
> | $k$ | Estimateur | Bias | Variance |
> | :---: | :--- | :---: | :---: |
> | 1 | $\hat G_t^{(1)} = r_t + \gamma V(s_{t+1})$ | Élevé | Faible |
> | 2 | $\hat G_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$ | Modéré | Modéré |
> | $\infty$ | $\hat G_t^{(\infty)} = r_t + \gamma r_{t+1} + \ldots$ | Zéro | Élevé |
>
> Et leurs versions advantage. Choisir $k$ intermédiaire pour le bon trade-off.

### G. Mini-projet — REINFORCE sur CartPole

> [!example] CartPole-v0 avec logistic regression policy
> Environnement : équilibrer un pendule sur un chariot. Observation 4D, actions binaires (gauche/droite). Reward +1 par step debout. Termine à +200 ou chute.

> [!note]- Setup
> ```python
> import gym
> env = gym.make('CartPole-v0')
> # observation_space = Box(4,), action_space = Discrete(2)
> ```

**Policy :** logistic regression simple plutôt qu'un NN.

$$\pi_\theta(0 \mid x) = \frac{1}{1 + e^{-\theta \cdot x}}, \quad \pi_\theta(1 \mid x) = 1 - \pi_\theta(0 \mid x).$$

**Score function dérivé manuellement :**

$$\nabla_\theta \log \pi_\theta(0 \mid x) = x - x \pi_\theta(0 \mid x), \quad \nabla_\theta \log \pi_\theta(1 \mid x) = -x \pi_\theta(0 \mid x).$$

> [!note]- Implémentation Python
> ```python
> import numpy as np
> 
> class LogisticPolicy:
>     def __init__(self, theta, alpha, gamma):
>         self.theta = theta; self.alpha = alpha; self.gamma = gamma
>     
>     def logistic(self, y):
>         return 1 / (1 + np.exp(-y))
>     
>     def probs(self, x):
>         y = x @ self.theta
>         prob0 = self.logistic(y)
>         return np.array([prob0, 1 - prob0])
>     
>     def act(self, x):
>         probs = self.probs(x)
>         action = np.random.choice([0, 1], p=probs)
>         return action, probs[action]
>     
>     def grad_log_p(self, x):
>         y = x @ self.theta
>         grad_log_p0 = x - x * self.logistic(y)
>         grad_log_p1 = -x * self.logistic(y)
>         return grad_log_p0, grad_log_p1
>     
>     def discount_rewards(self, rewards):
>         discounted = np.zeros(len(rewards))
>         cum = 0
>         for i in reversed(range(len(rewards))):
>             cum = cum * self.gamma + rewards[i]
>             discounted[i] = cum
>         return discounted
>     
>     def update(self, rewards, obs, actions):
>         grad_log_p = np.array([self.grad_log_p(o)[a] for o, a in zip(obs, actions)])
>         discounted = self.discount_rewards(rewards)
>         dot = grad_log_p.T @ discounted
>         self.theta += self.alpha * dot
> ```

**Résultat.** Après ~500 épisodes, l'agent commence à équilibrer 200 steps. Les courbes sont **très bruitées** typique du policy gradient.

![[algo-1.png]]

À l'évaluation, la policy finale obtient 200 sur tous les 100 épisodes test.

![[algo-2.png]]

> 💡 **Pour aller plus loin.** Variance reduction (plus d'épisodes par update), learning rate scheduling, multi-seed runs pour estimer la perf moyenne avec std.

### H. Trust Regions

*À développer.*

> 💡 **L'idée.** Le vanilla PG fait des updates non bornées sur $\theta$, ce qui peut **catastrophiquement dégrader** la policy. **TRPO** (Schulman et al. 2015) et son successeur **PPO** (Schulman et al. 2017) bornent l'update via une contrainte KL entre l'ancienne et la nouvelle policy. PPO en particulier est devenu le **standard de fait** en deep RL moderne (utilisé pour entraîner ChatGPT via RLHF).

---

## III. Actor-Critic (Hybride)

*À développer.*

> 💡 **L'idée.** Combiner le meilleur des deux mondes :
> - **Actor** = policy paramétrique $\pi_\theta$ (comme en policy gradient).
> - **Critic** = value function paramétrique $\hat V_\mathbf{w}$ ou $\hat Q_\mathbf{w}$ (comme en value-based).
>
> Le critic estime $V$ ou $Q$ pour calculer l'advantage en temps réel (au lieu de Monte Carlo). Cela réduit la variance et permet du **bootstrapping**, donc apprentissage online.
>
> **Algos clés** :
> - **A2C / A3C** (Advantage Actor-Critic, synchrone et asynchrone).
> - **DDPG** (Deep Deterministic Policy Gradient — actor-critic en action continue).
> - **TD3** (Twin Delayed DDPG — corrige l'overestimation de DDPG).
> - **SAC** (Soft Actor-Critic — entropy-regularized, état de l'art en continu).

---

## Annexe — récapitulatif des algorithmes

| Algo | Famille | Action space | Key feature |
| :--- | :--- | :--- | :--- |
| **MC VFA Linear** | Value-based | Discret/continu | Pas d'oracle, target = retour |
| **TD(0) VFA Linear** | Value-based | Discret/continu | Bootstrap, faible variance |
| **DQN** | Value-based | Discret | NN + replay + fixed targets |
| **Double DQN** | Value-based | Discret | Réduit overestimation |
| **Dueling DQN** | Value-based | Discret | Sépare $V$ et $A$ |
| **Behavioral Cloning** | Imitation | Tout | Supervised learning sur expert |
| **DAGGER** | Imitation | Tout | Aggregation itérative de données |
| **MaxEnt IRL** | Imitation | Tout | Récupère reward function |
| **REINFORCE** | Policy-based | Tout | MC policy gradient |
| **Vanilla PG + baseline** | Policy-based | Tout | Réduit la variance via $\hat A$ |
| **TRPO/PPO** | Policy-based | Tout | Trust region, état de l'art |
| **A2C/A3C** | Actor-Critic | Tout | Bootstrap pour réduire variance |
| **DDPG/TD3/SAC** | Actor-Critic | Continu | Actions continues |

> 💡 **Le résumé en une phrase.** Pour des **actions discrètes avec espace d'états visuel** (jeux Atari, images) → DQN/Double/Dueling. Pour des **actions continues** (robotique) → SAC ou PPO. Pour des **rewards sparses avec un expert disponible** → DAGGER ou IRL. Pour un **prototype rapide** → REINFORCE avec baseline.
