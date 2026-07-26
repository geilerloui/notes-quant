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

> [!info] Le fil rouge de toute la section E : Atari
> Jusqu'ici (B, C, D), l'état était un petit vecteur de quelques nombres (position, vitesse...), et les features étaient construites à la main. **DQN change d'échelle** : Mnih et al. (2015, *Nature*) l'appliquent aux **jeux Atari**, où l'état est une **image brute** (pixels) — impossible d'y construire des features à la main comme en B/C/D. C'est précisément ce qui motive le passage à un réseau de neurones profond : il apprend lui-même les features utiles directement depuis les pixels. Tout ce qui suit (E.1 à E.6) — architecture, preprocessing, entraînement, résultats — porte sur ce même papier et ce même jeu de benchmarks Atari, du début à la fin.

#### E.1 Contexte historique

> 💡 **Historique en bref.**
> - **1994** : TD-Backgammon (Tesauro), premier succès neural net en RL.
> - **1995-1998** : focus sur function approximation off-policy + bootstrapping (TD). **Beaucoup de divergence** → la communauté devient prudente.
> - **Mid-2000s** : renaissance avec Deep NN.
> - **2014** : DeepMind sort **DQN sur Atari** — révolution.

![[neural-1.png]]

**Pourquoi NN — limites du linéaire**

Comme dans le **XOR-world**, certaines value functions ne se modélisent pas linéairement. On peut ajouter des termes d'interaction :

$$\hat U(s; \mathbf w) = x_1 w_1 + x_2 w_2 + x_1 x_2 w_3 + w_4.$$

![[neural-2.png]]

Mais ça reste manuel. Mieux : un **NN profond** qui apprend les features automatiquement :

![[neural-3.png]]

> [!warning] Trois familles d'approximateurs
> - **(i) Linéaire** — combinaison pondérée de features. Marche bien si les features sont bien choisies, mais demande du feature engineering.
> - **(ii) Kernel-based** — plus riche, sans features explicites. Convergence sous certaines conditions, mais pas scalable.
> - **(iii) Deep NN** — universal approximator, distributed representations, exponentiellement moins de paramètres qu'un shallow net pour la même fonction. SGD pour apprendre.

#### E.3 Prétraitement des frames et architecture du réseau

> 💡 **Référence.** Mnih et al. 2015 (*Nature*). Réseau qui prend en entrée des images Atari préprocessées et sort un vecteur de Q-values, une par action.

![[dqn-5.png|483]]
**Figure.** La boucle agent-environnement classique, ici sur un jeu Atari (Space Invaders) : l'agent observe $s_t$ (l'écran), choisit une action $a_t$ (le joystick), reçoit une récompense $r_t$ (le score).

**Préprocessing — l'essentiel.** L'état donné au réseau, ce sont les **4 dernières frames de l'écran, empilées** comme 4 canaux d'une même entrée (fenêtre glissante : à chaque nouveau pas, on ajoute la frame la plus récente et on enlève la plus ancienne). Pourquoi 4 et pas 1 : une seule image ne dit pas dans quelle direction va la balle (Pong) ou un ennemi — deux situations visuellement identiques peuvent avoir des dynamiques opposées selon d'où vient l'objet. En empilant 4 frames successives, le réseau peut déduire vitesse et direction à partir des différences entre elles. Chaque frame est aussi convertie en **niveaux de gris** (on ne garde pas la couleur) et réduite en taille avant d'être empilée (détail d'implémentation, sans grand intérêt conceptuel).

Au final, l'entrée du réseau est un tensor $s \in \mathbb{R}^{4 \times 84 \times 84}$ — exactement le même rôle que le $\mathbf x(s)$ des sections B/C, sauf que ce ne sont plus des features construites à la main mais des pixels bruts empilés.

**Architecture — de $s$ à $\hat Q(s,\cdot)$.**

![[dqn-4.png]]
**Figure.** Schéma de l'architecture (version simplifiée du papier original) : stack de 4 frames → 2 couches convolutives → 1 couche fully-connected → sortie linéaire. La version Nature 2015, utilisée dans la suite de cette note, a une couche convolutive de plus et davantage de filtres (32, puis 64, puis 64 ; 512 unités en FC) — le principe est identique, seule la taille change.

C'est exactement la même construction que celle vue plus haut dans notre discussion : trois couches convolutives, puis on **aplatit** (*flatten*) en un vecteur, puis une couche fully-connected classique, puis une sortie linéaire :

$$h_1 = \text{ReLU}(W_1 * [s]_{4 \times 84 \times 84} + b_1), \quad h_2 = \text{ReLU}(W_2 * h_1 + b_2), \quad h_3 = \text{ReLU}(W_3 * h_2 + b_3),$$

$$\hat Q(s, \cdot\, ; \mathbf w) = W_5 \cdot \text{ReLU}\big(W_4 \cdot \text{flatten}(h_3) + b_4\big) + b_5,$$

où $*$ est la convolution (poids partagés, connectivité locale — cf. [[03_CNN]]), et $\mathbf w = \{W_1, \ldots, W_5, b_1, \ldots, b_5\}$ regroupe **tous** les filtres et poids du réseau (potentiellement des millions de paramètres, contre 5 à 10 en VFA linéaire). La sortie $\hat Q(s,\cdot;\mathbf w)$ est un **vecteur** de taille $|\mathcal A|$ — une Q-valeur par action, calculées toutes en un seul forward pass, sans jamais construire de feature vector par action à la main (contrairement au block stacking de C.1).


#### E.4 Loss, target et les deux innovations clés

> [!warning] Loss DQN
> $$J(\mathbf w) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1})}\big[(y_t^{DQN} - \hat q(s_t, a_t; \mathbf w))^2\big],$$
>
> avec target one-step :
>
> $$y_t^{DQN} = r_t + \gamma \max_{a'} \hat q(s_{t+1}, a'; \mathbf w^-),$$
>
> où $\mathbf w^-$ sont les paramètres du **target network** (fixés temporairement), et le target $y_t$ est traité comme **fixe** lors du SGD sur $\mathbf w$.

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

#### E.5 Pseudo-code DQN

> [!note] Algorithme complet
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

> 💡 **Quelques détails pratiques du papier original**, en complément du pseudo-code ci-dessus : reward clipping à $[-1, +1]$ (même learning rate sur tous les jeux, malgré des scores d'échelles très différentes) ; RMSProp, mini-batch de 32 ; $\epsilon$ décroît linéairement de $1.0$ à $0.1$ sur le premier million de steps puis reste fixé à $0.1$ ; à l'évaluation (une fois entraîné), $\epsilon = 0.05$.

#### E.6 Visualisations et résultats

*(Toujours le même papier/jeu de benchmarks Atari annoncé en ouverture de la section E — ces figures illustrent ce qu'on vient de construire, pas une nouvelle application.)*

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

### B. Pourquoi une stochastic policy est parfois meilleure

> 💡 **Le fil rouge de B.** Jusqu'ici (01, et I. ci-dessus), $\pi(s) = \arg\max_a Q(s,a)$ : une politique **déterministe**, une seule action par état. Les deux exemples ci-dessous montrent des cas où la politique optimale ne peut **pas** être déterministe — pas à cause de l'exploration, mais structurellement.

#### B.1 Rock-paper-scissors (environnement adversarial)

> [!info] Adversarial = un adversaire intelligent en face
> Contrairement à FrozenLake (où le seul "hasard" vient de la glace qui glisse), ici un **adversaire observe ta stratégie et s'adapte** pour te battre — un autre joueur, pas juste de l'aléatoire fixe.

Toute politique fixe non-uniforme est exploitable (l'adversaire devine le biais et joue le coup qui bat). La seule politique non-exploitable est uniformément aléatoire :

$$P(\text{rock}) = P(\text{paper}) = P(\text{scissors}) = \tfrac{1}{3}.$$

#### B.2 Aliased gridworld (environnement partiellement observable)

> [!info] Partiellement observable = l'agent ne voit pas l'état réel
> L'agent perçoit une **observation** (ici : les murs autour de lui), pas sa position exacte. Deux états différents peuvent produire la même observation — l'agent ne peut alors pas les distinguer ("aliasing"). Nom savant : **POMDP**.

> [!example] Couloir à 5 cases : squelette — case grise — 💰 — case grise — squelette
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im1.png]]
>
> Les deux cases grises ont la **même config de murs** → même observation, indistinguables pour l'agent.
>
> Une politique **déterministe** doit choisir la même action dans les deux (puisqu'elles sont perçues identiquement) — ici "toujours gauche" : correct pour la case grise de droite, mais envoie l'agent tout droit dans le squelette depuis la case grise de gauche :
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im2.png]]
>
> Une politique **stochastique** (gauche/droite 50/50 sur les cases grises) donne, depuis n'importe laquelle des deux, une bonne chance de partir dans la bonne direction — l'agent atteint le but avec haute probabilité au lieu d'être condamné dans l'un des deux cas :
>
> ![[images/3-Apprentissage automatique/07_Reinforcement learning/Deep RL/Policy based/im3 (1).png]]

> 💡 **Conclusion.** Dans un environnement adversarial ou partiellement observable, la politique optimale peut être *intrinsèquement* aléatoire — chose que seul le policy-based représente directement (le value-based ne produit qu'un $\arg\max$, donc du déterministe).

### C. Policy Gradient — REINFORCE

> [!info] D'où vient le nom
> REINFORCE (Williams, 1992) est un acronyme rétroactif : *"**RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einforcement × **C**haracteristic **E**ligibility"* — en gros, une vieille formulation de la règle de mise à jour qu'on va retrouver ci-dessous. Retiens juste : c'est le premier algo de policy gradient, celui qui sert de base à tous les autres (C.3).

**Le plan de C.** On veut ajuster $\theta$ (les poids de $\pi_\theta$) pour maximiser la performance de la politique — donc de la **descente/montée de gradient** sur un objectif $V(\theta)$. Le problème : $V(\theta)$ est une espérance sur des trajectoires, et $\theta$ influence *quelles trajectoires sont probables* — pas juste une valeur qu'on dérive normalement. C.1 construit le gradient malgré ça (l'astuce du *likelihood ratio*), C.2 le simplifie en exploitant la structure temporelle, C.3 en fait un algo utilisable — **avec, à chaque étape, le même exemple qui se concrétise au fur et à mesure : CartPole.**

> [!example] Le fil rouge de C : CartPole-v0
> Équilibrer un pendule sur un chariot. Observation $x \in \mathbb{R}^4$, 2 actions (gauche/droite), $+1$ de reward par step debout, épisode terminé à $200$ steps ou à la chute.
>
> Politique choisie : une simple **régression logistique** plutôt qu'un NN (assez pour ce problème) :
>
> $$\pi_\theta(0 \mid x) = \frac{1}{1 + e^{-\theta \cdot x}}, \qquad \pi_\theta(1 \mid x) = 1 - \pi_\theta(0 \mid x).$$
>
> On va calculer, avec ce $\pi_\theta$ précis, chaque quantité abstraite introduite en C.1/C.2/C.3.

#### C.1 Objectif et gradient

> [!warning] Objectif
> $$V(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \sum_\tau P(\tau; \theta) R(\tau)$$
>
> où $\tau = (s_0, a_0, r_0, \ldots, s_T)$ est une trajectoire. On veut $\theta^* = \arg\max_\theta V(\theta)$.

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

> [!example] Concrètement sur CartPole
> Le score function $\nabla_\theta \log \pi_\theta(a\mid x)$ se calcule à la main pour la régression logistique :
>
> $$\nabla_\theta \log \pi_\theta(0 \mid x) = x - x\,\pi_\theta(0 \mid x), \qquad \nabla_\theta \log \pi_\theta(1 \mid x) = -x\,\pi_\theta(0 \mid x).$$
>
> C'est exactement le $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$ de la formule ci-dessus — juste explicité pour ce $\pi_\theta$ précis, sans avoir besoin de $\mathbf P$ ni de $\mu(s_0)$.

> [!warning] Forme finale
> $$\boxed{\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(\tau^{(i)}) \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})}$$

#### C.2 Exploiter la structure temporelle

L'action à $t$ ne peut affecter que les **rewards à $t' \geq t$**, pas avant. En décomposant $R(\tau) = \sum_t r_t$ :

$$\nabla_\theta V(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T-1} r_{t'}\right] = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right],$$

où $G_t = \sum_{t' \geq t} r_{t'}$ est le **retour à partir de $t$**.

> [!warning] Forme REINFORCE finale
> $$\boxed{\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^{T-1} G_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})}$$

> [!example] Concrètement sur CartPole — calculer $G_t$
> Sur CartPole, chaque step debout rapporte $r_t = 1$. Épisode très court de 3 steps avant la chute, $\gamma = 0.9$ : $r_0 = r_1 = r_2 = 1$. En partant de la fin :
>
> $$G_2 = 1, \qquad G_1 = 1 + 0.9 \times 1 = 1.9, \qquad G_0 = 1 + 0.9 \times 1.9 = 2.71.$$
>
> C'est exactement le $G_t$ qu'on multiplie au score function dans la formule encadrée ci-dessus — plus l'épisode dure longtemps *après* $t$, plus $G_t$ est grand, donc plus la mise à jour pousse fort dans la direction des actions qui ont mené à un épisode long.

#### C.3 Pseudo-code REINFORCE

> [!note]- REINFORCE (Monte Carlo policy gradient)
> 1. Initialiser $\theta$ arbitrairement.
> 2. Pour chaque épisode $\{s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T\} \sim \pi_\theta$ :
>    - Pour $t = 1$ à $T - 1$ :
>      - $\theta \leftarrow \theta + \alpha \cdot G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.
> 3. Retourner $\theta$.

> [!example] Concrètement sur CartPole — toutes les pièces assemblées
> 1. $\theta$ initialisé à $0$ (ou aléatoire).
> 2. Jouer un épisode complet avec $\pi_\theta$ (à chaque pas : tirer $a \sim \pi_\theta(\cdot\mid x)$, cf. la policy logistique du début de C) → on récupère $x_0, a_0, r_0, \ldots, x_{T-1}, a_{T-1}, r_{T-1}$.
> 3. Calculer $G_t$ pour chaque $t$ avec `discount_rewards` (cf. C.2).
> 4. Calculer $\nabla_\theta \log \pi_\theta(a_t \mid x_t)$ pour chaque $t$ avec les deux formules de C.1 (selon que $a_t = 0$ ou $1$).
> 5. Mettre à jour : $\theta \leftarrow \theta + \alpha \sum_t G_t \, \nabla_\theta \log \pi_\theta(a_t \mid x_t)$ — exactement la forme REINFORCE finale de C.2, ligne par ligne.
> 6. Répéter depuis l'étape 2, épisode après épisode.
>
> **Résultat.** Après ~500 épisodes, l'agent tient les 200 steps. Les courbes d'entraînement sont très bruitées (typique du policy gradient, cf. E — c'est justement le problème que la baseline va résoudre) :
>
> ![[algo-1 (1).png]]
>
> À l'évaluation (politique figée), 200/200 sur les 100 épisodes de test :
>
> ![[algo-2.png]]
>
> 💡 **Pour aller plus loin** : plus d'épisodes par update (réduit la variance), learning rate scheduling, multi-seed runs pour estimer la perf moyenne avec écart-type.

### D. Classes de policies différentiables

| Action space             | Policy                                                                                       | Score function $\nabla_\theta \log \pi_\theta(a\mid s)$     |
| :----------------------- | :------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **Discret** (softmax)    | $\pi_\theta(a \mid s) = \dfrac{e^{\phi(s, a)^T \theta}}{\sum_{a'} e^{\phi(s, a')^T \theta}}$ | $\phi(s, a) - \mathbb{E}_{a' \sim \pi_\theta}[\phi(s, a')]$ |
| **Continu** (gaussienne) | $a \sim \mathcal{N}(\mu(s), \sigma^2), \; \mu(s) = \phi(s)^T \theta$                         | $\dfrac{(a - \mu(s)) \phi(s)}{\sigma^2}$                    |

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

| $k$ | Estimateur | Bias | Variance |
| :---: | :--- | :---: | :---: |
| 1 | $\hat G_t^{(1)} = r_t + \gamma V(s_{t+1})$ | Élevé | Faible |
| 2 | $\hat G_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$ | Modéré | Modéré |
| $\infty$ | $\hat G_t^{(\infty)} = r_t + \gamma r_{t+1} + \ldots$ | Zéro | Élevé |

Et leurs versions advantage. Choisir $k$ intermédiaire pour le bon trade-off.

### G. Trust Regions

*À développer.*

> 💡 **L'idée.** Le vanilla PG fait des updates non bornées sur $\theta$, ce qui peut **catastrophiquement dégrader** la policy. **TRPO** (Schulman et al. 2015) et son successeur **PPO** (Schulman et al. 2017) bornent l'update via une contrainte KL entre l'ancienne et la nouvelle policy. PPO en particulier est devenu le **standard de fait** en deep RL moderne (utilisé pour entraîner ChatGPT via RLHF).

---

## III. Actor-Critic (Hybride)

> 💡 **Le problème de fond.** En policy-based (II), on sait mettre à jour $\pi_\theta$ (le gradient), mais pour ça il faut d'abord *mesurer* si la politique est bonne — et on le fait avec $G_t$, un retour Monte Carlo bruité et coûteux (attendre la fin de l'épisode). En value-based (I), c'est l'inverse : on sait très bien mesurer une politique ($Q$, $V$), mais on ne sait pas en extraire directement une politique stochastique ou à actions continues. L'Actor-Critic combine les deux : un **acteur** ($\pi_\theta$) qui agit, un **critic** ($\hat q$ ou $\hat v$) qui le mesure en continu, à chaque pas. Cette section dérive l'algo pas à pas (A → D), en partant de REINFORCE.

### A. Le problème : REINFORCE dépend de $G_t$

> [!warning] Rappel — update REINFORCE
> $$\Delta \theta = \alpha \nabla_\theta\big(\log \pi(S_t, A_t, \theta)\big) \, R(\tau), \qquad R(\tau) = G_t = R_{t+1} + \gamma R_{t+2} + \ldots$$

$G_t$ est le retour Monte Carlo — il faut attendre la **fin de l'épisode** pour le calculer. Deux problèmes : ça ne marche que pour des tâches épisodiques, et ça empêche toute mise à jour en ligne, pas à pas.

### B. Q Actor-Critic — un critic appris en ligne à la place de $G_t$

**L'idée.** Remplacer $G_t$ par $\hat q(S_t, A_t; \mathbf w)$ — une estimation apprise, pas un retour observé :

> [!warning] Policy update (actor)
> $$\Delta \theta = \alpha \nabla_\theta\big(\log \pi(S_t, A_t, \theta)\big) \, \hat q(S_t, A_t, \mathbf w)$$

Reste à savoir d'où sort $\hat q$ : on le fait apprendre **en parallèle**, par TD (comme en I.B), avec son propre taux d'apprentissage $\beta$ :

> [!warning] Value update (critic)
> $$\Delta \mathbf w = \beta \Big( R_{t+1} + \gamma \hat q(S_{t+1}, A_{t+1}, \mathbf w) - \hat q(S_t, A_t, \mathbf w) \Big) \nabla_{\mathbf w} \hat q(S_t, A_t, \mathbf w)$$

![[Pasted image 20260726191933.png|593]]

$\theta$ et $\mathbf w$ sont deux jeux de paramètres différents, mais de même nature (deux approximateurs) : $\theta$ pour la policy (l'**acteur**, "quelle action jouer"), $\mathbf w$ pour $\hat q$ (le **critic**, "à quel point cette action était bonne").

![[Pasted image 20260726192249.png|301]]

**La boucle complète.** Au départ l'acteur est quasi-aléatoire. À chaque pas $t$ : l'acteur observe $S_t$, joue $A_t \sim \pi_\theta$ ; l'environnement renvoie $S_{t+1}, R_{t+1}$ ; le critic calcule $\hat q(S_t, A_t, \mathbf w)$ et sert cette valeur à l'acteur pour sa mise à jour ; le critic se met aussi à jour lui-même (TD, formule ci-dessus). **Tout se passe à chaque pas**, pas à la fin de l'épisode — c'est le gain direct par rapport à REINFORCE.

![[Pasted image 20260726192520.png|464]]

### C. Réduire la variance — l'Advantage

**Le problème.** $\hat q(s,a)$ varie beaucoup d'une action à l'autre — le signal utilisé pour l'update est bruité, l'apprentissage est instable.

**L'astuce.** Imaginons $Q(s,a)$ tiré d'une distribution centrée sur $V(s) = \mathbb{E}_\pi[Q(s,a)]$ (par définition de $V$, cf. I.A). Si on **soustrait $V(s)$** à $Q(s,a)$, la distribution résultante est centrée en $0$ — ça ne change rien en espérance, mais ça réduit la variance du signal :

![[Pasted image 20260726193140.png|471]]

> [!warning] Définition — Advantage
> $$A(s,a) = Q(s,a) - V(s)$$

**Pourquoi c'est le bon signal.** $Q(s,a)$ dit "combien je m'attends à gagner en jouant $a$ en $s$". $A(s,a)$ dit "combien je gagne **en plus**, par rapport à la moyenne des actions possibles en $s$" — exactement l'information utile pour savoir si $a$ vaut le coup d'être renforcée ou non.

### D. Le raccourci pratique — TD Actor-Critic

Utiliser $A(s,a)$ directement demanderait d'apprendre **deux** critics ($\hat q$ et $\hat v$) — cher, et redondant. Astuce : la **TD error**

$$\delta = R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf w) - \hat v(S_t, \mathbf w)$$

est déjà, en espérance, une bonne estimation de $A(s,a)$ — donc un seul critic ($\hat v$ seulement) suffit.

> [!warning] Forme finale — TD Actor-Critic
> $$\boxed{\Delta \theta = \alpha \nabla_\theta\big(\log \pi(S_t, A_t, \theta)\big) \underbrace{\Big(R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf w) - \hat v(S_t, \mathbf w)\Big)}_{\text{TD error } \delta \approx A(S_t,A_t)}}$$

C'est **l'algorithme final** — souvent appelé *"one-step Actor-Critic"* (Sutton & Barto). Ce n'est pas exactement A2C au sens strict (qui ajoute des retours n-step et plusieurs workers en parallèle), mais c'est le cœur conceptuel exact qu'A2C/A3C mettent à l'échelle.

> 💡 **Algos clés qui en découlent** :
> - **A2C / A3C** (Advantage Actor-Critic, synchrone et asynchrone — ce même algo + n-step + parallélisation).
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
| **REINFORCE** | Policy-based | Tout | MC policy gradient |
| **Vanilla PG + baseline** | Policy-based | Tout | Réduit la variance via $\hat A$ |
| **TRPO/PPO** | Policy-based | Tout | Trust region, état de l'art |
| **A2C/A3C** | Actor-Critic | Tout | Bootstrap pour réduire variance |
| **DDPG/TD3/SAC** | Actor-Critic | Continu | Actions continues |

> 💡 **Le résumé en une phrase.** Pour des **actions discrètes avec espace d'états visuel** (jeux Atari, images) → DQN/Double/Dueling. Pour des **actions continues** (robotique) → SAC ou PPO. Pour un **prototype rapide** → REINFORCE avec baseline. Pas de reward function du tout → [[03_Imitation Learning]].

---

## To do

> [!note] Points à approfondir plus tard
> - **Graphes de convergence de $\mathbf w$.** Générer un vrai graphique (code, pas un placeholder) qui trace l'évolution de $\mathbf w$ (ou d'une composante, ou de la performance/durée d'épisode) au fil des épisodes d'entraînement, pour visualiser concrètement la convergence.
> - **Comparer MC, SARSA et Q-learning (SARSAMAX) sur CartPole.** Un même graphe (ou une petite série de graphes) qui compare les trois méthodes côte à côte sur le même environnement — par exemple durée d'épisode moyenne en fonction du nombre d'épisodes d'entraînement, pour voir laquelle converge le plus vite et le plus stablement.
> - **Prudence de SARSA vs. agressivité de Q-learning.** Illustrer concrètement sur CartPole (pas seulement l'exemple cliff walking de 01) que Q-learning, en évaluant la politique gloutonne plutôt que la politique réellement suivie, peut apprendre une politique plus "risquée" qui se comporte mal pendant l'exploration — alors que SARSA, en tenant compte du coût de l'exploration ($\varepsilon$-greedy), apprend une politique plus prudente. Point déjà discuté en 01 (SARSA vs Q-learning, section IV.B) mais jamais illustré numériquement sur CartPole.
