---
title: Imitation Learning
---
# Imitation Learning

> Apprendre depuis des **rewards sparse** est lent et risqué (autonomous driving, médical). Une alternative : apprendre par **imitation** d'un expert. Contrairement à [[01_RL Tabulaire]] et [[02_Function Approximation (Deep RL)]], on ne suppose même pas l'existence d'une reward function — c'est un paradigme à part, pas une variante de value-based/policy-based/actor-critic.

> Pré-requis : [[01_RL Tabulaire]] (MDP, politique $\pi$).

---

## I. Setup

> [!warning] Learning from Demonstration
> On dispose de :
> - State space, action space.
> - Transition model $P(s' \mid s, a)$.
> - **Pas de reward function** $R$.
> - Set de trajectoires expertes $(s_0, a_0, s_1, a_1, \ldots)$ avec actions de la policy experte $\pi^*$.

## II. Behavioral Cloning

> 💡 **Idée la plus simple.** Apprendre $\pi$ par **supervised learning** sur les paires $\{(s_i, a_i)\}$. Exemple historique : ALVINN (Pomerleau 1989, sensors → steering angle).

> [!example] Concrètement — FrozenLake, en reprenant le fil rouge de [[01_RL Tabulaire]]
> On enregistre un expert (la politique quasi-optimale $\pi_*$) en train de jouer, et on note chaque paire (état, action jouée). Reprenons la trajectoire déjà utilisée dans 01 : $0\to4\to4\to8\to9\to13\to14\to15$.
>
> | $t$ | $s_t$ | $a_t^*$ (action de l'expert) |
> |:---:|:---:|:---:|
> | 0 | 0 | Bas |
> | 1 | 4 | Bas |
> | 2 | 4 | Bas |
> | 3 | 8 | Droite |
> | 4 | 9 | Bas |
> | 5 | 13 | Droite |
> | 6 | 14 | Droite |
>
> Ce tableau, c'est **exactement un dataset supervisé classique** : $s_t$ = la feature (l'état, ou son one-hot encoding), $a_t^*$ = le label à prédire. Aucune reward, aucun $\gamma$, aucun $Q$ n'intervient : uniquement "prédire la bonne action à partir de l'état".
>
> **Le classifieur, concrètement.** On prend la softmax policy de [[02_Function Approximation (Deep RL)]] II.D : $\pi_\theta(a\mid s) = \dfrac{e^{\phi(s,a)^T\theta}}{\sum_{a'} e^{\phi(s,a')^T\theta}}$, avec $\phi(s,a)$ le one-hot de $(s,a)$ (comme le block-stacking de la même note, I.C.1) — un score $\theta_{s,a}$ par case et par action, $16\times4$ paramètres au total. C'est un classifieur multiclasse tout ce qu'il y a de plus standard (4 classes = 4 actions), sauf que la feature d'entrée est un état de grille plutôt qu'une image.
>
> **La loss : cross-entropy**, exactement comme en classification d'images :
>
> $$L(\theta) = -\frac{1}{N}\sum_{t=1}^N \log \pi_\theta(a_t^* \mid s_t).$$
>
> **Un pas de descente de gradient, à la main.** Avant tout entraînement, $\theta = 0$ partout → $\pi_\theta(a\mid s) = \frac14$ pour les 4 actions, quel que soit $s$ (softmax uniforme). Prenons la première ligne du tableau, $(s_0=0,\, a_0^*=\text{Bas})$ :
>
> $$-\nabla_\theta \log \pi_\theta(\text{Bas}\mid 0) = -\big(\phi(0,\text{Bas}) - \mathbb{E}_{a'\sim\pi_\theta}[\phi(0,a')]\big)$$
>
> (la score function de la softmax, déjà vue en 02.D). Avec $\pi_\theta$ encore uniforme, $\mathbb{E}_{a'}[\phi(0,a')] = \frac14$ sur les 4 actions de l'état $0$ et $0$ ailleurs. Après un pas de taille $\alpha$ : $\theta_{0,\text{Bas}}$ augmente, $\theta_{0,\text{Gauche}}, \theta_{0,\text{Droite}}, \theta_{0,\text{Haut}}$ diminuent légèrement — donc $\pi_\theta(\text{Bas}\mid 0)$ devient $> \frac14$ : le classifieur vient d'apprendre, sur ce seul exemple, à préférer un peu plus "Bas" en $s=0$. On répète ça sur toutes les lignes du tableau, sur plusieurs passes (epochs) — exactement comme on entraînerait un classifieur d'images sur un dataset labellisé.
>
> **Le piège (compounding errors, ci-dessous) concrètement** : $\pi_*$ n'est jamais passée par l'état $6$ ou $10$ dans cette trajectoire (elle a évité les trous). Si $\pi_\theta$ mal entraînée pousse l'agent vers $6$ (à côté du trou $7$) par erreur, il n'existe **aucune ligne dans le tableau** pour lui dire quoi faire depuis $6$ — il improvise, et improviser à côté d'un trou finit mal.

> [!warning] Limite — compounding errors, l'analogie la plus simple
> C'est comme entraîner une **régression linéaire par interpolation** (elle n'a vu des données que sur une plage précise), puis un jour lui demander une prédiction en **extrapolation**, bien en dehors de cette plage : elle va quand même sortir un nombre, mais rien ne garantit qu'il soit correct — le modèle n'a simplement jamais rien appris sur cette zone.
>
> Ici, c'est pareil : $\pi_\theta$ n'a "interpolé" que sur les états visités par l'expert (les data ne sont **pas i.i.d.** dans l'espace d'états — concentrées sur les trajectoires expertes). Dès que l'agent dévie vers un état jamais vu, $\pi_\theta$ "extrapole" une action, sans aucune garantie.
>
> **La différence avec une régression classique : ça s'auto-aggrave.** Une mauvaise extrapolation ponctuelle ne dégrade pas les extrapolations futures. Ici si, car le résultat de l'action de $\pi_\theta$ devient le *prochain* état d'entrée : une extrapolation ratée pousse l'agent encore plus loin de la zone connue, où l'extrapolation suivante est encore pire, etc. Cet effet boule de neige est précisément ce qui fait que **l'erreur scale en $T^2$** sur la longueur de l'épisode (vs linéaire en RL standard, où une erreur ne se nourrit pas d'elle-même).

## III. DAGGER (Dataset Aggregation)

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

## IV. Inverse Reinforcement Learning (IRL)

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

## V. Apprenticeship Learning via IRL

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

## VI. Maximum Entropy IRL

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

---

## Récapitulatif des algorithmes

| Algo | Famille | Action space | Key feature |
| :--- | :--- | :--- | :--- |
| **Behavioral Cloning** | Imitation | Tout | Supervised learning sur expert |
| **DAGGER** | Imitation | Tout | Aggregation itérative de données |
| **MaxEnt IRL** | Imitation | Tout | Récupère reward function |

> 💡 **En une phrase.** Rewards sparses avec un expert disponible → DAGGER (si l'expert peut labelliser en temps réel) ou IRL (si on veut aussi récupérer la reward function elle-même).
