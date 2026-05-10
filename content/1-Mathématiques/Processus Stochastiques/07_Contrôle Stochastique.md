---
title: Contrôle Stochastique
date: 2026-05-10
tags: [probabilités, processus-stochastiques, contrôle, optimisation, finance, énergie]
---

## L'idée fondatrice

Le **contrôle stochastique** étudie comment **piloter un système qui évolue de façon aléatoire**, en prenant des décisions au fil du temps pour maximiser un gain espéré (ou minimiser un coût espéré). C'est l'analogue stochastique du contrôle optimal classique (Pontryagin, Bellman) que tu as peut-être croisé en optimisation déterministe.

Concrètement, tu as :
- Un système dont l'état $X_t$ évolue selon une **EDS contrôlée** : tu modifies son drift et/ou sa diffusion via une variable de décision $\alpha_t$
- Un **objectif** à optimiser sur l'horizon $[0, T]$ : typiquement une espérance d'utilité, de coût, de payoff
- Tu cherches la **politique** $\alpha^*(t, x)$ optimale qui dépend de l'état courant — c'est le **contrôle feedback**

L'outil central : l'équation de **Hamilton-Jacobi-Bellman (HJB)**, une EDP qui donne directement la politique optimale. Ce cours fait le pont entre tes notes [[03_Équations Différentielles Stochastiques]], [[05_Reverse-time SDE et Fokker-Planck]] et le monde de l'optimisation.

> [!example] Domaines d'application
> Ce qu'on va voir s'applique à des problèmes très différents qui partagent **le même formalisme** :
> - **Finance** : allocation de portefeuille (Merton), optimal execution (Almgren-Chriss), pricing avec contraintes
> - **Énergie / smart grids** : pilotage de batteries, demand response, intégration de renouvelables sous incertitude (météo, consommation)
> - **Robotique / véhicules autonomes** : planification de trajectoire sous bruit capteur
> - **ML moderne** : reinforcement learning continu, RL stochastique
> - **Logistique** : gestion de stocks avec demande aléatoire

---

## I. Le problème générique

### I.1 EDS contrôlée

> [!warning] Définition — EDS contrôlée
> On appelle **EDS contrôlée** une équation de la forme :
> 
> $$dX_t = \mu(t, X_t, \alpha_t)\,dt + \sigma(t, X_t, \alpha_t)\,dW_t$$
> 
> où :
> - $X_t \in \mathbb{R}^d$ est l'**état** du système à l'instant $t$
> - $\alpha_t \in \mathcal{A}$ est le **contrôle** (la décision) à l'instant $t$, dans un ensemble $\mathcal{A}$
> - $\mu, \sigma$ dépendent de $\alpha$ — c'est ça qui distingue d'une EDS classique

Le contrôle $\alpha_t$ peut être :
- **En boucle ouverte** : $\alpha_t$ ne dépend que du temps (politique préprogrammée)
- **En boucle fermée / feedback** : $\alpha_t = \alpha(t, X_t)$ — dépend de l'état observé. **C'est presque toujours mieux** parce que ça réagit aux fluctuations stochastiques.

### I.2 Visualisation : contrôler = modifier l'EDS

![[fig1_setup.png]]
*Figure 1. Trois politiques de contrôle sur la même EDS $dX_t = \alpha\,dt + 0.5\,dW_t$ avec $X_0 = 1$. **Gauche** — pas de contrôle ($\alpha = 0$) : le système diffuse librement. **Milieu** — contrôle constant ($\alpha = -0.5$) : le drift constant tire les trajectoires vers le bas, peu importe leur position. **Droite** — contrôle feedback ($\alpha = -2x$) : le drift dépend de l'état, ramène vers 0 plus fort quand on est loin. Les fluctuations sont rapidement amorties. Même bruit utilisé pour les 3 panneaux pour comparer équitablement.*

### I.3 L'objectif à optimiser

> [!warning] Définition — Critère d'objectif
> Sur l'horizon $[0, T]$, on cherche à maximiser le **gain total espéré** :
> 
> $$J(\alpha) = \mathbb{E}\left[\int_0^T f(t, X_t, \alpha_t)\,dt + g(X_T)\right]$$
> 
> où :
> - $f(t, x, \alpha)$ est le **gain courant** (running reward) — gagné à chaque instant
> - $g(x)$ est le **gain terminal** (terminal reward) — gagné à la fin
> 
> On cherche $\alpha^* = \arg\max_\alpha J(\alpha)$.

> [!example] Exemples concrets
> - **Merton** : $f = 0$, $g(W) = U(W)$ utilité de la richesse finale
> - **Optimal execution** : $f = -\eta v_t^2 - \lambda\sigma^2 X_t^2$ (impact instantané + risque), $g = 0$
> - **Smart grid** : $f$ = coût électricité dépensée + bonus stabilité, $g$ = coût final batterie
> - **RL** : $f$ = reward instantané, $g$ = reward terminal d'épisode

---

## II. L'équation de Hamilton-Jacobi-Bellman (HJB)

### II.1 La fonction valeur

> [!warning] Définition — Fonction valeur
> La **fonction valeur** $V : [0, T] \times \mathbb{R}^d \to \mathbb{R}$ est définie par :
> 
> $$V(t, x) = \sup_{\alpha} \mathbb{E}\left[\int_t^T f(s, X_s, \alpha_s)\,ds + g(X_T) \mid X_t = x\right]$$
> 
> C'est **le meilleur gain espéré** atteignable en partant de l'état $x$ à l'instant $t$.

À $t = 0$, $V(0, x_0)$ est le gain optimal du problème entier. À $t = T$, $V(T, x) = g(x)$ (condition terminale).

### II.2 L'équation HJB

> [!warning] Théorème — Équation HJB
> Sous des hypothèses de régularité, $V$ satisfait l'EDP :
> 
> $$\partial_t V + \sup_{\alpha \in \mathcal{A}} \left\{ \mu(t, x, \alpha) \cdot \nabla_x V + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2_x V) + f(t, x, \alpha) \right\} = 0$$
> 
> avec condition terminale $V(T, x) = g(x)$. Le **contrôle optimal feedback** est :
> 
> $$\alpha^*(t, x) = \arg\max_\alpha \left\{ \mu(t, x, \alpha) \cdot \nabla_x V + \frac{1}{2}\text{tr}(\sigma\sigma^T \nabla^2_x V) + f(t, x, \alpha) \right\}$$

> [!note]- Idée de la dérivation (principe de Bellman)
> Le **principe de Bellman** dit : pour être optimal sur $[t, T]$, il faut être optimal sur $[t + dt, T]$ partant de la position où on arrive. Formellement :
> 
> $$V(t, x) = \sup_\alpha \mathbb{E}\left[ f(t, x, \alpha)\,dt + V(t+dt, X_{t+dt}) \mid X_t = x \right]$$
> 
> Par formule d'Itô (cf [[02_Calcul d'Itô]]) :
> $$V(t+dt, X_{t+dt}) \approx V(t, x) + \partial_t V\,dt + \mu \cdot \nabla_x V\,dt + \frac{1}{2}\sigma^2 \nabla^2_x V\,dt + \sigma \nabla_x V\,dW_t$$
> 
> Le terme en $dW$ disparaît dans l'espérance (martingalité). En soustrayant $V(t,x)$ et en divisant par $dt$, on obtient HJB.

### II.3 Le contrôle optimal est feedback

C'est un fait fondamental : **le contrôle optimal s'exprime comme une fonction de l'état $X_t$**, pas du temps seul. Tu n'as pas besoin de "savoir le futur" — il suffit de regarder où tu es et de réagir. C'est ce qui rend HJB pratique : tu calcules $V$ une fois, et tu obtiens une politique $\alpha^*(t, x)$ qui te dit quoi faire dans toute situation.

---

## III. Exemple résolu : le problème de Merton

L'exemple historique (Merton 1969). Tu as un cash sans risque (taux $r$) et un actif risqué :

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

À chaque instant, tu choisis quelle **fraction $\alpha_t$ de ta richesse** investir dans l'actif risqué. La richesse $W_t$ évolue selon :

$$dW_t = W_t \left[ (r + \alpha_t(\mu - r))\,dt + \alpha_t \sigma\,dW_t \right]$$

Tu veux maximiser $\mathbb{E}[U(W_T)]$ pour une utilité **CRRA** (Constant Relative Risk Aversion) :

$$U(W) = \frac{W^{1-\gamma}}{1 - \gamma}$$

où $\gamma > 0$ est l'**aversion au risque** ($\gamma = 1$ donne le log-utilité).

> [!note]- Résolution par HJB
> **Étape 1 : ansatz.** On cherche $V(t, w) = h(t) \cdot \frac{w^{1-\gamma}}{1 - \gamma}$ par homogénéité du problème.
> 
> **Étape 2 : substitution dans HJB.** L'équation devient une EDO scalaire en $h(t)$ :
> $$h'(t) + h(t) \cdot \rho^* = 0, \quad h(T) = 1$$
> où $\rho^*$ est une constante optimale qu'on calcule en maximisant le sup sur $\alpha$.
> 
> **Étape 3 : maximisation explicite.** Le sup sur $\alpha$ dans HJB donne, après calcul de la dérivée :
> $$\boxed{\alpha^* = \frac{\mu - r}{\sigma^2 \gamma}}$$
> 
> C'est le **ratio de Merton**. Indépendant de $t$ et de $w$ — politique constante.

### III.1 Lecture du ratio de Merton

$$\alpha^* = \frac{\mu - r}{\sigma^2 \gamma}$$

- **Plus le rendement excédentaire $\mu - r$ est grand, plus on investit dans le risqué** (logique)
- **Plus la volatilité $\sigma$ est grande, moins on investit** (logique : risque)
- **Plus l'aversion $\gamma$ est grande, moins on investit** (logique : peureux)

C'est la **pierre angulaire de l'allocation optimale en finance théorique**.

![[fig2_merton.png]]
*Figure 2. **Gauche** — trajectoires de richesse pour 3 allocations. En vert, l'optimal Merton. En bleu, sous-investissement (peu de risque mais peu de rendement). En rouge, sur-investissement : rendement attendu plus élevé mais variance qui explose, et beaucoup de trajectoires terminent **plus bas** que l'optimal. **Droite** — l'utilité espérée $\mathbb{E}[U(W_T)]$ en fonction de l'allocation $\alpha$ : le maximum est atteint exactement en $\alpha^*$ Merton, calculé analytiquement. Paramètres : $\mu = 10\%$, $r = 2\%$, $\sigma = 20\%$, $\gamma = 3$, $T = 5$ ans.*

---

## IV. Optimal execution : le modèle d'Almgren-Chriss

Un classique du trading algorithmique (Almgren-Chriss 2000). Tu dois liquider une position de $X_0$ shares en $T$ minutes (ou heures, ou jours).

### IV.1 Le dilemme

- **Vendre vite** : forte impact de marché, tu déprécies ton propre prix de vente
- **Vendre lentement** : exposition prolongée au risque de prix (la valeur peut dériver pendant que tu attends)

> [!warning] Formulation
> L'état $X_t$ = nombre de shares restantes, le contrôle $v_t = -dX_t/dt$ = vitesse de vente. Coût total :
> 
> $$J = \mathbb{E}\left[\int_0^T (\eta\, v_t^2 + \lambda \sigma^2 X_t^2)\,dt\right]$$
> 
> où $\eta\, v_t^2$ = impact temporaire (vendre plus vite coûte plus cher) et $\lambda\sigma^2 X_t^2$ = pénalité pour exposition au risque.

### IV.2 La solution HJB

Sous hypothèses standard, l'équation HJB se résout analytiquement et donne le **schedule optimal** :

$$X^*_t = X_0 \cdot \frac{\sinh(\kappa(T - t))}{\sinh(\kappa T)}, \quad \kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$$

C'est une **décroissance exponentielle** (à peu près) qui dépend de $\kappa$ — l'arbitrage entre impact ($\eta$) et risque ($\lambda\sigma^2$).

![[fig3_execution.png]]
*Figure 3. **Gauche** — trois schedules de liquidation. En rouge, le **naïf** : on vend tout en 10% du temps, fort impact d'exécution. En bleu, **uniforme** : décroissance linéaire, simple mais pas optimal. En vert, **Almgren-Chriss** : décroissance exponentielle issue de HJB, balance impact et risque. **Droite** — coût total selon l'aversion au risque $\lambda$. L'optimal HJB **domine systématiquement**. Plus $\lambda$ est grand (vendeur risk-averse), plus le schedule optimal accélère vers le début pour réduire l'exposition.*

### IV.3 Pourquoi c'est important en finance

C'est le sujet **central des desks d'execution** chez Goldman, Citadel, Optiver, etc. Quand un fonds veut bouger 100 millions de dollars sur une action sans déstabiliser le marché, c'est ce type de calcul qui guide l'algorithme. Les variantes modernes (impact non-linéaire, signal d'alpha intra-journalier, contraintes de liquidité) sont toutes des extensions de Almgren-Chriss.

---

## V. La dualité HJB ↔ Fokker-Planck

C'est une connexion profonde qui boucle ce dossier sur les processus stochastiques.

### V.1 Le même générateur, deux questions différentes

Pour une diffusion $dX_t = \mu(t, x)\,dt + \sigma(t, x)\,dW_t$, le **générateur infinitésimal** est l'opérateur :

$$\mathcal{L} = \mu \cdot \nabla_x + \frac{1}{2}\sigma^2 \nabla^2_x$$

Cet opérateur apparaît **dans deux EDP fondamentales** mais qui décrivent des choses différentes :

> [!warning] Dualité HJB / Fokker-Planck
> - **Fokker-Planck (forward)** : $\partial_t p_t = \mathcal{L}^* p_t$ (cf [[05_Reverse-time SDE et Fokker-Planck]])
>   - Décrit l'évolution **forward dans le temps** de la **densité** $p_t(x)$
>   - Question : "où est la masse de probabilité à l'instant $t$ ?"
>   - Condition initiale : $p_0$ donnée
>   
> - **HJB (backward)** : $\partial_t V + \sup_\alpha \{\mathcal{L}^\alpha V + f\} = 0$
>   - Décrit l'évolution **backward dans le temps** de la **fonction valeur** $V(t, x)$
>   - Question : "quel est le meilleur gain espéré partant de $x$ à $t$ ?"
>   - Condition terminale : $V(T, x) = g(x)$

Les deux EDP partagent **le même opérateur**, mais agissent sur des objets différents et dans des sens du temps différents. C'est la **dualité Fokker-Planck ↔ HJB** au sens fonctionnel.

![[fig4_duality.png]]
*Figure 4. La dualité conceptuelle entre Fokker-Planck (évolution de la densité, forward) et HJB (évolution de la fonction valeur, backward). Les deux EDP utilisent **le même générateur infinitésimal** $\mathcal{L}$ — l'opérateur qui caractérise localement la diffusion. Fokker-Planck répond à "comment se distribue la probabilité ?", HJB répond à "quelle est la stratégie optimale ?". Applications respectives en bas : sampling et diffusion models pour FP ; finance, énergie, RL pour HJB.*

### V.2 Pourquoi cette dualité est belle

C'est l'aboutissement du formalisme : le **même opérateur** $\mathcal{L}$ que tu as vu dans Fokker-Planck pour étudier l'évolution des densités te sert maintenant à résoudre des problèmes d'optimisation. Une fois que tu maîtrises le générateur d'une diffusion, tu as les deux faces de la médaille.

---

## VI. Lien avec le Reinforcement Learning

Le **RL moderne** (Q-learning, DQN, PPO, etc.) est essentiellement du contrôle stochastique discret.

### VI.1 La correspondance

| Contrôle stochastique continu | Reinforcement Learning |
|---|---|
| État $X_t \in \mathbb{R}^d$ | État $s$ (continu ou discret) |
| Contrôle $\alpha_t$ | Action $a$ |
| EDS $dX = \mu\,dt + \sigma\,dW$ | Transition $P(s' \mid s, a)$ |
| Fonction valeur $V(t, x)$ | Value function $V^\pi(s)$ |
| HJB | Équation de Bellman |
| Contrôle optimal $\alpha^*(t, x)$ | Policy $\pi^*(s)$ |
| Sup sur $\alpha$ | $\max_a$ |

### VI.2 RL = HJB approchée

Le Q-learning peut être vu comme une **approximation discrète de l'équation HJB** :

$$Q(s, a) \approx r(s, a) + \gamma\,\mathbb{E}_{s'}[\max_{a'} Q(s', a')]$$

C'est l'analogue de :

$$V(t, x) = \sup_\alpha \{f(t, x, \alpha)\,dt + V(t+dt, X_{t+dt})\}$$

en passant à la limite continue.

> [!example] Applications RL en finance et énergie
> - **Market making** : optimal bid/ask sous flux d'ordres aléatoires (Avellaneda-Stoikov 2008 + variantes RL)
> - **Trading systématique** : politiques d'allocation dynamiques apprises sur historique
> - **Smart grids** : pilotage de batteries / véhicules électriques pour minimiser coût d'électricité sous prix variable et consommation aléatoire
> - **Demand response** : réguler la charge réseau via incitations dynamiques

---

## VII. Pour aller plus loin

**Contrôle avec contraintes** : si $\alpha$ doit rester dans un compact (positions limitées, contraintes de risque), HJB devient une **inégalité variationnelle**. Cas typique : pricing d'options américaines.

**Contrôle de type impulse** : décisions discrètes ponctuelles plutôt que continues (genre rebalancement de portefeuille à coûts fixes). HJB devient un système d'inéquations quasi-variationnelles.

**Contrôle robuste** : pour quand on ne fait pas confiance au modèle, on optimise contre le pire cas. Lié à la théorie des jeux (Isaacs 1965).

**Mean-field control** : contrôle d'une grande population d'agents qui interagissent. Lié aux mean-field games (Lasry-Lions 2007). Applications : marchés financiers à plusieurs traders, gestion de flotte de véhicules.

**Stochastic differential games** : deux contrôleurs qui s'opposent. HJB devient l'équation d'Isaacs-Bellman.

**Solutions de viscosité** : quand $V$ n'est pas régulière (souvent le cas), on utilise le formalisme des solutions de viscosité (Crandall-Lions 1983) pour donner sens à HJB. C'est le cadre moderne pour la théorie du contrôle stochastique.

---

## Récapitulatif

| Concept | Définition / formule |
|---|---|
| EDS contrôlée | $dX_t = \mu(t, X_t, \alpha_t)\,dt + \sigma(t, X_t, \alpha_t)\,dW_t$ |
| Fonction valeur | $V(t, x) = \sup_\alpha \mathbb{E}[\int_t^T f + g(X_T) \mid X_t = x]$ |
| Équation HJB | $\partial_t V + \sup_\alpha\{\mathcal{L}^\alpha V + f\} = 0$, $V(T, x) = g(x)$ |
| Contrôle optimal feedback | $\alpha^*(t, x) = \arg\max_\alpha\{\ldots\}$ |
| Ratio de Merton | $\alpha^* = (\mu - r)/(\sigma^2 \gamma)$ |
| Schedule Almgren-Chriss | $X^*_t = X_0 \sinh(\kappa(T-t))/\sinh(\kappa T)$ |
| Dualité FP / HJB | Même générateur $\mathcal{L}$, FP forward sur densité, HJB backward sur valeur |
| Lien RL | Bellman discret = HJB discrétisée, $Q$ = approximation de $V$ |

---

## Pour aller plus loin

**Références** :
- Pham (2009) — *Continuous-time Stochastic Control and Optimization with Financial Applications*
- Fleming-Soner (2006) — *Controlled Markov Processes and Viscosity Solutions*
- Touzi (2013) — *Optimal Stochastic Control, Stochastic Target Problems, and Backward SDE*
- Cartea-Jaimungal-Penalva (2015) — *Algorithmic and High-Frequency Trading* (très orienté pratique)
- Sutton-Barto (2018) — *Reinforcement Learning: An Introduction* (pour le pendant ML)

**Dans le vault** :
- Prérequis : [[01_Mouvement Brownien]], [[02_Calcul d'Itô]], [[03_Équations Différentielles Stochastiques]]
- Dualité avec : [[05_Reverse-time SDE et Fokker-Planck]]
- Lien sampling : [[04_Dynamique de Langevin]] (autre application du générateur)

---

## Suite logique

**Précédent ← [[06_Rough Paths]]** : on a vu comment étendre Itô aux chemins très irréguliers. Ici on revient au cadre standard ($H = 1/2$) mais on s'intéresse à l'optimisation plutôt qu'à l'intégration.

**Suivant → [[08_Volterra Signatures]]** : (placeholder) extension de la signature pour le feature engineering ML sur séries temporelles à mémoire longue.

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
7. **[[07_Contrôle Stochastique]]** — (cette note) HJB, Merton, optimal execution, lien RL
8. [[08_Volterra Signatures]] — (placeholder) extension signature pour mémoire longue
