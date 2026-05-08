---
title: Multi-agents (RL)
---
# Multi-agents (RL)

> Quand plusieurs agents interagissent dans le même environnement, on quitte le cadre MDP classique : chaque agent doit prendre en compte les actions des autres. Le cadre formel s'appelle **stochastic game** (ou **Markov game**) — c'est l'analogue multi-agent du MDP. Cette note couvre la généralisation du Q-learning aux jeux stochastiques (zero-sum, general-sum), les concepts d'équilibre (Nash, correlated), et les valeurs coco pour les jeux coopératifs-compétitifs.

> Pré-requis : [[RL Tabulaire]] (MDP, Bellman, Q-learning, value iteration), notions de théorie des jeux (Nash equilibrium).

> 💡 **L'analogie centrale** :
>
> $$\text{MDP} : \text{RL} :: \text{Stochastic Game} : \text{Multi-agent RL}.$$

---

## I. Introduction

### A. Le shift conceptuel

Stochastic games (jeux de Shapley) = généralisation à la fois du **MDP** et des **repeated games**. Ils donnent un modèle formel pour le multi-agent RL.

### B. Exemple — Grid 3×3 à deux joueurs

> [!example] Course au dollar
> Grille $3 \times 3$ avec joueurs A et B. Chacun peut aller N, S, E, O ou rester. Transitions déterministes sauf :
> - **Murs épais** = passage impossible.
> - **Semi-murs** = si A va au nord, 50% il monte, 50% il reste.
>
> ![[images/3-Apprentissage automatique/Reinforcement learning/multi-agents/im1.png|325]]
>
> Le but : atteindre `$` pour gagner $100. **Premier arrivé gagne**. Si les deux arrivent en même temps, pile ou face.

> [!warning] Analyse stratégique
> - **Si A ignore B** : meilleure stratégie = E puis N. Symétriquement pour B.
> - **Mais ils ne peuvent pas occuper la même case** → conflit.
> - **Stratégie coopérative** (les deux passent par le semi-mur) :
>   - 25% les deux passent → conflit, pile ou face.
>   - 25% personne ne passe.
>   - 50% un seul passe → il gagne.
> - **Si A passe par le semi-mur et pas B** : A gagne 2/3 du temps, B gagne 2/3 du temps (asymétrie due au choix).

> 💡 **Multiple Nash equilibria.** Il y a un Nash où B prend le centre, un autre où A prend le centre. Pas d'équilibre unique → pierre angulaire de la difficulté du multi-agent.

---

## II. Stochastic Games — formalisme

### A. Définition

> [!warning] Définition (Stochastic Game)
> Un stochastic game est un tuple $\langle S, A_i, T, R_i, \gamma \rangle$ :
>
> - **$S$** : ensemble des états $s$.
> - **$A_i$** : actions du joueur $i$. On focus sur deux joueurs : $a \in A_1$, $b \in A_2$.
> - **$T$** : transitions $T(s, (a, b), s')$ — proba d'atteindre $s'$ depuis $s$ après l'**action jointe** $(a, b)$ (les joueurs agissent simultanément).
> - **$R_i$** : reward function du joueur $i$, $R_1(s, (a, b))$ et $R_2(s, (a, b))$.
> - **$\gamma$** : discount factor.

> 💡 **Note historique.** Shapley a publié les stochastic games **avant** que Bellman publie les MDPs. Donc techniquement, MDP est un cas particulier (mono-agent) de stochastic game, pas l'inverse.

---

## III. Zero-sum stochastic games

### A. Le challenge

Ce qui rend les stochastic games plus intéressants que les repeated games : l'action courante affecte non seulement le reward mais aussi les **futurs états**. Même problème qu'en MDP → on définit une **value function**.

### B. Bellman optimiste (max-max)

Première tentative pour étendre Bellman :

$$Q_i^*(s, (a, b)) = R_i(s, (a, b)) + \gamma \sum_{s'} T(s, (a, b), s') \, \max_{a', b'} Q_i^*(s', (a', b')).$$

> 💡 **Lecture.** Le $\max$ sur la joint action $(a', b')$ rend le joueur **optimiste** : on suppose qu'il y aura toujours une action jointe favorable. Ça ne reflète pas la dynamique adversariale.

### C. Minimax-Q (Littman 1994)

> [!warning] Pour zero-sum
> Pour les jeux à somme nulle (deux joueurs aux intérêts opposés), on remplace $\max$ par **minimax** :
>
> $$\boxed{Q_i^*(s, (a, b)) = R_i(s, (a, b)) + \gamma \sum_{s'} T(s, (a, b), s') \, \underset{a', b'}{\min\max} \, Q_i^*(s', (a', b'))}$$

Forme online (Q-learning) :

$$\langle s, (a, b), (r_1, r_2), s' \rangle : \quad Q_i(s, (a, b)) \xleftarrow{\alpha} r_i + \gamma \, \underset{a', b'}{\min\max} \, Q_i(s', (a', b')).$$

> [!warning] Propriétés de minimax-Q
> - **Convergence** sous les mêmes conditions que Q-learning standard.
> - **$Q^*$ unique** — value iteration converge vers une seule solution.
> - **Policies indépendantes** — si chaque joueur tourne minimax-Q sans coordination, ils convergent quand même vers des minimax-optimal policies.
> - Mais **pas linéaire** comme Q-learning (résoudre le minimax intérieur demande de la programmation linéaire).

> 💡 **Pourquoi ça marche.** En zero-sum à deux joueurs, il y a toujours une **valeur unique** du jeu (théorème de von Neumann). Pas d'ambiguïté → convergence.

---

## IV. General-sum games

### A. Nash-Q

> 💡 **Le problème.** En general-sum, minimax n'a plus de sens. Au lieu de minimax, on calcule un **équilibre de Nash** des Q-values :
>
> $$Q_i^*(s, (a, b)) = R_i(s, (a, b)) + \gamma \sum_{s'} T(s, (a, b), s') \, \underset{a', b'}{\text{Nash}} \, Q_i^*(s', (a', b')).$$

Forme online :

$$\langle s, (a, b), (r_1, r_2), s' \rangle : \quad Q_i(s, (a, b)) \xleftarrow{\alpha} r_i + \gamma \, \underset{a', b'}{\text{Nash}} \, Q_i(s', (a', b')).$$

### B. Pourquoi Nash-Q ne marche pas bien

> [!warning] Cinq problèmes
> - **Value iteration ne converge pas** sur ce système.
> - **Pas de $Q^*$ unique** — multiples Nash equilibria possibles.
> - **Policies non indépendantes** — les joueurs doivent se coordonner.
> - **Update non efficace** — sauf si P = PPAD (improbable, PPAD est proche de NP).
> - **Q functions insuffisantes** pour spécifier la policy (à cause de la multiplicité des équilibres).

> 💡 **Conclusion.** Le general-sum est **fondamentalement difficile**. C'est pour ça qu'on cherche d'autres concepts de solution.

---

## V. Solution concepts

### A. Vue d'ensemble

> 💡 **Définition informelle.** Un *solution concept* est une règle pour prédire comment le jeu sera joué. Le plus connu : **Nash equilibrium**. D'autres existent :
> - **Correlated equilibrium** (Aumann) — souvent meilleur en pratique, on l'utilise tout le temps en vrai.
> - **Trembling hand equilibrium** (Selten) — robuste aux petites erreurs.
> - **Stackelberg equilibrium** — leader-follower.
> - **Subgame perfect** — chaque sous-jeu doit être un équilibre.

### B. Correlated equilibrium — exemple General Tsu Chicken

> [!example] Le jeu du chicken
> Deux voitures s'affrontent. Actions : **Chicken** (dévier) ou **Dare** (tenir bon). Si les deux Dare → crash.
>
> ![[correlated-1.png]]

**Stratégie mixte symétrique** où Dare est joué avec proba $1/3$ :

| | Probabilité | Payoff joueur 1 |
| :--- | :---: | :---: |
| (C, C) | $2/3 \times 2/3 = 4/9$ | 6 |
| (C, D) | $2/3 \times 1/3 = 2/9$ | 2 |
| (D, C) | $1/3 \times 2/3 = 2/9$ | 7 |
| (D, D) | $1/3 \times 1/3 = 1/9$ | 0 |

Espérance : $\frac{1}{9}(4 \cdot 6 + 2 \cdot 2 + 2 \cdot 7 + 1 \cdot 0) = \frac{42}{9} \approx 4.67$.

> 💡 **L'idée du Correlated Equilibrium.** Et si on faisait mieux avec un **médiateur** ?

### C. Le mediator de Chris

> [!example] Correlated GTC avec médiateur
> On introduit un tiers (Chris) avec **3 cartes** : (C, C), (D, C), (C, D). **Pas de (D, D)**. Chris tire une carte uniformément, et chuchote à chaque joueur **uniquement son action** (sans dévoiler celle de l'autre).
>
> ![[correlated-2.png]]
>
> **Pourquoi c'est un équilibre :**
> - Si Chris dit "C" au joueur 1, le joueur 1 sait que Chris a tiré soit (C, C) soit (C, D), donc le joueur 2 fait C ou D avec proba 1/2 chacune.
> - Le joueur 1 préfère-t-il dévier vers D ? Calcul d'utilité montre que non.
> - Idem pour "D".

> [!warning] Trois faits sur les correlated equilibria (CE)
> - **Calculables en temps polynomial** (vs Nash qui est PPAD-complete).
> - **Tout Nash mixte est un CE** → les CE existent toujours.
> - **Toute combinaison convexe de Nash mixtes est un CE** → l'ensemble des CE est convexe.

> 💡 **Pourquoi c'est utile.** Les CE sont **plus larges** et **plus faciles à calculer** que Nash. En applications réelles (mécanismes d'enchères, traffic routing), on utilise souvent des CE implicitement via signalisation publique.

---

## VI. Valeurs Coco (Cooperative-Competitive)

### A. Motivation — le jeu des bananes

> [!example] Curly et Smooth
> Curly est trop petit pour atteindre les bananes. Smooth peut prendre 2 bananes seul. **Coopération** : ils peuvent obtenir 4 bananes au total.
>
> ![[images/3-Apprentissage automatique/Reinforcement learning/multi-agents/im2.png]]

**Analyse Nash :**
- NE 1 : "don't boost - reach" (Smooth se sert seul, prend 2).
- NE 2 : "boost - climb" (coopération, 4 bananes).

NE 2 est **plus bénéfique au total** mais Curly ne reçoit rien sans transfert.

### B. L'idée des side payments

> 💡 **Coco values** (Kalai & Kalai). On autorise des **transferts entre joueurs** (un joueur peut "donner une banane" à l'autre). Le calcul se fait en deux étapes :
> 1. Trouver l'action jointe maximisant la **somme des payoffs** (cooperative).
> 2. Redistribuer équitablement via une part **competitive** (zero-sum).

### C. Définition formelle

> [!warning] Coco value
> Soit $U$ le payoff de "moi" et $\bar U$ celui de l'autre joueur :
>
> $$\boxed{\text{coco}(U, \bar U) = \underset{\text{max max}}{\underbrace{\max\!\left(\frac{U + \bar U}{2}\right)}} + \underset{\text{minimax}}{\underbrace{\text{minimax}\!\left(\frac{U - \bar U}{2}\right)}}}$$

> 💡 **Lecture.**
> - Premier terme : **partie coopérative** — somme totale qu'on peut maximiser ensemble, divisée en deux. Solvable par max-max.
> - Second terme : **partie compétitive** (zero-sum) — qu'est-ce que je peux extraire de l'autre quand on est en désaccord ? Solvable par programmation linéaire (minimax classique).
>
> Les deux termes étant tractables, **le coco est calculable** — contrairement aux Nash de general-sum.

### D. Exemple chiffré — banane

$$U = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}, \quad \bar U = \begin{pmatrix} 2 & 0 \\ 2 & 4 \end{pmatrix}.$$

> [!note]- Calcul des matrices coopérative et compétitive
> $$\frac{U + \bar U}{2} = \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix}, \quad \frac{U - \bar U}{2} = \begin{pmatrix} -1 & 0 \\ -1 & -2 \end{pmatrix}, \quad \frac{\bar U - U}{2} = \begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix}.$$

**Résultat :**
- Coco value pour $U$ (Curly) : 1.
- Coco value pour $\bar U$ (Smooth) : 3.

**Side payments :**
- $p = \text{coco}(U, \bar U) - U(a^*, \bar a^*) = 1$.
- $\bar p = \text{coco}(\bar U, U) - \bar U(a^*, \bar a^*) = -1$.

> 💡 **Interprétation.** Smooth récupère 4 bananes via coopération, en donne 1 à Curly. Smooth net = 3, Curly net = 1. C'est l'**unique répartition équitable** au sens coco.

### E. Propriétés

> [!warning] Cinq propriétés des coco values
> - **Calculables efficacement** (polynomial time).
> - **Maximisent l'utilité totale**.
> - **Décomposent le jeu** en somme d'un coopératif + compétitif.
> - **Coco unique** (contrairement à Nash multiples).
> - **Généralisables aux stochastic games** via l'algorithme **Coco-Q**.

> [!warning] Limite majeure
> Les coco values **ne se généralisent pas** au-delà de **deux joueurs**. C'est leur principal défaut.

### F. Coco-Q

*À développer.*

> 💡 **Mini-orientation.** Coco-Q applique l'idée coco à un stochastic game : à chaque step, les joueurs jouent l'action jointe coco-optimale et échangent des side payments. Convergence prouvée à deux joueurs. Vs minimax-Q qui suppose adversarial pur, coco-Q permet de capturer la coopération partielle.

---

## VII. Récapitulatif

| Cadre | Concept de solution | Algorithme | Convergence | Tractable |
| :--- | :--- | :--- | :---: | :---: |
| **MDP (mono-agent)** | Optimal policy | Q-learning, VI | ✓ | ✓ (linéaire) |
| **Zero-sum stochastic game** | Minimax value | Minimax-Q | ✓ | ✓ (LP) |
| **General-sum stochastic game** | Nash equilibrium | Nash-Q | ✗ | ✗ (PPAD) |
| **General-sum, mediator OK** | Correlated equilibrium | LP-based | ✓ | ✓ (poly) |
| **2-player coop-comp** | Coco value | Coco-Q | ✓ (2 joueurs) | ✓ |

> 💡 **Le résumé en une phrase.** En multi-agent RL, la difficulté vient de la **multiplicité des équilibres** et de la **non-stationnarité induite par les autres apprenants**. Pour 2 joueurs zero-sum, **minimax-Q est propre et convergent**. Pour 2 joueurs general-sum, **Nash-Q est inutilisable** mais **CE et coco-Q** offrent des alternatives tractables. Au-delà de 2 joueurs, c'est largement un domaine de recherche ouvert.

> 💡 **Lien finance quant.** Le multi-agent RL est utilisé en :
> - **Market making** — agents qui apprennent à coter des prix face à d'autres market makers.
> - **Optimal execution** — un trader qui exécute un ordre face à d'autres traders qui peuvent le détecter et front-run.
> - **Algo trading** — modélisation de la dynamique des order books comme un jeu stochastique multi-agents (cf. les papers de Buehler, Cartea, Jaimungal).
>
> Pas un standard industriel encore, mais c'est un sujet de R&D actif chez les hedge funds quant les plus avancés (CFM, Two Sigma).
