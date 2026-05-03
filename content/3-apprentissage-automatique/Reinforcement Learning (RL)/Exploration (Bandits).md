---
title: Exploration et Bandits manchots
---
# Exploration et Bandits manchots

> Cette partie traite *séparément* du compromis exploration/exploitation, qui est central en RL mais qu'on a jusqu'ici contourné par des heuristiques (ε-greedy + GLIE). Pour comprendre le problème en lui-même, on l'isole dans un cadre dégénéré : un MDP réduit à un seul état, le **Multi-Armed Bandit** (MAB). Une fois les algorithmes maîtrisés sur ce cas pur (UCB, Thompson Sampling), on les remontera vers le cadre MDP général en fin de chapitre.

## I. Introduction

**Le problème.** Dans tout ce qu'on a vu jusqu'ici (DP, MC, TD), l'exploration était traitée comme un détail technique : on rajoutait un ε-greedy par-dessus l'algorithme principal, et on espérait que GLIE ferait converger les choses. Mais en pratique, sur des applications réelles — santé, robotique, finance — la **vitesse** d'apprentissage compte autant que la convergence asymptotique. Une stratégie qui converge vers $\pi_*$ après 10 ans d'apprentissage est inutile si on veut trader demain matin.

**Exploration vs exploitation.** Le compromis est fondamental :

- **Exploitation** : prendre la meilleure action selon ce qu'on sait actuellement. Maximise le gain immédiat.
- **Exploration** : prendre une action sous-optimale *à dessein*, pour récolter de l'information sur sa distribution. Sacrifie du gain à court terme pour mieux décider plus tard.

> 💡 **Pourquoi un chapitre dédié.** Les algorithmes RL classiques (Q-learning, SARSA…) explorent **mal** : ε-greedy explore *uniformément* au hasard, sans tenir compte de quelles actions sont incertaines. Un bon algorithme d'exploration explore **stratégiquement** — il alloue son budget d'exploration aux actions où il y a le plus à apprendre. C'est tout l'objet de ce chapitre.

> [!example] Fil rouge — 3 stratégies de trading
> Tu es PM dans un fonds quant. Trois stratégies systématiques sont candidates pour recevoir ton capital :
> 
> - **A : Momentum**
> - **B : Mean Reversion**
> - **C : Carry**
> 
> Chaque stratégie produit un PnL journalier (en bps) tiré d'une distribution **inconnue de toi**. Chaque jour pendant 300 jours, tu choisis une seule stratégie, tu observes son PnL, et tu mets à jour tes croyances. **Tu n'observes pas le PnL des stratégies que tu n'as pas choisies** (c'est le point clé du MAB : pas de contrefactuel gratuit). Objectif : maximiser le PnL cumulé.
> 
> **Vérité cachée** (que toi rédacteur connais pour faire les calculs, mais que l'agent ignore complètement) :
> 
> $$\mu_A = 8\,\text{bps},\quad \mu_B = 10\,\text{bps},\quad \mu_C = 5\,\text{bps}.$$
> 
> **Oracle.** Si on connaissait les $\mu_i$ d'avance, on prendrait toujours B → PnL cumulé sur 300 jours $= 300 \times 10 = 3000$ bps. C'est le **maximum** que l'agent essaie d'approcher. Le **regret** mesure l'écart à cet oracle.
> 
> **Pourquoi c'est un MAB et pas un MDP.** Il n'y a pas d'état — chaque jour est indépendant, choisir A aujourd'hui ne change pas la distribution de C demain. Donc pas de notion de "récompense différée", pas de transitions à apprendre. Juste : 3 distributions inconnues, comment les explorer efficacement ?

## II. Multi-Armed Bandits

### A. Cadre formel

> [!warning] Multi-Armed Bandit
> Un **MAB** est un couple $(\mathcal{A}, \mathcal{R})$ :
> 
> - $\mathcal{A}$ : ensemble fini d'**actions** (les "bras" de la machine — à tirer le levier de la machine à sous).
> - $\mathcal{R}$ : collection de **distributions de récompense**, une par action : $\mathcal{R}^a(r) = \mathbb{P}(r \mid a)$.
> 
> À chaque pas $t$, l'agent choisit une action $a_t \in \mathcal{A}$ et reçoit une récompense $r_t \sim \mathcal{R}^{a_t}$. Objectif : maximiser $\sum_t r_t$.

Notations utiles, qui reviendront partout :

- $Q(a) = \mathbb{E}[r \mid a]$ : la **vraie valeur** de l'action $a$ (espérance de la récompense). Inconnue.
- $\hat Q_t(a)$ : notre **estimation** de $Q(a)$ à l'instant $t$.
- $N_t(a)$ : le **nombre de fois** où l'action $a$ a été choisie jusqu'à $t$.
- $a^* = \arg\max_a Q(a)$ : la meilleure action (oracle).
- $V^* = Q(a^*)$ : la valeur optimale.

**Estimation de $Q(a)$ par moyenne empirique.** Sans modèle de la récompense, on l'estime par la moyenne des récompenses observées en jouant $a$ :

$$\hat Q_t(a) = \frac{1}{N_t(a)} \sum_{\tau = 1}^{t} r_\tau \cdot \mathbb{1}\{a_\tau = a\}.$$

Cette moyenne se calcule **incrémentalement** sans stocker l'historique :

$$\hat Q_t(a) = \hat Q_{t-1}(a) + \frac{1}{N_t(a)}\big( r_t - \hat Q_{t-1}(a) \big).$$

> [!note]- Dérivation de la moyenne incrémentale
> Vue déjà en III.A des notes RL Tabulaire (avec les retours $G_t$ comme observations) : si $\mu_n = \frac{1}{n}\sum_{k=1}^n x_k$, alors
> 
> $$\mu_n = \frac{1}{n}\big(x_n + (n-1)\mu_{n-1}\big) = \mu_{n-1} + \frac{1}{n}(x_n - \mu_{n-1}).$$
> 
> On retrouve la **forme universelle** du RL : $\text{nouvelle} \leftarrow \text{ancienne} + \alpha(\text{cible} - \text{ancienne})$, avec $\alpha = 1/N$.

### B. Stratégies naïves

On part des heuristiques les plus simples pour comprendre où elles échouent. Cela motivera UCB et Thompson Sampling.

#### Greedy

> [!warning] Greedy
> $$a_t = \arg\max_a \hat Q_t(a).$$
> 
> On joue toujours l'action qui a la meilleure estimation actuelle.

**Problème.** Greedy peut **se verrouiller** sur une action sous-optimale. Si la première récompense de la vraie meilleure action est mauvaise (par malchance), on l'abandonne pour toujours et on n'a aucune chance de corriger l'estimation.

#### ε-greedy et ε-greedy décroissant

> [!warning] ε-greedy
> Avec probabilité $1 - \varepsilon$, on joue $\arg\max_a \hat Q_t(a)$ ; avec probabilité $\varepsilon$, on tire une action **uniformément au hasard**.

C'est exactement la même politique qu'on a utilisée en III.B (MC Control) et IV.B (TD Control). Avec $\varepsilon$ constant, on continue à explorer indéfiniment — donc on accumule du regret linéairement même après avoir identifié la meilleure action. La parade : faire **décroître $\varepsilon$ avec le temps** (ε-greedy décroissant), par exemple $\varepsilon_t = 1/t$. C'est la condition GLIE qu'on connaît.

#### Optimistic Initialization

Une alternative astucieuse : **initialiser $\hat Q_0(a)$ à une valeur volontairement très grande**, plus grande que toute récompense possible. Du coup, dès qu'on essaie une action, son estimation **diminue** (parce que la vraie récompense est plus petite que l'estimation gonflée). Les actions non testées gardent leur grande estimation initiale, donc elles attirent l'agent — qui les essaie naturellement.

> [!warning] Optimistic Initialization
> Initialiser $\hat Q_0(a) = Q_{\max}$ pour toute $a$, avec $Q_{\max} \gg Q(a^*)$. Combiner avec une politique purement greedy.
> 
> *L'optimisme force l'exploration sans avoir besoin d'$\varepsilon$.*

> 💡 **L'image à retenir.** Optimistic init = "donner sa chance à toutes les actions". L'agent explore parce qu'il *croit* (à tort) que les actions non testées sont géniales — et il le découvre seulement en les essayant. C'est élégant, ça marche bien quand on connaît à peu près l'échelle des récompenses, mais c'est sensible à la valeur initiale.

> [!example] Fil rouge — comparaison des 3 stratégies naïves
> 
> Pour rendre tangible ce que coûtent ces heuristiques, on compare le PnL cumulé sur 300 jours.
> 
> **Setup rappelé** : $\mu_A = 8$, $\mu_B = 10$, $\mu_C = 5$ bps. Oracle = 3000 bps.
> 
> | Stratégie | Mécanisme | PnL cumulé attendu | Regret |
> |---|:---|:---:|:---:|
> | **Pur exploration** (uniforme) | 100 jours par stratégie | $100 \cdot 8 + 100 \cdot 10 + 100 \cdot 5 = 2300$ | $700$ |
> | **Pur exploitation** (3 jours d'essai puis greedy) | Risque de se verrouiller sur A ou C par malchance | $\sim 2400$–$2900$ selon les premières observations | $100$–$600$ |
> | **ε-greedy** ($\varepsilon = 0.1$) | 30 jours d'exploration uniforme + 270 jours sur la meilleure (en moyenne B) | $\approx 30 \cdot \bar\mu + 270 \cdot 10 = 30 \cdot 7.67 + 2700 \approx 2930$ | $\approx 70$ |
> 
> où $\bar\mu = (8 + 10 + 5)/3 = 7.67$.
> 
> **Lecture.** Pure exploration est très mauvaise (on ignore complètement ce qu'on apprend). Pure exploitation est bonne en moyenne mais a une **variance énorme** : si par malchance les 3 jours d'essai donnent une mauvaise observation pour B, on finit collé à A ou C pour 297 jours. ε-greedy équilibre les deux et gagne en pratique.
> 
> Mais aucune de ces 3 stratégies ne fait mieux que **regret linéaire en $T$** : leur regret croît proportionnellement au nombre de jours. UCB et Thompson Sampling, qu'on va voir, atteignent du **regret logarithmique** — un saut qualitatif énorme.

### C. Régret

Pour comparer les algorithmes proprement, il faut une métrique. La métrique standard dans la littérature MAB est le **regret**.

> [!warning] Regret
> Plusieurs quantités liées :
> 
> - **Gap** d'une action : $\Delta_a = V^* - Q(a) \geq 0$. Mesure de combien $a$ est sous-optimale.
> - **Regret instantané** au pas $t$ : $\ell_t = \mathbb{E}[V^* - Q(a_t)]$.
> - **Regret total** sur $T$ pas : $L_T = \mathbb{E}\Big[\sum_{t=1}^T (V^* - Q(a_t))\Big]$.

**Maximiser la récompense $\Leftrightarrow$ minimiser le regret total.** En effet,

$$L_T = T \cdot V^* - \mathbb{E}\Big[\sum_t Q(a_t)\Big],$$

donc minimiser $L_T$ revient à maximiser le PnL cumulé attendu (le terme $T \cdot V^*$ étant fixé par l'oracle).

**Décomposition fondamentale.** Si $\bar N_t(a) = \mathbb{E}[N_t(a)]$ est le nombre attendu de fois où on a joué $a$,

$$L_T \;=\; \sum_{a \in \mathcal{A}} \bar N_T(a) \cdot \Delta_a.$$

> 💡 **L'intuition cruciale.** Le regret est une **somme pondérée** : on additionne, pour chaque action sous-optimale, combien de fois on l'a jouée multiplié par à quel point elle est mauvaise. Donc un bon algorithme doit faire en sorte que $\bar N_T(a)$ soit **petit pour les grands $\Delta_a$** (peu jouer les actions très mauvaises) et **plus grand pour les petits $\Delta_a$** (on peut se permettre de jouer souvent les actions presque optimales). Le hic : les $\Delta_a$ ne sont pas connus d'avance — il faut les apprendre en jouant.

#### Regret linéaire vs sublinéaire

| Algorithme | Regret asymptotique |
|---|:---:|
| Greedy | $L_T = \Theta(T)$ — linéaire (peut se verrouiller) |
| ε-greedy constant | $L_T = \Theta(T)$ — linéaire (explore toujours $\varepsilon$) |
| ε-greedy décroissant | sublinéaire **si schedule bien choisi** |
| Optimistic Init | sublinéaire si $Q_{\max}$ assez optimiste |
| **UCB1** | $L_T = O(\log T)$ |
| **Thompson Sampling** | $L_T = O(\log T)$ |

**Sublinéaire = bon, linéaire = mauvais.** Un regret linéaire signifie qu'on perd un montant *constant* à chaque pas, indéfiniment. Un regret logarithmique signifie qu'à long terme, on joue presque toujours $a^*$ et le regret cesse pratiquement de croître.

#### Borne inférieure de Lai et Robbins

Existe-t-il une limite théorique au regret qu'aucun algorithme ne peut battre ? Oui — et elle est logarithmique.

> [!warning] Théorème (Lai & Robbins, 1985)
> Pour tout algorithme MAB *consistant*, le regret total a une borne inférieure asymptotique
> 
> $$\liminf_{T \to \infty} \frac{L_T}{\log T} \;\geq\; \sum_{a \mid \Delta_a > 0} \frac{\Delta_a}{\mathrm{KL}(\mathcal{R}^a \,\|\, \mathcal{R}^{a^*})},$$
> 
> où $\mathrm{KL}$ est la divergence de Kullback-Leibler entre la distribution de l'action $a$ et celle de l'action optimale.

> 💡 **Lecture.** Le regret minimum dépend de **deux choses** : la taille des gaps ($\Delta_a$) et la **séparabilité** des distributions ($\mathrm{KL}$). Un problème est dur quand les actions sous-optimales ont des distributions *proches* de celle de l'action optimale (KL petit) — il faut beaucoup d'échantillons pour les distinguer. Un problème est facile quand $a^*$ a une distribution clairement séparée des autres.
> 
> Cette borne est **atteinte** (à constante multiplicative près) par UCB1 et Thompson Sampling — les deux sont donc **asymptotiquement optimaux**.

### D. UCB1 — Optimisme face à l'incertitude

**Principe.** Au lieu de jouer l'action avec la meilleure *moyenne empirique*, on joue celle avec la meilleure **borne supérieure de confiance** sur sa vraie valeur. Une action peu jouée a une grande incertitude, donc une grande borne, donc on a envie de la jouer pour réduire l'incertitude. Une action très jouée a une petite incertitude, donc sa borne est proche de sa moyenne — elle ne sera choisie que si sa moyenne est élevée.

> 💡 **L'image à retenir.** *"Optimism in the face of uncertainty"*. On donne le bénéfice du doute aux actions sous-explorées. Soit elles sont bonnes (et on le découvre), soit elles ne le sont pas (et on l'apprend). Dans les deux cas on gagne de l'information.

**Construction.** On veut une borne $\hat U_t(a)$ telle que $Q(a) \leq \hat Q_t(a) + \hat U_t(a)$ avec haute probabilité, puis on choisit

$$a_t = \arg\max_a \big[\,\hat Q_t(a) + \hat U_t(a)\,\big].$$

Pour calculer $\hat U_t(a)$, on utilise une inégalité de concentration.

> [!warning] Inégalité de Hoeffding
> Soient $X_1, \ldots, X_n$ des variables i.i.d. dans $[0, 1]$, $\bar X_n = \frac{1}{n}\sum X_i$. Alors pour tout $u > 0$,
> 
> $$\mathbb{P}\big[\,\mathbb{E}[X] > \bar X_n + u\,\big] \leq \exp(-2 n u^2).$$
> 
> *Probabilité que la vraie moyenne dépasse l'empirique de plus de $u$ décroît exponentiellement en $n$.*

**Application au MAB.** Pour chaque action $a$, on a $N_t(a)$ observations de récompense, dont la moyenne est $\hat Q_t(a)$. Hoeffding donne

$$\mathbb{P}\big[\,Q(a) > \hat Q_t(a) + U_t(a)\,\big] \leq \exp\big(-2 N_t(a) \, U_t(a)^2\big).$$

On fixe une probabilité $p$ que la vraie valeur dépasse la borne, et on inverse :

$$\exp(-2 N_t(a) U_t(a)^2) = p \quad \Longleftrightarrow \quad U_t(a) = \sqrt{\frac{-\log p}{2 N_t(a)}}.$$

**Choix de $p = t^{-4}$** (qui décroît en $t$ pour resserrer la borne au fur et à mesure) : $-\log p = 4 \log t$, et on obtient l'algorithme **UCB1**.

> [!warning] UCB1
> $$a_t = \arg\max_{a \in \mathcal{A}} \;\Bigg\{ \hat Q_t(a) + \sqrt{\frac{2 \log t}{N_t(a)}} \;\Bigg\}.$$
> 
> Le premier terme est l'**exploitation** (moyenne empirique), le second est l'**exploration** (bonus pour les actions peu visitées).

> [!note]- Lecture du bonus d'exploration
> - $N_t(a)$ petit (action peu jouée) → bonus grand → on est attiré vers $a$.
> - $N_t(a)$ grand (action bien explorée) → bonus petit → on décide essentiellement sur la moyenne empirique.
> - $\log t$ croît avec $t$ → même les actions déjà jouées beaucoup voient leur bonus remonter lentement, donc on continue à les ré-essayer occasionnellement. Cela évite que l'algorithme se verrouille.

> [!warning] Théorème (Auer, Cesa-Bianchi, Fischer, 2002)
> UCB1 atteint un regret total
> 
> $$L_T \;\leq\; 8 \log T \sum_{a \mid \Delta_a > 0} \Delta_a + O(1).$$
> 
> Donc $L_T = O(\log T)$ — **regret logarithmique**, asymptotiquement optimal au sens de Lai-Robbins (à constante près).

> [!example] Fil rouge — UCB1 sur les 3 stratégies, pas par pas
> 
> **Setup.** $\gamma$ ne joue pas ici (pas de discount en MAB). On lance UCB1 avec $\hat Q_0(a) = 0$, $N_0(a) = 0$ pour les 3 stratégies. Quand $N_t(a) = 0$, le bonus est infini (par convention) — donc UCB1 commence par jouer chaque stratégie une fois.
> 
> **Pas 1-3 : initialisation forcée.** L'agent joue A, B, C à tour de rôle. Récompenses observées (tirées avec un peu de bruit autour des vraies moyennes — comme en pratique) :
> 
> | $t$ | $a_t$ | $r_t$ | $N_t(A)$ | $N_t(B)$ | $N_t(C)$ | $\hat Q(A)$ | $\hat Q(B)$ | $\hat Q(C)$ |
> |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
> | 1 | A | 6 | 1 | 0 | 0 | 6 | 0 | 0 |
> | 2 | B | 12 | 1 | 1 | 0 | 6 | 12 | 0 |
> | 3 | C | 7 | 1 | 1 | 1 | 6 | 12 | 7 |
> 
> *(Observations bruitées autour de $\mu_A = 8, \mu_B = 10, \mu_C = 5$.)*
> 
> **Pas 4 : premier choix UCB1.** Le bonus pour chaque action est $\sqrt{2 \log 4 / 1} = \sqrt{2.77} \approx 1.665$. Indices UCB :
> 
> $$\text{UCB}(A) = 6 + 1.665 = 7.665,\quad \text{UCB}(B) = 12 + 1.665 = 13.665,\quad \text{UCB}(C) = 7 + 1.665 = 8.665.$$
> 
> Argmax : **B**. On joue B. Disons qu'on observe $r_4 = 9$. Mise à jour : $\hat Q(B) = (12 + 9)/2 = 10.5$, $N(B) = 2$.
> 
> **Pas 5 : nouveau choix.** Bonus : $\sqrt{2 \log 5 / N_t(a)} = \sqrt{3.22 / N}$. 
> 
> $$\text{UCB}(A) = 6 + \sqrt{3.22} = 7.795,\quad \text{UCB}(B) = 10.5 + \sqrt{1.61} = 11.769,\quad \text{UCB}(C) = 7 + \sqrt{3.22} = 8.795.$$
> 
> Argmax : **B** encore. On continue à exploiter B, mais on n'oublie pas A et C — leur bonus continue de croître en $\sqrt{\log t}$, ce qui les fera occasionnellement remonter dans le classement.
> 
> **Comportement asymptotique.** Sur 300 jours, UCB1 va jouer B environ $\sim 290$ fois et répartir les $\sim 10$ jours restants entre A et C, avec **plus de visites à A** (proche de B, gap petit donc plus dur à différencier) qu'à C (loin de B, gap grand donc identifié vite comme mauvais). C'est exactement la prédiction théorique : le regret se concentre sur les actions à *petit gap*.
> 
> **PnL attendu** : $\approx 290 \cdot 10 + 7 \cdot 8 + 3 \cdot 5 = 2971$ bps. Regret $\approx 29$, contre $\approx 70$ pour ε-greedy. Et l'écart se creuse à mesure que $T$ augmente, parce que UCB est en $\log T$ et ε-greedy en $T$ (si $\varepsilon$ ne décroît pas).

### E. Thompson Sampling

**Une autre philosophie.** Au lieu de borner l'incertitude par une formule de concentration (UCB), on l'**exprime probabilistiquement** : on maintient une distribution de probabilité (un *posterior*) sur la vraie valeur $Q(a)$ de chaque action, et on choisit l'action avec la **plus grande probabilité d'être optimale**.

> 💡 **Probability matching.** L'idée : à chaque pas, on choisit l'action $a$ avec probabilité égale à la probabilité que $a$ soit la meilleure. Formellement :
> 
> $$\pi(a \mid h_t) = \mathbb{P}\big[\,Q(a) > Q(a'),\; \forall a' \neq a \mid h_t\,\big].$$
> 
> Calculer ça analytiquement est en général dur. **Thompson Sampling est une astuce simple pour l'implémenter** : on tire un échantillon de la posterior de chaque action et on prend l'argmax des échantillons.

#### L'algorithme

> [!warning] Thompson Sampling
> 1. Maintenir une posterior $p(\theta_a \mid h_t)$ sur la vraie distribution de récompense de chaque action $a$.
> 2. À chaque pas $t$ :
>    - Pour chaque $a$, **tirer un échantillon** $\tilde\theta_a \sim p(\theta_a \mid h_t)$.
>    - Calculer la valeur estimée $\tilde Q(a) = \mathbb{E}[r \mid \theta = \tilde\theta_a]$.
>    - Jouer $a_t = \arg\max_a \tilde Q(a)$.
> 3. Observer $r_t$ et mettre à jour la posterior de $a_t$ par Bayes.

L'astuce : en tirant des échantillons aléatoires de la posterior, on **explore proportionnellement à l'incertitude**. Une action très incertaine a une posterior large, donc l'échantillon $\tilde\theta_a$ peut tomber sur de très grandes valeurs — et l'action sera choisie. Une action très certaine a une posterior étroite — l'échantillon sera proche de la vraie moyenne.

#### Cas Bernoulli avec prior Beta

Le cas le plus simple : récompenses **binaires** (succès / échec). On suppose que la vraie probabilité de succès $\theta_a$ a un prior Beta. C'est le **conjugué** de Bernoulli, donc la posterior reste Beta — mise à jour analytique.

> [!warning] Thompson Sampling pour Bernoulli MAB
> Soit $S_a$ = nombre de succès observés pour l'action $a$, $F_a$ = nombre d'échecs. Prior uniforme : $\text{Beta}(1, 1)$.
> 
> 1. Pour chaque $a$, tirer $\tilde\theta_a \sim \text{Beta}(S_a + 1, F_a + 1)$.
> 2. Jouer $a_t = \arg\max_a \tilde\theta_a$.
> 3. Observer $r_t \in \{0, 1\}$. Si $r_t = 1$ : $S_{a_t} \mathrel{+}= 1$. Sinon $F_{a_t} \mathrel{+}= 1$.

> [!note]- Pourquoi la Beta est conjuguée de la Bernoulli
> Si la vraisemblance est $\text{Bernoulli}(\theta)$ et le prior est $\text{Beta}(\alpha, \beta)$, alors la posterior après une observation $r$ est
> 
> $$p(\theta \mid r) \propto \theta^r (1-\theta)^{1-r} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{\alpha + r - 1}(1-\theta)^{\beta + (1-r) - 1},$$
> 
> qui est encore une Beta : $\text{Beta}(\alpha + r, \beta + 1 - r)$. La forme du prior est préservée — on appelle ça une **famille conjuguée**. Énorme avantage en pratique : pas de calcul d'intégrale, juste deux compteurs $(S_a, F_a)$ à entretenir.

#### Cas Gaussien avec prior Gaussien

Si les récompenses sont supposées $\mathcal{N}(\mu_a, \sigma^2)$ avec $\sigma^2$ connue, et qu'on a un prior $\mu_a \sim \mathcal{N}(\mu_0, 1/\tau_0)$ (où $\tau_0$ est la précision), alors la posterior reste gaussienne — encore un cas conjugué. Après $n$ observations $x_1, \ldots, x_n$ de l'action $a$ :

$$\tau_0 \leftarrow \tau_0 + n\tau, \qquad \mu_0 \leftarrow \frac{\tau_0 \mu_0 + \tau \sum_i x_i}{\tau_0 + n\tau},$$

où $\tau = 1/\sigma^2$ est la précision des observations.

> 💡 **Bilan.** Thompson Sampling est conceptuellement très simple, et empiriquement il est *au moins aussi bon* que UCB1, souvent meilleur en pratique. Il atteint la borne de Lai-Robbins. Il est aussi très utilisé en industrie (recommandation, A/B testing, ad placement) parce qu'il s'étend naturellement à des features contextuelles (contextual bandits).

> [!example] Fil rouge — Thompson Sampling avec récompenses gaussiennes
> 
> **Setup.** Stratégies A, B, C avec récompenses gaussiennes de variance $\sigma^2 = 25$ (donc $\sigma = 5$ bps). Prior plat : $\mu_a \sim \mathcal{N}(0, 100^2)$ pour chaque stratégie.
> 
> **Au début.** Posteriors très larges (variance ~100). Quand on tire $\tilde\mu_A, \tilde\mu_B, \tilde\mu_C$, ces tirages sont presque uniformes sur une large plage — chaque stratégie a environ 1/3 de chance d'avoir le plus grand échantillon. Donc l'agent **explore** uniformément au début, comme attendu.
> 
> **Après 30 essais** (~10 par stratégie, pour fixer les idées). Les posteriors se sont resserrées autour des vraies moyennes :
> 
> $$\mu_A \mid h_{30} \;\sim\; \mathcal{N}(8.2,\, 0.25^2), \quad \mu_B \mid h_{30} \sim \mathcal{N}(10.1,\, 0.25^2), \quad \mu_C \mid h_{30} \sim \mathcal{N}(5.3,\, 0.25^2).$$
> 
> Maintenant un tirage donne presque toujours $\tilde\mu_B > \tilde\mu_A > \tilde\mu_C$ (vu l'écart entre B et A est de 1.9 bps, soit ~7.6 écarts-types des posteriors). L'agent commence à exploiter B presque tout le temps.
> 
> **Quand explore-t-il encore ?** Quand par hasard l'échantillon $\tilde\mu_A$ dépasse $\tilde\mu_B$ — ce qui arrive avec une probabilité décroissante au fur et à mesure que les posteriors se resserrent. C'est *exactement* le mécanisme : exploration $\propto$ incertitude résiduelle.
> 
> **Comparaison qualitative avec UCB1.** Sur ce problème, les deux algorithmes ont des performances très proches. La différence opérationnelle :
> - UCB1 est **déterministe** (pour des observations données, le choix est fixé).
> - Thompson est **stochastique** (deux runs avec le même historique peuvent diverger).
> 
> En quant research, ça compte : Thompson est plus naturel quand on veut paralléliser plusieurs portefeuilles indépendants (chacun fait son propre tirage). UCB1 est plus naturel quand on veut un comportement reproductible.

### F. PAC Bandits

**Limite du regret.** Le regret cumulé est une mesure agrégée — il ne distingue pas entre "1000 petites erreurs" et "10 énormes erreurs". Dans certaines applications, on veut bien **borner le nombre de grandes erreurs**.

**Exemple médical.** Dans un essai clinique, donner un mauvais traitement avec léger effet secondaire à 1000 patients vaut probablement mieux que tuer 10 patients. Le regret cumulé peut être identique dans les deux cas — mais l'enjeu pratique est très différent.

> [!warning] Algorithme PAC ($\varepsilon, \delta$)
> Un algorithme est **PAC** s'il garantit, avec probabilité $\geq 1 - \delta$, que pour tous les pas sauf un nombre **polynomial** (en $\varepsilon, \delta, |\mathcal{A}|$), il choisit une action $\varepsilon$-optimale :
> 
> $$Q(a_t) \geq Q(a^*) - \varepsilon.$$

> 💡 **Lecture.** Au lieu de borner la **somme** des erreurs (regret), on borne le **nombre de gros loupés**. On accepte un nombre polynomial de pas où l'algorithme prend une action carrément mauvaise, mais après ce budget on garantit qu'on prend toujours une action $\varepsilon$-optimale (à proba $1 - \delta$ près).
> 
> Il existe des variantes PAC d'UCB et de Thompson Sampling. Elles sont plus pertinentes dans des contextes à fort enjeu individuel (médecine, robotique critique) que dans des contextes à enjeu agrégé (publicité, recommandation), où le regret cumulé reste la métrique reine.

## III. Information State Search

**Une autre façon de voir le problème.** Jusqu'ici on a traité le MAB comme un objet *sui generis*. Mais on peut le **reformuler comme un POMDP** (Partially Observable MDP — voir V des notes RL Tabulaire), ce qui ouvre la porte à des algorithmes plus puissants.

### Le MAB comme POMDP

**Idée centrale.** L'**état caché** du POMDP, c'est le vecteur des vraies moyennes $(\mu_1, \ldots, \mu_K)$. L'agent ne le voit pas, mais il maintient une **croyance** (belief state) — sa posterior actuelle sur ces moyennes. Chaque action / observation met à jour cette posterior.

- **Action** : tirer un bras.
- **Observation** : la récompense reçue.
- **État caché** : les vraies moyennes $\mu_a$.
- **Belief state** $\tilde s_t$ : la posterior $p(\mu_1, \ldots, \mu_K \mid h_t)$, où $h_t$ est l'historique.

Le belief state est une **statistique suffisante** de tout l'historique : tout ce qu'on a besoin de savoir pour décider est résumé dans la posterior actuelle. Donc le problème devient un **MDP sur l'espace des belief states** — appelé **information state space**.

### Bayes-adaptive RL

**L'objectif.** Trouver la politique optimale dans cet MDP étendu. Comme le belief state évolue de façon prévisible (selon Bayes), on peut faire de la **programmation dynamique** dessus, ou résoudre par RL classique.

> 💡 **Connexion avec Thompson Sampling.** Pour les Bernoulli bandits, le belief state est entièrement caractérisé par les compteurs $(S_a, F_a)$ — c'est ce qui rend Thompson Sampling si simple. Thompson Sampling est en fait une **heuristique** sur cet MDP étendu : au lieu de calculer la politique optimale (intractable), on tire un échantillon de la posterior et on agit gloutonnement par rapport à cet échantillon.

### Gittins Index

Pour certains MAB (récompenses i.i.d., horizon infini avec discount), il existe une **solution exacte** à l'MDP étendu, donnée par les **indices de Gittins**. L'idée : à chaque action $a$, on associe un index $G(a)$ qui ne dépend que de la posterior de $a$ (pas des autres). La politique optimale est alors $a_t = \arg\max_a G(a)$.

C'est un résultat magnifique théoriquement, mais **calculer les indices est intractable** dès que l'espace d'états devient grand. En pratique, on utilise des approximations par simulation.

## IV. Application aux MDP

Toutes les techniques vues dans le cas MAB s'étendent au cadre MDP général. Le défi : on a en plus de la stochasticité de la récompense, la stochasticité des **transitions** $P(s' \mid s, a)$. On doit donc explorer non seulement les actions, mais aussi les états.

### A. Optimistic Initialization — Rmax

**Principe.** Comme dans le cas MAB, on initialise les valeurs de manière optimiste pour forcer l'exploration. Spécifiquement, l'algorithme **Rmax** (Brafman & Tennenholtz, 2002) traite chaque paire $(s, a)$ encore peu visitée comme si elle menait, avec certitude, à un état fictif ultra-rentable.

> [!warning] Rmax (idée principale)
> On distingue deux types de paires $(s, a)$ :
> 
> - **"Connues"** : visitées au moins $m$ fois (seuil de confiance). On utilise les transitions et récompenses estimées par moyenne empirique.
> - **"Inconnues"** : visitées moins de $m$ fois. On les remplace par une transition fictive vers un état terminal de récompense $R_{\max}$ (la borne supérieure des récompenses possibles).
> 
> Puis on fait du Value Iteration sur ce MDP modifié, et on agit gloutonnement.

**Conséquence.** L'agent est **fortement incité à visiter les paires $(s, a)$ inconnues** — car les considérer comme menant à $R_{\max}$ gonfle leur valeur. Une fois qu'une paire a été visitée $m$ fois, elle bascule du côté "connu" et n'est plus artificiellement attirante.

> 💡 **Garantie PAC.** Rmax est un des premiers algorithmes RL à fournir une garantie PAC : avec probabilité $\geq 1 - \delta$, après un nombre polynomial de pas, l'agent suit une politique $\varepsilon$-optimale. C'est une raison pour laquelle l'algorithme reste un classique malgré sa simplicité.

### B. UCB Model-Based

**Extension directe de UCB1.** À chaque pas, on choisit l'action qui maximise la borne supérieure de confiance — mais maintenant la borne combine deux sources d'incertitude :

$$a_t = \arg\max_a \;\Big[\, Q(s_t, a) + U_1(s_t, a) + U_2(s_t, a) \,\Big],$$

où :

- $U_1(s, a)$ : incertitude liée à l'estimation de **la valeur** $Q(s, a)$ étant donné le modèle courant (analogue MAB).
- $U_2(s, a)$ : incertitude liée à l'estimation **du modèle** lui-même ($P, R$).

Le second terme est la nouveauté par rapport au MAB. Il est en pratique difficile à calculer rigoureusement — souvent on utilise des approximations ou on l'ignore.

### C. Thompson Sampling Model-Based

**Extension directe de Thompson MAB.** Au lieu de maintenir une posterior sur les moyennes des bras, on maintient une posterior sur **le MDP entier** : $\mathbb{P}[P, R \mid h_t]$.

À chaque pas (ou chaque épisode) :
1. On tire un MDP $\tilde M = (\tilde P, \tilde R)$ de la posterior.
2. On résout $\tilde M$ exactement (Value Iteration ou Policy Iteration) → politique $\tilde \pi$.
3. On suit $\tilde \pi$ pendant un épisode (ou un certain nombre de pas).
4. On observe les transitions et récompenses, on met à jour la posterior.

C'est l'algorithme **PSRL** (Posterior Sampling for Reinforcement Learning, Osband et al.). Il a d'excellentes garanties théoriques (regret bayésien sublinéaire) et fonctionne très bien en pratique sur les petits MDPs tabulaires.

### D. Information State Search en MDP

**Idée.** Comme en MAB, on peut augmenter l'espace d'états du MDP avec le **belief state** — la posterior sur le modèle inconnu. On obtient un **Bayes-adaptive MDP** dont la solution donne le compromis exploration/exploitation **optimal** étant donné le prior.

Le problème : l'espace d'états explose (on a maintenant des belief states sur des modèles complets). Donc en pratique on utilise des **méthodes simulation-based** (MCTS, *Monte Carlo Tree Search*) sur cet espace augmenté.

> 💡 **Bilan exploration en MDP.** Les trois grandes familles (Optimistic Init / Rmax, UCB-based, Thompson / PSRL) couvrent l'essentiel des approches modernes. Plus on monte en complexité (deep RL, espaces continus), plus ces approches deviennent approximatives — mais leur philosophie reste centrale : **explorer en proportion de l'incertitude**, soit par optimisme déterministe (UCB), soit par échantillonnage probabiliste (Thompson).
> 
> Pour aller plus loin : Bayesian Deep Q-Networks (Osband 2016+), Random Network Distillation (Burda 2018), curiosity-driven exploration (Pathak 2017). Tous ces algorithmes modernes construisent sur les fondations posées dans ce chapitre.
</content>