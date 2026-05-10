---
title: Calcul d'Itô
date: 2026-05-10
tags: [probabilités, processus-stochastiques, ito, calcul-stochastique]
---

## L'idée fondatrice

Dans [[01_Mouvement Brownien]], on a vu qu'on ne peut **pas** dériver $W_t$ — donc pas d'EDO bruitée classique. La solution proposée : passer par l'intégrale, $\int_0^t H_s\,dW_s$, définie comme limite de sommes $\sum H_{s_i}\Delta W_i$.

Cette note rend cette construction **rigoureuse**, en répondant aux questions que la définition heuristique laisse en suspens :

1. **Quelles fonctions $H$** peut-on intégrer contre $dW$ ? (Pas n'importe lesquelles.)
2. **En quel sens** la limite des sommes converge-t-elle ?
3. **Quel point d'évaluation** : $H_{s_i}$ (gauche), $H_{s_{i+1}}$ (droite), $H_{(s_i+s_{i+1})/2}$ (milieu) ? **Ce choix change la valeur** — c'est le cœur de la différence Itô vs Stratonovich.
4. Quelles sont les **propriétés** de l'objet construit ? (Espérance, variance, martingalité.)

Ces réponses sont le **socle technique** sur lequel reposent les EDS, le calcul d'Itô, Black-Scholes, et toute la finance quantitative moderne. On les voit ici de manière visuelle, avec Monte Carlo pour rendre tangible chaque concept.

---

## I. Le cadre probabiliste : filtration et adaptabilité

### I.1 La filtration $(\mathcal{F}_t)$ — l'information disponible

Pour parler "d'information à l'instant $t$" de manière rigoureuse, on a besoin d'un objet mathématique : une **filtration**, c'est-à-dire une famille croissante de tribus $(\mathcal{F}_t)_{t \geq 0}$ avec $\mathcal{F}_s \subseteq \mathcal{F}_t$ pour $s \leq t$.

> [!warning] Définition — Filtration brownienne
> La **filtration naturelle** du mouvement brownien $W$ est définie par :
> $$\mathcal{F}_t = \sigma(W_s : 0 \leq s \leq t)$$
> 
> C'est la tribu engendrée par les positions du brownien jusqu'à $t$. Intuitivement, $\mathcal{F}_t$ encode **toute l'information disponible jusqu'à l'instant $t$** — c'est ce que tu sais en regardant le passé du processus.

L'idée clé : à l'instant $t$, **on ne connaît que ce qui s'est passé avant**. Le futur est indéterminé — il existe toute une famille de continuations possibles, chacune avec sa probabilité.

![[fig1_filtration.png]]
*Figure 1. À l'instant $t = 0.45$, le passé (en bleu) est fixé une fois pour toutes : c'est ce que contient $\mathcal{F}_t$. Mais le futur est encore ouvert : on superpose 15 prolongements possibles (en rouge), chacun étant une continuation valide de la trajectoire actuelle. Plus le temps avance, plus le cône des futurs s'élargit (variance qui grandit en $\sqrt{t}$).*

### I.2 Processus adapté

Quand on veut intégrer $\int_0^t H_s\,dW_s$, le processus intégrande $H$ doit respecter une contrainte fondamentale : **à chaque instant $s$, $H_s$ ne doit dépendre que du passé jusqu'à $s$**, pas du futur. Sinon on ferait de la triche. Par contre rien n'empêche le calcul de l'intégrale d'aller jusqu'à la fin ie jusqu'à $T$. 

> [!warning] Définition — Processus adapté
> Un processus $(H_s)_{s \geq 0}$ est **adapté** à la filtration $(\mathcal{F}_s)$ si pour tout $s \geq 0$, $H_s$ est $\mathcal{F}_s$-mesurable.
> 
> Concrètement : la valeur de $H_s$ peut être calculée à partir de l'observation du brownien jusqu'à l'instant $s$, sans utiliser ce qui se passe après.

![[fig2_adapted.png]]
*Figure 2. **Haut** — un processus adapté : $H_s = W_s$. À chaque instant $s$, sa valeur est lisible directement sur la trajectoire passée. **Bas** — un processus anticipé : $H_s = W_T$ (la valeur finale du brownien). À $s = 0.4$, on aurait besoin de connaître $W_T$ qui est encore dans le futur — c'est de la triche, ce processus n'est pas adapté.*

> [!example] Exemples concrets
> - $H_s = W_s$ : adapté ✓ (lecture directe de la trajectoire à $s$)
> - $H_s = \int_0^s W_u^2\,du$ : adapté ✓ (l'intégrale n'utilise que le passé)
> - $H_s = f(W_s, s)$ pour $f$ déterministe : adapté ✓
> - $H_s = W_T$ avec $T > s$ : **anticipé ✗** (utilise le futur)
> - $H_s = \max_{u \in [0, T]} W_u$ avec $T > s$ : **anticipé ✗**

> 💡 **Bornes vs intégrande, à ne pas confondre.** L'adaptabilité ne dit rien sur les bornes de l'intégrale — tu peux intégrer jusqu'à $T$, jusqu'à $+\infty$, peu importe. Elle dit que **à chaque instant $s$ rencontré dans le parcours d'intégration, l'intégrande $H_s$ doit être calculable avec uniquement l'info disponible à $s$**. C'est une condition de causalité sur $H$ lui-même, pas sur où s'arrête l'intégration.

L'adaptabilité est une condition **non-négociable** pour que l'intégrale d'Itô soit bien définie.

---

## II. Construction de l'intégrale d'Itô

### II.1 Étape 1 — Sommes de Riemann (idée de la limite)

L'intuition de l'intégrale est familière : on subdivise $[0, T]$ en $n$ pas, on évalue $H$ à chaque pas, et on somme. Mais avant tout : **l'intégrale d'Itô est une variable aléatoire**, pas un nombre. Pour chaque trajectoire $\omega$ tirée (donc chaque réalisation possible du brownien), elle vaut un nombre différent. On écrit donc explicitement $\omega$ pour ne pas l'oublier :

> [!warning] Définition heuristique
> L'**intégrale d'Itô** de $H$ contre $W$ sur $[0, T]$ est définie comme :
> $$\int_0^T H_s(\omega)\,dW_s(\omega) = \lim_{n \to \infty} \sum_{i=0}^{n-1} H_{s_i}(\omega) \cdot \bigl(W_{s_{i+1}}(\omega) - W_{s_i}(\omega)\bigr)$$
> 
> où $0 = s_0 < s_1 < \cdots < s_n = T$ est une subdivision dont le pas tend vers 0.

> 💡 **Une trajectoire à la fois.** L'équation ci-dessus se lit pour **un $\omega$ fixé** : on prend une trajectoire brownienne particulière, on calcule la somme dessus, et on obtient un nombre. Sur $\omega'$ différent on obtiendrait une autre valeur. **L'objet $\int_0^T H_s\,dW_s$ est une fonction $\omega \mapsto \text{nombre}$, donc une variable aléatoire** — comme $W_T$ qui est aussi une fonction $\omega \mapsto W_T(\omega) \in \mathbb{R}$.

> [!example] Calcul à la main avec $n = 4$ pas
> Prenons $T = 1$ et 4 sous-intervalles de longueur $\Delta = 0.25$. Une trajectoire $\omega$ fixée qui passe par les valeurs suivantes :
> 
> | $i$ | $s_i$ | $W_{s_i}(\omega)$ |
> |---|---|---|
> | 0 | 0 | $0$ (toujours, $W_0 = 0$) |
> | 1 | 0.25 | $-0.3$ |
> | 2 | 0.5 | $0.4$ |
> | 3 | 0.75 | $0.1$ |
> | 4 | 1 | $0.6$ |
> 
> On calcule $\sum_{i=0}^{3} W_{s_i} \cdot \Delta W_i$ terme par terme :
> 
> - $i = 0$ : $W_0 \cdot (W_{0.25} - W_0) = 0 \cdot (-0.3) = 0$ (toujours nul, car $W_0 = 0$)
> - $i = 1$ : $W_{0.25} \cdot (W_{0.5} - W_{0.25}) = (-0.3) \cdot 0.7 = -0.21$
> - $i = 2$ : $W_{0.5} \cdot (W_{0.75} - W_{0.5}) = 0.4 \cdot (-0.3) = -0.12$
> - $i = 3$ : $W_{0.75} \cdot (W_1 - W_{0.75}) = 0.1 \cdot 0.5 = 0.05$
> 
> **Somme** : $S_4 = 0 - 0.21 - 0.12 + 0.05 = -0.28$.
> 
> C'est notre approximation de $\int_0^1 W_s\,dW_s$ avec $n = 4$. On observe deux choses :
> 1. **Le premier terme est toujours 0** parce que $W_0 = 0$ — c'est attendu, pas un bug.
> 2. À chaque pas on a deux choses bien différentes : $H_{s_i} = W_{s_i}$ (la **valeur** en début de pas, fixe pour cette trajectoire) et $\Delta W_i = W_{s_{i+1}} - W_{s_i}$ (le **mouvement** sur le pas, positif ou négatif).

**Convergence quand $n$ augmente.** En prenant la même trajectoire $\omega$ (les mêmes valeurs aux instants concernés) et en raffinant la grille — $n = 8$, $32$, $200$, etc. — la somme partielle $S_n$ converge vers une valeur limite : c'est, par définition, l'intégrale d'Itô sur cette trajectoire. La formule d'Itô (qu'on verra en §V) nous dit qu'elle vaut $\frac{W_T^2 - T}{2}$ — environ $-0.32$ pour notre trajectoire d'exemple.

![[fig3_riemann.png]]
*Figure 3. **Gauche** — visualisation de l'exemple manuel à $n=4$ ci-dessus : les 5 points $W_{s_i}$ du tableau, les paliers bleus matérialisant $H_{s_i} = W_{s_i}$ constant sur chaque sous-intervalle, et les flèches verticales représentant les incréments $\Delta W_i$ (vert si la contribution $W_{s_i}\cdot\Delta W_i$ est positive, rouge si négative). On retrouve $S_4 = -0.28$. **Droite** — sur une vraie trajectoire brownienne fine (pas l'exemple manuel), on calcule la somme partielle $S_n$ pour $n$ croissant. Au début ($n = 5, 10, 20$) la somme oscille parce qu'on capture mal les zigzags du brownien. Quand $n$ augmente, $S_n$ se stabilise vers la valeur exacte $(W_T^2 - T)/2$ donnée par la formule d'Itô (ligne pointillée rouge).*

Mais **trois** détails techniques qu'on a glissés sous le tapis vont être cruciaux :

1. **En quel sens** la limite converge-t-elle ? (réponse : $L^2$, voir §III)
2. **Quel point** d'évaluation dans $[s_i, s_{i+1}]$ ? (réponse : gauche pour Itô, voir §II.2)
3. **Pour quelles fonctions** $H$ ? (réponse : adaptées et $L^2$, voir §III.1)

### II.2 Le choix Itô vs Stratonovich — c'est crucial

Dans une intégrale de Riemann classique $\int_0^T f(s)\,ds$, on peut évaluer $f$ à n'importe quel point de chaque sous-intervalle $[s_i, s_{i+1}]$ — gauche, droite, milieu — et on obtient la même limite. C'est parce que $f$ est régulière et $ds$ est petit.

**Pour l'intégrale stochastique, c'est faux.** Les incréments $\Delta W$ ont des signes aléatoires et leur produit avec $H$ dépend de la corrélation entre $H_{s_i}$ et $\Delta W_i$. Le choix du point change la **moyenne** de la somme de manière systématique.

> [!warning] Itô (point gauche) vs Stratonovich (milieu)
> Deux conventions possibles pour évaluer $H$ sur $[s_i, s_{i+1}]$ :
> 
> $$\text{Itô :} \qquad \int_0^T H_s\,dW_s = \lim_n \sum_i H_{s_i} \cdot \Delta W_i$$
> 
> $$\text{Stratonovich :} \qquad \int_0^T H_s \circ dW_s = \lim_n \sum_i \frac{H_{s_i} + H_{s_{i+1}}}{2} \cdot \Delta W_i$$
> 
> Les deux limites existent (sous bonnes hypothèses) mais **elles ne donnent pas la même valeur**.

**Démonstration empirique frappante.** Calculons $\int_0^1 W_s\,dW_s$ par les deux méthodes sur 5000 trajectoires browniennes :

![[fig4_ito_strato.png]]
*Figure 4. Sur 5000 trajectoires browniennes, on calcule l'intégrale en deux versions. **Itô (bleu)** : moyenne empirique = 0.000, parfaitement centrée sur 0. **Stratonovich (rouge)** : moyenne = 0.500, centrée sur $T/2$. **Le panneau de droite** montre que la différence Stratonovich − Itô vaut systématiquement $T/2 = 0.5$, **trajectoire par trajectoire** — pas en moyenne, mais à chaque tirage. C'est un fait déterministe, pas un artefact statistique.*

> [!note]- Pourquoi cette différence de $T/2$ ?
> Pour Stratonovich avec $H = W$, on a $\frac{H_{s_i} + H_{s_{i+1}}}{2} \cdot \Delta W_i = \frac{W_{s_i} + W_{s_{i+1}}}{2} \cdot (W_{s_{i+1}} - W_{s_i}) = \frac{W_{s_{i+1}}^2 - W_{s_i}^2}{2}$ (différence de carrés). Donc :
> $$\sum_i \frac{H_{s_i} + H_{s_{i+1}}}{2} \Delta W_i = \frac{W_T^2 - W_0^2}{2} = \frac{W_T^2}{2}$$
> 
> Pour Itô avec point gauche, après calcul (voir Mouvement Brownien §VI.2 ou section IV ci-dessous) :
> $$\sum_i W_{s_i} \Delta W_i \xrightarrow{} \frac{W_T^2 - T}{2}$$
> 
> Différence : $\frac{W_T^2}{2} - \frac{W_T^2 - T}{2} = \frac{T}{2}$. C'est exactement la **variation quadratique** du brownien sur $[0, T]$ qui apparaît — Stratonovich "voit" cette correction, Itô la cache.

**Pourquoi on choisit Itô en finance.** Le choix du point gauche a une propriété cruciale qu'on va démontrer en §III : **il rend l'intégrale d'espérance nulle et martingale**, ce qui correspond à la notion de "stratégie non-anticipative" en finance (au moment de prendre position $H_{s_i}$ à $s_i$, on ne connaît pas $W_{s_{i+1}}$). Stratonovich a sa place en physique (où la règle de la chaîne classique est préservée), mais en finance c'est Itô.

---

## III. Construction rigoureuse — l'isométrie comme clé de voûte

On va construire $\int_0^T H_s\,dW_s$ en deux étapes : d'abord pour des processus **simples** (constants par morceaux), puis on étend à une classe plus large par **densité dans $L^2$**.

### III.1 Étape 1 — L'espace $\mathcal{H}^2$

> [!warning] Espace des processus admissibles
> L'espace des processus pour lesquels l'intégrale d'Itô est définie est :
> $$\mathcal{H}^2 = \left\{ H : H \text{ adapté à } (\mathcal{F}_s),\ \mathbb{E}\left[\int_0^T H_s^2\,ds\right] < \infty \right\}$$
> 
> Deux conditions : **adapté** (cf §I.2) et **carré intégrable** au sens de la mesure produit $\mathbb{P} \otimes ds$.

### III.2 Étape 2 — Définition pour les processus simples

Un **processus simple** est un processus constant par morceaux sur une subdivision $0 = t_0 < t_1 < \cdots < t_n = T$ :
$$H_s = \sum_{i=0}^{n-1} \xi_i \cdot \mathbf{1}_{[t_i, t_{i+1})}(s)$$
où chaque $\xi_i$ est $\mathcal{F}_{t_i}$-mesurable et borné.

Pour ces processus on **définit directement** :
$$\int_0^T H_s\,dW_s = \sum_{i=0}^{n-1} \xi_i \cdot (W_{t_{i+1}} - W_{t_i})$$

C'est juste une somme finie — pas de limite, pas de problème d'existence.

### III.3 Étape 3 — L'isométrie d'Itô

C'est la propriété qui permet d'étendre l'intégrale aux processus pas simples par densité.

> [!warning] Théorème — Isométrie d'Itô
> Pour tout processus simple $H$ :
> $$\boxed{\mathbb{E}\left[\left(\int_0^T H_s\,dW_s\right)^2\right] = \mathbb{E}\left[\int_0^T H_s^2\,ds\right]}$$

Côté gauche : la "norme" de l'intégrale stochastique. Côté droit : la "norme" $L^2$ classique du processus $H$. **Les deux sont égales** — d'où le mot "isométrie" : l'application $H \mapsto \int H\,dW$ préserve les normes entre $L^2(\mathbb{P} \otimes ds)$ et $L^2(\mathbb{P})$.

> [!note]- Démonstration pour un processus simple
> $\left(\int H\,dW\right)^2 = \left(\sum_i \xi_i \Delta W_i\right)^2 = \sum_i \xi_i^2 \Delta W_i^2 + 2 \sum_{i < j} \xi_i \xi_j \Delta W_i \Delta W_j$
> 
> En prenant l'espérance et en utilisant les indépendances :
> 
> - Pour $i < j$ : $\mathbb{E}[\xi_i \xi_j \Delta W_i \Delta W_j] = \mathbb{E}[\xi_i \xi_j \Delta W_i \cdot \mathbb{E}[\Delta W_j \mid \mathcal{F}_{t_j}]] = 0$ car $\Delta W_j$ est centré et indépendant de $\mathcal{F}_{t_j}$.
> - Pour $i = j$ : $\mathbb{E}[\xi_i^2 \Delta W_i^2] = \mathbb{E}[\xi_i^2 \cdot \mathbb{E}[\Delta W_i^2 \mid \mathcal{F}_{t_i}]] = \mathbb{E}[\xi_i^2] \cdot (t_{i+1} - t_i)$ (par indépendance et $\mathbb{E}[\Delta W_i^2] = t_{i+1} - t_i$).
> 
> Donc $\mathbb{E}\left[\left(\int H\,dW\right)^2\right] = \sum_i \mathbb{E}[\xi_i^2](t_{i+1} - t_i) = \mathbb{E}\left[\int_0^T H_s^2\,ds\right]$. $\square$

![[fig6_isometry.png]]
*Figure 6. Vérification empirique de l'isométrie sur $H_s = W_s$ pour différents horizons $T$. On calcule $\mathrm{Var}(\int_0^T W_s\,dW_s)$ (Monte Carlo, 3000 trajectoires) et on le compare à $\mathbb{E}[\int_0^T W_s^2\,ds]$ (mesuré sur les mêmes trajectoires). Les points tombent parfaitement sur la droite $y = x$ : l'isométrie est vérifiée pour tous les $T$.*

### III.4 Étape 4 — Extension par densité

L'argument est classique en analyse fonctionnelle : les processus simples sont **denses** dans $\mathcal{H}^2$. Donc pour tout $H \in \mathcal{H}^2$, on prend une suite $H^{(n)}$ de processus simples qui converge vers $H$ dans $L^2(\mathbb{P} \otimes ds)$, et on définit :

$$\int_0^T H_s\,dW_s = \lim_{n \to \infty} \int_0^T H^{(n)}_s\,dW_s \quad \text{(dans } L^2(\mathbb{P})\text{)}$$

L'isométrie garantit que la limite **existe et ne dépend pas du choix de la suite** : si $\|H^{(n)} - H^{(m)}\|_{L^2} \to 0$ alors $\|\int H^{(n)}\,dW - \int H^{(m)}\,dW\|_{L^2} \to 0$ également. C'est une suite de Cauchy dans $L^2(\mathbb{P})$ (qui est complet), donc elle converge.

**Bilan de la construction.** L'intégrale d'Itô est définie pour tout $H \in \mathcal{H}^2$, comme limite $L^2$ d'intégrales sur des processus simples. C'est rigoureux, et l'isométrie passe à la limite par construction.

---

## IV. Les trois propriétés magiques de l'intégrale d'Itô

Une fois l'intégrale construite, trois propriétés héritent immédiatement de la définition. Ce sont elles qui rendent le calcul d'Itô si puissant.

### IV.1 Espérance nulle

> [!warning] Théorème — Espérance nulle
> Pour tout $H \in \mathcal{H}^2$ :
> $$\mathbb{E}\left[\int_0^T H_s\,dW_s\right] = 0$$

C'est la propriété d'**absence de biais** : intégrer contre un brownien donne en moyenne zéro, comme si chaque incrément aléatoire $\Delta W_i$ se compensait avec son opposé sur l'ensemble des trajectoires.

![[fig5_expectation.png]]
*Figure 5. **Gauche** — 50 trajectoires de $I_t = \int_0^t W_s\,dW_s$ en gris, et leur moyenne empirique sur 1000 simulations en rouge (qui colle à $y = 0$). **Droite** — distribution de $I_T$ : centrée sur 0 (moyenne empirique = $-0.008$), avec une asymétrie à droite (la distribution exacte est $\frac{W_T^2 - T}{2}$, qui est minorée par $-T/2$ donc tronquée à gauche).*

### IV.2 Martingalité

C'est la propriété **centrale** pour la finance.

> [!warning] Théorème — L'intégrale d'Itô est une martingale
> Le processus $I_t = \int_0^t H_s\,dW_s$ est une **martingale** par rapport à $(\mathcal{F}_t)$ :
> $$\mathbb{E}[I_t \mid \mathcal{F}_s] = I_s \quad \text{pour tout } s \leq t$$

**Lecture** : la meilleure prédiction du futur de l'intégrale, sachant l'information disponible aujourd'hui, c'est sa **valeur actuelle**. L'intégrale ne dérive pas — elle fluctue autour de son niveau actuel.

![[fig7_martingale.png]]
*Figure 7. À l'instant $t = 0.4$, on connaît la valeur $I_t \approx -0.160$ (valeur actuelle, en pointillés verts). On simule 100 prolongements possibles (en rouge), chacun produisant une valeur finale $I_T$. La moyenne empirique de ces 100 futurs est $\mathbb{E}[I_T \mid \mathcal{F}_t] \approx -0.139$, très proche de $I_t$ (l'écart est dû au bruit Monte Carlo sur 100 échantillons). C'est la martingalité visualisée : les futurs s'écartent dans toutes les directions, mais leur **moyenne** reste à la valeur actuelle.*

> [!note]- Pourquoi l'intégrale est martingale
> Pour un processus simple $H$ et $s \in [t_k, t_{k+1}]$ :
> $$I_t - I_s = \sum_{i \geq k+1} \xi_i \Delta W_i + \xi_k (W_t - W_s)$$
> 
> Conditionnellement à $\mathcal{F}_s$ :
> - $\xi_k$ est connu (il est $\mathcal{F}_{t_k}$-mesurable et $t_k \leq s$)
> - $W_t - W_s$ est indépendant de $\mathcal{F}_s$ et centré, donc $\mathbb{E}[\xi_k(W_t - W_s) \mid \mathcal{F}_s] = \xi_k \cdot 0 = 0$
> - Pour les termes futurs ($i \geq k+1$), même argument par tour de conditionnement : $\mathbb{E}[\xi_i \Delta W_i \mid \mathcal{F}_s] = \mathbb{E}[\xi_i \cdot \mathbb{E}[\Delta W_i \mid \mathcal{F}_{t_i}] \mid \mathcal{F}_s] = 0$.
> 
> Donc $\mathbb{E}[I_t - I_s \mid \mathcal{F}_s] = 0$, c'est-à-dire $\mathbb{E}[I_t \mid \mathcal{F}_s] = I_s$. La propriété passe à la limite par convergence $L^2$. $\square$

**Pourquoi c'est important pour la finance.** En arbitrage, une stratégie auto-financée sans coût initial dont la valeur est une martingale ne peut **pas** générer de profit moyen — c'est l'absence d'opportunité d'arbitrage. La théorie d'évaluation des options (Black-Scholes, Harrison-Pliska 1981) repose entièrement sur la transformation de la dynamique des prix en martingale via un changement de mesure (théorème de Girsanov).

### IV.3 Variation quadratique

> [!warning] Théorème — Variation quadratique de l'intégrale d'Itô
> Pour $I_t = \int_0^t H_s\,dW_s$, la variation quadratique est :
> $$[I]_t = \int_0^t H_s^2\,ds$$

Cette formule **généralise** $[W]_t = t$ (cas $H \equiv 1$) qu'on avait dans [[01_Mouvement Brownien]] §V.2.

![[fig8_quad_var.png]]
*Figure 8. Sur une trajectoire de $I_t = \int_0^t W_s\,dW_s$ avec 5000 pas, on calcule la variation quadratique empirique $\sum_i (\Delta I_i)^2$ (en violet) et on la compare à la prédiction théorique $\int_0^t W_s^2\,ds$ (en pointillés noirs). Les deux courbes se superposent parfaitement — on **mesure** vraiment la variation quadratique, pas seulement on la **calcule** théoriquement.*

---

## V. La formule d'Itô — le chain rule stochastique

**Pourquoi cette section ?** On a défini rigoureusement l'intégrale d'Itô comme limite de sommes de Riemann (§II-III), mais cette définition est **impraticable pour calculer**. Sauf cas très simples, on ne va pas se taper la limite des sommes à la main. Il faut un **outil de calcul**, l'analogue stochastique de la règle de la chaîne en analyse classique : si tu as une primitive $F$ de $f$, tu calcules $\int_a^b f(s)\,ds = F(b) - F(a)$ sans repasser par les sommes. La **formule d'Itô** joue exactement ce rôle pour les intégrales stochastiques — avec en plus un terme correctif spécifique au cadre stochastique.

On a déjà rencontré la formule d'Itô heuristiquement (cf [[01_Mouvement Brownien]] §VI.4). On la redonne ici dans sa forme générale, avec sa logique.

> [!warning] Théorème — Formule d'Itô (cas général)
> Soit $X_t$ une diffusion : $dX_t = \mu_t\,dt + \sigma_t\,dW_t$. Pour toute fonction $f \in C^{1,2}(\mathbb{R}_+ \times \mathbb{R})$ :
> $$df(t, X_t) = \frac{\partial f}{\partial t}\,dt + \frac{\partial f}{\partial x}\,dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2}\,d[X]_t$$
> 
> avec $d[X]_t = \sigma_t^2\,dt$ (variation quadratique de $X$).

**Lecture** : c'est la règle de la chaîne classique, plus un terme correctif $\frac{1}{2} f''\,d[X]$. Ce terme **ne disparaît pas** parce que la variation quadratique de $X$ est d'ordre $dt$, pas $dt^2$.

> [!note]- Idée de démonstration
> Taylor à l'ordre 2 :
> $$f(t + dt, X_{t + dt}) - f(t, X_t) \approx \partial_t f\,dt + \partial_x f\,(X_{t+dt} - X_t) + \frac{1}{2} \partial_{xx} f\,(X_{t+dt} - X_t)^2 + \cdots$$
> 
> Pour une fonction lisse classique, le terme en $(\Delta X)^2$ est d'ordre $(\Delta t)^2$ et disparaît à la limite. Mais ici $\Delta X \sim \sigma \Delta W \sim \sigma \sqrt{\Delta t}$, donc $(\Delta X)^2 \sim \sigma^2 \Delta t$ — d'ordre $\Delta t$, **du même ordre que les autres termes**. Il faut le garder.
> 
> En revanche $(\Delta X)^3 \sim (\Delta t)^{3/2} \to 0$ et les termes plus hauts disparaissent. La formule s'arrête à l'ordre 2 exactement à cause de cette balance d'ordres. $\square$

**Application directe.** On vérifie maintenant analytiquement que $\int_0^T W_s\,dW_s = \frac{W_T^2 - T}{2}$. Posons $f(x) = x^2/2$, donc $f'(x) = x$, $f''(x) = 1$. Avec $X_t = W_t$ et $d[W]_t = dt$ :

$$d\left(\frac{W_t^2}{2}\right) = W_t\,dW_t + \frac{1}{2} \cdot 1 \cdot dt$$

En intégrant entre $0$ et $T$ : $\frac{W_T^2}{2} = \int_0^T W_s\,dW_s + \frac{T}{2}$, soit $\int_0^T W_s\,dW_s = \frac{W_T^2 - T}{2}$.

C'est exactement la "vraie valeur" qu'on avait en Figure 3, et le terme $-T/2$ est la signature d'Itô (cf §II.2 sur Itô vs Stratonovich).

### V.1 Deuxième exemple résolu : $\int_0^t s\,dW_s$

Un exemple plus complet où $H_s = s$ (poids déterministe qui croît linéairement). On va le calculer **deux fois** — par la formule d'Itô, puis par somme de Riemann directe — pour vérifier que les deux coïncident.

**Calcul par la formule d'Itô (intégration par parties stochastique).** Posons $f(s, x) = s \cdot x$. Alors $\partial_s f = x$, $\partial_x f = s$, $\partial_{xx} f = 0$. Avec la formule d'Itô appliquée à $f(t, W_t) = t W_t$ : $d(s W_s) = W_s\,ds + s\,dW_s + \tfrac{1}{2} \cdot 0 \cdot ds = W_s\,ds + s\,dW_s$. Le terme correctif d'Itô disparaît ici car $\partial_{xx} f = 0$.

En intégrant entre $0$ et $t$ : $t W_t = \int_0^t W_s\,ds + \int_0^t s\,dW_s$, d'où :

$\boxed{\int_0^t s\,dW_s = t W_t - \int_0^t W_s\,ds}$

C'est l'analogue stochastique de l'intégration par parties classique $\int_0^t s\,f'(s)\,ds = t f(t) - \int_0^t f(s)\,ds$ — même structure, sauf que $df = f'\,ds$ devient $dW$ pour le brownien (et il n'y a pas de terme correctif d'Itô ici parce que $f$ est linéaire en $x$).

**Visualisation des 4 ingrédients.**

![[fig9_example_t_dW.png]]
*Figure 9. Exemple résolu sur une trajectoire fixée $\omega$. **(a)** La trajectoire $W_s$. **(b)** Le poids déterministe $H_s = s$ qui croît linéairement — plus on est tard dans $[0, T]$, plus chaque saut $\Delta W$ comptera lourdement. **(c)** Les contributions $s_i \cdot \Delta W_i$ visualisées comme des barres colorées (vert si $> 0$, rouge si $< 0$) : on voit nettement que les barres sont en moyenne **plus larges à droite** parce que $s_i$ est plus grand, même si les $\Delta W_i$ sont en moyenne d'amplitude similaire (ce sont juste des incréments gaussiens iid). **(d)** L'intégrale accumulée $I_t$ calculée en bleu directement par somme de Riemann, en orange via la formule $tW_t - \int_0^t W_s\,ds$ : les deux courbes se superposent parfaitement, ce qui confirme la formule.*

> [!warning] Attention aux visualisations "3D" trompeuses
> Tu trouveras parfois sur YouTube ou des blogs des visualisations qui plottent la trajectoire brownienne dans le plan $(s, W_s)$, ajoutent un troisième axe "weight" pour le poids $h(s)$, puis "projettent" la trajectoire sur le plan $z = h(s)$ en disant que ça *donne* l'intégrale d'Itô. C'est **trompeur** : la projection $(s, W_s, h(s))$ ne calcule rien mathématiquement, c'est juste une mise en scène. La **vraie** intuition est celle du panneau (c) ci-dessus : on **multiplie** $h(s_i)$ par l'incrément $\Delta W_i$ et on **somme**. Les barres ne sont pas dans un plan 3D, ce sont les contributions signées à la somme.

---

## VI. Existence et unicité des solutions d'EDS

> [!warning] Théorème — Existence et unicité (Lipschitz)
> Soit $a, b : \mathbb{R}_+ \times \mathbb{R} \to \mathbb{R}$ telles que :
> - **Lipschitz** : $|a(t, x) - a(t, y)| + |b(t, x) - b(t, y)| \leq K|x - y|$
> - **Croissance linéaire** : $|a(t, x)| + |b(t, x)| \leq K(1 + |x|)$
> 
> Alors l'EDS $dX_t = a(t, X_t)\,dt + b(t, X_t)\,dW_t$ avec $X_0$ donné admet une **unique solution forte** dans $\mathcal{H}^2$.

L'analogue stochastique du théorème de Cauchy-Lipschitz pour les EDO. Les conditions sont les mêmes que dans le cas déterministe — Lipschitz + croissance linéaire — la preuve est aussi de même nature (point fixe de Picard sur l'application $X \mapsto X_0 + \int a\,ds + \int b\,dW$, en utilisant l'isométrie pour borner la partie stochastique).

**En finance.** Le mouvement brownien arithmétique ($a, b$ constants), le brownien géométrique ($a = \mu x$, $b = \sigma x$) et l'Ornstein-Uhlenbeck ($a = \theta(\mu - x)$, $b = \sigma$) vérifient tous Lipschitz + croissance linéaire, donc sont bien posés. C'est ce qui justifie qu'on peut écrire $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ et parler de **la** solution sans ambiguïté.

---

## VII. Pour aller plus loin

Trois extensions naturelles, qui généralisent ce qu'on a vu dans des directions différentes.

**Théorème de représentation des martingales (Itô).** Toute martingale $M_t$ adaptée à la filtration brownienne s'écrit de manière unique comme $M_t = M_0 + \int_0^t \phi_s\,dW_s$ pour un processus prévisible $\phi$. **Conséquence finance** : tout payoff $\mathcal{F}_T$-mesurable peut être *parfaitement* couvert par une stratégie $\phi$ (marché complet). C'est le théorème qui garantit l'existence d'un portefeuille de couverture pour Black-Scholes.

**Théorème de Girsanov.** Permet de **changer de mesure de probabilité** en gardant le calcul d'Itô valide. Si $W_t$ est un brownien sous $\mathbb{P}$ et qu'on définit $\widetilde{W}_t = W_t + \int_0^t \theta_s\,ds$ pour un processus $\theta$ raisonnable, alors $\widetilde{W}$ est un brownien **sous une nouvelle mesure** $\mathbb{Q}$ équivalente à $\mathbb{P}$. **Conséquence finance** : on peut transformer la dynamique réelle des prix (avec drift $\mu$) en dynamique martingale (drift = $r$, taux sans risque) sous la "mesure neutre au risque" $\mathbb{Q}$. C'est le mécanisme central de l'évaluation des dérivés.

**Au-delà des semi-martingales : rough paths (Lyons 1998).** L'intégrale d'Itô est définie pour des intégrateurs qui sont des **semi-martingales** (drift à variation finie + martingale locale). Mais que faire si $X$ est plus irrégulier — par exemple un mouvement brownien fractionnaire avec exposant de Hölder $H < 1/2$, ou un signal déterministe très bruité ? La théorie de Lyons définit alors une intégrale enrichie en utilisant non seulement $X$ mais aussi son **intégrale itérée** $\mathbb{X}_{s,t} = \int_s^t (X_r - X_s) \otimes dX_r$. Le couple $(X, \mathbb{X})$ est appelé "rough path", et l'intégration y devient une opération continue dans la bonne topologie. **Applications modernes** : path signature en machine learning (Kidger, Lyons), neural CDEs, modèles de volatilité rough en finance (Bayer-Friz-Gatheral 2016).

---

## Récapitulatif

| Concept | Définition / formule |
|---|---|
| Filtration | $\mathcal{F}_t = \sigma(W_s : s \leq t)$ — info disponible jusqu'à $t$ |
| Adapté | $H_s$ est $\mathcal{F}_s$-mesurable pour tout $s$ |
| Espace admissible | $\mathcal{H}^2 = \{H$ adapté, $\mathbb{E}[\int_0^T H^2\,ds] < \infty\}$ |
| Itô (point gauche) | $\int H\,dW = \lim \sum H_{s_i} \Delta W_i$ |
| Stratonovich (milieu) | $\int H \circ dW = \lim \sum \frac{H_{s_i} + H_{s_{i+1}}}{2} \Delta W_i$ |
| Différence Strato − Itô | Pour $H = W$ sur $[0, T]$ : exactement $T/2$ |
| Isométrie d'Itô | $\mathbb{E}[(\int H\,dW)^2] = \mathbb{E}[\int H^2\,ds]$ |
| Espérance | $\mathbb{E}[\int H\,dW] = 0$ |
| Martingalité | $\mathbb{E}[I_t \mid \mathcal{F}_s] = I_s$ |
| Variation quadratique | $[I]_t = \int_0^t H_s^2\,ds$ |
| Formule d'Itô | $df(t, X_t) = \partial_t f\,dt + \partial_x f\,dX + \frac{1}{2}\partial_{xx} f\,d[X]$ |
| Existence/unicité EDS | Lipschitz + croissance linéaire ⟹ solution unique dans $\mathcal{H}^2$ |

---

## Suite logique

**Précédent ← [[01_Mouvement Brownien]]** : pour la construction et les propriétés de $W_t$ — prerequis indispensable de cette note.

**Suivant → [[03_Équations Différentielles Stochastiques]]** : maintenant qu'on sait intégrer rigoureusement contre $dW$, on peut écrire des EDS et résoudre les modèles classiques (ABM, MBG = Black-Scholes, Ornstein-Uhlenbeck).

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. **[[02_Calcul d'Itô]]** — (cette note) formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
