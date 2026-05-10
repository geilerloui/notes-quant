---
title: Mouvement Brownien
date: 2026-05-09
tags: [probabilités, processus-stochastiques, brownien, finance]
---

## L'idée fondatrice

On veut construire un **processus aléatoire en temps continu** $(W_t)_{t \ge 0}$ qui modélise le hasard pur — pas de tendance, pas de mémoire, juste de la fluctuation. C'est l'objet central de la finance quantitative (le sous-jacent de Black-Scholes), de la physique (mouvement d'une particule dans un fluide, observation de Brown en 1827) et plus généralement de tout calcul stochastique moderne.

L'approche naturelle : partir d'un objet discret simple — la **marche aléatoire** — et regarder ce qu'on obtient en raffinant les pas.

> [!note]- Pourquoi pas définir directement $W_t$ ?
> On *peut* définir le mouvement brownien axiomatiquement par ses propriétés (incréments gaussiens indépendants, trajectoires continues). Mais sans la construction par limite, on perd l'intuition cruciale : **ce qui rend le brownien si étrange — sa non-différentiabilité, sa variation quadratique non nulle — vient directement du scaling en $\sqrt{\Delta t}$**. La construction nous *montre* d'où viennent ces propriétés.

## I. La marche aléatoire simple

On part d'une suite $\varepsilon_1, \varepsilon_2, \dots$ de variables iid avec $\mathbb{P}(\varepsilon_t = +1) = \mathbb{P}(\varepsilon_t = -1) = 1/2$. La marche aléatoire est définie par la **récurrence** :

$$X_t = X_{t-1} + \varepsilon_t, \qquad X_0 = 0$$

À chaque instant, on **ajoute un choc** $\varepsilon_t$ à la position précédente — on monte ou descend de $1$ avec probabilité $1/2$. En déroulant la récurrence, on obtient la **forme close** :

$$X_t = X_0 + \sum_{i=1}^{t} \varepsilon_i = \sum_{i=1}^{t} \varepsilon_i$$

Les deux formes sont équivalentes : la récurrence dit *comment* le processus évolue, la forme close dit *où* il est après $t$ pas.

**Propriétés immédiates** : $\mathbb{E}(\varepsilon_t) = 0$ et $\mathrm{Var}(\varepsilon_t) = 1$, donc par linéarité et indépendance des $\varepsilon_i$ :

$$\mathbb{E}(X_t) = 0, \qquad \mathrm{Var}(X_t) = t$$

![[Pasted image 20260509134038.png|575]]
*Figure 1. Trois trajectoires indépendantes de la marche aléatoire simple sur 200 pas. La dispersion est en $\sqrt{t}$ : à $t = 200$, on s'attend à des valeurs de l'ordre de $\pm \sqrt{200} \approx \pm 14$.*

À ce stade, le "temps" $t$ n'est qu'un compteur de tirages. Pour passer en temps continu, il faut décider **combien de pas se produisent dans une unité de temps**.

## II. Le passage en temps continu — pourquoi $\sqrt{\Delta t}$ ?

On fixe un horizon $T$ (par exemple $T = 1$ an) et on découpe $[0, T]$ en $n$ morceaux de taille $\Delta t = T/n$. On veut faire $n \to \infty$ pour obtenir un processus à temps continu. Première tentative naïve : sur la grille $\{0, \Delta t, 2\Delta t, \dots, T\}$, on garde les sauts $\pm 1$. Avec $T=1, n=10$ on a les valeurs suivantes :

|     $k$     |  0  |  1  |  2  |  3  | $\dots$ | 10  |
| :---------: | :-: | :-: | :-: | :-: | :-----: | :-: |
| $k\Delta t$ |  0  | 0.1 | 0.2 | 0.3 | $\dots$ | 1.0 |

Problème : à $t = T$ on a fait $n$ pas, donc

$$\mathrm{Var}(X_n) = n \xrightarrow{n \to \infty} +\infty$$

> [!note]- Preuve : pourquoi $\mathrm{Var}(X_n) = n$ ?
> On a notre marche aléatoire sur la grille $\{0, \Delta t, 2\Delta t, \dots, T\}$. À chaque pas de la grille, on saute de $\pm 1$. Au total après $n$ pas (donc à l'instant $T$) :
> $$X_n = \varepsilon_1 + \varepsilon_2 + \dots + \varepsilon_n$$
>
> Chaque saut $\varepsilon_i$ vaut $+1$ ou $-1$ avec proba $1/2$, donc :
> - $\mathbb{E}(\varepsilon_i) = \tfrac{1}{2}(+1) + \tfrac{1}{2}(-1) = 0$
> - $\mathrm{Var}(\varepsilon_i) = \mathbb{E}(\varepsilon_i^2) = \tfrac{1}{2}(1)^2 + \tfrac{1}{2}(-1)^2 = 1$
>
> Comme les $\varepsilon_i$ sont **indépendants**, la variance de la somme est la somme des variances :
> $$\mathrm{Var}(X_n) = \mathrm{Var}(\varepsilon_1) + \dots + \mathrm{Var}(\varepsilon_n) = \underbrace{1 + 1 + \dots + 1}_{n \text{ fois}} = n \quad\square$$

Donc l'écart-type est $\sqrt{n}$, et avec $T = 1$ an :

| $n$ | $\Delta t$ | $\sqrt{\mathrm{Var}(X_n)}$ |
| :---: | :---: | :---: |
| 10 | 0.1 | $\sqrt{10} \approx 3.2$ |
| 100 | 0.01 | $10$ |
| $10\,000$ | $10^{-4}$ | $100$ |
| $1\,000\,000$ | $10^{-6}$ | $1000$ |

Plus on raffine, plus l'écart-type grandit. La Figure 2 le montre concrètement : à $n=10$ une trajectoire reste entre $\pm 6$ environ, mais à $n=10,000$ elle atteint des valeurs de l'ordre de $\pm 100$, et à $n=\infty$ elle peut prendre des valeurs dans $\pm \infty$. 
Mais "explose" ne signifie pas qu'une trajectoire individuelle diverge - chaque trajectoire finit à une valeur finie. Ce qui explose, c'est la zone où la trajectoire finale peut atterrir:

![[Pasted image 20260509134101.png]]
*Figure 2. Sans rescaling, raffiner la grille (augmenter $n$) fait exploser l'amplitude. À $n = 10\,000$ pas sur $[0,1]$, l'écart-type final est $\sqrt{n} = 100$.*

**L'idée** : pour que la variance reste finie, il faut diminuer la taille des sauts. On remplace $\varepsilon_i \in \{-1, +1\}$ par $\alpha \cdot \varepsilon_i$ pour un facteur $\alpha$ à déterminer. Le calcul de variance refait avec ce facteur donne :

$$\mathrm{Var}(X_n) = n \cdot \alpha^2$$

> [!note]- Preuve : d'où vient $\mathrm{Var}(X_n) = n \alpha^2$ avec le scaling ?
> Avec les nouveaux sauts $\alpha \varepsilon_i$, la marche devient :
> $$X_n = \alpha \varepsilon_1 + \alpha \varepsilon_2 + \dots + \alpha \varepsilon_n$$
>
> Pour chaque terme :
> $$\mathrm{Var}(\alpha \varepsilon_i) = \alpha^2 \cdot \mathrm{Var}(\varepsilon_i) = \alpha^2 \cdot 1 = \alpha^2$$
>
> Et par indépendance :
> $$\mathrm{Var}(X_n) = \sum_{i=1}^{n} \mathrm{Var}(\alpha \varepsilon_i) = n \cdot \alpha^2 \quad\square$$

On veut que ça reste **fini** quand $n \to \infty$. Le choix naturel : viser $\mathrm{Var}(X_n) = T$ (constant en $n$). Alors $n \alpha^2 = T$, donc :

$$\alpha^2 = \frac{T}{n} = \Delta t \quad\Longrightarrow\quad \alpha = \sqrt{\Delta t}$$

*Vérification.* Avec $\alpha = \sqrt{\Delta t}$ et $T = 1$ :

| $n$ | $\Delta t$ | $\alpha = \sqrt{\Delta t}$ | $\sqrt{\mathrm{Var}(X_n)} = \sqrt{n}\,\alpha$ |
| :---: | :---: | :---: | :---: |
| 10 | 0.1 | $\approx 0.32$ | $\sqrt{10 \cdot 0.1} = 1$ |
| 100 | 0.01 | $0.1$ | $\sqrt{100 \cdot 0.01} = 1$ |
| $10\,000$ | $10^{-4}$ | $0.01$ | $\sqrt{10\,000 \cdot 10^{-4}} = 1$ |

L'écart-type final reste à $\sqrt{T} = 1$ peu importe $n$ — directement comparable aux $3.2, 10, 100$ d'avant. **C'est exactement ce qu'on voulait.**

> [!note]- Pourquoi viser exactement $\mathrm{Var} = T$ ?
> C'est un choix de **normalisation**. On veut que sur $[0, T]$, l'écart-type de la trajectoire finale soit $\sqrt{T}$ — c'est ce qui rend la limite indépendante de $n$ et donne au mouvement brownien standard la propriété $\mathrm{Var}(W_t) = t$. Tout autre choix donnerait un processus proportionnel.

On avait $X_n$, où $n$ est le numéro du pas (1, 2, 3, ...) — il servait juste à compter les sauts. Maintenant qu'on a un pas de temps $\Delta t$, on peut indexer le processus par le **temps physique** $t \in [0, T]$ plutôt que par le numéro de pas : à l'instant $t$, on est arrivé après $\lfloor t/\Delta t \rfloor$ sauts. On change aussi de lettre, $X \to W$, pour préparer la limite : la **marche rescalée (*scaled random walk*)** va converger vers le mouvement brownien $W$. L'exposant $(n)$ marque la **résolution** — à chaque $n$ correspond un processus différent (la marche sur 10 points n'est pas la même que celle sur 1000 points).

$$W^{(n)}(t) = \sum_{i=1}^{\lfloor t/\Delta t \rfloor} \varepsilon_i \sqrt{\Delta t}$$

Le nombre de termes $k_n(t) = \lfloor t/\Delta t \rfloor = \lfloor nt/T \rfloor$ vérifie $k_n(t) \cdot \Delta t \to t$ quand $n \to \infty$, et donc :

$$\mathrm{Var}\bigl(W^{(n)}(t)\bigr) = k_n(t) \cdot \Delta t \xrightarrow{n \to \infty} t$$

## III. Le théorème central limite donne le brownien

Sur la grille à $n$ pas, $W^{(n)}(t)$ est une somme de $k_n(t)$ variables iid centrées de variance $\Delta t$. Le théorème central limite donne :

$$W^{(n)}(t) \xrightarrow[n \to \infty]{\text{loi}} \mathcal{N}(0, t)$$

> [!note]- Dérivation détaillée par le TCL
> On écrit $W^{(n)}(t) = \sqrt{\Delta t} \cdot S_{k_n}$ où $S_{k_n} = \sum_{i=1}^{k_n} \varepsilon_i$. Le TCL classique donne :
> $$\frac{S_{k_n}}{\sqrt{k_n}} \xrightarrow{\text{loi}} \mathcal{N}(0, 1)$$
> Donc :
> $$W^{(n)}(t) = \sqrt{\Delta t \cdot k_n} \cdot \frac{S_{k_n}}{\sqrt{k_n}} \xrightarrow{\text{loi}} \sqrt{t} \cdot \mathcal{N}(0,1) = \mathcal{N}(0, t)$$
> car $\Delta t \cdot k_n \to t$. $\square$

Le résultat plus fort est le **théorème de Donsker** : ce n'est pas seulement la loi marginale en $t$ qui converge, c'est *toute la trajectoire* (vue comme élément de $C([0,T])$) qui converge en loi vers le **mouvement brownien** $W$.

![[Pasted image 20260509134123.png]]
*Figure 3. La même réalisation du bruit, vue à différents niveaux de raffinement. À $n = 10$, on voit clairement les sauts. À $n = 10\,000$, la trajectoire devient (apparemment) continue — c'est le mouvement brownien.*

## IV. Définition axiomatique du mouvement brownien

Une fois construit, on peut résumer le brownien par quatre propriétés qui le caractérisent uniquement :

> [!abstract] Définition — Mouvement brownien standard
> Un processus $(W_t)_{t \ge 0}$ est un **mouvement brownien standard** si :
> 1. $W_0 = 0$
> 2. **Incréments indépendants** : pour $0 \le s_1 < t_1 \le s_2 < t_2$, les variables $W_{t_1} - W_{s_1}$ et $W_{t_2} - W_{s_2}$ sont indépendantes
> 3. **Incréments gaussiens stationnaires** : pour $0 \le s < t$, $W_t - W_s \sim \mathcal{N}(0, t-s)$
> 4. **Trajectoires continues** : $t \mapsto W_t(\omega)$ est continue presque sûrement

Toutes ces propriétés héritent directement de la construction par marche rescalée :
* (1) car $X_0 = 0$
* (2) car les $\varepsilon_i$ sont indépendants
* (3) car le TCL appliqué à un sous-bloc $[s, t]$ donne $\mathcal{N}(0, t-s)$
* (4) résulte du théorème de Donsker (continuité de la limite dans $C([0,T])$)

![[Pasted image 20260509134140.png]]
*Figure 4. 50 trajectoires browniennes superposées sur $[0,1]$. L'enveloppe rouge $\pm \sqrt{t}$ délimite une zone qui contient $\approx 68\%$ des trajectoires (un écart-type) ; $\pm 2\sqrt{t}$ en contient $\approx 95\%$. La forme parabolique reflète $\mathrm{Var}(W_t) = t$.*

![[Pasted image 20260509181448.png]]
*Figure 4bis. À chaque instant $t$ fixé, $W_t$ suit une gaussienne $\mathcal{N}(0, t)$. L'écart-type $\sigma = \sqrt{t}$ grandit avec le temps : la distribution s'élargit en racine du temps. C'est exactement la même information que l'enveloppe parabolique de la Figure 4, vue "de profil".*

## V. Les propriétés "bizarres" — non-différentiabilité et variation quadratique

Le brownien hérite d'une propriété étrange du scaling $\sqrt{\Delta t}$ : **ses incréments sont d'ordre $\sqrt{\Delta t}$**. Cette particularité a deux conséquences opposées selon comment on la regarde :

- Si on **divise** $\Delta W$ par $\Delta t$ pour calculer une dérivée → ça diverge ($\to \infty$). **Le brownien n'est dérivable nulle part.**
- Si on **élève** $\Delta W$ au carré → ça donne un terme d'ordre $\Delta t$, qui s'accumule en une quantité finie. **La variation quadratique est égale à $t$.**

Ces deux propriétés sont au cœur du calcul d'Itô : la première dit qu'on ne peut pas faire de calcul différentiel classique sur $W_t$, la seconde dit qu'on peut quand même faire *un autre* calcul, à condition de garder un terme de plus dans le développement de Taylor.

### V.1 Non-différentiabilité

**Heuristique** : Pour rappel le taux d'accroissement sur un intervalle $\Delta t$ pour eg $f(t)=t^2$

$$f^{\prime}(t)=\lim _{\Delta t \rightarrow 0} \frac{f(t+\Delta t)-f(t)}{\Delta t} = 2t$$

Mais pour un processus brownien :

$$\frac{W_{t + \Delta t} - W_t}{\Delta t} \sim \frac{\sqrt{\Delta t}}{\Delta t} = \frac{1}{\sqrt{\Delta t}} \xrightarrow{\Delta t \to 0} \infty$$

Quand on zoome, la pente apparente diverge. Une trajectoire brownienne est un objet **fractal** : sa rugosité se conserve à toute échelle.

![[Pasted image 20260509134157.png]]
*Figure 5. Zoom successif autour de $t = 0.5$ sur une trajectoire à 100 000 pas. La largeur passe de $1$ à $0.001$, mais la rugosité reste — on ne lisse jamais la courbe, contrairement à une fonction $C^1$ qu'on pourrait approcher localement par une droite.*

![[Pasted image 20260509182003.png]]
*Figure 5bis. Zoom successif sur une fonction lisse $f(t) = \sin(3t) + 0.5\cos(7t)$. À mesure qu'on zoome, la courbe devient de plus en plus droite : au dernier sous-graphe (largeur 0.001), elle ressemble à une droite de pente $f'(0.5) \approx 1.45$. **C'est ce que veut dire "être dérivable" : à l'échelle infinitésimale, la courbe se confond localement avec sa tangente.** À comparer avec la Figure 5 (brownien), où la rugosité persiste à toute échelle — d'où l'absence de dérivée.*

### V.2 Variation quadratique

Si la variation "première" $\sum |\Delta W|$ est infinie (le brownien bouge trop pour avoir une longueur finie) et que la variation cubique $\sum |\Delta W|^3$ tend vers 0 (les incréments sont quand même petits), il existe un **exposant intermédiaire** où la somme converge vers une quantité finie non nulle. Cet exposant est exactement 2 :

$$\sum_{i} (W_{t_{i+1}} - W_{t_i})^2 \xrightarrow[\|\Delta\| \to 0]{\mathbb{P}} t$$

Pour comparer, sur une fonction $f$ régulière, $\sum_i (f(t_{i+1}) - f(t_i))^2 \to 0$ quand le pas tend vers 0 (la somme est dominée par $(\max \Delta t) \cdot \sum_i |f'|^2 \cdot \Delta t \to 0$). Pour le brownien, **la quantité reste finie et vaut exactement $t$** — c'est précisément le scaling $\sqrt{\Delta t}$ qui rend ça possible : $(\Delta W)^2 \sim \Delta t$, donc en sommant on obtient $\sum \Delta t = t$.

C'est la propriété centrale qui rend le calcul d'Itô différent du calcul classique : **le carré des incréments du brownien est d'ordre $dt$, pas $dt^2$**.

![[Pasted image 20260509134213.png]]
*Figure 6. La somme des carrés des incréments, mesurée sur la même trajectoire à différents niveaux de raffinement. À $n = 10$, c'est trop bruité ; à partir de $n = 1000$, la convergence vers $y = t$ est claire.*

> [!note]- Pourquoi $\sum (\Delta W)^2 \to t$ exactement
> Chaque incrément $\Delta W_i = W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t_i)$, donc $\mathbb{E}[(\Delta W_i)^2] = \Delta t_i$. Par sommation :
> $$\mathbb{E}\Bigl[\sum_i (\Delta W_i)^2\Bigr] = \sum_i \Delta t_i = t$$
> Et la variance $\mathrm{Var}\bigl[\sum_i (\Delta W_i)^2\bigr] = 2 \sum_i (\Delta t_i)^2 \to 0$ (car les $(\Delta W_i)^2$ sont indépendants et $\mathrm{Var}((\Delta W_i)^2) = 2(\Delta t_i)^2$ pour une gaussienne). La convergence en probabilité suit. $\square$

## VI. De la non-différentiabilité au calcul d'Itô

### VI.1 Le problème : on ne peut pas écrire d'équation différentielle bruitée

En sciences, quand on veut modéliser une quantité qui évolue dans le temps, on écrit une **équation différentielle ordinaire (EDO)** :

$$\frac{dx_t}{dt} = a(x_t, t)$$

"À chaque instant, le taux de variation dépend de l'état actuel". Cette équation **génère** la trajectoire complète une fois résolue. Quelques exemples classiques :

- **Particule dans un courant** : $\frac{dx_t}{dt} = v$ (vitesse constante) → $x_t = x_0 + vt$
- **Croissance d'une obligation** : $\frac{dB_t}{dt} = r B_t$ (taux constant $r$) → $B_t = B_0 e^{rt}$
- **Désintégration radioactive** : $\frac{dN_t}{dt} = -\lambda N_t$ → $N_t = N_0 e^{-\lambda t}$

Maintenant on veut modéliser une quantité **bruitée** — par exemple le prix d'une action $S_t$, ou la position d'une particule de pollen dans un fluide (Einstein 1905). On voudrait écrire :

$$\frac{dx_t}{dt} = a(x_t, t) + b(x_t, t) \cdot \frac{dW_t}{dt}$$

"Tendance + bruit brownien". Mais $\frac{dW_t}{dt}$ **n'existe pas** (cf. section V). Donc cette équation **n'a aucun sens**. On est coincés : pas de dérivée du brownien → pas d'EDO avec bruit → on ne peut rien modéliser.

### VI.2 La solution : passer par l'intégrale

L'idée d'Itô tient en une phrase :

> **Y'a pas de dérivée du brownien. Mais pour résoudre une équation différentielle, on peut passer par l'intégrale — pas besoin de dérivée.**

Reprenons l'obligation. L'équation $\frac{dB_t}{dt} = r B_t$ peut s'écrire **équivalemment** sous forme intégrale :

$$B_t = B_0 + \int_0^t r B_s\,ds$$

*Lecture* : à chaque instant $s$, on gagne un petit intérêt $r B_s\,ds$ (taux × solde × durée). L'intégrale accumule tous ces petits gains entre $0$ et $t$. C'est la **même équation** que la version différentielle, juste réécrite en mode "accumulation" plutôt qu'en mode "taux de variation".

Pour le prix d'une action $S_t$, on fait pareil — on écrit la dynamique sous forme intégrale **dès le départ** :

$$S_t = S_0 + \int_0^t \mu S_s\,ds + \int_0^t \sigma S_s\,dW_s$$

Le premier terme intégral est ordinaire (Riemann) — c'est la croissance déterministe au taux $\mu$, identique à l'obligation. **Le deuxième est nouveau** : on intègre contre le brownien $W_s$ lui-même, pas contre le temps. C'est l'**intégrale d'Itô**, définie comme limite de sommes :

$$\int_0^t \sigma S_s\,dW_s = \lim_{\|\Delta\| \to 0} \sum_i \sigma S_{s_i} \cdot \underbrace{(W_{s_{i+1}} - W_{s_i})}_{\Delta W_{s_i}}$$

*Lecture* : à chaque petit pas de temps, on prend la valeur courante $\sigma S_{s_i}$ (l'intensité du bruit à cet instant), on la multiplie par le mouvement du brownien $\Delta W_{s_i}$ sur ce pas, et on accumule. **Pas de dérivée nulle part.**

> [!note]- Attention : $ds$ et $dW_s$ ne sont pas la même chose
> | Symbole | Sens | Unité | Signe |
> | :---: | :---: | :---: | :---: |
> | $ds$ | petit pas de **temps** | secondes | toujours $> 0$ |
> | $dW_s$ | mouvement du brownien sur ce pas | sans dimension | aléatoire ($\pm$) |
>
> Dans $\int r B_s\,ds$, on accumule des **intérêts** (taux × solde × durée). Dans $\int \sigma S_s\,dW_s$, on accumule des **gains/pertes aléatoires** (intensité × mouvement de prix). Même structure mathématique, sens très différents.

#### Pourquoi cette construction marche (et pas la dérivée)

Une dérivée demande au quotient $\Delta W / \Delta t$ de converger — il diverge. Une intégrale demande à la **somme** $\sum \sigma S_{s_i} \Delta W_i$ de converger — et ça, ça marche, parce que les $\Delta W_i$ ont des **signes aléatoires qui se compensent**. Une dérivée veut une limite ponctuelle ; une intégrale tolère du bruit qui s'annule en moyenne.

C'est toute la finesse de la construction d'Itô : on ne peut pas traiter $W_t$ point par point (dérivation), mais on peut le traiter en moyenne accumulée (intégration).

### VI.3 La notation différentielle : un raccourci pour l'équation intégrale

Une fois l'intégrale construite, on s'autorise une écriture compacte. Au lieu d'écrire :

$$S_t = S_0 + \int_0^t \mu S_s\,ds + \int_0^t \sigma S_s\,dW_s$$

on écrit :

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

C'est **strictement la même équation**. Les $dS_t$, $dt$, $dW_t$ ne sont pas des "infiniment petits" au sens classique — ce sont des **abréviations** pour les intégrales correspondantes. C'est un langage, pas un calcul.

Cette notation est appelée **équation différentielle stochastique (EDS)**. La forme générale :

$$dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t$$

- $a(X_t, t)$ : **drift** — la tendance déterministe (ce que ferait $X_t$ s'il n'y avait pas de bruit)
- $b(X_t, t)$ : **diffusion** — l'intensité du bruit brownien à chaque instant

Pour l'action : drift = $\mu S_t$ (croissance proportionnelle), diffusion = $\sigma S_t$ (volatilité proportionnelle au prix). C'est le **mouvement brownien géométrique** (MBG), le modèle Black-Scholes.

![[fig7_mbg.png]]
*Figure 7. Mouvement brownien géométrique avec $S_0 = 100$, $\mu = 10\%$, $\sigma = 30\%$. **(a)** Drift seul : exponentielle pure $S_0 e^{\mu t}$, comme une obligation — le prix monte de façon déterministe. **(b)** Diffusion seule : le prix fluctue autour de $S_0$ sans tendance. **(c)** Les deux ensemble : exponentielle bruitée — le prix monte en moyenne mais avec des fluctuations aléatoires. Le pointillé vert montre la trajectoire "sans bruit" pour comparaison. **Important** : $S_t$ reste toujours positif (contrairement à $W_t$ qui peut être négatif), ce qui en fait un bon modèle pour des prix d'actifs.*

### VI.4 Calculer avec : la formule d'Itô

Maintenant qu'on sait écrire la dynamique de $S_t$, une question naturelle se pose. En finance on s'intéresse rarement au prix brut $S_t$ — on s'intéresse à des **fonctions de ce prix** : le prix d'une option $V(S_t, t)$, le log-return $\log S_t$, etc.

**Question** : si $S_t$ suit une EDS, quelle est la dynamique de $f(S_t)$ ?

En calcul classique, la réponse serait la règle de la chaîne :

$$\frac{df}{dt} = f'(S_t) \cdot \frac{dS_t}{dt}$$

En calcul stochastique, il y a un terme en plus, à cause de la variation quadratique $(dW_t)^2 = dt$. C'est la **formule d'Itô** :

$$df(S_t) = f'(S_t)\,dS_t + \tfrac{1}{2} f''(S_t)\,(dS_t)^2$$

Le terme $\frac{1}{2} f''(S_t)\,(dS_t)^2$ est **nouveau par rapport au calcul classique**. Pour une fonction $f$ régulière, $(dS_t)^2$ serait d'ordre $(dt)^2 \to 0$ et on le jetterait. Pour le brownien, $(dW)^2 = dt$ ne disparaît pas — il faut le garder.

C'est ce terme correctif qui rend toute la finance moderne possible. Sans lui, pas de prix d'option.

**Exemple concret : le log-prix.** Appliquons la formule d'Itô à $f(S_t) = \log S_t$, avec $S_t$ qui suit le MBG. Après calcul (détaillé dans [[02_Calcul d'Itô]]) on obtient :

$$
d(\log S_t) = \Bigl(\mu - \tfrac{1}{2}\sigma^2\Bigr) dt + \sigma\,dW_t
$$

Le terme $-\tfrac{1}{2}\sigma^2$ est **exactement** le terme correctif d'Itô. Sans lui, on aurait naïvement $d(\log S_t) = \mu\,dt + \sigma\,dW_t$ (en appliquant bêtement la règle de la chaîne classique au $\mu S_t\,dt$ du MBG). Mais c'est faux : le log-prix dérive en moyenne au taux $\mu - \tfrac{1}{2}\sigma^2$, pas $\mu$.

![[fig8_ito_log.png]]
*Figure 8. La formule d'Itô en action sur le log-prix. **(a)** Trajectoire du prix $S_t$ qui suit un MBG. **(b)** Trajectoire du log-prix $\log S_t$ : c'est un brownien avec drift, de pente théorique $\mu - \tfrac{1}{2}\sigma^2$ (vert). Le pointillé rouge montre la pente "naïve" $\mu$ qu'on aurait sans le terme correctif d'Itô — elle ne colle pas. C'est la différence concrète entre calcul classique et calcul d'Itô.*

> **Pour aller plus loin** — les équations différentielles stochastiques classiques (mouvement brownien arithmétique, brownien géométrique alias Black-Scholes, Ornstein-Uhlenbeck pour les processus mean-reverting), avec leurs solutions, distributions, calibration sur données réelles et simulation, sont traitées dans **[[03_Équations Différentielles Stochastiques]]**.

## Récapitulatif

| Concept | Formule clé |
|---|---|
| Marche aléatoire | $X_t = X_{t-1} + \varepsilon_t$, $\mathrm{Var}(X_t) = t$ |
| Scaling | $W^{(n)}(t) = \sum \varepsilon_i \sqrt{\Delta t}$, $\Delta t = T/n$ |
| Limite (Donsker) | $W^{(n)} \xrightarrow{\text{loi}} W$ dans $C([0,T])$ |
| Loi marginale | $W_t \sim \mathcal{N}(0, t)$ |
| Incrément | $W_t - W_s \sim \mathcal{N}(0, t-s)$, indépendant du passé |
| Variation quadratique | $\sum (\Delta W)^2 \to t$, soit $(dW)^2 = dt$ |
| Régularité | continu partout, différentiable nulle part (p.s.) |
| EDS | $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ = raccourci pour l'équation intégrale |
| Formule d'Itô | $df(S_t) = f'\,dS_t + \tfrac{1}{2} f''\,(dS_t)^2$ |

---

## Suite logique

**Suivant → [[02_Calcul d'Itô]]** : pour rendre rigoureuse la définition de $\int H_s\,dW_s$ qu'on a juste esquissée ici. On y voit la filtration, la construction par densité dans $\mathcal{H}^2$, l'isométrie d'Itô, le choix Itô vs Stratonovich, et la formule d'Itô dans sa forme générale. **C'est la note technique** qui justifie tout ce qu'on a admis.

Progression complète du sujet :
1. **[[01_Mouvement Brownien]]** — (cette note) construction de $W_t$, propriétés bizarres (non-dérivable, variation quadratique)
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
