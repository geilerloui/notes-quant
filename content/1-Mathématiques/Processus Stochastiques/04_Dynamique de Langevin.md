---
title: Dynamique de Langevin
date: 2026-05-10
tags: [probabilités, processus-stochastiques, langevin, sampling, score-based]
---

## L'idée fondatrice

L'**équation de Langevin** est une EDS particulière qui fait le pont entre trois mondes : la physique (mouvement brownien dans un fluide visqueux), les statistiques bayésiennes (sampling MCMC), et le machine learning moderne (modèles de diffusion / score-based).

C'est aussi un **cas particulier important d'EDS** au-dessus des modèles classiques (cf [[03_Équations Différentielles Stochastiques]] pour ABM/MBG/OU). Ce qui la rend spéciale, ce sont ses **propriétés magiques** : distribution stationnaire connue explicitement, lien direct avec Fokker-Planck, et pont avec le score $\nabla \log \pi$ qui sous-tend tout le ML génératif récent.

> [!warning] Définition — Équation de Langevin
> Soit $U : \mathbb{R}^d \to \mathbb{R}$ un potentiel régulier (typiquement $C^2$). L'**équation de Langevin** associée à $U$ est l'EDS :
> 
> $$dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t$$
> 
> Deux termes :
> - **Drift** $-\nabla U(X_t)$ : pousse $X$ dans la direction qui fait **descendre** $U$. C'est de la descente de gradient.
> - **Diffusion** $\sqrt{2}\,dW_t$ : bruit thermique qui empêche $X$ de se figer dans un minimum local.

L'image physique : une **particule dans un paysage potentiel**. Le drift est la force qui la pousse vers les vallées. Le bruit thermique l'agite, lui permet d'explorer, et au besoin de franchir les barrières entre vallées.

---

## I. La particule dans le potentiel double-puits

Pour fixer les idées, on prend le **potentiel double-puits** :

$$U(x) = (x^2 - 1)^2$$

Il a deux minima en $x = \pm 1$ (les deux vallées) et un maximum local en $x = 0$ (la barrière, $U(0) = 1$). Le gradient : $\nabla U(x) = 4x(x^2 - 1)$.

![[fig1_double_well.png]]
*Figure 1. **Haut** — le potentiel $U(x) = (x^2-1)^2$ avec ses deux minima et la barrière au milieu. **Bas** — une trajectoire Langevin partant de $X_0 = -1$ (au fond du puits gauche). On voit qu'elle oscille longtemps autour de $-1$ (drift vers le minimum + bruit), puis à $t \approx 13$ le bruit l'emporte par-dessus la barrière et elle se retrouve dans le puits droit. Sur un temps assez long, elle visitera les deux puits dans des proportions précises données par $\pi$.*

**Lecture intuitive.** Sans bruit ($\sqrt{2} = 0$), la trajectoire serait une descente de gradient classique : partant de $X_0 = -1$, elle resterait pour toujours en $-1$ (minimum). Avec bruit, elle **explore** : la majorité du temps elle reste près d'un minimum (où le drift la rappelle), et de temps en temps une fluctuation suffisamment forte la fait basculer dans l'autre puits.

---

## II. La distribution stationnaire — la propriété magique

Le résultat central qui fait la beauté de Langevin :

> [!warning] Théorème — Distribution stationnaire
> L'équation de Langevin $dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t$ admet (sous bonnes hypothèses sur $U$, par exemple $e^{-U}$ intégrable) la distribution stationnaire :
> 
> $$\boxed{\pi(x) = \frac{1}{Z} e^{-U(x)}, \quad Z = \int_{\mathbb{R}^d} e^{-U(x)}\,dx}$$
> 
> C'est-à-dire que pour $t$ grand, la loi de $X_t$ converge vers $\pi$, indépendamment de la condition initiale $X_0$.

**Lecture pratique.** Cette formule est extraordinaire : si tu veux **sampler** une distribution $\pi$, tu n'as pas besoin de connaître $\pi$ explicitement (constante de normalisation $Z$ inconnue). Il suffit de connaître $U = -\log \pi$ à constante près, et de simuler la dynamique de Langevin assez longtemps. C'est ce qu'on appelle le **sampling par Langevin**.

> [!note]- Démonstration via Fokker-Planck
> La densité $p_t$ de $X_t$ vérifie l'équation de Fokker-Planck (cf [[04_Reverse-time SDE et Fokker-Planck]]) :
> 
> $$\partial_t p_t = -\nabla \cdot (p_t \cdot (-\nabla U)) + \frac{1}{2} \cdot 2 \cdot \Delta p_t = \nabla \cdot (p_t \nabla U) + \Delta p_t$$
> 
> Pour une distribution stationnaire $\pi$, on cherche $\partial_t \pi = 0$, c'est-à-dire :
> 
> $$0 = \nabla \cdot (\pi \nabla U + \nabla \pi)$$
> 
> Une solution évidente : $\pi \nabla U + \nabla \pi = 0$, c'est-à-dire $\nabla \pi = -\pi \nabla U$, soit $\nabla \log \pi = -\nabla U$, donc $\log \pi = -U + C$, et finalement $\pi(x) \propto e^{-U(x)}$. $\square$

![[fig2_stationary.png]]
*Figure 2. Convergence vers la stationnaire $\pi(x) \propto e^{-U(x)}$. On part de 2000 trajectoires toutes initialisées en $x = -1.5$ (loin du régime stationnaire). **Gauche** ($t \approx 0.1$) : les particules sont concentrées autour de $-1.5$, complètement loin de $\pi$. **Milieu** ($t \approx 1$) : elles ont commencé à se rassembler autour des deux puits mais le poids n'est pas équilibré. **Droite** ($t \approx 20$) : la distribution empirique colle à la cible $\pi$ (en rouge), avec deux pics également peuplés. La courbe rouge $\pi$ est **bimodale** parce que $U$ a deux minima.*

---

## III. Cas particulier important : $U$ quadratique = Ornstein-Uhlenbeck

Si $U(x) = \frac{1}{2}x^2$, alors $\nabla U(x) = x$ et l'équation de Langevin devient :

$$dX_t = -X_t\,dt + \sqrt{2}\,dW_t$$

C'est exactement un **processus d'Ornstein-Uhlenbeck** avec paramètres $\theta = 1$, $\mu = 0$, $\sigma = \sqrt{2}$. Tu reconnais cette EDS de [[03_Équations Différentielles Stochastiques]].

La distribution stationnaire prédite par la théorie : $\pi(x) \propto e^{-x^2/2}$, soit la **gaussienne standard** $\mathcal{N}(0, 1)$. Et c'est cohérent avec ce qu'on sait d'OU : la loi stationnaire d'OU avec $\theta = 1, \sigma = \sqrt{2}$ est $\mathcal{N}(0, \sigma^2/(2\theta)) = \mathcal{N}(0, 1)$. ✓

![[fig3_quadratic_OU.png]]
*Figure 3. **Gauche** — le potentiel quadratique $U(x) = x^2/2$ a un seul minimum en $x = 0$. **Droite** — 5 trajectoires Langevin partant de différents $X_0$ : toutes convergent vers $0$ en moyenne, avec des fluctuations gaussiennes. C'est le **retour à la moyenne** de l'OU.*

Cette équivalence Langevin (avec $U$ quadratique) = OU est utile : elle ancre Langevin dans le familier. Et inversement, on peut voir l'OU comme **le cas le plus simple** d'une dynamique de sampling — il sample une gaussienne.

---

## IV. Sampling : Langevin Monte Carlo en pratique

L'idée : pour sampler $\pi$, on simule numériquement Langevin par **Euler-Maruyama** :

$$X_{k+1} = X_k - h \nabla U(X_k) + \sqrt{2h}\,\varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)$$

C'est l'équivalent stochastique de la descente de gradient. **C'est même mieux** : c'est de la descente de gradient + bruit gaussien à chaque pas, ce qui empêche le piégeage dans un minimum local.

> [!note]- Pourquoi ça ressemble à de la descente de gradient bruitée
> Pour un pas de descente de gradient classique : $x_{k+1} = x_k - h \nabla U(x_k)$. C'est exactement le terme déterministe de l'Euler-Maruyama de Langevin. Le terme $\sqrt{2h}\,\varepsilon_k$ ajoute simplement un bruit gaussien d'écart-type $\sqrt{2h}$. **Donc Langevin = SGD avec bruit calibré pour avoir $\pi \propto e^{-U}$ comme stationnaire.**

### Exemple 2D : sampler une mixture de gaussiennes

Pour un cas non-trivial, prenons une **mixture gaussienne 2D** à 3 modes (3 paquets). On définit $\pi(x_1, x_2)$ comme cette mixture, et $U = -\log \pi$. On lance Langevin et on regarde où atterissent les trajectoires.

![[fig4_2d_sampling.png]]
*Figure 4. **Gauche** — la densité cible $\pi$ : 3 modes à des positions différentes (mixture gaussienne). **Droite** — 1000 samples obtenus en simulant Langevin pendant 2000 pas. On voit que les samples couvrent les 3 modes avec à peu près les bonnes proportions. C'est pour ça qu'on parle de **Langevin Monte Carlo** : on a transformé un problème de sampling en simulation d'EDS.*

### Effet du pas de discrétisation $h$

Euler-Maruyama introduit un **biais** quand $h$ est trop grand : la distribution des $X_k$ converge vers une distribution **proche** mais **pas exactement** $\pi$. C'est le prix de la discrétisation.

![[fig5_discretization.png]]
*Figure 5. Effet du pas $h$ sur la qualité du sampling. **Gauche** ($h = 0.01$) — distribution empirique très proche de $\pi$. **Droite** ($h = 0.5$) — biais visible : les pics sont mal placés et l'amplitude est fausse. En pratique, choisir $h$ est un compromis : petit $h$ = précis mais lent à converger, grand $h$ = rapide mais biaisé.*

### MALA — corriger le biais avec Metropolis

Pour éliminer le biais de discrétisation, on combine Euler-Maruyama avec un **pas Metropolis-Hastings** : à chaque proposition $X_{k+1}$, on accepte ou rejette selon la probabilité $\min(1, \pi(X_{k+1}) / \pi(X_k))$ ajustée par le ratio des densités de propositions. C'est le **MALA (Metropolis-Adjusted Langevin Algorithm)**, qui combine la précision MCMC avec l'efficacité du gradient.

C'est l'algorithme par défaut quand on veut un sampler exact (cf [[01_Inférence Bayésienne]] pour le contexte MCMC).

---

## V. Le pont avec les diffusion models en ML

C'est ici que Langevin devient **central pour le ML moderne**. L'observation clé :

$$\nabla \log \pi(x) = -\nabla U(x)$$

Le **drift de Langevin n'est rien d'autre que le score de la distribution cible**. Donc :

$$dX_t = \nabla \log \pi(X_t)\,dt + \sqrt{2}\,dW_t$$

> [!warning] Insight central — Langevin = sampling par le score
> Pour sampler $\pi$, on n'a pas besoin de $\pi$ explicitement, ni même de $U$. **Il suffit de connaître le score $\nabla \log \pi$**. Et le score peut être appris — c'est ce que font les modèles de diffusion modernes.

Cette idée fonde tout un pan du deep learning génératif :

1. **Score matching** (Hyvärinen 2005) : on apprend $s_\theta(x) \approx \nabla \log \pi(x)$ à partir de samples de $\pi$, sans connaître $\pi$ ni sa constante de normalisation.

2. **Score-based generative models** (Song et al. 2021) : on apprend le score à plusieurs niveaux de bruit, puis on sample par Langevin (ou plus précisément par reverse-time SDE, cf [[04_Reverse-time SDE et Fokker-Planck]]).

3. **DDPM, score SDE, flow matching** : différentes variantes qui exploitent toutes l'idée centrale "apprendre le score, sampler par EDS".

![[fig6_score_link.png]]
*Figure 6. **Gauche** — sur le potentiel double-puits, on voit que $U(x)$ et le score $\nabla \log \pi(x) = -\nabla U(x)$ sont liés : le score pointe vers les modes (positif quand on est à gauche du mode gauche, négatif quand on est à droite). C'est le score qui guide la trajectoire Langevin. **Droite** — plusieurs trajectoires Langevin partant de positions très différentes ($X_0 = -2, -0.3, 0.5, 2.2$) convergent toutes vers la même densité $\pi$ après burn-in. Le sampler est **insensible à l'initialisation** — c'est ce qui le rend pratique pour le ML.*

---

## VI. Pour aller plus loin

**Underdamped Langevin (avec moment).** Au lieu de $X_t$ seul, on suit le couple $(X_t, V_t)$ où $V_t$ joue le rôle de moment. La dynamique est plus rapide à converger pour des potentiels mal conditionnés. C'est l'analogue de la descente de gradient avec momentum pour le sampling.

**Stochastic Gradient Langevin Dynamics (SGLD, Welling-Teh 2011).** En deep learning, on remplace $\nabla U$ exact par un estimateur stochastique calculé sur un mini-batch (comme dans SGD). On obtient un sampler approximatif de la posterior bayesienne sur les poids du réseau. Ça donne une **alternative bayésienne à SGD** pour entraîner des réseaux de neurones.

**Hamiltonian Monte Carlo (HMC).** Variante d'underdamped Langevin où on supprime le bruit pendant le mouvement (dynamique hamiltonienne déterministe) et on n'ajoute du bruit que par le pas Metropolis. Algorithme par défaut en inférence bayésienne moderne (Stan, NumPyro). Cf [[02_MCMC]] pour le contexte.

**Score-based diffusion models.** Voir [[04_Reverse-time SDE et Fokker-Planck]] pour le lien rigoureux : on apprend le score pendant un processus *forward* qui rajoute du bruit, puis on sample en suivant le **reverse-time SDE** qui utilise ce score appris.

---

## Récapitulatif

| Concept | Définition / formule |
|---|---|
| Équation de Langevin | $dX_t = -\nabla U(X_t)\,dt + \sqrt{2}\,dW_t$ |
| Distribution stationnaire | $\pi(x) \propto e^{-U(x)}$ |
| Score | $\nabla \log \pi = -\nabla U$ |
| Cas $U$ quadratique | Langevin = OU (cf [[03_Équations Différentielles Stochastiques]]) |
| Euler-Maruyama | $X_{k+1} = X_k - h \nabla U(X_k) + \sqrt{2h}\,\varepsilon_k$ |
| MALA | Euler-Maruyama + pas Metropolis pour corriger le biais |
| SGLD | $\nabla U$ remplacé par estimateur sur mini-batch |
| Pont ML | Apprendre le score $\to$ sampler par Langevin / reverse-time SDE |

---

## Suite logique

**Précédent ← [[03_Équations Différentielles Stochastiques]]** : Langevin est un cas particulier important d'EDS, avec drift $= -\nabla U$ et diffusion constante.

**Suivant → [[05_Reverse-time SDE et Fokker-Planck]]** : pour le pont rigoureux entre Langevin et les diffusion models, et le théorème d'Anderson sur l'inversion du temps des EDS.

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. **[[04_Dynamique de Langevin]]** — (cette note) sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
7. [[07_Contrôle Stochastique]] — HJB, Merton, optimal execution, lien RL
8. [[08_Volterra Signatures]] — (placeholder) extension signature pour mémoire longue
