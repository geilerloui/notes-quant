---
title: Reverse-time SDE et Fokker-Planck
date: 2026-05-09
tags: [probabilités, processus-stochastiques, fokker-planck, reverse-sde, diffusion]
---

## L'idée fondatrice

Jusqu'ici on a vu les EDS sous l'angle des **trajectoires** : un processus $X_t$ qui évolue dans le temps selon $dX_t = a\,dt + b\,dW_t$. Cette note présente trois outils complémentaires :

1. **Fokker-Planck** : l'EDP qui décrit l'évolution de la **densité** $p_t(x)$ d'une EDS — la même information que les trajectoires, vue "en bloc"
2. **Score** $\nabla \log p_t(x)$ : un objet probabiliste fondamental, dérivé directement de la densité
3. **Reverse-time SDE** : un théorème (Anderson 1982) qui dit qu'on peut **inverser** une EDS, et que l'EDS inverse fait apparaître le score

Ces trois outils sont les pré-requis probabilistes pour comprendre les **modèles génératifs par diffusion** (DDPM, score-based generative models). On reste ici en proba pure : la partie "comment apprendre le score avec un réseau de neurones" est traitée ailleurs, en ML.

## I. Équation de Fokker-Planck

### Trajectoires vs densité

Pour une EDS $dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t$, on a deux descriptions équivalentes :

- **Trajectoires** : on simule $X_t$ pour des bruits $W$ différents et on regarde les courbes individuelles (ce qu'on a fait avec ABM/MBG/OU)
- **Densité** $p_t(x)$ : à chaque instant $t$, la distribution de probabilité de $X_t$ — *où le processus est susceptible d'être*

Les deux contiennent la même information, mais la densité est parfois plus facile à manipuler (notamment quand on s'intéresse à des limites, des distributions stationnaires, ou des transformations).

### L'équation

La densité $p_t(x)$ d'une EDS satisfait l'**équation de Fokker-Planck** (cas 1D) :

$$\partial_t p_t(x) = -\partial_x \bigl[a(x, t)\,p_t(x)\bigr] + \tfrac{1}{2}\,\partial_x^2 \bigl[b^2(x, t)\,p_t(x)\bigr]$$

C'est une EDP linéaire :
- Le **premier terme** (drift) déplace la densité — comme un courant qui transporte une concentration
- Le **deuxième terme** (diffusion) étale la densité — comme la chaleur qui se répand

### Exemple : Fokker-Planck pour OU

Pour le processus OU $dX_t = \kappa(\theta - X_t)\,dt + \sigma\,dW_t$, on a $a = \kappa(\theta - x)$ et $b = \sigma$ constants. L'équation devient :

$$\partial_t p_t = -\partial_x [\kappa(\theta - x)\,p_t] + \tfrac{\sigma^2}{2}\,\partial_x^2 p_t$$

Si on part d'une condition initiale $p_0(x) = \delta(x - X_0)$ (on est sûr que $X_0$ est en un point), la solution est :

$$p_t(x) = \mathcal{N}\bigl(X_0 e^{-\kappa t} + \theta(1 - e^{-\kappa t}),\ \tfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa t})\bigr)$$

Quand $t \to \infty$, $p_t$ converge vers la **distribution stationnaire** $\mathcal{N}(\theta, \sigma^2/(2\kappa))$ qu'on connaissait déjà depuis [[03_Équations Différentielles Stochastiques]].

![[fig_rev_fokker_planck_ou.png]]
*Figure 1. Évolution de la densité $p_t(x)$ d'un OU avec $\theta = 0$, $\kappa = 1$, $\sigma = 1$, partant d'un Dirac en $X_0 = 3$. À $t = 0.05$, la densité est concentrée près de $X_0$. Au fur et à mesure, elle dérive vers $\theta = 0$ (mean-reversion) et s'élargit. À $t = 3$, elle a pratiquement atteint la distribution stationnaire $\mathcal{N}(0, 0.5)$ (pointillé noir).*

> [!note]- Pourquoi ça marche : intuition
> Imagine un nuage de particules qui évoluent toutes selon la même EDS, indépendamment. La densité $p_t(x)$ est la concentration de ce nuage à l'endroit $x$ et l'instant $t$. Elle évolue parce que :
> - Les particules sont **transportées** par le drift : le nuage se déplace en bloc vers où le drift pointe
> - Les particules sont **étalées** par la diffusion : le nuage devient plus diffus
>
> Fokker-Planck est juste la formalisation de ce bilan local de concentration — exactement comme l'équation de la chaleur (cas $a=0$, $b=1$).

### Distribution stationnaire

Quand $\partial_t p_t = 0$ (la densité ne change plus), on dit que $p_t$ est **stationnaire**. Pour OU, c'est $\mathcal{N}(\theta, \sigma^2/(2\kappa))$. Pour ABM et MBG, **il n'y a pas de distribution stationnaire** (la variance grandit indéfiniment).

C'est pour ça qu'OU joue un rôle central pour les modèles de diffusion : on a besoin d'une EDS qui *converge* vers une distribution simple (typiquement gaussienne standard), pour ensuite pouvoir faire du sampling depuis cette distribution simple.

## II. Score function $\nabla \log p$

### Définition

Étant donnée une densité $p(x)$, on définit le **score** comme le gradient du log de la densité :

$$s(x) = \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}$$

C'est un **champ de vecteurs** sur $\mathbb{R}^d$ : à chaque point $x$, il indique la direction où la log-densité augmente le plus.

### Intuition : "force vers les modes"

Géométriquement, $s(x)$ pointe vers les **régions de haute densité** :
- Aux endroits où $p$ croît, $s > 0$
- Aux modes (maxima locaux), $s = 0$
- Aux endroits où $p$ décroît, $s < 0$

Penser au score comme une **force** qui, si on la suivait, ramènerait n'importe quel point vers le pic le plus proche de la distribution.

### Exemple 1 : score d'une gaussienne

Pour $p(x) = \mathcal{N}(\mu, \sigma^2)$ :

$$\log p(x) = -\frac{(x - \mu)^2}{2\sigma^2} + \text{const} \quad\Longrightarrow\quad s(x) = -\frac{x - \mu}{\sigma^2}$$

C'est une **fonction linéaire** qui s'annule à $\mu$, négative à droite, positive à gauche. Plus on est loin du mode, plus le score est grand en valeur absolue.

### Exemple 2 : score d'un mélange de gaussiennes

Pour une cible bimodale, le score n'est plus linéaire : il s'annule à chaque mode et change de signe entre les modes (à un point selle où la densité est minimale entre les deux pics).

![[fig_rev_score.png]]
*Figure 2. Score $\nabla \log p(x)$ pour deux densités. **(a, b)** Une gaussienne $\mathcal{N}(0, 1)$ : score linéaire $-(x - \mu)/\sigma^2$, qui s'annule au mode et tire vers $\mu = 0$. **(c, d)** Un mélange de deux gaussiennes en $\pm 2$ : le score s'annule à chaque mode, et il y a un point intermédiaire (vers $0$) où il change brusquement de signe — c'est la frontière entre les bassins d'attraction des deux modes.*

### Pourquoi le score nous intéresse

Le score $\nabla \log p$ a une propriété **magique** pour les modèles génératifs : il caractérise complètement la densité (à une constante près), mais **sans nécessiter le calcul de la constante de normalisation** $Z = \int p(x)\,dx$. C'est crucial parce que pour des distributions complexes (par exemple, l'ensemble des images réalistes de visages), $Z$ est totalement intractable.

Si on connaît $s(x)$, on peut **sampler** depuis $p$ par dynamique de Langevin :

$$x_{k+1} = x_k + \epsilon\,s(x_k) + \sqrt{2\epsilon}\,Z_k, \quad Z_k \sim \mathcal{N}(0, I)$$

C'est exactement une EDS discrétisée avec drift = score et diffusion = identité. Pour $\epsilon$ petit et beaucoup d'étapes, $x_k$ converge en loi vers $p$.

![[fig_rev_score_sampling.png]]
*Figure 3. Sampling Langevin sur une cible bimodale $\frac{1}{2}\mathcal{N}(-2, 0.5^2) + \frac{1}{2}\mathcal{N}(2, 0.5^2)$. On part de 800 points distribués comme du bruit (gaussienne large, à gauche). À chaque étape, on déplace les points dans la direction du score (qui pointe vers les modes), avec un peu de bruit ajouté. Après 30 étapes les points commencent à former les deux modes, et après 300 étapes la distribution empirique colle bien à la cible (pointillé noir).*

## III. Reverse-time SDE

### La question

On a une EDS forward :

$$dX_t = a(X_t, t)\,dt + b(t)\,dW_t, \quad t \in [0, T]$$

partant d'une distribution initiale $X_0 \sim p_0$ (la **vraie distribution** des données), et arrivant à $X_T \sim p_T$ (typiquement une gaussienne simple, en choisissant les bons coefficients).

**Question** : peut-on faire l'inverse ? Partir de $X_T \sim p_T$ (qu'on sait sampler) et remonter le temps pour obtenir $X_0 \sim p_0$ ?

Si oui, on a un **modèle génératif** : on échantillonne du bruit, on remonte le temps, et on obtient une donnée de la distribution cible.

### Le théorème (Anderson 1982)

**Oui, c'est possible**, et le processus inversé est lui-même une EDS. Si on note $\tau = T - t$ le temps inversé, le processus $\overline{X}_\tau = X_{T-\tau}$ satisfait :

$$d\overline{X}_\tau = \Bigl[-a(\overline{X}_\tau, T - \tau) + b^2(T - \tau)\,\nabla \log p_{T-\tau}(\overline{X}_\tau)\Bigr]d\tau + b(T - \tau)\,d\overline{W}_\tau$$

C'est une EDS ! Avec :
- Le **drift original inversé** $-a$ (on revient en arrière)
- **Plus un terme correctif** $b^2 \cdot \nabla \log p_t$ — c'est le **score**
- Le **même coefficient de diffusion** $b$ (mais avec un nouveau brownien $\overline{W}$)

> [!note]- Pourquoi le score apparaît
> L'idée : pour inverser le processus, il ne suffit pas d'inverser le drift $a$. On doit aussi compenser le fait que **la diffusion étale** la densité dans la direction forward. Pour la "ré-concentrer" en sens inverse, il faut pousser les particules vers les régions de haute densité — exactement ce que fait $\nabla \log p_t$.
>
> Le coefficient $b^2$ devant le score est exactement la quantité de "compensation d'étalement" qu'il faut, et elle vient directement de Fokker-Planck.

### Exemple visuel

![[fig_rev_forward_reverse.png]]
*Figure 4. Forward vs Reverse sur une distribution bimodale. **Ligne du haut (forward, bleu)** : on part de la bimodale (deux pics en $\pm 2$), on applique l'EDS forward $dX_t = -\frac{1}{2}X_t\,dt + dW_t$ (un OU). À $t = 0.5$ les pics commencent à s'élargir, à $t = 1.5$ ils se rejoignent, et à $t = 4$ la distribution est essentiellement gaussienne. **Ligne du bas (reverse, rouge)** : on lit la même évolution dans l'autre sens — on part du bruit gaussien et on remonte vers la bimodale. C'est ce que ferait la reverse-time SDE si on connaissait le score.*

![[fig_rev_trajectoires.png]]
*Figure 5. Trajectoires individuelles forward et reverse. À gauche : 8 trajectoires partent de la bimodale (concentrées autour de $\pm 2$) et finissent dispersées autour de $0$. À droite : les mêmes trajectoires lues à l'envers — elles partent dispersées et finissent concentrées sur les deux modes. La reverse-time SDE est la formalisation mathématique de "lire les trajectoires forward à l'envers".*

### Le lien avec les diffusion models

Le pipeline des modèles génératifs par diffusion est :

1. **Choisir une forward SDE** simple, fixée (typiquement OU non-stationnaire)
2. **Apprendre le score** $\nabla \log p_t(x)$ avec un réseau de neurones (la partie ML, pas couverte ici)
3. **Au moment de générer** : sampler $X_T$ depuis $p_T$ (gaussienne simple), puis simuler la reverse-time SDE jusqu'à $\tau = T$ pour obtenir un sample $X_0$ de la distribution cible

Tout ce qui est probabiliste — Fokker-Planck, score, reverse-time SDE — est dans cette note. **Tout ce qui est ML** (comment apprendre le score : score matching, denoising score matching, U-Net, etc.) est traité ailleurs.

## Récapitulatif

| Concept | Formule clé | Rôle |
|---|---|---|
| Fokker-Planck | $\partial_t p_t = -\partial_x(a p_t) + \tfrac{1}{2}\partial_x^2(b^2 p_t)$ | évolution de la densité d'une EDS |
| Distribution stationnaire | $\partial_t p_\infty = 0$ | limite quand $t \to \infty$ (existe pour OU, pas ABM/MBG) |
| Score | $s(x) = \nabla \log p(x)$ | "force" vers les modes de la densité |
| Sampling Langevin | $x_{k+1} = x_k + \epsilon\,s(x_k) + \sqrt{2\epsilon}\,Z_k$ | sampler depuis $p$ en connaissant juste son score |
| Reverse-time SDE (Anderson) | $d\overline{X}_\tau = [-a + b^2 \nabla \log p]\,d\tau + b\,d\overline{W}_\tau$ | inverser une EDS forward |

## Pour aller plus loin

- **Score matching** (Hyvärinen 2005) : comment estimer $\nabla \log p$ par optimisation, sans connaître $p$. Base théorique de tous les diffusion models. → ML/score matching
- **DDPM** (Ho et al. 2020) : la version discrète et pratique des diffusion models qui a tout démarré. → ML/diffusion models
- **Score-based SDE** (Song et al. 2021) : la formulation continue unifiée, qui montre que DDPM et score matching sont équivalents à des reverse-time SDEs. → ML/diffusion models
- **Flow matching** : alternative aux diffusion models qui apprend directement un champ de vélocité au lieu d'un score. Plus rapide à sampler.

---

## Suite logique

**Précédent ← [[04_Dynamique de Langevin]]** : on a vu un cas particulier important d'EDS dont le drift est $-\nabla U$. Sa distribution stationnaire $\pi \propto e^{-U}$ et son lien avec le score $\nabla \log \pi$ préparent directement les diffusion models qu'on va aborder ici via le retournement du temps.

**Suivant → [[06_Rough Paths]]** : pour aller au-delà du cadre des semi-martingales et travailler avec des intégrateurs plus irréguliers (brownien fractionnaire, Hurst $H < 1/2$). Note encore en construction.

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. **[[05_Reverse-time SDE et Fokker-Planck]]** — (cette note) EDP de l'évolution de la densité et inversion du temps
6. [[06_Rough Paths]] — au-delà des semi-martingales (Lyons 1998)
7. [[07_Contrôle Stochastique]] — HJB, Merton, optimal execution, lien RL
8. [[08_Volterra Signatures]] — (placeholder) extension signature pour mémoire longue
