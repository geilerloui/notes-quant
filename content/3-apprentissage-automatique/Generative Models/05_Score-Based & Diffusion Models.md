---
title: Score-Based & Diffusion Models
---
# Score-Based & Diffusion Models

> Cinquième famille — et la plus puissante en pratique aujourd'hui pour la génération d'images, d'audio, et de plus en plus de séries temporelles financières. Historiquement, deux écoles se sont développées en parallèle et se sont **révélées équivalentes** : les *score-based models* (Song & Ermon 2019), qui apprennent le gradient du log de la densité, et les *diffusion models* (Ho et al. 2020, DDPM), qui apprennent à inverser un processus de bruitage progressif. [Song et al. (2021)](https://arxiv.org/abs/2011.13456) ont unifié les deux formulations sous le cadre des **équations différentielles stochastiques** : les deux approches résolvent le même problème, vu sous deux angles complémentaires. On commence par la formulation *score-based*, qui s'enracine naturellement dans ce qu'on a vu sur les `[[00_Fondations|EBM]]` ; on construira ensuite la formulation diffusion ; et on terminera par le pont SDE.

## I. Pourquoi apprendre le score plutôt que la densité ?

### A. Le problème laissé ouvert par les EBM

On a vu quatre façons de représenter $p(x)$ jusqu'ici :

| Famille                                 | Représentation                                | Apprentissage          |
| --------------------------------------- | --------------------------------------------- | ---------------------- |
| `[[01_Modèles autoregressifs\|Autorégressif]]` | $\prod_i p_\theta(x_i \mid x_{<i})$           | MLE direct             |
| `[[02_VAE\|VAE]]`                         | $\int p_\theta(x\mid z) p(z)\, dz$            | Borne ELBO             |
| `[[03_Normalizing Flows\|Flow]]`        | $p_Z(f_\theta^{-1}(x)) \,\lvert\det J\rvert$  | MLE direct             |
| `[[04_GAN\|GAN]]`                         | Implicite : $x = g_\theta(\varepsilon)$       | Two-sample test        |

À cette liste s'ajoutent les **EBM** (energy-based models), une famille très flexible où

$$p_\theta(x) = \frac{e^{-f_\theta(x)}}{Z_\theta}, \qquad Z_\theta = \int e^{-f_\theta(x)}\, dx.$$

L'exponentielle garantit la positivité, et $Z_\theta$ normalise. Le problème central des EBM est que **$Z_\theta$ est intractable** dès que $x$ vit en grande dimension : on ne peut pas calculer la log-vraisemblance directement, et toutes les méthodes (contrastive divergence, MCMC) ne sont que des approximations bruitées du MLE.

> [!note] L'idée-clé
> Et si, au lieu de fitter la densité $p_\theta(x)$ elle-même, on fittait son **gradient** $\nabla_x \log p_\theta(x)$ ? On verra que dans le cas des EBM, ce gradient **ne dépend plus de $Z_\theta$** — la constante de normalisation s'évanouit. C'est l'amorce du paradigme score-based.

### B. La fonction score

Quand la densité $p(x)$ est différentiable, on définit la **fonction score** :

$$\boxed{\;s(x) := \nabla_x \log p(x)\;}$$

C'est un **champ de vecteurs** $\mathbb{R}^D \to \mathbb{R}^D$ (et non un scalaire comme $p(x)$) : à chaque point $x$, le score pointe dans la direction d'augmentation locale la plus rapide de la log-densité. Une analogie physique aide à fixer les idées : si on voit $\log p(x)$ comme un **potentiel électrostatique**, alors $s(x) = \nabla_x \log p(x)$ est le **champ électrique** associé — qui pointe des hautes densités vers les basses (à un signe près).

![[im37 1.png|400]]
*Champ de scores pour un mélange gaussien : les flèches pointent vers les modes (zones sombres = haute densité).*

L'argument central de tout le cadre score-based est le suivant :

> [!important] Le score caractérise la distribution
> Sous des conditions de régularité raisonnables (densité strictement positive et différentiable partout), $s(x) = \nabla_x \log p(x)$ **détermine $p(x)$ à une constante multiplicative près** — exactement comme un champ conservatif détermine son potentiel à une constante additive près. Or cette constante est fixée par la contrainte $\int p(x)\, dx = 1$. Donc connaître le score, c'est connaître la distribution.

### C. Le score tue la constante de normalisation

Reprenons l'EBM $p_\theta(x) = e^{-f_\theta(x)}/Z_\theta$ et calculons son score :

$$\nabla_x \log p_\theta(x) = -\nabla_x f_\theta(x) - \underbrace{\nabla_x \log Z_\theta}_{= \, 0}.$$

Le second terme est nul parce que $Z_\theta = \int e^{-f_\theta(x)}\, dx$ est une **constante** (un nombre, pas une fonction de $x$). Vérification sur deux exemples standards :

- **Gaussienne** : $p(x) = \tfrac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$. Ici $Z = \sqrt{2\pi}\sigma$, et

$$\nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}.$$

Pas de trace de $Z$.

- **Gamma** : $p(x) = \tfrac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$. Ici $Z = \Gamma(\alpha)/\beta^\alpha$, et

$$\nabla_x \log p(x) = \frac{\alpha - 1}{x} - \beta.$$

Pas de trace de $Z$.

> [!tip] Pourquoi c'est puissant
> Pour les EBM, ce résultat veut dire qu'on peut **apprendre $\theta$ en fittant un score** $s_\theta(x) = -\nabla_x f_\theta(x)$ sur le score des données, **sans jamais évaluer $Z_\theta$**. La normalisation, qui était le principal obstacle des EBM, n'apparaît tout simplement plus dans l'objectif.

Reste un problème : on ne connaît pas $\nabla_x \log p_{\text{data}}(x)$ — on n'a que des échantillons i.i.d. $\{x_1, \ldots, x_N\}$. Comment fitter le score d'une distribution qu'on ne connaît qu'à travers ses tirages ?

![[im39.png|475]]




![[im40.png|522]]
*Le pipeline complet du score matching. À gauche : la vraie distribution $p_{\text{data}}(x)$, inaccessible. Au centre : on n'a que des échantillons i.i.d. tirés d'elle. À droite : on entraîne un modèle paramétrique $s_\theta(x)$ — typiquement un réseau de neurones — à reproduire le champ de scores $\nabla_x \log p_{\text{data}}(x)$. Une fois $s_\theta$ appris, on peut générer de nouveaux échantillons via Langevin sans jamais avoir évalué ni la densité ni la constante de normalisation.*

Cette logique est la **boucle de fermeture** de tout le paradigme score-based :

1. **Constat EBM** : la log-vraisemblance est intractable à cause de $Z_\theta$.
2. **Pivot** : on passe du fit de $p_\theta$ au fit de $\nabla_x \log p_\theta$, qui ne dépend plus de $Z_\theta$.
3. **Nouvelle difficulté** : la cible $\nabla_x \log p_{\text{data}}$ est elle aussi inconnue (on n'a que des samples).
4. **Solution** : le **score matching**, qui réécrit l'objectif de façon à ce que la cible inconnue disparaisse — c'est l'objet de la section suivante.

## II. Score matching : fitter un score sans connaître le vrai

### A. Le cadre et l'objectif naïf - La divergence de Fisher

Le cadre est simple. On a un jeu d'entraînement i.i.d.

$$\{x_1, x_2, \ldots, x_N\} \stackrel{\text{i.i.d.}}{\sim} p(x),$$

avec $p$ inconnue, et on cherche à entraîner un **modèle de score** — typiquement un réseau de neurones — paramétré par $\theta$ :

$$s_\theta(x) : \mathbb{R}^D \to \mathbb{R}^D, \qquad s_\theta(x) \approx \nabla_x \log p(x).$$

La façon la plus naturelle de comparer deux champs de vecteurs est de regarder leur écart en norme $L^2$, moyenné sous la distribution des données. C'est la **divergence de Fisher** :

$$\boxed{\;\mathcal{L}_{\text{Fisher}}(\theta) = \frac{1}{2}\, \mathbb{E}_{p(x)}\!\left[\left\|\nabla_x \log p(x) - s_\theta(x)\right\|_2^2\right]\;}$$

![[images/3-Apprentissage automatique/Generative Models/score based/im1 (3).png|454]]

> [!warning] L'objectif est intractable tel quel
> On ne peut pas calculer cette divergence directement : elle dépend de $\nabla_x \log p(x)$, c'est-à-dire **exactement la quantité qu'on cherche à apprendre**. On semble tourner en rond. La suite est l'astuce d'intégration par parties qui résout le problème.

### B. L'astuce d'intégration par parties (1D)

On déroule la divergence en 1D pour rendre les manipulations lisibles, puis on généralisera. On développe le carré :

$$\frac{1}{2}\mathbb{E}_{p(x)}\!\left[\big(\nabla_x \log p(x) - s_\theta(x)\big)^2\right]
= \underbrace{\tfrac{1}{2}\!\int\! p(x)\big(\nabla_x \log p(x)\big)^2 dx}_{\text{(A) constant en } \theta}
+ \underbrace{\tfrac{1}{2}\!\int\! p(x)\, s_\theta(x)^2\, dx}_{\text{(B) ok}}
- \underbrace{\int\! p(x)\, \nabla_x \log p(x)\, s_\theta(x)\, dx}_{\text{(C) le terme gênant}}.$$

- Le terme **(A)** ne dépend pas de $\theta$ : on peut l'ignorer pour l'optimisation.
- Le terme **(B)** est facile : c'est juste $\tfrac{1}{2}\mathbb{E}_{p(x)}[s_\theta(x)^2]$, qu'on estime par Monte Carlo sur les données.
- Le terme **(C)** est le terme problématique : il contient le score inconnu.

**Traitons (C)**. On utilise l'identité $\nabla_x \log p(x) = \nabla_x p(x) / p(x)$, et le $p(x)$ se simplifie :

$$-\int p(x)\, \nabla_x \log p(x)\, s_\theta(x)\, dx
= -\int p(x)\, \frac{\nabla_x p(x)}{p(x)}\, s_\theta(x)\, dx
= -\int \nabla_x p(x)\, s_\theta(x)\, dx.$$

On reconnaît une intégration par parties. Avec $u = s_\theta(x)$ et $dv = \nabla_x p(x)\, dx$ :

$$-\int \nabla_x p(x)\, s_\theta(x)\, dx
= \underbrace{-\big[p(x)\, s_\theta(x)\big]_{-\infty}^{+\infty}}_{\text{(C}_1\text{)}}
+ \underbrace{\int p(x)\, \nabla_x s_\theta(x)\, dx}_{\text{(C}_2\text{)}}.$$

> [!note] Hypothèse de décroissance à l'infini
> Le terme de bord $(C_1)$ s'annule sous l'hypothèse que $p(x)\, s_\theta(x) \to 0$ aux bornes — c'est-à-dire que la densité décroît assez vite à l'infini pour dominer la croissance éventuelle du modèle de score. C'est vrai pour les distributions usuelles (gaussienne, gamma, distributions à support compact, etc.) et c'est l'hypothèse standard du score matching (Hyvärinen 2005).

Il ne reste donc que $(C_2) = \int p(x)\, \nabla_x s_\theta(x)\, dx = \mathbb{E}_{p(x)}[\nabla_x s_\theta(x)]$. En recollant **(B)** et **(C₂)**, et en oubliant la constante **(A)**, l'objectif devient :

$$\mathcal{L}(\theta) = \mathbb{E}_{p(x)}\!\left[\tfrac{1}{2}\, s_\theta(x)^2 + \nabla_x s_\theta(x)\right] + \text{const}.$$

**Le score inconnu a disparu.** L'objectif ne dépend plus que de $s_\theta$ et de ses dérivées, qu'on calcule par backprop, et de l'espérance sous $p$, qu'on estime par moyenne empirique.

### C. La formule générale et le problème de scalabilité

En dimension $D$, le carré devient une norme et la dérivée scalaire devient une trace de Jacobien. On obtient l'**objectif de score matching** d'[Hyvärinen (2005)](https://www.jmlr.org/papers/v6/hyvarinen05a.html) :

$$\boxed{\;\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p(x)}\!\left[\tfrac{1}{2}\,\|s_\theta(x)\|_2^2 + \operatorname{tr}\!\big(\underbrace{\nabla_x s_\theta(x)}_{\text{Jacobien de } s_\theta}\big)\right]\;}$$

Estimé par Monte Carlo sur le dataset :

$$\widehat{\mathcal{L}}_{\text{SM}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[\tfrac{1}{2}\,\|s_\theta(x_i)\|_2^2 + \operatorname{tr}\!\big(\nabla_x s_\theta(x_i)\big)\right].$$

Cette formule est *propre* : pas de $Z$, pas de score inconnu, juste deux termes qu'on calcule à partir du réseau. Mais elle cache un piège qui va bloquer le passage à l'échelle.

> [!warning] Le coût de la trace
> Le terme $\|s_\theta(x)\|_2^2$ est trivial : un seul forward pass donne le vecteur $s_\theta(x) \in \mathbb{R}^D$, dont on prend la norme. Le terme $\operatorname{tr}(\nabla_x s_\theta(x))$ est nettement plus problématique. Le Jacobien est une matrice $D \times D$, et calculer sa trace via autodiff demande **$D$ passes de backprop** (une par dimension de sortie, pour extraire chaque entrée diagonale $\partial s_{\theta,i}/\partial x_i$). Pour $D = 2$ sur un jouet 2D, c'est trivial. Pour $D = 3 \times 256 \times 256 \approx 2 \cdot 10^5$ sur ImageNet, c'est **infaisable**.

C'est le verrou qu'on devra lever pour passer à des images réelles. Deux familles de solutions existent — *sliced score matching* (estimateur Hutchinson de la trace) et *denoising score matching* (réécriture de l'objectif en termes de données bruitées) — qu'on développe dans la section suivante. Avant ça, on conclut cette section par la procédure de génération une fois qu'un score a été appris.

### D. Échantillonner avec Langevin dynamics

Supposons qu'on ait entraîné $s_\theta$ tel que $s_\theta(x) \approx \nabla_x \log p(x)$ partout. Comment générer un nouvel échantillon $x \sim p$ à partir de ce seul gradient ?

L'idée vient de la physique statistique : la **dynamique de Langevin** est une descente de gradient stochastique dans le paysage de log-densité. On part d'un point quelconque $x_0$ (typiquement tiré d'une gaussienne) et on itère :

$$\boxed{\;x_{t+1} = x_t + \frac{\epsilon}{2}\, s_\theta(x_t) + \sqrt{\epsilon}\, z_t, \qquad z_t \sim \mathcal{N}(0, I)\;}$$

Deux termes, deux rôles :

- $\tfrac{\epsilon}{2}\, s_\theta(x_t)$ : un pas de **gradient ascent** sur $\log p$ — on grimpe vers les zones de haute densité.
- $\sqrt{\epsilon}\, z_t$ : un **bruit gaussien** qui empêche l'algorithme de s'effondrer sur un mode et lui permet d'explorer toute la distribution.

Sans le bruit, on ferait juste une montée de gradient et on convergerait vers le mode le plus proche (MAP). Le bruit transforme cette descente en un **MCMC** qui échantillonne $p$ entièrement.

> [!important] Résultat de convergence
> [Welling & Teh (2011)](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) ont montré que sous $\epsilon \to 0$ et $t \to \infty$, $x_t$ converge en distribution vers un échantillon exact de $p(x)$. En pratique on prend $\epsilon$ petit mais fini et on itère un nombre fixé de pas, ce qui introduit un biais — biais que les méthodes annealed (NCSN) et les diffusion models vont apprendre à gérer proprement.

![[langevin_sampling_2d.png|400]]
*Trajectoire de Langevin sur un mélange gaussien 2D : partant de $x_0 = (1.5, -1.5)$, l'itération est attirée par les modes tout en explorant grâce au bruit.*

À ce stade on a un pipeline complet : (1) on entraîne $s_\theta$ par score matching, (2) on génère avec Langevin. Sur des jouets 2D, ça marche. Sur des images réelles, ça plante — pour deux raisons : le coût du Jacobien (vu plus haut), et un problème plus subtil de **support concentré** des données réelles qui rend le score mal défini partout sauf au voisinage des données. La section suivante traite les deux.


(1) Inputs.  Let a data input matrix $[X]_{1000 \times 2}$ that is each data points is of dimension two. We also define a neural net $s_{\theta}(\mathbf{x})$ which is a sequence of 

![[images/3-Apprentissage automatique/Generative Models/score based/im1-4.png|342]]
![[images/3-Apprentissage automatique/Generative Models/score based/im1-1.png|215]]

\textbf{(2) Computing the Two-norm:} First we consider a toy case to calculate the two-norm: we need one feed forward propagation then we can compute the two norm

![[images/3-Apprentissage automatique/Generative Models/score based/im2 (3).png|178]]

Mathematically this is equivalent to letting the matrix X through the neural nets $s_{\theta}(\mathbf{x})$ and we obtain for every single vector a dimension two gradient we compute the norm for every observation that is $\sqrt{x_{11}^2+x_{12}^2}$

$$
[X]_{1000 \times 2} \rightarrow [s_{\theta}(\mathbf{x})]_{1000 \times 2} \rightarrow \Big[||s_{\theta}(\mathbf{x})||_2^2\Big]_{1000 \times 1}
$$

\textbf{(3) Computing the Trace:} \textbf{(i) Computing the Jacobian.} We first do forward prop to get the output $s_{\theta, 1}(x)$, we do backprop and we only take the first derivative.
![[im3 (4).png]]



We continue this procedure for the second component we repeat the exact same process until we have all the elements of the diagonal we denote $D$ to denote the input dimension. \textcolor{oceanblue}{\textbf{In Image-net we have million of dimensions it won't be feasible because of the size thus it cannot be used to calculate the trace.}}\\

Mathematically we obtain

$$
[\nabla_{\mathbf{x}} s_{\theta}(\mathbf{x})]_{1000 \times 2 \times 2}
$$

\textbf{(ii) Calculating the Trace.} When our gradient is a tensor of dimension three the trace is just a stack of the diagonal of the tensor. For example if we have the tensor 


![[images/3-Apprentissage automatique/Generative Models/score based/im4 (1).png|484]]


\textbf{(4) Plotting.} From what we have said previously we argue that 

$$
[s_{\theta}(\mathbf{x})]_{10k \times 2} \approx \nabla_{x} \log p(x)
$$

To visualize the gradient field, we first plot the set of datapoints $[X]$. Next for every single point we have their vectors $s_{\theta}(x)$ which is the endpoint of the vector.

![[images/3-Apprentissage automatique/Generative Models/score based/im1-2.png|295]]


\newpage
\textbf{(5) Sampling.} Once the vector field has been learned we can leverage the Langevin Dynamics to produce true samples from the density $\mathbf{p(x)}$ by relying only on $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. The sampling is defined in a way very similar to MCMC approaches, by applying recursively where here we have set $x_0 = (1.5, -1.5)$

$$
\begin{aligned}
\mathbf{x}_{t+1}&=\mathbf{x}_{t}+\frac{\epsilon}{2} \nabla_{\mathbf{x}_{t}} \log p\left(\mathbf{x}_{t}\right)+\sqrt{\epsilon} \mathbf{z}_{t}\\
&= \mathbf{x}_{t}+\frac{\epsilon}{2} s_{\theta}(x_t)+\sqrt{\epsilon} \mathbf{z}_{t}
\end{aligned}
$$

where $\mathbf{z}_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) .$ It has been shown in $\underline{\text { Wellinget al. }}(\underline{2011})$ that under $\epsilon \rightarrow 0, t \rightarrow$ inf: $\mathbf{x}_{t}$ converges to an exact sample from $p(\mathbf{x}) .$ This is a key idea behind the score-based generative modeling approach.
![[im1-3 (1).png|330]]

