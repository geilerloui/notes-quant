---
title: Score-Based & Diffusion Models
---
# Score-Based & Diffusion Models

> Cinquième famille — et la plus puissante en pratique aujourd'hui pour la génération d'images, d'audio, et de plus en plus de séries temporelles financières. Historiquement, deux écoles se sont développées en parallèle et se sont **révélées équivalentes** : les *score-based models* (Song & Ermon 2019), qui apprennent le gradient du log de la densité, et les *diffusion models* (Ho et al. 2020, DDPM), qui apprennent à inverser un processus de bruitage progressif. [Song et al. (2021)](https://arxiv.org/abs/2011.13456) ont unifié les deux formulations sous le cadre des **équations différentielles stochastiques** : les deux approches résolvent le même problème, vu sous deux angles complémentaires. On commence par la formulation *score-based*, qui s'enracine naturellement dans ce qu'on a vu sur les `[[02_Energy-models|EBM]]` ; on construira ensuite la formulation diffusion ; et on terminera par le pont SDE.

## I. Pourquoi apprendre le score plutôt que la densité ?

### A. Le problème laissé ouvert par les EBM

On a vu quatre façons de représenter $p(x)$ jusqu'ici :

| Famille                                 | Représentation                                | Apprentissage          |
| --------------------------------------- | --------------------------------------------- | ---------------------- |
| `[[01_Modèles autoregressifs\|Autorégressif]]` | $\prod_i p_\theta(x_i \mid x_{<i})$           | MLE direct             |
| `[[03_VAE\|VAE]]`                         | $\int p_\theta(x\mid z) p(z)\, dz$            | Borne ELBO             |
| `[[04_Normalizing Flows\|Flow]]`        | $p_Z(f_\theta^{-1}(x)) \,\lvert\det J\rvert$  | MLE direct             |
| `[[05_GAN\|GAN]]`                         | Implicite : $x = g_\theta(\varepsilon)$       | Two-sample test        |

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

### E. Exemple complet

La fonction de perte :

$$\widehat{\mathcal{L}}_{\text{SM}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left[\tfrac{1}{2}\,\|s_\theta(x_i)\|_2^2 + \operatorname{tr}\!\big(\nabla_x s_\theta(x_i)\big)\right].$$

**(1) Inputs.** Soit une matrice de données en entrée $[X]_{1000 \times 2}$ : chaque point de données est de dimension deux. On définit également un réseau de neurones $s_{\theta}(\mathbf{x})$ qui est une séquence de couches.

| Réseau de neurones                                                                   | Sortie                                                                               |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| ![[images/3-Apprentissage automatique/Generative Models/score based/im1-4.png\|368]] | ![[images/3-Apprentissage automatique/Generative Models/score based/im1-1.png\|263]] |
*Figures. blabla*

**(2) Calcul de la norme deux.** On commence par considérer un cas jouet pour calculer la norme deux : il suffit d'une propagation avant (feed forward), puis on peut calculer la norme deux.

![[images/3-Apprentissage automatique/Generative Models/score based/im2 (3).png|178]]

Mathématiquement, cela revient à faire passer la matrice $X$ à travers le réseau de neurones $s_{\theta}(\mathbf{x})$ : pour chaque vecteur on obtient un gradient de dimension deux, et on calcule la norme pour chaque observation, c'est-à-dire $\sqrt{x_{11}^2+x_{12}^2}$.

$$
[X]_{1000 \times 2} \rightarrow [s_{\theta}(\mathbf{x})]_{1000 \times 2} \rightarrow \Big[||s_{\theta}(\mathbf{x})||_2^2\Big]_{1000 \times 1}
$$

**(3) Calcul de la trace. (i) Calcul du Jacobien.** On fait d'abord une propagation avant pour obtenir la sortie $s_{\theta, 1}(x)$, puis on fait une rétropropagation et on ne prend que la première dérivée.

![[im3 (4).png]]

On poursuit cette procédure pour la deuxième composante : on répète exactement le même processus jusqu'à avoir tous les éléments de la diagonale. On note $D$ la dimension de l'entrée. **Sur ImageNet, on a des millions de dimensions : ce ne sera pas faisable à cause de la taille, donc cette approche ne peut pas être utilisée pour calculer la trace.**

Mathématiquement, on obtient

$$
[\nabla_{\mathbf{x}} s_{\theta}(\mathbf{x})]_{1000 \times 2 \times 2}
$$

**(ii) Calcul de la trace.** Quand notre gradient est un tenseur de dimension trois, la trace est simplement l'empilement des diagonales du tenseur. Par exemple, si on a le tenseur suivant :

![[images/3-Apprentissage automatique/Generative Models/score based/im4 (1).png|484]]

**(4) Visualisation (plotting).** À partir de ce qu'on a dit précédemment, on affirme que

$$
[s_{\theta}(\mathbf{x})]_{10k \times 2} \approx \nabla_{x} \log p(x)
$$

Pour visualiser le champ de gradient, on commence par tracer l'ensemble des points de données $[X]$. Ensuite, pour chaque point on a son vecteur $s_{\theta}(x)$, qui constitue le point d'arrivée du vecteur.

![[images/3-Apprentissage automatique/Generative Models/score based/im1-2.png|295]]

**(5) Échantillonnage (sampling).** Une fois le champ de vecteurs appris, on peut tirer parti de la dynamique de Langevin pour produire de vrais échantillons de la densité $\mathbf{p(x)}$ en s'appuyant uniquement sur $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. L'échantillonnage est défini d'une manière très similaire aux approches MCMC, en appliquant récursivement la formule suivante, où on a posé ici $x_0 = (1.5, -1.5)$ :

$$
\begin{aligned}
\mathbf{x}_{t+1}&=\mathbf{x}_{t}+\frac{\epsilon}{2} \nabla_{\mathbf{x}_{t}} \log p\left(\mathbf{x}_{t}\right)+\sqrt{\epsilon} \mathbf{z}_{t}\\
&= \mathbf{x}_{t}+\frac{\epsilon}{2} s_{\theta}(x_t)+\sqrt{\epsilon} \mathbf{z}_{t}
\end{aligned}
$$

où $\mathbf{z}_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Il a été montré par [Welling et al. (2011)](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) que sous $\epsilon \rightarrow 0$ et $t \rightarrow \infty$, $\mathbf{x}_{t}$ converge vers un échantillon exact de $p(\mathbf{x})$. C'est l'idée-clé derrière l'approche de modélisation générative basée sur les scores.

![[im1-3 (1).png|330]]

> [!warning] Le verrou : la trace coûte $D$ backward
> Sur le jouet 2D ci-dessus, $D=2$ donc 2 backward auxiliaires par évaluation de la loss : OK. Sur ImageNet, $D \approx 2 \cdot 10^5$ : **infaisable**. D'où deux familles de solutions pour remplacer $\operatorname{tr}(\nabla_x s_\theta(x))$ :
> - **Sliced score matching** : estimateur d'Hutchinson, $\operatorname{tr}(A) \approx v^\top A v$ → 1 backward au lieu de $D$.
> - **Denoising score matching** : on bruite les données pour faire disparaître tout Jacobien — il ne reste qu'une MSE. Cette piste mènera aux diffusion models.


## III. Scalable Score Matching

La clé pour rendre l'objectif de score matching utilisable est de résoudre le problème de la trace. Deux méthodes :

- Denoising Score Matching
- Sliced Score Matching

### A. Denoising Score Matching

*Pascal Vincent, 2011.*

**Idée centrale.** Au lieu d'estimer le score de la distribution originale, on estime celui d'une version perturbée par bruit. On prend un noyau de perturbation $q_\sigma(\tilde{x}\mid x)$. Quand $\sigma$ est très petit, $q$ et $p$ sont très proches (l'approximation est fidèle). Intuitivement, $q$ est obtenue en ajoutant un bruit à $p(x)$, et quand $\sigma = 0$ il n'y a plus de bruit donc $q = p$ exactement.

$$
\begin{aligned}
q_{\sigma}(\tilde{x} \mid x) &= \mathcal{N}(\tilde{x}; x, \sigma^{2} I), \qquad q_{\sigma}(\tilde{x}) = \int p(x)\, q_{\sigma}(\tilde{x} \mid x)\, dx \\
q_{\sigma}(\tilde{x}) &\approx p(\tilde{x})
\end{aligned}
$$

Avec ce noyau, on réécrit la fonction de score matching :

$$
\begin{aligned}
& \frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{x})}\!\left[\left\|\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x}) - s_{\theta}(\tilde{x})\right\|_{2}^{2}\right] \\
=& \frac{1}{2} \mathbb{E}_{p(x)}\, \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x)}\!\left[\left\|\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x) - s_{\theta}(\tilde{x})\right\|_{2}^{2}\right] + \text{const.}
\end{aligned}
$$

Cette nouvelle forme est tractable, chaque composante est facile à calculer. La première espérance s'évalue par échantillonnage de $p$. La seconde, par rapport au noyau de perturbation, est facile aussi : le gradient est relié à une gaussienne et se calcule explicitement :

$$\nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^{2}}.$$

Au final :

$$\boxed{\;l(\theta;\sigma) = \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x)}\, \mathbb{E}_{x \sim p(x)}\!\left[\left\| s_{\theta}(\tilde{x}) + \frac{\tilde{x}-x}{\sigma^{2}} \right\|_{2}^{2} \right]\;}$$

**Limite.** L'estimation du score est biaisée parce qu'on l'évalue sur une version perturbée de $p$. Et le choix de $\sigma$ est délicat : on le veut petit, mais pas trop petit non plus, sinon le dénominateur risque de tendre vers zéro.

**Algorithme.**

1. On part du dataset $[X]_{1000 \times 2}$ et on perturbe chaque observation par un bruit gaussien : $[\tilde{X}]_{1000 \times 2}$.
2. On calcule le terme de droite : $\frac{\tilde{x} - x}{\sigma^{2}}$.
3. Le terme de gauche se calcule comme d'habitude, mais cette fois sur le dataset perturbé : $s_{\theta}(\tilde{x})$.
4. Il ne reste qu'à calculer la norme.

#### Exemple

**(1) Inputs.** Soit la matrice d'entrée $[X]_{1000 \times 2}$ (chaque point en dimension deux). On définit un réseau de neurones $s_{\theta}(x)$ :

| Réseau de neurones                                                                   | Inputs                                                                               |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| ![[images/3-Apprentissage automatique/Generative Models/score based/im1-4.png\|368]] | ![[images/3-Apprentissage automatique/Generative Models/score based/im1-1.png\|263]] |
*Figures. blabla*

**(2) Fonction de perte.** On rappelle que le réseau optimal vérifiant $s_{\theta}(x) \approx \nabla_x \log p(x)$ est obtenu en minimisant :

$$l(\theta;\sigma) = \mathbb{E}_{q_{\sigma}(\tilde{x} \mid x)}\, \mathbb{E}_{x \sim p(x)}\!\left[\left\| s_{\theta}(\tilde{x}) + \frac{\tilde{x}-x}{\sigma^{2}} \right\|_{2}^{2} \right]$$

Attention : $s_{\theta}(x) = \nabla_x \log q_\sigma(x) \approx \nabla_x \log p(x)$ n'est vrai que si le bruit est suffisamment petit pour avoir $q_\sigma(x) \approx p(x)$.

**(a) Matrice perturbée.** On fixe $\sigma = 0.01$ :

$$
\begin{aligned}
\tilde{X} &\sim q_{\sigma}(\tilde{x} \mid x) = \mathcal{N}(\tilde{x} \mid x, \sigma^{2} I) \\
\tilde{X} &= [X]_{1k \times 2} + [U]_{1k \times 2} \times \sigma, \qquad U \sim \mathcal{N}(0, I)
\end{aligned}
$$

**(b) Terme de droite.** Différence entre les deux matrices, divisée par une constante.

**(c) Terme de gauche.** On passe la matrice perturbée dans le MLP, qui retourne $[s_{\theta}(\tilde{x})]_{1k \times 2}$.

**(3) Visualisation.** Pour visualiser le champ de gradient, on part du dataset $[X]$ ; le point d'arrivée de chaque vecteur est $s_{\theta}(x)$.

![[images/3-Apprentissage automatique/Generative Models/score based/im1-2.png|295]]

### B. Sliced score matching

*Yang Song et al., UAI 2019.*

**Idée centrale.** Faire le calcul en dimension 1 est plus facile qu'en dimension multiple. On y arrive par **projections aléatoires** : on projette les champs de vecteurs sur des directions aléatoires (lignes en pointillé), de sorte qu'ils deviennent des champs scalaires.

![[images/3-Apprentissage automatique/Generative Models/score based/im5.png]]

Ensuite, on projette tous les vecteurs sur ces deux droites aléatoires. On obtient des scalaires faciles à manipuler. Les deux champs vectoriels sont proches l'un de l'autre si et seulement si leurs projections aléatoires le sont aussi.

![[images/3-Apprentissage automatique/Generative Models/score based/im6 (2).png]]

**Fonction objectif.** On définit une nouvelle fonction objectif, la **Sliced Fisher Divergence**. C'est l'erreur quadratique entre les projections du score de $p$ et du score modèle :

$$\frac{1}{2}\, \mathbb{E}_{p_v}\, \mathbb{E}_{p(x)}\!\left[\left(v^{\top} \nabla_{x} \log p(x) - v^{\top} s_{\theta}(x)\right)^{2}\right]$$

où :

- $v$ est la direction de projection aléatoire
- $p_v$ est la distribution de ce vecteur

**Dérivation.** Comme d'habitude, on résout par intégration par parties et on retombe sur le même schéma qu'avant : on a un terme intractable, le $\nabla_x \log p(x)$. Après dérivation, on obtient :

$$\boxed{\;\mathbb{E}_{p_v}\, \mathbb{E}_{p(x)}\!\left[v^{\top} \nabla_{x} s_{\theta}(x)\, v + \frac{1}{2}\left(v^{\top} s_{\theta}(x)\right)^{2}\right] + \text{const}\;}$$

**(i) Premier terme.** Ce n'est plus la trace du Jacobien : ça dépend uniquement d'un **produit Jacobien-vecteur**, similaire à une multiplication matricielle en dimension 3. La seule question : ce produit est-il rapide à calculer ?

$$\begin{pmatrix} v_{1} & v_{2} & v_{3} \end{pmatrix} \begin{pmatrix} \frac{\partial s_{\theta, 1}}{\partial x_{1}} & \frac{\partial s_{\theta, 1}}{\partial x_{2}} & \frac{\partial s_{\theta, 1}}{\partial x_{3}} \\ \frac{\partial s_{\theta, 2}}{\partial x_{1}} & \frac{\partial s_{\theta, 2}}{\partial x_{2}} & \frac{\partial s_{\theta, 2}}{\partial x_{3}} \\ \frac{\partial s_{\theta, 3}}{\partial x_{1}} & \frac{\partial s_{\theta, 3}}{\partial x_{2}} & \frac{\partial s_{\theta, 3}}{\partial x_{3}} \end{pmatrix} \begin{pmatrix} v_{1} \\ v_{2} \\ v_{3} \end{pmatrix}$$

**(ii) Second terme.** Rapide : c'est juste un produit scalaire.

**Le produit Jacobien-vecteur est scalable.** On le réécrit :

$$v^{\top} \nabla_{x} s_{\theta}(x)\, v = v^{\top} \nabla_{x}\!\left(v^{\top} s_{\theta}(x)\right)$$

Les deux expressions sont égales parce que $v$ ne dépend pas de $x$, on peut donc échanger l'ordre du produit scalaire. Cette alternative est facile à calculer :

On fait d'abord un forward pour calculer $s_{\theta}(x)$, puis le produit scalaire (équivalent à ajouter un nœud, c'est-à-dire une couche linéaire). On obtient un scalaire (le produit interne). On peut ensuite rétropropager à travers ce calcul : ça donne le gradient du produit interne.

![[images/3-Apprentissage automatique/Generative Models/score based/im7.png|494]]

Le dernier produit scalaire est la procédure inverse. **Une seule passe de backprop suffit** pour évaluer l'objectif du sliced score matching — bien mieux que le score matching original.

![[images/3-Apprentissage automatique/Generative Models/score based/im8 (1).png]]

**Limite.** Le sliced peut être plus lent que le denoising score matching. L'objectif du sliced demande un backprop, alors que le denoising ne demande qu'un forward — ce qui peut rendre le sliced jusqu'à 4× plus lent.

**Vérification empirique de la scalabilité.** En abscisse la dimension des données, en ordonnée le temps par itération pour calculer l'objectif (plus bas = mieux). En marron : score matching original ; SSM : sliced ; SSM-VR : une variante ; DSM : denoising ; CP et approx-BP : autres méthodes d'accélération, sans garantie théorique et parfois mauvaises en pratique. Le sliced est nettement plus rapide que le score matching original, qui sature en mémoire au-delà de 300 dimensions. DSM est légèrement meilleur.

![[im9 (2).png|288]]

#### Exemple

**(1) Inputs.** Soit la matrice d'entrée $[X]_{1000 \times 2}$ (chaque point en dimension deux). On définit un réseau de neurones $s_{\theta}(x)$ :

| Réseau de neurones                                                                   | Inputs                                                                               |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| ![[images/3-Apprentissage automatique/Generative Models/score based/im1-4.png\|368]] | ![[images/3-Apprentissage automatique/Generative Models/score based/im1-1.png\|263]] |
*Figures. blabla*

**(2) Fonction de perte.** On rappelle :

$$\mathbb{E}_{p_v}\, \mathbb{E}_{p(x)}\!\left[v^{\top} \nabla_{x} s_{\theta}(x)\, v + \frac{1}{2}\left(v^{\top} s_{\theta}(x)\right)^{2}\right] + \text{const}$$

où $v$ est la direction de projection aléatoire et $p_v$ sa distribution.

**(a) Projection aléatoire.** On tire d'abord $[\tilde{v}]_{1k \times 2} \sim \mathcal{N}(0, I)$, puis on normalise :

$$v = \frac{\tilde{v}}{\|\tilde{v}\|}$$

Les deux termes sont calculés avec la fonction `autograd.functional.jvp()`, qui retourne deux vecteurs : `logp` et `jvp`.

**(b) Terme de droite.** La fonction $s_\theta(x)$ est `logp`. Il suffit de faire le produit scalaire entre le vecteur aléatoire et `logp`.

![[images/3-Apprentissage automatique/Generative Models/score based/im7 (1).png|544]]

**(c) Terme de gauche.** Le gradient de $v^\top s_\theta(x)$ est le terme `jvp`. On refait un produit scalaire à droite avec $v$ pour obtenir le terme de gauche.

![[im8 (2).png]]

**(3) Visualisation.** Pour visualiser le champ de gradient, on part du dataset $[X]$ ; le point d'arrivée de chaque vecteur est $s_{\theta}(x)$.

![[images/3-Apprentissage automatique/Generative Models/score based/im1-2.png|295]]

## IV. Denoising Score Matching avec Langevin Dynamics (SMLD)

*Yang Song & Stefano Ermon, NeurIPS 2019.*

**De l'estimation du score à la génération.** Depuis le début du chapitre, on suppose que disposer du champ de scores permet de générer de nouveaux échantillons. L'idée : tirer des points aléatoires, puis les déplacer le long du gradient pour qu'ils ressemblent à la distribution originale. Les échantillons suivent alors la direction du gradient :

![[images/3-Apprentissage automatique/Generative Models/score based/im12.png]]

Avec uniquement le gradient, les échantillons restent piégés (image de gauche). En ajoutant un bruit aléatoire, le résultat est bien meilleur (image de droite).

![[images/3-Apprentissage automatique/Generative Models/score based/im13.png]]

**Échantillonnage par dynamique de Langevin.** Technique standard pour échantillonner $p(x)$ en n'utilisant que son score $\nabla_x \log p(x)$ :

1. Initialisation : $\tilde{x}_0 \sim \pi(x)$ (distribution prior simple).
2. Pour $t = 1, 2, \ldots, T$ :

$$\begin{aligned} z_t &\sim \mathcal{N}(0, I) \\ \tilde{x}_t &\leftarrow \tilde{x}_{t-1} + \frac{\epsilon}{2}\, \nabla_x \log p(\tilde{x}_{t-1}) + \sqrt{\epsilon}\, z_t \end{aligned}$$

À chaque étape on tire un vecteur gaussien et on perturbe le score avec une version mise à l'échelle. Toute la procédure ne dépend que du score. En pratique on ne prend pas $\epsilon$ infiniment petit ni $T$ infiniment grand : il y a une petite erreur, mais plusieurs papiers montrent qu'on peut l'ignorer.

**Modélisation générative basée sur le score.** Le framework est formalisé par [Song & Ermon (2019)](https://arxiv.org/abs/1907.05600). On dispose uniquement d'échantillons de la distribution de données ; on apprend le score par score matching, puis on échantillonne par dynamique de Langevin.

![[images/3-Apprentissage automatique/Generative Models/score based/im14.png|292]]

### Trois pièges qui font tout planter

**Piège 1 : l'hypothèse de variété (manifold hypothesis).** En pratique, une grande partie des données vit sur une variété de dimension plus faible que celle de l'espace ambiant. Sur l'image ci-dessous, les points 3D vivent en réalité sur une bande 2D. Dans ce cas, le score n'est **pas défini** :

![[images/3-Apprentissage automatique/Generative Models/score based/im15.png|231]]

$$\nabla_x \log p_{\text{data}}(x) \;\text{ n'existe pas — la densité est singulière sur la variété.}$$

Intuition : si la distribution est concentrée autour d'un anneau, plus l'anneau devient fin, plus le score devient grand. À la limite (épaisseur nulle, variété 1D dans un espace 2D), le score explose.

![[images/3-Apprentissage automatique/Generative Models/score based/im16.png|254]]

*Test empirique.* On ajuste les données avec un algorithme de variété linéaire (PCA). Sur MNIST, les images font 784 dimensions, mais PCA trouve une sous-variété linéaire de dimension 595 qui reconstruit fidèlement les images d'origine. Avec une variété non-linéaire la dimension serait encore plus petite.

![[images/3-Apprentissage automatique/Generative Models/score based/im17.png|514]]

![[images/3-Apprentissage automatique/Generative Models/score based/im18.png|438]]

*Conséquence sur CIFAR-10.* La courbe d'apprentissage avec SSM décroît rapidement puis explose en oscillations à cause de problèmes numériques.

![[images/3-Apprentissage automatique/Generative Models/score based/im19.png|190]]

**Piège 2 : score imprécis dans les zones de faible densité.** L'estimateur Monte Carlo de l'objectif s'écrit :

$$\frac{1}{2}\, \mathbb{E}_{p_{\text{data}}}\!\left[\|\nabla_x \log p_{\text{data}}(x) - s_\theta(x)\|_2^2\right] \approx \frac{1}{2N} \sum_{i=1}^N \|\nabla_x \log p_{\text{data}}(x_i) - s_\theta(x_i)\|_2^2$$

Vu comme un jeu enseignant/élève : l'enseignant tire $x \sim p_{\text{data}}$, l'élève approxime le score en ce point. Le problème est que dans les régions jamais échantillonnées (les zones « ? » sur le graphe), l'élève n'a aucune information.

![[images/3-Apprentissage automatique/Generative Models/score based/im25.png|402]]

Sur le même jouet 2D : à gauche le score idéal, à droite le score appris — précis uniquement dans la zone rouge (où il y a des données), faux ailleurs.

![[images/3-Apprentissage automatique/Generative Models/score based/im20.png]]

**Piège 3 : mauvais mélange de Langevin entre modes.** Soit un mélange à deux modes à supports disjoints :

$$p_{\text{data}}(x) = \pi p_1(x) + (1-\pi)\, p_2(x).$$

Sur le support du premier mode, le score ne dépend pas de $\pi$ (et idem pour le second). Comme Langevin n'utilise *que* le score, **il ne peut pas retrouver $\pi$** : il va échantillonner les deux modes mais avec des poids relatifs faux. En pratique on retrouve la position des modes mais pas leurs poids — c'est un gros problème.

![[images/3-Apprentissage automatique/Generative Models/score based/im26.png]]

![[images/3-Apprentissage automatique/Generative Models/score based/im27.png|519]]

### Solution unifiée : perturbation gaussienne multi-échelle

**Perturbation gaussienne.** L'astuce qui résout les trois pièges d'un coup : perturber les données avec du bruit gaussien. Le gradient de la log-densité perturbée est alors **bien défini partout**, et en pratique ça marche très bien.

![[images/3-Apprentissage automatique/Generative Models/score based/im28.png|527]]

Mais une perturbation trop grande détruit l'information utile. D'où un trade-off : gros bruit pour couvrir les zones de faible densité, petit bruit pour rester fidèle à $p_{\text{data}}$. La solution est d'utiliser **une séquence de variances** décroissantes pour récupérer l'information à plusieurs granularités :

$$\sigma_1 > \sigma_2 > \cdots > \sigma_{L-1} > \sigma_L$$

![[images/3-Apprentissage automatique/Generative Models/score based/im21.png|530]]

**Annealed Langevin Dynamics.** L'idée est simple :

- Échantillonner successivement avec $\sigma_1, \sigma_2, \ldots, \sigma_L$ via Langevin à chaque échelle.
- Réduire progressivement le pas (step size).

![[images/3-Apprentissage automatique/Generative Models/score based/im27.png|519]]

Algorithme :

![[im31 (1).png|350]]

Comparaison avec la dynamique de Langevin vanilla :

![[images/3-Apprentissage automatique/Generative Models/score based/im22.png|558]]

**Noise Conditional Score Networks (NCSN).** Plutôt que d'entraîner $L$ réseaux séparés (un par niveau de bruit), on utilise **un seul réseau conditionné sur la variance** : il prend $(x, \sigma)$ en entrée et estime le score correspondant à chaque niveau.

![[images/3-Apprentissage automatique/Generative Models/score based/im23.png|556]]

Distribution perturbée :

$$q_\sigma(x) \equiv \int p_{\text{data}}(t)\, \mathcal{N}(x \mid t, \sigma^2 I)\, dt$$

Réseau :

$$s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)$$

![[images/3-Apprentissage automatique/Generative Models/score based/im32.png|483]]

Loss finale, somme pondérée des objectifs de denoising score matching à chaque échelle :

$$\boxed{\;\mathcal{L}_{\{\sigma_i\}_{i=1}^L}(\theta) = \frac{1}{L} \sum_{i=1}^L \sigma_i^2\, \mathbb{E}_{p_{\text{data}}}\, \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)}\!\left[\left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x}-x}{\sigma_i^2}\right\|_2^2\right]\;}$$

### Résultats

**Échantillonnage.** On part d'un gros bruit et on raffine : à mesure que $\sigma$ diminue, les samples deviennent réalistes. Pour la première fois, un modèle non-adversarial égale ou surpasse les GAN sur la qualité visuelle — alors que les modèles à vraisemblance n'y parvenaient pas.

![[images/3-Apprentissage automatique/Generative Models/score based/im24.png|273]]

![[im34.png|339]]

**Plus proche voisin.** À gauche les vraies images, à droite leurs voisins les plus proches dans les samples générés — preuve que le modèle ne fait pas que recopier les données d'entraînement.

![[images/3-Apprentissage automatique/Generative Models/score based/im33.png|299]]

**Inpainting.** La moitié droite de l'image est masquée, le réseau conditionné complète la zone manquante. À droite : reconstruction complète.

![[im35.png|295]]

#### Exemple complet

**(1) Inputs.** Matrice d'entrée $[X]_{10 \times 2}$ (chaque point en dimension deux).

**(2) Fonction de perte.**

$$\mathcal{L}_{\{\sigma_i\}_{i=1}^L}(\theta) = \frac{1}{L} \sum_{i=1}^L \sigma_i^2\, \mathbb{E}_{p_{\text{data}}}\, \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)}\!\left[\left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x}-x}{\sigma_i^2}\right\|_2^2\right]$$

**(a) Matrice perturbée.** Cette fois on ne fixe pas $\sigma$. Pour chaque observation, on tire uniformément un entier dans $\{0, 1, 2, 3\}$. Chaque entier correspond à une variance, par exemple $\sigma_0 = 1.0$, $\sigma_2 = 0.046$ :

$$[\text{labels}]_{10k \times 1} = \begin{pmatrix} 3 \\ 2 \\ \vdots \\ 1 \end{pmatrix}, \quad \tilde{\sigma} = \begin{pmatrix} 1.0 & 0.21 & 0.046 & 0.01 \end{pmatrix} \;\Rightarrow\; [\sigma]_{10k \times 1} = \begin{pmatrix} 0.01 \\ 0.046 \\ \vdots \\ 0.21 \end{pmatrix}$$

On perturbe ensuite le dataset :

$$[\tilde{X}]_{10k \times 2} = X + [U]_{10k \times 2} \times [\sigma]_{10k \times 1}, \qquad U \sim \mathcal{N}(0, I)$$

**(b) Terme de droite.** Différence entre les deux matrices, pondérée ligne par ligne par les variances correspondantes.

**(c) Terme de gauche.** On nourrit le réseau $s_\theta(\tilde{x}, \sigma_i)$ avec la matrice perturbée et ses labels associés. Le réseau apprend des embeddings pour les quatre niveaux de bruit.

![[images/3-Apprentissage automatique/Generative Models/score based/im4-1 (1).png]]

**(3) Visualisation.** Pour visualiser le champ de gradient, on utilise le dataset $X$ avec les vecteurs $s_\theta(\tilde{x}, \sigma_i)$ comme points d'arrivée (cela demande les labels et les données perturbées).

![[images/3-Apprentissage automatique/Generative Models/score based/im1-2.png|295]]

## V. Modèles de diffusion discrets

### A. Sohl-Dickstein (2015) — Non-Equilibrium Thermodynamics

*Sohl-Dickstein et al., ICML 2015.* Inspiré de la thermodynamique hors équilibre : on définit un processus de bruitage progressif (forward) qui détruit la donnée, et on apprend à le renverser (reverse).

![[images/3-Apprentissage automatique/Generative Models/score based/im5-1.png]]

**Processus forward (diffusion).** À chaque étape on injecte une quantité de bruit pré-définie. Le noyau de diffusion markovien avec un taux $\beta_t$ est :

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\big).$$

![[im5-4.png]]

La distribution complète sur toute la trajectoire est appelée le **processus de diffusion** :

$$q(x_{0:T}) = q(x_0) \prod_{t=1}^T q(x_t \mid x_{t-1}).$$

**Forward à un instant arbitraire.** Avantage important : on peut échantillonner $x_t$ directement à un pas $t$ quelconque sans passer par toute la chaîne. Avec les notations $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ :

$$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, I\big).$$

En pratique, cela dépend du *variance schedule* $\beta_1, \ldots, \beta_T$ fixé à l'avance. Ici on prend $T = 100$ :

$$
\begin{aligned}
[\beta]_{1 \times 100} &= \begin{pmatrix} 3.47 \cdot 10^{-5} & 3.79 \cdot 10^{-5} & 4.15 \cdot 10^{-5} & \cdots & 9.97 \cdot 10^{-3} \end{pmatrix} \\
[\alpha]_{1 \times 100} &= 1 - \beta = \begin{pmatrix} 1.0000 & 1.0000 & \cdots & 0.9900 \end{pmatrix} \\
[\bar{\alpha}]_{1 \times 100} &= \begin{pmatrix} 1.00 & 0.999 & \cdots & 0.605 \end{pmatrix} \\
x_t &= \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I).
\end{aligned}
$$

**Modèle paramétrique (reverse).** On apprend un modèle qui transforme un bruit aléatoire en image de haute qualité. C'est conceptuellement similaire à l'inférence variationnelle :

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\big).$$

La moyenne $\mu_\theta(x_t, t)$ et la covariance $\Sigma_\theta(x_t, t)$ sont paramétrées par des réseaux de neurones. Notez qu'elles sont aussi paramétrées par $t$ : un seul modèle sert pour tous les pas de temps.

**Processus paramétrique.** Le reverse part de la distribution tractable $p(x_T) = \pi(x_T)$ (typiquement $\mathcal{N}(0, I)$) :

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t).$$

#### Exemple

**(1) Inputs.** Données $[X_0]_{10k \times 2} \sim q(x_0)$. On travaille en batches de taille 128, construits par assignation aléatoire.

**(a) Vecteur de permutation.** On construit un vecteur de permutation où chaque ligne est l'indice d'une ligne à permuter. On ne garde que les 128 premières lignes (le batch). $[\text{batch}_x]$ contient les valeurs réelles correspondant aux indices.

$$[\text{perm}]_{10k \times 1} = \begin{pmatrix} 1073 \\ 9503 \\ \vdots \\ 5733 \end{pmatrix} \Rightarrow [\text{perm}]_{128 \times 1} \begin{pmatrix} 1073 \\ 9503 \\ \vdots \end{pmatrix} \Rightarrow [\text{batch}_x]_{128 \times 2}$$

**(b) Pas de temps aléatoires (antithetic sampling).** On choisit un $t$ aléatoire pour chaque observation du batch. Cet échantillonnage antithétique permet d'entraîner conjointement des points symétriques dans les différentes chaînes.

$$[t]_{1 \times 128} = \begin{pmatrix} 59 & 79 & 53 & \cdots & 36 \end{pmatrix}$$

**(c) Diffusion.** Chaque observation est mappée à un pas de temps différent. On peut donc échantillonner :

$$
\begin{aligned}
\begin{pmatrix} \vline & t_{59} \\ x_0 & t_{53} \\ \vline & t_{36} \end{pmatrix}
&\Rightarrow q(x_{60} \mid x_{59}) \sim \alpha_{59} x_0 + \sqrt{1 - \alpha_{59}}\, \varepsilon \\
&\Rightarrow q(x_{53} \mid x_{52}) \sim \alpha_{53} x_0 + \sqrt{1 - \alpha_{53}}\, \varepsilon
\end{aligned}
$$

Ayant $x_t$, $x_0$ et $t$, on peut calculer la postérieure forward :

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I\big)$$

avec

$$\tilde{\mu}_t(x_t, x_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t, \qquad \tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\, \beta_t.$$

**(d) Reverse.** On passe le dataset (après diffusion) avec le vecteur de temps $[t]_{1 \times 128}$ dans le réseau :

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\big)$$

![[im5-2.png]]

Le réseau retourne les deux matrices $[\mu]$ et $[\Sigma]$.

**(e) Loss.** Borne variationnelle (similaire à un ELBO) :

$$K = -\mathbb{E}_q\Big[D_{\mathrm{KL}}\big(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\big) + H_q(X_T \mid X_0) - H_q(X_1 \mid X_0) - H_p(X_T)\Big]$$

**(f) Échantillonnage.** Partant de l'extrême gauche de la figure : on génère $[X]_{10k \times 2} \sim \mathcal{N}(0, I)$. Avec $T = 100$, on commence à $t = 99$. On tire $[\mu_{99}]$ et $[\Sigma_{99}]$ depuis le réseau et :

$$[\text{sample}]_{10k \times 2} = [\mu_{99}]_{10k \times 2} + [\Sigma_{99}]_{10k \times 2} \times \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I).$$

On trace toutes les 10 itérations, de 100 à 0 :

![[im5-3.png]]

### B. DDPM (Ho et al., 2020)

*[Ho, Jain & Abbeel, NeurIPS 2020](https://arxiv.org/abs/2006.11239).* Reformulation moderne du modèle de Sohl-Dickstein avec une **loss simplifiée** qui prédit directement le bruit $\epsilon$ plutôt que de manipuler les KL — c'est ce qui rend DDPM élégant et entraînable à grande échelle.

#### Exemple

Le setup d'entraînement est identique à la section précédente (batches, permutation, pas de temps aléatoires). On se concentre ici sur ce qui change.

**(1) Diffusion.** Postérieure forward conditionnée sur $x_0$ (tractable) :

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t I\big)$$

Moyenne et variance correspondantes :

$$
\begin{aligned}
\tilde{\mu}_t(x_t, x_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\bar{\alpha}_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\, \beta_t
\end{aligned}
$$

**(2) Reverse.** *On n'a pas besoin d'utiliser ces formules en pratique car on utilise une "loss simplifiée".* Le réseau reverse reste de la forme :

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1};\, \mu_\theta(x_t, t),\, \Sigma_\theta(x_t, t)\big)$$

où

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\Big).$$

Le bruit $[\epsilon_\theta(\cdot)]_{10k \times 2}$ est prédit par le réseau :

![[im6-1.png]]

Pour générer un point, on utilise plutôt :

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\Big) + \sigma_t z.$$

**(3) Fonction de perte (similaire à la loss du denoising score matching).** On tire $[\epsilon]_{10k \times 2} \sim \mathcal{N}(0, I)$. On calcule les $\alpha$ pour chaque observation, on passe le tout dans le réseau, qui retourne $[\text{out}]_{10k \times 2}$ à comparer à $\epsilon$ :

$$\boxed{\;\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\, t\big)\big\|^2\right]\;}$$

**(4) Visualisation.** On tire $[z]_{10k \times 2} \sim \mathcal{N}(0, I)$ et on passe la matrice initiale $[X]$ dans le réseau avec les pas de temps aléatoires $t$ :

$$[\text{sample}] = \text{mean} + \sigma_t \times z$$

c'est-à-dire

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\Big) + \sigma_t z.$$

On trace toutes les 10 itérations :

![[im6-2.png]]

### C. DDIM — Sampling rapide et déterministe

*[Song, Meng & Ermon, ICLR 2021](https://arxiv.org/abs/2010.02502).* DDPM nécessite typiquement **1000 pas** de sampling pour générer une image — c'est lent. **DDIM** (Denoising Diffusion Implicit Models) réduit ça à **10–50 pas** tout en réutilisant le **même réseau DDPM entraîné** (pas de réentraînement nécessaire).

**Idée clé.** DDIM redéfinit le processus forward comme une **famille non-markovienne** paramétrée par un paramètre de stochasticité $\sigma_t \in [0,\, \sqrt{\tilde{\beta}_t}]$ :

- $\sigma_t = \sqrt{\tilde{\beta}_t}$ → on retombe sur DDPM (sampling stochastique).
- $\sigma_t = 0$ → **DDIM déterministe** (sampling purement ODE-like, sans bruit ajouté à chaque étape).

Le point crucial : tous ces processus **partagent les mêmes marginales** $q(x_t \mid x_0)$ que DDPM. Donc le réseau $\epsilon_\theta(x_t, t)$ déjà appris pour DDPM est valide pour toute la famille.

**Formule de sampling DDIM (cas déterministe $\sigma_t = 0$).** À chaque pas, on procède conceptuellement en deux étapes :

$$\boxed{\;x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\, \underbrace{\left(\frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{prédiction de } x_0 \text{ depuis } x_t} + \sqrt{1 - \bar{\alpha}_{t-1}}\, \epsilon_\theta(x_t, t)\;}$$

Lecture intuitive :

1. **Estimer $x_0$** à partir de $(x_t,\, \epsilon_\theta)$ en inversant la formule du forward $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$.
2. **Projeter en avant** depuis cet $\hat{x}_0$ vers le pas $t-1$.

Comme le processus est déterministe, on peut **sauter directement** de $t$ à $t-k$ (au lieu de $t \to t-1$) sans accumuler de bruit — c'est ce qui permet d'utiliser 10–50 pas au lieu de 1000.

**Avantages pratiques.**

- **10–50× plus rapide.** 20 pas DDIM donnent une qualité comparable à 1000 pas DDPM.
- **Reproductible.** Sampling déterministe ⇒ même seed = même image, à chaque fois.
- **Inversion possible.** On peut **remonter** de $x_0$ vers $x_T$ déterministiquement, ce qui permet l'**édition d'image** (img2img, inpainting déterministe, interpolation entre deux images dans l'espace des bruits).
- **Pas de réentraînement.** Un modèle DDPM préentraîné s'utilise directement avec DDIM.

> [!important] DDIM est la base de tous les samplers modernes
> Tous les samplers utilisés dans Stable Diffusion (**DPM-Solver**, **Euler**, **Euler a**, **Heun**, **DPM++ 2M Karras**…) sont des raffinements de l'idée DDIM : sampling déterministe par résolution numérique d'une ODE associée à la diffusion. Quand tu vois ces options dans une UI de génération d'image, ce sont toutes des variantes de DDIM avec des solveurs d'ODE plus précis ou plus rapides.

### Le pont DDPM ↔ SMLD

À première vue DDPM (qui prédit le bruit $\epsilon$) et SMLD (qui prédit le score $\nabla_x \log p_t(x)$) semblent deux paradigmes distincts. **C'est en fait la même tâche, à un facteur d'échelle près.**

Le forward DDPM est $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$, ce qui donne le score conditionnel exact :

$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}\, x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}.$$

Le réseau DDPM apprend $\epsilon_\theta(x_t, t) \approx \epsilon$ et le réseau SMLD apprend $s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$. Les deux quantités sont reliées par une simple normalisation :

$$\boxed{\;s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}\;}$$

Donc *prédire le bruit* (paramétrisation Ho 2020) et *prédire le score* (paramétrisation Song 2019) ne sont **que deux conventions** pour la même fonction sous-jacente. C'est ce pont qui motive l'unification SDE de la section suivante.

## VI. Modèles de diffusion continus (SDE)

### A. Le cadre score-based / SDE

*[Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, ICLR 2021](https://arxiv.org/abs/2011.13456).* Généralisation continue qui unifie SMLD et DDPM sous un même formalisme : une équation différentielle stochastique.

**Définition (processus de diffusion).** Un processus de diffusion est un processus stochastique similaire au mouvement brownien. Soit $\{x(t) \in \mathbb{R}^d\}_{t=0}^T$ un processus indexé par $t \in [0, T]$. Il est défini par une SDE d'Itô :

$$\boxed{\;dx = \underbrace{f(x, t)}_{\text{coef. de drift}}\, dt + \underbrace{g(t)}_{\text{coef. de diffusion}}\, \underbrace{dw}_{\text{mvt. brownien}}\;}$$

où $f(\cdot, t) : \mathbb{R}^d \to \mathbb{R}^d$ est le **coefficient de drift**, $g(t) \in \mathbb{R}$ est le **coefficient de diffusion**, et $w$ est un mouvement brownien standard.

**Intuition physique.** Des particules suivant une SDE ne se contentent pas de suivre le drift déterministe $f(x, t)$ : elles sont aussi affectées par le bruit aléatoire $g(t)\, dw$. C'est comme la trajectoire d'une particule plongée dans un fluide en mouvement, qui bouge aléatoirement à cause de collisions imprévisibles. On note $p_t(x)$ la distribution de $x(t)$.

**Setup score-based.** On choisit un processus de diffusion tel que $x(0) \sim p_0$ (distribution des données, accessible par échantillons) et $x(T) \sim p_T$ (distribution prior, tractable et facile à échantillonner). La perturbation est suffisamment forte pour que $p_T$ ne dépende plus de $p_0$.

**Processus reverse (Anderson 1982).** En partant d'un échantillon de $p_T$ et en renversant le processus, on retrouve un échantillon de $p_0$. Le reverse est lui-même un processus de diffusion, défini par la **reverse-time SDE** :

$$\boxed{\;dx = \big[f(x, t) - g^2(t)\, \nabla_x \log p_t(x)\big]\, dt + g(t)\, d\bar{w}\;}$$

où $\bar{w}$ est un mouvement brownien en temps inverse et $dt$ représente un pas de temps infinitésimal négatif. La reverse SDE se calcule dès qu'on connaît le drift et la diffusion du forward, plus le score $\nabla_x \log p_t(x)$ pour chaque $t$.

![[im2-1.jpeg|511]]

### L'unification SMLD ↔ DDPM ↔ SDE

> [!important] Le résultat-clé du papier Song 2021
> Les deux modèles discrets vus dans la section V ne sont pas des paradigmes distincts du framework SDE : ce sont des **discrétisations** de deux SDE continues différentes. En passant à la limite continue, on obtient deux familles canoniques de SDE qui correspondent exactement à SMLD et DDPM.

| Famille     | SDE forward                                                                | Variance de $x_t$               | Modèle discret correspondant |
| ----------- | -------------------------------------------------------------------------- | ------------------------------- | ---------------------------- |
| **VE-SDE**  | $dx = \sqrt{\dfrac{d[\sigma^2(t)]}{dt}}\, dw$                              | $\to \infty$ quand $t \to T$    | SMLD / NCSN                  |
| **VP-SDE**  | $dx = -\tfrac{1}{2}\, \beta(t)\, x\, dt + \sqrt{\beta(t)}\, dw$            | Reste $\approx 1$               | DDPM                         |
| **sub-VP**  | Variante de VP avec drift modifié pour borner plus serré la variance       | $< 1$                           | Variante de Song 2021        |

Les noms s'expliquent par le comportement de la variance de $x_t$ pendant le forward :

- **VE (Variance Exploding).** Pas de drift, juste un bruit qui s'accumule. La variance de $x_t$ croît sans borne — c'est la version continue de l'astuce *multi-noise scale* de NCSN, où $\sigma_T \gg 1$.
- **VP (Variance Preserving).** Le drift $-\tfrac{1}{2}\beta(t)\, x$ contrarie l'accumulation du bruit en ramenant $x$ vers zéro. La variance de $x_t$ reste bornée autour de $1$. C'est la formulation continue du forward DDPM $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ : on perd progressivement le signal $x_0$ mais on ajoute juste assez de bruit pour que la variance totale reste constante.
- **sub-VP.** Modification de VP où la variance reste strictement sous $1$, ce qui améliore la likelihood en pratique.

> [!tip] Pourquoi VP en pratique
> Garder la variance bornée rend l'entraînement et l'échantillonnage **numériquement bien plus stables**. C'est pour ça que DDPM, Stable Diffusion et la plupart des modèles modernes utilisent VP. L'exemple gaussien $dx = \sigma^t\, dw$ qui suit est en revanche une instance de VE.

#### Exemple gaussien (forward)

On spécifie la SDE qui perturbe $p_0$ en $p_T$ :

$$dx = \sigma^t\, dw, \qquad t \in [0, 1].$$

Dans ce cas :

$$\boxed{\;p_{0t}(x(t) \mid x(0)) = \mathcal{N}\Big(x(t);\, x(0),\, \tfrac{1}{2 \log \sigma}(\sigma^{2t} - 1)\, I\Big)\;}$$

et on peut choisir la fonction de pondération $\lambda(t) = \tfrac{1}{2 \log \sigma}(\sigma^{2t} - 1)$.

**Remarque.** Quand $\sigma$ est grand, la distribution prior est :

$$\int p_0(y)\, \mathcal{N}\Big(x;\, y,\, \tfrac{1}{2 \log \sigma}(\sigma^2 - 1)\, I\Big)\, dy \approx \mathcal{N}\Big(x;\, 0,\, \tfrac{1}{2 \log \sigma}(\sigma^2 - 1)\, I\Big)$$

quasiment indépendante des données et facile à échantillonner. Intuitivement, cette SDE est un **continuum** de perturbations gaussiennes de variance $\tfrac{1}{2 \log \sigma}(\sigma^{2t} - 1)$, qui transforme graduellement $p_0$ en une gaussienne simple $p_1$.

#### Algorithme d'entraînement

**(1) Inputs.** Ensemble d'images $x(0) = [X]_{32 \times 1 \times 28 \times 28}$.

**(2) Calcul de l'écart-type.** On assigne à chaque observation un pas de temps aléatoire $t$. On tire $[\text{rand}_t]_{32} \sim \mathcal{N}(0, 1)$ et on transforme :

$$[\text{rand}_t]_{32} = \text{rand}_t (1 - \epsilon) + \epsilon \sim \mathcal{N}(\varepsilon, 1 - \varepsilon)$$

Étant donnés ces 32 pas de temps, on calcule l'écart-type de $p_{0t}(x(t) \mid x(0))$ et on l'étend en tenseur :

$$[\sigma_t]_{32} = \tfrac{1}{2 \log \sigma}(\sigma^{2t} - 1)\, I \;\Rightarrow\; [\sigma]_{32 \times 1 \times 28 \times 28}$$

**(3) Échantillonnage depuis $p_{0t}$.** On tire $[z]_{32 \times 1 \times 28 \times 28} \sim \mathcal{N}(0, 1)$ et on perturbe :

$$\tilde{X} = x(0) + z\, \sigma_t$$

**(4) Loss.** On passe le tout dans un U-net :

$$s_\theta(\tilde{X}, t) = [\text{score}]_{32 \times 1 \times 28 \times 28} = \text{U-net}(\tilde{X}, \text{rand}_t)$$

ce qui donne la loss :

$$\text{loss} = \text{mean}\!\big(\text{sum}\!\big(\text{score} \times [\sigma]_{32 \times 1 \times 1 \times 1} + z\big)^2\big)$$

On rétropropage pour apprendre le score $s_\theta(\tilde{x}, t)$, conformément à l'objectif théorique :

$$\min_\theta \mathbb{E}_{t \sim \mathcal{U}(0, T)}\!\Big[\lambda(t)\, \mathbb{E}_{x(0) \sim p_0}\, \mathbb{E}_{x(t) \sim p_{0t}}\!\big[\|s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t) \mid x(0))\|_2^2\big]\Big]$$

#### Reverse SDE (échantillonnage)

La reverse-time SDE associée est :

$$\boxed{\;dx = -\sigma^{2t}\, \nabla_x \log p_t(x)\, dt + \sigma^t\, d\bar{w}\;}$$

Pour échantillonner depuis le modèle score-based $s_\theta(x, t)$, on tire d'abord depuis $p_1 \approx \mathcal{N}(x;\, 0,\, \tfrac{1}{2}(\sigma^2 - 1)\, I)$, puis on résout numériquement la reverse SDE. En substituant $s_\theta$ au vrai score :

$$dx = -\sigma^{2t}\, s_\theta(x, t)\, dt + \sigma^t\, d\bar{w}$$

**Schéma d'Euler-Maruyama.** Discrétisation simple : on remplace $dt$ par $\Delta t$ et $dw$ par $z \sim \mathcal{N}(0, g^2(t) \Delta t\, I)$. Pour notre reverse SDE :

$$x_{t - \Delta t} = x_t + \sigma^{2t}\, s_\theta(x_t, t)\, \Delta t + \sigma^t \sqrt{\Delta t}\, z_t, \qquad z_t \sim \mathcal{N}(0, I).$$

**Algorithme.**

1. *Initialisation.* On va de $t = 1$ à $t = 0.001$. Distribution de départ :

$$x(t = 1) = p_{t = 1} = \mathcal{N}\Big(x;\, 0,\, \tfrac{1}{2 \log \sigma}(\sigma^2 - 1)\, I\Big)$$

2. *Boucle.*
    - $\text{mean}_x(t) = x(t) + \sigma^{2t}\, s_\theta(x_t, t)\, \Delta t$
    - $x(t) = \text{mean}_x(t) + \sigma^t \sqrt{\Delta t}\, z_t$, avec $z_t \sim \mathcal{N}(0, 1)$

3. *Résultat.* À la dernière étape on renvoie uniquement $\text{mean}_x(0.001)$ — pas de bruit ajouté au dernier pas. Comme $x(t)$ est vectoriel, on a plusieurs trajectoires : on ne garde que le dernier point de chacune.

#### Prédicteur-Correcteur

*Méthode hybride combinant un solveur SDE (prédicteur) avec un pas de Langevin (correcteur) à chaque étape pour améliorer la qualité des samples. Section à compléter.*

#### Sampling par ODE (probability flow ODE)

Pour toute SDE $dx = f(x, t)\, dt + g(t)\, dw$, il existe une **ODE associée** :

$$\boxed{\;dx = \Big[f(x, t) - \tfrac{1}{2}\, g(t)^2\, \nabla_x \log p_t(x)\Big]\, dt\;}$$

dont les trajectoires ont la **même densité marginale** $p_t(x)$ que la SDE. En résolvant cette ODE dans le sens inverse du temps, on échantillonne donc depuis la même distribution. On l'appelle la **probability flow ODE**. La figure ci-dessous montre les différences de trajectoires entre la SDE et son ODE, alors qu'elles produisent la même distribution :

![[im2-2.jpeg]]

On peut donc partir de $p_T$, intégrer cette ODE en temps inverse, et obtenir un échantillon de $p_0$. Pour notre exemple, on intègre de $t = T$ à $0$ :

$$dx = -\tfrac{1}{2}\, \sigma^{2t}\, s_\theta(x, t)\, dt.$$

Faisable avec n'importe quel solveur ODE black-box, par exemple ceux de `scipy`.

**Algorithme.** $\sigma^{2t}$ est le coefficient de diffusion, $s_\theta(x, t)$ est le réseau de neurones (le score). On va à l'envers, $t \in [1, 0.001]$ :

$$dx = -\tfrac{1}{2}\, \sigma^{2t}\, s_\theta(x, t)\, dt \;\iff\; \text{ode\_func}$$

- Il y a autant de tableaux $[y]$ que de conditions initiales.
- On ne garde que la dernière valeur de chaque trajectoire $[y]$ — c'est elle qui est censée reconstruire l'image.

#### Calcul de la vraisemblance

**Formule instantanée de changement de variable.** Un sous-produit du probability flow ODE est le calcul de la log-vraisemblance. Si $h$ est une bijection différentiable transformant $x \sim p_0$ en $h(x) \sim p_T$, la formule classique du changement de variable donne :

$$p_0(x) = p_T(h(x))\, |\det(J_h(x))|.$$

Les trajectoires d'une ODE définissent une bijection $x(0) \mapsto x(T)$. Pour une ODE $dx = f(x, t)\, dt$, la formule **instantanée** correspondante est :

$$p_0(x(0)) = e^{\int_0^1 \operatorname{div} f(x(t), t)\, dt}\, p_1(x(1))$$

où $\operatorname{div}$ désigne la divergence (la trace du Jacobien).

**Estimateur de Skilling.** La divergence est la trace du Jacobien :

$$\operatorname{div} f(x) = \operatorname{tr}(J_f) = \sum_{i=1}^n \frac{\partial f_i}{\partial x_i}$$

Elle peut être intractable. L'**estimateur de Skilling** affirme :

$$\operatorname{tr}(A) = \mathbb{E}[z^\top A z]$$

avec $z$ gaussien ou Rademacher. On obtient un algorithme randomisé en estimant l'espérance par Monte Carlo (évaluer la forme quadratique, moyenner).

*Preuve.* Considérons $z \sim \mathcal{N}(m, \Sigma)$. La propriété des formes quadratiques donne $\mathbb{E}[zz^\top] = \Sigma + mm^\top$. Pour $z \sim \mathcal{N}(0, I)$, on a $\mathbb{E}[zz^\top] = I$. D'où :

$$\operatorname{tr}(A) = \operatorname{tr}(AI) = \operatorname{tr}\!\big(A\, \mathbb{E}[zz^\top]\big) = \mathbb{E}\!\big[\operatorname{tr}(Azz^\top)\big] = \mathbb{E}[z^\top A z].$$

Quand $z$ est tiré d'une Rademacher (entrées $\pm 1$ avec proba $0.5$), c'est l'**estimateur de Hutchinson**.

![[images/3-Apprentissage automatique/Generative Models/score based/im3-1 (2).png]]

**Estimateur de Skilling-Hutchinson.** En pratique, la divergence d'une fonction vectorielle $f$ est dure à évaluer, mais on peut utiliser cet estimateur non biaisé. Soit $\epsilon \sim \mathcal{N}(0, I)$ :

$$\operatorname{div} f(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\!\big[\epsilon^\top\, J_f(x)\, \epsilon\big].$$

On tire $\epsilon$ et on calcule $\epsilon^\top J_f(x)\, \epsilon$ — seul le **produit Jacobien-vecteur** $J_f(x)\, \epsilon$ est nécessaire, ce qui est typiquement efficace.

**Forme finale.** Pour notre probability flow ODE, la log-vraisemblance s'écrit :

$$\log p_0(x(0)) = \log p_1(x(1)) - \tfrac{1}{2} \int_0^1 \frac{d[\sigma^2(t)]}{dt}\, \operatorname{div} s_\theta(x(t), t)\, dt$$

et avec l'estimateur de Skilling-Hutchinson :

$$\operatorname{div} s_\theta(x(t), t) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\!\big[\epsilon^\top\, J_{s_\theta}(x(t), t)\, \epsilon\big].$$

Numériquement, on utilise `torch.autograd.grad` pour calculer le Jacobien du score, puis on multiplie à gauche et à droite par $\epsilon$ aléatoire et on somme. Ensuite on intègre numériquement — un estimateur non biaisé de la vraie vraisemblance, qu'on peut affiner en répétant et moyennant. L'intégrateur a besoin de $x(t)$ comme fonction de $t$, fourni par le probability flow ODE sampler.

### B. SDE plus générale

Dans le papier précédent, le coefficient de diffusion ne dépendait pas de $x(t)$. Le framework s'étend à des coefficients plus généraux.

**Forward SDE générale :**

$$dx = f(x, t)\, dt + G(x, t)\, dw$$

où $G(x, t)$ peut maintenant dépendre de $x$ (on suit l'interprétation d'Itô).

**Reverse-time SDE générale** (Anderson 1982) :

$$dx = \Big\{f(x, t) - \nabla \cdot \big[G(x, t) G(x, t)^\top\big] - G(x, t) G(x, t)^\top\, \nabla_x \log p_t(x)\Big\}\, dt + G(x, t)\, d\bar{w}$$

où pour une fonction matricielle $F(x) = (f^1(x), \ldots, f^d(x))^\top$, on note $\nabla \cdot F(x) := (\nabla \cdot f^1(x), \ldots, \nabla \cdot f^d(x))^\top$.

**Probability flow ODE générale.**

$$dx = \Big\{f(x, t) - \tfrac{1}{2}\, \nabla \cdot \big[G(x, t) G(x, t)^\top\big] - \tfrac{1}{2}\, G(x, t) G(x, t)^\top\, \nabla_x \log p_t(x)\Big\}\, dt$$

**Génération conditionnelle.** Pour la SDE générale, on résout la reverse-time SDE conditionnelle :

$$
\begin{aligned}
dx = \Big\{&f(x, t) - \nabla \cdot \big[G(x, t) G(x, t)^\top\big] - G(x, t) G(x, t)^\top\, \nabla_x \log p_t(x) \\
&- G(x, t) G(x, t)^\top\, \nabla_x \log p_t(y \mid x)\Big\}\, dt + G(x, t)\, d\bar{w}
\end{aligned}
$$

**Problème.** Quand le drift et le coefficient de diffusion ne sont pas affines, le **noyau de transition** $p_{0t}(x(t) \mid x(0))$ peut ne pas avoir de forme fermée. Or la loss d'entraînement le requiert :

$$\theta^* = \arg\min_\theta\, \mathbb{E}_t\!\left\{\lambda(t)\, \mathbb{E}_{x(0)}\, \mathbb{E}_{x(t) \mid x(0)}\!\left[\|s_\theta(x(t), t) - \nabla_{x(t)} \log p_{0t}(x(t) \mid x(0))\|_2^2\right]\right\}$$

Deux solutions :

- **Sliced score matching** : on évite le calcul du noyau de transition en repassant par l'objectif sliced :

$$\theta^* = \arg\min_\theta\, \mathbb{E}_t\!\left\{\lambda(t)\, \mathbb{E}_{x(0)}\, \mathbb{E}_{x(t)}\, \mathbb{E}_{v \sim p_v}\!\left[\tfrac{1}{2}\|s_\theta(x(t), t)\|_2^2 + v^\top s_\theta(x(t), t)\, v\right]\right\}$$

- **Restreindre la SDE** à des formes qui admettent un noyau de transition tractable (typiquement affines en $x$).

### C. Diffusion dans l'espace latent

Plutôt que de diffuser dans l'espace ambiant (typiquement $3 \times 512 \times 512 \approx 8 \cdot 10^5$ dimensions pour une image 512²), on diffuse dans un espace latent **bien plus petit** appris par un autoencodeur. Deux approches dominantes : LSGM entraîne l'autoencodeur **conjointement** avec le SGM, LDM utilise un autoencodeur **fixe pré-entraîné** — c'est la seconde qui a gagné en pratique et fait tourner Stable Diffusion.

#### LSGM (Vahdat et al., 2021)

*[Vahdat, Kreis & Kautz, NeurIPS 2021](https://arxiv.org/abs/2106.05931).*

**Architecture.** Les données sont mappées vers un latent via un encodeur $z_0 \sim q_\phi(z_0 \mid x)$, et le processus de diffusion s'applique dans cet espace latent ($z_0 \to z_1$). La génération part de la base $p(z_1) = \mathcal{N}(z_1;\, 0, I)$ et débruite ($z_0 \leftarrow z_1$) pour échantillonner $p_\theta(z_0)$ via le score conditionné sur le temps $\nabla_{z_t} \log p_\theta(z_t)$. Les samples latents sont enfin mappés vers l'espace data par un décodeur $p_\psi(x \mid z_0)$. Le processus génératif s'écrit :

$$p(z_0, x) = p_\theta(z_0)\, p_\psi(x \mid z_0).$$

L'entraînement apprend $\{\phi, \theta, \psi\}$ : paramètres de l'encodeur, du score, et du décodeur.

![[images/3-Apprentissage automatique/Generative Models/score based/latent space/im1-1 (1).png]]

**Entraînement.** Deux étapes :

1. On entraîne le backbone VAE (NVAE, *"nouveau VAE"*) en supposant un prior gaussien standard.
2. On remplace le prior gaussien par un prior score-based, et on entraîne **conjointement** le backbone VAE et le prior score-based en bout en bout.

**Loss.** Borne variationnelle supérieure sur la log-vraisemblance négative $-\log p(x)$ :

$$
\begin{aligned}
\mathcal{L}(x, \phi, \theta, \psi) &= \mathbb{E}_{q_\phi(z_0 \mid x)}\!\left[-\log p_\psi(x \mid z_0)\right] + \mathrm{KL}\!\left(q_\phi(z_0 \mid x)\, \|\, p_\theta(z_0)\right) \\
&= \underbrace{\mathbb{E}_{q_\phi}\!\left[-\log p_\psi(x \mid z_0)\right]}_{\text{reconstruction}} + \underbrace{\mathbb{E}_{q_\phi}\!\left[\log q_\phi(z_0 \mid x)\right]}_{\text{entropie négative}} + \underbrace{\mathbb{E}_{q_\phi}\!\left[-\log p_\theta(z_0)\right]}_{\text{cross-entropie}}
\end{aligned}
$$

avec la cross-entropie qui s'exprime en termes de score :

$$\mathrm{CE}(q(z_0 \mid x) \,\|\, p(z_0)) = \mathbb{E}_{t \sim \mathcal{U}[0, 1]}\!\left[\tfrac{g(t)^2}{2}\, \mathbb{E}_{q(z_t, z_0 \mid x)}\!\left[\|\nabla_{z_t} \log q(z_t \mid z_0) - \nabla_{z_t} \log p(z_t)\|_2^2\right]\right] + \tfrac{D}{2} \log(2 \pi e \sigma_0^2)$$

**Nouvelle paramétrisation.** On paramétrise le score latent comme un **mélange géométrique** entre une normale et un SGM appris, ce qui permet au SGM de modéliser seulement l'écart entre la distribution latente et le prior gaussien. En 1D :

$$p(z_t) \propto \mathcal{N}(z_t;\, 0, 1)^{1 - \alpha}\, p_\theta'(z_t)^\alpha$$

où $p_\theta'(z_t)$ est un prior SGM entraînable et $\alpha \in [0, 1]$ un coefficient de mélange appris. La cross-entropie devient :

$$\mathrm{CE}(q_\phi(z_0 \mid x) \,\|\, p_\theta(z_0)) = \mathbb{E}_{t \sim \mathcal{U}[0, 1]}\!\left[\tfrac{w(t)}{2}\, \mathbb{E}_{q_\phi(z_t, z_0 \mid x), \epsilon}\!\left[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\right]\right] + \tfrac{D}{2} \log(2 \pi e \sigma_0^2)$$

**Réduction de variance.** Techniques de réduction de variance par une nouvelle SDE et par des schémas d'importance sampling. On se concentre sur les **variance preserving SDEs (VP-SDE)** :

$$\boxed{\;dz = -\tfrac{1}{2}\, \beta(t)\, z\, dt + \sqrt{\beta(t)}\, dw\;}$$

avec $\beta(t) = \beta_0 + (\beta_1 - \beta_0)\, t$ qui interpole linéairement dans $[\beta_0, \beta_1]$.

#### LDM / Stable Diffusion (Rombach et al., 2022)

*[Rombach, Blattmann, Lorenz, Esser & Ommer, CVPR 2022](https://arxiv.org/abs/2112.10752).* Les **Latent Diffusion Models** sont la simplification radicale de LSGM qui a permis Stable Diffusion. L'idée tient en une ligne : *fige l'autoencodeur, ne fais de la diffusion que sur le latent*.

**Architecture.** Trois composants entraînés **séquentiellement** (pas conjointement) :

- $\mathcal{E}$ : encodeur image → latent ($x \in \mathbb{R}^{3 \times H \times W} \;\mapsto\; z \in \mathbb{R}^{c \times h \times w}$, avec typiquement $h = H/8$, $w = W/8$).
- $\mathcal{D}$ : décodeur latent → image ($z \mapsto \hat{x}$).
- $\epsilon_\theta$ : réseau de diffusion (UNet) qui opère **dans l'espace latent**.

Pipeline :

1. **Étape 1 — Entraîner l'autoencodeur** $\mathcal{E}, \mathcal{D}$ comme un VQ-VAE ou un VAE classique avec perte de reconstruction, perte perceptuelle (LPIPS) et discriminateur GAN. Régularisation KL **très légère** (objectif : reconstruction fidèle, pas une bonne distribution latente).
2. **Étape 2 — Geler $\mathcal{E}, \mathcal{D}$**.
3. **Étape 3 — Entraîner $\epsilon_\theta$** par la loss DDPM standard sur les latents $z = \mathcal{E}(x)$ :

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), \epsilon, t}\!\left[\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}\, \mathcal{E}(x) + \sqrt{1 - \bar{\alpha}_t}\, \epsilon,\, t,\, c\big)\big\|^2\right]$$

où $c$ est un conditionnement optionnel (texte, classe, image…).

**Inférence** :
1. Tirer $z_T \sim \mathcal{N}(0, I)$.
2. Sampler $z_0$ par DDIM (ou autre solveur).
3. Décoder : $\hat{x} = \mathcal{D}(z_0)$.

**Pourquoi c'est puissant.**

- **Compression spatiale 8×.** Une image $512 \times 512$ devient un latent $64 \times 64$ ⇒ **64× moins de pixels** à traiter par étape de diffusion. C'est ce qui rend la génération 512² faisable sur une GPU consumer (8–12 GB de VRAM).
- **Sémantique vs détails.** La diffusion se concentre sur la sémantique (composition, structure, objets), pendant que le décodeur gère les détails haute fréquence (textures, peau, cheveux). Bien plus efficace que de tout faire en pixel-space.
- **Modularité.** Un autoencodeur pré-entraîné se réutilise pour entraîner plusieurs diffusion models (SD 1.4, SD 1.5, SD 2.0 partagent un autoencodeur compatible).

**Cross-attention pour le conditionnement texte.** LDM introduit du **cross-attention** dans le UNet entre les features image et les embeddings du prompt (typiquement issus du text encoder de CLIP). C'est le mécanisme qui permet à un prompt comme *"a cat wearing a hat in the style of Van Gogh"* d'influencer la génération à toutes les résolutions du UNet. Le réseau devient :

$$\epsilon_\theta(z_t,\, t,\, \tau_\xi(c))$$

où $\tau_\xi$ est un encodeur texte (CLIP, T5) qui transforme le prompt $c$ en une séquence de tokens, injectés au UNet via cross-attention.

> [!important] Pourquoi LDM a gagné face à LSGM
> La séparation stricte autoencodeur / diffusion (au lieu d'un entraînement conjoint) permet de **scaler à des résolutions impossibles en pixel-space**, de **réutiliser** un autoencodeur pré-entraîné pour plusieurs modèles, et de **modulariser** complètement l'entraînement. C'est cette simplicité d'architecture qui a permis Stable Diffusion comme modèle open-source grand public. LSGM, plus élégant théoriquement, a été éclipsé.

**Comparatif rapide.**

| Aspect                       | LSGM (Vahdat 2021)         | LDM (Rombach 2022)            |
| ---------------------------- | -------------------------- | ----------------------------- |
| Autoencodeur                 | Entraîné **conjointement** | **Pré-entraîné fixe**         |
| Loss globale                 | ELBO conjoint              | DDPM standard sur le latent   |
| Conditionnement texte        | Non développé              | **Cross-attention CLIP/T5**   |
| Scalabilité                  | Limitée                    | **Excellente** (1024² faisable) |
| Modèles industriels          | Aucun                      | Stable Diffusion, SDXL, SD3   |

## VII. Génération conditionnelle : classifier-free guidance

Jusqu'ici on a généré des samples non conditionnels — un tirage de $p_{\text{data}}$ sans contrainte. Pour des applications comme **Stable Diffusion** (texte → image), **DALL-E 2** ou la génération class-conditional, on veut conditionner sur un label $y$ : générer $x \sim p(x \mid y)$ plutôt que $x \sim p(x)$. Deux techniques se sont imposées, dont la seconde est devenue la norme.

### A. Classifier guidance

*[Dhariwal & Nichol, NeurIPS 2021](https://arxiv.org/abs/2105.05233).* L'idée part de Bayes :

$$\log p(x \mid y) = \log p(x) + \log p(y \mid x) - \log p(y).$$

En prenant le gradient en $x$ (le $\log p(y)$ ne dépend pas de $x$, donc disparaît) :

$$\nabla_x \log p(x \mid y) = \underbrace{\nabla_x \log p(x)}_{\text{score inconditionnel}} + \underbrace{\nabla_x \log p(y \mid x)}_{\text{gradient d'un classifieur}}.$$

Le premier terme est le score qu'on apprend déjà (ou $\epsilon_\theta$ via le pont). Le second est le gradient d'un **classifieur** $p_\phi(y \mid x_t)$, qu'on entraîne séparément sur des images **bruitées** à tous les niveaux $t$. À l'inférence, on échantillonne avec le score augmenté :

$$\tilde{s}_\theta(x_t, t, y) = s_\theta(x_t, t) + w\, \nabla_{x_t} \log p_\phi(y \mid x_t)$$

où $w \ge 1$ est le **guidance scale** : plus $w$ est grand, plus la génération colle au label.

**Limite.** Il faut entraîner un classifieur **séparé** capable de classer des images bruitées à tous les niveaux de bruit — ce qui est laborieux et limite la qualité.

### B. Classifier-free guidance (CFG)

*[Ho & Salimans, NeurIPS Workshop 2021](https://arxiv.org/abs/2207.12598).* **C'est la technique qui fait marcher Stable Diffusion, DALL-E 2, Imagen, Midjourney et tous les diffusion modernes.** L'idée : se passer du classifieur en utilisant le modèle de diffusion lui-même.

**Entraînement.** On entraîne un **seul** réseau $\epsilon_\theta(x_t, t, y)$ qui prend en argument un label $y$ (texte, classe…), avec une astuce : pendant l'entraînement, on remplace $y$ par un token spécial $\emptyset$ (« null », équivalent à « pas de condition ») avec une probabilité $p \approx 10$–$20\%$. Le même réseau apprend donc à faire les **deux tâches** :

- $\epsilon_\theta(x_t, t, y)$ : prédire le bruit *conditionnellement* à $y$
- $\epsilon_\theta(x_t, t, \emptyset)$ : prédire le bruit *inconditionnellement*

**Inférence.** À la génération, on combine les deux prédictions par **extrapolation** :

$$\boxed{\;\tilde{\epsilon}_\theta(x_t, t, y) = (1 + w)\, \epsilon_\theta(x_t, t, y) - w\, \epsilon_\theta(x_t, t, \emptyset)\;}$$

où $w \ge 0$ est le **guidance scale**. Interprétation :

- $w = 0$ → génération conditionnelle pure ($\tilde{\epsilon} = \epsilon_\theta(x_t, t, y)$).
- $w > 0$ → on extrapole *au-delà* du conditionnel, dans la direction qui éloigne de l'inconditionnel. Cela **amplifie** l'influence de $y$ sur la génération.
- $w$ trop grand → l'image colle fortement au prompt mais perd en diversité et fait apparaître des artefacts (sur-saturation, déformations).

En pratique $w \in [3, 15]$ selon les modèles (Stable Diffusion utilise typiquement $w = 7.5$).

> [!tip] À retenir pour les interviews
> Quand quelqu'un te demande « qu'est-ce qui rend Stable Diffusion utilisable ? », la réponse en une phrase est : **classifier-free guidance**. Sans CFG, les diffusion models conditionnels génèrent des images « raisonnables » mais qui ne suivent pas vraiment le prompt. Avec CFG ($w \approx 7$), elles le suivent réellement — c'est ce qui rend l'expérience utilisateur de Midjourney/Stable Diffusion possible.

**Pourquoi c'est l'état de l'art.** Trois raisons :
1. Pas de classifieur séparé à entraîner — un seul réseau gère tout.
2. Le même modèle gère conditionnel et inconditionnel, donc on peut faire varier $y$ ou utiliser $\emptyset$ sans changer de réseau.
3. La qualité visuelle obtenue avec un grand $w$ surpasse largement classifier guidance, et l'effet du $w$ se règle au moment de l'inférence (pas besoin de réentraîner pour ajuster la « force » de la guidance).
