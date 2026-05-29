---
title: Flow Matching
---
# Flow Matching

> Septième famille — mathématiquement **héritière des Continuous Normalizing Flows** (CNF), pédagogiquement liée aux diffusion models (`[[06_Score-Based & Diffusion Models|06]]`), et **devenue le standard de fait depuis 2024** pour la génération haute-résolution (Stable Diffusion 3, FLUX, Movie Gen…). L'idée tient en une phrase : apprendre un **champ de vitesse** $v_\theta(x, t)$ qui transporte une distribution simple (gaussienne) vers la distribution des données via une **ODE déterministe**. Le tour de force de [Lipman et al. (2023)](https://arxiv.org/abs/2210.02747) est d'avoir trouvé une **loss simulation-free** — pas besoin d'intégrer l'ODE pendant l'entraînement, contrairement à CNF — ce qui rend la méthode scalable. Plus fort encore : ils montrent que **la diffusion est un cas particulier** de Flow Matching avec un chemin gaussien. Flow Matching est donc une généralisation strictement plus large qui permet de choisir des chemins **plus simples** (notamment les chemins **rectifiés**, linéaires), qui se génèrent en 5–10 NFE au lieu de 50+ pour diffusion.

## I. Généalogie et positionnement

### A. Le paysage des générateurs continus

| Famille                                          | Trajectoire          | Apprentissage            | NFE génération |
| ------------------------------------------------ | -------------------- | ------------------------ | -------------- |
| `[[04_Normalizing Flows\|Normalizing Flows]]` (discrets) | Bijection discrète   | MLE direct               | 1              |
| **Continuous NF** (FFJORD, 2018)                 | ODE déterministe     | MLE par intégration (lent) | 50–100         |
| `[[06_Score-Based & Diffusion Models\|Diffusion (SDE)]]` | SDE stochastique     | Score matching           | 25–1000        |
| **Probability flow ODE** (Song 2021)             | ODE déterministe     | Score matching           | 25–50          |
| **Flow Matching** (Lipman 2023)                  | ODE déterministe     | **Simulation-free**      | **5–10** (rectifié) |

Flow Matching est l'héritier direct des **Continuous Normalizing Flows**, pas des diffusion models — mais il leur emprunte leur astuce d'entraînement (apprendre sur des trajectoires courtes, pas via intégration complète de l'ODE).

### B. Le problème laissé ouvert par les CNF

Les CNF apprennent une ODE déterministe

$$\frac{dx}{dt} = v_\theta(x, t)$$

et calculent la log-vraisemblance via la **formule instantanée de changement de variable** (cf. la section *Calcul de la vraisemblance* de `[[06_Score-Based & Diffusion Models|06]]`). C'est élégant, mais **pour entraîner**, il faut intégrer l'ODE forward et backward sur chaque mini-batch — prohibitif. C'est ce qui a fait que les CNF sont restés théoriquement intéressants mais peu utilisés en pratique entre 2018 et 2022.

### C. L'idée de Flow Matching

> [!important] L'idée-clé
> Au lieu d'entraîner sur la log-vraisemblance (qui nécessite l'intégration de l'ODE), on **fabrique** un champ de vitesse cible $u_t(x)$ qu'on **sait calculer en forme fermée**, et on entraîne $v_\theta$ par **régression MSE** sur ce champ-cible. L'entraînement devient une MSE simple — exactement comme DDPM qui apprend à prédire $\epsilon$ par MSE.

## II. Le cadre Flow Matching

### A. Setup

- **Distribution de base** : $p_0 = \mathcal{N}(0, I)$ (bruit gaussien).
- **Distribution cible** : $p_1 = p_{\text{data}}$.
- **But** : un champ $v_\theta(x, t)$ tel que résoudre $\frac{dx}{dt} = v_\theta(x, t)$ de $t=0$ à $t=1$ avec $x_0 \sim p_0$ donne $x_1 \sim p_1$.

Convention temporelle de Lipman : $t=0$ = bruit, $t=1$ = données. (Attention, certains papiers utilisent l'inverse.)

### B. L'astuce : conditionner sur les paires (x_0, x_1)

On choisit explicitement un **chemin** $\psi_t(x_0, x_1)$ qui interpole entre $x_0$ et $x_1$, avec $\psi_0 = x_0$ et $\psi_1 = x_1$. Le choix le plus simple est le **chemin linéaire** :

$$\psi_t(x_0, x_1) = (1 - t)\, x_0 + t\, x_1$$

Le champ de vitesse **conditionnel** correspondant (la vélocité instantanée le long du chemin) est alors la **constante** :

$$u_t(x \mid x_0, x_1) = \frac{d\psi_t}{dt} = x_1 - x_0$$

> [!tip] Pourquoi c'est génial
> Le champ conditionnel $u_t(x \mid x_0, x_1) = x_1 - x_0$ est **trivial à évaluer** : c'est juste la différence entre l'image cible et le bruit de départ. Pas d'intégrale, pas de score à apprendre, juste une soustraction.

### C. Loss Flow Matching

$$\boxed{\;\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x_0 \sim p_0,\; x_1 \sim p_1}\!\left[\big\|v_\theta(\psi_t(x_0, x_1),\, t)\, -\, (x_1 - x_0)\big\|^2\right]\;}$$

**Procédure d'entraînement** :

1. Tirer $t \sim \mathcal{U}[0, 1]$, $x_0 \sim \mathcal{N}(0, I)$, $x_1 \sim p_{\text{data}}$.
2. Construire $x_t = (1-t)\, x_0 + t\, x_1$.
3. Forward le réseau : $v_\theta(x_t, t)$.
4. MSE contre $x_1 - x_0$.
5. Backprop standard.

**Pas d'intégration d'ODE pendant l'entraînement** — d'où le terme *simulation-free*.

### D. Pourquoi ça marche : théorème central

Le résultat-clé de Lipman 2023 :

> [!note] Marginalisation du champ conditionnel
> Si chaque champ conditionnel $u_t(x \mid x_0, x_1)$ génère le chemin individuel $(x_0 \to x_1)$, alors le champ **marginal** défini par
> $$v^*(x, t) = \mathbb{E}\!\big[u_t(x \mid x_0, x_1) \,\big|\, \psi_t(x_0, x_1) = x\big]$$
> génère la distribution marginale $p_t$. Et c'est précisément ce que $v_\theta$ apprend en minimisant $\mathcal{L}_{\text{FM}}$ — l'argmin de la MSE conditionnelle est l'espérance conditionnelle.

Donc en entraînant sur la MSE conditionnelle (facile), on apprend bien le champ marginal (la vraie quantité d'intérêt).

### E. Génération

Une fois $v_\theta$ appris, on génère par **résolution d'ODE** standard :

1. Tirer $x_0 \sim \mathcal{N}(0, I)$.
2. Intégrer $\frac{dx}{dt} = v_\theta(x, t)$ de $t=0$ à $t=1$ avec un solveur (Euler, RK4…).
3. $x_1$ est l'échantillon généré.

Avec un chemin **linéaire**, les trajectoires sont quasi-droites ⇒ **5–10 pas d'Euler suffisent** (vs 25–50 pour diffusion).

## III. Rectified Flow

Présenté par [Liu, Gong & Liu (2022)](https://arxiv.org/abs/2209.03003) **en parallèle** de Lipman, avec une perspective complémentaire centrée sur le caractère **rectiligne** des trajectoires.

**Idée géométrique.** Entre deux points $x_0$ et $x_1$, le chemin **optimal** est une ligne droite. Diffusion impose un chemin **courbé** (passage par une gaussienne intermédiaire). Rectified Flow le **redresse** :

$$x_t = (1 - t)\, x_0 + t\, x_1 \qquad\Rightarrow\qquad u_t = x_1 - x_0.$$

C'est exactement le Flow Matching avec chemin linéaire — les deux papiers sont essentiellement équivalents pour ce cas.

### Reflow : redresser encore plus

Astuce de Liu et al. pour pousser plus loin : après avoir entraîné un premier modèle, **résoudre l'ODE** pour générer des paires $(x_0, x_1)$ couplées **par le modèle lui-même**, puis **réentraîner** sur ces paires.

Les trajectoires apprises lors du second tour sont **encore plus droites** (le modèle a "déjà résolu" le couplage optimal). Après quelques itérations de reflow, on peut générer en **1 seul pas**.

> [!important] InstaFlow et la 1-step generation
> [InstaFlow (Liu et al. 2023)](https://arxiv.org/abs/2309.06380) applique reflow + distillation à un modèle Stable Diffusion, et obtient une **génération en 1 NFE** avec une qualité comparable à 50 NFE du modèle original. C'est l'analogue rapide de Stable Diffusion.

## IV. Le lien avec diffusion

**Théorème de Lipman.** Le probability flow ODE de la diffusion (cf. `[[06_Score-Based & Diffusion Models|06]]`, section VI.A) est un cas particulier de Flow Matching avec un **chemin gaussien** :

$$\psi_t(x_0, x_1) = \mu_t(x_1) + \sigma_t\, x_0$$

où $\mu_t = \sqrt{\bar\alpha_t}$ et $\sigma_t = \sqrt{1 - \bar\alpha_t}$ (dans la convention t=1 = data). Le champ conditionnel correspondant fait apparaître un **terme score-like** — d'où la trajectoire courbée.

Comparatif des chemins :

| Modèle               | Chemin $\psi_t(x_0, x_1)$                                  | Champ $u_t$                  | Trajectoire | NFE  |
| -------------------- | ---------------------------------------------------------- | ---------------------------- | ----------- | ---- |
| **Diffusion (VP)**   | $\sqrt{\bar\alpha_t}\, x_1 + \sqrt{1 - \bar\alpha_t}\, x_0$ | Score-like (complexe)        | Courbée     | 25–50 |
| **Rectified Flow**   | $(1-t)\, x_0 + t\, x_1$                                    | $x_1 - x_0$ (constante)      | **Linéaire** | 5–10 |
| **InstaFlow**        | Ultra-droite (après reflow)                                | Quasi-constante              | **Très linéaire** | **1** |

> [!tip] L'intuition géométrique
> La diffusion oblige les trajectoires à **traverser une variété gaussienne** courbée, ce qui nécessite beaucoup de petits pas pour bien suivre la courbure. Rectified Flow va **directement en ligne droite**, ce qui demande très peu de pas d'intégration. C'est pour ça que la génération est si rapide.

## V. Applications en production (2024–2025)

Flow Matching / Rectified Flow est devenu **le standard** pour les modèles génératifs de pointe :

- **Stable Diffusion 3** (Stability AI, 2024) — architecture MMDiT (Multimodal Diffusion Transformer) entraînée par Rectified Flow.
- **FLUX.1** (Black Forest Labs, 2024) — rectified flow + transformer, état de l'art pour texte → image open-source.
- **Movie Gen** (Meta, 2024) — Flow Matching pour la génération vidéo.
- **ESM3** (Evolutionary Scale, 2024) — Flow Matching pour les protéines.

Le passage de diffusion à Flow Matching dans Stable Diffusion 3 a été un des facteurs clés du gain de qualité — entraînement plus stable, génération plus rapide.

## VI. Pour aller plus loin

Quelques directions de recherche actives, à ajouter à terme :

- **Conditional Flow Matching** (Tong et al. 2023) — formulation plus générale qui inclut FM et Rectified Flow comme cas particuliers.
- **Stochastic Interpolants** (Albergo & Vanden-Eijnden 2023) — cadre unifié qui englobe FM, diffusion et bien plus, avec des couplages stochastiques entre $x_0$ et $x_1$.
- **Optimal Transport Flow Matching** — utiliser un couplage de transport optimal $(x_0, x_1)$ pour des trajectoires encore plus droites dès la première itération (pas besoin de reflow).
- **Flow Matching sur variétés** (Chen & Lipman 2023) — extension aux données vivant sur des sphères, des groupes de Lie, des graphes, etc. Utile en biologie computationnelle.
