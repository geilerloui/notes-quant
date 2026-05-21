---
title: Variational Autoencoders (VAE)
---
# Variational Autoencoders (VAE)

> Deuxième famille concrète après les autorégressifs. Les VAE introduisent **un espace latent explicite** : on génère $x$ via une variable cachée $z$ qu'on tire d'un prior simple ($\mathcal{N}(0, I)$). Cette structure change tout — on gagne l'interpolation, le débridage du parallélisme à la génération, et une représentation utilisable ailleurs ; mais on perd la densité exacte (remplacée par une borne, l'**ELBO**). C'est aussi la note où s'introduisent l'**inférence variationnelle**, l'**amortization**, et le **reparametrization trick** — trois outils qu'on retrouvera dans les normalizing flows et les diffusion models.

## I. Pourquoi un modèle à variable latente ?

### A. Le projet : générer via un latent

Reprenons la grille de `[[00_Fondations#III. Les trois questions fondamentales]]`. Un autorégressif fait le choix structurel d'une **chain rule sur les composantes de $x$** ; un VAE fait un choix radicalement différent : il introduit une **variable cachée $z$** entre une distribution simple (le prior) et l'observation $x$.

$$z \sim p(z), \qquad x \sim p_\theta(x \mid z).$$

![[im9 (1).png|97]]
Figure Graphical model perspective of the VAE


Concrètement, pour générer une image MNIST :

1. on tire $z \in \mathbb{R}^d$ depuis $\mathcal{N}(0, I)$ (distribution simple, $d$ petit, typiquement 16-128),
2. on passe $z$ dans un réseau de neurones (le **décodeur**) qui produit les paramètres d'une distribution sur $x$,
3. on échantillonne $x$ depuis cette distribution.

L'image générée est alors entièrement spécifiée par $z$. C'est ce $z$ qu'on appelle la **représentation latente** : un vecteur de petite dimension qui code les facteurs de variation des données (style d'écriture, angle, épaisseur, etc.).

### B. Pourquoi pas un autoencodeur classique ?

Un **autoencodeur** (AE) est une architecture neuronale qui compresse $x$ en un vecteur caché $h$ via un encodeur, puis reconstruit $\hat x$ depuis $h$ via un décodeur. L'objectif d'entraînement est purement reconstructif :

$$\min_{\theta, \phi} \; \frac{1}{m} \sum_{i=1}^{m} \| x^{(i)} - \hat x^{(i)} \|^2.$$

Un AE entraîné apprend bien à représenter $x$ par un $h$ compact, et le décodeur sait reconstruire depuis $h$. **Tentation naturelle :** retirer l'encodeur, prendre un $h$ au hasard, le passer au décodeur, et obtenir une nouvelle image.

![[im1 (2).png|174]]
Figure l'autoencodeur


> [!warning] Pourquoi ça ne marche pas
> Un AE classique n'est **pas génératif**. Deux raisons :
> 
> - **L'espace caché est sparse.** Le décodeur a été entraîné à reconstruire depuis les $h$ que l'encodeur produit pour les images du dataset — un sous-ensemble *minuscule* de $\mathbb{R}^d$. Si on tire un $h$ ailleurs (au hasard, ou par interpolation entre deux points connus), on tombe dans une zone que le décodeur n'a jamais vue. Résultat : du bruit.
> - **Le décodeur est déterministe.** Pour un $h$ donné, il sort *une seule* image. Or modéliser des données réelles demande de la stochasticité — beaucoup d'images peuvent être plausibles pour un même concept (différentes écritures du chiffre 3).
> 
> L'AE compresse efficacement, mais n'apprend pas une *distribution* sur $x$. Pour générer, il manque les deux ingrédients.



![[im2 (2).png|175]]
Figure On a appris les poides de l'autoencodeur et on vire "x" pour prédire x hat

### C. Le saut conceptuel du VAE

Le VAE résout les deux problèmes simultanément par un changement de paradigme : **au lieu de mapper $x$ à un point $h$, on mappe $x$ à une distribution $q_\phi(z \mid x)$**. L'encodeur ne produit plus un vecteur mais les paramètres d'une distribution — typiquement une gaussienne $\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$.

Et on impose à cette distribution un **comportement régulier** : elle doit ressembler à un prior simple $p(z) = \mathcal{N}(0, I)$. Cette contrainte (qu'on verra apparaître comme un terme KL dans la loss) garantit deux choses :

- les distributions $q_\phi(z \mid x)$ pour différents $x$ se **chevauchent** dans $\mathbb{R}^d$ — pas d'îlots isolés ;
- on peut **échantillonner** un $z$ depuis le prior $\mathcal{N}(0, I)$ et avoir une bonne chance de tomber dans une zone où le décodeur sait quoi faire.

C'est ce qui transforme l'AE en générateur. Le reste de la note est la formalisation mathématique de cette idée, et c'est là qu'on rencontre l'**ELBO**.

## II. Le modèle génératif latent

### A. Le DAG

Le VAE est un modèle graphique dirigé minimal — deux nœuds :

![[im9 (1).png|97]]

- **$z \in \mathbb{R}^d$** est la variable latente, jamais observée.
- **$x \in \mathbb{R}^n$** est la variable observée (image, etc.), $d \ll n$.

La jointe se factorise naturellement :

$$p_\theta(x, z) = p(z) \, p_\theta(x \mid z),$$

avec :

- **$p(z) = \mathcal{N}(0, I)$** : le prior, fixé (pas appris).
- **$p_\theta(x \mid z)$** : la *likelihood* (vraisemblance), paramétrée par un réseau de neurones — le **décodeur**. Sa forme dépend du type de données :
  - $x$ continu (images en réel) : $p_\theta(x \mid z) = \mathcal{N}(f_\theta(z), I)$ — gaussienne centrée sur la sortie du décodeur.
  - $x$ binaire (MNIST) : $p_\theta(x \mid z) = \prod_j \text{Bern}\big(f_\theta(z)_j\big)$ — Bernoulli par pixel.

### B. Ce qu'on veut maximiser

L'objectif est classique (cf. `[[00_Fondations#V bis. Apprentissage par maximum de vraisemblance]]`) : maximiser la log-vraisemblance des données :

$$\log p_\theta(x) = \log \int p_\theta(x \mid z) \, p(z) \, dz.$$

L'intégrale marginalise sur toutes les valeurs possibles de $z$. Et là, problème majeur.

$$
P(X)=\int P(X \mid z) P(z) d z 
=\iint \cdots \int P\left(X \mid z_{1}, z_{2}, \ldots, z_{n}\right) P\left(z_{1}, z_{2}, \ldots, z_{n}\right) d z_{1}, \ldots d z_{n}
$$

> [!warning] L'intégrale est intractable
> Pour calculer $p_\theta(x)$, il faudrait intégrer $p_\theta(x \mid z) p(z)$ sur tout $\mathbb{R}^d$. Avec $d = 64$ et $z$ continu, c'est une intégrale en dimension 64 — incalculable analytiquement, et l'approximation numérique naïve (Monte Carlo uniforme) est désastreuse (cf. §III).
> 
> Conséquence : on ne peut **pas** maximiser directement $\log p_\theta(x)$. Il faut trouver un substitut. Ce substitut sera l'**ELBO**.

## III. Approximer l'intégrale : trois tentatives

On déroule trois approches pour estimer $\log p_\theta(x)$. Seule la troisième mène quelque part — c'est l'ELBO.

### A. Monte Carlo naïf

Idée : approximer l'intégrale par une moyenne empirique sur des $z$ tirés uniformément.

$$p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz = \mathbb{E}_{z \sim p(z)}[p_\theta(x \mid z)] \approx \frac{1}{K} \sum_{k=1}^{K} p_\theta(x \mid z^{(k)}), \quad z^{(k)} \sim p(z).$$

**Problème de variance.** Pour la plupart des $z$ tirés du prior, $p_\theta(x \mid z)$ est microscopique : $z$ correspond à une "image générique" qui n'a rien à voir avec le $x$ qu'on essaie d'expliquer. Seuls quelques $z$ rares (ceux qui produisent une image proche de $x$) ont une grande contribution. La moyenne est donc dominée par des événements très rares et a une variance énorme. Inutilisable en pratique.

### B. Importance sampling

Idée : ne pas tirer les $z$ uniformément depuis le prior, mais depuis une distribution $q(z)$ qui privilégie les $z$ *pertinents* pour le $x$ donné. On corrige par le ratio :

$$p_\theta(x) = \int q(z) \cdot \frac{p_\theta(x \mid z) p(z)}{q(z)} \, dz = \mathbb{E}_{z \sim q(z)}\!\left[ \frac{p_\theta(x \mid z) p(z)}{q(z)} \right].$$

Beaucoup mieux : si $q(z)$ est proche de la *posterior* $p_\theta(z \mid x)$ (la distribution des $z$ ayant pu produire $x$), la variance s'effondre.

**Mais on veut $\log p_\theta(x)$, pas $p_\theta(x)$.** Et c'est là que ça coince :

$$\log \mathbb{E}_{z \sim q(z)}\!\left[ \frac{p_\theta(x \mid z) p(z)}{q(z)} \right] \neq \mathbb{E}_{z \sim q(z)}\!\left[ \log \frac{p_\theta(x \mid z) p(z)}{q(z)} \right].$$

Le log d'une espérance n'est pas l'espérance d'un log. On ne peut pas estimer $\log p_\theta(x)$ par Monte Carlo en passant juste le log à l'intérieur.

### C. Jensen → ELBO

Mais l'inégalité de Jensen nous dit que, pour une fonction concave comme le log :

$$\log \mathbb{E}_{z \sim q(z)}[f(z)] \;\geq\; \mathbb{E}_{z \sim q(z)}[\log f(z)].$$

En appliquant à $f(z) = p_\theta(x, z) / q(z)$ qui est l'estimateur d'importance sampling de l'intégrale $\int p_{\theta}(x,z)dz$ avec $x$ qui est fixe c'est une image (eg chat) qu'on a choisi sinon on ne pourrait pas décrire $f(z)$ comme ça : 

$$\log p_\theta(x) \;=\; \log \mathbb{E}_{z \sim q(z)}\!\left[\frac{p_\theta(x, z)}{q(z)}\right] \;\geq\; \mathbb{E}_{z \sim q(z)}\!\left[\log \frac{p_\theta(x, z)}{q(z)}\right].$$

Cette **borne inférieure** sur la log-vraisemblance est l'**Evidence Lower BOund**, notée $\mathcal{L}(x; \theta, q)$ ou simplement ELBO :

$$\boxed{\;\mathcal{L}(x; \theta, q) \;=\; \mathbb{E}_{z \sim q(z)}\!\left[\log \frac{p_\theta(x, z)}{q(z)}\right] \;\leq\; \log p_\theta(x)\;}$$

**Le coup de génie :** au lieu de maximiser $\log p_\theta(x)$ (intractable), on va maximiser sa borne inférieure $\mathcal{L}(x; \theta, q)$ — qui est, elle, calculable par Monte Carlo. Si on pousse la borne vers le haut, on pousse aussi $\log p_\theta(x)$.

## IV. L'ELBO : décomposition et interprétation

### A. Décomposition reconstruction + régularisation

L'ELBO se réécrit de façon plus interprétable :

$$\begin{aligned}
\mathcal{L}(x; \theta, q) &= \mathbb{E}_{z \sim q(z)}\!\left[\log \frac{p_\theta(x \mid z) p(z)}{q(z)}\right] \\
&= \mathbb{E}_{z \sim q(z)}[\log p_\theta(x \mid z)] + \mathbb{E}_{z \sim q(z)}\!\left[\log \frac{p(z)}{q(z)}\right] \\
&= \underbrace{\mathbb{E}_{z \sim q(z)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q(z) \,\|\, p(z))}_{\text{régularisation}}.
\end{aligned}$$

> [!warning] L'ELBO = reconstruction − régularisation
> Deux termes avec des interprétations très claires :
> 
> - **Reconstruction** : $\mathbb{E}_{z \sim q(z)}[\log p_\theta(x \mid z)]$. Pour les $z$ tirés selon $q$, le décodeur doit attribuer une forte probabilité à $x$. Concrètement : *le $z$ que tu m'as proposé doit me permettre de reconstruire $x$*.
> 
> - **Régularisation** : $D_{\text{KL}}(q(z) \,\|\, p(z))$. Le $q$ qu'on utilise doit rester proche du prior $p(z) = \mathcal{N}(0, I)$. Sans ce terme, $q$ pourrait dégénérer (concentration sur un point unique → meilleur reconstruction term, mais latent inutilisable).
> 
> **Maximiser l'ELBO = minimiser l'erreur de reconstruction tout en gardant $q$ proche du prior.** C'est l'équilibre entre fidélité aux données et régularité de l'espace latent.

### B. Le gap : à quel point la borne est-elle serrée ?

Une borne est utile si elle est *serrée*. À quel point l'ELBO sous-estime-t-elle $\log p_\theta(x)$ ? On peut le calculer exactement :

$$\log p_\theta(x) - \mathcal{L}(x; \theta, q) = D_{\text{KL}}(q(z) \,\|\, p_\theta(z \mid x)).$$

> [!warning] La borne est serrée quand $q$ approxime le vrai postérieur
> Le gap entre la log-vraisemblance et son ELBO est *exactement* la KL entre la distribution d'inférence $q(z)$ et la vraie posterior $p_\theta(z \mid x)$. Conséquences :
> 
> - Si on choisit un $q$ très proche du vrai postérieur → l'ELBO est presque égale à $\log p_\theta(x)$.
> - Si $q$ est mauvais → la borne devient lâche, on optimise quelque chose de très en dessous de la vraisemblance réelle.
> 
> **Maximiser l'ELBO sur $q$ revient donc à faire de l'inférence variationnelle** : on cherche le meilleur $q$ dans une famille paramétrée pour approcher le postérieur intractable $p_\theta(z \mid x)$.

> [!note]- Preuve du gap (calcul direct)
> $$\begin{aligned}
> \log p_\theta(x) - \mathcal{L}(x; \theta, q) &= \log p_\theta(x) - \mathbb{E}_{z \sim q}\!\left[\log \frac{p_\theta(x, z)}{q(z)}\right] \\
> &= \mathbb{E}_{z \sim q}[\log p_\theta(x)] - \mathbb{E}_{z \sim q}[\log p_\theta(x, z)] + \mathbb{E}_{z \sim q}[\log q(z)] \\
> &= \mathbb{E}_{z \sim q}\!\left[\log \frac{p_\theta(x) \, q(z)}{p_\theta(x, z)}\right] \\
> &= \mathbb{E}_{z \sim q}\!\left[\log \frac{q(z)}{p_\theta(z \mid x)}\right] \\
> &= D_{\text{KL}}(q(z) \,\|\, p_\theta(z \mid x)).
> \end{aligned}$$
> 
> Le passage clé est $p_\theta(x, z) = p_\theta(z \mid x) p_\theta(x)$, c'est-à-dire la définition du postérieur.

## V. Amortization : de $q(z)$ à $q_\phi(z \mid x)$

### A. Le bon $q$ dépend de $x$

Le calcul du gap (§IV.B) a une conséquence qu'on n'avait pas explicitée : il dit que **l'optimum de $q$ dépend de $x$**.

En effet, le $q$ qui rend la borne serrée pour un $x^{(1)}$ donné est $p_\theta(z \mid x^{(1)})$ — la posterior pour ce $x$ précis. Pour un autre $x^{(2)}$, l'optimum est $p_\theta(z \mid x^{(2)})$, qui est une **distribution différente**. Pour $x^{(1)}$ = image d'un 7, la posterior est concentrée sur les $z$ qui décodent en 7 ; pour $x^{(2)}$ = image d'un 3, ailleurs.

Donc la notation $q(z)$ qu'on a utilisée depuis §III était trompeuse. **Honnêtement, il faudrait noter $q^{(i)}(z)$** : un $q$ différent par exemple du dataset. Pour chaque image $x^{(i)}$, on veut une distribution propre sur $z$.


en gros pour comprendre le p_theta on peut voir le graphe comme ça si tes images sont toutes en 1D

j'ai une distribution marginale X qui est la distribution des images et Z la distribution marginale de l'espace latent quand je note $p_{\theta}(z \mid X=x^{(1)})$ ça me donne la distribution conditionnel de l'espace latent pour cette image $x^{(1)}$ qu'on va ensuite renormaliser pour que ça intègre à 1 . On peut aussi dessiner la jointe $p_{\theta}(x,z)$   c'est une surface 2D ou de façon équivalente un nuage de densité dans le plan $(x,z)$.

et en gros le délire c'est que $p_{\theta}(z \mid X=x^{(1)})$ est incalculable et c'est $q_{\phi}(z \mid x)$ qu'on apprend à approximer cette distribution-là (où plutôt sa moyenne et sa variance dans la famille gaussienne).

![[im3-5 (1).png|515]]

et c'est la ou c'est compliqué c'est que en grande dimension p_theta(z | X=x^{(1)}) est un petit nuage gaussien dans R^64 centré sur un point mu^{(1)} le truc c'est qu'on dit que c'est entangled (représentations enchevêtrées) car tout est mélangé dans ce nuage je vais pas avoir précisément une information très précise du type "style penché, fin" ou "style droit, épais" si je dois détecter l'image du chiffre "7".

il parle apres que y'a eu trois améliorations : 
1. interpolation marche
2. directions sémantiques existent
3. le beta-VAE fait du désentanglement on obtient des axes interprétables



autre image
![[im4-4.png]]





### B. Variational inference classique : un $q$ par exemple

L'approche directe s'appelle **inférence variationnelle classique** (variational EM stochastique). Procédure :

1. **Pour chaque $x^{(i)}$ du dataset**, paramétriser $q^{(i)}(z) = \mathcal{N}(\mu^{(i)}, \sigma^{(i)2} I)$ et **optimiser** $\mu^{(i)}, \sigma^{(i)}$ par descente de gradient pour maximiser l'ELBO sur cet exemple.
2. Mettre à jour les paramètres $\theta$ du décodeur sur ces mêmes exemples.
3. Répéter.

**Problème de coût.** L'étape 1 demande une descente de gradient **par exemple d'entraînement**. Sur MNIST (60k exemples) c'est encore faisable ; sur ImageNet ou un dataset moderne (millions à milliards d'exemples), c'est rédhibitoire. À chaque epoch, on devrait relancer l'optimisation des $q^{(i)}$.

Pire : un nouvel exemple (à l'inférence, sur des données non vues) demande une optimisation complète pour obtenir son $q$. Pas de mode "inférence rapide".

### C. L'idée d'amortization : apprendre un mapping

> [!quote] Amortization (Kingma & Welling, 2014)
> Plutôt qu'apprendre les $q^{(i)}$ **séparément** pour chaque exemple, observer que la collection $\{q^{(i)}\}_{i=1}^{m}$ est en réalité **un mapping** $x \mapsto (\mu, \sigma)$ : pour chaque $x^{(i)}$, on aimerait calculer les paramètres $\mu^{(i)}, \sigma^{(i)}$. Apprenons directement ce mapping avec un seul réseau de neurones — l'**encodeur** $\phi$.

On paramétrise donc $q$ comme une fonction de $x$ :

$$q_\phi(z \mid x) = \mathcal{N}\!\big(\mu_\phi(x), \, \sigma_\phi^2(x) \, I\big),$$

où $\mu_\phi$ et $\sigma_\phi$ sont les sorties d'un réseau de neurones. En pratique le réseau sort $\mu$ et $\log \sigma$ (pour garantir $\sigma > 0$), avec une tête $\mu$ et une tête $\log \sigma$ partageant le tronc.

**Décodage de la notation.** $q_\phi(z \mid x)$ se lit "$q$ paramétré par $\phi$, conditionné sur $x$". Pour un $x$ fixé, c'est une distribution sur $z$ — exactement le rôle qu'avait $q^{(i)}(z)$. Mais le **réseau $\phi$ est partagé** entre tous les $x$ : on apprend une recette générale, pas une infinité de distributions séparées.

> [!note] L'analogie NADE/FVSBN
> C'est *exactement* le même saut conceptuel que FVSBN → NADE dans `[[01_Modèles autoregressifs#III. Premières solutions paramétrées]]`.
> 
> - **FVSBN** : un prédicteur par position $i$, avec ses propres poids $w^{(i)}$ — analogue de "un $q^{(i)}$ par exemple".
> - **NADE** : un réseau partagé qui produit la bonne CPD pour toute position $i$ — analogue de "un $q_\phi(z \mid x)$ qui produit le bon $q$ pour tout $x$".
> 
> L'amortization, c'est NADE appliqué à la posterior.

|    Amortization    |     |
| :----------------: | :-: |
| ![[im11.png\|306]] |     |

> [!warning] Important : $q_\phi$ approxime, ne reproduit pas
> $q_\phi(z \mid x)$ ne vise pas à reproduire **exactement** la posterior $p_\theta(z \mid x)$ (qu'on ne sait pas calculer de toute façon — elle nécessiterait $p_\theta(x)$ par Bayes, l'intractable de départ). Le réseau $\phi$ va trouver, par SGD conjointe avec $\theta$, le **meilleur compromis** dans la famille restreinte qu'on lui permet (gaussiennes diagonales).
> 
> C'est pour ça que les VAE produisent des échantillons un peu flous : la posterior réelle est souvent multimodale ou complexe, mais $q_\phi$ ne sait sortir qu'une gaussienne unimodale. Le compromis est imparfait par construction.

### D. L'ELBO avec amortization

L'ELBO devient maintenant fonction de deux ensembles de paramètres :

$$\mathcal{L}(x; \theta, \phi) \;=\; \mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)] \;-\; D_{\text{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big).$$

- $\phi$ = paramètres de l'**encodeur** (inference network).
- $\theta$ = paramètres du **décodeur** (generative network).
- L'entraînement se fait **conjointement** sur $(\theta, \phi)$ par SGD — sur les **mêmes mini-batches**. Pour chaque $x^{(i)}$, un seul forward de l'encodeur donne $\mu, \sigma$ ; plus besoin d'optimiser un $q^{(i)}$ séparé.

## VI. Le reparametrization trick

### A. Le problème : un nœud aléatoire bloque la backprop

L'ELBO contient une espérance $\mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$. En pratique on l'estime par Monte Carlo avec un seul échantillon par $x$ (la variance est acceptable) :

$$\mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)] \;\approx\; \log p_\theta(x \mid z), \quad z \sim q_\phi(z \mid x).$$

**Mais comment rétropropager le gradient à travers le tirage $z \sim q_\phi(z \mid x)$ ?** Un nœud stochastique dans le graphe de calcul est une **fonction non-différentiable** par rapport à ses paramètres : on ne peut pas dériver "tirer aléatoirement" par rapport à $\mu_\phi, \sigma_\phi$.

### B. La solution : déplacer le tirage hors du graphe

Le reparametrization trick observe que tirer $z \sim \mathcal{N}(\mu, \sigma^2 I)$ est équivalent à :

$$\boxed{\; z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I) \;}$$

où $\varepsilon$ est tiré **indépendamment de $\phi$**. Tout le bruit aléatoire est isolé dans $\varepsilon$ ; le reste du calcul est déterministe en $(\mu_\phi, \sigma_\phi)$ et donc différentiable.

Le diagramme ci-dessus illustre la différence : à gauche, le nœud $z \sim \mathcal{N}(\mu, \sigma^2)$ casse le graphe et la backprop ne peut pas passer ; à droite, $\varepsilon$ est tiré hors du graphe et $z = \mu + \sigma \odot \varepsilon$ devient une opération différentiable par rapport à $\mu$ et $\sigma$.

> [!note]- Argument de variance : reparam vs REINFORCE
> Il existe une alternative au reparam — l'estimateur **REINFORCE** (gradient via la log-derivative trick) :
> 
> $$\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)] = \mathbb{E}_{z \sim q_\phi}[f(z) \, \nabla_\phi \log q_\phi(z)].$$
> 
> REINFORCE marche pour des $z$ discrets comme continus (avantage), mais sa **variance est très élevée** parce qu'il ne tire aucun parti de la structure différentiable de $f$.
> 
> Le reparam, lui, exige $z$ continu et $f$ différentiable, mais sa variance est **bien plus faible** (de plusieurs ordres de grandeur en pratique). C'est ce qui rend l'entraînement de VAE stable et faisable.
> 
> Une expérience simple sur $\min_\theta \mathbb{E}_{q_\theta}[x^2]$ avec $q_\theta = \mathcal{N}(\theta, 1)$ montre que la variance de REINFORCE décroît en $1/N$ classique, mais à un niveau ~10× supérieur à celle du reparam, pour le même nombre d'échantillons. Pour les latents discrets, on utilise des relaxations continues (Gumbel-Softmax) ou des estimateurs hybrides (REBAR, RELAX) — voir `[[Estimateurs de gradient]]` (à venir).

|          Avant          | Après reparametrization |     |
| :---------------------: | :---------------------: | :-: |
| ![[im2-1 (3).png\|198]] | ![[im10 (1).png\|194]]  |     |

## VII. L'architecture complète

### A. Vue d'ensemble

> [!warning] Anatomie d'un VAE
> 1. **Encodeur** $\phi$ : $x \mapsto (\mu_\phi(x), \log \sigma_\phi(x))$.
> 2. **Sampling reparametrisé** : $\varepsilon \sim \mathcal{N}(0, I)$, $\; z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$.
> 3. **Décodeur** $\theta$ : $z \mapsto$ paramètres de $p_\theta(x \mid z)$ (typiquement la moyenne $f_\theta(z)$ d'une gaussienne ou les logits d'une Bernoulli).
> 4. **Loss** : négatif de l'ELBO,
>    $$-\mathcal{L}(x; \theta, \phi) \;=\; \underbrace{-\log p_\theta(x \mid z)}_{\text{reconstruction}} \;+\; \underbrace{D_{\text{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{régularisation}}.$$
> 5. **Backprop** sur $\theta$ et $\phi$ simultanément, SGD.

### B. Calcul des deux termes

**Le terme KL est analytique.** Pour deux gaussiennes diagonales $q_\phi(z \mid x) = \mathcal{N}(\mu, \sigma^2 I)$ et $p(z) = \mathcal{N}(0, I)$ :

$$D_{\text{KL}}\!\big(\mathcal{N}(\mu, \sigma^2 I) \,\|\, \mathcal{N}(0, I)\big) = \frac{1}{2} \sum_{j=1}^{d} \big( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \big).$$

Calcul direct, pas d'approximation. C'est cadeau.

**Le terme de reconstruction se calcule selon la nature de $x$.** C'est un point souvent mal compris :

> [!warning] La sortie du décodeur n'est PAS l'image
> Beaucoup de tutoriels représentent $f_\theta(z)$ comme "l'image reconstruite". C'est trompeur : $f_\theta(z)$ est la **paramètre** d'une distribution sur $x$, pas l'image. La perte de reconstruction est $-\log p_\theta(x \mid z)$, dont la forme dépend du choix de $p_\theta(x \mid z)$ :
> 
> - **$p_\theta(x \mid z) = \mathcal{N}(f_\theta(z), I)$** (images en réel, audio) → $-\log p_\theta(x \mid z) = \frac{1}{2} \|x - f_\theta(z)\|^2 + \text{const}$. **MSE.**
> - **$p_\theta(x \mid z) = \prod_j \text{Bern}(f_\theta(z)_j)$** (MNIST binaire) → $-\log p_\theta(x \mid z) = \sum_j \text{BCE}(x_j, f_\theta(z)_j)$. **Binary cross-entropy.**
> - **$p_\theta(x \mid z) = \prod_j \text{Cat}_{256}(f_\theta(z)_j)$** (images 8-bit) → cross-entropy catégorielle sur 256 classes par pixel.
> 
> Sur MNIST en flottants [0, 1], les deux premiers donnent visuellement des résultats similaires (d'où la confusion), mais **mathématiquement le MSE suppose une vraisemblance gaussienne**. Bien choisir $p_\theta(x \mid z)$ change la nature du modèle.

### C. Pseudocode

```
for x in batch:
    mu, log_sigma = encoder(x)          # phi
    eps = randn_like(mu)
    z = mu + exp(log_sigma) * eps       # reparametrization
    x_recon_params = decoder(z)         # theta
    
    recon_loss = -log_p(x | x_recon_params)
    kl_loss = 0.5 * sum(mu**2 + exp(2*log_sigma) - 1 - 2*log_sigma)
    
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()
```

## VIII. Génération et inférence

### A. Génération (échantillonnage)

Une fois le modèle entraîné, on jette l'encodeur :

1. tirer $z \sim p(z) = \mathcal{N}(0, I)$ ;
2. passer $z$ dans le décodeur, obtenir les paramètres de $p_\theta(x \mid z)$ ;
3. tirer $x$ depuis cette distribution (ou prendre sa moyenne $f_\theta(z)$ comme estimateur MAP).

**Pourquoi ça marche ?** Parce que le terme KL de la loss a forcé $q_\phi(z \mid x)$ à rester proche de $\mathcal{N}(0, I)$ pour tous les $x$ du dataset. Donc tirer $z$ du prior tombe dans une zone que le décodeur sait décoder — précisément ce qu'un AE classique ne pouvait pas garantir.

**Avantage massif sur les autorégressifs :** la génération est **un seul forward pass** du décodeur. Pas de séquentialité, pas de $n$ étapes. C'est rapide, parallélisable, et c'est la raison pour laquelle on a longtemps préféré les VAE aux AR pour la génération d'images.

### B. Inférence (abstraction)

À partir d'un $x$ donné, on peut récupérer une représentation latente :

1. passer $x$ dans l'encodeur, obtenir $\mu_\phi(x)$, $\sigma_\phi(x)$ ;
2. tirer $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$ (ou prendre $\mu_\phi(x)$ si on veut une représentation déterministe).

Ce $z$ est un **vecteur de petite dimension** qui résume $x$. Usages :

- **Interpolation.** Entre deux images, on peut interpoler entre leurs $z$ latents et décoder le résultat — on obtient un morphing fluide. Impossible avec un AE classique.
- **Génération conditionnelle.** Manipuler $z$ dans une direction interprétée (style, posture, expression).
- **Représentation pour tâches downstream** (classification, retrieval) — souvent meilleure qu'une représentation supervisée classique sur peu de données.

## IX. Positionnement par rapport aux autres familles

> [!warning] Bilan VAE
> **Forces :**
> - **Espace latent $z$ explicite** — interpolation, manipulation, représentation utilisable.
> - **Génération rapide** — un seul forward pass du décodeur.
> - **Cadre probabiliste complet** — l'ELBO est une borne sur la log-vraisemblance, principe d'apprentissage théoriquement bien fondé.
> - **Extensible** — VAE conditionnels, β-VAE pour le désentanglement, VQ-VAE pour les latents discrets, etc.
> 
> **Faiblesses :**
> - **Densité $p_\theta(x)$ non exacte** — on n'a qu'une borne (l'ELBO), souvent assez lâche en pratique. Mauvais pour la compression, comparaison de modèles par log-likelihood, détection d'anomalies fine.
> - **Échantillons un peu flous** — la sortie du décodeur tend à moyenner sur plusieurs modes de $p(x \mid z)$, donnant des images moins nettes qu'un GAN ou un diffusion model. Phénomène lié au choix de la likelihood gaussienne pour des données complexes.
> - **Posterior collapse** — phénomène où $q_\phi(z \mid x) \approx p(z)$ pour tous les $x$ (le latent est ignoré, le décodeur reproduit la moyenne du dataset). Cf. β-VAE, KL annealing pour mitiger.

### Tableau mis à jour

| Aspect | Autorégressif | **VAE** | Flow | GAN | Diffusion |
|---|---|---|---|---|---|
| Structure | Chain rule, ordre total | **Latent $z$ + décodeur** | Bijection $z \leftrightarrow x$ | Latent $z$ + générateur | Markov de débruitage |
| Densité $p(x)$ | Exacte | **Borne (ELBO)** | Exacte | Inaccessible | Approchée |
| Apprentissage | MLE direct | **ELBO (variational)** | MLE | Adversarial | Score matching |
| Échantillonnage | Lent (séquentiel) | **Rapide (1 forward)** | Rapide | Rapide | Itératif (lent) |
| Espace latent | Non | **Oui, structuré** | Oui, bijectif | Oui, non-régulier | Implicite |

Le VAE est le point d'équilibre du tableau : il sacrifie la densité exacte pour gagner un latent structuré et une génération rapide. C'est le compromis qui définit la famille.

## X. Suite et extensions

Les notes suivantes du dossier exploreront :

- **`[[03_Normalizing Flows]]`** — comment garder un latent (comme VAE) **et** une densité exacte (comme AR), en imposant que le décodeur soit bijectif.
- **`[[04_GAN]]`** — abandonner complètement la vraisemblance, apprendre par jeu adversarial, gagner en qualité visuelle au prix de l'inférence.
- **`[[05_Diffusion]]`** — un VAE *hiérarchique* avec $T$ latents emboîtés et une structure markovienne de débruitage. L'ELBO du VAE se généralise directement.

> [!todo] Extensions à étoffer ultérieurement
> - **β-VAE** (Higgins et al., 2017) : pondérer le terme KL pour pousser au désentanglement.
> - **IWAE** (Burda et al., 2015) : borne ELBO plus serrée via importance sampling avec $K$ échantillons.
> - **VQ-VAE** (van den Oord et al., 2017) : latent discret, base de DALL-E et de nombreux modèles modernes.
> - **Conditional VAE** : $p_\theta(x \mid z, y)$ pour génération conditionnée.

---

## Pour aller plus loin

- **Article fondateur.** D.P. Kingma, M. Welling. *Auto-Encoding Variational Bayes.* ICLR 2014. arXiv:1312.6114.
- **Tutoriel pédagogique.** C. Doersch. *Tutorial on Variational Autoencoders.* arXiv:1606.05908. Excellente exposition complémentaire.
- **Survey complet.** D.P. Kingma, M. Welling. *An Introduction to Variational Autoencoders.* Foundations and Trends in ML, 2019.
- **Reparametrization trick.** Article original Kingma-Welling + le blog post de G. Gundersen, *The Reparameterization Trick* (2018), pour l'angle gradient-estimator.
- **Variational inference, vue d'ensemble.** D.M. Blei, A. Kucukelbir, J.D. McAuliffe. *Variational Inference: A Review for Statisticians.* JASA 2017.
