---
title: Evaluating Generative Models
---
# Evaluating Generative Models

> Une fois qu'on a vu plusieurs familles de modèles génératifs (autorégressif, VAE, flow, GAN, EBM, diffusion, flow matching), reste la question pratique : **comment dire qu'un modèle est meilleur qu'un autre ?** L'évaluation dépend critiquement de la tâche qu'on cible. Trois grands axes : (1) **density estimation**, (2) **qualité d'échantillonnage**, (3) **représentations latentes**. Aucun de ces trois critères n'est trivial, et selon le modèle (à likelihood tractable ou non), des techniques très différentes s'appliquent.

## I. Trois axes d'évaluation

Évaluer un modèle génératif est subtil : il faut spécifier la **tâche** qu'on vise.

- **Density estimation.** Quelle probabilité le modèle assigne-t-il à un point de données issu de la vraie distribution ? Centrale pour les modèles à vraisemblance.
- **Sampling / Generation.** Quelle est la qualité visuelle (ou autre) des échantillons générés ? Important pour GAN, diffusion, etc.
- **Latent Representation Learning.** Le modèle apprend-il des patterns cachés utiles ? Important pour les tâches downstream (semi-supervisé, image translation, compressed sensing).

Ces axes ne sont pas indépendants : un VAE qui maximise un ELBO apprend simultanément un latent. Mais sans **mesure quantitative**, impossible de comparer deux modèles. On parcourt les trois en détail.

## II. Density Estimation

### A. Modèles à likelihood tractable

Pour les modèles autorégressifs et les flows, la densité est directement calculable. Procédure standard :

1. Split du dataset en train / validation / test.
2. Calculer les gradients sur le train.
3. Tuner les hyperparamètres (learning rate, architecture) sur la validation.
4. **Évaluer la log-vraisemblance finale sur le test** — seul chiffre qui compte pour la généralisation.

> [!tip] Seul le test log-likelihood compte
> Si tu reportes une vraisemblance dans un papier, c'est **la test log-likelihood**, pas la train. Le train sert à entraîner, la validation à tuner, le test à reporter.

### B. Modèles à likelihood intractable

Pour VAE et GAN, on n'a pas accès direct à $p_\theta(x)$ :

- **VAE** : on a l'ELBO, qui est une **borne inférieure** sur la vraie log-vraisemblance. Suffit pour comparer deux VAE entre eux (en sachant que la vraie likelihood sera toujours plus haute).
- **GAN** : aucune évaluation de likelihood. Il faut une mesure qui dépende uniquement des **samples** — c'est l'objet du **Kernel Density Estimation** ci-dessous.

### C. Density estimation par binning

Soit un modèle $p_\theta(x)$ dont la densité est mal définie ou intractable. On tire $\mathcal{S} = \{x^{(1)}, \ldots, x^{(6)}\}$ et on veut estimer $p_\theta(-0.5)$ :

| $x^{(1)}$ | $x^{(2)}$ | $x^{(3)}$ | $x^{(4)}$ | $x^{(5)}$ | $x^{(6)}$ |
| --------- | --------- | --------- | --------- | --------- | --------- |
| -2.1      | -1.3      | -0.4      | 1.9       | 5.1       | 6.2       |

**Réponse naïve.** Si on regarde la fréquence exacte, $p_\theta(-0.5) = 0$ car $-0.5 \notin \mathcal{S}$. Inutile.

**Approche binning.** Diviser l'axe en bins (ici largeur $2$). Pour chaque bin, compter les points et diviser par $N \times \text{largeur}$ pour normaliser :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im1 (3).png]]

> [!warning] Problème du binning
> Très sensible aux **frontières** : $p_\theta(-1.99) = 1/6$ mais $p_\theta(-2.01) = 1/12$, alors qu'ils sont quasi-identiques. Discontinuités absurdes, haute variance.

### D. Kernel Density Estimation (KDE)

**Idée.** Au lieu de bins durs, on pose un **noyau** autour de chaque point d'entraînement, et l'estimateur final est la moyenne de ces noyaux :

$$\boxed{\;\hat{p}(x) = \frac{1}{n} \sum_{x^{(i)} \in \mathcal{S}} K\!\left(\frac{x - x^{(i)}}{\sigma}\right)\;}$$

- $\sigma$ : **bandwidth** (largeur du noyau).
- $K$ : **fonction noyau** (typiquement gaussienne).

**Propriétés requises d'un noyau** : $\int K(u)\, du = 1$ (normalisation, pour que KDE soit une densité), et $K(u) = K(-u)$ (symétrie).

**Exemple 1 — noyau linéaire.** Sample $[-2, -1, 0, 1, 2]$, $K(a) = 1 - |a|/h$ avec $h = 10$. Pour $x = 0$ :

$$p(0) = \frac{1}{5 \times 10}\, (0.8 + 0.9 + 1 + 0.9 + 0.8) = 0.088.$$

**Exemple 2 — noyau gaussien.** $K(u) = (1/\sqrt{2\pi})\, \exp(-u^2/2)$. On pose une gaussienne par point, on les somme :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im22.png]]

Résultat lisse, $p(-1.99) \approx p(-2.01)$ — plus de discontinuités.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im2 (3).png]]

**Choix de $\sigma$.** Critique : trop petit, on sur-fitte les points ; trop grand, on lisse trop et la densité devient plate. On **cross-valide** sur un set de validation.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im3 (4).png]]

Sur la figure : courbe noire = $\sigma$ optimal, recouvre presque parfaitement la vraie densité (grise). Vert = $\sigma$ trop grand (squashed), trop petit = oscille.

> [!warning] KDE meurt en haute dimension
> KDE était utilisé pour le GAN original (2014), mais **plus personne** depuis ~2019. En haute dimension, la notion de distance perd son sens : décaler une image de quelques pixels change drastiquement la distance, même si l'image est sémantiquement la même.

### E. Likelihood Weighting

Pour un modèle latent on a $p(x) = \mathbb{E}_{p(z)}[p(x \mid z)]$. Estimateur Monte Carlo naïf : tirer $z \sim p(z)$ et moyenner $p(x \mid z)$.

> [!warning] Variance énorme
> Si le prior $p(z)$ est très différent de la postérieure $p(z \mid x)$, l'estimateur a une variance gigantesque — on tire majoritairement dans des zones où $p(x \mid z) \approx 0$. C'est précisément ce qui motive **l'inférence variationnelle** (cf. VAE).

### F. Annealed Importance Sampling (AIS)

**Technique générique** pour estimer la constante de normalisation d'une distribution, ou plus généralement la vraisemblance d'un modèle latent. L'idée : interpoler graduellement entre deux distributions $p(z)$ et $p(x \mid z)$ via une **chaîne de distributions intermédiaires**, et tracker le ratio cumulé.

> [!note] Biais log-likelihood
> AIS donne un estimateur **non biaisé** de la likelihood. Mais par l'inégalité de Jensen, un estimateur non biaisé de $k$ donne un estimateur **biaisé** de $\log k$. Donc pour la log-vraisemblance, attention. Implémentation disponible dans TensorFlow Probability.

## III. Qualité d'échantillonnage

### A. Pourquoi c'est difficile

Comment dire qu'une image générée est meilleure qu'une autre ?

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im4 (1).png]]

- **Évaluations humaines** (Mechanical Turk, etc.) : chères, biaisées, dures à reproduire.
- **Mémorisation du training set** donnerait des samples parfaits — clairement indésirable. Difficile de définir formellement la généralisation.
- **Tâche qualitative** ⇒ pas de réponse unique.

Métriques principales : **Inception Score (IS)**, **Frechet Inception Distance (FID)**, **Kernel Inception Distance (KID)**.

### B. Fidélité et diversité

Un bon modèle génératif doit produire des samples qui satisfont **deux critères** :

**(i) Sharpness (Fidélité, $S$).** Le classifieur est confiant sur chaque image générée — sa distribution prédictive $c(y \mid x)$ a une **entropie faible** :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im5.png]]

Mathématiquement : on tire $x$ depuis le générateur, on regarde l'entropie négative de $c(y \mid x)$ — plus elle est élevée, plus le modèle est sharp.

**(ii) Diversité ($D$).** L'**ensemble** des prédictions doit être varié — la marginale $c(y) = \mathbb{E}_{x \sim p}[c(y \mid x)]$ doit avoir une **entropie élevée** (on ne génère pas que des chiens) :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im6 (2).png]]

### C. Distance pixel vs distance feature

**Distance pixel.** Difference brute des pixels — parfaite ici (distance nulle) :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im13.png]]

**Mais non-fiable.** Décaler de un pixel donne une distance de $900$ pour des images sémantiquement identiques :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im14.png]]

**Distance feature.** Plutôt comparer des **features haut niveau** (l'image a-t-elle deux yeux ? de la fourrure ?) qui sont **insensibles** aux petits décalages. On extrait ces features depuis un **classifieur pré-entraîné** (typiquement **Inception-v3**, entraîné sur **ImageNet** — 14M images, 20k catégories).

**Architecture Inception-v3.** 42 couches, conçue pour la classification mais utile comme feature extractor. On coupe le softmax final et on récupère l'embedding de la dernière couche de pooling, de dimension $2048$ :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im15.png]]

On note $\Phi(x) \in \mathbb{R}^{2048}$ l'embedding d'une image $x$. Pour comparer real vs fake dogs : on compare leurs distributions d'embeddings.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im16.png]]

### D. Fréchet Inception Distance (FID)

**Fréchet distance** entre deux distributions :

$$d^2(F, G) = \min_{X, Y} \mathbb{E}\big[\|X - Y\|^2\big].$$

Pour deux **gaussiennes multivariées** $X_r \sim \mathcal{N}(\mu_r, \Sigma_r)$ et $X_g \sim \mathcal{N}(\mu_g, \Sigma_g)$, la formule a une forme fermée :

$$\boxed{\;\mathrm{FID} = \|\mu_X - \mu_Y\|^2 + \mathrm{Tr}\!\big(\Sigma_X + \Sigma_Y - 2\sqrt{\Sigma_X \Sigma_Y}\big)\;}$$

**FID** = Fréchet distance entre les distributions des activations Inception-v3 (pool3, dim 2048), en supposant qu'elles sont gaussiennes.

> [!important] Plus FID est bas, mieux c'est
> $\mathrm{FID} = 0$ : distributions identiques. FID est la métrique standard pour comparer GAN, diffusion, etc.

**Limites du FID.**

- Dépend du dataset d'entraînement d'Inception (ImageNet). Sur MNIST par exemple, Inception n'a vu aucun chiffre — features inadéquates.
- Biais à petit sample size — il faut $\sim 50k$ images pour un FID stable.
- Calcul relativement lent.
- Ne dépend que des deux premiers moments (moyenne, covariance) — on perd l'asymétrie et le kurtosis.

### E. Inception Score (IS)

Proposé par [Salimans et al. 2016](https://arxiv.org/abs/1606.03498). Plus utilisé depuis FID, mais reste cité.

On garde Inception-v3 intact (incluant le softmax) — pour chaque sample on a une distribution de classe $p(y \mid x)$ :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im17.png]]

**Objectif** : combiner haute fidélité (entropie de $p(y \mid x)$ basse, le modèle est sûr) et haute diversité (entropie de $p(y) = \mathbb{E}_x[p(y \mid x)]$ haute, on couvre toutes les classes) :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im23.png]]

$$\boxed{\;\mathrm{IS} = \exp\!\Big(\mathbb{E}_{x \sim p_g}\, \mathrm{KL}\big(p(y \mid x)\, \|\, p(y)\big)\Big)\;}$$

L'IS minimal est $1$ ; le maximum est le nombre de classes du classifieur.

**Exemple numérique.** Avec 3 images, $p(y \mid x)$ en lignes :

$$p(y \mid x) = \begin{pmatrix} 0.8 & 0.1 & 0.1 \\ 0.2 & 0.6 & 0.2 \\ 0.55 & 0.25 & 0.30 \end{pmatrix}$$

Marginale (moyenne par colonne) : $p(y) = (0.52, 0.32, 0.2)$. KL par image (en supposant un broadcast standard) :

$$\mathrm{KL} = p(y \mid x) \odot \big[\log(p(y \mid x)) - \log(p(y))\big] \;\Rightarrow\; \mathrm{KL}_{1,2,3} = (0.17, 0.19, 0.1).$$

Espérance : $\overline{\mathrm{KL}} = 0.15$, puis $\mathrm{IS} = e^{0.15} \approx 1.17$.

**Intuition de la marginale.** On veut que la marginale soit **uniforme** (= maximum d'entropie) — c'est exactement la diversité :

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im24.png]]

### F. Kernel Inception Distance (KID)

**Maximum Mean Discrepancy (MMD).** Comparer deux distributions $p$ et $q$ uniquement par leur **moyenne** est insuffisant — deux distributions de même moyenne mais variances différentes seraient considérées identiques.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im25.png]]

L'idée MMD : mapper chaque point dans un espace de plus haute dimension où les moments d'ordre supérieur sont capturés, par exemple $x \mapsto (x, x^2)$. La distance entre les moyennes dans cet espace devient :

$$\big\|p - q\big\|^2 = \Big\| \frac{1}{n}\sum_i \phi(x_i) - \frac{1}{m}\sum_j \phi(y_j) \Big\|^2.$$

**Astuce du noyau.** On n'a pas besoin de calculer $\phi$ explicitement — on développe le carré :

$$
\begin{aligned}
\big\|p - q\big\|^2 &= \tfrac{1}{n^2}\sum_{i,j} K(x_i, x_j) + \tfrac{1}{m^2}\sum_{i,j} K(y_i, y_j) - \tfrac{2}{nm}\sum_{i,j} K(x_i, y_j) \\
&= \mathbb{E}_{x, x' \sim p}[K(x, x')] + \mathbb{E}_{y, y' \sim q}[K(y, y')] - 2\, \mathbb{E}_{x \sim p, y \sim q}[K(x, y)]
\end{aligned}
$$

Tous les noyaux ne marchent pas (il faut capturer tous les moments). Un bon choix est le **RBF** :

$$K(x_i, x_j) = \exp\!\Big(\!-\frac{\|x_i - x_j\|^2}{\gamma}\Big).$$

**KID** : MMD calculée sur les features Inception.

**FID vs KID.**

| Aspect          | FID                  | KID                       |
| --------------- | -------------------- | ------------------------- |
| Biais           | Biaisé (≥ 0)         | **Non biaisé**            |
| Coût            | $O(n)$               | $O(n^2)$                  |
| Hypothèse       | Gaussienne sur features | Aucune (free-form via noyau) |

### G. Truncation trick

Astuce post-entraînement qui **trade fidélité contre diversité**. On échantillonne le vecteur de bruit $z \sim \mathcal{N}(0, I)$, mais on **tronque** la distribution : on rejette les $z$ dont la norme dépasse un seuil.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im26.png]]

- **Tronquer fort** (rester près de $0$) → **haute fidélité**, faible diversité. Le générateur a beaucoup de pratique dans cette zone.
- **Tronquer peu** (queues complètes) → **haute diversité**, mais beaucoup de samples bizarres ($z$ dans les queues, peu vus à l'entraînement).

Le FID empire avec une troncature forte (manque de diversité), mais pour une application downstream où on veut des images très propres, c'est utile.

### H. Precision / Recall

Pour décomposer fidélité et diversité en deux mesures séparées (analogues au précision/rappel supervisé). Soit $P_r$ la distribution réelle et $P_g$ celle du générateur. Le scénario idéal : $P_g = P_r$.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im18.png|348]]

**Precision.** Fraction des samples générés qui sont dans le support réel :

$$\mathrm{Precision} = \frac{|\text{samples générés} \cap P_r|}{|\text{samples générés}|}.$$

Reflète la **fidélité** : haute precision = ce qu'on génère a l'air réel (peu de tennis-ball-dogs ou autres bidules).

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im19.png|396]]

**Recall.** Fraction des données réelles couvertes par le générateur :

$$\mathrm{Recall} = \frac{|\text{samples générés} \cap P_r|}{|\text{samples réels}|}.$$

Reflète la **diversité** : haut recall = on couvre toutes les variations du dataset, pas juste un sous-mode.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im20.png|434]]

**Constat empirique** : les state-of-the-art GAN ont souvent un **bon recall** mais une **précision médiocre** (ils couvrent tout le manifold avec beaucoup de gunk). Le truncation trick aide à corriger la précision au prix du recall.

![[images/3-Apprentissage automatique/05_Generative Models/evaluation/im21.png]]

### I. Perceptual Path Length

*Section à compléter.* (Métrique de StyleGAN : mesure la continuité de l'espace latent en regardant la distance perceptuelle entre samples produits par des $z$ proches.)

## IV. Évaluer les représentations latentes

Si le modèle apprend un latent $z$, comment juger sa qualité ? Trois angles classiques : **clustering**, **compression**, **disentanglement**.

### A. Clustering

L'idée : si le modèle apprend un bon latent, **les points d'une même classe** devraient se retrouver **proches dans l'espace latent**. On compare le clustering induit par le latent au clustering vrai (étiquettes connues).

Exemple : deux modèles génératifs sur MNIST, chacun produit une structure 2D dans le latent. On colore les points par leur classe vraie pour comparer.

**Trois métriques classiques** (pour évaluation avec labels disponibles) :

- **Completeness** : score haut quand tous les membres d'une même classe vraie atterrissent dans le même cluster prédit.
- **Homogeneity** : score haut quand chaque cluster prédit ne contient qu'une seule classe vraie.
- **V-measure** : moyenne harmonique des deux (équivalent à la **NMI**, *Normalized Mutual Information*) :

$$\mathrm{NMI}(\mathcal{C}, \mathcal{T}) = \frac{I(\mathcal{C}, \mathcal{T})}{\sqrt{H(\mathcal{C}) \cdot H(\mathcal{T})}}$$

NMI ∈ $[0, 1]$. $1$ = clustering parfait, $0$ = aucun rapport entre prédiction et vérité.

Références :
- [V-measure paper](https://www.aclweb.org/anthology/D07-1043.pdf)
- [NMI tutorial](https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf)
- Voir la lib `disentanglement_lib` pour les implémentations.

### B. Compression

Un bon modèle latent doit permettre de **compresser** les données. Lien fort avec la théorie de l'information (rate-distortion). Beaucoup de travaux sur la **compression avec perte** via VAE / flows. C'est un domaine à part entière, on ne creuse pas ici.

### C. Disentanglement

L'idée : un bon latent doit avoir des **dimensions interprétables** (couleur de peau, âge, sexe, orientation, etc.) — chaque axe du latent contrôle un **attribut** distinct.

**Utilité concrète** : génération contrôlable. Sur l'exemple typique, l'axe $z_1$ contrôle la taille, l'axe $z_2$ la rotation, etc. Si on bouge $z_1$ en gardant $z_2, z_3$ constants, seule la taille change.

**Métriques** : nombreuses (BetaVAE score, Factor-VAE score, MIG, SAP, DCI…) — toutes implémentées dans `disentanglement_lib`.

> [!warning] Le disentanglement non supervisé est impossible
> [Locatello et al., ICML 2019](https://arxiv.org/abs/1811.12359) ont prouvé **théoriquement et empiriquement** que sans **supervision implicite** (architecture biaisée, données structurées, inductive bias), le disentanglement unsupervised est impossible. Avant ce papier, des dizaines de travaux prétendaient le contraire — ils étaient en fait soit non-robustes au choix de seed, soit utilisaient implicitement des labels. Aujourd'hui le consensus est qu'il faut au moins un peu de signal extérieur pour obtenir un latent disentangled.
