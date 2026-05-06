---
title: MCMC
---
# MCMC — Markov Chain Monte Carlo

> Quand le posterior bayésien n'est pas conjugué, on ne peut ni l'écrire en forme fermée ni en échantillonner directement (cf. les méthodes de [[Échantillonnage]]). **MCMC** résout ce problème en construisant une chaîne de Markov dont la distribution stationnaire est exactement le posterior ciblé. Au lieu de tirer des échantillons i.i.d. (impossible), on génère une trajectoire corrélée qui finit par "explorer" le posterior dans les bonnes proportions.

## I. Pourquoi MCMC ?

### A. Le problème

En bayésien, on veut le posterior $f(\theta \mid y) \propto f(y \mid \theta) f(\theta)$. Trois scénarios :

1. **Conjugaison** : on reconnaît la forme du posterior (Beta, Gamma, Normal…), tout est analytique. Pas besoin de MCMC.
2. **Pas de conjugaison** : on connaît le posterior à une constante près, $f(\theta \mid y) \propto g(\theta)$, mais $g$ ne correspond à aucune loi standard. Impossible de calculer la constante de normalisation $\int g(\theta) \, \mathrm{d}\theta$ analytiquement, et impossible d'échantillonner directement.
3. **Posterior haute dimension** : même si on a la forme, l'intégration en dimension $d \gg 1$ est intractable.

**MCMC est la réponse aux cas 2 et 3.**

> [!example] Exemple canonique — vraisemblance Normale, prior Student
> Soit $Y_i \overset{iid}{\sim} \mathcal{N}(\mu, 1)$ et $\mu \sim t(0, 1, 1)$ (Student à 1 ddl, qui n'est *pas* conjugué à la Normale). Le posterior s'écrit :
>
> $$f(\mu \mid \boldsymbol{y}) \propto \frac{\exp\!\left[n(\bar{y} \mu - \mu^2 / 2)\right]}{1 + \mu^2}.$$
>
> Le numérateur ressemble à un noyau gaussien, mais le dénominateur $1 + \mu^2$ casse tout : ce n'est plus une distribution standard. On connaît la **forme** du posterior, mais on ne sait pas en échantillonner directement. C'est exactement le terrain de chasse de MCMC.

### B. Pourquoi pas Rejection Sampling ou Importance Sampling ?

Dans `Échantillonnage`, on a vu Rejection Sampling et Importance Sampling. Ils marchent en faible dimension, mais cassent en haute dimension : trouver une enveloppe $M q(\theta) \geq g(\theta)$ avec un $M$ raisonnable devient impossible quand $\theta \in \mathbb{R}^d$ avec $d$ grand. La masse de probabilité du posterior se concentre sur une région *typique* très petite par rapport au volume total — une proposition globale échoue presque toujours.

**MCMC contourne le problème** en proposant des coups *locaux* : on bouge depuis l'état courant, on n'essaie pas de couvrir tout l'espace d'un coup.

## II. Chaînes de Markov — le minimum à savoir

> [!note]- Rappel — chaîne de Markov
> Une suite $(\theta_0, \theta_1, \theta_2, \dots)$ est une **chaîne de Markov** si la loi de $\theta_{t+1}$ ne dépend que de $\theta_t$ (et pas du passé antérieur) :
>
> $$f(\theta_{t+1} \mid \theta_t, \theta_{t-1}, \dots, \theta_0) = f(\theta_{t+1} \mid \theta_t).$$
>
> La chaîne est caractérisée par son **noyau de transition** $K(\theta' \mid \theta) = f(\theta_{t+1} = \theta' \mid \theta_t = \theta)$.

**Distribution stationnaire.** Une distribution $\pi$ est dite **stationnaire** pour la chaîne si

$$\pi(\theta') = \int K(\theta' \mid \theta) \, \pi(\theta) \, \mathrm{d}\theta.$$
Autrement dit : si on tire $\theta_t \sim \pi$, alors $\theta_{t+1} \sim \pi$ aussi. Une fois qu'on est "dans" $\pi$, on y reste.

**Convergence.** Sous des conditions techniques (irréductibilité, apériodicité), la chaîne converge vers $\pi$ peu importe le point de départ :

$$\theta_t \xrightarrow{d} \pi \quad \text{quand } t \to \infty.$$
**Stratégie de MCMC.** Construire un noyau $K$ tel que la **distribution stationnaire soit le posterior** $\pi(\theta) = f(\theta \mid y)$. Faire tourner la chaîne assez longtemps, et à partir d'un certain point les $\theta_t$ sont des échantillons (corrélés) du posterior.

### A. Bilan détaillé (detailed balance)

Une condition **suffisante** pour que $\pi$ soit stationnaire :

$$\pi(\theta) \, K(\theta' \mid \theta) = \pi(\theta') \, K(\theta \mid \theta') \quad \forall \theta, \theta'.$$
Égalité symétrique : la masse qui passe de $\theta$ à $\theta'$ égale celle qui passe de $\theta'$ à $\theta$. La chaîne est dite **réversible**. Tous les algorithmes MCMC standards (Metropolis-Hastings, Gibbs) vérifient le detailed balance par construction.

## III. Metropolis-Hastings

L'idée centrale : pour chaque itération, on **propose** un nouveau point depuis une distribution $q$ qu'on sait échantillonner, puis on **décide** s'il faut l'accepter ou rester où on est. Le critère d'acceptation est calibré pour que la chaîne vérifie le detailed balance avec $\pi(\theta) \propto g(\theta)$.

### A. L'algorithme

> [!summary] Metropolis-Hastings
> **Entrée :** densité cible non-normalisée $g(\theta)$, distribution de proposition $q(\theta' \mid \theta)$, point initial $\theta_0$, nombre d'itérations $m$.
>
> Pour $i = 1, \dots, m$ :
> 1. **Proposer** un candidat $\theta^* \sim q(\theta^* \mid \theta_{i-1})$.
> 2. **Calculer le ratio d'acceptation** :
> $$\alpha = \frac{g(\theta^*) \, q(\theta_{i-1} \mid \theta^*)}{g(\theta_{i-1}) \, q(\theta^* \mid \theta_{i-1})}.$$
> 3. **Accepter ou rejeter** : tirer $u \sim \mathrm{Unif}[0, 1]$.
> - Si $u < \alpha$ : $\theta_i \leftarrow \theta^*$ (accept).
> - Sinon : $\theta_i \leftarrow \theta_{i-1}$ (reject, on reste).
>
> **Sortie :** trajectoire $(\theta_0, \theta_1, \dots, \theta_m)$.

**Lecture du ratio.** Le ratio $g(\theta^*) / g(\theta_{i-1})$ compare les densités cibles : si le candidat a une densité plus élevée, on est *toujours* tenté d'y aller. Le ratio $q(\theta_{i-1} \mid \theta^*) / q(\theta^* \mid \theta_{i-1})$ corrige le biais introduit par la proposition asymétrique. Quand $q$ est symétrique ($q(\theta' \mid \theta) = q(\theta \mid \theta')$), ce facteur disparaît.

> [!note]- Pourquoi $\alpha$ assure la stationnarité du posterior
> On veut que la chaîne vérifie le detailed balance avec $\pi(\theta) \propto g(\theta)$. Le noyau effectif de Metropolis-Hastings est :
>
> $$K(\theta' \mid \theta) = q(\theta' \mid \theta) \cdot \min\!\left(1, \alpha(\theta, \theta')\right) + \delta_\theta(\theta') \cdot r(\theta)$$
>
> où $r(\theta)$ est la probabilité de rejeter et $\delta_\theta$ est la masse de Dirac en $\theta$.
>
> En substituant $\alpha = \frac{g(\theta')q(\theta \mid \theta')}{g(\theta) q(\theta' \mid \theta)}$ et en vérifiant
>
> $$g(\theta) \, q(\theta' \mid \theta) \cdot \min(1, \alpha) = g(\theta') \, q(\theta \mid \theta') \cdot \min(1, 1/\alpha)$$
>
> on retrouve bien $\pi(\theta) K(\theta' \mid \theta) = \pi(\theta') K(\theta \mid \theta')$. La constante de normalisation s'annule — c'est *exactement* pour ça que MCMC peut travailler avec $g$ au lieu de $\pi$.

### B. Le cas symétrique — random walk Metropolis

Si $q$ est symétrique (typiquement $\theta^* \sim \mathcal{N}(\theta_{i-1}, s^2)$), le ratio se simplifie :

$$\alpha = \frac{g(\theta^*)}{g(\theta_{i-1})}.$$
C'est le **random walk Metropolis** : on perturbe l'état courant par un bruit gaussien et on accepte avec probabilité $\min(1, g(\theta^*) / g(\theta_{i-1}))$.

**Stabilité numérique.** Pour éviter les overflows quand $g$ est très petit, on travaille en log :

$$\log \alpha = \log g(\theta^*) - \log g(\theta_{i-1}), \quad \text{accepter si } \log u < \log \alpha.$$
C'est la version standard à implémenter en pratique.

### C. Choix de la proposition — le compromis du step size

Pour random walk Metropolis avec $\theta^* \sim \mathcal{N}(\theta_{i-1}, s^2)$, le paramètre crucial est $s$ (la *step size*).

**Trop petit** ($s$ tout petit) : tous les candidats sont proches de l'état courant, donc $g(\theta^*) \approx g(\theta_{i-1})$, donc $\alpha \approx 1$ et on accepte presque tout. Mais la chaîne se déplace à pas de fourmi : il faut un nombre énorme d'itérations pour explorer le posterior. **Taux d'acceptation trop élevé (> 50%)** = mauvais signe.

**Trop grand** ($s$ énorme) : les candidats tombent souvent dans des régions de très faible densité, donc $\alpha$ très petit et on rejette quasi tout le temps. La chaîne reste coincée. **Taux d'acceptation trop bas (< 20%)** = mauvais signe aussi.

> 💡 **Règle pratique.** Pour random walk Metropolis, viser un taux d'acceptation entre **23% et 50%**. La valeur asymptotiquement optimale (Roberts, Gelman, Gilks 1997) est $\approx 23\%$ en haute dimension, $\approx 44\%$ en dimension 1.

En pratique : on ajuste $s$ pendant une phase de tuning ("adaptive Metropolis"), puis on fixe $s$ pour la phase de sampling officielle.

### D. Exemple — le posterior Normal $\times$ Student

Reprenons le cas du I.A : $f(\mu \mid \boldsymbol{y}) \propto \dfrac{\exp[n(\bar{y}\mu - \mu^2/2)]}{1 + \mu^2}$, avec $\boldsymbol{y} = (1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9)$ donc $\bar{y} = 0.99$, $n = 10$.

On applique random walk Metropolis avec proposition $\mu^* \sim \mathcal{N}(\mu_{i-1}, s^2)$. Trois scénarios typiques selon $s$ :

| $s$ (step size) | Taux d'acceptation | Comportement |
|---|---|---|
| 3.0 (trop grand) | ~12% | la chaîne reste bloquée longtemps sur la même valeur |
| 0.05 (trop petit) | ~95% | la chaîne se déplace mais explore très lentement |
| 0.9 (correct) | ~38% | exploration efficace |

![[mcmc_random_walk_trace.png]]
**Figure 1.** Trace plots des trois régimes de step size pour random walk Metropolis. À gauche ($s = 3.0$, trop grand) : la chaîne reste bloquée sur des plateaux car la majorité des candidats sont rejetés. Au milieu ($s = 0.05$, trop petit) : la chaîne se déplace sans cesse mais à pas minuscules, donc explore très lentement. À droite ($s = 0.9$, calibré) : oscillation rapide autour de la valeur stationnaire, comportement souhaité.

Après convergence (avec $s = 0.9$), le posterior empirique se concentre autour de $\mu \approx 0.9$, qui penche bien plus côté données ($\bar{y} = 0.99$) que côté prior ($t(0,1,1)$ centré sur 0). C'est l'effet attendu : avec $n = 10$ observations, le prior est dominé par les données.

![[mcmc_random_walk_posterior.png|507]]
**Figure 2.** Comparaison du prior Student $t(0,1,1)$ (en pointillés), de la vraisemblance gaussienne (centrée sur $\bar{y} = 0.99$), et du posterior empirique obtenu par random walk Metropolis avec $s = 0.9$ (histogramme). Le posterior se déplace nettement vers les données, illustrant la mise à jour bayésienne.

### E. Limites de Metropolis-Hastings

**En haute dimension**, random walk Metropolis devient inefficace : la step size optimale décroît en $\mathcal{O}(d^{-1/2})$, et le nombre d'itérations nécessaires explose. C'est pour ça qu'on est passé à des méthodes plus sophistiquées comme HMC (cf. section VI).

**Multi-modal** : si le posterior a plusieurs modes séparés, random walk Metropolis a du mal à sauter de l'un à l'autre. Solutions : tempering, parallel tempering, ou méthodes globales (HMC ne résout pas ce problème non plus).

## IV. Gibbs sampling

Quand $\theta = (\theta_1, \dots, \theta_d)$ est multidimensionnel et qu'on connaît les **conditionnelles complètes** (full conditionals) en forme fermée, on peut s'épargner les rejections et échantillonner *exactement* une coordonnée à la fois. C'est le Gibbs sampler.

### A. L'idée

Factoriser le posterior par la règle de la chaîne :

$$f(\theta_1, \dots, \theta_d \mid y) = f(\theta_1 \mid \theta_{-1}, y) \cdot f(\theta_2 \mid \theta_{-2}, y) \cdots$$
où $\theta_{-k}$ désigne toutes les coordonnées sauf la $k$-ième. La clé : **chaque conditionnelle complète $f(\theta_k \mid \theta_{-k}, y)$ ne dépend que d'une seule coordonnée** (les autres sont fixées), donc beaucoup plus simple que le posterior complet.

Dans beaucoup de modèles (notamment hiérarchiques avec priors conjugués par bloc), ces conditionnelles ont une **forme analytique standard**. On les échantillonne directement.

### B. L'algorithme

> [!summary] Gibbs sampler
> **Entrée :** conditionnelles complètes $f(\theta_k \mid \theta_{-k}, y)$ pour $k = 1, \dots, d$, point initial $\theta^{(0)}$, nombre d'itérations $m$.
>
> Pour $i = 1, \dots, m$ :
> 1. Tirer $\theta_1^{(i)} \sim f(\theta_1 \mid \theta_2^{(i-1)}, \dots, \theta_d^{(i-1)}, y)$.
> 2. Tirer $\theta_2^{(i)} \sim f(\theta_2 \mid \theta_1^{(i)}, \theta_3^{(i-1)}, \dots, \theta_d^{(i-1)}, y)$.
> 3. ...
> $d$. Tirer $\theta_d^{(i)} \sim f(\theta_d \mid \theta_1^{(i)}, \dots, \theta_{d-1}^{(i)}, y)$.
>
> **Sortie :** trajectoire $(\theta^{(0)}, \theta^{(1)}, \dots, \theta^{(m)})$.

**Pas de rejet.** Contrairement à Metropolis-Hastings, chaque tirage est *accepté d'office* (taux d'acceptation = 100%). C'est parce qu'on tire de la vraie conditionnelle, pas d'une proposition approximative.

### C. Lien avec Metropolis-Hastings

Gibbs est en fait **un cas particulier de Metropolis-Hastings** : on prend $q(\theta_k^* \mid \theta_k, \theta_{-k}) = f(\theta_k \mid \theta_{-k}, y)$, c'est-à-dire la conditionnelle complète elle-même comme proposition. Le ratio devient :

$$\alpha = \frac{f(\theta_k^* \mid \theta_{-k}, y) \cdot f(\theta_k \mid \theta_{-k}, y)}{f(\theta_k \mid \theta_{-k}, y) \cdot f(\theta_k^* \mid \theta_{-k}, y)} = 1.$$
Donc on accepte toujours. Gibbs n'est pas une méthode "différente" de MH — c'est MH avec la proposition optimale.

### D. Exemple — Normal-InverseGamma

Modèle typique : $X_i \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ avec **les deux** paramètres inconnus, priors indépendants $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$ et $\sigma^2 \sim \mathrm{IG}(\nu_0, \beta_0)$.

Le posterior conjoint $f(\mu, \sigma^2 \mid \boldsymbol{x})$ n'est pas une loi standard (ni Normale, ni IG). Mais les **conditionnelles complètes** le sont :

**$\mu$ sachant $\sigma^2$ :** En traitant $\sigma^2$ comme connu, on retombe sur le cas Normal-Normal classique :

$$\mu \mid \sigma^2, \boldsymbol{x} \sim \mathcal{N}\!\left( \frac{n\bar{x}/\sigma^2 + \mu_0/\sigma_0^2}{n/\sigma^2 + 1/\sigma_0^2}, \; \frac{1}{n/\sigma^2 + 1/\sigma_0^2} \right).$$
**$\sigma^2$ sachant $\mu$ :** En traitant $\mu$ comme connu, on a une vraisemblance Normale en $\sigma^2$ avec moyenne fixée, et un prior IG :

$$\sigma^2 \mid \mu, \boldsymbol{x} \sim \mathrm{IG}\!\left( \nu_0 + \frac{n}{2}, \; \beta_0 + \frac{1}{2}\sum_{i=1}^n (x_i - \mu)^2 \right).$$
> [!note]- Dérivation des conditionnelles
> Le posterior conjoint :
>
> $$f(\mu, \sigma^2 \mid \boldsymbol{x}) \propto (\sigma^2)^{-n/2} \exp\!\left(-\frac{1}{2\sigma^2} \sum (x_i - \mu)^2\right) \cdot \exp\!\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right) \cdot (\sigma^2)^{-(\nu_0+1)} \exp\!\left(-\frac{\beta_0}{\sigma^2}\right).$$
>
> **Pour $f(\mu \mid \sigma^2, \boldsymbol{x})$ :** on garde uniquement les facteurs en $\mu$ ($\sigma^2$ traité comme constante absorbée dans la normalisation). Reste :
>
> $$f(\mu \mid \sigma^2, \boldsymbol{x}) \propto \exp\!\left(-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2\right) \exp\!\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right).$$
>
> Complétion du carré (cf. `Inférence Bayésienne` section VII.G) → forme gaussienne.
>
> **Pour $f(\sigma^2 \mid \mu, \boldsymbol{x})$ :** on garde uniquement les facteurs en $\sigma^2$ :
>
> $$f(\sigma^2 \mid \mu, \boldsymbol{x}) \propto (\sigma^2)^{-n/2 - \nu_0 - 1} \exp\!\left(-\frac{1}{\sigma^2}\left[\beta_0 + \frac{1}{2}\sum(x_i - \mu)^2\right]\right).$$
>
> C'est exactement le noyau d'une IG avec paramètres mis à jour.

**Le sampler Gibbs alterne** entre tirer $\mu$ (sachant $\sigma^2$ courant) puis $\sigma^2$ (sachant le nouveau $\mu$). Très simple à implémenter, et chaque échantillon est accepté.

### E. Quand Gibbs marche bien (et quand il rame)

**Gibbs brille quand** les conditionnelles complètes sont des lois standard. C'est typiquement le cas dans les **modèles hiérarchiques avec priors conjugués par bloc** (cf. `Modèles Bayésiens`).

**Gibbs rame quand** les coordonnées sont fortement corrélées a posteriori. Imagine un posterior 2D en forme d'ellipse très allongée selon la diagonale : Gibbs ne peut bouger que selon les axes, donc avance par minuscules pas en zigzag. Solution : reparamétrer pour décorréler, ou passer à HMC.

![[mcmc_gibbs_trajectory.png|355]]
**Figure 5.** Trajectoire d'un Gibbs sampler sur un posterior 2D gaussien fortement corrélé (corrélation $\rho = 0.95$). Les contours montrent les iso-densités du posterior cible, la ligne brisée montre les premières itérations du sampler. Gibbs est contraint de bouger uniquement selon les axes (mises à jour coordonnée par coordonnée), ce qui produit ce z-pattern caractéristique. Quand les variables sont fortement corrélées, ce comportement axe-aligné explore la distribution très lentement le long de l'axe principal de l'ellipse.

## V. Diagnostics de convergence

MCMC garantit la convergence vers le posterior **à l'infini**. En pratique on s'arrête après $m$ itérations finies. Comment savoir si $m$ est assez grand ?

### A. Trace plot

Graphique de $\theta_t$ contre $t$. C'est le diagnostic visuel de base.

**Bonne chaîne :** oscille rapidement autour d'une valeur stable, sans tendance ni dérive. Ressemble à du bruit blanc.

**Mauvaise chaîne :** dérive lente, plateaux, sauts brusques, ou oscillation très lente. Signes que la chaîne n'a pas convergé ou que le sampler est mal réglé.

![[mcmc_trace_good_vs_bad.png]]
**Figure 3.** En haut : trace plot d'une chaîne bien mélangée — oscillation rapide autour de la moyenne, pas de tendance, ressemble à du bruit blanc. En bas : trace plot d'une chaîne problématique — dérive lente et autocorrélation visible, la chaîne n'a pas exploré le posterior assez longtemps ou la step size est mal réglée.

> [!warning] Le trace plot ne *prouve* pas la convergence
> Une chaîne peut avoir l'air convergente sur un trace plot et être en réalité coincée dans un mode local. Le trace plot est nécessaire mais pas suffisant. **Toujours combiner avec d'autres diagnostics** (autocorrélation, ESS, multi-chaînes).

### B. Autocorrélation et thinning

Les échantillons MCMC sont **corrélés** dans le temps : $\theta_t$ et $\theta_{t+1}$ ne sont pas indépendants. La fonction d'autocorrélation à lag $\ell$ :

$$\rho(\ell) = \frac{\mathrm{Cov}(\theta_t, \theta_{t+\ell})}{\mathrm{Var}(\theta_t)}.$$
**Lecture du graphe d'autocorrélation :**
- Décroissance rapide vers 0 (en quelques lags) = bonne chaîne.
- Décroissance lente (centaines de lags avant d'atteindre 0) = forte corrélation, exploration lente.

![[mcmc_autocorrelation.png]]
**Figure 4.** À gauche : autocorrélation d'une bonne chaîne — chute rapide vers 0 dès les premiers lags, les échantillons deviennent vite indépendants. À droite : autocorrélation d'une chaîne avec step size trop petit — corrélation persistante même à 30+ lags, signe que la chaîne avance trop lentement et que l'ESS sera très faible.

**Thinning.** Garder seulement $\theta_t$ tous les $T$ pas (par ex. $T = 50$). Ça réduit la taille du fichier de sortie et casse la corrélation, mais **ne réduit pas l'erreur Monte Carlo** (on jette de l'info utile). Aujourd'hui considéré comme inutile sauf contraintes mémoire — mieux vaut garder tous les échantillons et utiliser l'ESS comme mesure d'efficacité.

### C. Effective Sample Size (ESS)

Vu l'autocorrélation, $m$ échantillons MCMC contiennent moins d'information que $m$ échantillons i.i.d. L'**Effective Sample Size** quantifie cette perte :

$$\mathrm{ESS} = \frac{m}{1 + 2 \sum_{\ell=1}^{\infty} \rho(\ell)}.$$
C'est le "nombre équivalent d'échantillons i.i.d." — ce dont on aurait besoin pour avoir la même précision Monte Carlo qu'avec un sampler parfait.

**Exemple typique :** une chaîne de 100 000 itérations avec ESS = 373 signifie que pour la précision, c'est comme si on avait tiré 373 échantillons indépendants. Le reste est "gaspillé" à cause de la corrélation.

> 💡 **Règles de pouce sur l'ESS.**
> - **ESS > 100–1000** : assez pour estimer la moyenne posterior.
> - **ESS > 10 000** : assez pour estimer des quantiles extrêmes (95%, 99%).
> - **ESS / $m$ < 0.01** : signal d'alarme, le sampler est très inefficace.

### D. Burn-in

Les premières itérations sont biaisées par le point de départ $\theta_0$, qui n'est pas tiré du posterior. On **jette** ces itérations — c'est le **burn-in**.

Pas de règle universelle pour la longueur du burn-in : on regarde le trace plot et on coupe avant le moment où la chaîne se stabilise. Typiquement 10–20% des itérations totales.

**Bonne pratique :** même quand la chaîne a l'air stable dès le début, garder un burn-in par sécurité.

### E. Multi-chaînes et Gelman-Rubin ($\hat{R}$)

Lancer **plusieurs chaînes** en parallèle, depuis des points de départ différents. Si toutes convergent vers la même distribution, c'est un bon signe.

**Statistique de Gelman-Rubin.** Pour chaque paramètre, on compare la variance *intra-chaîne* (W) et la variance *inter-chaînes* (B) :

$$\hat{R} = \sqrt{\frac{\frac{m-1}{m} W + \frac{1}{m} B}{W}}.$$
Si les chaînes ont convergé vers la même distribution, $\hat{R} \approx 1$. Si elles divergent, $\hat{R} > 1$.

> 💡 **Règle :** $\hat{R} < 1.01$ pour chaque paramètre = convergence acceptable. $\hat{R} > 1.05$ = problème, faire tourner plus longtemps.

### F. Résumé — le checklist

> [!summary] Avant d'utiliser les échantillons MCMC pour l'inférence
> 1. **Trace plot** sur chaque paramètre — pas de tendance, oscillation rapide.
> 2. **Autocorrélation** — décroissance vers 0 dans un nombre raisonnable de lags.
> 3. **ESS** — au moins 1000 par paramètre pour des moyennes, 10 000+ pour des quantiles.
> 4. **$\hat{R}$** sur multi-chaînes — < 1.01.
> 5. **Burn-in** appliqué et documenté.
>
> Si un seul de ces diagnostics échoue : faire tourner plus longtemps, ajuster le sampler, ou changer de paramétrisation.

## VI. MCMC moderne — au-delà de Metropolis-Hastings

Metropolis-Hastings et Gibbs sont les algorithmes "historiques" et restent omniprésents. Mais en haute dimension, ils sont battus par des méthodes plus sophistiquées. Tour d'horizon rapide.

### A. Hamiltonian Monte Carlo (HMC)

L'idée est inspirée de la mécanique : on considère $-\log f(\theta \mid y)$ comme une **énergie potentielle**, on introduit une variable auxiliaire $p$ (le "moment"), et on simule des trajectoires hamiltoniennes :

$$H(\theta, p) = -\log f(\theta \mid y) + \frac{1}{2} p^\top M^{-1} p.$$
Les équations de Hamilton font évoluer $(\theta, p)$ de façon à conserver $H$. Après un certain temps d'intégration (par leapfrog), on propose le $\theta$ obtenu et on accepte avec un critère Metropolis classique sur $H$.

**Pourquoi c'est mieux.** Les trajectoires hamiltoniennes parcourent de **grandes distances** dans l'espace des $\theta$ tout en restant dans la région de forte densité (parce que $H$ est conservé). Pas de "random walk" : on se déplace de manière déterministe et cohérente avec la géométrie du posterior.

**Coût.** Chaque pas leapfrog nécessite un calcul du gradient $\nabla \log f(\theta \mid y)$. C'est plus cher par itération que MH, mais le gain en ESS compense largement (souvent par plusieurs ordres de grandeur).

### B. NUTS — No-U-Turn Sampler

HMC a un hyperparamètre pénible : le nombre de pas leapfrog par itération. Trop peu → on n'explore pas. Trop → la trajectoire fait demi-tour ("U-turn") et revient sur ses pas, on gaspille du calcul.

**NUTS** (Hoffman & Gelman 2014) ajuste automatiquement la longueur de trajectoire en détectant le moment où la trajectoire commence à "reculer". C'est l'algorithme par défaut de **Stan**, **PyMC**, **NumPyro**, **Turing.jl**.

### C. Langevin Monte Carlo (MALA)

Variante plus simple que HMC : on prend un seul pas dans la direction du gradient, plus un bruit gaussien :

$$\theta^* = \theta_{i-1} + \frac{\epsilon^2}{2} \nabla \log f(\theta_{i-1} \mid y) + \epsilon Z, \quad Z \sim \mathcal{N}(0, I).$$
Puis correction Metropolis-Hastings. C'est essentiellement une **descente de gradient bruitée**, ce qui le rend très utilisé en optimisation bayésienne et dans les liens MCMC ↔ SGD ("SGLD" — Stochastic Gradient Langevin Dynamics, Welling & Teh 2011).

### D. En pratique — quel outil utiliser

Aujourd'hui, en bayésien moderne, on écrit le modèle dans un langage probabiliste (Stan, PyMC, NumPyro) et le solveur fait tourner NUTS automatiquement. On ne code plus Metropolis-Hastings à la main sauf pour des cas très spécifiques ou à des fins pédagogiques.

Gibbs reste utile dans des contextes où les conditionnelles complètes sont analytiques (modèles hiérarchiques simples), notamment via **JAGS** ou **BUGS** — mais Stan/PyMC le surpassent dans la majorité des cas.

## VII. À retenir

> [!summary] Les idées centrales
> 1. **MCMC = construire une chaîne de Markov dont la stationnaire est le posterior**, pour échantillonner d'un posterior intractable. Marche même quand on ne connaît la cible qu'à une constante près.
> 2. **Metropolis-Hastings** : propose un candidat depuis $q$, accepte avec proba $\min(1, \alpha)$ où $\alpha$ implique un ratio des densités cibles et un ratio correctif des propositions. Random walk Metropolis = cas symétrique, le plus courant.
> 3. **Gibbs sampling** : échantillonne une coordonnée à la fois depuis sa conditionnelle complète. Pas de rejet (cas particulier de MH avec $\alpha = 1$). Marche bien quand les conditionnelles sont des lois standard.
> 4. **Tuning critique** : random walk Metropolis vise un taux d'acceptation 23–50%. Trop haut = step trop petit, trop bas = step trop grand.
> 5. **Diagnostics obligatoires** : trace plot, autocorrélation, ESS, $\hat{R}$ multi-chaînes, burn-in. Aucun n'est suffisant seul.
> 6. **MCMC moderne** : HMC (gradient-based, suit la géométrie du posterior), NUTS (auto-tuning de HMC), Langevin (descent de gradient bruité). Standard aujourd'hui via Stan / PyMC / NumPyro.
