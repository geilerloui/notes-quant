---
title: Variational Inference
---
# Inférence Variationnelle

> Cette note couvre l'**inférence variationnelle** (VI) : une famille d'algorithmes d'inférence approchée **déterministes** qui transforment un problème de calcul de postérieure en un problème d'**optimisation**. L'idée centrale : on choisit une approximation $q(z)$ dans une famille tractable, et on essaie de la rendre aussi proche que possible de la vraie postérieure $p(z \mid x)$. En relâchant les contraintes ou en approximant l'objectif, on troque de la précision contre de la vitesse. Bilan : VI donne souvent les bénéfices de vitesse de l'estimation MAP avec les bénéfices statistiques de l'approche bayésienne.

---

## I. Cadre général

### A. Pourquoi l'inférence variationnelle ?

Soit le modèle graphique suivant :

![[images/3-Apprentissage automatique/09_PGM/Représentation/im15.png]]

À partir des dépendances du graphe, on peut écrire la loi jointe :

$$p(X_{1:5}) = p(X_5 \mid X_3) \, p(X_4 \mid X_2, X_3) \, p(X_3 \mid X_1) \, p(X_2 \mid X_1) \, p(X_1).$$

En revanche, si on cherche une **distribution conditionnelle**, on tombe sur :

$$p(X_3, X_4 \mid X_1, X_2, X_5) \;=\; \frac{p(X_1, X_2, X_3, X_4, X_5)}{\displaystyle\int_{X_3} \int_{X_4} p(X_1, X_2, X_3, X_4, X_5) \, dX_3 \, dX_4}.$$

C'est beaucoup plus compliqué : on connaît la jointe, mais il faut aussi calculer une intégrale qui est très souvent **intractable**.

> 💡 **L'idée de l'inférence variationnelle.** On contourne le calcul de l'intégrale en l'approchant. C'est l'analogue déterministe de **MCMC** (qui est exact mais très coûteux en temps de calcul). VI est plus rapide mais ne donne qu'une approximation.

### B. Idée centrale

Comme on ne connaît pas $p(z \mid x)$, on va l'approcher par $q(z)$. Le point de départ est la décomposition fondamentale :

$$\boxed{\text{KL} + \mathcal{L} \;=\; \log p(x)}$$

où $\text{KL}$ et $\mathcal{L}$ peuvent varier en fonction du choix de $q(z)$, mais $\log p(x)$ est **fixe**. Pourquoi fixe ? Parce qu'on calcule $p(z \mid x)$ avec $x$ comme variable connue (l'évidence).

> [!note]- Dérivation
> On part de la KL divergence entre $q(z)$ et la postérieure $p(z \mid x)$ :
>
> $$\text{KL}\big(q(z) \,\|\, p(z \mid x)\big) = -\sum_z q(z) \log \frac{p(z \mid x)}{q(z)}.$$
>
> En utilisant $p(z \mid x) = p(x, z) / p(x)$ :
>
> $$\begin{aligned}
> \text{KL}\big(q(z) \,\|\, p(z \mid x)\big) &= -\sum_z q(z) \log \frac{p(x, z)}{p(x)} \cdot \frac{1}{q(z)} \\
> &= -\sum_z q(z) \left[ \log \frac{p(x, z)}{q(z)} + \log \frac{1}{p(x)} \right] \\
> &= -\sum_z q(z) \left[ \log \frac{p(x, z)}{q(z)} - \log p(x) \right] \\
> &= -\sum_z q(z) \log \frac{p(x, z)}{q(z)} + \log p(x) \underbrace{\sum_z q(z)}_{=1}.
> \end{aligned}$$
>
> En réarrangeant :
>
> $$\text{KL}\big(q(z) \,\|\, p(z \mid x)\big) + \underbrace{\sum_z q(z) \log \frac{p(x, z)}{q(z)}}_{\text{ELBO}} = \log p(x),$$
>
> où $p(x, z)$ est connue. D'où la décomposition :
>
> $$\text{KL} + \text{ELBO} = \log p(x).$$

> [!warning] Décomposition fondamentale
> $$\text{KL}\big(q(z) \,\|\, p(z \mid x)\big) + \text{ELBO}(q) \;=\; \log p(x).$$
>
> Avec :
> - $\text{KL} \geq 0$ (par définition).
> - $\text{ELBO}(q) = \sum_z q(z) \log \dfrac{p(x, z)}{q(z)}$ : **Evidence Lower BOund**.
> - $\log p(x)$ : terme fixe.
>
> Comme $\log p(x)$ est constant et $\text{KL} + \text{ELBO} = \log p(x)$, **maximiser l'ELBO revient exactement à minimiser la KL**.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im1 (4).png]]
**Figure 1.** En augmentant la borne inférieure $\mathcal{L}$ (ELBO), on réduit la KL divergence. On n'a même pas besoin de connaître $p(z \mid x)$ — l'approximation s'améliore quoi qu'il arrive.

| KL | $\mathcal{L}$ | = | $\log p(x)$ |
| :---: | :---: | :---: | :---: |
| 4 | −8 | = | −4 |
| 3 | −7 | = | −4 |
| 2 | −6 | = | −4 |

> 💡 **Le point clé.** Quand on approche la distribution conditionnelle $p(z \mid x)$ par $q(z)$, **au lieu** de minimiser la KL divergence entre les deux (qui requiert de connaître $p(z \mid x)$), on **maximise l'ELBO** — ce qui est mathématiquement équivalent et beaucoup plus simple à manipuler.

---

## II. Mean field — la méthode la plus populaire

### A. L'approximation factorisée

Une des formes les plus populaires d'inférence variationnelle est l'**approximation mean field** (Opper & Saad, 2001). On suppose que la postérieure se factorise complètement :

$$\boxed{q(\mathbf{x}) \;=\; \prod_i q_i(\mathbf{x}_i)}$$

> [!warning] Variantes selon ce qu'on infère
> - **Variational Inference standard** : on infère les variables latentes $z_i$ en supposant que les paramètres $\theta$ du modèle sont connus.
> - **Variational Bayes (VB)** : on infère les paramètres eux-mêmes, avec une approximation factorisée.
> - **Variational Bayes EM** : on infère **à la fois** les latentes et les paramètres, avec
>
> $$p(\boldsymbol{\theta}, \mathbf{z}_{1:N} \mid \mathcal{D}) \;\approx\; q(\boldsymbol{\theta}) \prod_i q_i(\mathbf{z}_i).$$

### B. Dérivation

**Setting.** On a deux ensembles de variables :

$$x = \{x_1, x_2, x_3\}, \qquad z = \{z_1, z_2, z_3\}.$$

On veut estimer la postérieure $p(z \mid x)$ en connaissant la jointe $p(z, x)$. L'astuce : utiliser $q(z)$ comme estimateur, et maximiser l'ELBO

$$\text{ELBO} = \sum_z q(z) \log \frac{p(x, z)}{q(z)}.$$

**Idée mean field.** Au lieu de laisser $q(z_1, z_2, z_3)$ être quelconque, on impose une hypothèse d'**indépendance** :

$$q(z_1, z_2, z_3) = q(z_1) \, q(z_2) \, q(z_3) = \prod_{i=1}^{3} q(z_i).$$

En injectant dans l'ELBO :

$$\begin{aligned}
\text{ELBO} &= \sum_{z_1} \sum_{z_2} \sum_{z_3} q(z_1) q(z_2) q(z_3) \log \frac{p(x, z)}{q(z_1) q(z_2) q(z_3)} \\
&= \sum_{z_1} \sum_{z_2} \sum_{z_3} q(z_1) q(z_2) q(z_3) \big[ \log p(x, z) - \log q(z_1) - \log q(z_2) - \log q(z_3) \big].
\end{aligned}$$

> [!note]- Découpage en trois parties (avec $q(z_1)$ inconnue, $q(z_2), q(z_3)$ connues)
> **(1) Premier terme.** Avec la définition de l'espérance $\mathbb{E}[f(x)] = \sum f(x) p(x)$ :
>
> $$\sum_{z_1} \sum_{z_2} \sum_{z_3} q(z_1) q(z_2) q(z_3) \log p(x, z) = \sum_{z_1} q(z_1) \, \mathbb{E}_{z_2, z_3}\big[ \log p(x, z) \big].$$
>
> **(2) Deuxième terme.**
>
> $$\sum_{z_1} \sum_{z_2} \sum_{z_3} q(z_1) q(z_2) q(z_3) \log q(z_1) = \sum_{z_1} q(z_1) \log q(z_1) \underbrace{\sum_{z_2} \sum_{z_3} q(z_2) q(z_3)}_{=1} = \sum_{z_1} q(z_1) \log q(z_1).$$
>
> **(3) Troisième terme.** Comme $q(z_2), q(z_3)$ sont connues, le terme entre crochets est une constante $K$ :
>
> $$\sum_{z_1} \sum_{z_2} \sum_{z_3} q(z_1) q(z_2) q(z_3) \big[ \log q(z_2) + \log q(z_3) \big] = \sum_{z_1} q(z_1) \cdot K.$$

En recollant les trois morceaux :

$$\text{ELBO} = \sum_{z_1} q(z_1) \big[ \mathbb{E}_{z_2, z_3}[\log p(x, z)] + K_1 + K_2 \big] - \sum_{z_1} q(z_1) \log q(z_1).$$

On définit $\log f(x, z) := \mathbb{E}_{z_2, z_3}[\log p(x, z)] + K_1$, soit

$$f(x, z) = c \cdot \exp\big(\mathbb{E}_{z_2, z_3}[\log p(x, z)]\big).$$

L'ELBO se réécrit alors :

$$\text{ELBO} = \sum_{z_1} q(z_1) \log \frac{f(x, z)}{q(z_1)} + \text{const} = -\text{KL}\big(q(z_1) \,\|\, f(x, z)\big) + \text{const}.$$

> [!warning] Mise à jour mean field
> Pour minimiser cette KL, on prend $q(z_1) = f(x, z)$. D'où la **règle de mise à jour** pour chaque facteur :
>
> $$q(z_1) = c_1 \exp\!\left( \sum_{z_2} \sum_{z_3} q(z_2) q(z_3) \log p(x, z) \right)$$
> $$q(z_2) = c_2 \exp\!\left( \sum_{z_1} \sum_{z_3} q(z_1) q(z_3) \log p(x, z) \right)$$
> $$q(z_3) = c_3 \exp\!\left( \sum_{z_1} \sum_{z_2} q(z_1) q(z_2) \log p(x, z) \right)$$
>
> Le problème : on ne connaît pas $q(z_1), q(z_2), q(z_3)$ — ils sont définis l'un en fonction de l'autre. D'où l'**algorithme itératif** (CAVI).

### C. Exemple — distribution exponentielle

Soit la jointe

$$p(x, y, z) = \lambda_1 \lambda_2 \lambda_3 \, e^{-\lambda_1 x - \lambda_2 y - \lambda_3 z}.$$

On résout pour $q(x)$ :

$$\begin{aligned}
\log q(x) &= \mathbb{E}_{y, z}[\log p(x, y, z)] + K \\
&= \mathbb{E}_y\Big[\mathbb{E}_z\big[\log(\lambda_1 \lambda_2 \lambda_3) - \lambda_1 x - \lambda_2 y - \lambda_3 z\big]\Big] + K \\
&= \log(\lambda_1 \lambda_2 \lambda_3) - \lambda_1 x - \lambda_2 \mathbb{E}[y] - \lambda_3 \mathbb{E}[z] + K.
\end{aligned}$$

En exponentiant :

$$q(x) = e^{\log(\lambda_1 \lambda_2 \lambda_3) + K - \lambda_2 \mathbb{E}[y] - \lambda_3 \mathbb{E}[z]} \cdot e^{-\lambda_1 x} = c \cdot e^{-\lambda_1 x}.$$

C'est une **distribution exponentielle**, donc $c = \lambda_1$ par normalisation, ce qui donne :

$$q(x) = \lambda_1 e^{-\lambda_1 x}.$$

Par le même raisonnement :

$$q(y) = \lambda_2 e^{-\lambda_2 y}, \qquad q(z) = \lambda_3 e^{-\lambda_3 z}.$$

> 💡 **L'algorithme implicite.**
> 1. Calculer l'espérance $\mathbb{E}_{i \neq j}[\log p(X, Z)]$.
> 2. Reconnaître la forme d'une distribution connue.
> 3. Deviner les paramètres restants par normalisation.
>
> Si on n'arrive pas à éliminer $\mathbb{E}[y]$ et $\mathbb{E}[z]$, on les initialise aléatoirement. Cela donne $q(x)$, qui permet ensuite de calculer $q(y)$ puis $q(z)$, et on itère.

### D. Algorithme CAVI (David Blei)

> [!note]- Pseudo-code CAVI (Coordinate Ascent Variational Inference)
> ```
> Entrée : Distribution jointe p(X, Z)
> Sortie : Densité variationnelle q(Z) = ∏_j q_j(z_j)
>
> Initialiser les facteurs variationnels q_j(z_j)
> ELBO = E[log p(X, Z)] - E[log q(Z)]
>
> Tant que l'ELBO n'a pas convergé :
>     Pour j = 1, ..., m :
>         log q_j(z_j) ← E_{i≠j}[log p(X, Z)]
>     Calculer ELBO
> ```

---

## III. Exemple — estimation des paramètres d'une gaussienne 1D

On applique VB pour inférer la postérieure des paramètres d'une gaussienne 1D : $p(\mu, \lambda \mid \mathcal{D})$, où $\lambda = 1/\sigma^2$ est la **précision**. Pour simplifier, on utilise une **prior conjuguée** :

$$p(\mu, \lambda) = \mathcal{N}\!\left(\mu \mid \mu_0, (\kappa_0 \lambda)^{-1}\right) \cdot \mathrm{Ga}(\lambda \mid a_0, b_0).$$

Et une postérieure approchée factorisée :

$$q(\mu, \lambda) = q_\mu(\mu) \, q_\lambda(\lambda).$$

> 💡 **On ne spécifie pas la forme de $q_\mu$ et $q_\lambda$** : les formes optimales vont **émerger automatiquement** de la dérivation (et il se trouvera qu'elles sont gaussienne et gamma respectivement).

### A. Démonstration théorique

**(i) Distribution cible :**

$$\begin{aligned}
\log \tilde p(\mu, \lambda) &= \log p(\mu, \lambda, \mathcal{D}) \\
&= \log p(\mathcal{D} \mid \mu, \lambda) + \log p(\mu \mid \lambda) + \log p(\lambda) \\
&= \frac{N}{2} \log \lambda - \frac{\lambda}{2} \sum_{i=1}^N (x_i - \mu)^2 - \frac{\kappa_0 \lambda}{2}(\mu - \mu_0)^2 \\
&\quad + \frac{1}{2} \log(\kappa_0 \lambda) + (a_0 - 1) \log \lambda - b_0 \lambda + \text{const}.
\end{aligned}$$

**(ii) Mise à jour de $q_\mu(\mu)$ :**

$$\begin{aligned}
\log q_\mu(\mu) &= \mathbb{E}_{q_\lambda}[\log p(\mathcal{D} \mid \mu, \lambda) + \log p(\mu \mid \lambda)] + \text{const} \\
&= -\frac{\mathbb{E}_{q_\lambda}[\lambda]}{2} \left\{ \kappa_0 (\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2 \right\} + \text{const}.
\end{aligned}$$

En complétant le carré, on montre que $q_\mu(\mu) = \mathcal{N}(\mu \mid \mu_N, \kappa_N^{-1})$, avec :

$$\mu_N = \frac{\kappa_0 \mu_0 + N \bar x}{\kappa_0 + N}, \qquad \kappa_N = (\kappa_0 + N) \, \mathbb{E}_{q_\lambda}[\lambda].$$

À ce stade, on ne connaît pas encore $q_\lambda(\lambda)$, donc on ne peut pas calculer $\mathbb{E}[\lambda]$ — on le dérive juste après.

**(iii) Mise à jour de $q_\lambda(\lambda)$ :**

$$\begin{aligned}
\log q_\lambda(\lambda) &= \mathbb{E}_{q_\mu}[\log p(\mathcal{D} \mid \mu, \lambda) + \log p(\mu \mid \lambda) + \log p(\lambda)] + \text{const} \\
&= (a_0 - 1) \log \lambda - b_0 \lambda + \frac{1}{2} \log \lambda + \frac{N}{2} \log \lambda \\
&\quad - \frac{\lambda}{2} \mathbb{E}_{q_\mu}\!\left[ \kappa_0 (\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2 \right] + \text{const}.
\end{aligned}$$

On reconnaît le log d'une **distribution Gamma** : $q_\lambda(\lambda) = \mathrm{Ga}(\lambda \mid a_N, b_N)$, avec :

$$\begin{aligned}
a_N &= a_0 + \frac{N + 1}{2}, \\
b_N &= b_0 + \frac{1}{2} \mathbb{E}_{q_\mu}\!\left[ \kappa_0 (\mu - \mu_0)^2 + \sum_{i=1}^N (x_i - \mu)^2 \right].
\end{aligned}$$

**(iv) Calcul des espérances.** Pour implémenter les mises à jour, on a besoin des espérances. Comme $q(\mu) = \mathcal{N}(\mu \mid \mu_N, \kappa_N^{-1})$ :

$$\mathbb{E}_{q(\mu)}[\mu] = \mu_N, \qquad \mathbb{E}_{q(\mu)}[\mu^2] = \frac{1}{\kappa_N} + \mu_N^2.$$

Et comme $q(\lambda) = \mathrm{Ga}(\lambda \mid a_N, b_N)$ :

$$\mathbb{E}_{q(\lambda)}[\lambda] = \frac{a_N}{b_N}.$$

> [!warning] Équations finales — distinguer fixes et itératives
> **Pour $q(\mu)$ :**
>
> $$\textcolor{blue}{\mu_N = \frac{\kappa_0 \mu_0 + N \bar x}{\kappa_0 + N}}, \qquad \textcolor{red}{\kappa_N = (\kappa_0 + N) \frac{a_N}{b_N}}.$$
>
> **Pour $q(\lambda)$ :**
>
> $$\textcolor{blue}{a_N = a_0 + \frac{N + 1}{2}}, \qquad \textcolor{red}{b_N = b_0 + \kappa_0 \big(\mathbb{E}[\mu^2] + \mu_0^2 - 2\mathbb{E}[\mu] \mu_0\big) + \frac{1}{2} \sum_{i=1}^N \big(x_i^2 + \mathbb{E}[\mu^2] - 2\mathbb{E}[\mu] x_i\big)}.$$
>
> 💡 **$\mu_N$ et $a_N$ sont en fait des constantes fixes** (en bleu). **Seuls $\kappa_N$ et $b_N$ doivent être mis à jour itérativement** (en rouge).

**(v) Fonction de perte (ELBO).** En recollant tout, on peut montrer que :

$$\mathcal{L}(q) = \frac{1}{2} \log \frac{1}{\kappa_N} + \log \Gamma(a_N) - a_N \log b_N + \text{const}.$$

Cette quantité **augmente monotone** après chaque mise à jour VB.

### B. Mini-projet — simulation en Python

#### B.1 Initialisation et boucle principale

Seuls les termes en **rouge** ($\kappa_N$ et $b_N$) entrent dans la boucle, parce que ce sont les seuls dont les paramètres dépendent d'eux-mêmes.

```python
def make_approx_posterior(data, plot):
    # Paramètres de la prior
    # mu ~ Normal(mu0, 1/(lambda*k0))
    # lambda ~ Gamma(a0, b0)
    mu_0 = 0.0
    k_0 = 1.0
    a_0 = 1
    b_0 = 1

    # Paramètres de la postérieure (constants)
    N = len(data)
    mu_N = (k_0 * mu_0 + N * np.mean(data)) / (k_0 + N)
    a_N = a_0 + (N + 1) / 2
    k_N = 1  # Initialisation arbitraire

    num_steps = 10
    ELBO_prev = -1e9
    for step in range(num_steps):
        E_mu2 = 1 / k_N + np.power(mu_N, 2)
        b_N = b_0 + k_0 * (E_mu2 + mu_0**2 - 2 * mu_N * mu_0) \
              + 0.5 * np.sum(np.power(data, 2) + E_mu2 - 2 * mu_N * data)
        k_N = (k_0 + N) * (a_N / b_N)

        # Vérifier que l'ELBO augmente bien à chaque itération
        ELBO_current = ELBO(k_N, a_N, b_N)
        assert ELBO_current >= ELBO_prev
        ELBO_prev = ELBO_current
        print('ELBO at step %3i/%3i is %8.5f' % (step, num_steps, ELBO_current))

        q_mean_field = q_mean_field_distro(mu_N, k_N, a_N, b_N)
        plot(q_mean_field, num_points_plot=50, basename='approx_posterior', stepname=step)
```

#### B.2 Calcul de l'ELBO

Rappel de la formule :

$$\mathcal{L}(q) = \frac{1}{2} \log \frac{1}{\kappa_N} + \log \Gamma(a_N) - a_N \log b_N + \text{const}.$$

```python
from scipy.special import gamma as gamma_func

def ELBO(k, a, b):
    """Implémente l'équation 21.98 de Murphy."""
    return -0.5 * np.log(k) + np.log(gamma_func(a)) - a * np.log(b)
```

#### B.3 Postérieure approchée

La postérieure approchée combine les deux distributions $q_\mu$ et $q_\lambda$ :

```python
def q_mean_field_distro(mu_par, k_par, a_par, b_par):
    """Combine les distributions approchées de mu et lambda."""
    q_mu_instance = q_mu_distro(mu_par, k_par)
    q_lambda_instance = q_lambda_distro(a_par, b_par)

    def q_mean_field(mu, sigma2):
        # Attention : l'argument est sigma2, pas lambda !
        lam = 1 / (sigma2 + 1e-9)  # Éviter l'erreur numérique
        return q_mu_instance(mu) * q_lambda_instance(lam)
    return q_mean_field
```

Avec, pour $q(\mu) = \mathcal{N}(\mu \mid \mu_N, \kappa_N^{-1})$ :

```python
from scipy.stats import norm

def q_mu_distro(mu_par, k_par):
    """Implémente l'équation 21.71 de Murphy."""
    def q_mu(mu):
        return norm.pdf(mu, loc=mu_par, scale=np.sqrt(1 / k_par))
    return q_mu
```

Et pour $q_\lambda(\lambda) = \mathrm{Ga}(\lambda \mid a_N, b_N)$ :

```python
def q_lambda_distro(a_par, b_par):
    """Implémente l'équation 21.73 de Murphy."""
    def q_lambda(lam):
        return np.power(b_par, a_par) / gamma_func(a_par) \
               * np.power(lam, a_par - 1) * np.exp(-b_par * lam)
    return q_lambda
```

---

## IV. Pour aller plus loin

### A. Stochastic Variational Inference (SVI)

À développer. Voir [Hoffman, Blei, Wang & Paisley (2013) — *Stochastic Variational Inference*](http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf).

Bibliographie :
- [Slides résumant SVI par David Blei](https://www.research.ibm.com/haifa/Workshops/ml2014/papers/blei.pdf)
- [Exemple de code SVI pour LDA](https://github.com/qlai/stochasticLDA)
- [Quora sur SVI](https://www.quora.com/What-is-the-difference-between-full-Bayesian-inference-variational-inference-variational-Bayes-variational-EM-and-stochastic-variational-inference)

### B. EM comme cas particulier de l'inférence variationnelle

À développer. Voir [ce billet](http://stillbreeze.github.io/Variational-Inference-and-Expectation-Maximization/).

### C. Bibliographie générale

- [Code + démo (RobRomijnders)](https://github.com/RobRomijnders/vi_normal)
- [CAVI : pseudo-code et démo propre (Suzy Ahyah)](https://suzyahyah.github.io/bayesian%20inference/machine%20learning/variational%20inference/2019/03/20/CAVI.html)
- [Variational Bayes et mean field (Brian Keng)](http://bjlkeng.github.io/posts/variational-bayes-and-the-mean-field-approximation/)
- [Site avec code Python (Zhiya Zuo)](https://zhiyzuo.github.io/VI/)
