---
title: Intervalles de confiance
order: 2
---

# Intervalles de confiance

## I. Définition

> [!warning] Définition — Intervalle de confiance
> Soit $(E, (\mathbb{P}_\theta)_{\theta \in \Theta})$ un modèle statistique basé sur les observations $X_1, \ldots, X_n$, et supposons $\Theta \subseteq \mathbb{R}$. Soit $\alpha \in (0, 1)$.
> 
> - **Intervalle de confiance (IC) de niveau $1 - \alpha$** pour $\theta$ : tout intervalle aléatoire $\mathcal{I}$ (dépendant de $X_1, \ldots, X_n$) dont les bornes ne dépendent pas de $\theta$ et tel que :
> 
> $$\mathbb{P}_\theta[\mathcal{I} \ni \theta] \ge 1 - \alpha, \quad \forall \theta \in \Theta$$
> 
> - **IC de niveau asymptotique $1 - \alpha$** pour $\theta$ : tout intervalle aléatoire $\mathcal{I}$ dont les bornes ne dépendent pas de $\theta$ et tel que :
> 
> $$\lim_{n \to \infty} \mathbb{P}_\theta[\mathcal{I} \ni \theta] \ge 1 - \alpha, \quad \forall \theta \in \Theta$$

Un intervalle de confiance asymptotique correspond au cas où $n \to \infty$.

$q_\alpha$ = quantile de la loi normale pour le niveau de l'intervalle de confiance. On prend typiquement $95\% = 1 - \alpha$ donc $\alpha = 0.05$.

---

## II. Intervalle de confiance pour la moyenne

On collecte des données $Y_1, \ldots, Y_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$, où $\mu$ est inconnue et $\sigma^2$ connue. Quelles valeurs de $\mu$ sont plausibles vu $Y_1, \ldots, Y_n$ ?

L'estimateur du maximum de vraisemblance est :

$$\bar{Y} = \frac{1}{n} \sum_{i=1}^n Y_i$$

Par le TCL :

$$\frac{\bar{Y} - \mu}{\sigma/\sqrt{n}} \sim \mathcal{N}(0, 1)$$

On fait un test bilatéral avec $\alpha = 5\%$, donc on a $+1.96$ et $-1.96$ des deux côtés :

$$
\begin{aligned}
&\Rightarrow P\left(-q_{\alpha/2} \le \frac{\bar{Y} - \mu}{\sigma/\sqrt{n}} \le q_{\alpha/2}\right) = 1 - \alpha \\
&\Rightarrow P\left(-\frac{q_{\alpha/2} \sigma}{\sqrt{n}} \le \bar{Y} - \mu \le \frac{q_{\alpha/2} \sigma}{\sqrt{n}}\right) = 1 - \alpha \\
&\Rightarrow P\left(\bar{Y} - \frac{q_{\alpha/2} \sigma}{\sqrt{n}} \le \mu \le \bar{Y} + \frac{q_{\alpha/2} \sigma}{\sqrt{n}}\right) = 1 - \alpha
\end{aligned}
$$

On peut alors remplacer $\alpha$ et $q$ par des valeurs numériques.

---

## III. Intervalle de confiance pour une proportion (Bernoulli)

> [!example] Setup — couples qui s'embrassent
> Soit $p$ la proportion de couples qui tournent la tête à droite en s'embrassant. On observe $n$ couples, avec $R_i = 1$ si le couple $i$ tourne à droite, $R_i = 0$ sinon. Estimateur de $p$ :
> 
> $$\hat{p} = \bar{R}_n = \frac{1}{n} \sum_{i=1}^n R_i$$

On observe $R_1, \ldots, R_n \overset{iid}{\sim} \text{Ber}(p)$, $p \in [0, 1]$ inconnu. C'est un modèle statistique $(\{0, 1\}, (\text{Ber}(p))_{p \in (0, 1)})$. L'estimateur de $p$ est $\hat{p} = \bar{R}_n$.

Par le TCL :

$$\sqrt{n} \frac{\bar{X}_n - p}{\sqrt{p(1-p)}} \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, 1)$$

On peut calculer l'IC asymptotique :

$$
\begin{aligned}
&\lim_{n \to \infty} \mathbb{P}\left(\left|\sqrt{n}\frac{\bar{X}_n - p}{\sqrt{p(1-p)}}\right| \le q_{\alpha/2}\right) = 1 - \alpha \\
&\iff \lim_{n \to \infty} \mathbb{P}\left(\bar{X}_n - \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}} \le p \le \bar{X}_n + \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}\right) = 1 - \alpha
\end{aligned}
$$

On a l'estimateur $\bar{R}_n$. On veut que la probabilité que la distance entre l'estimateur et le vrai paramètre $p$ dépasse $x$ soit faible (probabilité $\alpha$) :

$$P(|\bar{R}_n - p| \ge x) = \alpha$$

Graphiquement, on veut que l'erreur (distance entre estimateur et vraie valeur) soit $\alpha$-petite.

![[images/3-Apprentissage automatique/Generative Models/vae/im2 (2).png]]

> [!note]- Preuve
> $$
> \begin{aligned}
> &P\left(\sqrt{n}\frac{|\bar{R}_n - p|}{\sqrt{p(1-p)}} \ge \sqrt{n}\frac{x}{\sqrt{p(1-p)}}\right) = \alpha \\
> &P\left(|Z| \ge \sqrt{n}\frac{x}{\sqrt{p(1-p)}}\right) = \alpha \\
> &2 \times P\left(Z \ge \sqrt{n}\frac{x}{\sqrt{p(1-p)}}\right) = \alpha \\
> &2 \times \left(1 - \Phi\left(\sqrt{n}\frac{x}{\sqrt{p(1-p)}}\right)\right) = \alpha \\
> &\Phi\left(\sqrt{n}\frac{x}{\sqrt{p(1-p)}}\right) = 1 - \alpha/2 \\
> &x = \frac{\sqrt{p(1-p)} \Phi^{-1}(1 - \alpha/2)}{\sqrt{n}} \\
> &x = \frac{\sqrt{p(1-p)} q_{\alpha/2}}{\sqrt{n}}
> \end{aligned}
> $$

Pour $\alpha \in (0, 1)$ fixé et $q_{\alpha/2}$ le $(1 - \alpha/2)$-quantile de $\mathcal{N}(0, 1)$, avec probabilité $\simeq 1 - \alpha$ (si $n$ est assez grand) :

$$\boxed{\bar{R}_n \in \left[p - \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}, p + \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}\right]}$$

L'IC asymptotique :

$$\lim_{n \to \infty} \mathbb{P}\left(p \in \left[\bar{R}_n - \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}, \bar{R}_n + \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}\right]\right) = 1 - \alpha$$

> ⚠️ **Ce n'est pas un vrai intervalle de confiance !** Il dépend de $p$, qui est inconnu.

### Trois solutions pour s'en sortir

#### Solution 1 — Borne conservative

Quelle que soit la valeur (inconnue) de $p$ :

$$p(1-p) \le \frac{1}{4}$$

Donc avec probabilité au moins $1 - \alpha$ :

$$\bar{R}_n \in \left[p - \frac{q_{\alpha/2}}{2\sqrt{n}}, p + \frac{q_{\alpha/2}}{2\sqrt{n}}\right]$$

D'où l'IC asymptotique :

$$\mathcal{I}_{\text{conserv}} = \left[\bar{R}_n - \frac{q_{\alpha/2}}{2\sqrt{n}}, \bar{R}_n + \frac{q_{\alpha/2}}{2\sqrt{n}}\right]$$

avec $\lim_{n \to \infty} \mathbb{P}(\mathcal{I}_{\text{conserv}} \ni p) \ge 1 - \alpha$.

#### Solution 2 — Résoudre l'équation quadratique en $p$

On a le système d'inéquations en $p$ :

$$\bar{R}_n - \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}} \le p \le \bar{R}_n + \frac{q_{\alpha/2}\sqrt{p(1-p)}}{\sqrt{n}}$$

Chacune est une inégalité quadratique en $p$ de la forme :

$$(p - \bar{R}_n)^2 \le \frac{q_{\alpha/2}^2 p(1-p)}{n}$$

On cherche les racines $p_1 < p_2$ de :

$$\left(1 + \frac{q_{\alpha/2}^2}{n}\right) p^2 - \left(2\bar{R}_n + \frac{q_{\alpha/2}^2}{n}\right) p + \bar{R}_n^2 = 0$$

D'où un nouvel IC $\mathcal{I}_{\text{solve}} = [p_1, p_2]$ tel que :

$$\lim_{n \to \infty} \mathbb{P}(\mathcal{I}_{\text{solve}} \ni p) = 1 - \alpha$$

(complexe à écrire en générique, on attend des valeurs numériques pour $n, \alpha, \bar{R}_n$).

#### Solution 3 — Plug-in

Par la LGN, $\hat{p} = \bar{R}_n \xrightarrow[n \to \infty]{(\mathbb{P})} p$.

Par Slutsky :

$$\sqrt{n}\frac{\bar{R}_n - p}{\sqrt{\hat{p}(1-\hat{p})}} \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, 1)$$

D'où le nouvel IC :

$$\mathcal{I}_{\text{plug-in}} = \left[\bar{R}_n - \frac{q_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}}, \bar{R}_n + \frac{q_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})}}{\sqrt{n}}\right]$$

avec $\lim_{n \to \infty} \mathbb{P}(\mathcal{I}_{\text{plug-in}} \ni p) = 1 - \alpha$.

### Application numérique

> [!example] Exemple — couples
> Dans l'exemple des couples : $n = 124$, $\bar{R}_n = 0.645$, $\alpha = 5\%$.
> 
> Pour $\mathcal{I}_{\text{solve}}$, racines de $1.03 p^2 - 1.32 p + 0.41 = 0$ : $p_1 = 0.53$, $p_2 = 0.75$.
> 
> Les IC de niveau asymptotique $95\%$ :
> - $\mathcal{I}_{\text{conserv}} = [0.56, 0.73]$
> - $\mathcal{I}_{\text{solve}} = [0.53, 0.75]$
> - $\mathcal{I}_{\text{plug-in}} = [0.56, 0.73]$

---

## IV. Intervalle de confiance pour une exponentielle

[Bon exemple Stanford](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture18.pdf)

> [!example] Setup — temps entre arrivées du métro
> On observe les temps entre arrivées du métro à Kendall : $T_1, \ldots, T_n$. On suppose que ces temps sont :
> - Mutuellement indépendants
> - Variables exponentielles de paramètre commun

**Modèle.** Propriétés de la variable exponentielle :

$$f(t) = \lambda e^{-\lambda t}, \forall t \ge 0 \quad \mathbb{E}[T_1] = \frac{1}{\lambda} \quad \text{Var}(T_1) = \frac{1}{\lambda^2}$$

Estimation naturelle de $1/\lambda$ :

$$\bar{T}_n := \frac{1}{n} \sum_{i=1}^n T_i$$

Estimateur naturel de $\lambda$ :

$$\hat{\lambda} := \frac{1}{\bar{T}_n} \xrightarrow[n \to \infty]{(\mathbb{P})} \lambda$$

Par la LGN, $\bar{T}_n \xrightarrow{(\mathbb{P})} 1/\lambda$, donc $\hat{\lambda} \xrightarrow{(\mathbb{P})} \lambda$.

Par le TCL :

$$\sqrt{n}\left(\bar{T}_n - \frac{1}{\lambda}\right) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, \lambda^{-2})$$

Cela donnerait un IC pour $1/\lambda$, mais on veut un IC pour $\lambda$ directement. D'où la **delta-méthode**.

### Delta-méthode

> [!warning] Définition — Delta-méthode
> Soit $(Z_n)_{n \ge 1}$ une suite de variables aléatoires telle que :
> 
> $$\sqrt{n}(Z_n - \theta) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, \sigma^2)$$
> 
> pour $\theta \in \mathbb{R}$ et $\sigma^2 > 0$ (la suite $(Z_n)_{n \ge 1}$ est dite **asymptotiquement normale autour de $\theta$**).
> 
> Soit $g : \mathbb{R} \to \mathbb{R}$ continûment différentiable au point $\theta$. Alors :
> 
> - $(g(Z_n))_{n \ge 1}$ est asymptotiquement normale autour de $g(\theta)$
> - Plus précisément :
> 
> $$\boxed{\sqrt{n}(g(Z_n) - g(\theta)) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, (g'(\theta))^2 \sigma^2)}$$

> [!note]- Preuve
> Par développement de Taylor au 1er ordre :
> 
> $$g(\theta + h) = g(\theta) + g'(\theta) h$$
> 
> On pose $h = Z_n - \theta$ :
> 
> $$
> \begin{aligned}
> g(Z_n) &= g(\theta) + (Z_n - \theta) g'(\bar{\theta}), \quad Z_n \le \bar{\theta} \le \theta \\
> g(Z_n) - g(\theta) &= (Z_n - \theta) g'(\bar{\theta}) \\
> \iff \sqrt{n}(g(Z_n) - g(\theta)) &= \sqrt{n}(Z_n - \theta) g'(\bar{\theta})
> \end{aligned}
> $$
> 
> $\bar{\theta}$ est sandwichée entre $\theta$ (paramètre optimal fixe) et $Z_n$ (qui s'améliore vers $\theta$). De plus, $\sqrt{n}(Z_n - \theta) \xrightarrow{(d)} \mathcal{N}(0, \sigma^2)$ par le TCL et $g'(\bar{\theta}) \xrightarrow{(\mathbb{P})} g'(\theta)$. Par Slutsky, le produit converge en distribution vers $g'(\theta) \cdot \mathcal{N}(0, \sigma^2) = \mathcal{N}(0, (g'(\theta))^2 \sigma^2)$.

### Application à l'exponentielle

On a $\hat{\lambda} = 1/\bar{T}_n$, $\lambda = 1/(1/\lambda)$, et $g(x) = 1/x$ donne $g'(1/\lambda) = -\lambda^2$ avec $\theta = 1/\lambda$ :

$$\sqrt{n}(\hat{\lambda} - \lambda) \xrightarrow[n \to \infty]{(d)} \mathcal{N}\left(0, \frac{1}{\lambda^2} \cdot g'\left(\frac{1}{\lambda}\right)^2\right)$$

$$\sqrt{n}(\hat{\lambda} - \lambda) \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, \lambda^2)$$

Pour $\alpha \in (0, 1)$ et $n$ assez grand :

$$|\hat{\lambda} - \lambda| \le \lambda \cdot \frac{q_{\alpha/2}}{\sqrt{n}}$$

avec probabilité $\simeq 1 - \alpha$. Donc :

$$\lambda \in \left[\hat{\lambda} \pm \frac{q_{\alpha/2} \lambda}{\sqrt{n}}\right]$$

Mais on ne peut pas calculer ça directement, on ne connaît pas $\lambda$. On le remplace par $\hat{\lambda}$ via Slutsky. Plus précisément :

$$\sqrt{n}\frac{\hat{\lambda} - \lambda}{\hat{\lambda}} \cdot \frac{\lambda}{\hat{\lambda}} \to \mathcal{N}(0, 1)$$

Le terme de gauche converge vers $\mathcal{N}(0, 1)$ par TCL, et le terme de droite converge vers 1 par LGN. Par Slutsky, le produit converge vers $\mathcal{N}(0, 1)$, et il ne reste donc que $\sqrt{n}(\hat{\lambda} - \lambda)/\hat{\lambda}$.

### Trois solutions

**1. Borne conservative.** On n'a pas de borne *a priori* sur $\lambda$, donc cette solution ne marche pas ici.

**2. Résoudre pour $\lambda$.**

$$
\begin{aligned}
|\hat{\lambda} - \lambda| \le \frac{q_{\alpha/2} \lambda}{\sqrt{n}} &\iff \lambda\left(1 - \frac{q_{\alpha/2}}{\sqrt{n}}\right) \le \hat{\lambda} \le \lambda\left(1 + \frac{q_{\alpha/2}}{\sqrt{n}}\right) \\
&\iff \frac{\hat{\lambda}}{1 + \frac{q_{\alpha/2}}{\sqrt{n}}} \le \lambda \le \frac{\hat{\lambda}}{1 - \frac{q_{\alpha/2}}{\sqrt{n}}}
\end{aligned}
$$

D'où :

$$\mathcal{I}_{\text{solve}} = \left[\hat{\lambda}\left(1 + \frac{q_{\alpha/2}}{\sqrt{n}}\right)^{-1}, \hat{\lambda}\left(1 - \frac{q_{\alpha/2}}{\sqrt{n}}\right)^{-1}\right]$$

**3. Plug-in.**

$$\mathcal{I}_{\text{plug-in}} = \left[\hat{\lambda}\left(1 - \frac{q_{\alpha/2}}{\sqrt{n}}\right), \hat{\lambda}\left(1 + \frac{q_{\alpha/2}}{\sqrt{n}}\right)\right]$$

> [!example] Exemple numérique
> $n = 64$, $\bar{T}_n = 6.23$, $\alpha = 5\%$. IC de niveau asymptotique $95\%$ :
> - $\mathcal{I}_{\text{solve}} = [0.13, 0.21]$
> - $\mathcal{I}_{\text{plug-in}} = [0.12, 0.20]$

---

## V. Signification d'un intervalle de confiance

Prenons $\mathcal{I}_{\text{plug-in}} = [0.12, 0.20]$. Que signifie *"$\mathcal{I}_{\text{plug-in}}$ est un IC de niveau asymptotique $95\%$"* ?

Est-ce que cela veut dire :

$$\lim_{n \to \infty} \mathbb{P}(\lambda \in [0.12, 0.20]) \ge 0.95 \quad ? \quad \text{NON}$$

> [!warning] Interprétation fréquentiste correcte
> Si on **répétait** cette expérience (collecter 64 observations), alors $\lambda$ serait dans l'IC résultant environ $95\%$ du temps.

![[images/1-Mathématiques/Optimal transport/im3 (2).png]]

Une autre interprétation, plus pratique : imaginons qu'on pèse un échantillon de souris femelles d'une population, on calcule leur moyenne. Puis on considère un grand nombre d'échantillons tirés par bootstrap sur lesquels on calcule leurs moyennes.

![[images/3-Apprentissage automatique/Generative Models/score based/im4 (1).png|419]]

![[images/2-Statistiques/Frequentist/Inférence statistique/im5 (1).png|431]]
