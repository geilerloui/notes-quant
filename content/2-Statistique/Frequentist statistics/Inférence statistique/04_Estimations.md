---
title: Estimations
order: 4
---

# Estimations — Maximum de Vraisemblance (MLE)

Cette note traite du **principe du maximum de vraisemblance** (MLE, *Maximum Likelihood Estimation*), méthode d'estimation ponctuelle de loin la plus utilisée en statistique. On la présente dans son cadre le plus simple — l'estimation des paramètres d'une gaussienne à partir d'un échantillon iid — pour fixer les idées géométriquement avant tout formalisme avancé. Le MLE est aussi la **fondation** sur laquelle s'appuient EM (`[[05_Expectation_Maximization]]`) et toutes les méthodes d'apprentissage probabiliste modernes (régression logistique, modèles graphiques, deep learning génératif).

## I. Le setup

On a un échantillon $\{x_1, \ldots, x_n\}$ de $n$ observations, qu'on suppose **iid** (indépendantes et identiquement distribuées) selon une distribution paramétrée par $\theta \in \Theta$. Par exemple, une gaussienne avec $\theta = (\mu, \sigma^2)$ — on dit que $X_i \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$.

**Le problème** : on connaît la *famille* de distributions (gaussienne ici), mais pas les *paramètres* qui ont effectivement généré les données. Le MLE va dire :

> *"Choisissons les paramètres qui rendent les données observées les plus plausibles."*

C'est l'intuition la plus naturelle qu'on puisse avoir. On va la formaliser pas à pas.

## II. Étape 1 — fixer $(\mu, \sigma)$ et mesurer les densités

Imaginons qu'on *devine* une paire $(\mu, \sigma)$. On peut alors dessiner la gaussienne $\mathcal{N}(\mu, \sigma^2)$. Pour chaque observation $x_i$ du dataset, on remonte verticalement depuis l'axe jusqu'à la courbe : ça donne la **densité** $p(x_i \mid \mu, \sigma)$ — la hauteur de la courbe au-dessus du point.

Cette hauteur dit "à quel point ce point $x_i$ était plausible sous $(\mu, \sigma)$" :

- Si la gaussienne est bien placée (les points sont dans la bosse), les hauteurs sont grandes.
- Si elle est mal placée (les points sont dans les queues), les hauteurs sont microscopiques.

C'est l'opération **fondamentale** du MLE : on évalue la densité du modèle aux points observés.

![[mle_step1_densities.png]]
*Figure. Étape 1 du MLE en image. On fixe une paire $(\mu, \sigma)$ (ici la MLE finale, $\mu \approx 5.18$, $\sigma \approx 1.30$) et on trace la gaussienne associée. Les points rouges sur l'axe sont les $n = 8$ observations du dataset. Les flèches grises remontent depuis chaque $x_i$ jusqu'à la courbe — leur hauteur est la **densité** $p(x_i \mid \mu, \sigma)$. Cette mesure quantifie à quel point chaque point est plausible sous la gaussienne choisie : les points au cœur de la bosse ont des hauteurs grandes ($\sim 0.27$), ceux dans les queues ont des hauteurs petites ($\sim 0.05$).*

## III. Étape 2 — la vraisemblance

On combine toutes les hauteurs en un seul nombre : la **vraisemblance** (likelihood) des paramètres sachant les données :

$$L(\mu, \sigma \mid x_1, \ldots, x_n) \;=\; \prod_{i=1}^{n} p(x_i \mid \mu, \sigma).$$

Pour la gaussienne :

$$L(\mu, \sigma \mid x_1, \ldots, x_n) \;=\; \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right).$$

> [!warning] Vraisemblance ≠ densité
> Numériquement, $L(\mu, \sigma \mid x_1, \ldots, x_n)$ est égal à $\prod_i p(x_i \mid \mu, \sigma)$. Mais le **rôle des variables est inversé** :
> 
> - **Densité** $p(x \mid \mu, \sigma)$ : on fixe les paramètres $(\mu, \sigma)$ et on regarde la fonction de $x$. Elle intègre à $1$ sur l'espace des $x$.
> - **Vraisemblance** $L(\mu, \sigma \mid \text{data})$ : on fixe les données observées et on regarde la fonction de $(\mu, \sigma)$. **Elle n'intègre pas à $1$** sur l'espace des paramètres — ce n'est pas une distribution sur $(\mu, \sigma)$.
> 
> C'est une distinction conceptuelle essentielle. La vraisemblance est une *fonction des paramètres*, qu'on va chercher à **maximiser**.

> [!note] Pourquoi le produit ?
> L'hypothèse iid signifie que la densité jointe de l'échantillon se factorise :
> 
> $$p(x_1, \ldots, x_n \mid \mu, \sigma) = \prod_{i=1}^{n} p(x_i \mid \mu, \sigma).$$
> 
> Donc la vraisemblance est la probabilité jointe d'observer exactement *cet* échantillon-là, vue comme fonction des paramètres. Sans l'hypothèse iid (par exemple si les $x_i$ sont corrélés), il faudrait écrire la jointe complète.

## IV. Étape 3 — maximiser

On fait *varier* $\mu$ (ou $\sigma$) et on regarde comment la vraisemblance change.

- Tracer $L(\mu, \sigma_{\text{fixé}})$ comme fonction de $\mu$ seul donne une **courbe avec un sommet** : la valeur de $\mu$ qui rend les données les plus plausibles.
- Idem pour $\sigma$ avec $\mu$ fixé.

L'estimateur du maximum de vraisemblance est défini par :

$$(\hat\mu_{\text{MLE}}, \hat\sigma_{\text{MLE}}) \;=\; \arg\max_{\mu, \sigma} \; L(\mu, \sigma \mid x_1, \ldots, x_n).$$

Géométriquement, on cherche **la gaussienne qui maximise les hauteurs au-dessus des points observés**. C'est la gaussienne qui "épouse" le mieux le nuage de données dans la famille permise.

![[mle_likelihood.png]]
*Figure. La vraisemblance en fonction de chacun des deux paramètres. **À gauche** : $L(\mu, \sigma_{\text{fixé}})$ en fonction de $\mu$ seul ($\sigma$ étant fixé à sa valeur MLE) — la courbe a un sommet à $\hat\mu_{\text{MLE}}$, qui correspond à la moyenne empirique des données. **À droite** : pareil mais en fonction de $\sigma$ avec $\mu$ fixé — sommet à $\hat\sigma_{\text{MLE}}$. La cible du MLE est de trouver ces deux sommets simultanément.*

## V. Étape 4 — passer au log

En pratique, on ne maximise jamais $L$ directement. On maximise $\log L$ pour deux raisons.

**Raison numérique.** Un produit de $n$ petits nombres sous-déborde rapidement : pour $n = 1000$ et des densités moyennes $\sim 0.1$, on a $L \sim 10^{-1000}$, en-dessous de la plus petite valeur représentable en double précision. Avec $\log L$, on travaille avec des sommes de $\log$, qui restent dans une plage numérique manipulable.

**Raison mathématique.** $\log$ est strictement croissante, donc :

$$\arg\max_\theta L(\theta) \;=\; \arg\max_\theta \log L(\theta).$$

Le sommet est au **même endroit**, mais $\log$ transforme le produit en somme :

$$\log L(\mu, \sigma \mid x_1, \ldots, x_n) \;=\; \sum_{i=1}^{n} \log p(x_i \mid \mu, \sigma),$$

ce qui est beaucoup plus facile à dériver.

![[mle_loglikelihood.png]]
*Figure. La log-vraisemblance en fonction de chacun des deux paramètres — à comparer avec la figure précédente. **À gauche** : $\log L(\mu, \sigma_{\text{fixé}})$ en fonction de $\mu$. **À droite** : en fonction de $\sigma$. La forme de la courbe est différente (somme au lieu de produit), mais le **sommet est exactement au même endroit** que dans la vraisemblance directe : $\hat\mu_{\text{MLE}}$ et $\hat\sigma_{\text{MLE}}$ inchangés. C'est précisément ce qui permet de travailler avec $\log L$ en pratique sans changer la solution.*

## VI. Calcul explicite pour la gaussienne

Tirons les estimateurs MLE concrets. Pour la gaussienne :

$$\log L(\mu, \sigma \mid x_1, \ldots, x_n) \;=\; \sum_{i=1}^{n} \log\!\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)\right].$$

En développant le $\log$ :

$$\log L = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2.$$

### A. Estimateur de $\mu$

On dérive par rapport à $\mu$ :

$$\frac{\partial \log L}{\partial \mu} \;=\; \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i - \mu).$$

On annule :

$$\sum_{i=1}^{n} (x_i - \mu) = 0 \quad\Longleftrightarrow\quad \mu = \frac{1}{n} \sum_{i=1}^{n} x_i.$$

> [!warning] Estimateur MLE de la moyenne
> $$\boxed{\;\hat\mu_{\text{MLE}} \;=\; \bar{x}_n \;=\; \frac{1}{n} \sum_{i=1}^{n} x_i\;}$$
> 
> C'est la **moyenne empirique** — exactement le résultat qu'on aurait deviné intuitivement.

### B. Estimateur de $\sigma^2$

On dérive par rapport à $\sigma^2$ (en posant $v = \sigma^2$ pour clarifier) :

$$\frac{\partial \log L}{\partial v} \;=\; -\frac{n}{2v} + \frac{1}{2v^2} \sum_{i=1}^{n} (x_i - \mu)^2.$$

On annule (et on multiplie par $2v^2$) :

$$-nv + \sum_{i=1}^{n} (x_i - \mu)^2 = 0 \quad\Longleftrightarrow\quad v = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2.$$

En remplaçant $\mu$ par son estimateur $\hat\mu_{\text{MLE}}$ :

> [!warning] Estimateur MLE de la variance
> $$\boxed{\;\hat\sigma^2_{\text{MLE}} \;=\; \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x}_n)^2\;}$$
> 
> C'est la **variance empirique**, avec un dénominateur $n$ (pas $n-1$).

### C. Pourquoi $n$ et pas $n-1$ ?

L'estimateur MLE divise par $n$, ce qui le rend **biaisé** : $\mathbb{E}[\hat\sigma^2_{\text{MLE}}] = \frac{n-1}{n} \sigma^2 \neq \sigma^2$. Pour obtenir un estimateur **non biaisé**, on divise par $n-1$ :

$$\hat\sigma^2_{\text{non biaisé}} \;=\; \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x}_n)^2.$$

C'est ce qu'on appelle parfois l'estimateur de **Bessel**. Quand $n$ est grand, la différence est négligeable. Quand $n$ est petit, elle compte — d'où le choix entre les deux selon le contexte.

> [!note] MLE biaisé, ce n'est pas forcément un défaut
> Le biais n'est qu'une propriété parmi d'autres. Le MLE a plein de **bonnes propriétés asymptotiques** (consistance, normalité asymptotique, efficacité au sens de Cramér-Rao) qui en font l'estimateur de choix en pratique, même biaisé à $n$ fini.

## VII. Propriétés générales du MLE

Sans entrer dans les preuves, on liste les propriétés clés que tout statisticien doit avoir en tête :

- **Consistance** : $\hat\theta_{\text{MLE}} \xrightarrow{p} \theta^*$ quand $n \to \infty$ (sous des conditions de régularité). L'estimateur converge vers la vraie valeur.
- **Normalité asymptotique** : $\sqrt{n}(\hat\theta_{\text{MLE}} - \theta^*) \xrightarrow{d} \mathcal{N}(0, I(\theta^*)^{-1})$, où $I(\theta)$ est l'**information de Fisher**. C'est ce qui permet de construire des intervalles de confiance autour de $\hat\theta_{\text{MLE}}$.
- **Efficacité** : asymptotiquement, le MLE atteint la borne de Cramér-Rao — aucun autre estimateur ne peut avoir une variance plus petite à grand $n$.
- **Invariance par reparamétrisation** : si $\hat\theta_{\text{MLE}}$ estime $\theta$, alors $g(\hat\theta_{\text{MLE}})$ estime $g(\theta)$ pour toute fonction $g$. Pratique : pour estimer $\sigma$ plutôt que $\sigma^2$, il suffit de prendre la racine de l'estimateur.

> [!todo] À détailler dans une future note
> Information de Fisher, borne de Cramér-Rao, tests de rapport de vraisemblance (likelihood ratio test). Ces sujets méritent leur propre note.

## VIII. Au-delà de la gaussienne

Le principe MLE s'applique à n'importe quelle famille paramétrique. Quelques exemples canoniques :

- **Bernoulli** $p$ : $\hat p_{\text{MLE}} = \bar{x}_n$ (proportion empirique de succès).
- **Poisson** $\lambda$ : $\hat\lambda_{\text{MLE}} = \bar{x}_n$ (moyenne empirique).
- **Exponentielle** $\lambda$ : $\hat\lambda_{\text{MLE}} = 1/\bar{x}_n$.
- **Uniforme** $[0, \theta]$ : $\hat\theta_{\text{MLE}} = \max_i x_i$ (cas instructif où le MLE n'est pas dérivable en $\theta$).

Pour des modèles plus complexes (régression linéaire, régression logistique, deep learning), le principe est identique : on écrit la log-vraisemblance et on la maximise, généralement par descente de gradient.

## IX. Le pont vers EM

Le MLE classique s'applique tant qu'on peut écrire $p_\theta(x)$ explicitement et calculer sa log-vraisemblance. Mais dans beaucoup de modèles modernes (mixtures, modèles à variable latente), la quantité naturelle est $p_\theta(x, z)$ avec $z$ caché, et la marginale s'écrit comme une somme :

$$p_\theta(x) \;=\; \sum_z p_\theta(x, z).$$

La log-vraisemblance devient alors :

$$\log p_\theta(x) \;=\; \log \sum_z p_\theta(x, z),$$

et **le $\log$ d'une somme n'est plus dérivable directement** par rapport à $\theta$ — pas de forme fermée en général.

L'algorithme **EM** (cf. `[[05_Expectation_Maximization]]`) est précisément l'astuce qui permet de faire du MLE quand le modèle a des variables latentes, en alternant deux étapes :
- **E-step** : deviner la distribution probable de $z$ sachant les observations.
- **M-step** : faire un MLE classique sur la "log-vraisemblance des données complètes", pondérée par le résultat du E-step.

EM est donc une **généralisation du MLE** à la situation des modèles latents. Si tu maîtrises cette note, EM ne sera qu'une astuce calculatoire de plus.

---

## Pour aller plus loin

- **Casella, Berger, *Statistical Inference*, chapitre 7.** Le traitement de référence du MLE et de ses propriétés.
- **van der Vaart, *Asymptotic Statistics*, chapitre 5.** Pour les preuves rigoureuses de consistance, normalité asymptotique, efficacité.
- **StatQuest sur YouTube — "Maximum Likelihood, clearly explained".** Excellente intuition visuelle pour les estimateurs gaussiens, dans la lignée de la présentation §II–V.
- **Lehmann & Casella, *Theory of Point Estimation*.** Référence avancée pour l'estimation ponctuelle.
