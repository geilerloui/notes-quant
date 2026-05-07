---
title: Tests d'hypothèses
order: 3
---

# Tests d'hypothèses

> Cette note couvre les tests d'hypothèses statistiques : pourquoi en faire, comment les construire, et les principaux tests paramétriques (Z-test, T-test, ANOVA, ANCOVA). Le fil rouge est le dataset `Auto` (ISLR), qu'on retrouve aussi dans la note [[Régression Linéaire Multiple]].

## 0. Pourquoi tester ?

### A. Une question naïve qui devient rigoureuse

Sur le dataset `Auto`, on observe que la mpg moyenne des voitures **japonaises** est de $30.45$ et celle des **américaines** de $20.03$. Différence observée : $+10.42$ mpg en faveur du Japon.

> [!example] Deux niveaux de question
> **Question naïve** : *"C'est marrant, les voitures japonaises consomment moins."*
> 
> **Question rigoureuse** : *"Est-ce que cette différence reflète une vraie différence dans la population (toutes les voitures jamais produites), ou est-ce juste du hasard d'échantillonnage sur les 392 voitures qu'on a observées ?"*

Le point central : **on observe un échantillon, mais on veut tirer des conclusions sur la population**. Le bruit d'échantillonnage peut créer des différences apparentes même quand il n'existe aucune différence réelle.

> 💡 **L'idée du test d'hypothèse en une phrase.** Un test est une procédure pour décider si un effet observé dans les données est *trop gros pour être du hasard* — auquel cas on conclut qu'il existe vraiment dans la population.

![[Pasted image 20260506185601.png|609]]

**Figure 1.** Distribution de la mpg pour les voitures **américaines** (rouge, $n = 245$) et **japonaises** (vert, $n = 79$). Les losanges noirs marquent les moyennes empiriques : $\bar{y}_{\text{USA}} = 20.03$ et $\bar{y}_{\text{Japon}} = 30.45$, soit un écart observé $\Delta = +10.42$ mpg. **Le point clé** : les deux nuages se recouvrent largement — il existe des voitures américaines plus économes que beaucoup de japonaises, et inversement. La vraie question n'est donc pas *"est-ce que les moyennes diffèrent ?"* (oui, manifestement), mais *"l'écart de $10.42$ mpg est-il assez grand pour exclure le hasard d'échantillonnage ?"*


### B. Le raisonnement — supposer $H_0$, regarder si les données collent

L'idée centrale du test est **à l'envers de l'intuition** : on ne cherche pas à prouver $H_1$ directement. À la place, on **suppose $H_0$ vraie par défaut**, puis on regarde si les données sont *trop incompatibles* avec ce scénario pour le maintenir. C'est exactement la présomption d'innocence en justice.

| Cadre judiciaire | Cadre statistique |
|---|---|
| Présomption d'innocence | $H_0$ : pas d'effet, pas de différence |
| Hypothèse de culpabilité | $H_1$ : effet présent |
| Preuves accumulées | Données observées |
| Seuil "preuve au-delà du doute raisonnable" | Niveau de signification $\alpha$ |
| Verdict : coupable ou non-coupable | Rejeter $H_0$ ou ne pas rejeter $H_0$ |

**Sur Auto** : on suppose par défaut qu'il n'y a *aucune différence* dans la population entre les voitures japonaises et américaines. En notant $\mu_{\text{Japon}}$ et $\mu_{\text{USA}}$ les **vraies** mpg moyennes (inconnues) des deux populations :

$$H_0 : \mu_{\text{Japon}} = \mu_{\text{USA}} \qquad \text{vs} \qquad H_1 : \mu_{\text{Japon}} \neq \mu_{\text{USA}}$$

**Comment on décide ?** On construit une **statistique de test** $T$ — la version empirique de ce qu'on veut tester. Sur Auto, naturellement :

$$T = \bar{Y}_{\text{Japon}} - \bar{Y}_{\text{USA}}$$

Si $H_0$ vraie, $T$ devrait être **proche de 0** (à du bruit d'échantillonnage près). Si $H_1$ vraie, $T$ devrait être **loin de 0**. La question devient : *$T_{\text{obs}} = 10.42$ est-il "loin" de 0 ou "pas loin" ?* Pour répondre il faut connaître la **distribution de $T$ sous $H_0$**.

> [!warning] La mécanique du test en 3 temps
> 1. **On construit** une statistique $T$ qui résume "à quel point les données s'écartent de $H_0$".
> 2. **On connaît** la distribution de $T$ sous $H_0$ (étape technique : TCL, Student, etc.).
> 3. **On rejette** $H_0$ si la valeur observée $T_{\text{obs}}$ est extrême sous cette distribution.
> 
> Tout le formalisme qui suit ($\Theta_0$, $\psi$, $\alpha$, p-value…) ne fait que rendre cette idée rigoureuse.

> ⚠️ **Asymétrie fondamentale.** On ne peut pas *prouver* $H_0$ — on peut seulement la rejeter ou *ne pas la rejeter*. Si on observait $\Delta = 0.5$ mpg, on ne rejetterait pas $H_0$, mais cela ne **prouverait pas** que $\mu_{\text{Japon}} = \mu_{\text{USA}}$ — ça voudrait juste dire qu'on n'a pas assez de preuves pour conclure le contraire. C'est pour cette raison qu'on dit *"on ne rejette pas $H_0$"* et jamais *"on accepte $H_0$"*.

---

## I. Cadre général

Maintenant qu'on a l'intuition (section 0), on rend tout ça rigoureux. Le formalisme qui suit n'invente rien de nouveau — il met juste un nom propre sur ce qu'on a déjà décrit.

### A. Le modèle statistique

Un test commence toujours par fixer un **modèle statistique** : on dit *quelle famille de distributions* est censée avoir généré nos données, en fonction d'un paramètre $\theta$ inconnu.

> [!warning] Définition — Modèle statistique
> Un modèle statistique est une paire $(E, (\mathbb{P}_\theta)_{\theta \in \Theta})$ où :
> - $E$ est l'**espace d'échantillonnage** (l'ensemble des valeurs possibles pour une observation)
> - $(\mathbb{P}_\theta)_{\theta \in \Theta}$ est une **famille de distributions** indexée par un paramètre $\theta$ qui vit dans l'**espace des paramètres** $\Theta$
> 
> Les données $X_1, \ldots, X_n$ sont supposées i.i.d. selon $\mathbb{P}_\theta$ pour un certain $\theta \in \Theta$ (inconnu).

**Sur Auto.** On suppose que les mpg des voitures suivent une loi normale dans chaque population, avec une moyenne propre à chaque origine :

$$Y_i^{\text{Japon}} \overset{iid}{\sim} \mathcal{N}(\mu_{\text{Japon}}, \sigma^2) \qquad Y_i^{\text{USA}} \overset{iid}{\sim} \mathcal{N}(\mu_{\text{USA}}, \sigma^2)$$

Le paramètre inconnu est $\theta = (\mu_{\text{Japon}}, \mu_{\text{USA}}) \in \mathbb{R}^2$. Donc ici :
- **Espace d'échantillonnage** $E = \mathbb{R}$ (chaque mpg est un nombre réel)
- **Espace des paramètres** $\Theta = \mathbb{R}^2$ (chaque couple $(\mu_J, \mu_U)$ est admissible *a priori*)

### B. Les hypothèses comme partition de $\Theta$

On cherche à départager **deux scénarios** sur le vrai $\theta$. Mathématiquement, ça revient à partitionner $\Theta$ en deux sous-ensembles disjoints.

> [!warning] Définition — Hypothèse nulle et alternative
> On choisit deux sous-ensembles disjoints $\Theta_0, \Theta_1 \subset \Theta$ et on pose :
> 
> $$H_0 : \theta \in \Theta_0 \qquad \text{vs} \qquad H_1 : \theta \in \Theta_1$$
> 
> $H_0$ est l'**hypothèse nulle** (la présomption d'innocence), $H_1$ est l'**hypothèse alternative**.

**Sur Auto** :

$$\Theta_0 = \{(\mu_J, \mu_U) \in \mathbb{R}^2 : \mu_J = \mu_U\} \qquad \Theta_1 = \{(\mu_J, \mu_U) \in \mathbb{R}^2 : \mu_J \neq \mu_U\}$$

Géométriquement, $\Theta_0$ est la **droite diagonale** $\mu_J = \mu_U$ dans le plan $\mathbb{R}^2$, et $\Theta_1$ est tout le reste du plan.

> [!note]- Hypothèse simple vs composite
> - **$H_0$ simple** : $\Theta_0$ est réduit à un point, par exemple $H_0: \mu = 0$ avec $\Theta_0 = \{0\}$.
> - **$H_0$ composite** : $\Theta_0$ est un sous-ensemble plus gros, par exemple $H_0: \mu \le 30$ avec $\Theta_0 = (-\infty, 30]$, ou notre cas Auto où $\Theta_0$ est une droite entière du plan.
> 
> Cette distinction aura son importance plus loin (section sur le niveau d'un test) parce qu'avec $H_0$ composite il faut contrôler l'erreur de type 1 *uniformément* sur tout $\Theta_0$, pas juste en un point.

### C. Le test comme fonction $\psi$

Un test, c'est juste une **règle de décision** : au vu des données, je rejette $H_0$ ou pas ? Formellement :

> [!warning] Définition — Test
> Un test est une fonction $\psi$ à valeurs dans $\{0, 1\}$ qui dépend des données :
> - $\psi = 0$ → on **ne rejette pas** $H_0$
> - $\psi = 1$ → on **rejette** $H_0$, on conclut $H_1$

En pratique, $\psi$ s'écrit toujours sous la forme :

$$\psi = \mathbb{1}\{T \in \mathcal{R}\}$$

où $T$ est la **statistique de test** (la quantité calculée sur les données, comme $T = \bar{Y}_{\text{Japon}} - \bar{Y}_{\text{USA}}$ sur Auto) et $\mathcal{R}$ est la **région de rejet** (les valeurs de $T$ pour lesquelles on rejette).

> [!warning] Définition — Région de rejet et région d'acceptation
> - **Région de rejet** : $R_\psi = \{x \in E^n : \psi(x) = 1\}$ — les données qui mènent à rejeter $H_0$.
> - **Région d'acceptation** : son complémentaire — les données qui mènent à ne pas rejeter $H_0$.

**Sur Auto.** Notre statistique de test naturelle est $T = \bar{Y}_{\text{Japon}} - \bar{Y}_{\text{USA}}$ et la région de rejet a typiquement la forme $\mathcal{R} = \{|T| > c\}$ pour un seuil $c$ à déterminer. Toute la difficulté va être de **choisir $c$ correctement** — ni trop grand (on raterait des vraies différences), ni trop petit (on rejetterait $H_0$ pour rien). C'est ce qui motive les notions d'erreur de type 1, type 2, niveau, p-value qui suivent.

![[Pasted image 20260506192739.png]]

**Figure 2.** Anatomie d'un test illustrée sur Auto, avec $T = \bar{Y}_{\text{Japon}} - \bar{Y}_{\text{USA}}$ et un seuil $c$ correspondant à un niveau $\alpha = 5\%$. **À gauche** : la distribution théorique de $T$ sous $H_0$ (centrée en 0, comme attendu si les deux populations ont la même mpg moyenne). La **région d'acceptation** (zone verte) correspond aux valeurs de $T$ "compatibles avec $H_0$" ; la **région de rejet** $\mathcal{R} = \{|T| > c\}$ (zones rouges hachurées) regroupe les valeurs trop extrêmes pour être crédibles sous $H_0$. L'aire totale des deux zones rouges est exactement $\alpha$ — la probabilité de rejeter à tort. **À droite** : on superpose la valeur observée $T_{\text{obs}} = 10.42$. Elle tombe à environ 11 écart-types de 0, **très loin** dans la queue droite — bien au-delà du seuil $c$. Conclusion visuelle immédiate : $T_{\text{obs}} \in \mathcal{R}$, donc $\psi = 1$, donc on rejette $H_0$. La différence de mpg entre voitures japonaises et américaines n'est pas du hasard d'échantillonnage.

> [!warning] Récapitulatif — les ingrédients d'un test
> Pour construire un test il faut quatre choses :
> 1. **Un modèle statistique** $(E, (\mathbb{P}_\theta)_{\theta \in \Theta})$
> 2. **Une partition** de $\Theta$ en $\Theta_0$ (hypothèse nulle) et $\Theta_1$ (alternative)
> 3. **Une statistique de test** $T$ — une quantité calculée sur les données qui résume "à quel point on s'écarte de $H_0$"
> 4. **Une région de rejet** $\mathcal{R}$ — le seuil au-delà duquel on rejette $H_0$

### D. Erreurs de type 1 et type 2

Quand on prend une décision avec un test, on peut se tromper de deux façons. Comme dans un procès :

|  | $H_0$ vraie (innocent) | $H_1$ vraie (coupable) |
|---|:---:|:---:|
| $\psi = 0$ (acquittement) | ✅ OK | ❌ Erreur de **type 2** |
| $\psi = 1$ (condamnation) | ❌ Erreur de **type 1** | ✅ OK |

> [!warning] Erreur de type 1 d'un test $\psi$ (rejeter $H_0$ alors qu'elle est vraie)
> $$
> \begin{aligned}
> \alpha_\psi : \Theta_0 &\rightarrow \mathbb{R} \\
> \theta &\mapsto \mathbb{P}_\theta[\psi = 1]
> \end{aligned}
> $$
> En clair : **probabilité de condamner un innocent**.

> [!warning] Erreur de type 2 d'un test $\psi$ (ne pas rejeter $H_0$ alors que $H_1$ est vraie)
> $$
> \begin{aligned}
> \beta_\psi : \Theta_1 &\rightarrow \mathbb{R} \\
> \theta &\mapsto \mathbb{P}_\theta[\psi = 0]
> \end{aligned}
> $$
> En clair : **probabilité d'acquitter un coupable**.

> 💡 **Pourquoi $\alpha_\psi$ est définie sur $\Theta_0$ (pas un seul point) ?** Parce que sous $H_0$ il peut y avoir *plusieurs* valeurs admissibles pour $\theta$ (cas composite). Pour chacune, la probabilité de rejeter peut être différente. On doit donc voir l'erreur de type 1 comme une **fonction** de $\theta$, pas un nombre. Idem pour $\beta_\psi$ sur $\Theta_1$.

**Sur Auto, lien avec la figure 2.** L'erreur de type 1 a une interprétation visuelle directe : c'est l'**aire totale des zones rouges hachurées** sous la courbe sous $H_0$. En effet, sous $H_0$ ($\mu_J = \mu_U$), la probabilité que $T$ tombe dans la région de rejet $\mathcal{R} = \{|T| > c\}$ est exactement la probabilité de la zone rouge — donc la probabilité de rejeter à tort. C'est précisément le $\alpha = 5\%$ qu'on a fixé en construisant le test.

L'erreur de type 2 est plus subtile à visualiser : elle dépend de "à quel point $H_1$ est vraie" (i.e. à quel point la vraie différence $\mu_J - \mu_U$ est grande). Si la vraie différence est énorme (comme les 10.42 mpg observés), $T$ tombera presque toujours dans la zone rouge → erreur de type 2 quasi-nulle, on rejette presque toujours quand on doit. Si la vraie différence est minuscule (genre 0.3 mpg), la distribution de $T$ sous $H_1$ est presque identique à celle sous $H_0$, et on risque souvent de ne pas rejeter à tort.

> ⚠️ **Compromis fondamental.** Diminuer $c$ → on rejette plus → moins d'erreur de type 2 mais plus d'erreur de type 1. Augmenter $c$ → on rejette moins → l'inverse. **Les deux erreurs sont en tension** : on ne peut pas les minimiser simultanément. La pratique standard est de **fixer $\alpha$** (typiquement 5%) et de chercher à **maximiser la puissance** (= minimiser l'erreur de type 2) sous cette contrainte.

### E. Puissance d'un test

> [!warning] Puissance d'un test $\psi$
> Probabilité de rejeter quand on devrait vraiment rejeter. Quand cette quantité est grande, on dit qu'on a un **test puissant**.
> 
> $$\pi_\psi = \inf_{\theta \in \Theta_1} (1 - \beta_\psi(\theta))$$

L'idée : la **fonction de puissance** $\theta \mapsto \mathbb{P}_\theta[\psi = 1]$ donne, pour chaque vraie valeur $\theta$ possible, la probabilité que le test rejette. Ce qu'on veut :

- Pour $\theta \in \Theta_0$ : que cette fonction soit **petite** (peu d'erreur de type 1).
- Pour $\theta \in \Theta_1$ : que cette fonction soit **grande** (puissance élevée, on rejette bien quand on doit).

La **puissance** $\pi_\psi$ est définie comme le **pire cas** sur $\Theta_1$ : la valeur de $\theta \in \Theta_1$ pour laquelle on a le plus de mal à rejeter. C'est conservateur, comme pour le niveau.

Pour visualiser concrètement à quoi ressemble cette fonction et pourquoi on veut qu'elle ait une certaine forme, sortons un instant d'Auto sur un cas plus simple où on peut tout dessiner en 1D. Imaginons un test $H_0 : \mu \le 30$ vs $H_1 : \mu > 30$ (modèle gaussien à un échantillon, $\sigma$ connue, $n = 25$). La figure 3 ci-dessous trace la fonction de puissance pour ce test, avec un code couleur qui sépare $\Theta_0$ (vert, où on veut peu rejeter) de $\Theta_1$ (rouge, où on veut beaucoup rejeter).

![[im3_fonction_puissance.png]]

**Figure 3.** Fonction de puissance $\mu \mapsto \mathbb{P}_\mu[\psi = 1]$ pour le test $H_0 : \mu \le 30$ vs $H_1 : \mu > 30$ au niveau $\alpha = 5\%$. Sur la zone verte ($\Theta_0$), la courbe doit rester **basse** ($\le \alpha$) — c'est la contrainte du niveau, qui borne l'erreur de type 1. Sur la zone rouge ($\Theta_1$), la courbe doit être **haute** (proche de 1) — c'est la puissance, qui correspond à la capacité du test à détecter une vraie alternative. À la frontière $\mu = 30$, la fonction vaut exactement $\alpha = 0.05$ (point noir). La courbe est sigmoïdale parce qu'elle transitionne continûment entre les deux régimes : plus la vraie $\mu$ est éloignée de la frontière, plus le test rejette facilement (à droite) ou rarement (à gauche).

> 💡 **Et sur Auto ?** On ne peut pas tracer la fonction de puissance comme une courbe 1D simple parce que $\Theta = \mathbb{R}^2$ — il faudrait un graphique 3D où l'axe horizontal serait $(\mu_J, \mu_U)$ et l'axe vertical $\mathbb{P}_\theta[\psi = 1]$. La fonction serait nulle sur la diagonale $\mu_J = \mu_U$ (i.e. égale à $\alpha$ au plus), et augmenterait à mesure qu'on s'éloigne de cette diagonale. Mais le principe est exactement le même qu'en 1D.

### F. Niveau d'un test

> [!warning] Niveau d'un test
> Un test $\psi$ a **niveau $\alpha$** (penser $\alpha = 5\%, 1\%, \ldots$) si :
> 
> $$\alpha_\psi(\theta) \le \alpha, \quad \forall \theta \in \Theta_0$$

**Interprétation.** Le niveau est une **borne supérieure** sur l'erreur de type 1 dans le pire cas sous $H_0$. Quand on dit "test au niveau 5%", on garantit qu'on rejettera $H_0$ à tort dans **au plus 5% des cas**, peu importe la vraie valeur de $\theta$ dans $\Theta_0$.

**Sur la figure 2** : c'est exactement ce qu'on a vu — l'aire totale des zones rouges sous la courbe sous $H_0$ vaut $\alpha$. Le seuil $c$ a été choisi *précisément* pour que cette aire fasse 5%.

> [!warning] Niveau asymptotique
> Un test $\psi$ a **niveau asymptotique $\alpha$** si :
> 
> $$\lim_{n \to \infty} \alpha_{\psi_n}(\theta) \le \alpha, \quad \forall \theta \in \Theta_0$$

> 💡 **Pourquoi cette version asymptotique ?** Quand on n'a pas la distribution exacte de $T$ sous $H_0$ pour $n$ fini (par exemple parce qu'on s'appuie sur le TCL), le contrôle de l'erreur de type 1 n'est valide qu'à la limite $n \to \infty$. Pour $n$ grand mais fini, on a $\alpha_\psi(\theta) \approx \alpha$. La distinction "niveau exact" vs "niveau asymptotique" deviendra cruciale dans la section sur le T-test (qui est *non-asymptotique*) vs Z-test (*asymptotique*).

En général, un test a la forme :

$$\psi = \mathbb{1}\{T_n > c\}$$

pour une statistique $T_n$ et un seuil $c \in \mathbb{R}$. La région de rejet est $R_\psi = \{T_n > c\}$ et $c$ est calibré pour atteindre exactement le niveau $\alpha$ voulu.

### G. p-value

Plutôt que de fixer un seuil $c$ et de regarder si $T_{\text{obs}}$ tombe au-delà, on peut renverser la question : **à quel point $T_{\text{obs}}$ est-il extrême sous $H_0$ ?** C'est l'idée de la p-value.

> 💡 **Intuition opérationnelle.** La p-value est la **probabilité d'observer une statistique de test au moins aussi extrême que celle observée, sous l'hypothèse que $H_0$ est vraie**.
> 
> Sur la figure 2, c'est l'aire **au-delà** de $T_{\text{obs}}$ sous la courbe sous $H_0$ (à droite + à gauche en bilatéral). Plus cette aire est petite, plus l'observation est *surprenante* sous $H_0$, plus on est légitime à la rejeter.

> [!warning] Définition — p-value
> La p-value (asymptotique) d'un test $\psi_\alpha$ est le plus petit niveau (asymptotique) $\alpha$ auquel $\psi_\alpha$ rejette $H_0$. Elle est aléatoire — elle dépend de l'échantillon.
> 
> **Règle d'or** : $p\text{-value} \le \alpha \iff H_0$ est rejetée par $\psi_\alpha$ au niveau (asymptotique) $\alpha$.

**Pourquoi cette définition est équivalente à l'intuition opérationnelle ?** Si la p-value vaut $0.03$, ça veut dire : "sous $H_0$, on observerait une statistique aussi extrême que la mienne avec probabilité $3\%$". Si on fixe un niveau $\alpha = 5\%$, alors $0.03 \le 0.05$ → on rejette. Si on fixe un niveau $\alpha = 1\%$, alors $0.03 > 0.01$ → on ne rejette pas. La p-value est donc *exactement* la frontière entre "je rejette" et "je ne rejette pas".

**Sur Auto.** $T_{\text{obs}} = 10.42$ tombe à environ 11 écart-types de 0 sous $H_0$. La probabilité d'observer un $T$ aussi extrême sous $H_0$ est **astronomiquement petite** (de l'ordre de $10^{-26}$). Quel que soit le niveau $\alpha$ raisonnable qu'on choisisse, on rejette $H_0$ — la différence entre voitures japonaises et américaines n'est clairement pas du hasard.

Pour visualiser concrètement la p-value, illustrons-la sur un cas générique avec $T_{\text{obs}} = 2.5$ (sur Auto, $T_{\text{obs}} \approx 11\sigma$ rendrait la zone hachurée invisible à l'œil, mais le principe est exactement le même).

![[Pasted image 20260506195235.png]]

**Figure 4.** Visualisation de la p-value sur un exemple générique avec $T_{\text{obs}} = 2.5$. Sous $H_0$, $T$ suit la distribution grise centrée en 0. La **p-value est l'aire totale des zones orange hachurées** — la probabilité, sous $H_0$, d'observer une valeur de $T$ aussi extrême ou plus extrême que $T_{\text{obs}}$ (test bilatéral → on regarde des deux côtés). Ici p-value $\approx 0.012 < \alpha = 0.05$ → on rejette $H_0$. Les lignes rouges en pointillés marquent les seuils $\pm c$ correspondant à $\alpha = 5\%$ (comme dans la figure 2) : on voit que $T_{\text{obs}}$ dépasse le seuil, ce qui est équivalent à dire "p-value $< \alpha$". **C'est la règle d'or** : *p-value $\le \alpha \iff$ on rejette*. Plus $T_{\text{obs}}$ s'éloigne de 0, plus la zone orange rétrécit, plus la p-value devient petite, plus le rejet devient évident.


> ⚠️ **Une p-value n'est PAS la probabilité que $H_0$ soit vraie.** C'est la probabilité d'observer les données *sous* $H_0$. Confondre les deux est une erreur classique : on compare $\mathbb{P}(\text{données} \mid H_0)$ avec $\mathbb{P}(H_0 \mid \text{données})$, qui sont deux quantités très différentes (lien : théorème de Bayes).

### H. Vocabulaire — un échantillon vs deux échantillons

> [!warning] One-sample test
> Un test où un paramètre inconnu $\mu$ est comparé à une valeur de référence connue. Exemple : comparer la moyenne $\mu$ d'aujourd'hui à la moyenne historique connue de 5.5.

> [!warning] Two-sample test
> Un test où deux paramètres inconnus sont comparés entre eux. Exemple : comparer $\mu_{\text{drug}}$ inconnu à $\mu_{\text{control}}$ inconnu pour quantifier l'effet d'un médicament.

- Si $H_1: \theta \neq \theta_0$ : test **bilatéral** (two-sided)
- Si $H_1: \theta > \theta_0$ ou $H_1: \theta < \theta_0$ : test **unilatéral** (one-sided)

**Sur Auto.** On compare $\mu_{\text{Japon}}$ et $\mu_{\text{USA}}$ tous deux **inconnus** → c'est un **two-sample test**. Et $H_1: \mu_J \neq \mu_U$ → c'est **bilatéral**. C'est précisément ce que fera le test de Welch que l'on construira plus loin.

---

## II. p-value selon le type d'alternative

La p-value se calcule comme l'aire au-delà de $T_{\text{obs}}$ sous la distribution de $T$ sous $H_0$ — mais **"au-delà" dépend de la forme de $H_1$**. Trois cas selon que l'alternative est unilatérale (à droite, à gauche) ou bilatérale.

### A. Test unilatéral à droite

$$H_0: \theta = \theta_0 \qquad H_1: \theta > \theta_0$$

On rejette $H_0$ uniquement quand $T_{\text{obs}}$ est **trop grand**. La p-value est l'aire de la queue **droite** au-delà de $T_{\text{obs}}$ :

$$\text{p-value} = \mathbb{P}_{H_0}[T \ge T_{\text{obs}}]$$

![[im9.png|307]]

### B. Test unilatéral à gauche

$$H_0: \theta = \theta_0 \qquad H_1: \theta < \theta_0$$

Symétriquement, on rejette uniquement quand $T_{\text{obs}}$ est **trop petit**. La p-value est l'aire de la queue **gauche** :

$$\text{p-value} = \mathbb{P}_{H_0}[T \le T_{\text{obs}}]$$

![[images/2-Statistiques/Frequentist/Inférence statistique/im10 (1).png]]

### C. Test bilatéral

$$H_0: \theta = \theta_0 \qquad H_1: \theta \neq \theta_0$$

On rejette $H_0$ quand $T_{\text{obs}}$ est trop loin de 0 dans **n'importe quelle direction**. La p-value somme les deux queues :

$$\text{p-value} = \mathbb{P}_{H_0}[|T| \ge |T_{\text{obs}}|] = 2 \cdot \mathbb{P}_{H_0}[T \ge |T_{\text{obs}}|]$$

(La dernière égalité utilise la symétrie de la distribution de $T$ sous $H_0$ — vraie pour la gaussienne, vraie pour Student.)

![[images/2-Statistiques/Frequentist/Inférence statistique/im11.png]]

> 💡 **Cas Auto.** L'hypothèse alternative est $H_1: \mu_J \neq \mu_U$ → **test bilatéral** → on multiplie par 2. Et c'est ce qu'on a fait dans la figure 4 (les deux zones orange hachurées des deux côtés).

---

## III. Taxonomie des tests

Avant d'attaquer les tests concrets, il est utile de prendre du recul : **les tests d'hypothèses se classent par familles selon ce qu'ils testent**. Cette note couvre principalement la première famille (tests de moyenne) ; les autres font l'objet de notes séparées.

| Famille                                  | Ce qu'on teste                                                  | Exemples de tests                                                         |
| ---------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Tests de moyenne**                     | Une moyenne $\mu$ ou comparaison de moyennes                    | Z-test, T-test, ANOVA, ANCOVA                                             |
| **Tests de proportion**                  | Une proportion $p$ ou comparaison de proportions                | Z-test pour une proportion, $\chi^2$ d'homogénéité                        |
| **Tests de variance**                    | Une variance ou comparaison de variances                        | $\chi^2$ pour la variance, test F de Fisher                               |
| **Tests d'adéquation** (goodness-of-fit) | Si une distribution observée colle à une distribution théorique | $\chi^2$ d'adéquation, Kolmogorov-Smirnov, Shapiro-Wilk, Anderson-Darling |
| **Tests d'indépendance / corrélation**   | Si deux variables sont liées                                    | $\chi^2$ d'indépendance, Pearson, Spearman, Kendall                       |
| **Tests de rang** (non-paramétriques)    | Comparer des distributions sans hypothèse gaussienne            | Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Friedman            |

> 💡 **Le fil conducteur des tests de moyenne.** Tous les tests qu'on va voir (Z-test, T-test, ANOVA, ANCOVA) partagent la même structure : on construit une statistique qui ressemble à
> $$\frac{\text{moyenne(s) observée(s)} - \text{valeur(s) sous } H_0}{\text{erreur-type}}$$
> et on regarde si elle est extrême sous une distribution de référence (gaussienne, Student, ou Fisher).

**Quand utiliser quoi ?** Vue d'ensemble pour les tests de moyenne :

| Test | Cadre |
|---|---|
| **Z-test** | $\sigma$ connu, ou cas asymptotique via TCL ($n$ grand). Peu utilisé en pratique sauf pour Bernoulli/proportions. |
| **T-test (one-sample)** | Une moyenne vs valeur de référence, $\sigma$ inconnu, échantillon gaussien |
| **T-test (two-sample / Welch)** | Comparer **2 moyennes** inconnues, $\sigma$ inconnus |
| **ANOVA** | Comparer **$k \ge 3$ moyennes** ($H_0: \mu_1 = \cdots = \mu_k$) |
| **ANCOVA** | ANOVA + covariables continues (contrôler pour des facteurs supplémentaires) |

---

## IV. Tests de moyenne

### A. Z-test : tests asymptotiques

Le Z-test repose sur le **TCL** : sous $H_0$, la statistique standardisée tend vers $\mathcal{N}(0,1)$ quand $n \to \infty$. Test au niveau **asymptotique** $\alpha$ — le contrôle de l'erreur de type 1 n'est garanti qu'à la limite, pour $n$ grand mais fini il y a une approximation.

#### A.1 Cas bilatéral — Bernoulli

> [!example] Setup
> Soit $X_1, \ldots, X_n \overset{iid}{\sim} \text{Ber}(p)$, $p \in [0,1]$ inconnu. On teste :
> $$H_0: p = 1/2 \quad \text{vs} \quad H_1: p \neq 1/2$$

Par le TCL, sous $H_0$ :

$$T_n = \sqrt{n}\frac{|\hat{p}_n - 1/2|}{\sqrt{1/2 \cdot (1-1/2)}} \xrightarrow[n \to \infty]{(d)} |\mathcal{N}(0, 1)|$$

Le test au niveau asymptotique $\alpha$ rejette si $T_n > q_{\alpha/2}$, où $q_{\alpha/2}$ est le $(1 - \alpha/2)$-quantile de $\mathcal{N}(0,1)$ (typiquement $q_{0.025} = 1.96$).

> [!example] Application — pièce truquée
> Une pièce est lancée $n = 30$ fois, on obtient pile $13$ fois. La pièce est-elle biaisée ? Avec $\hat{p}_n = 13/30 \approx 0.43$ et $\alpha = 5\%$ :
> 
> $$T_n = \sqrt{30}\frac{|0.43 - 0.5|}{0.5} \approx 0.77 < 1.96 = q_{2.5\%}$$
> 
> On **ne rejette pas** $H_0$ — pas assez de preuves pour affirmer que la pièce est truquée.

#### A.2 Cas unilatéral

> [!example] Setup
> Soit $X_1, \ldots, X_n \overset{iid}{\sim} \text{Ber}(p)$. On teste :
> $$H_0: p \ge 0.33 \quad \text{vs} \quad H_1: p < 0.33$$

Ici $\Theta_0 = [0.33, 1]$ est composite. Pour contrôler l'erreur de type 1 *uniformément* sur $\Theta_0$, on évalue au **pire cas** $p_0 = 0.33$ (à la frontière). Le test rejette si :

$$\sqrt{n}\frac{\hat{p}_n - 0.33}{\sqrt{0.33 \cdot 0.67}} < -q_\alpha$$

> [!example] Application — sondage YouTube
> Sondage 2017 sur $n = 4971$ Américains : $32\%$ déclarent obtenir au moins une partie de leur info sur YouTube. Peut-on conclure qu'**au plus** un tiers s'informent sur YouTube ?
> 
> $$\sqrt{4971}\frac{0.32 - 0.33}{\sqrt{0.33 \cdot 0.67}} \approx -1.50$$
> 
> Avec $\alpha = 5\%$, $-q_\alpha = -1.645$. Comme $-1.50 > -1.645$, on **ne rejette pas** $H_0$.

> [!note]- Cas $\Theta_0 = [a, b]$ avec deux frontières
> Si $\Theta_0 = [0.5, 0.6]$ par exemple, il faut calculer la statistique pour chacune des deux frontières et prendre celle qui donne la plus grande probabilité d'erreur de type 1. C'est plus rare mais ça arrive.

### B. T-test : tests non-asymptotiques

Le T-test est utilisable même pour de **petits échantillons** (typiquement $n < 30$), à condition que les données soient **gaussiennes**. La distribution de la statistique sous $H_0$ est connue **exactement** (loi de Student), pas seulement asymptotiquement.

#### B.1 Pré-requis — distributions $\chi^2$ et Student

> [!warning] Définition — Distribution $\chi^2$
> Pour un entier $d > 0$, la distribution $\chi^2_d$ est la loi de $Z_1^2 + \cdots + Z_d^2$, où $Z_1, \ldots, Z_d \overset{iid}{\sim} \mathcal{N}(0, 1)$.

> [!warning] Définition — Distribution de Student
> Pour un entier $d > 0$, la distribution de Student $t_d$ est la loi de $\frac{Z}{\sqrt{V/d}}$, où $Z \sim \mathcal{N}(0, 1)$, $V \sim \chi^2_d$, et $Z \perp V$.

> [!note]- Pourquoi cette construction marche
> Si $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ et $S_n^2 = \frac{1}{n-1}\sum(X_i - \bar{X}_n)^2$, alors par le **théorème de Cochran** :
> - $\sqrt{n}\frac{\bar{X}_n - \mu}{\sigma} \sim \mathcal{N}(0, 1)$
> - $\frac{(n-1) S_n^2}{\sigma^2} \sim \chi^2_{n-1}$
> - et ces deux quantités sont **indépendantes**.
> 
> Donc en formant le ratio :
> $$\frac{\bar{X}_n - \mu}{S_n / \sqrt{n}} = \frac{\sqrt{n}(\bar{X}_n - \mu)/\sigma}{\sqrt{S_n^2/\sigma^2}} \sim t_{n-1}$$
> 
> Le $\sigma$ inconnu se simplifie au numérateur et au dénominateur. Cette distribution ne dépend ni de $\mu$ ni de $\sigma$, donc on peut tabuler ses quantiles une fois pour toutes.

#### B.2 T-test à un échantillon (one-sample)

> [!example] Setup
> $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ avec $\mu, \sigma^2$ inconnus. On teste $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$.

Statistique de test :

$$T_n = \frac{\bar{X}_n - \mu_0}{S_n / \sqrt{n}}$$

Sous $H_0$, $T_n \sim t_{n-1}$ (loi exacte, pas asymptotique). Le test au niveau $\alpha$ :

$$\psi_\alpha = \mathbb{1}\{|T_n| > q_{\alpha/2}\}$$

où $q_{\alpha/2}$ est le $(1 - \alpha/2)$-quantile de $t_{n-1}$.

> [!example] Application — Pizza Hut
> Pizza Hut prétend qu'une part contient en moyenne $4$ pepperonis : $H_0: \mu = 4$ vs $H_1: \mu \neq 4$. On échantillonne $n = 9$ parts : $\bar{X} = 4.3$, $S = 1.2$.
> 
> $$T_n = \frac{4.3 - 4.0}{1.2 / \sqrt{9}} = 0.75$$
> 
> p-value bilatérale : $2 \cdot \mathbb{P}[t_8 > 0.75] = 0.4747$. **On ne rejette pas** $H_0$.

#### B.3 T-test à deux échantillons (Welch) — résolution du fil rouge

C'est le test qu'on attend depuis la section 0. Cadre général :

> [!example] Setup
> Deux échantillons indépendants :
> - $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu_X, \sigma_X^2)$
> - $Y_1, \ldots, Y_m \overset{iid}{\sim} \mathcal{N}(\mu_Y, \sigma_Y^2)$
> 
> avec $\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2$ tous inconnus. On teste $H_0: \mu_X = \mu_Y$ vs $H_1: \mu_X \neq \mu_Y$.

Statistique de test :

$$T = \frac{\bar{X}_n - \bar{Y}_m}{\sqrt{\frac{S_X^2}{n} + \frac{S_Y^2}{m}}}$$

Sous $H_0$, $T$ suit **approximativement** une loi de Student $t_N$ avec un degré de liberté ajusté par la formule de **Welch-Satterthwaite** :

$$N = \frac{\left(S_X^2/n + S_Y^2/m\right)^2}{\frac{S_X^4}{n^2(n-1)} + \frac{S_Y^4}{m^2(m-1)}} \ge \min(n, m)$$

> [!warning] Application sur Auto — la résolution attendue
> On a deux échantillons gaussiens (sous l'hypothèse de modèle de la section I.A) :
> - **Japon** : $n_J = 79$, $\bar{Y}_J = 30.45$, $S_J^2 \approx 37.4$
> - **USA** : $n_U = 245$, $\bar{Y}_U = 20.03$, $S_U^2 \approx 40.9$
> 
> Le test Japon vs USA donne :
> 
> $$T_{\text{obs}} = \frac{30.45 - 20.03}{\sqrt{37.4/79 + 40.9/245}} = \frac{10.42}{\sqrt{0.473 + 0.167}} = \frac{10.42}{0.800} \approx 13.0$$
> 
> Degrés de liberté Welch-Satterthwaite : $N \approx 142$. Sous $H_0$, $T \sim t_{142}$, qui est essentiellement indistinguable de $\mathcal{N}(0,1)$.
> 
> p-value bilatérale : $2 \cdot \mathbb{P}[t_{142} > 13.0] \approx 10^{-26}$.
> 
> **Conclusion** : on rejette $H_0$ à n'importe quel niveau raisonnable. La différence de mpg moyenne entre voitures japonaises et américaines (~10 mpg) est **statistiquement écrasante** et ne peut pas s'expliquer par le hasard d'échantillonnage. C'est la réponse rigoureuse à la question de la section 0.

> [!note]- Subtilité — variances égales ou pas ?
> Si on suppose en plus $\sigma_X^2 = \sigma_Y^2$, on retombe sur le **T-test classique de Student** avec une variance regroupée (pooled variance) et $N = n + m - 2$ degrés de liberté exacts. Le test de Welch est plus général : il ne suppose **pas** l'égalité des variances et est donc plus robuste. C'est l'option par défaut dans la plupart des logiciels modernes (`scipy.stats.ttest_ind(equal_var=False)` en Python).

#### B.4 Discussion — avantages et limites du T-test

> 💡 **Avantage.** Non-asymptotique : le T-test est valide pour de petits échantillons (où le TCL est trop approximatif), et reste valide pour de grands échantillons.
> 
> ⚠️ **Limite.** Repose sur l'hypothèse que les données sont **gaussiennes**. Pour de grands échantillons, ce n'est pas critique (le TCL fait que $\bar{X}_n$ est gaussien quel que soit l'échantillon original). Pour de petits échantillons, il faut vérifier la normalité — typiquement avec un test de **Shapiro-Wilk** ou un **QQ-plot**.

### C. ANOVA (ANalysis Of VAriance) — comparer $k \ge 3$ moyennes

Et si on voulait comparer **plus de deux groupes** simultanément ? Sur Auto, on a en fait **trois origines** : Japon, USA, et Europe. Question naturelle : *"y a-t-il une différence de mpg entre ces trois origines ?"*

On pourrait faire trois T-tests (Japon vs USA, Japon vs Europe, USA vs Europe), mais cela soulève un problème : le **multiple testing**. Avec trois tests à $\alpha = 5\%$ chacun, la probabilité de faire au moins une erreur de type 1 monte à $1 - 0.95^3 \approx 14\%$. L'ANOVA résout ce problème en posant **une seule hypothèse globale**.

#### C.1 Cadre formel

> [!warning] Modèle ANOVA à un facteur (one-way ANOVA)
> On a $k$ groupes indépendants, le groupe $j$ contenant $n_j$ observations $y_{ij}$ pour $i = 1, \ldots, n_j$. Le modèle s'écrit :
> 
> $y_{ij} = \mu + \tau_j + \varepsilon_{ij}, \qquad \varepsilon_{ij} \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$
> 
> où :
> - $\mu$ est la **moyenne globale**
> - $\tau_j$ est l'**effet du groupe $j$** (écart de la moyenne du groupe $j$ par rapport à la moyenne globale)
> - $\varepsilon_{ij}$ est le **résidu individuel** au sein du groupe
> 
> On teste :
> 
> $H_0: \tau_1 = \tau_2 = \cdots = \tau_k = 0 \quad (\text{i.e. } \mu_1 = \cdots = \mu_k) \qquad \text{vs} \qquad H_1: \exists\, j, \tau_j \neq 0$

#### C.2 Décomposition de la variance — l'idée centrale

L'ANOVA repose sur une **décomposition fondamentale** de la variance totale en deux parties orthogonales : ce que les groupes expliquent, et ce qu'il reste comme variabilité individuelle.

Pour formaliser, notons :
- $\bar{y}$ la **moyenne globale** ($= \frac{1}{N}\sum_{j,i} y_{ij}$ avec $N = \sum_j n_j$)
- $\bar{y}_j$ la **moyenne du groupe $j$** ($= \frac{1}{n_j}\sum_i y_{ij}$)

Pour chaque observation $y_{ij}$, on peut écrire son écart à la moyenne globale comme la somme de **deux écarts** :

$\underbrace{(y_{ij} - \bar{y})}_{\text{total}} = \underbrace{(\bar{y}_j - \bar{y})}_{\text{inter-groupes }\tau_j} + \underbrace{(y_{ij} - \bar{y}_j)}_{\text{intra-groupe }\varepsilon_{ij}}$

En élevant au carré et en sommant sur toutes les observations, le double produit s'annule (orthogonalité — c'est Pythagore en dimension finie) et on obtient :

> [!warning] Décomposition fondamentale (équation Pythagore)
> 
> $\underbrace{\sum_{j,i} (y_{ij} - \bar{y})^2}_{\text{SC}_{\text{tot}}} = \underbrace{\sum_j n_j (\bar{y}_j - \bar{y})^2}_{\text{SC}_{\text{inter}}} + \underbrace{\sum_{j,i} (y_{ij} - \bar{y}_j)^2}_{\text{SC}_{\text{intra}}}$
> 
> - **$\text{SC}_{\text{tot}}$** : variance totale de $Y$ autour de la moyenne globale
> - **$\text{SC}_{\text{inter}}$** : variance **expliquée par les groupes** (à quel point les moyennes de groupes diffèrent de la moyenne globale)
> - **$\text{SC}_{\text{intra}}$** : variance **résiduelle** (variabilité individuelle au sein de chaque groupe)

![[im5_anova_decomposition.png]]

**Figure 5.** Décomposition de la variance illustrée sur Auto avec les trois origines (USA, Europe, Japon). La ligne pointillée noire est la **moyenne globale** $\mu = \bar{y} \approx 23.45$ mpg. Les trois lignes pointillées colorées sont les **moyennes par groupe** $\bar{y}_j$. Pour un point d'exemple dans chaque groupe (rond bordé de noir), l'écart **total** à la moyenne globale se décompose en : (1) un écart **inter-groupes** $\tau_j = \bar{y}_j - \bar{y}$ (de la moyenne globale à la moyenne du groupe) et (2) un écart **intra-groupe** $\varepsilon_{ij} = y_{ij} - \bar{y}_j$ (de la moyenne du groupe à l'observation). En sommant les carrés sur toutes les observations, ces écarts s'agrègent en $\text{SC}_{\text{inter}}$ et $\text{SC}_{\text{intra}}$ — c'est cette décomposition qui fonde le test de Fisher.

> 💡 **L'intuition statistique.** Sous $H_0$ (toutes les moyennes égales), $\text{SC}_{\text{inter}}$ devrait être **petite** (les $\bar{y}_j$ ne diffèrent que par bruit d'échantillonnage). Sous $H_1$, $\text{SC}_{\text{inter}}$ devient **grande** par rapport à $\text{SC}_{\text{intra}}$. Le test compare ces deux quantités.

#### C.3 Statistique F et tableau ANOVA

Pour comparer $\text{SC}_{\text{inter}}$ et $\text{SC}_{\text{intra}}$, on les normalise chacune par leurs **degrés de liberté** pour obtenir des **variances estimées** (carrés moyens) :

- $\text{CM}_{\text{inter}} = \dfrac{\text{SC}_{\text{inter}}}{k-1}$ — estime la variance entre groupes ($k-1$ ddl car $k$ moyennes contraintes par leur somme)
- $\text{CM}_{\text{intra}} = \dfrac{\text{SC}_{\text{intra}}}{N-k}$ — estime la variance résiduelle ($N-k$ ddl car $N$ obs contraintes par les $k$ moyennes)

> [!warning] Statistique de Fisher
> 
> $F = \frac{\text{CM}_{\text{inter}}}{\text{CM}_{\text{intra}}} = \frac{\text{SC}_{\text{inter}} / (k-1)}{\text{SC}_{\text{intra}} / (N-k)}$
> 
> Sous $H_0$, $F \sim \mathcal{F}_{k-1,\, N-k}$ (loi de Fisher à $(k-1, N-k)$ degrés de liberté). On rejette $H_0$ si $F$ est trop grand.

> 💡 **F = ratio signal/bruit.** Numérateur = ce que les groupes expliquent (signal). Dénominateur = variance résiduelle (bruit). Sous $H_0$ les deux estiment la même chose ($\sigma^2$) → $F \approx 1$. Sous $H_1$ le signal explose → $F \gg 1$.

**Présentation classique : le tableau ANOVA.** En pratique, on rassemble tous ces calculs dans un tableau standardisé qu'on retrouve dans tous les logiciels (`anova()` en R, `statsmodels` en Python) :

| Source | Degrés de liberté | Somme des carrés | Carré moyen | Statistique F |
|---|:---:|:---:|:---:|:---:|
| **Inter-groupes** | $k - 1$ | $\text{SC}_{\text{inter}}$ | $\text{CM}_{\text{inter}} = \dfrac{\text{SC}_{\text{inter}}}{k-1}$ | $F = \dfrac{\text{CM}_{\text{inter}}}{\text{CM}_{\text{intra}}}$ |
| **Intra-groupes** | $N - k$ | $\text{SC}_{\text{intra}}$ | $\text{CM}_{\text{intra}} = \dfrac{\text{SC}_{\text{intra}}}{N-k}$ | |
| **Total** | $N - 1$ | $\text{SC}_{\text{tot}}$ | | |

#### C.4 Application sur Auto — Japon vs USA vs Europe

> [!warning] Test ANOVA sur les trois origines
> Sur le dataset complet ($N = 392$), on a trois groupes :
> - **USA** : $n_U = 245$, $\bar{y}_U = 20.03$
> - **Europe** : $n_E = 68$, $\bar{y}_E = 27.89$
> - **Japon** : $n_J = 79$, $\bar{y}_J = 30.45$
> 
> Avec une moyenne globale $\bar{y} \approx 23.45$, on calcule :
> 
> | Source | ddl | SC | CM | F |
> |---|:---:|:---:|:---:|:---:|
> | Inter-groupes (origin) | $2$ | $\sim 7\,500$ | $\sim 3\,750$ | $\mathbf{96.8}$ |
> | Intra-groupes (résidus) | $389$ | $\sim 15\,070$ | $\sim 38.7$ | |
> | Total | $391$ | $\sim 22\,570$ | | |
> 
> Avec $F \approx 96.8$ sous $\mathcal{F}_{2, 389}$, la p-value est de l'ordre de $10^{-35}$ — **on rejette $H_0$ très fortement**. Au moins une des trois origines a une mpg moyenne différente des autres.

#### C.5 Lien Fisher ↔ Student

Le T-test et le test de Fisher ne sont pas deux outils indépendants. Pour $k = 2$ groupes, l'ANOVA est **mathématiquement équivalente** au T-test à variances égales :

> [!warning] Équivalence Fisher-Student pour $k=2$
> 
> $F_{1, N-2} = (T_{N-2})^2$
> 
> où $T$ est la statistique du T-test à variances égales sur les deux groupes. Donc :
> 
> $\text{p-value (ANOVA, } k=2\text{)} = \text{p-value (T-test bilatéral)}$

> 💡 **Ce que ça veut dire.** L'ANOVA **généralise** le T-test à $k \ge 3$ groupes. Quand tu fais un T-test bilatéral sur deux groupes, tu fais en fait une mini-ANOVA sans le savoir. La statistique de Fisher est juste *le carré* de la statistique de Student dans ce cas — d'où l'équivalence des p-values.

> [!note]- Pourquoi $F = T^2$ exactement ?
> Le T-test bilatéral à variances égales rejette quand $|T|$ est grand, c'est-à-dire quand $T^2$ est grand. La distribution de $T^2$ sous $H_0$ est précisément $\mathcal{F}_{1, N-2}$ (par construction de la loi de Fisher : ratio de deux $\chi^2$ normalisées, où le numérateur $\chi^2_1$ est exactement $Z^2$). Donc rejeter $H_0$ via $|T| > q$ ou via $F > q^2$ est strictement équivalent.

#### C.6 Limites — au-delà de l'ANOVA

> ⚠️ **Limite 1 — pas d'identification des paires.** L'ANOVA dit qu'**au moins une** moyenne diffère, mais pas **lesquelles**. Pour identifier les paires significativement différentes, on enchaîne avec des **comparaisons multiples post-hoc** (Tukey HSD, Bonferroni) — qui font justement la correction du problème de multiple testing évoqué plus haut.

> ⚠️ **Limite 2 — hypothèse d'homoscédasticité.** L'ANOVA classique suppose **même variance** dans tous les groupes ($\sigma^2$ commun). Si les variances diffèrent fortement, on utilise l'**ANOVA de Welch** (équivalent du T-test de Welch en multi-groupes), ou des tests non-paramétriques comme **Kruskal-Wallis**.

> 💡 **Lien avec la régression linéaire.** L'ANOVA peut s'écrire comme une **régression linéaire sur des variables dummy** : on régresse $y$ sur des indicatrices de groupe ($\mathbb{1}_{\text{Europe}}, \mathbb{1}_{\text{Japon}}$ avec USA = référence) et on teste la nullité jointe de leurs coefficients via un **test F de modèles emboîtés**. Voir [[Régression Linéaire Multiple]] section III.E pour cette perspective et le test joint en cadre régression.

### D. ANCOVA (ANalysis of COVAriance) — ANOVA avec covariables

Dernier raffinement. Sur Auto, on a vu que les voitures japonaises ont une mpg moyenne plus élevée que les américaines. Mais on sait aussi que les voitures japonaises sont en moyenne **plus légères** — et le poids influence directement la mpg. **Une partie de la différence Japon/USA pourrait donc s'expliquer par le poids, pas par l'origine.**

L'ANCOVA répond à cette question : *"après avoir contrôlé pour le poids, reste-t-il une différence d'origine ?"*

> [!warning] Cadre ANCOVA
> Modèle de régression linéaire avec une variable catégorielle (le **facteur** : origine) et une variable continue (la **covariable** : poids) :
> $$Y_{i,j} = \mu + \tau_i + \beta \cdot W_{i,j} + \varepsilon_{i,j}$$
> où :
> - $Y_{i,j}$ est la mpg de la voiture $j$ du groupe $i$
> - $\tau_i$ est l'**effet de l'origine** $i$ (avec $\sum \tau_i = 0$)
> - $\beta$ est l'**effet du poids**
> - $\varepsilon_{i,j} \overset{iid}{\sim} \mathcal{N}(0, \sigma^2)$
> 
> On teste $H_0: \tau_1 = \tau_2 = \tau_3 = 0$ (pas d'effet de l'origine *après contrôle du poids*).

**Idée.** On régresse `mpg` sur `weight + origin`. Le test ANCOVA mesure si l'ajout de `origin` au modèle améliore significativement l'ajustement par rapport au modèle ne contenant que `weight`. C'est encore un **test F**, mais sur une comparaison de modèles emboîtés (nested).

> [!warning] Application sur Auto
> Modèle réduit : $\text{mpg} \sim \text{weight}$. Modèle complet : $\text{mpg} \sim \text{weight} + \text{origin}$.
> 
> En ajustant ces deux modèles par moindres carrés et en comparant les sommes des carrés résiduels, le test F donne typiquement $F \approx 8.3$ avec $(2, 388)$ degrés de liberté, p-value $\approx 0.0003$.
> 
> **Conclusion** : même après avoir contrôlé pour le poids, l'origine a un effet **résiduel significatif** sur la mpg. La différence entre Japon, USA, et Europe ne s'explique pas uniquement par les différences de poids — il y a aussi un effet "design / efficacité moteur" propre à l'origine.

> 💡 **Pourquoi c'est utile.** L'ANCOVA permet d'isoler l'effet d'un facteur catégoriel **toutes choses égales par ailleurs** sur une covariable continue. C'est très proche d'une régression linéaire avec interactions — d'ailleurs en pratique on utilise souvent directement la régression (notes [[Régression Linéaire Multiple]]). L'ANCOVA est juste le **vocabulaire historique** pour ce type d'analyse.
