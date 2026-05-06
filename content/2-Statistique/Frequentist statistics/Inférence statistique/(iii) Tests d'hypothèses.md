---
title: Tests d'hypothèses
order: 3
---

# Tests d'hypothèses

> Cette note couvre les tests d'hypothèses statistiques : pourquoi en faire, comment les construire, et les principaux tests paramétriques (Z-test, T-test, ANOVA, ANCOVA). Le fil rouge est le dataset `Auto` (ISLR), qu'on retrouve aussi dans la note [[Régression Linéaire]].

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

$H_0 : \mu_{\text{Japon}} = \mu_{\text{USA}} \qquad \text{vs} \qquad H_1 : \mu_{\text{Japon}} \neq \mu_{\text{USA}}$

**Comment on décide ?** On construit une **statistique de test** $T$ — la version empirique de ce qu'on veut tester. Sur Auto, naturellement :

$T = \bar{Y}_{\text{Japon}} - \bar{Y}_{\text{USA}}$

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

### A. Formulation statistique

On considère un échantillon $X_1, \ldots, X_n$ de variables aléatoires i.i.d. et un modèle statistique $(E, (\mathbb{P}_\theta)_{\theta \in \Theta})$. Soient $\Theta_0$ et $\Theta_1$ deux sous-ensembles disjoints de $\Theta$. On considère les deux hypothèses :

$$
\begin{aligned}
&H_0: \theta \in \Theta_0 \\
&H_1: \theta \in \Theta_1
\end{aligned}
$$

$H_0$ est l'**hypothèse nulle**, $H_1$ est l'**hypothèse alternative**.

![[stat_form.png|467]]

### B. Test, région de rejet, région d'acceptation

> [!warning] Définition — Test
> Un test est une statistique $\psi \in \{0, 1\}$ telle que :
> - Si $\psi = 0$, $H_0$ n'est pas rejetée
> - Si $\psi = 1$, $H_0$ est rejetée, on conclut $H_1$

|  | $H_0$ vraie | $H_1$ vraie |
|---|:---:|:---:|
| $\psi = 0$ | ok | Erreur de type 2 |
| $\psi = 1$ | Erreur de type 1 | ok |

Sur les cellules de gauche, on écrit en général avec $\psi$ plutôt que $H_0$ et $H_1$ parce qu'on ne dit pas que $H_1$ est vraie.

> [!warning] Région de rejet d'un test $\psi$
> $$R_\psi = \{x \in E^n : \psi(x) = 1\}$$
> 
> C'est l'ensemble des points qui sont dans la région de rejet.

> [!warning] Région d'acceptation
> Ce sont les points qui ne sont pas dans la région de rejet.

### C. Erreurs de type 1 et type 2

> [!warning] Erreur de type 1 d'un test $\psi$ (rejeter $H_0$ alors qu'elle est vraie)
> $$
> \begin{aligned}
> \alpha_\psi : \Theta_0 &\rightarrow \mathbb{R} \\
> \theta &\mapsto \mathbb{P}_\theta[\psi = 1]
> \end{aligned}
> $$

> [!example] Exemple — erreur de type 1
> Soit $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, 1)$ où $\mu$ est un paramètre inconnu. On veut répondre à : est-ce que $\mu = 0$ ? On construit :
> 
> $$
> \begin{aligned}
> &H_0: \mu = 0 \\
> &H_1: \mu \neq 0
> \end{aligned}
> $$
> 
> et on choisit le seuil $C = q_{0.05}$. La probabilité d'erreur de type 1 $\alpha_\psi$ est :
> 
> $$\alpha_\psi(0) = \mathbb{P}_0(\psi_C = 1) = \mathbb{P}_0\left(\sqrt{n}|\overline{X}_n| > q_{0.05}\right) = 0.1$$

> [!warning] Erreur de type 2 d'un test $\psi$ (ne pas rejeter $H_0$ alors que $H_1$ est vraie)
> $$
> \begin{aligned}
> \beta_\psi : \Theta_1 &\rightarrow \mathbb{R} \\
> \theta &\mapsto \mathbb{P}_\theta[\psi = 0]
> \end{aligned}
> $$

> [!example] Exemple — interprétation des erreurs
> Soit
> 
> $$
> \begin{aligned}
> &H_0: p \in \Theta_0 = \{1/2\} \\
> &H_1: p \in \Theta_1 = \{3/4\}
> \end{aligned}
> $$
> 
> - **Que représente $\alpha_\psi(1/2)$ ?** La probabilité de rejeter $p = 1/2$ en faveur de $p = 3/4$ alors qu'en réalité $p = 1/2$. Si $\psi = 1$ alors on rejette l'hypothèse nulle $p \in \Theta_0 = \{1/2\}$. Donc $\alpha_\psi(1/2) = \mathbb{P}_{1/2}(\psi = 1)$ est la probabilité de rejeter $p^* \in \Theta_0 = \{1/2\}$ en faveur de $p^* \in \Theta_1 = \{3/4\}$ alors qu'en fait $p^* \in \Theta_0$.
> 
> - **Que représente $\beta_\psi(3/4)$ ?** La probabilité de ne pas rejeter $p = 1/2$ en faveur de $p = 3/4$ alors qu'en réalité $p = 3/4$. Si $\psi = 0$ alors on ne rejette pas l'hypothèse nulle $p^* \in \Theta_0 = \{1/2\}$ en faveur de l'alternative $p^* \in \Theta_1 = \{3/4\}$. Donc $\beta_\psi(3/4) = \mathbb{P}_{3/4}(\psi = 0)$.

### D. Puissance d'un test

> [!warning] Puissance d'un test $\psi$
> Probabilité de rejeter quand on devrait vraiment rejeter. Quand cette quantité est grande, on dit qu'on a un **test puissant**.
> 
> $$\pi_\psi = \inf_{\theta \in \Theta_1} (1 - \beta_\psi(\theta))$$

On prend $\mu < 30$ pour $\Theta_0$. On dessine la probabilité sous $\mu$ qu'un test rejette, qui est exactement $1 - \beta_\psi(\theta)$. On veut que cette probabilité de rejet soit petite pour $\Theta_0$ et grande pour $\Theta_1$.

![[power-test.png]]

### E. Niveau d'un test

> [!warning] Niveau d'un test
> Un test $\psi$ a **niveau $\alpha$** (penser $\alpha = 5\%, 1\%, \ldots$) si :
> 
> $$\alpha_\psi(\theta) \le \alpha, \quad \forall \theta \in \Theta_0$$

Interprétation du niveau :
- Le niveau d'un test est une borne supérieure sur l'erreur de type 1
- Le niveau donne une borne supérieure sur la probabilité d'erreur dans le pire cas sous l'hypothèse nulle

> [!warning] Niveau asymptotique
> Un test $\psi$ a **niveau asymptotique $\alpha$** si :
> 
> $$\lim_{n \to \infty} \alpha_{\psi_n}(\theta) \le \alpha, \quad \forall \theta \in \Theta_0$$

En général, un test a la forme :

$$\psi = \mathbb{1}\{T_n > c\}$$

pour une statistique $T_n$ et un seuil $c \in \mathbb{R}$. $T_n$ est appelée la **statistique de test**. La région de rejet est :

$$R_\psi = \{T_n > c\}$$

### F. p-value

> [!warning] Définition — p-value
> La p-value (asymptotique) d'un test $\psi_\alpha$ est le plus petit niveau (asymptotique) $\alpha$ auquel $\psi_\alpha$ rejette $H_0$. Elle est aléatoire, elle dépend de l'échantillon.
> 
> **Règle d'or** : $p\text{-value} \le \alpha \iff H_0$ est rejetée par $\psi_\alpha$ au niveau (asymptotique) $\alpha$.

Plus la p-value est petite, plus on peut rejeter $H_0$ avec confiance.

> 💡 **Intuition.** La p-value est la probabilité d'observer une statistique de test au moins aussi extrême que celle observée, en supposant $H_0$ vraie. On suppose $H_0$ vraie parce que dans le cadre des tests d'hypothèses, on suppose $H_0$ vraie jusqu'à ce qu'on ait des preuves contre elle.

### G. Vocabulaire — un échantillon vs deux échantillons

> [!warning] One-sample test
> Un test où un paramètre inconnu $\mu$ est comparé à une valeur de référence connue. Exemple : comparer la moyenne $\mu$ d'aujourd'hui à la moyenne historique connue de 5.5.

> [!warning] Two-sample test
> Un test où deux paramètres inconnus sont comparés entre eux. Exemple : comparer $\mu_{\text{drug}}$ inconnu à $\mu_{\text{control}}$ inconnu pour quantifier l'effet d'un médicament.

- Si $H_1: \theta \neq \theta_0$ : test **bilatéral** (two-sided)
- Si $H_1: \theta > \theta_0$ ou $H_1: \theta < \theta_0$ : test **unilatéral** (one-sided)

---

## II. Hypothèse simple vs composite, règle de décision

> [!warning] Définition — Hypothèse
> Un énoncé sur un paramètre de population.
> - **Hypothèse simple** : le paramètre est égal à un point unique (ex : $\mu = 10$)
> - **Hypothèse composite** : le paramètre est dans un intervalle (ex : $\mu > 10$)

> [!example] Exemple — règle de décision
> $X \sim \mathcal{N}(\mu, 36)$ et $\mu$ vaut soit $50$ soit $55$.
> 
> $$H_0: \mu = 50 \quad H_1: \mu = 55$$
> 
> Pour décider de rejeter $H_0$ ou non, on utilise une règle :
> - Si $(X_1, \ldots, X_n) \in C$, on rejette $H_0$ en faveur de $H_1$
> - Si $(X_1, \ldots, X_n) \in C'$, on ne rejette pas $H_0$
> 
> où l'espace d'échantillonnage de $(X_1, \ldots, X_n)$ est partitionné en $C$ et $C'$.
> 
> Concrètement, on peut prendre :
> - $C = \{(X_1, \ldots, X_n) : \bar{X} \ge 53\}$ (cohérent car sous $H_0$, $\mu = 50$)
> - $C' = \{(X_1, \ldots, X_n) : \bar{X} < 53\}$
> 
> $C$ et $C'$ ne se chevauchent pas.

### Calcul des erreurs

![[im6 (1).png]]

$$P(\text{erreur de type 1}) = \text{niveau de signification} = \alpha$$
$$P(\text{erreur de type 2}) = \beta$$

Règle de décision (suite de l'exemple) :
- Rejeter si $\bar{X} \ge 53$
- Ne pas rejeter si $\bar{X} < 53$

> [!example] Calcul de $\alpha$ et $\beta$
> Soit $X \sim \mathcal{N}(\mu, 36)$, $H_0: \mu = 50$ vs $H_1: \mu = 55$. On prend $n = 16$. Soit $Z \sim \mathcal{N}(0, 1)$.
> 
> **Calcul de $\alpha$** :
> 
> $$
> \begin{aligned}
> \alpha &= P(\text{rejeter } H_0 \mid H_0 \text{ vraie}) = P(\bar{X} \ge 53 \mid \mu = 50) \\
> &= P\left(\frac{\bar{X} - 50}{6/4} \ge \frac{53 - 50}{6/4} \mid \mu = 50\right) \\
> &= P(Z > 2) = 0.0228
> \end{aligned}
> $$
> 
> où $6/4 = \sqrt{36}/\sqrt{16}$. La valeur de $\alpha$ nous dit qu'en réalité, si $H_0$ est vraie ($\mu = 50$), alors environ $2.28\%$ du temps on rejettera l'hypothèse nulle à tort.
> 
> **Calcul de $\beta$** :
> 
> $$
> \begin{aligned}
> \beta &= P(\text{ne pas rejeter } H_0 \mid H_0 \text{ fausse}) = P(\bar{X} < 53 \mid \mu = 55) \\
> &= P\left(\frac{\bar{X} - 55}{6/4} < \frac{53 - 55}{6/4} \mid \mu = 55\right) = P(Z < -4/3) \\
> &= 0.0913
> \end{aligned}
> $$
> 
> Si on prend un échantillon de taille 16 dix millions de fois, on ne rejetterait pas $H_0$ à tort dans environ $9.13\%$ des cas.

> 💡 **Remarque.** Dans la vraie vie, on ne sait pas si $H_0$ ou $H_1$ est correcte. Tout ce qu'on peut faire, c'est prendre la décision de rejeter ou non.

![[im7.png]]

---

## III. Calcul de p-value

> [!warning] Définition — p-value
> Probabilité d'obtenir une statistique de test aussi extrême ou plus extrême que celle observée, en supposant $H_0$ vraie. Les p-values mesurent la **preuve contre $H_0$**. Plus la p-value est petite, plus la preuve contre $H_0$ est forte. On rejette $H_0$ si la p-value est "assez petite" — plus petite qu'un seuil prédéterminé appelé **niveau de signification $\alpha$**.

> [!example] Exemple — une moyenne, $\sigma^2$ connue
> $X \sim \mathcal{N}(\mu, \sigma^2)$, $\mu$ inconnue, $\sigma^2 = 100$.
> 
> $$H_0: \mu = 60 \quad H_1: \mu > 60$$
> 
> Avec $n = 52$ et $\bar{X} = 62.75$. La p-value est :
> 
> $$\text{p-val} = P(\bar{X} \ge 62.75 \mid \mu = 60)$$
> 
> Or $\bar{X} \sim \mathcal{N}(60, 100/52)$ sous $H_0$, donc :
> 
> $$
> \begin{aligned}
> \text{p-val} &= P\left(\frac{\bar{X} - 60}{10/\sqrt{52}} \ge \frac{62.75 - 60}{10/\sqrt{52}} \mid \mu = 60\right) \\
> &= P\left(Z \ge \frac{62.75 - 60}{10/\sqrt{52}}\right) = 0.0237
> \end{aligned}
> $$

![[im8.png]]

### Trois cas selon l'hypothèse alternative

**(i) Test unilatéral à droite.** On dessine la gaussienne dont la moyenne est celle de $H_0$. Pour savoir quelle partie hachurer, on regarde l'hypothèse alternative.

$$H_0: \mu = 60 \quad H_1: \mu > 60$$

![[im9.png|307]]

**(ii) Test unilatéral à gauche.** Si on prend un échantillon et qu'on obtient $\bar{X} = 58.1$, on le dessine, et la p-value est la probabilité d'obtenir une statistique de test plus extrême sous $H_0$ (avec $\mu = 60$).

$$H_0: \mu = 60 \quad H_1: \mu < 60$$

![[im10 (1).png]]

**(iii) Test bilatéral.** Soit $\bar{X} = 58.1$. On dessine la moyenne sous $H_0$ qui est $60$. Sur l'image de gauche, on regarde la queue inférieure — mais c'est seulement la moitié de l'histoire car $\mu \neq 60$ est une alternative bilatérale. On calcule cette probabilité et on la **double**. Autre façon de voir : image de droite.

$$H_0: \mu = 60 \quad H_1: \mu \neq 60$$

Comme $\text{p-val} < 0.05$, on rejette $H_0$ au niveau $\alpha = 0.05$.

![[im11.png]]

---

## IV. Test pour la moyenne avec $\sigma^2$ inconnue

Rappel : si $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ avec $\mu$ et $\sigma^2$ inconnues, alors :

$$\frac{\bar{X} - \mu}{s/\sqrt{n}} \sim T_{n-1}$$

Précédemment, notre statistique de test était $\bar{X}$ avec une distribution d'échantillonnage $\mathcal{N}(\mu, \sigma^2/n)$ sous $H_0$, mais $\sigma^2$ était connue. De manière équivalente, la statistique de test aurait pu être :

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$$

et sa distribution sous $H_0$ était $\mathcal{N}(0, 1)$. Ces deux statistiques de test produisent des résultats équivalents (mêmes p-values).

Maintenant que $\sigma^2$ est inconnue, notre statistique de test pour $H_0: \mu = \mu_0$ est :

$$\frac{\bar{X} - \mu_0}{s/\sqrt{n}}$$

et sa distribution d'échantillonnage sous $H_0$ est $T_{n-1}$.

> [!example] Exemple — Pizza Hut
> Pizza Hut prétend que chaque part de pizza pepperoni a en moyenne 4 pepperonis :
> 
> $$H_0: \mu = 4 \quad H_1: \mu \neq 4$$
> 
> On prend un échantillon de 9 parts : $\bar{X} = 4.3$, $s = 1.2$. La statistique de test est :
> 
> $$\frac{4.3 - 4.0}{1.2/\sqrt{9}} = 0.75$$
> 
> et la distribution de $(\bar{X} - 4)/(s/\sqrt{n})$ est $T_8$ sous $H_0$. La p-value est :
> 
> $$2 \times P(T_8 > 0.75) = 0.4747$$
> 
> **Pourquoi multiplier par 2 ?** Parce que le test est bilatéral, donc il faut multiplier par 2 pour tenir compte des deux côtés.

![[im12 (1).png]]

---

## V. Z-test : tests asymptotiques

### A. Test bilatéral

> [!example] Expérience de Bernoulli
> Soit $X_1, \ldots, X_n \overset{iid}{\sim} \text{Ber}(p)$, pour $p \in [0, 1]$ inconnu. On veut tester :
> 
> $$H_0: p = 1/2 \quad \text{vs} \quad H_1: p \neq 1/2$$
> 
> avec niveau asymptotique $\alpha \in [0, 1]$.

Soit la statistique de test :

$$T_n = \sqrt{n} \frac{|\hat{p}_n - p|}{\sqrt{p(1-p)}}$$

On veut calculer l'erreur de type 1 :

$$\alpha(\theta) = \mathbb{P}_\theta[R_\psi] = \mathbb{P}_\theta[\psi = 1], \quad \forall \theta \in \Theta_0$$

Rappel : $\psi = 1$ signifie que $H_0$ est rejetée. $R_\psi$ est la région où tous les points sont rejetés. L'erreur de type 1 est la probabilité de rejeter $H_0$ alors qu'elle est vraie.

On remplace $\theta$ par le paramètre d'intérêt $p$ sous $H_0$ :

$$\mathbb{P}_p[T_n \in \mathcal{R}] = \alpha(p)$$

On veut contrôler ceci uniquement quand $p \in \Theta_0 = \{1/2\}$ :

$$\mathbb{P}_{1/2}\left[\sqrt{n}\frac{|\hat{p}_n - 0.5|}{\sqrt{0.5(1 - 0.5)}} \in \mathcal{R}\right] = \alpha(1/2) = \alpha$$

On peut écrire $\alpha(1/2) = \alpha$ car :

$$\sup_{\theta \in \Theta_0} \alpha(\theta) \le \alpha$$

et ici $\alpha$ ne prend qu'une seule valeur, donc $\sup_{\theta \in \Theta_0} \alpha(\theta) = \alpha$.

Il reste à définir la constante $C$. On standardise :

$$
\begin{aligned}
&= 2(1 - \Phi(S)) = \alpha \\
&\Rightarrow \Phi(S) = 1 - \frac{\alpha}{2} \\
&\Rightarrow S = q_{\alpha/2} \text{, le } (1 - \alpha/2)\text{-quantile de } \mathcal{N}(0, 1)
\end{aligned}
$$

D'où :

$$\mathbb{P}_{1/2}\left[\sqrt{n}\frac{|\hat{p}_n - 1/2|}{1/2} > q_{\alpha/2}\right] \xrightarrow[n \to \infty]{} \alpha$$

Par le **TCL**, sous $H_0$ :

$$T_n = \sqrt{n}\frac{|\hat{p}_n - 0.5|}{\sqrt{0.5(1-0.5)}} \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, 1)$$

On spécifie le test :

$$\psi = \mathbb{1}\left\{\sqrt{n}\frac{|\hat{p}_n - 1/2|}{\sqrt{0.5(1-0.5)}} > C\right\}$$

> [!example] Exemple 1 — pièce truquée
> Une pièce est lancée 30 fois et on obtient pile 13 fois. La pièce est-elle significativement biaisée ? Soit $n = 30$, $X_1, \ldots, X_n \overset{iid}{\sim} \text{Ber}(p)$ et $\hat{p}_n = 13/30 \approx 0.43$, $\alpha = 5\%$.
> 
> $$
> \begin{aligned}
> &H_0: p = 1/2, \quad \Theta_0 = \{1/2\} \\
> &H_1: p \neq 1/2, \quad \Theta_1 = (0, 1) \setminus \{1/2\}
> \end{aligned}
> $$
> 
> ![[quantile.png|323]]
> 
> Avec $\alpha = 5\%$, on a $q_{\alpha/2} = 1.96$.
> 
> $$\sqrt{n}|\bar{X}_n - 1/2| \cdot 2 = 0.77 < 1.96$$
> 
> **Interprétation** : on compare la vraie valeur de $p$ (à 0 sur la gaussienne standard) avec la statistique de test (à 0.77 écart-types du centre). Plus la valeur de la statistique de test est grande, plus on rejette $H_0$.
> 
> $H_0$ n'est **pas rejetée** au niveau asymptotique $5\%$ par le test $\psi_{5\%}$.
> 
> ![[standardized.png|320]]
> 
> Calcul de la p-value : on regarde la Z-table pour $2.482$, on obtient $0.00226$.
> 
> $$
> \begin{aligned}
> \eta &= \mathbb{P}[|Z| > T_n] \\
> &= 2 \cdot \mathbb{P}[|Z| > 2.842] \\
> &= 0.00452
> \end{aligned}
> $$
> 
> **Conclusion** : on est confiant qu'on ne rejette pas l'hypothèse nulle, et la p-value confirme cela.
> 
> ![[z-table.png]]

### B. Test unilatéral

> [!example] Exemple 2 — sondage Youtube
> Selon un sondage de 2017 sur 4 971 Américains, $32\%$ déclarent obtenir au moins une partie de leur info sur Youtube. Peut-on conclure qu'au plus un tiers des Américains s'informent sur Youtube ? Soit $n = 4971$, $X_1, \ldots, X_n \overset{iid}{\sim} \text{Ber}(p)$ et $\bar{X}_n = 0.32$ :
> 
> $$
> \begin{aligned}
> &H_0: p \ge 0.33 \\
> &H_1: p < 0.33
> \end{aligned}
> $$

On rejette si :

$$\mathbb{P}_p\left[\sqrt{n}\frac{\hat{p} - p}{\sqrt{p(1-p)}} < C\right] \xrightarrow[n \to \infty]{} \alpha(p)$$

Et :

$$\sup_{\theta \in \Theta_0} \alpha(\theta) \le \alpha$$

Mais $p \in \Theta_0 = [0.33, 1]$. Différents $p$ donnent différents $\alpha(p)$. On note d'abord que :

$$\mathbb{P}_p\left[\sqrt{n}\frac{\hat{p} - p}{\sqrt{p(1-p)}} < q_{1-\alpha}\right] \xrightarrow[n \to \infty]{} \alpha(p)$$

Par symétrie : $q_{1-\alpha} = -q_\alpha$.

On doit choisir un $p$ dans $\Theta_0$. Si on prend $0.46$, l'expression n'est valide que pour $p = 0.46$. Le pire cas est $p = 0.33$ (à la frontière).

$$f_n(p_0) = \sup_p \mathbb{P}_p\left[\sqrt{n}\frac{\hat{p} - p}{\sqrt{p(1-p)}} < -q_\alpha\right] \xrightarrow[n \to \infty]{} \alpha(p)$$

On prend $p_0 = 0.33$ (la valeur frontière).

> 📌 **TODO** : ajouter un exemple où $\Theta_0 = [0.5, 0.6]$. C'est galère car il faut calculer les deux valeurs et regarder laquelle donne la plus grande erreur de type 1.

Statistique de test :

$$\sqrt{n}\frac{\hat{X}_n - 0.33}{\sqrt{0.33(1-0.33)}} = -1.50 < -1.645$$

On ne rejette pas.

---

## VI. T-test : tests non-asymptotiques

### A. Cas à deux échantillons (gaussien, variances connues)

> [!example] Exemple — essai clinique cholestérol
> - Soit $\Delta_d > 0$ la baisse attendue de LDL (en mg/dL) pour un patient sous médicament
> - Soit $\Delta_c \ge 0$ la baisse attendue pour un patient sous placebo
> 
> On veut savoir si $\Delta_d > \Delta_c$. On observe deux échantillons indépendants :
> - $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\Delta_d, \sigma_d^2)$ groupe test
> - $Y_1, \ldots, Y_m \overset{iid}{\sim} \mathcal{N}(\Delta_c, \sigma_c^2)$ groupe contrôle

**Hypothèses** :

$$H_0: \Delta_c = \Delta_d \quad \text{vs} \quad H_1: \Delta_d > \Delta_c$$

Comme les données sont gaussiennes par hypothèse, on n'a pas besoin du TCL. On a :

$$\bar{X}_n \sim \mathcal{N}\left(\Delta_d, \frac{\sigma_d^2}{n}\right) \quad \text{et} \quad \bar{Y}_m \sim \mathcal{N}\left(\Delta_c, \frac{\sigma_c^2}{m}\right)$$

Donc :

$$\frac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\frac{\sigma_d^2}{n} + \frac{\sigma_c^2}{m}}} \sim \mathcal{N}(0, 1)$$

Supposons $m = cn$ et $n \to \infty$. Par le lemme de Slutsky :

$$\frac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\frac{\hat{\sigma}_d^2}{n} + \frac{\hat{\sigma}_c^2}{m}}} \xrightarrow[n \to \infty]{(d)} \mathcal{N}(0, 1)$$

où :

$$\hat{\sigma}_d^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X}_n)^2 \quad \text{et} \quad \hat{\sigma}_c^2 = \frac{1}{m-1}\sum_{i=1}^m (Y_i - \bar{Y}_m)^2$$

Test au niveau asymptotique $\alpha$ :

$$R_\psi = \left\{\frac{\bar{X}_n - \bar{Y}_m}{\sqrt{\frac{\hat{\sigma}_d^2}{n} + \frac{\hat{\sigma}_c^2}{m}}} > q_\alpha\right\}$$

C'est un test **unilatéral à deux échantillons**.

> [!example] Application numérique
> $n = 70$, $m = 50$, $\bar{X}_n = 156.4$, $\bar{Y}_m = 132.7$, $\hat{\sigma}_d^2 = 5198.4$, $\hat{\sigma}_c^2 = 3867.0$.
> 
> $$\frac{156.4 - 132.7}{\sqrt{\frac{5198.4}{70} + \frac{3867.0}{50}}} = 1.57$$
> 
> Comme $q_{5\%} = 1.645$, on **ne rejette pas** $H_0$.
> 
> p-value : $\mathbb{P}(\mathcal{N}(0, 1) > 1.57) = 0.0582$.

### B. Motivation pour le T-test — petits échantillons

Que faire si $n = 20$ et $m = 12$ ? On ne peut pas appliquer Slutsky de manière réaliste. On avait besoin de Slutsky pour trouver la distribution (asymptotique) de quantités du type :

$$\frac{\bar{X}_n - \mu}{\sqrt{\sigma^2}}$$

quand $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$. Il s'avère que cette distribution ne dépend pas de $\mu$ ni $\sigma$, donc on peut calculer ses quantiles.

> [!warning] Définition — Distribution $\chi^2$
> Pour un entier positif $d$, la distribution $\chi^2$ à $d$ degrés de liberté est la loi de la variable aléatoire $Z_1^2 + Z_2^2 + \cdots + Z_d^2$, où $Z_1, \ldots, Z_d \overset{iid}{\sim} \mathcal{N}(0, 1)$.

> [!note]- Propriétés du $\chi^2$
> Si $V \sim \chi_k^2$, alors :
> - $\mathbb{E}[V] = E[Z_1^2] + \cdots + \mathbb{E}[Z_d^2] = d$
> - $\text{Var}[V] = \text{Var}[Z_1^2] + \cdots + \text{Var}[Z_d^2] = 2d$

> [!warning] Définition — Distribution de Student
> Pour un entier positif $d$, la distribution de Student à $d$ degrés de liberté (notée $t_d$) est la loi de la variable aléatoire $\frac{Z}{\sqrt{V/d}}$, où $Z \sim \mathcal{N}(0, 1)$, $V \sim \chi_d^2$, et $Z \perp V$ ($Z$ indépendante de $V$).

### C. Test de Student à un échantillon

> [!example] Un échantillon, bilatéral
> Soit $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$ avec $\mu$ et $\sigma^2$ inconnus. On veut tester $H_0: \mu = 0$ vs $H_1: \mu \neq 0$.

On construit la statistique de test :

$$T_n = \frac{\bar{X}_n}{\sqrt{\tilde{S}_n / n}} = \frac{\sqrt{n}\frac{\bar{X}_n - \mu}{\sigma}}{\sqrt{\tilde{S}_n / \sigma^2}}$$

Comme $\sqrt{n}\bar{X}_n / \sigma \sim \mathcal{N}(0, 1)$ (sous $H_0$) et $\tilde{S}_n / \sigma^2 \sim \chi^2_{n-1}/(n-1)$ sont indépendants par le **théorème de Cochran**, on a :

$$T_n \sim t_{n-1}$$

Test de Student au niveau (non-asymptotique) $\alpha \in (0, 1)$ :

$$\psi_\alpha = \mathbb{1}\{|T_n| > q_{\alpha/2}\}$$

où $q_{\alpha/2}$ est le $(1 - \alpha/2)$-quantile de $t_{n-1}$.

### D. Test de Welch — deux échantillons, petits effectifs

> [!example] Retour à l'exemple cholestérol — petits échantillons
> Que se passe-t-il pour de petits échantillons ? On veut connaître la distribution de :
> 
> $$\frac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\frac{\hat{\sigma}_d^2}{n} + \frac{\hat{\sigma}_c^2}{m}}}$$

On a approximativement :

$$\frac{\bar{X}_n - \bar{Y}_m - (\Delta_d - \Delta_c)}{\sqrt{\frac{\hat{\sigma}_d^2}{n} + \frac{\hat{\sigma}_c^2}{m}}} \sim t_N$$

où :

$$N = \frac{(\hat{\sigma}_d^2/n + \hat{\sigma}_c^2/m)^2}{\frac{\hat{\sigma}_d^4}{n^2(n-1)} + \frac{\hat{\sigma}_c^4}{m^2(m-1)}} \ge \min(n, m)$$

(formule de **Welch-Satterthwaite**)

> [!example] Application — $n = 70$, $m = 50$
> $\bar{X}_n = 156.4$, $\bar{Y}_m = 132.7$, $\hat{\sigma}_d^2 = 5198.4$, $\hat{\sigma}_c^2 = 3867.0$.
> 
> $$\frac{156.4 - 132.7}{\sqrt{\frac{5198.4}{70} + \frac{3867.0}{50}}} = 1.57$$
> 
> Avec la formule courte $N = \min(n, m) = 50$ : $q_{5\%} = 1.68$ et p-value $= \mathbb{P}[t_{50} > 1.57] = 0.0614$.
> 
> Avec la formule W-S :
> 
> $$N = \frac{\left(\frac{5198.4}{70} + \frac{3867.0}{50}\right)^2}{\frac{5198.4^2}{70^2(70-1)} + \frac{3867.0^2}{50^2(50-1)}} = 113.78$$
> 
> Arrondi à 113. p-value $= \mathbb{P}[t_{113} > 1.57] = 0.0596$.

> [!example] Application — $n = 20$, $m = 12$
> Mêmes moyennes et variances qu'avant.
> 
> $$\frac{156.4 - 132.7}{\sqrt{\frac{5198.4}{20} + \frac{3867.0}{12}}} = 0.982$$
> 
> Avec $N = \min(n, m) = 12$ : $q_{5\%} = 1.78$ et p-value $= \mathbb{P}[t_{12} > 0.982] = 17.27\%$.
> 
> Avec W-S :
> 
> $$N = \frac{\left(\frac{5198.4}{20} + \frac{3867.0}{12}\right)^2}{\frac{5198.4^2}{20^2(20-1)} + \frac{3867.0^2}{12^2(12-1)}} = 26.07$$
> 
> Arrondi à 26. p-value $= \mathbb{P}[t_{26} > 0.982] = 16.76\%$.

### E. Discussion — avantages et inconvénients du T-test

> 💡 **Avantage du test de Student.** Non-asymptotique : peut être utilisé sur de petits échantillons, et reste valide pour les grands échantillons.
> 
> ⚠️ **Inconvénient.** Repose sur l'hypothèse que l'échantillon est gaussien (on verra plus tard comment tester cette hypothèse).
