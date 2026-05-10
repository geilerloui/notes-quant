---
title: Échantillonnage
---
# Échantillonnage

> Ce fichier couvre les méthodes d'**échantillonnage** (sampling) : comment générer numériquement des réalisations d'une loi de probabilité, et comment s'en servir pour calculer des espérances qu'on ne sait pas calculer analytiquement. C'est le socle commun à la simulation stochastique, à l'intégration de Monte Carlo, et à l'inférence bayésienne (où le posterior est en général inaccessible analytiquement et nécessite ces outils ou leur extension MCMC).

## I. Le problème fondamental

**Cadre général.** Soit $\mathbf{z}$ une variable aléatoire à valeurs dans $\mathbb{R}^d$ de densité $p(\mathbf{z})$, et $f : \mathbb{R}^d \to \mathbb{R}$ une fonction. La quantité d'intérêt est l'espérance

$$\mathbb{E}_p[f] = \int f(\mathbf{z}) \, p(\mathbf{z}) \, \mathrm{d}\mathbf{z}.$$
Dans la grande majorité des cas pratiques, cette intégrale n'est **pas calculable analytiquement** : soit la densité $p$ est trop compliquée (posterior bayésien, modèle hiérarchique), soit la fonction $f$ l'est, soit la dimension $d$ est trop grande pour les méthodes déterministes (quadrature, méthode des rectangles).

![[ech_pf_setup.png]]
**Figure 1.** Configuration typique : une densité $p(z)$ concentrée sur une partie de l'espace, et une fonction $f(z)$ qu'on doit moyenner contre cette densité. Le but est de calculer $\mathbb{E}_p[f]$ sans résoudre l'intégrale.

**L'idée de Monte Carlo.** Si l'on dispose de $L$ tirages indépendants $\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(L)} \sim p$, on approche l'espérance par la moyenne empirique

$$\widehat{f} = \frac{1}{L} \sum_{l=1}^{L} f(\mathbf{z}^{(l)}).$$
> 💡 **L'idée en une phrase.** Calculer une intégrale, c'est calculer une espérance ; calculer une espérance, c'est faire la moyenne d'une fonction sur des tirages aléatoires. Tout le reste — Inverse Transform, Rejection, Importance Sampling — ne sert qu'à savoir **comment** produire ces tirages quand on ne sait pas le faire directement.

**Propriétés de l'estimateur.** Tant que les tirages sont vraiment iid selon $p$ :

1. **Sans biais :** $\mathbb{E}[\widehat{f}] = \mathbb{E}_p[f]$, par linéarité de l'espérance.
2. **Variance :** $\mathrm{Var}[\widehat{f}] = \frac{1}{L} \, \mathrm{Var}_p[f]$, qui décroît en $1/L$.
3. **Indépendance de la dimension.** La variance de $\widehat{f}$ ne dépend **pas** explicitement de $d$, seulement de la variance de $f$ sous $p$. C'est ce qui rend Monte Carlo redoutable en grande dimension, là où les méthodes de quadrature classiques explosent.

> [!warning] Le théorème central limite (TCL, central limit theorem) en pratique
> Pour $L$ grand, l'estimateur est approximativement gaussien :
>
> $$\widehat{f} \;\sim\; \mathcal{N}\!\left(\mathbb{E}_p[f], \; \frac{\mathrm{Var}_p[f]}{L}\right).$$
>
> L'**erreur standard** (standard error) $\mathrm{SE} = \sqrt{\mathrm{Var}_p[f] / L}$ se lit directement comme la largeur de l'intervalle de confiance : on est à environ $\pm 2 \, \mathrm{SE}$ de la vraie valeur avec probabilité $\approx 95\%$. C'est l'outil canonique pour évaluer la qualité d'une estimation de Monte Carlo.

**Deux difficultés pratiques.** En théorie l'algorithme est trivial, mais deux problèmes apparaissent :

- **Tirages corrélés.** Les méthodes pratiques (rejection, MCMC) produisent souvent des $\mathbf{z}^{(l)}$ qui ne sont pas indépendants. La taille effective d'échantillon (effective sample size, ESS) est alors plus petite que $L$, et la variance de l'estimateur est plus grande que $\mathrm{Var}_p[f] / L$.
- **Mauvais alignement entre $f$ et $p$.** Si $f$ est grande là où $p$ est petite, les tirages tombent rarement dans les zones qui contribuent à l'intégrale. Cela force des tailles d'échantillon énormes pour atteindre une précision donnée. C'est exactement ce que résout l'**Importance Sampling** (section V).

## II. Loi d'une fonction d'une variable aléatoire

Avant de pouvoir échantillonner, il faut savoir manipuler les **transformations** de variables aléatoires : si on sait tirer $X$ et qu'on définit $Y = g(X)$, quelle est la loi de $Y$ ? Cette mécanique est le pré-requis direct de l'Inverse Transform Sampling, où on construit volontairement une transformation qui produit la loi voulue.

### A. Méthode en deux étapes

**Procédure générale.** Soit $X$ une v.a. continue de densité $f_X$, et $Y = g(X)$ pour une fonction $g$ quelconque. Pour trouver la densité de $Y$ :

1. **Calculer la fonction de répartition (CDF, cumulative distribution function) de $Y$ :**
$$F_Y(y) = \mathbb{P}(Y \le y) = \mathbb{P}(g(X) \le y).$$
2. **Dériver :** $f_Y(y) = \dfrac{\mathrm{d} F_Y}{\mathrm{d} y}(y).$

L'étape (1) revient à exprimer l'événement $\{g(X) \le y\}$ en termes de $X$, ce qui dépend de la forme de $g$.

> [!example] Exemple — $Y = X^3$ avec $X \sim \mathrm{Unif}([0,2])$
>
> ![[images/1-Mathématiques/Probabilité/Échantillonage/im2.png]]
> **Figure 2.** À gauche, la densité de $X$ uniforme sur $[0,2]$ (constante à $1/2$). À droite, le support attendu de $Y$, qui ira de $0$ à $8$.
>
> *(1)* La fonction $x \mapsto x^3$ est strictement croissante donc inversible. Pour $0 \le y \le 8$ :
>
> $$F_Y(y) = \mathbb{P}(X^3 \le y) = \mathbb{P}(X \le y^{1/3}) = \frac{y^{1/3}}{2}$$
>
> où la dernière égalité vient de la CDF de l'uniforme : $F_X(x) = x/2$ sur $[0,2]$. Car pour rappel $F(x)=\frac{(x-a)}{(b-a)}$ où $a \le x \le b$
>![[images/1-Mathématiques/Probabilité/Échantillonage/im3 (1).png]]
> *(2)* On dérive :
>
> $$f_Y(y) = \frac{1}{2} \cdot \frac{1}{3} y^{-2/3} = \frac{1}{6 \, y^{2/3}}, \qquad y \in [0, 8].$$
>
> ![[im4 1.png]]
> **Figure 3.** La densité de $Y = X^3$ explose près de $0$ (en $y^{-2/3}$) et est très plate vers $8$. C'est cohérent : le cube écrase les petites valeurs ensemble et étire les grandes.

### B. Formule générale pour $g$ monotone

Quand $g$ est strictement monotone et dérivable, on peut court-circuiter le calcul de la CDF et passer directement à la densité. On note $h = g^{-1}$ la fonction inverse.

**Cas $g$ strictement croissante.** L'événement $\{g(X) \le y\}$ équivaut à $\{X \le h(y)\}$, donc $F_Y(y) = F_X(h(y))$. En dérivant :

$$f_Y(y) = f_X(h(y)) \cdot \frac{\mathrm{d}h}{\mathrm{d}y}(y).$$
**Cas $g$ strictement décroissante.** L'événement s'inverse : $\{g(X) \le y\} = \{X \ge h(y)\}$, donc $F_Y(y) = 1 - F_X(h(y))$ et

$$f_Y(y) = -f_X(h(y)) \cdot \frac{\mathrm{d}h}{\mathrm{d}y}(y).$$
Le signe moins compense le fait que $\mathrm{d}h/\mathrm{d}y < 0$ dans ce cas, de sorte que $f_Y$ reste bien positive.

> [!warning] Formule générale (changement de variable, change of variables)
> Pour $g$ strictement monotone et dérivable, $h = g^{-1}$ :
>
> $$\boxed{\; f_Y(y) = f_X(h(y)) \, \left| \frac{\mathrm{d}h}{\mathrm{d}y}(y) \right| \;}$$
>
> La valeur absolue unifie les deux cas. Le facteur $|h'(y)|$ est le **jacobien** scalaire (Jacobian), qui mesure comment $g$ étire ou compresse localement la mesure : là où $g$ étire (dérivée grande), la densité de $Y$ se dilue ; là où $g$ compresse, elle se concentre.

> [!note]- Et si $g$ n'est pas monotone ?
> Si $g$ n'est pas inversible globalement (par exemple $g(x) = x^2$ sur $\mathbb{R}$), on découpe le support de $X$ en morceaux où $g$ est monotone, on applique la formule sur chacun, et on **somme** les contributions :
>
> $$f_Y(y) = \sum_{x \in g^{-1}(\{y\})} f_X(x) \, \left| \frac{\mathrm{d}h}{\mathrm{d}y}(y) \right|.$$
>
> Pour $Y = X^2$ avec $X \sim \mathcal{N}(0,1)$, cette somme sur les deux antécédents $\pm \sqrt{y}$ redonne la densité du chi-deux à un degré de liberté.

## III. Inverse Transform Sampling

**Motivation.** Tous les générateurs aléatoires d'un ordinateur (RNG, random number generator) produisent en réalité une seule chose : des tirages $U \sim \mathrm{Unif}([0,1])$. Pour échantillonner depuis n'importe quelle autre loi, il faut une recette qui transforme l'uniforme en la loi voulue. C'est exactement le rôle de l'Inverse Transform.

**Principe.** On cherche une fonction $g$ telle que, si $U \sim \mathrm{Unif}([0,1])$, alors $g(U)$ ait la CDF cible $F_X$. Le miracle, c'est que cette fonction est explicite : c'est l'**inverse de la CDF cible**.

![[ech_inverse_box.png]]
**Figure 4.** L'idée : on a un boîtier RNG qui produit du bruit uniforme $U$, et on cherche à le transformer en samples $X \sim F_X$.

### A. Cas continu

**Théorème (Inverse Transform).** Soit $F_X$ une CDF strictement croissante et continue. Si $U \sim \mathrm{Unif}([0,1])$, alors

$$X = F_X^{-1}(U) \;\sim\; F_X.$$
> [!note]- Preuve
> On vérifie que $X = F_X^{-1}(U)$ a bien la CDF $F_X$. Pour tout $c \in \mathbb{R}$ :
>
> $$\mathbb{P}(X \le c) = \mathbb{P}(F_X^{-1}(U) \le c) = \mathbb{P}(U \le F_X(c)) = F_X(c)$$
>
> où l'avant-dernière égalité utilise la monotonie de $F_X$ (on peut appliquer $F_X$ aux deux côtés sans changer le sens), et la dernière utilise le fait que pour $U$ uniforme sur $[0,1]$, $\mathbb{P}(U \le u) = u$ pour tout $u \in [0,1]$. $\blacksquare$

![[Pasted image 20260502173811.png|471]]
**Figure 5.** Lecture graphique : on tire $u$ uniformément sur l'axe vertical (entre $0$ et $1$, là où vit la CDF), on lit horizontalement jusqu'à toucher la courbe $F_X$, puis on descend pour récupérer la valeur $x$ correspondante. La concentration des $x$ là où $F_X$ monte vite reproduit exactement la densité $f_X$.

> [!example] Exemple fil rouge — la loi exponentielle
> On prend $X \sim \mathrm{Exp}(1)$, de densité $f_X(x) = e^{-x}$ et de CDF $F_X(x) = 1 - e^{-x}$ pour $x \ge 0$. On inverse :
>
> $$u = 1 - e^{-x} \;\;\Longleftrightarrow\;\; x = -\log(1 - u).$$
>
> **Recette.** Pour générer un tirage exponentiel, on tire $U \sim \mathrm{Unif}([0,1])$ et on retourne $-\log(1-U)$. (En pratique on utilise souvent $-\log(U)$, qui suit la même loi puisque $1-U$ est aussi uniforme.) Cette loi exponentielle nous servira de fil rouge dans Rejection (section IV) et Monte Carlo (section VI), où l'on calculera $\mathbb{E}[X^2]$ par simulation.

### B. Cas discret

Pour une v.a. discrète, la CDF est une fonction en escalier. Inverser revient à découper $[0,1]$ en intervalles, un par valeur possible, dont les longueurs sont les probabilités.

**Recette.** Soit $X$ à valeurs $\{x_1, \dots, x_K\}$ avec probabilités $\{p_1, \dots, p_K\}$. On définit les seuils cumulés $c_0 = 0$, $c_k = p_1 + \dots + p_k$. Pour tirer $X$ :

1. Tirer $U \sim \mathrm{Unif}([0,1])$.
2. Retourner $x_k$ tel que $c_{k-1} \le U < c_k$.

![[ech_inverse_discret.png]]
**Figure 6.** Exemple à trois valeurs avec probabilités $(2/6, 3/6, 1/6)$. On découpe $[0,1]$ en trois intervalles de ces longueurs ; le tirage uniforme tombe dans l'un d'eux et désigne la valeur correspondante. Sur la CDF (à droite), c'est exactement la lecture horizontale puis verticale, sauf que la CDF est en escalier.

### C. Limites de la méthode

L'Inverse Transform est élégant et exact, mais ne s'applique pas toujours :

1. **$F_X^{-1}$ doit être calculable.** Pour la loi normale par exemple, ni la CDF ($\Phi$) ni son inverse n'ont d'expression élémentaire — il faut passer par des approximations numériques ou par d'autres méthodes (Box-Muller, ratio uniforme).
2. **Densités définies à une constante près.** En statistique bayésienne, on connaît typiquement $\tilde{p}(\mathbf{z}) \propto p(\mathbf{z})$ sans connaître la constante de normalisation. Dans ce cas la CDF n'est même pas définie, et l'Inverse Transform est inutilisable. Cette limitation motive directement les deux méthodes suivantes.

## IV. Rejection Sampling

**Motivation.** On veut échantillonner $X \sim p$, mais $p$ est compliquée (CDF non inversible, densité connue seulement à une constante près). En revanche, on a une densité **proposale** $q$ — facile à échantillonner, par exemple uniforme ou gaussienne — qui domine $p$ à un facteur près.

**Hypothèse clé.** Il existe une constante $M \ge 1$ telle que $p(x) \le M \cdot q(x) \text{ pour tout } x.$
Autrement dit, la courbe $M q$ forme une **enveloppe** au-dessus de $p$.
### A. Algorithme

**Procédure.** Pour produire un tirage $X \sim p$ :

1. Tirer un candidat $x \sim q$.
2. Tirer $u \sim \mathrm{Unif}([0,1])$.
3. **Accepter** $x$ si $u \le \dfrac{p(x)}{M \, q(x)}$ ; sinon **rejeter** et recommencer.

Le ratio $p(x) / (M q(x)) \in [0,1]$ joue le rôle de probabilité d'acceptation conditionnelle à $x$.

> [!note]- Pourquoi ça marche
> On cherche à montrer que conditionnellement à l'acceptation, le candidat $x$ a bien densité $p$. La densité jointe d'un candidat $x \sim q$ accepté est
>
> $$q(x) \cdot \underbrace{\frac{p(x)}{M q(x)}}_{\text{prob. d'accepter}} = \frac{p(x)}{M}.$$
>
> La probabilité totale d'accepter (en marginalisant sur $x$) est
>
> $$\mathbb{P}(\text{accept}) = \int \frac{p(x)}{M} \, \mathrm{d}x = \frac{1}{M}.$$
>
> Par définition de la loi conditionnelle, la densité de $x$ sachant qu'on a accepté est
>
> $$\frac{p(x) / M}{1 / M} = p(x). \qquad \blacksquare$$
### B. Taux d'acceptation et limites

**Taux d'acceptation.** Comme on vient de le voir, la probabilité d'accepter un candidat à chaque tour est $1/M$. En espérance, il faut donc $M$ tirages de $q$ pour produire un tirage de $p$. Conséquence pratique :

- Plus $q$ ressemble à $p$, plus $M$ peut être pris petit (dans le meilleur cas $M = 1$, jamais de rejet).
- Plus $q$ s'écarte de $p$, plus $M$ doit être grand, plus on rejette, plus c'est inefficace.

> [!warning] Le fléau de la dimension (curse of dimensionality)
> En dimension élevée, trouver une enveloppe serrée devient quasi impossible. Pour $p$ et $q$ deux gaussiennes de variances proches en dimension $d$, la constante optimale $M$ croît **exponentiellement** en $d$. Le taux d'acceptation $1/M$ devient alors astronomiquement petit, et la méthode est inutilisable en grande dimension. C'est l'une des raisons historiques du passage à l'**Importance Sampling** (qui ne rejette rien) puis aux **chaînes de Markov** (MCMC), qui contournent le problème en construisant un parcours guidé de l'espace plutôt qu'un échantillonnage indépendant.

> [!example] Exemple — sampler un mélange de gaussiennes
> On prend comme cible un **mélange de gaussiennes** (Gaussian mixture), un objet qu'on rencontre partout en ML (modèles de mélange, GMM) et en finance (rendements bimodaux selon le régime de marché) :
>
> $p(x) = 0.4 \cdot \mathcal{N}(x; -2, 1) + 0.6 \cdot \mathcal{N}(x; 3, 1.5).$
>
> Sa CDF est une somme de fonctions d'erreur (erf), pas inversible analytiquement → Inverse Transform est hors-jeu. Comme **proposale**, on prend une gaussienne large $q = \mathcal{N}(0.5, 4)$, centrée entre les deux modes et assez étalée pour les recouvrir tous les deux. On calcule numériquement $M = \max_x \, p(x)/q(x) \approx 2.5$.
>
> ![[Pasted image 20260502174413.png]]
> **Figure 7bis.** *Gauche :* la cible $p$ (rouge) avec ses deux bosses, la proposale $q$ (bleu pointillé), et l'enveloppe $M \cdot q$ (bleu plein) qui domine $p$ partout. La zone grise entre $p$ et $Mq$ est la **zone de rejet**. Les points verts sont les candidats acceptés (sous $p$), les rouges sont les rejetés (dans la zone grise). *Droite :* l'histogramme des candidats acceptés colle exactement à la vraie densité $p$ — preuve concrète que la mécanique a marché.
>
> **Bilan.** Avec $M \approx 2.5$, le taux d'acceptation théorique est $1/M \approx 40\%$ : on jette environ 60% des candidats. C'est le coût à payer pour pouvoir sampler depuis une cible dont on ne sait pas inverser la CDF.

> [!note]- Et le cas bayésien — densité connue à une constante près
> En statistique bayésienne, on connaît typiquement le posterior à une constante près : $\tilde{p}(x) \propto p(x)$. Le miracle de Rejection : la constante inconnue est **absorbée dans $M$**, et la mécanique produit quand même des tirages exacts de $p$. C'est pour ça que Rejection (et plus tard MCMC) sont des outils centraux du bayésien.

## V. Importance Sampling

**Motivation.** Les méthodes précédentes produisent des tirages de $p$. Mais souvent on n'a pas besoin de tirages — on a besoin d'**une espérance** $\mathbb{E}_p[f]$. L'Importance Sampling (IS) court-circuite l'étape de sampling : on calcule directement l'espérance en samplant depuis une autre loi $q$, plus facile.

> 💡 **L'idée en une phrase.** Au lieu de sampler depuis $p$ (difficile) et de moyenner $f$, on sample depuis $q$ (facile) et on moyenne $f$ **pondéré** par le ratio $p/q$, qui corrige le décalage entre les deux lois. Ce n'est pas une méthode d'échantillonnage, c'est une méthode d'**estimation**.

### A. La formule

**Identité algébrique.** En multipliant et divisant par $q(x)$ dans l'intégrale :

$$\mathbb{E}_p[f] = \int f(x) \, p(x) \, \mathrm{d}x = \int f(x) \, \frac{p(x)}{q(x)} \, q(x) \, \mathrm{d}x = \mathbb{E}_q\!\left[ f(X) \, \frac{p(X)}{q(X)} \right].$$
**Estimateur Importance Sampling.** Si l'on tire $x_1, \dots, x_n \sim q$ iid :

$$\widehat{\mu}_q = \frac{1}{n} \sum_{i=1}^{n} f(x_i) \, \frac{p(x_i)}{q(x_i)}.$$
Le ratio $w(x) = p(x)/q(x)$ s'appelle le **poids d'importance** (importance weight). Il corrige le biais qui viendrait du fait qu'on a samplé depuis la mauvaise loi.

**Vocabulaire :**
- $p$ est la **loi cible** (target distribution) — celle dont on veut l'espérance.
- $q$ est la **loi proposale** (proposal distribution) — celle qu'on sait sampler.
- Hypothèse minimale : $q(x) > 0$ partout où $p(x) > 0$ (sinon le ratio diverge sur des zones non négligeables).

### B. Variance et choix de la proposale

**Sans biais.** $\mathbb{E}_q[\widehat{\mu}_q] = \mathbb{E}_p[f]$, par construction de l'identité ci-dessus.

**Variance.** $\mathrm{Var}_q[\widehat{\mu}_q] = \dfrac{\sigma_q^2}{n}$ où

$$\sigma_q^2 = \int \frac{\big(f(x) p(x) - \mu \, q(x)\big)^2}{q(x)} \, \mathrm{d}x, \qquad \mu = \mathbb{E}_p[f].$$
Le choix de $q$ contrôle entièrement cette variance. Et c'est là que ça devient intéressant : un bon choix de $q$ peut donner une variance **plus petite** que le Monte Carlo vanilla qui sample directement depuis $p$. Un mauvais choix peut la faire **exploser**, parfois jusqu'à l'infini.

> [!warning] Heuristique du bon choix de $q$
> La variance est minimale quand $q(x) \propto |f(x)| \, p(x)$, c'est-à-dire quand $q$ concentre sa masse là où le **produit** $|f| \cdot p$ est grand. En pratique :
>
> - **$q$ trop concentrée à un endroit où $p$ est diffuse** → certains poids $p/q$ explosent → variance énorme.
> - **$q$ étalée loin du support de $f \cdot p$** → la plupart des $f(x_i)$ sont nuls ou les poids $p/q$ sont minuscules → on perd tout le signal.
> - **$q$ qui ressemble à $|f| \cdot p$** → les contributions sont uniformes, variance minimale.
>
> Le mantra : **on veut sampler là où ça compte** (là où $f \cdot p$ est grand), pas là où $p$ seule est grande.

> [!example] Exemple — l'effet catastrophique d'une mauvaise proposale
> On veut estimer $\mathbb{E}_p[f]$ avec
>
> $f(x) = \frac{1}{1 + e^{-x}} \quad \text{(sigmoïde)}, \qquad X \sim \mathcal{N}(3.5, 1).$
>
> Comme $p$ (la densité de $X$) est concentrée vers $x = 3.5$ où la sigmoïde vaut $\approx 0.97$, la vraie valeur est $\mu \approx 0.95$.
>
> ![[Pasted image 20260502175001.png]]
> **Figure 8.** Deux choix de proposale $q$. À gauche, $q = \mathcal{N}(3, 1)$ recouvre bien le support de $p$ : les poids $p/q$ restent modérés. À droite, $q = \mathcal{N}(6, 1)$ est décalée : très peu de tirages de $q$ tombent là où $p$ est grande, mais quand un rare tirage tombe vers $x = 3$, le poids $p(x)/q(x)$ est colossal et fait sauter l'estimateur.
>
> **Résultats numériques** (avec $n$ identique dans tous les cas) :
>
> | Méthode | Estimation | Variance |
> |---|---|---|
> | Monte Carlo vanilla ($X \sim p$) | $0.954$ | référence |
> | IS avec $q = \mathcal{N}(3, 1)$ | $0.951$ | $\approx 0.30$ ✅ |
> | IS avec $q = \mathcal{N}(6, 1)$ | $0.684$ | $\approx 14.1$ ❌ |
>
> Avec une bonne proposale, l'IS bat même le MC vanilla en variance. Avec une mauvaise, l'estimateur est biaisé en pratique (à $n$ fini) et la variance explose. C'est le couteau à double tranchant qu'il faut garder en tête à chaque fois qu'on déploie de l'IS.

> [!note]- Self-normalized Importance Sampling (SNIS)
> Quand $p$ n'est connue qu'à une constante près (cas bayésien), on ne peut pas calculer le poids $w(x) = p(x)/q(x)$ exactement. On utilise alors la version auto-normalisée :
>
> $$\widehat{\mu}_q^{\text{SNIS}} = \frac{\sum_i \tilde{w}(x_i) \, f(x_i)}{\sum_i \tilde{w}(x_i)}, \qquad \tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}$$
>
> où $\tilde{p} \propto p$. La constante de normalisation inconnue se simplifie entre numérateur et dénominateur. Cet estimateur est **biaisé à $n$ fini** (le rapport de deux moyennes empiriques n'est pas la moyenne du rapport) mais consistant ($n \to \infty$).

## VI. Monte Carlo

L'**intégration de Monte Carlo** est l'application directe des sections précédentes : on a appris à produire des tirages, on s'en sert pour calculer des intégrales sous forme d'espérances. Cette section regroupe les usages classiques (espérance, variance, probabilités, modèles hiérarchiques) et l'évaluation de la qualité des estimations.

### A. Méthode des rectangles (déterministe) — pour comparer

**Méthode des rectangles.** Pour $\int_a^b f(x) \, \mathrm{d}x$, on découpe $[a,b]$ en $N$ pas réguliers $x_i = a + i \cdot (b-a)/N$ et on approche par

$$\int_a^b f(x) \, \mathrm{d}x \;\approx\; \frac{b-a}{N} \sum_{i=0}^{N-1} f(x_i).$$
C'est la méthode déterministe la plus simple. Elle marche très bien en dimension 1 ou 2, mais sa précision se dégrade en $N^{-2/d}$ en dimension $d$ : pour garder la même précision quand $d$ augmente, il faut un nombre de points qui explose exponentiellement. C'est le **fléau de la dimension** sous sa forme la plus brute.

> [!note]- Pourquoi MC bat la quadrature en haute dimension
> La méthode des rectangles a une erreur en $\mathcal{O}(N^{-2/d})$ — elle dépend de $d$. Monte Carlo a une erreur en $\mathcal{O}(N^{-1/2})$ — **indépendante** de $d$. Il existe donc une dimension critique au-delà de laquelle Monte Carlo (même mauvais) bat la quadrature (même optimale). En pratique, au-delà de $d \approx 4$, on passe systématiquement au stochastique. C'est ce qui fait de Monte Carlo l'outil universel en finance quantitative (panier d'actifs, portefeuilles), en physique statistique, et en bayésien.

### B. Monte Carlo comme espérance

**Recette générale.** Pour calculer une intégrale, on l'écrit comme une espérance, puis on simule. Toute intégrale $\int g(x) \, \mathrm{d}x$ peut être réécrite ainsi pour une infinité de choix de découpage $g(x) = f(x) \cdot p(x)$. Le choix change la variance de l'estimateur — exactement comme dans IS, c'est le choix de $p$ qui contrôle l'efficacité.

> [!example] Exemple fil rouge — calcul de $I = \int_0^\infty x^2 e^{-x} \, \mathrm{d}x$
> La vraie valeur (calculable analytiquement) est $\Gamma(3) = 2$. On va l'estimer par MC de **deux** manières différentes pour montrer l'effet du découpage.
>
> **Technique 1 : exponentielle.** On reconnaît $e^{-x}$ comme la densité de $\mathrm{Exp}(1)$. On pose $f(x) = x^2$, $p(x) = e^{-x}$. Alors
>
> $$I = \mathbb{E}_{X \sim \mathrm{Exp}(1)}[X^2] \;\approx\; \frac{1}{m} \sum_{i=1}^{m} (x_i^*)^2, \quad x_i^* \sim \mathrm{Exp}(1).$$
>
> Pour générer les $x_i^*$, on utilise l'Inverse Transform de la section III : $x_i^* = -\log(U_i)$ avec $U_i \sim \mathrm{Unif}([0,1])$. On obtient $\widehat{I} \approx 2$.
>
> **Technique 2 : uniforme tronquée.** Comme $x^2 e^{-x}$ devient négligeable au-delà de $x \approx 15$, on peut approcher l'intégrale par $\int_0^{50} x^2 e^{-x} \, \mathrm{d}x$ et utiliser une uniforme. On pose $f(x) = x^2 e^{-x}$, $p(x) = \mathbf{1}_{[0,50]}/50$ :
>
> $$I \;\approx\; \frac{50}{m} \sum_{i=1}^{m} (x_i^*)^2 e^{-x_i^*}, \quad x_i^* \sim \mathrm{Unif}(0, 50).$$
>
> Les deux estimateurs convergent vers 2, mais la première est **bien plus efficace** : la grande majorité de la masse de $x^2 e^{-x}$ est près de l'origine, donc l'uniforme gaspille la plupart de ses tirages dans la queue où l'intégrande est négligeable. C'est exactement l'intuition d'IS qu'on a vu en section V : il faut sampler là où **l'intégrande** est grande, pas n'importe où.
>
> ![[Pasted image 20260502175315.png]]
> **Figure 9.** Le graphe de $x^2 e^{-x}$. La masse est concentrée vers $x \in [0, 5]$ ; au-delà l'intégrande est essentiellement nulle.

**Interprétation géométrique.** L'estimateur de Monte Carlo calcule l'aire sous la courbe $f \cdot p$. Chaque réplique de la procédure (tirer un nouvel ensemble de $m$ samples) produit une nouvelle estimation de cette aire. La distribution de ces estimations, par TCL, est gaussienne autour de la vraie valeur, de largeur $\propto 1/\sqrt{m}$.

![[Pasted image 20260502175340.png|605]]
**Figure 10.** Distribution des estimateurs de Monte Carlo sur des réplicats. Plus $m$ est grand, plus cette distribution est piquée autour de la vraie valeur — c'est le sens concret de "réduire la variance".

### C. Cas particuliers utiles

**1. Espérance d'une fonction quelconque.** Pour $\theta \sim p$ et $h$ une fonction :

$$\int h(\theta) p(\theta) \, \mathrm{d}\theta = \mathbb{E}[h(\theta)] \;\approx\; \frac{1}{m} \sum_{i=1}^m h(\theta_i^*), \quad \theta_i^* \sim p.$$
Ce schéma générique inclut **tous** les calculs qu'on fait sur une distribution : moyenne ($h(\theta) = \theta$), variance ($h(\theta) = (\theta - \bar\theta)^2$), moment d'ordre $k$ ($h(\theta) = \theta^k$), espérance d'une transformation ($h(\theta) = e^\theta$), etc.

**2. Probabilités via indicatrices.** Cas particulier majeur : prendre $h = \mathbf{1}_A$ l'indicatrice d'un événement.

$$\mathbb{P}(\theta \in A) = \mathbb{E}[\mathbf{1}_A(\theta)] \;\approx\; \frac{1}{m} \sum_{i=1}^m \mathbf{1}_A(\theta_i^*).$$
Concrètement : on tire $m$ échantillons et on compte la proportion qui tombe dans $A$. Pour estimer $\mathbb{P}(\theta < 5)$, on tire $m$ échantillons et on compte combien sont $< 5$. Pour $\mathbb{P}(X < Y)$ avec $(X,Y)$ joints, on tire $m$ paires et on compte celles où $x_i^* < y_i^*$.

> [!example] Exemple fil rouge — moyenne et probabilité d'une Gamma
> Soit $\theta \sim \mathrm{Gamma}(a=2, b=1/3)$. La vraie moyenne est $a/b = 6$, la vraie variance $a/b^2 = 18$.
>
> En tirant $m = 10\,000$ échantillons et en calculant la moyenne empirique, on obtient $\widehat{\mathbb{E}[\theta]} \approx 6.02$ et $\widehat{\mathrm{Var}[\theta]} \approx 18.01$. La précision est typique d'un Monte Carlo : erreur en $\sim 1/\sqrt{m}$.
>
> Pour $\mathbb{P}(\theta < 5)$, on calcule la fréquence empirique : $\frac{1}{m} \sum_i \mathbf{1}_{\theta_i^* < 5} \approx 0.50$. Cela revient juste à compter la proportion d'échantillons sous le seuil.

### D. Modèles hiérarchiques

**Cadre.** Un modèle hiérarchique (hierarchical model) est défini par une suite de lois conditionnelles. Exemple canonique :

$$\phi \sim \mathrm{Beta}(2, 2), \qquad y \mid \phi \sim \mathrm{Bin}(10, \phi).$$
La loi jointe se factorise par la **règle de chaîne** (chain rule) :

$$p(y, \phi) = p(\phi) \, p(y \mid \phi).$$
**Échantillonnage par tirage successif (ancestral sampling).** Pour produire des tirages joints $(y_i^*, \phi_i^*)$ :

1. Tirer $\phi_i^* \sim \mathrm{Beta}(2,2)$.
2. Tirer $y_i^* \sim \mathrm{Bin}(10, \phi_i^*)$ en utilisant la valeur précédente.
3. Le couple $(y_i^*, \phi_i^*)$ est un tirage de la loi jointe.

> [!warning] Marginalisation gratuite (free marginalization)
> Une fois qu'on a $m$ tirages joints $(y_i^*, \phi_i^*)$, on peut **lire n'importe quelle marginale en ignorant les autres coordonnées**. Pour la marginale de $y$ : on garde juste les $y_i^*$ et on ignore les $\phi_i^*$. C'est une propriété générale, profondément utile : *toute* marginale d'un modèle joint se calcule en samplant le joint et en projetant. Cela explique pourquoi le sampling est l'outil de base du bayésien — on génère le joint, et on récupère gratuitement toutes les marginales et toutes les conditionnelles d'intérêt.

> [!example] Marginale de $y$ dans le modèle Beta-Binomial
> Avec $m = 10^5$ tirages $(\phi_i^*, y_i^*)$ et un comptage des fréquences de $y$ :
>
> | $y$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
> |---|---|---|---|---|---|---|---|---|---|---|---|
> | $\widehat{\mathbb{P}}(Y=y)$ | 0.04 | 0.07 | 0.09 | 0.11 | 0.12 | 0.13 | 0.12 | 0.11 | 0.09 | 0.07 | 0.04 |
>
> On retrouve la **loi Beta-Binomiale** (Beta-Binomial distribution), qui est exactement la marginale de $y$ ici. On l'a calculée sans jamais la dériver analytiquement, juste en tirant des paires et en projetant. C'est la force du sampling.

### E. Erreur de Monte Carlo

**Question pratique.** À quel point peut-on faire confiance à $\widehat{f}$ ? Le TCL répond directement.

**Erreur standard.** Pour un estimateur $\widehat{f} = \frac{1}{m} \sum_i f(x_i^*)$ :

$$\widehat{f} \;\sim\; \mathcal{N}\!\left( \mathbb{E}_p[f], \; \frac{\mathrm{Var}_p[f]}{m} \right) \quad \text{(asymptotiquement)}.$$
On approche $\mathrm{Var}_p[f]$ par sa version empirique $\widehat{\mathrm{Var}}[f] = \frac{1}{m} \sum_i (f(x_i^*) - \widehat{f})^2$, et l'**erreur standard** est

$$\mathrm{SE} = \sqrt{\frac{\widehat{\mathrm{Var}}[f]}{m}}.$$
**Intervalle de confiance à 95%.** $\widehat{f} \pm 2 \cdot \mathrm{SE}$. C'est l'outil de base pour décider si $m$ est assez grand.

> [!example] Précision pour la moyenne d'une Gamma
> Reprenons $\theta \sim \mathrm{Gamma}(2, 1/3)$ avec $m = 10\,000$. On obtient $\widehat{\mathbb{E}[\theta]} \approx 6.02$ et un écart-type empirique $\approx 4.24$. L'erreur standard est
>
> $$\mathrm{SE} = \frac{4.24}{\sqrt{10\,000}} \approx 0.042.$$
>
> L'intervalle de confiance à 95% est donc $[5.94, \; 6.11]$, qui contient bien la vraie valeur $6$. Pour resserrer l'IC d'un facteur 10, il faut multiplier $m$ par 100 — c'est le prix de la convergence en $1/\sqrt{m}$, que ne change ni la dimension du problème ni le choix de $f$.

> [!summary] À retenir
> 1. **Tout calcul probabiliste = espérance + sampling.** Espérances, probabilités, variances, marginales, transformations : tout passe par $\frac{1}{m} \sum h(x_i^*)$ avec $x_i^* \sim p$.
> 2. **Trois mécanismes pour produire les $x_i^*$.** Inverse Transform si on a $F^{-1}$ ; Rejection si on a une enveloppe ; sinon (ou en grande dimension) MCMC, traité ailleurs.
> 3. **L'Importance Sampling court-circuite le sampling de $p$** en samplant depuis $q$ et en pondérant. Le choix de $q$ contrôle entièrement la variance — c'est aussi son talon d'Achille.
> 4. **L'erreur converge en $1/\sqrt{m}$**, indépendamment de la dimension. C'est la propriété qui rend Monte Carlo universel et qui justifie son usage en finance quantitative, en physique statistique, et en bayésien.

## VII. Arbre de décision — quelle méthode choisir ?

Face à un problème concret, deux questions suffisent à trancher entre les quatre méthodes vues plus haut.

**Question 1 : tu veux *des tirages* de $p$, ou *une espérance* $\mathbb{E}_p[f]$ ?**

- Tirages → III ou IV (ou MCMC, hors de ce fichier).
- Espérance → V ou VI.

**Question 2 : sous quelle hypothèse es-tu ?**

- Tu sais inverser la CDF de $p$ → **III. Inverse Transform**.
- Tu sais majorer $p$ par $M \cdot q$ avec $q$ facile à sampler → **IV. Rejection**.
- Tu sais sampler $p$ directement → **VI. Monte Carlo vanilla**.
- Tu ne sais pas sampler $p$, mais tu sais sampler une autre loi $q$ → **V. Importance Sampling**.

**Récap :**

| Méthode | Tu veux quoi ? | Hypothèse clé | Quand l'utiliser |
|---|---|---|---|
| **III. Inverse Transform** | Tirages de $p$ | $F_X^{-1}$ analytique | Lois "standards" 1D (exponentielle, Pareto, Cauchy, Weibull...) |
| **IV. Rejection** | Tirages de $p$ | $p \le M \cdot q$ avec $q$ facile | $p$ tordue (mélange, posterior bayésien à constante près), petite dimension |
| **V. Importance Sampling** | Espérance $\mathbb{E}_p[f]$ | $q(x) > 0$ partout où $p(x) > 0$ | $p$ dur à sampler, ou événements rares où $f \cdot p$ est concentré |
| **VI. Monte Carlo (vanilla)** | Espérance $\mathbb{E}_p[f]$ | Tu sais sampler $p$ | Cas standard, $f$ et $p$ alignées |

> [!note]- Le piège classique
> Ne pas confondre **"produire des samples"** (III, IV) et **"estimer une espérance"** (V, VI). L'Importance Sampling en particulier n'est *pas* une méthode pour sampler $p$ — c'est une méthode pour estimer une espérance *sans* sampler $p$. Si tu as besoin des samples eux-mêmes (pour faire un histogramme, calculer un quantile, alimenter un modèle aval), IS ne te les donne pas.

> [!note]- Et MCMC ?
> Quand aucune des quatre méthodes ne s'applique — typiquement en grande dimension, ou pour un posterior bayésien complexe sans enveloppe utilisable — on passe aux **chaînes de Markov** (MCMC) : Metropolis-Hastings, Gibbs, HMC. Ces méthodes produisent des samples *corrélés* (pas iid) en construisant un parcours guidé de l'espace. Traité dans le fichier MCMC.
