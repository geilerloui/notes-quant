---
title: Rough Paths
date: 2026-05-10
tags: [probabilités, processus-stochastiques, rough-paths, à-développer]
---

## L'idée fondatrice

La théorie des **rough paths** (Lyons 1998) étend le calcul stochastique d'Itô à des intégrateurs **trop irréguliers** pour être traités par les outils classiques. Pour comprendre ce que ça veut dire et pourquoi c'est utile, on a besoin de deux concepts préalables : **processus gaussien** et **mouvement brownien fractionnaire**. On les voit dans cet ordre, puis on voit pourquoi Itô casse, puis on esquisse la solution de Lyons.

---

## I. Préliminaire : qu'est-ce qu'un processus gaussien ?

Tu connais déjà l'archétype : le brownien standard $W_t$ est un processus gaussien. Mais on n'a jamais formalisé le terme. Le voici :

> [!warning] Définition — Processus gaussien
> Un processus stochastique $(X_t)_{t \geq 0}$ est dit **gaussien** si pour tout choix fini d'instants $t_1, \ldots, t_n$, le vecteur aléatoire $(X_{t_1}, \ldots, X_{t_n})$ suit une **loi gaussienne multivariée**.
> 
> Un processus gaussien est entièrement caractérisé par :
> - sa **fonction moyenne** $m(t) = \mathbb{E}[X_t]$
> - sa **fonction de covariance** $K(s, t) = \mathbb{E}[(X_s - m(s))(X_t - m(t))]$

C'est une propriété puissante : **deux fonctions** ($m$ et $K$) suffisent à caractériser entièrement la loi du processus, alors qu'en général il faudrait connaître toutes les lois finies-dimensionnelles à tout ordre.

> [!example] Trois processus gaussiens classiques que tu connais déjà
> - **Brownien standard** : $m(t) = 0$, $K(s, t) = \min(s, t)$ — cf [[01_Mouvement Brownien]]
> - **Ornstein-Uhlenbeck stationnaire** : $m(t) = \mu$, $K(s, t) = \frac{\sigma^2}{2\theta}e^{-\theta|s-t|}$ — cf [[03_Équations Différentielles Stochastiques]]
> - **Mouvement brownien fractionnaire** $B^H$ : $m(t) = 0$, $K(s,t) = \frac{1}{2}(|s|^{2H} + |t|^{2H} - |s-t|^{2H})$ — voir §II ci-dessous

> 💡 **Processus gaussiens en Machine Learning.** En ML, "Gaussian Process" (GP) désigne un usage différent — non pas un processus indexé par le temps, mais une **distribution sur des fonctions**. On modélise une fonction inconnue $f : \mathcal{X} \to \mathbb{R}$ comme un GP, on choisit un kernel $K(x, x')$ qui encode la corrélation, et on conditionne sur des observations pour obtenir une posterior sur $f$. C'est de la **régression bayésienne non-paramétrique** (kriging). Techniquement c'est le même cadre mathématique que ce qu'on utilise ici (un processus gaussien est défini par sa covariance), mais l'usage est différent : on indexe par les features plutôt que par le temps. À ne pas confondre avec un brownien donc.

---

## II. Le mouvement brownien fractionnaire (fBm)

> [!warning] Définition — Mouvement brownien fractionnaire
> Le **mouvement brownien fractionnaire** d'exposant de Hurst $H \in (0, 1)$ est le processus gaussien centré $B^H_t$ avec $B^H_0 = 0$ et fonction de covariance :
> 
> $$\mathbb{E}[B^H_s B^H_t] = \frac{1}{2}\left(|s|^{2H} + |t|^{2H} - |s-t|^{2H}\right)$$
> 
> **Cas particulier** : pour $H = 1/2$, on retrouve le brownien standard (la covariance se réduit à $\min(s,t)$).

L'exposant $H$ (du nom de l'hydrologue Harold Hurst, 1951, qui étudiait les niveaux du Nil) règle la **rugosité** des trajectoires. Trois régimes très différents :

### II.1 Trois régimes selon $H$

![[fig1_compare_H.png]]
*Figure 1. Trajectoires de $B^H_t$ pour 3 valeurs de l'exposant de Hurst, chaque panneau contenant 3 trajectoires indépendantes. **$H = 0.2$** (rouge) : trajectoires très **rugueuses**, oscillant violemment, anti-persistantes. **$H = 0.5$** (noir) : brownien standard. **$H = 0.8$** (vert) : trajectoires beaucoup plus **lisses**, persistantes — quand ça monte, ça continue de monter.*

**$H < 1/2$ — incréments anti-persistants**
- Si $B^H$ monte sur $[0, t]$, il a tendance à descendre sur $[t, t']$ (corrélation négative entre incréments)
- Trajectoires **plus rugueuses** que le brownien (Hölder $\alpha < H < 1/2$)
- C'est *ce régime* qui pose problème pour Itô (cf §III)

**$H = 1/2$ — brownien standard**
- Incréments **indépendants** (la propriété de Markov classique, qu'on a vue dans [[01_Mouvement Brownien]])
- Régularité Hölder $\alpha < 1/2$ (presque)
- Le cadre habituel : Itô, Black-Scholes, etc.

**$H > 1/2$ — incréments persistants**
- Si $B^H$ monte sur $[0, t]$, il a tendance à continuer à monter sur $[t, t']$ (corrélation positive)
- Trajectoires **plus lisses** que le brownien
- Modélise les phénomènes à mémoire longue : trafic réseau, niveaux du Nil, certains signaux climatiques

### II.2 Auto-similarité

Comme le brownien standard, $B^H$ est **auto-similaire** mais avec un autre exposant :

$$B^H_{ct} \stackrel{\text{loi}}{=} c^H \cdot B^H_t$$

(Pour le brownien standard tu te souviens : $W_{ct} = \sqrt{c}\,W_t$, donc $H = 1/2$.) Ça veut dire que si on zoome sur un intervalle 10× plus petit et qu'on rescale les valeurs par $10^H$, on retombe sur quelque chose de la même loi.

![[fig2_self_similarity.png]]
*Figure 2. Auto-similarité du fBm. Sur chaque panneau, la courbe colorée est $B^H_t$ sur $[0, 1]$. La courbe noire pointillée est le zoom sur $[0, 0.1]$ rescalé : $t \to 10t$ et $B \to B/10^H$. Les deux courbes ont la même statistique. **Gauche** ($H = 0.2$) : trajectoires rugueuses à toutes les échelles. **Droite** ($H = 0.8$) : trajectoires lisses à toutes les échelles. Le rescaling $c^H$ est le bon facteur pour rendre les deux échelles comparables.*

---

## III. Pourquoi Itô casse pour $H < 1/2$

Dans [[02_Calcul d'Itô]] (§IV.3 et le théorème associé), on a vu que la **variation quadratique du brownien** est finie et égale à $t$ :

$$\sum_{i} (W_{t_{i+1}} - W_{t_i})^2 \xrightarrow[\text{pas} \to 0]{} t$$

C'est **cette propriété qui fait fonctionner Itô** : la formule d'Itô utilise la variation quadratique pour son terme correctif $\frac{1}{2}f''\,d[X]$. Si la variation quadratique n'existait pas (limite infinie ou zéro), tout l'édifice s'écroulerait.

> [!warning] Variation quadratique du fBm
> Pour le mouvement brownien fractionnaire $B^H$ :
> 
> $$\sum_i (B^H_{t_{i+1}} - B^H_{t_i})^2 \xrightarrow[\text{pas} \to 0]{} \begin{cases} +\infty & \text{si } H < 1/2 \\ t & \text{si } H = 1/2 \\ 0 & \text{si } H > 1/2 \end{cases}$$

**Pour $H < 1/2$, la variation quadratique explose.** Les trajectoires sont trop irrégulières — chaque incrément contribue trop fort, et la somme diverge. Ça veut dire :

1. $B^H$ pour $H < 1/2$ **n'est pas une semi-martingale**
2. L'intégrale d'Itô $\int H_s\,dB^H_s$ **n'est pas définie** au sens classique
3. La formule d'Itô **ne s'applique pas**

![[fig3_quadratic_variation.png]]
*Figure 3. Variation quadratique empirique pour 3 valeurs de $H$, calculée sur la même trajectoire à différentes résolutions $n$. **$H = 0.5$** (noir) : converge vers $T = 1$ (la valeur théorique du brownien standard). **$H = 0.2$** (rouge) : explose en montant — variation quadratique infinie. **$H = 0.8$** (vert) : tend vers 0 — variation quadratique nulle. Donc Itô ne peut s'appliquer **que** pour $H = 1/2$.*

C'est ça la motivation centrale des rough paths : trouver un cadre nouveau qui permette d'intégrer contre $B^H$ même quand $H < 1/2$.

---

## IV. La solution de Lyons (1998) — enrichir le chemin

La motivation est claire : pour $H < 1/2$, l'intégrale d'Itô ne marche plus. Lyons (1998) propose une solution. Mais avant de comprendre cette solution, il faut comprendre **pourquoi le sujet ne se pose vraiment qu'en multi-dimensionnel** ($d \geq 2$), et **ce qu'on "perd" sur un chemin rugueux** : l'aire qu'il balaie.

### IV.1 Préliminaire — l'aire d'une boucle (formule de Green)

Reprenons l'analyse classique. Étant donné un chemin fermé et lisse $(x(t), y(t))$ dans le plan, son **aire enclose** se calcule par la formule de Green :

$$A = \frac{1}{2} \oint_C (x\,dy - y\,dx)$$

Le symbole $\oint$ note simplement "intégrale sur un chemin fermé" — même chose que $\int_0^T$ pour un chemin paramétrisé entre 0 et $T$ qui revient à son point de départ. La formule à retenir : on intègre $x\,dy - y\,dx$ le long du chemin, et on divise par 2.

![[fig5_area_classical.png]]
*Figure 5. **Gauche** — une ellipse paramétrisée par $(a\cos t, b\sin t)$ avec $a=2, b=1.2$. La formule de Green donne $A = \pi a b \approx 7.54$, l'aire colorée en bleu. Les flèches rouges montrent les vecteurs tangents $(dx, dy)$ à quelques points : c'est en suivant ces vecteurs et en accumulant $x\,dy - y\,dx$ qu'on "balaie" l'aire. **Droite** — construction numérique : on cumule $\int x\,dy$ (bleu) et $\int y\,dx$ (rouge), puis $\frac{1}{2}$ de leur différence (vert) converge bien vers l'aire vraie $\pi a b = 7.54$ (ligne pointillée noire).*

L'idée clé à retenir : **l'aire d'un chemin 2D est une intégrale qui dépend de la façon dont le chemin tourne**, pas seulement des positions $(x, y)$ qu'il visite.

### IV.2 L'aire de Lévy — le cas multi-dimensionnel

Maintenant on généralise au cas stochastique. Soit un chemin **2D** $X_t = (X^1_t, X^2_t)$ (par exemple, deux composantes d'un mouvement brownien). On définit son **aire de Lévy** :

$$A_{s,t} = \frac{1}{2} \int_s^t \left[ (X^1_r - X^1_s)\,dX^2_r - (X^2_r - X^2_s)\,dX^1_r \right]$$

C'est exactement l'analogue stochastique de la formule de Green vue en §IV.1, sauf qu'on ne ferme pas la boucle : on regarde l'aire balayée par les segments reliant le départ $X_s$ aux points courants $X_r$.

> [!note]- Décode la notation : $s$, $t$, $r$ — qui est qui ?
> Trois lettres, trois rôles différents. C'est le truc qui peut bloquer à la première lecture :
> - **$s$ et $t$ sont fixés** : ce sont les bornes de l'intégrale, choisies une fois pour toutes au début (par exemple $s = 0$ et $t = 1$). **Elles ne bougent pas** pendant le calcul.
> - **$r$ est la variable muette d'intégration** : c'est elle qui parcourt $[s, t]$ pendant qu'on intègre. Comme le $x$ dans $\int_0^1 f(x)\,dx$.
>
> Donc :
> - $X_s$ = la **valeur du chemin au temps $s$** (point de départ, fixé)
> - $X_r$ = la **valeur du chemin au temps $r$** (point courant, qui bouge)
> - $dX_r$ = un petit incrément du chemin entre $r$ et $r + dr$
>
> **Pourquoi $r$ et pas $s$ pour la variable muette ?** Parce que $s$ est déjà pris (c'est la borne gauche). Convention classique : $s, t$ pour les temps fixés, $r$ ou $u$ pour la variable muette d'intégration.

> [!note]- Pourquoi $X_r - X_s$ et pas juste $X_r$ ?
> On veut que l'aire de Lévy soit **invariante par translation du chemin**. Si tu déplaces tout le chemin de 1000 unités vers la droite, ça ne change rien à la "rotation" du chemin — c'est juste le même chemin ailleurs. Mais avec $X_r$ tout seul, l'intégrale exploserait à cause de la translation.
>
> En soustrayant le point de départ $X_s$, on **recentre le chemin sur l'origine** : on mesure vraiment la rotation, indépendamment de la position absolue. C'est l'analogue de la formule de Green qu'on a vue en §IV.1, mais appliquée à un chemin **recentré sur son point de départ** (parce que le chemin n'est pas fermé, il faut choisir un point de référence — le départ est le choix naturel).

### Métaphore : la somme des petits triangles

La formule de Lévy a une **interprétation géométrique très concrète**. À chaque instant $r$, on a 3 points dans le plan :

- $X_s$ — le départ (**fixé**)
- $X_r$ — le point courant
- $X_{r + dr}$ — la position juste après, $dr$ plus tard

Ces 3 points forment un **petit triangle**. Et la quantité $\frac{1}{2}\left[(X^1_r - X^1_s)\,dX^2_r - (X^2_r - X^2_s)\,dX^1_r\right]$ calcule **l'aire signée de ce triangle** :

- **Positive** si la particule tourne dans le sens anti-horaire autour de $X_s$
- **Négative** si elle tourne dans le sens horaire

L'intégrale **somme toutes ces petites aires de triangles** le long du chemin. C'est exactement le déterminant des deux vecteurs $\binom{X^1_r - X^1_s}{X^2_r - X^2_s}$ (du départ au point courant) et $\binom{dX^1_r}{dX^2_r}$ (déplacement instantané) divisé par 2.

**Image mentale : l'élastique.** Imagine que tu attaches un élastique au point de départ $X_s$. Tu tiens l'autre bout et tu suis la particule qui parcourt le chemin. À chaque instant, l'élastique balaie une petite surface triangulaire :
- Si la particule s'éloigne **en tournant** systématiquement dans un sens, les triangles s'accumulent → grande aire de Lévy
- Si elle s'éloigne **en zigzags symétriques**, les triangles se compensent → aire ≈ 0
- Si elle s'éloigne **en ligne droite** depuis $X_s$, le triangle est dégénéré (3 points alignés) → aire nulle

![[fig7_triangles.png]]
*Figure 7. Décomposition explicite de l'aire de Lévy en somme de petits triangles. Chaque triangle a pour sommets $X_s$ (étoile verte), $X_r$, et $X_{r+1}$. Les rayons gris pointillés sont "l'élastique" attaché à $X_s$. **Vert** = aire signée positive (rotation anti-horaire). **Orange** = aire signée négative (rotation horaire). **Gauche** — chemin A presque droit : les triangles alternent vert et orange, leurs aires se compensent presque, total ≈ +0.031. **Droite** — chemin B en spirale anti-horaire : les triangles sont quasi tous verts (rotation systématique), les aires s'additionnent, total ≈ +0.200.*

### Construction progressive de l'aire

Un autre angle : on peut tracer l'aire de Lévy **cumulée** $A_{s,r}$ pour $r$ qui parcourt $[s, t]$. Ça montre comment l'aire se construit pas à pas le long du chemin.

![[fig8_levy_cumulative.png]]
*Figure 8. Aire de Lévy cumulée $A_{s,r}$ en fonction de $r$ (la variable d'intégration qui parcourt $[s, t] = [0, 1]$). **Gauche** — chemin A : la courbe oscille autour de 0. Chaque oscillation correspond à une période où le chemin tourne dans un sens, puis dans l'autre. Au final ≈ 0. **Droite** — chemin B en spirale : la courbe **croît globalement** (avec des oscillations dues à la spirale qui passe parfois plus près, parfois plus loin de $X_s$, mais sans changer de sens). Final ≈ +0.339, valeur nettement positive.*

**L'observation centrale** : pour deux chemins 2D qui ont **trajectoire visuellement similaire** (mêmes départ, arrivée, et même allure générale), l'aire de Lévy peut être **complètement différente**. Cette aire encode "comment le chemin tourne" — information **invisible** quand on ne regarde que les positions.

![[fig6_levy_area.png]]
*Figure 6. Deux chemins 2D allant de $(0,0)$ à $(2, 1)$ avec une trajectoire enveloppe similaire. **Gauche** — chemin A presque droit, oscillations symétriques. Aire de Lévy $\approx 0$ (les contributions positives et négatives s'annulent). **Droite** — chemin B en spirale anti-horaire (CCW). Même départ, même arrivée, même trajectoire moyenne. Mais aire de Lévy nettement positive (≈ 0.4) parce que le chemin tourne systématiquement dans le même sens. **La trajectoire seule ne distingue pas A de B — l'aire de Lévy oui.***

> 💡 **Pourquoi en 1D ça n'a pas d'intérêt.** En dimension 1, $X_t \in \mathbb{R}$ et l'aire de Lévy est triviale ($A_{s,t} = 0$ ou se ramène à $(X_t-X_s)^2/2$ qui est connu dès qu'on connaît la trajectoire). Le sujet rough paths est **intrinsèquement multi-dimensionnel** — il devient intéressant dès $d \geq 2$. C'est ce qu'on n'a pas dit clairement dans les sections précédentes : nos figures 1D étaient de la motivation, pas le vrai sujet.

### IV.3 La solution de Lyons : enrichir avec l'aire de Lévy

L'idée centrale de Terence Lyons : pour intégrer contre un signal trop irrégulier, **enrichir** $X$ avec son intégrale itérée (qui contient l'aire de Lévy) :

$$\mathbb{X}_{s,t} = \int_s^t (X_r - X_s) \otimes dX_r$$

C'est une matrice $d \times d$ dont la partie antisymétrique est l'aire de Lévy. Le couple $(X, \mathbb{X})$ est appelé **rough path** d'ordre 2.

> [!note]- Décode de la notation $\otimes$ et structure de la matrice
> Le symbole $\otimes$ est le **produit tensoriel** (ou produit extérieur). Pour deux vecteurs $u, v \in \mathbb{R}^d$, $u \otimes v$ est la matrice $d \times d$ définie par $(u \otimes v)_{ij} = u_i v_j$. C'est l'équivalent de $u v^\top$ en notation matricielle classique.
> 
> Du coup en dimension 2, $\mathbb{X}_{s,t}$ est une matrice $2 \times 2$ :
> $\mathbb{X}_{s,t} = \begin{pmatrix} \int (X^1 - X^1_s)\,dX^1 & \int (X^1 - X^1_s)\,dX^2 \\ \int (X^2 - X^2_s)\,dX^1 & \int (X^2 - X^2_s)\,dX^2 \end{pmatrix}$
> 
> **Ce que contient chaque entrée** :
> - **Diagonale** ($i = j$) : $\int (X^i - X^i_s)\,dX^i = \frac{1}{2}(X^i_t - X^i_s)^2$ — info purement 1D, calculable depuis la trajectoire.
> - **Hors-diagonale** ($i \neq j$) : c'est là que vit l'aire de Lévy. Plus précisément, l'aire de Lévy entre composantes $i$ et $j$ est $A^{ij}_{s,t} = \frac{1}{2}(\mathbb{X}^{ij}_{s,t} - \mathbb{X}^{ji}_{s,t})$ — la **partie antisymétrique** de la matrice. Tu retrouves bien la formule de §IV.2.
> 
> **Pourquoi une matrice et pas juste un nombre ?** En dimension $d$, il y a $\binom{d}{2}$ paires de composantes, donc autant d'aires de Lévy distinctes. La matrice est juste une façon compacte de toutes les ranger en même temps que l'info diagonale (les $(X^i_t - X^i_s)^2$). En 2D il n'y a qu'une aire (entre composantes 1 et 2). En 3D il y en a 3 (1-2, 1-3, 2-3). Etc.

**Ce que ça résout** : pour un chemin lisse, $\mathbb{X}_{s,t}$ se calcule depuis $X$ — pas d'info nouvelle. Pour un chemin rugueux ($H < 1/2$), $\mathbb{X}_{s,t}$ **ne se déduit plus** de $X$ — il faut la donner séparément. Une fois ces deux infos disponibles, Lyons montre qu'on peut définir :

- L'intégrale $\int H_s\,dX_s$ proprement
- Les EDS $dY_t = f(Y_t)\,dX_t$ avec existence et unicité
- **Continuité par rapport à $(X, \mathbb{X})$** : si $(X^{(n)}, \mathbb{X}^{(n)}) \to (X, \mathbb{X})$, alors les solutions $Y^{(n)} \to Y$

C'est cette continuité qui permet d'approximer numériquement, de prendre des limites, de prouver des théorèmes.

> [!note]- Pour aller plus loin techniquement
> Le théorème principal de Lyons : si $X$ est un rough path $\alpha$-Hölder avec $\alpha > 1/3$ (ce qui inclut le fBm pour $H > 1/3$ avec niveau 2), alors la solution de l'EDS $dY = f(Y)\,dX$ existe et dépend continûment du couple $(X, \mathbb{X})$. Pour des chemins encore plus irréguliers (Hölder $\alpha \in (1/4, 1/3]$), il faut monter au niveau 3 (intégrale doublement itérée), etc. Référence : Friz-Hairer (2014) *A Course on Rough Paths* pour le formalisme complet.

### IV.4 D'où vient $\mathbb{X}$ en pratique ?

Question légitime : on a vu en Figure 7 qu'on calcule l'aire de Lévy par discrétisation (somme de petits triangles). Pourquoi ne suffit-il pas de faire ça pour un fBm avec $H < 1/2$ ?

Réponse : **parce que ça diverge**. C'est le même phénomène que la variation quadratique vue en §III. Plus le chemin est rugueux, plus les sommes de petits triangles sur la grille fine **ne convergent pas** vers une limite stable. Concrètement, si tu prends un fBm 2D avec $H = 0.2$ et que tu calcules l'aire de Lévy par discrétisation :

- Avec $n = 100$ points : tu obtiens une valeur
- Avec $n = 1000$ : tu obtiens une autre valeur
- Avec $n = 10000$ : encore une autre
- **Ça ne converge pas**

Donc même la formule de Lévy n'est pas calculable directement par discrétisation pour un chemin trop rugueux. Trois cas selon le contexte :

**Cas 1 — Chemin lisse ou Hölder $\geq 1/2$** : on calcule $\mathbb{X}_{s,t}$ par discrétisation, ça converge sans souci. C'est ce qu'on a fait pour Figure 7.

**Cas 2 — Brownien standard ($H = 1/2$)** : on calcule $\mathbb{X}_{s,t}$ comme limite d'intégrales d'Itô (ou de Stratonovich, le choix change la valeur — mais les deux donnent un $\mathbb{X}$ valide). Les sommes de Riemann convergent **en $L^2$**, comme en §III de [[02_Calcul d'Itô]] (isométrie d'Itô).

**Cas 3 — fBm avec $H < 1/2$** : on **renonce à calculer** $\mathbb{X}_{s,t}$ par discrétisation. À la place, on le **construit** par d'autres méthodes :
- Approximation par des chemins lisses qui convergent vers $X$, et on prend la limite des $\mathbb{X}^{(n)}$ associés (méthode de Wong-Zakai)
- Constructions explicites pour les processus gaussiens (existe pour le fBm)
- Ou simplement : on **suppose** $\mathbb{X}$ donné en entrée, comme une donnée du problème

C'est le tour de force conceptuel de Lyons : **on découple** le calcul de l'aire (qui peut diverger) et la définition de l'intégrale (qui utilise $\mathbb{X}$ une fois donné). Au lieu de calculer l'aire à partir du chemin, on **enrichit** le chemin avec son aire.

> [!note]- La relation de Chen : condition de cohérence sur $\mathbb{X}$
> Pour qu'un objet $\mathbb{X}_{s,t}$ donné en entrée soit un "vrai" rough path, il doit satisfaire une condition algébrique appelée **relation de Chen** :
> $\mathbb{X}_{s,t} - \mathbb{X}_{s,u} - \mathbb{X}_{u,t} = (X_u - X_s) \otimes (X_t - X_u)$
> pour tout $s \leq u \leq t$. 
> 
> **Lecture** : si on découpe l'intervalle $[s, t]$ en deux morceaux $[s, u] \cup [u, t]$, l'aire totale = aire de la première moitié + aire de la deuxième moitié + un terme correctif qui dépend des positions aux extrémités. C'est l'analogue stochastique de l'additivité des intégrales $\int_s^t = \int_s^u + \int_u^t$, sauf qu'ici un terme "croisé" apparaît à cause de l'aire (qui est sensible à la position absolue, pas seulement aux incréments).
> 
> Toute construction valide de $\mathbb{X}$ doit vérifier Chen — c'est la marque qu'elle représente bien une intégrale itérée.

> [!warning] Sujet difficile
> Rough paths est un sujet **techniquement avancé** qui demande de l'analyse fonctionnelle. L'objectif de cette note est :
> 1. Que tu saches que ça existe et **pourquoi** (Itô casse pour $H < 1/2$, l'aire de Lévy est l'info manquante)
> 2. Que tu saches **où chercher** si un jour tu en as besoin (Friz-Hairer 2014)
> 
> **Tu n'as pas besoin de manipuler rough paths** pour les interviews quant standard — c'est un sujet de recherche. Sauf cas très précis (rough volatility en finance), tu n'en feras jamais.

---

## V. Applications modernes

### V.1 Rough volatility (Bayer-Friz-Gatheral 2016)

L'application la plus marquante en finance. **Observation empirique** : la volatilité réalisée des actifs financiers, observée sur des données haute fréquence, suit un processus dont l'exposant de Hurst est environ $H \approx 0.1$. Donc **très loin du brownien standard** ($H = 0.5$), et même très loin de Heston classique.

Le modèle "rough volatility" :

$$\sigma_t = \sigma_0 \exp(\eta B^H_t), \quad H \approx 0.1$$

ajuste **beaucoup mieux** le smile de volatilité observé sur les options que les modèles classiques. C'est devenu un sujet de recherche très actif en finance quantitative.

![[fig4_rough_vol.png]]
*Figure 4. Comparaison de deux modèles de volatilité $\sigma_t = \sigma_0 \exp(\eta B^H_t)$. **Haut** : avec $H = 0.5$ (vol classique type Heston), trajectoires modérément rugueuses. **Bas** : avec $H = 0.1$ (rough vol Bayer-Friz-Gatheral), trajectoires extrêmement rugueuses — c'est ce qu'on observe empiriquement sur les données réelles de volatilité.*

### V.2 Path signature en Machine Learning

La **path signature** (Chen 1957, redécouverte et développée par Lyons et Kidger) est une feature transform qui transforme une trajectoire en une suite infinie de coefficients (intégrales itérées de tous les ordres). Propriétés exceptionnelles :

- **Caractérise complètement** la trajectoire (à reparamétrisation près)
- **Robuste au bruit** et aux variations de paramétrisation
- **Linéarise** beaucoup de problèmes d'apprentissage sur séries temporelles

Applications : médical (signaux EEG), finance (prédiction haute fréquence), reconnaissance d'écriture manuscrite. Voir Chevyrev-Kormilitzin (2016) pour un primer ML.

### V.3 Neural CDEs et Neural Rough DEs

Extension récente des Neural ODEs (Chen et al. 2018) aux entrées continues très bruitées. Au lieu de modéliser $\dot y = f(y, x)$, on modélise $dy = f(y)\,dX$ où $X$ est traité comme un rough path. Permet d'apprendre des dynamiques sur des séries temporelles irrégulièrement échantillonnées (c'est *le* cas pour les données médicales, par exemple).

---

## Récapitulatif

| Concept | Définition / formule |
|---|---|
| Processus gaussien | $(X_{t_1}, \ldots, X_{t_n})$ gaussien multivarié pour tout $n$, caractérisé par $m, K$ |
| Brownien standard | Gaussien centré, $K(s,t) = \min(s,t)$, $H = 1/2$ |
| fBm | Gaussien centré, $K(s,t) = \frac{1}{2}(\|s\|^{2H} + \|t\|^{2H} - \|s-t\|^{2H})$ |
| Auto-similarité fBm | $B^H_{ct} \stackrel{loi}{=} c^H B^H_t$ |
| Variation quadratique fBm | $+\infty$ si $H < 1/2$, $t$ si $H = 1/2$, $0$ si $H > 1/2$ |
| Pourquoi Itô casse | $B^H$ pour $H < 1/2$ n'est pas une semi-martingale |
| Solution Lyons | Enrichir $X$ avec $\mathbb{X}_{s,t} = \int (X-X_s) \otimes dX$ |
| Rough volatility | $\sigma_t = \sigma_0 \exp(\eta B^H_t)$, $H \approx 0.1$ (Bayer-Friz-Gatheral 2016) |

---

## Pour aller plus loin

**Références** :
- Lyons (1998) — *Differential equations driven by rough signals* (article fondateur)
- Friz-Hairer (2014) — *A Course on Rough Paths* (livre de référence pédagogique)
- Bayer-Friz-Gatheral (2016) — *Pricing under rough volatility*
- Chevyrev-Kormilitzin (2016) — *A Primer on the Signature Method in Machine Learning*

**Dans le vault** :
- Prérequis : [[01_Mouvement Brownien]], [[02_Calcul d'Itô]], [[03_Équations Différentielles Stochastiques]]
- Lien ML : [[04_Dynamique de Langevin]] (sampling et score)
- Évolutions des densités : [[05_Reverse-time SDE et Fokker-Planck]]

---

## Suite logique

**Précédent ← [[05_Reverse-time SDE et Fokker-Planck]]** : on a vu comment retourner le temps des EDS et le lien avec les diffusion models.

**Suivant → [[07_Contrôle Stochastique]]** : on revient au cadre standard ($H = 1/2$) et on s'intéresse à l'optimisation — comment piloter une EDS pour maximiser un objectif. HJB, allocation de Merton, optimal execution, lien avec le RL.

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. **[[06_Rough Paths]]** — (cette note) au-delà des semi-martingales (Lyons 1998)
7. [[07_Contrôle Stochastique]] — HJB, Merton, optimal execution, lien RL
8. [[08_Volterra Signatures]] — (placeholder) extension signature pour mémoire longue
