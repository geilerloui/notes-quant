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

## IV. La solution de Lyons (1998) — esquisse

L'idée centrale de Terence Lyons : pour intégrer contre un signal trop irrégulier, **enrichir** $X$ avec son intégrale itérée :

$$\mathbb{X}_{s,t} = \int_s^t (X_r - X_s) \otimes dX_r$$

Le couple $(X, \mathbb{X})$ est appelé **rough path** d'ordre 2. L'intégration et les EDS deviennent alors des **opérations continues** dans la bonne topologie (le théorème de continuité de Lyons).

L'intuition : un signal très rugueux ne peut pas être caractérisé uniquement par sa trajectoire, il faut aussi spécifier comment il "tourne" autour d'elle (l'intégrale itérée). Une fois ces deux infos données, on peut faire du calcul stochastique cohérent, même hors du cadre semi-martingale.

> [!note]- Pour aller plus loin techniquement
> Le théorème principal de Lyons : si $X$ est un rough path $\alpha$-Hölder avec $\alpha > 1/3$ (ce qui inclut le fBm pour $H > 1/3$ avec niveau 2), alors la solution de l'EDS $dY = f(Y)\,dX$ existe et dépend continûment du couple $(X, \mathbb{X})$. Pour des chemins encore plus irréguliers (Hölder $\alpha \in (1/4, 1/3]$), il faut monter au niveau 3 (intégrale doublement itérée), etc. Référence : Friz-Hairer (2014) *A Course on Rough Paths* pour le formalisme complet.

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

Progression complète du sujet :
1. [[01_Mouvement Brownien]] — construction de $W_t$, propriétés bizarres
2. [[02_Calcul d'Itô]] — formalisme rigoureux de $\int H\,dW$
3. [[03_Équations Différentielles Stochastiques]] — modèles classiques (ABM, MBG, OU)
4. [[04_Dynamique de Langevin]] — sampling et lien score-based ML
5. [[05_Reverse-time SDE et Fokker-Planck]] — EDP de l'évolution de la densité et inversion du temps
6. **[[06_Rough Paths]]** — (cette note) au-delà des semi-martingales (Lyons 1998)
