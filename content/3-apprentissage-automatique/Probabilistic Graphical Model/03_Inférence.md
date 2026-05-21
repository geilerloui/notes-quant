---
title: Inférence - PGM
---
# Inférence

> Cette note couvre la deuxième des trois parties des PGM : **comment poser des questions au modèle**. On voit d'abord le cadre général (types de requêtes, sum-product), puis les algorithmes principaux (Variable Elimination, Belief Propagation, MAP), puis l'inférence dans les **modèles temporels** (HMM, Kalman) et enfin l'**inférence par échantillonnage** (ancestral, Gibbs).

> Pré-requis : [[02_Représentation]] (Bayesian Networks, Markov Random Fields, factorisation, indépendance).

---

## I. Vue d'ensemble

On peut poser plein de requêtes sur un graphe, mais la plus commune est probablement la **conditional probability query** :

- **Évidence** : $E = e$
- **Requête** : un sous-ensemble de variables $Y$
- **Objectif** : calculer $p(Y \mid E = e)$

> [!example] BN — exemple de l'étudiant complété
> On reprend le BN de l'étudiant ([[02_Représentation]]) avec une variable supplémentaire $C$ (cohérence du cours, etc.). L'inférence sur les PGM utilise la notion de **facteur** : $P(G \mid I, D)$ se convertit en $\phi_G(G, I, D)$.
>
> Si l'on veut calculer $P(J)$, il suffit de **marginaliser** par rapport à toutes les variables sauf $J$.

> 💡 **Sum-product.** On parle de "sum-product" parce qu'on a une **somme** sur un **produit** de facteurs.

> [!example] MRF — sum-product avec normalisation
> Pour un MRF, le sum-product donne une mesure **non normalisée** :
>
> $$p(D) = \frac{1}{Z} \sum \phi(A, B).$$

**Évidence dans un MRF.** Par définition :

$$\boxed{P(Y \mid E = e) = \frac{P(Y, E = e)}{P(E = e)}.}$$

On peut écrire la jointe comme un sum-product :

$$P(Y, E = e) = \sum_W P(Y, W, E = e) = \sum_W \frac{1}{Z} \prod_k \phi_k(D_k, E = e) = \sum_W \frac{1}{Z} \prod_k \phi_k'(D_k').$$

### A. Algorithmes pour calculer une probabilité conditionnelle

> [!warning] Familles d'algorithmes
> - **Variable Elimination** (élimination de variables)
> - **Belief Propagation** (passage de messages) — généralisation, avec ses variantes : sum-product et max-product
> - **Approximations variationnelles** ([[Variational Inference]])
> - **Sampling-based** (méthodes Monte-Carlo, [[MCMC]])

---

## II. Variable Elimination

*À développer.*

---

## III. Belief Propagation

*À développer.*

### A. Message passing matriciel

*À développer.*

Bibliographie :
- [Vidéo](https://www.youtube.com/watch?v=ijmxpItkRjc&t=203s)
- [Notebook GitHub](https://github.com/zjost/blog_code/blob/master/gcn_numpy/message_passing.ipynb)
- [Blog de l'auteur](https://blog.zakjost.com/)
- [Slides IPAM/UCLA](http://helper.ipam.ucla.edu/publications/gss2013/gss2013_11344.pdf)

### B. MAP inference

*À développer.*

---

## IV. Inférence dans les modèles temporels

> 💡 **Le contexte large.** Dans le cours de Daphne Koller, cette partie est traitée sous le nom de **Dynamic Bayesian Networks (DBN)**. Un DBN est une généralisation où les variables se répètent au fil du temps, avec une structure de dépendance entre tranches temporelles. Le **HMM** est le cas particulier le plus simple : une seule variable latente discrète + une seule variable d'observation par tranche temporelle. Le **Kalman Filter** est son analogue continu (latent gaussien).

### A. Hidden Markov Models (HMM) — variables latentes discrètes

#### A.1 Trois représentations

> [!warning] Définition (HMM)
> Un HMM modélise des données séquentielles avec une **chaîne de Markov de variables latentes** $z_t$, où chaque observation $x_t$ est conditionnée à sa variable latente correspondante.

**(i) Graphe avec chaîne de latentes.**

![[images/3-Apprentissage automatique/PGM/Inférence/HMM/im1.png|431]]
**Figure 1.** Représentation graphique : la chaîne de Markov des $z_t$ avec les observations $x_t$ qui en dépendent.

**(ii) Diagramme de transition** (3 états cachés possibles).

![[images/3-Apprentissage automatique/PGM/Inférence/HMM/im2.png|335]]
**Figure 2.** Diagramme de transition entre les états latents.

**(iii) Représentation en treillis (lattice / trellis).**

![[images/3-Apprentissage automatique/PGM/Inférence/HMM/im3 (1).png|436]]
**Figure 3.** Représentation en treillis — déroulement temporel des états latents.

**Distribution jointe.**

$$\boxed{p(\mathbf{x}_1, \ldots, \mathbf{x}_T, \mathbf{z}_1, \ldots, \mathbf{z}_T) = p(\mathbf{z}_1) \left[\prod_{t=2}^T p(\mathbf{z}_t \mid \mathbf{z}_{t-1})\right] \prod_{t=1}^T p(\mathbf{x}_t \mid \mathbf{z}_t)}$$

#### A.2 Problèmes d'intérêt

> [!warning] Inférence — calculer la probabilité des états cachés sachant les observations
> - **Filtering** : $p(z_t \mid x_1, \ldots, x_t)$ — distribution de l'état caché courant sachant les observations passées et présentes.
> - **Prediction** : $p(z_{t+k} \mid x_1, \ldots, x_t)$, $k > 0$ — anticipation d'un état caché futur.
> - **Smoothing** : $p(z_k \mid x_1, \ldots, x_T)$ — distribution d'un état caché passé sachant **toute** la séquence (passé et futur).
> - **Most likely sequence** :
>
> $$\arg\max_{z_1, \ldots, z_T} p(z_1, \ldots, z_T \mid x_1, \ldots, x_T) = \arg\max_{z_1, \ldots, z_T} p(z_1, \ldots, z_T, x_1, \ldots, x_T).$$

**Apprentissage** : déterminer les paramètres des matrices de transition (cf. [[#A.6 Apprentissage des paramètres — Baum-Welch|Baum-Welch]]).

Bibliographie :
- Applications : [time series](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/)
- [Markov model intro](https://blog.quantinsti.com/markov-model/)

#### A.3 Forward Algorithm — étape α (de gauche à droite)

**Dérivation théorique.**

$$\begin{aligned}
\alpha(i, t+1) &= p(X_{1:t+1} = x_{1:t+1}, Z_{t+1} = i) \\
&= \sum_j p(X_{1:t+1} = x_{1:t+1}, Z_t = j, Z_{t+1} = i) \\
&= \sum_j p(X_{1:t} = x_{1:t}, Z_t = j, Z_{t+1} = i, X_{t+1} = x_{t+1}) \\
&= \underbrace{p(X_{t+1} = x_{t+1} \mid Z_{t+1} = i)}_{p(x_3 \mid x_1, x_2)} \sum_j \underbrace{p(Z_{t+1} = i \mid Z_t = j)}_{p(x_2 \mid x_1)} \underbrace{p(X_{1:t} = x_{1:t}, Z_t = j)}_{p(x_1)} \\
&= p(X_{t+1} = x_{t+1} \mid Z_{t+1} = i) \sum_j p(Z_{t+1} = i \mid Z_t = j) \, \alpha(j, t).
\end{aligned}$$

(On a utilisé $Z_{t+1} \perp X_{1:t}$ et la chain rule $p(x_1, x_2, x_3) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2)$.)

> [!warning] Récursion forward
> $$\boxed{\alpha(i, t+1) = p(X_{t+1} = x_{t+1} \mid Z_{t+1} = i) \sum_j p(Z_{t+1} = i \mid Z_t = j) \, \alpha(j, t)}$$
>
> **Initialisation :**
>
> $$\boxed{\alpha(i, 1) = p(X_1 = x_1, Z_1 = i) = p(X_1 = x_1 \mid Z_1 = i) \, p(Z_1 = i) \quad \forall i}$$

Une fois la table $\alpha$ remplie :

$$p(X_{1:T} = x_{1:T}) = \sum_j p(X_{1:T}, Z_T = j) = \sum_j \alpha(j, T).$$

> [!example] Exemple — message sur canal bruité ($T = 4$)
> Observations : $x_1 = 0, x_2 = 0, x_3 = 0, x_4 = 1$.
>
> **Matrice de transition $A$** :
>
> | | $Z_{t-1} = 0$ | $Z_{t-1} = 1$ |
> | :--- | :---: | :---: |
> | $p(Z_t = 0 \mid Z_{t-1})$ | 0.3 | 0.6 |
> | $p(Z_t = 1 \mid Z_{t-1})$ | 0.7 | 0.4 |
>
> **Matrice d'observation $B$** :
>
> | | $Z_t = 0$ | $Z_t = 1$ |
> | :--- | :---: | :---: |
> | $p(X_t = 0 \mid Z_t)$ | 0.9 | 0.2 |
> | $p(X_t = 1 \mid Z_t)$ | 0.1 | 0.8 |
>
> **Distribution initiale $\pi$** :
>
> | | $Z_1 = 0$ | $Z_1 = 1$ |
> | :--- | :---: | :---: |
> | $p(Z_1)$ | 0.5 | 0.5 |
>
> **(i) Initialisation.**
>
> $$\begin{aligned}
> \alpha(0, 1) &= p(X_1 = 0 \mid Z_1 = 0) \, p(Z_1 = 0) = 0.9 \times 0.5 = 0.45 \\
> \alpha(1, 1) &= p(X_1 = 0 \mid Z_1 = 1) \, p(Z_1 = 1) = 0.2 \times 0.5 = 0.1
> \end{aligned}$$
>
> **(ii) Étape 2.**
>
> $$\begin{aligned}
> \alpha(0, 2) &= 0.9 \times (0.3 \times 0.45 + 0.6 \times 0.1) = 0.1755 \\
> \alpha(1, 2) &= 0.2 \times (0.7 \times 0.45 + 0.4 \times 0.1) = 0.071
> \end{aligned}$$
>
> **(iii) Étape 3.**
>
> $$\alpha(0, 3) = 0.9 \times (0.3 \times 0.1755 + 0.6 \times 0.071) = 0.085725$$
>
> **(iv) Étape 4.** En répétant le même processus, on obtient la table finale :
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | 0.45 | 0.1755 | 0.085725 | 0.004387 |
> | 1 | 0.1 | 0.071 | 0.03025 | 0.057686 |
>
> **(v) Inférence.**
>
> Marginale de l'observation :
>
> $$p(X_1 = 0, X_2 = 0, X_3 = 0, X_4 = 1) = \alpha(0, 4) + \alpha(1, 4) = 0.06207.$$
>
> **Filter** au temps 4 :
>
> $$\begin{aligned}
> p(Z_4 = 0 \mid X_{1:4}) &= \frac{\alpha(0, 4)}{\alpha(0, 4) + \alpha(1, 4)} = \frac{0.004387}{0.062073} \approx 0.0707 \\
> p(Z_4 = 1 \mid X_{1:4}) &= \frac{0.057686}{0.062073} \approx 0.9293
> \end{aligned}$$

#### A.4 Backward Algorithm — étape β (de droite à gauche)

**Dérivation théorique.**

$$\begin{aligned}
\beta(i, t-1) &= p(X_{t:T} = x_{t:T} \mid Z_{t-1} = i) \\
&= \sum_j p(X_{t:T} = x_{t:T}, Z_t = j \mid Z_{t-1} = i) \\
&= \sum_j \underbrace{p(X_{t+1:T} = x_{t+1:T}}_{x_2}, \underbrace{X_t = x_t}_{x_3}, \underbrace{Z_t = j}_{x_1} \mid \underbrace{Z_{t-1} = i)}_{x_4} \\
&= \sum_j p(X_t = x_t \mid Z_t = j) \, p(Z_t = j \mid Z_{t-1} = i) \, p(X_{t+1:T} = x_{t+1:T} \mid Z_t = j) \\
&= \sum_j p(X_t = x_t \mid Z_t = j) \, p(Z_t = j \mid Z_{t-1} = i) \, \beta(j, t).
\end{aligned}$$

(On a utilisé $p(x_1, x_2, x_3 \mid x_4) = p(x_1 \mid x_4) p(x_2 \mid x_1, x_4) p(x_3 \mid x_1, x_2, x_4)$.)

> [!warning] Récursion backward
> $$\boxed{\beta(i, t-1) = \sum_j p(X_t = x_t \mid Z_t = j) \, p(Z_t = j \mid Z_{t-1} = i) \, \beta(j, t)}$$
>
> **Initialisation :**
>
> $$\beta(i, T) = 1 \quad \forall i.$$

Une fois la table $\beta$ remplie :

$$p(X_{1:T} = x_{1:T}) = \sum_j \beta(j, 1) \, p(X_1 = x_1 \mid Z_1 = j) \, p(Z_1 = j).$$

> [!example] Exemple — backward sur la même séquence
> Mêmes paramètres :
>
> $$A = \begin{pmatrix} 0.3 & 0.6 \\ 0.7 & 0.4 \end{pmatrix}, \quad B = \begin{pmatrix} 0.9 & 0.2 \\ 0.1 & 0.8 \end{pmatrix}, \quad \pi = \begin{pmatrix} 0.5 & 0.5 \end{pmatrix}.$$
>
> **(i) Initialisation.** $\beta(i, 4) = 1$ pour tout $i$.
>
> **(ii) Étape $t = 3$.**
>
> $$\begin{aligned}
> \beta(0, 3) &= 0.1 \times 0.3 \times 1 + 0.8 \times 0.7 \times 1 = 0.59 \\
> \beta(1, 3) &= 0.1 \times 0.6 \times 1 + 0.8 \times 0.4 \times 1 = 0.38
> \end{aligned}$$
>
> **(iii) Étape $t = 2$.**
>
> $$\beta(0, 2) = 0.9 \times 0.3 \times 0.59 + 0.2 \times 0.7 \times 0.38 = 0.2125$$
>
> **(iv) Table finale.**
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | 0.106235 | 0.2125 | 0.59 | 1 |
> | 1 | 0.14267 | 0.349 | 0.38 | 1 |
>
> **(v) Smoothing.** En combinant $\alpha$ et $\beta$, on dérive la formule de smoothing :
>
> $$\begin{aligned}
> p(Z_k = i \mid X_{1:T}) &= \frac{p(Z_k = i, X_{1:k}) \, p(X_{k+1:T} \mid Z_k = i)}{\gamma} \\
> &= \frac{\alpha(i, k) \, \beta(i, k)}{\gamma}.
> \end{aligned}$$
>
> Probabilité de smoothing à $t = 2$ :
>
> $$\begin{aligned}
> p(Z_2 = 0 \mid X_{1:4}) &= \frac{\alpha(0, 2) \, \beta(0, 2)}{\alpha(0, 2) \beta(0, 2) + \alpha(1, 2) \beta(1, 2)} \\
> &= \frac{0.1755 \times 0.2125}{0.1755 \times 0.2125 + 0.071 \times 0.349} \approx 0.6008 \\
> p(Z_2 = 1 \mid X_{1:4}) &\approx 0.3992
> \end{aligned}$$
>
> On peut aussi faire un smoothing avec deux variables cachées adjacentes : $p(Z_k = i, Z_{k+1} = j \mid X_{1:T})$ — *à compléter*.

Bibliographie :
- [HMM notes (MIT)](https://people.csail.mit.edu/rameshvs/content/hmms.pdf)
- [HMM (San Jose State)](https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf) avec son [implémentation from scratch](https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e)

#### A.5 Prediction

Soit $\pi(i, k) = p(Z_{t+k} = i \mid X_{1:t} = x_{1:t})$.

**Dérivation de la récursion :**

$$\begin{aligned}
\pi(i, k+1) &= p(Z_{t+k+1} = i \mid X_{1:t} = x_{1:t}) \\
&= \sum_j p(Z_{t+k+1} = i, Z_{t+k} = j \mid X_{1:t}) \\
&= \sum_j p(Z_{t+k+1} = i \mid Z_{t+k} = j) \, p(Z_{t+k} = j \mid X_{1:t}) \\
&= \sum_j p(Z_{t+k+1} = i \mid Z_{t+k} = j) \, \pi(j, k).
\end{aligned}$$

> [!warning] Récursion de prediction
> $$\pi(i, k+1) = \sum_j p(Z_{t+k+1} = i \mid Z_{t+k} = j) \, \pi(j, k)$$
>
> **Initialisation** (à $k = 0$, c'est juste un filter) :
>
> $$\pi(i, 0) = p(Z_t = i \mid X_{1:t}) = \frac{\alpha(i, t)}{\sum_j \alpha(j, t)} \quad \forall i.$$

On peut aussi prédire $X_{t+k}$ :

$$p(X_{t+k} = x \mid X_{1:t} = x_{1:t}) = \sum_j p(X_{t+k} = x \mid Z_{t+k} = j) \, \pi(j, k).$$

> [!example] Exemple — prediction sur la même séquence
> Mêmes paramètres et table $\alpha$ qu'avant.
>
> **(i) Initialisation** ($k = 0$, filter en $t = 4$).
>
> $$\begin{aligned}
> \pi(0, 0) &= \frac{0.004387}{0.004387 + 0.057686} = 0.07071 \\
> \pi(1, 0) &= \frac{0.057686}{0.062073} = 0.92929
> \end{aligned}$$
>
> **(ii) Étape $k = 1$.**
>
> $$\begin{aligned}
> \pi(0, 1) &= 0.3 \times 0.07071 + 0.6 \times 0.92929 = 0.57879 \\
> \pi(1, 1) &= 0.7 \times 0.07071 + 0.4 \times 0.92929 = 0.42121
> \end{aligned}$$
>
> **(iii) Étape $k = 2$.**
>
> | $i \mid k$ | 0 | 1 | 2 |
> | :---: | :---: | :---: | :---: |
> | 0 | 0.07071 | 0.57879 | 0.42637 |
> | 1 | 0.92929 | 0.42121 | 0.57363 |
>
> **(iv) Inférence.** On peut prédire le prochain état caché :
>
> $$p(Z_6 = 0 \mid X_{1:4}) = \pi(0, 2) = 0.42637.$$

#### A.6 Max-Product Algorithm (Viterbi)

On a vu filtering, smoothing, prediction. La dernière inférence est la **most likely explanation** : la séquence complète d'états cachés qui explique le mieux la séquence d'observations. On a maintenant un **max** au lieu d'une somme.

> 💡 **Approche naïve = exponentiel.** Si on fait le max naïvement, le problème est exponentiel. Sur un exemple à 3 temps :
>
> $$\max_{z^*_{1:3}} p(z_1^*) p(x_1 \mid z_1^*) p(z_2^* \mid z_1^*) p(x_2 \mid z_2^*) p(z_3^* \mid z_2^*) p(x_3 \mid z_3^*)$$
>
> $$= \max_{z_3^*} p(x_3 \mid z_3^*) \, \max_{z_2^*} p(x_2 \mid z_2^*) p(z_3^* \mid z_2^*) \, \max_{z_1^*} p(z_2^* \mid z_1^*) p(z_1^*) p(x_1 \mid z_1^*).$$
>
> Pour retirer la complexité exponentielle, on stocke les $\max$ partiels dans une table — c'est exactement Viterbi.

Soit $\alpha^*(i, t) = p(X_{1:t} = x_{1:t}, Z_{1:t-1} = z_{1:t-1}^*, Z_t = i)$.

> [!warning] Récursion Viterbi
> $$\alpha^*(i, t+1) = p(X_{t+1} = x_{t+1} \mid Z_{t+1} = i) \, \max_j p(Z_{t+1} = i \mid Z_t = j) \, \alpha^*(j, t)$$
>
> **Initialisation** :
>
> $$\alpha^*(i, 1) = p(X_1 = x_1 \mid Z_1 = i) \, p(Z_1 = i) \quad \forall i.$$
>
> **Inférence** :
>
> $$p(X_{1:T} = x_{1:T}, Z_{1:T} = z_{1:T}^*) = \max_j \alpha^*(j, T).$$

Pour trouver la séquence optimale $z_{1:T}^*$, une fois $\alpha^*$ complétée, on **backtrack** stage par stage en prenant l'argmax dans chaque colonne.

> [!example] Exemple — Viterbi sur la même séquence
>
> **(i) Initialisation.**
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | 0.45 | | | |
> | 1 | 0.1 | | | |
>
> **(ii) Étape 2.**
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | 0.45 | 0.1215 | | |
> | 1 | 0.1 | 0.063 | | |
>
> **(iii) Étapes 3 et 4.**
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | 0.45 | 0.1215 | 0.03402 | 0.001021 |
> | 1 | 0.1 | 0.063 | 0.01701 | 0.019051 |
>
> **(iv) Inférence.** On backtrack et on prend les max colonne par colonne (en gras) :
>
> | $i \mid t$ | 1 | 2 | 3 | 4 |
> | :---: | :---: | :---: | :---: | :---: |
> | 0 | **0.45** | 0.1215 | **0.03402** | 0.001021 |
> | 1 | 0.1 | **0.063** | 0.01701 | **0.019051** |
>
> Séquence optimale : $Z_1 = 0, Z_2 = 1, Z_3 = 0, Z_4 = 1$.

#### A.7 Apprentissage des paramètres — Baum-Welch

*À développer.*

> 💡 **L'idée.** Baum-Welch est un cas particulier de l'algorithme **EM** (Expectation-Maximization) appliqué aux HMM. À chaque itération : (i) E-step calcule $\alpha$ et $\beta$ et les espérances de transitions/émissions, (ii) M-step met à jour $A$, $B$, $\pi$ par maximum de vraisemblance.

### B. Kalman Filter — variables latentes continues

*À développer.*

> 💡 **L'idée.** Kalman Filter = HMM avec latentes gaussiennes au lieu de discrètes, et transitions/émissions linéaires gaussiennes. Les opérations forward/backward deviennent des opérations sur les paramètres des gaussiennes (moyennes et covariances) au lieu de tables.

---

## V. Inférence par échantillonnage

> 💡 **L'idée générale.** Au lieu de calculer une espérance exactement, on tire des échantillons de la distribution et on moyenne. Famille complémentaire aux méthodes déterministes (Variable Elimination, BP) et aux approximations variationnelles ([[Variational Inference]]).

### A. Ancestral sampling

![[sampling1.png|466]]

L'algorithme :
1. Tirer un échantillon de $p(x_1)$.
2. Tirer un échantillon de $p(x_2 \mid x_1)$.
3. Tirer un échantillon de $p(x_3 \mid x_2)$.
4. Tirer un échantillon de $p(x_4 \mid x_3)$.

On obtient un échantillon $\{x_1, x_2, x_3, x_4\}$ de la distribution jointe.

![[sampling2.png|455]]

À partir de la jointe, on peut calculer les **probabilités marginales** :

| | val 0 | val 1 |
| :--- | :---: | :---: |
| marginale $x_1$ | 0.7000 | 0.3000 |
| marginale $x_2$ | 0.4300 | 0.5700 |
| marginale $x_3$ | 0.4580 | 0.5420 |
| marginale $x_4$ | 0.6084 | 0.3916 |

> 💡 **Limite.** Ancestral sampling marche pour les BN (graphe orienté → ordre topologique des nœuds). Pour les MRF (non orienté), on n'a pas d'ordre naturel — il faut Gibbs sampling.

### B. Gibbs sampling (graphes non orientés)

*Voir [[MCMC]] — Gibbs sampling y est traité en détail comme cas particulier de MCMC.*

> 💡 **L'idée en une phrase.** À chaque itération, on échantillonne une variable conditionnellement à toutes les autres : $x_i^{(t+1)} \sim p(x_i \mid x_{-i}^{(t)})$. Pour un MRF, ces conditionnelles ne dépendent que des voisins de $x_i$ dans le graphe, donc le calcul est local.
