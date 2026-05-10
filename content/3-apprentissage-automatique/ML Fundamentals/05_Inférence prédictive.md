# (v) Inférence prédictive

> *Predictive inference* = ce qu'on peut dire de **fiable** sur la prédiction d'un point individuel, au-delà du score brut. Deux outils principaux qui ne servent pas exactement la même chose : **calibration** (les probabilités prédites ont-elles vraiment un sens ?) et **conformal prediction** (peut-on construire un intervalle/ensemble de prédiction avec une garantie statistique ?).

> 💡 **Pourquoi c'est important.** En production, un score brut ne suffit pas. Un modèle qui dit "0.9 de probabilité" doit avoir 90% de raison s'il est calibré. Et pour un patient ou un client, on veut souvent dire *"la vraie valeur est entre X et Y avec 90% de chances"* plutôt qu'un point estimate. Conformal prediction est en train de devenir un standard industriel pour ça.

---

## I. Calibration

### A. Le problème

Beaucoup de modèles de classification produisent des "probabilités" qui ne sont **pas vraiment des probabilités**. Un modèle qui dit $\hat p = 0.9$ pour 100 clients devrait, si bien calibré, voir environ 90 d'entre eux être réellement positifs. Ce n'est pas garanti par défaut.

> [!warning] Comportement par modèle
> - **Logistic regression (sans repondération)** : naturellement bien calibrée — c'est un cas particulier des modèles linéaires généralisés où une loi de probabilité (Bernoulli) est associée à $Y$ par construction.
> - **Random Forest, SVM, Naive Bayes** : mal calibrés — scores poussés vers les extrêmes ou écrasés au milieu.
> - **Réseaux de neurones profonds** : tendance à l'**overconfidence** (prédisent 0.99 souvent à tort, surtout avec batch norm + dropout).

> 💡 **Pourquoi le SVM n'est pas calibré.** SVM n'a pas de méthode `predict_proba` mais une `decision_function` qui renvoie la **distance à l'hyperplan séparateur**. Cette distance n'est pas comprise entre 0 et 1 — ce n'est pas une probabilité. La calibration sigmoïde (Platt scaling) sert précisément à transformer ces distances en probabilités.

> 💡 **Random Forest et la sigmoïde inverse.** Le RF montre souvent un histogramme de probabilités avec des **pics à 0.2 et 0.8** mais peu de valeurs proches de 0 ou 1. Raison : pour avoir une moyenne d'arbres très proche de 0 ou 1, il faudrait que **tous** les weak learners soient d'accord — rare à cause du bagging qui injecte du bruit. Le RF est donc *underconfident* aux extrêmes.

### B. Méthodes de calibration post hoc

> [!warning] Idée générale (post hoc calibration)
> Deux stratégies possibles : (a) concevoir un algo qui produit naturellement des probas calibrées, ou (b) **post-traiter** les probas du modèle déjà entraîné. La voie (b) est de loin la plus utilisée.

**Pseudo-code post hoc :**

> [!note]- Pipeline en 3 étapes
> 1. Diviser les données en **training set** et **calibration set**.
> 2. Entraîner le classifieur normalement sur le training set (max accuracy, AUC, etc.).
> 3. Apprendre une **fonction de calibration** $g$ qui transforme $\hat p$ en $g(\hat p)$ bien calibrée.
>
> ![[im0-1.png]]

> 💡 **Le piège du leakage.** On entraîne $g$ sur le **calibration set**, pas sur le training set. Sinon on crée du bias parce que le modèle a déjà vu ces points. Et il faut **un troisième set de validation** pour évaluer si la calibration a bien marché.
>
> ![[im0-2.png]]

### C. Diagnostic — Reliability diagram

> [!warning] Définition
> Pour un problème de classification binaire, le **reliability diagram** trace, pour chaque bin de probabilités prédites, la **fréquence empirique** des positifs vs la **moyenne des probabilités prédites**.

![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/im1-1.png]]
**Figure 1.** En binaire, on entraîne un modèle à estimer $f(x_i) = p(y_i = 1 \mid x_i)$.

**Construction du diagramme** (3 étapes) :

> [!note]- Étape (i) — bucketization
> Diviser $[0, 1]$ en $M$ buckets (souvent $M = 10$). Buckets de largeur égale ou de quantiles égaux selon $\hat p(y = 1 \mid x)$.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/im1-2.png]]

> [!note]- Étape (ii) — calculs par bin
> Pour le bin $b \in \{1, \ldots, M\}$, soit $B_b$ l'ensemble des points dont la proba tombe dans $B_b$.
>
> **Fréquence relative** (fraction de prédictions correctes) :
>
> $$\hat P(B_b) = \frac{1}{|B_b|} \sum_{x \in B_b} \mathbb{I}[y = 1].$$
>
> **Moyenne des probabilités prédites** :
>
> $$\hat p(B_b) = \frac{1}{|B_b|} \sum_{x \in B_b} \hat p(y = 1 \mid x).$$
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/im1-3.png]]
> ![[im1-4.png]]

**Étape (iii) — le plot.**

![[im1-5.png]]
**Figure 2.** $x$-axis : moyenne des probas prédites ; $y$-axis : fréquence empirique des positifs. Diagonale = calibration parfaite.

> 💡 **Lecture.** Si la proba moyenne prédite vaut 0.17 mais que 33% des prédictions sont positives → le modèle est **sous-confiant** dans cette zone. Inversement, proba 0.82 et 80% positifs → légèrement sous-confiant aussi.

#### C.1 Exemple visuel — comparaison de modèles

> [!example] Lecture comparative
> - **Régression logistique** : modèle le mieux calibré, proche de la diagonale.
> - **SVM** : sigmoïde encore plus éloignée de la diagonale (les distances à l'hyperplan ne sont pas des probas).
> - **Random Forest** : pics à 0.2 et 0.8, sigmoïde indiquant une **sous-confiance** aux extrêmes.
>
> ![[im1-6.png]]
> ![[im1-7.png]]

> [!example] Lecture KNN, perceptron, NN profond
> - **KNN** : sigmoïde — la "proba" est la fraction des $k$ voisins qui sont 1, pondérée par 1/distance. Concept particulier de probabilité.
> - **Perceptron 1 couche** : assez bien calibré (fonction linéaire + sigmoïde en sortie).
> - **NN Keras 2+ couches** : moins bien calibré. Les techniques pour éviter vanishing gradient et dying ReLU dégradent la calibration.
>
> ![[im1-9.png]]

> [!warning] Attention au déséquilibre de classes
> Les techniques de **sous/sur-échantillonnage** pour résoudre le problème des classes déséquilibrées modifient la distribution a priori et engendrent des probabilités très mal calibrées. Si tu fais du SMOTE et que tu veux des probas calibrées en prod, **recalibre après**.

### D. Quand calibrer ?

> 💡 **Référence.** [Niculescu-Mizil & Caruana 2005](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) ont étudié empiriquement cette question. Les algos qui bénéficient le plus de la calibration : **SVM, decision trees baggés, random forests**.

**Cas d'usage typiques où calibrer matter :**
- **Marketing** : on évalue la *Customer Lifetime Value* en multipliant le prix par la proba d'achat. CLV $= 200\text{€} \times 0.1 = 20\text{€}$. Si la proba n'est pas calibrée, le calcul est faux.
- **Médecine** : l'ordre des probas de cancer importe peu au patient, ce qui compte c'est la **vraie** probabilité.
- **Assurance** : pricing fonction de la proba de sinistre.
- **Credit scoring** : décisions seuillées.

### E. Métriques d'évaluation — Brier score

> 💡 **Motivation.** Deux modèles qui prédisent correctement le temps ensoleillé. L'un avec $p = 0.51$, l'autre avec $p = 0.93$. Même accuracy au seuil 0.5. Mais le second est clairement meilleur. Le **Brier score** capture cette nuance.

> [!warning] Définition
> $$BS = \frac{1}{n} \sum_{i=1}^n (p_i - o_i)^2$$
>
> où $p_i$ = proba prédite, $o_i = 1$ si l'événement s'est produit, 0 sinon. C'est essentiellement une **MSE** sur les probas. Plus bas = mieux.

> [!example] Calcul
> Si on prédit 60% pour un événement qui ne se produit pas : $BS = (0.6 - 0)^2 = 0.36$.

> [!example] Pièges de l'AUC
> Reliability curve d'un SVM ill-calibré : $\text{AUC} = 0.89$, $BS = 0.49$. **L'AUC élevée nous trompe** — elle nous dit que le ranking est bon, mais le Brier score révèle que les probas elles-mêmes sont fausses.
>
> ![[im4-9.png]]

### F. Méthodes de recalibration (binaire)

#### F.1 Binning

> [!warning] Méthode par bins fixes
> 1. Trier les données par $\hat p$.
> 2. $B_1, \ldots, B_M$ chacun de largeur $1/M$.
> 3. Estimer $\hat P(B_b)$ pour chaque bin.
> 4. $g(\hat p) = \hat P(B_b)$ pour le bin $B_b$ contenant $\hat p$.

> [!example] Exemple visuel
> Bin $[0.9, 1]$ — la proba moyenne prédite est 0.95 ($x$-axis), mais la fréquence empirique est 0.78 → on remplace 0.95 par 0.78 dans la version calibrée.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-1 (2).png]]

**Variante quantile bins.** Définir les bins de sorte que chacun contienne $1/M$ des données d'entraînement. Plus stable quand la distribution des probas est inégale.

#### F.2 Platt scaling

> 💡 **Origine.** John Platt 1999, motivé par le SVM (qui ne sort pas des probas). Idée : appliquer une **sigmoïde** aux scores bruts.

$$\boxed{g(\hat p; a, b) = \frac{1}{1 + e^{a + b \hat p}}}$$

Les paramètres $a, b$ sont appris par MLE sur le calibration set (régression logistique sur les scores).

> [!example] Adult dataset — calibration SVM
> Le reliability diagram fit bien à une sigmoïde. SVM score $= -1$ → proba calibrée 0.2. SVM score entre 1 et 2 → proba 0.82.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-2.png]]

> 💡 **Limite de Platt.** Suppose que le décalage est de **forme sigmoïdale**. Si la mauvaise calibration n'a pas cette forme, ça ne marche pas. → Isotonic regression.

#### F.3 Isotonic regression

> [!warning] Définition
> Régression sous **contrainte de monotonie** :
>
> $$\min_f \sum_{i=1}^n (y_i - \hat y_i)^2 \quad \text{s.t.} \quad f(x_1) \leq f(x_2) \leq \ldots \leq f(x_k).$$

> 💡 **Intuition métier.** La contrainte de monotonie permet d'**injecter du domain knowledge** : si tu sais que ta proba doit toujours croître avec une feature (ex : grain size → porosity), tu l'imposes.

**Algorithme : Pool-Adjacent Violators (PAV).** Résolution itérative du problème ci-dessus.

> [!note]- Hyperparamètre $K$
> $K$ = nombre de seuils (isotonic constraints). Trop petit → underfit, trop grand → overfit. À tuner par validation croisée.

**Fitting du modèle.** Le modèle est paramétrisé par les prédictions aux seuils $f(x_1), \ldots, f(x_K)$. Pour un nouveau point :

$$f(x_0) = f(x_{k-1}) + (x_0 - x_{k-1}) \cdot \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}, \quad x_{k-1} \leq x_0 \leq x_k.$$

![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im2-2.png]]

#### F.4 Variantes avancées

> [!warning] mPAVA (Hastie & Tibshirani 2011)
> Ajoute un terme de régularisation pour produire une régression "**near-isotonic**" qui tolère quelques petites violations :
>
> $$\frac{1}{2} \sum_{i=1}^N (\hat P_i - \hat p_i)^2 + \lambda \sum_{i=1}^{N-1} (\hat P_i - \hat P_{i+1}) \mathbb{I}[\hat P_i > \hat P_{i+1}]$$

> [!warning] ENIR — Naeini & Cooper 2018
> Ensemble de near-isotonic regressions. Calcule le BIC score de chaque $g_t$, le normalise, et fait une **moyenne pondérée** pour obtenir $g$.

> [!example] Visualisation isotonic
> Bin de tailles différentes avec proba empirique sur l'$y$-axis — c'est essentiellement un **binning adaptatif**.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-3.png]]

#### F.5 Modèles avancés

> 💡 **Au-delà de Platt et isotonic.**
> - **Splines** (Lucena 2018, arxiv 1809.07751)
> - **Piecewise linear** via décomposition par arbre (Leathart, Frank, Holmes, Pfahringer 2017)
> - **Gaussian Processes** (Song, Kull, Flach 2018)

#### F.6 Temperature scaling pour les NN

> 💡 **Cas particulier des NN.** On divise les **logits** par une température $T$ avant le softmax :
>
> $$\text{softmax}\!\left( \frac{z}{T} \right).$$
>
> Un seul paramètre $T$ à optimiser sur le calibration set. $T > 1$ adoucit, $T < 1$ rend plus aigu. C'est la méthode standard pour les NN modernes (Guo et al. 2017).

### G. Calibration multiclasse

> [!warning] Méthode One-vs-Rest normalisée
> Pour chaque classe $k$, apprendre une fonction de calibration binaire $g_k$ basée sur un classifieur **one-vs-rest**, puis renormaliser pour que les probas somment à 1.

---

## II. Conformal Prediction

### A. Pourquoi la calibration ne suffit pas

> [!warning] Limites de la calibration
> - Elle ne donne pas des **probas parfaitement calibrées** mais juste **mieux calibrées**.
> - Elle reste un **point estimate** — elle ne porte pas les attributs attendus d'une vraie quantification d'incertitude.

> 💡 **L'analogie.** Calibration ≈ "mes scores moyens sont corrects" mais ne dit rien sur la **distribution** autour de la prédiction. Conformal prediction donne un **ensemble de prédictions plausibles**.

### B. Définition

> 💡 **Référence.** [Vovk et al. 2005](https://link.springer.com/book/10.1007/b106715), approche frequentiste autour de tests d'hypothèse, fournit des bornes d'erreur **par instance** sans spécifier de prior.

> [!warning] Définition (Conformal Prediction)
> Approche qui produit des bornes d'erreur autour des prédictions :
> - **Régresseurs** : les régions sont des **intervalles** autour de la prédiction.
> - **Classifieurs** : prédictions **set-valued** dans le power-set des classes.

> [!example] Cas médical — IRM cerveau
> Imagine que tu es médecin et que tu reçois une IRM avec la sortie d'un algo ML. Tu veux un point estimate ("normal") mais en tant que médecin tu fais du *differential diagnosis*. Tu veux savoir s'il y a 5% de chance que ce soit un cancer.
>
> Le **prediction set** te dit :
>
> $$\mathbb{P}\big[\text{vrai diagnostic} \in \{\text{normal, concussion, cancer}\}\big] \geq 90\%.$$
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im2-1 (2).png]]

### C. Setup et garantie

**Inputs (cas classification) :**
- **Calibration dataset** $\{(x_i, y_i)\}_{i=1}^n \sim \mathbb{P}$ i.i.d. (par exemple $x_i \in \mathbb{R}^d$ image, $y_i \in \{1, \ldots, K\}$).
- **Modèle** $\hat \pi_y(x)$ qui estime $\pi_y(x) = \mathbb{P}[Y = y \mid X = x]$.
- **Nouvelle image** $x_{n+1}$.

**Sortie :** un set $\tau(x_{n+1}) \subseteq \mathcal{Y}$.

> [!warning] Coverage — la garantie centrale
> $$\boxed{\mathbb{P}[y_{n+1} \in \tau(x_{n+1})] \geq 1 - \alpha}$$
>
> où $\alpha$ est l'**error rate** (souvent 0.1 → coverage 90%). $\geq$ donne un **conservative coverage**, $=$ donne **exact coverage**.

> [!example] Adaptive prediction sets
> À gauche cas facile, prediction set d'une seule classe. À droite, image complexe (marmotte ?), prediction set large.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im2-2.png]]

> [!warning] Trois objectifs pour un bon prediction set
> - **Exact coverage** — au moins conservative, le plus proche possible d'exact.
> - **Petite taille** $|\tau(x)|$ — sinon trivial d'avoir exact coverage en prenant tout.
> - **Adaptatif** — petit pour les exemples faciles, grand pour les durs.

### D. Algorithme de base — exemple MNIST

> [!example] Vovk et al. — pas-à-pas
> 1. **Calibration set** $\{(X_i, y_i)\}_{i=1}^n$ sur MNIST.
> 2. Pour chaque point de calibration, on récupère le **score de la vraie classe** depuis le softmax. C'est le **conformal score** $E_i$.
> 3. On plotte la distribution des $[E_1, \ldots, E_n]$ et on prend le **quantile à 10%** (= $\hat q$).
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-1 (2).png]]
>
> 4. Sur la query $x_{n+1}$, on passe par le NN qui sort un softmax. On **threshold** avec $\hat q$. Le prediction set est l'ensemble des classes dont le score dépasse $\hat q$ :
>
> $$\tau(x_{n+1}) = \{2, 5, 8\}.$$
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-2.png]]

> [!warning] Garantie de coverage (encadrement)
> $$1 - \alpha \leq \mathbb{P}[y_{n+1} \in \tau(X_{n+1})] \leq 1 - \alpha + \frac{1}{n + 1}$$
>
> Non seulement on a $1 - \alpha$ coverage, mais le set n'est pas trop gros.

### E. Cadre général de Conformal Prediction

> [!note]- Pseudo-code générique
> 1. **Notion heuristique d'incertitude.** Ex : softmax score (confiance que $x$ appartient à $y$).
> 2. **Score function** $s(X, Y) \in \mathbb{R}$. Plus grand = pire fit entre $Y$ et $X$.
> 3. **Quantile.** Calculer $\hat q = \dfrac{\lceil (n + 1)(1 - \alpha) \rceil}{n}$ quantile des $s(X_1, Y_1), \ldots, s(X_n, Y_n)$.
> 4. **Build prediction set** :
>
> $$\tau(x) = \{y : s(x, y) \leq \hat q\}.$$

> 💡 **Pourquoi ça marche — intuition mathématique.** Si les data points sont **échangeables** (≈ i.i.d.), un nouveau point a la même chance de tomber dans n'importe quel quantile que les points de calibration. Donc en prenant le quantile $1 - \alpha$, on garantit qu'avec proba $\geq 1 - \alpha$ le score du nouveau point est en-dessous, donc dans le set.

### F. Adaptive Prediction Sets (APS)

> 💡 **Limite du score "vraie classe simple".** Sur MNIST ça marche, mais ne capte pas bien l'incertitude pour les images vraiment ambiguës. APS exploite **toutes** les classes du softmax.

![[im3-5 (1).png]]

> [!note]- APS — algorithme
> **(1) Construction du score.** Pour chaque obs du calibration set, on **trie** le softmax par ordre décroissant. On somme les probabilités jusqu'à atteindre la vraie classe. Le score est cette **masse cumulée**.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im3-3.png]]
>
> **(2) Quantile** $\hat q$ comme avant.
>
> **(3) Prediction set.** Sur la query, on trie le softmax. On rajoute les classes dans l'ordre décroissant **jusqu'à ce que la somme dépasse** $\hat q$.
>
> ![[im3-4.png]]

### G. Conformalized Quantile Regression (CQR)

> 💡 **Cas régression.** Pour la régression, on combine **quantile regression** + conformal prediction. Considéré comme la meilleure approche pour la régression conforme (Romano et al. 2019).

> [!note]- Algorithme
> **(0) Heuristique.** Quantile regression donne $\hat t_{\alpha/2}(X)$ et $\hat t_{1 - \alpha/2}(X)$ (ex : entraîner un NN avec **pinball loss**). Mais ces quantiles ne sont pas exacts (overfitting, bruit).
>
> **(1) Score function.** Encoder l'écart aux quantiles estimés : pour chaque point de calibration,
>
> $$E_i = \max\!\big(\hat t_{\alpha/2}(X_i) - Y_i, \; Y_i - \hat t_{1 - \alpha/2}(X_i)\big).$$
>
> Si $Y_i$ est **dans** la bande, $E_i$ est négatif.
>
> **(2) Quantile** $\hat q$ — peut être négatif si les bandes initiales sont déjà trop larges.
>
> **(3) Prediction interval** sur une nouvelle query :
>
> $$\tau(x) = \big[\hat t_{\alpha/2}(x) - \hat q, \; \hat t_{1 - \alpha/2}(x) + \hat q\big].$$
>
> Garantie de coverage exacte.

![[im3-6.png]]

### H. Marginal vs Conditional Coverage

> [!warning] Marginal coverage (la garantie de base)
> $$\mathbb{P}[Y \in \tau(X)] \geq 1 - \alpha.$$
>
> "Sur tous les points en moyenne, on a 90% de coverage." Mais on peut très bien avoir **toutes les erreurs concentrées** dans un sous-groupe.

> [!warning] Conditional coverage (idéal)
> $$\mathbb{P}[Y \in \tau(X) \mid X = x] \geq 1 - \alpha.$$
>
> Coverage garanti **partout dans l'espace des features**. Conformal **ne le garantit pas** au sens strict, mais des méthodes comme APS et CQR s'en approchent en pratique.

> [!example] Visualisation
> - **No coverage** : 50% de miscoverage dans le groupe 2.
> - **Marginal** (conformal de base) : 100% en haut, 80% en bas → moyenne 90%.
> - **Conditional** : 90% partout — l'idéal.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im4-1 (1).png]]
>
> En régression : sans coverage les bandes sont trop étroites ; conformal donne 90% en moyenne ; conditional adapte la largeur — large dans les régions difficiles, étroite dans les faciles.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/2.Conformal/im4-2.png]]

> 💡 **Comment se rapprocher du conditional coverage.** Les méthodes APS et CQR sont conçues pour ça — leur score function dépend explicitement de $x$, ce qui leur permet d'adapter localement la taille du set.

### I. Évaluer un modèle conformal

Trois axes d'évaluation :

> [!warning] (1) Coverage empirique
> Coverage est une **quantité aléatoire** dépendant du training/calibration split. On peut prouver que sa distribution suit une **Beta** :
>
> $$\text{Coverage} \sim \text{Beta}(l, n - l + 1), \quad l = \lceil (n + 1)(1 - \alpha) \rceil.$$
>
> ![[im4-5.png]]

> [!note]- Protocole d'évaluation
> 1. Random split données → train/calib/val.
> 2. Calculer $\hat q$ sur le calib set.
> 3. Calculer le coverage sur le val set.
> 4. Répéter $T$ fois → histogramme de coverages.
>
> Comparer avec la distribution Beta théorique.
>
> ![[im4-6.png]]

> [!warning] (2) Set size
> Plus c'est petit, mieux c'est. Idéalement la distribution des sizes est **bimodale** : la plupart sont faciles (size 1) et quelques-unes sont dures (size grande).
>
> ![[im4-7.png]]

> [!warning] (3) Adaptiveness / proxy de conditional coverage
> Le conditional coverage exact n'est pas calculable. **Proxy** : **label-stratified coverage**. Si on classifie chats, chiens, éléphants, on veut 90% sur chaque classe séparément. Tu stratifies par classe et tu mesures le coverage. Si les barres sont **égales**, tu es proche du conditional.
>
> ![[im4-8.png]]

### J. Outputs additionnels

En plus de $C(x_q)$, on peut sortir :
- $\hat y_q = \arg\max_k p^k$ — la meilleure prédiction.
- $p_q = \max_k p^k$ — la $p$-value de la meilleure prédiction.
- $1 - \max_{k \neq \hat y_q} p^k$ — la **confiance**. Plus la 2e meilleure $p$-value est petite, plus la confiance est élevée.

---

## III. Quantile Regression

> 💡 **Pourquoi en parler ici.** La quantile regression est le building block de CQR (Conformalized Quantile Regression). Elle a aussi sa propre utilité comme outil d'incertitude.

### A. Motivation

OLS classique cible la **moyenne conditionnelle** :

$$\mathbb{E}(y \mid x) = x \beta_m.$$

On peut viser autre chose :

$$\text{Médiane}(y \mid x) = x \beta_{0.5},$$

ou plus généralement le **$\tau$-percentile conditionnel** :

$$\text{Percentile}_\tau(y \mid x) = x \beta_\tau.$$

### B. Définitions

> [!warning] Quantile
> Le $\tau$-quantile ($\tau \in (0, 1)$) de $y$ est $\mu_\tau$ tel que :
>
> $$\tau = \mathbb{P}(y \leq \mu_\tau) \equiv F_y(\mu_\tau), \quad \mu_\tau = F_y^{-1}(\tau).$$

> [!warning] Quantile conditionnel
> $$\mu_\tau(x) = F_{y \mid x}^{-1}(\tau \mid x).$$

> 💡 **Vocabulaire.** Le 0.23-quantile = 23e percentile. Pareil.

### C. Visualisation

![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im1-1.png]]
**Figure 3.** Dataset d'illustration.

Pour OLS on prend l'espérance conditionnelle. Ici on calcule plutôt les quantiles conditionnels à $\tau \in \{0, 5, 10, \ldots, 100\}$ :

![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im1-2.png]]
**Figure 4.** À droite : quantile conditionnel $\mu_\tau(x)$ pour différents $\tau$.

Pour construire la régression de quantile, on répète pour chaque $x_i$ et on plotte la ligne de régression à $\tau = 0.95$ par exemple :

![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im1-3.png]]

### D. Quantile loss (pinball loss)

On pondère les résidus positifs par $\tau$ et les négatifs par $1 - \tau$ :

$$Q(\beta_\tau) = \sum_i (\tau - \mathbb{I}[y_i < x_i \beta_\tau])(y_i - x_i \beta_\tau) \equiv \rho_\tau(y_i - x_i \beta_\tau).$$

Forme équivalente plus lisible :

$$L = \begin{cases} \tau (y - X\theta) & \text{si } y - X\theta \geq 0 \\ (\tau - 1)(y - X\theta) & \text{si } y - X\theta < 0 \end{cases}$$

> 💡 **Lecture.** On pénalise plus la loss si :
> - $\tau$ est faible mais la prédiction est haute.
> - $\tau$ est élevé mais la prédiction est basse.
>
> Pour $\tau = 0.5$, poids symétriques → médiane.

### E. Cas d'usage industriel — Instacart

> [!example] ETA des livraisons
> Étant donnée une distance, prédire le temps de livraison. Le pipeline donne un ETA, et après livraison on a l'ATA :
>
> $$\text{ATA} = \text{ETA} + \text{prediction error} < \text{Due Time}.$$
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im2-1 (2).png]]
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im2-2.png]]
>
> **Le buffer.**
>
> $$\text{ETA} + \text{Buffer} < \text{Due Time}.$$
>
> Approche naïve : pourcentage de livraisons en retard par ville. Suboptimal.
>
> Approche par quantile regression : entraîner $\tau \in \{0.1, 0.5, 0.9\}$. Donne directement un upper bound. Pour 10 miles → entre 45 et 65 min. L'intervalle s'élargit avec la distance (plus de variance, moins de données).
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im2-3.png]]
>
> **Solution finale Instacart** : quantile regression à $q = 0.9$ pour avoir un upper bound, et on dispatche aux shoppers seulement si :
>
> $$\text{Prédiction du 90e quantile} < \text{Due time}.$$

### F. Quantile Random Forest

> 💡 **L'idée.** Au lieu de moyenner les prédictions des arbres d'une random forest, on garde **la distribution complète** des prédictions par arbre. Ça donne une estimation de la **CDF conditionnelle** :
>
> $$y \mapsto \mathbb{P}(Y \leq y \mid X = x).$$

> [!example] Données superconductivité
> Target : température critique en Kelvin sous laquelle un matériau devient supraconducteur.
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im3-1 (2).png]]
>
> Distribution prédite par matériau :
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im4-1 (1).png]]
>
> **Prediction interval :**
>
> $$I_{\text{pred}}(x) = [Q_\alpha(x), Q_{1-\alpha}(x)].$$
>
> Avec $\alpha = 0.05$, on obtient des intervalles parfois larges (= les arbres sont en désaccord).
>
> ![[images/3-Apprentissage automatique/ML Fundamentals/Calibration/3.Quantile_reg/im4-2.png]]

> [!warning] Applications
> - **Mesurer l'incertitude** — en médecine ou anti-fraude, ne rien faire si l'intervalle est trop large.
> - **Data gathering** — un intervalle énorme indique manque de données ou réponse très bruitée.
> - **Outlier detection** — intervalle énorme = probable outlier dans train ou test.

### G. Lien avec l'incertitude bayésienne

> 💡 **Question fréquente : différence entre prediction interval (quantile reg) et bayesian linear regression ?** Les deux donnent des intervalles, mais :
> - **Prediction interval (quantile reg)** = approche **fréquentiste**, pas d'hypothèse de prior, intervalle marginal.
> - **Bayesian linear regression** = approche bayésienne, prior sur les paramètres, intervalle de **crédibilité** qui dépend du prior.
> - **Conformal prediction** = encore différent, distribution-free, garantie statistique non asymptotique.

---

## IV. Calibration vs Conformal — récapitulatif

| | Calibration | Conformal Prediction |
| :--- | :--- | :--- |
| **Question** | Mon $\hat p$ correspond-il à la vraie probabilité ? | Sur quelle plage la vraie sortie a-t-elle de fortes chances d'être ? |
| **Type de garantie** | Marginale, en moyenne | Marginale (default) ou conditionnelle (avancé), non asymptotique |
| **Sortie** | Score recalibré | Intervalle (régression) ou ensemble de classes |
| **Hypothèse** | Données de calibration | **Échangeabilité** (≈ i.i.d.) |
| **Distribution-free** | Non (Platt suppose une forme sigmoïdale, etc.) | **Oui** |
| **Cas d'usage** | Risk management, marketing CLV | Médical, justice, applications haut-stakes |

> 💡 **Les deux peuvent se combiner.** Modèle calibré → conformal sur les scores calibrés → intervalle plus serré et plus interprétable.

---

## V. Bibliographie

**Conformal Prediction :**
- Vovk, Gammerman, Shafer 2005 — *Algorithmic Learning in a Random World* (livre fondateur)
- Romano et al. 2019 — *Conformalized Quantile Regression*
- Angelopoulos & Bates 2021 — *A Gentle Introduction to Conformal Prediction* (tutorial très accessible)
- [https://arxiv.org/html/2501.19047v2](https://arxiv.org/html/2501.19047v2) — review récente

**Calibration :**
- Niculescu-Mizil & Caruana 2005 — *Predicting Good Probabilities with Supervised Learning* ([PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf))
- Guo et al. 2017 — *On Calibration of Modern Neural Networks* (temperature scaling)
- Naeini & Cooper 2018 — *Binary Classifier Calibration Using an Ensemble of Near Isotonic Regression Models* (ENIR)

**Tutoriel pratique :** [Measuring Models' Uncertainty with Conformal Prediction](https://medium.com/data-from-the-trenches/measuring-models-uncertainty-with-conformal-prediction-f6aa8debb50e)

---

## Annexe — récap des méthodes

| Méthode | Famille | Sortie | Garantie |
| :--- | :--- | :--- | :--- |
| **Reliability diagram** | Calibration | Diagnostic visuel | — |
| **Brier score** | Calibration | Métrique scalaire | — |
| **Platt scaling** | Calibration post hoc | Proba recalibrée | Forme sigmoïde supposée |
| **Isotonic regression** | Calibration post hoc | Proba recalibrée | Monotone, non paramétrique |
| **Temperature scaling** | Calibration NN | Logits divisés | 1 paramètre, softmax |
| **Conformal classique** | Conformal | Set de classes | Coverage marginal |
| **APS** | Conformal | Set adaptatif | Coverage marginal, plus proche du conditional |
| **CQR** | Conformal | Intervalle régression | Coverage marginal, exact |
| **Quantile regression** | Incertitude | Intervalle | Pas de garantie distribution-free |
| **Quantile RF** | Incertitude | Distribution complète | Pas de garantie distribution-free |

> 💡 **À retenir.** Pour de la classification ou régression avec garantie statistique : **conformal prediction**. Pour avoir des probas calibrées (credit scoring, marketing) : **Platt ou isotonic** sur tabulaire, **temperature scaling** sur NN. Pour de l'estimation d'intervalle sans garantie formelle mais utile : **quantile regression** (et son extension Quantile RF).
