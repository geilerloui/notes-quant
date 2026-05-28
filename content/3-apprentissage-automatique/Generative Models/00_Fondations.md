---
title: Fondations - Modèles génératifs
---
# Fondations des modèles génératifs

> Cette première note pose le décor de tout le cours : que cherche-t-on à apprendre, pourquoi est-ce difficile, et quels sont les outils conceptuels (factorisation, indépendance conditionnelle, paramétrisation) qui permettront de construire tous les modèles vus par la suite — autorégressifs, VAE, flows, GANs, diffusion. On suit la première leçon du cours CS236 de Stanford (Ermon) en réorganisant la matière pour mettre en évidence **l'unique fil d'argumentation** qui traverse la leçon.

## I. Pourquoi modéliser $p(x)$ ?

### A. Le point de départ : "What I understand, I can create"

Tout l'objet du cours tient dans une intuition reformulée à partir de Feynman :

> [!quote] Feynman → Génératif
> *"What I cannot create, I do not understand"* — Feynman parlait des démonstrations mathématiques. En modélisation générative, on retourne la phrase : **"What I understand, I can create"**.
> 
> Si un modèle a vraiment compris la structure des données — la grammaire de l'anglais, l'anatomie d'un chien, la dynamique d'un sous-jacent — alors il doit être capable d'en produire de nouveaux exemples plausibles. Générer est un *test de compréhension*.

On adopte une vision **probabiliste** du monde : on suppose que les données observées $\mathcal{D} = \{x_1, \ldots, x_N\}$ sont des échantillons i.i.d. d'une distribution sous-jacente $p_{\text{data}}$ inconnue. L'objectif de tout modèle génératif est alors d'approximer cette distribution à partir du seul dataset $\mathcal{D}$.

### B. Trois usages d'un bon $p(x)$

Une fois qu'on a un modèle $p_\theta(x) \approx p_{\text{data}}(x)$, on peut s'en servir de trois façons :

- **Génération (sampling).** Tirer $x_{\text{new}} \sim p_\theta(x)$ et obtenir un échantillon qui ressemble aux données réelles. C'est l'usage le plus visible : synthèse d'images, de texte, de musique, de paths financiers.
- **Estimation de densité.** Évaluer $p_\theta(x)$ pour un $x$ donné. Cela permet la **détection d'anomalies** : $p_\theta(x)$ est grand si $x$ est typique, petit si $x$ est aberrant.
- **Apprentissage de représentations non supervisées.** Pour bien modéliser des chiens, un modèle doit avoir intériorisé les concepts de patte, museau, fourrure. Ces concepts forment un **espace latent** réutilisable pour des tâches de classification, régression, ou pour conditionner la génération.

Les trois usages ne sont pas toujours alignés : un GAN excelle en génération mais ne donne pas accès à $p_\theta(x)$ ; un flow donne $p_\theta(x)$ exactement mais peut être moins bon en qualité d'échantillon. Cette tension structurera la suite du cours.

## II. Discriminatif vs génératif

Avant d'attaquer la représentation, on clarifie la différence entre **modèle discriminatif** et **modèle génératif**, car c'est le malentendu le plus fréquent.

### A. Conditionnel vs joint

On note $X$ l'entrée (par exemple une image) et $Y$ l'étiquette (par exemple "chambre à coucher").

- Un **modèle discriminatif** apprend la conditionnelle $p(Y \mid X)$. C'est ce que fait une régression logistique ou un CNN de classification : on lui donne une image, il sort une distribution sur les classes.
- Un **modèle génératif** apprend la jointe $p(X, Y)$, ou la marginale $p(X)$ si l'on travaille sans étiquette.

![[images/3-Apprentissage automatique/Generative Models/introduction/im8.png|167]]
Figure graphe bayésien de generative vs discriminative models

![[images/3-Apprentissage automatique/Generative Models/introduction/im9.png|348]]
Autre graphe

> [!warning] Comment lire ces DAG : le sens des flèches
> Une flèche $A \to B$ dans un DAG bayésien encode un **ordre de factorisation** : $A$ vient avant $B$ dans la chain rule. Les deux schémas représentent la **même jointe $p(X, Y)$**, factorisée dans deux ordres différents.
> 
> **Génératif ($Y \to X$).** Ordre : on tire d'abord la classe, ensuite l'image conditionnellement.
> 
> $$p(X, Y) = p(Y) \, p(X \mid Y)$$
> 
> C'est l'ordre naturel de **production** : la classe est le concept, l'image est sa réalisation. Un modèle génératif apprend ce prior $p(Y)$ et ce générateur conditionnel $p(X \mid Y)$.
> 
> **Discriminatif ($X \to Y$).** Ordre inverse : l'image est donnée, on prédit la classe.
> 
> $$p(X, Y) = p(X) \, p(Y \mid X)$$
> 
> L'image est traitée comme une **entrée fixe**. On ignore complètement $p(X)$ et on apprend uniquement $p(Y \mid X)$.
> 
> Les deux factorisations sont **mathématiquement équivalentes** (reliées par Bayes). Mais elles suggèrent des paramétrisations très différentes : modéliser $p(X \mid Y)$ est dur (haute dimension), modéliser $p(Y \mid X)$ est plus simple si on n'a besoin que de prédire $Y$.

> [!note]- Attention : le DAG complet n'est pas le modèle effectif
> Les images du cours dessinent **toute la chain rule** sans hypothèse d'indépendance. Sur le DAG discriminatif, chaque $X_i$ a pour parents tous les $X_{<i}$ — donc les images montrent un produit de conditionnelles entre features.
> 
> En pratique, ces flèches inter-$X_i$ ne sont *pas* modélisées explicitement : la régression logistique apprend directement $p(Y \mid X)$ comme une fonction et ignore complètement la structure conjointe des features. Les flèches du DAG montrent **ce que la chain rule impose si on voulait modéliser la jointe entière** ; le modèle effectif (LR) en garde seulement le dernier facteur.

> [!warning] Pont avec les modèles autorégressifs
> Le DAG de droite (discriminatif) **sans le nœud $Y$** est littéralement le DAG d'un **modèle autorégressif sur $X$** (chap. `[[01_Modèles autoregressifs]]`). Chaque $X_i$ y dépend de tous les $X_{<i}$ : c'est la chain rule complète sur les composantes de $X$.
> 
> La différence sémantique entre les deux cadres est :
> - **Modèle autorégressif** : on apprend **toutes** les conditionnelles $p(X_i \mid X_{<i})$ pour pouvoir échantillonner $X$ (générer du texte, des images, ...).
> - **Modèle discriminatif** : on n'apprend que **la dernière** conditionnelle $p(Y \mid X)$ pour classifier.
> 
> Un LLM comme GPT est un modèle autorégressif sur le texte. Un fine-tuning de classification (sentiment, toxicité, etc.) ajoute un nœud $Y$ à la fin et n'apprend que sa CPD. Le formalisme est rigoureusement le même.

> [!warning] Le génératif est strictement plus riche
> La jointe contient la conditionnelle (via Bayes) **et** la marginale $p(X)$ :
> 
> $$p(Y \mid X) = \frac{p(X, Y)}{p(X)}.$$
> 
> Avec un modèle génératif, on peut : classifier, échantillonner de nouvelles données, gérer les données manquantes en marginalisant les variables non observées. Avec un discriminatif, on ne peut que classifier — et seulement si $X$ est complètement observé.

### B. Le prix à payer

Si le génératif est plus riche, pourquoi ne pas l'utiliser systématiquement ?

- **Difficulté.** Modéliser $p(X)$ en haute dimension est un problème beaucoup plus dur que modéliser $p(Y \mid X)$, parce que $X$ vit dans un espace de dimension énorme alors que $Y$ est de petite dimension.
- **Trade-off biais/variance asymptotique** (Ng & Jordan, 2002). Les modèles génératifs convergent plus vite avec peu de données (moins de variance) mais ont un biais asymptotique plus élevé quand les hypothèses du modèle sont incorrectes. À la limite $N \to \infty$, un discriminatif bien spécifié est meilleur ; à $N$ petit, le génératif l'emporte souvent.
- **Optimisation directe.** Si la seule chose dont on a besoin est de prédire $Y$, modéliser $p(X)$ est du travail gâché.

Le choix dépend donc de la tâche : tâche purement supervisée avec beaucoup de données → discriminatif ; besoin de générer, détecter des anomalies, gérer du manquant, ou peu de données labellisées → génératif.

### C. Le génératif conditionnel

Entre les deux, on peut aussi vouloir $p(X \mid Y)$ : générer une image **conditionnée** sur une classe ou une description. C'est ce que font tous les modèles modernes de type *text-to-image* — formellement un modèle génératif sur $X$, paramétré par $Y$.

## III. Les trois questions fondamentales

Une fois posé qu'on veut apprendre $p(x)$, **trois questions structurent tout le cours**. Ce sont les trois axes orthogonaux le long desquels on évaluera chaque modèle qu'on rencontrera.

> [!warning] Les trois piliers
> 1. **Représentation** : comment paramétrer la famille de distributions $\{p_\theta\}_\theta$ ? Quelle forme donner à $p_\theta(x)$ pour qu'elle soit à la fois expressive et calculable ?
> 2. **Apprentissage (learning)** : étant donné $\mathcal{D}$, comment trouver le $\theta^*$ qui rend $p_{\theta^*}$ aussi proche que possible de $p_{\text{data}}$ ? Quelle notion de proximité (KL, Wasserstein, score matching, adversarial) ?
> 3. **Inférence** : une fois $\theta^*$ trouvé, comment **utiliser** le modèle ? Échantillonner, évaluer la densité, marginaliser, conditionner, calculer des espérances.


![[images/3-Apprentissage automatique/Generative Models/autorégressif/im1 (1) 1.png]]
**Figure 1.** Les trois questions. Représentation = choix de la famille de modèles (le set vert) ; learning = trouver le point le plus proche du target (le point rouge) selon une métrique ; inférence = utiliser le modèle appris.

Chaque famille de modèles (autorégressifs, VAE, flows, GANs, diffusion) fait des choix différents sur ces trois axes, et c'est ce qui les différencie. Par exemple :

- les **flows** privilégient une représentation où $p_\theta(x)$ est calculable exactement → learning par maximum de vraisemblance direct → inférence facile pour l'évaluation, légèrement plus coûteuse pour l'échantillonnage.
- les **GANs** privilégient une représentation où l'échantillonnage est facile mais $p_\theta(x)$ n'est pas accessible → learning adversarial (pas de vraisemblance) → inférence : on peut échantillonner mais pas évaluer la densité.

Cette grille de lecture sera reprise systématiquement dans chaque note. Pour le reste de la présente fondation, on se concentre sur l'axe **représentation**, qui est le point d'entrée naturel.

## IV. Représentation — une première approche : la voie autorégressive

> [!warning] Avertissement de cadrage
> Il existe **plusieurs stratégies** pour paramétrer $p(x)$ en haute dimension. Cette section développe la première — la plus directe : **décomposer la jointe par chain rule et paramétrer chaque conditionnelle**. C'est la voie qui mène aux **modèles autorégressifs** (chap. `[[01_Modèles autoregressifs]]`) et que les LLM exploitent à grande échelle.
> 
> Les autres voies — introduire une variable latente $z$ (VAE, diffusion), imposer une bijection (flows), apprendre par jeu adversariel (GAN) — sont des **stratégies alternatives qui n'utilisent *pas* la chain rule sur les composantes de $x$**. Elles sont introduites en §VI et développées dans les chapitres suivants.

L'enjeu commun à toutes les approches est simple à énoncer : $x$ vit en haute dimension. Une image $32 \times 32$ en noir et blanc, c'est déjà $2^{1024}$ valeurs possibles. Aucun ordinateur ne peut stocker une table avec autant d'entrées. Il faut donc trouver une **forme compacte** pour $p(x)$.

![[images/3-Apprentissage automatique/Generative Models/vae/im3 (3).png|429]]
Exemple on prend les trois chiffres 9, 3 et 6 en noir et blanc

### A. Le mur combinatoire

> [!example] Combien de paramètres pour une distribution jointe ?
> Soit $x = (x_1, \ldots, x_n)$ un vecteur de $n$ pixels binaires (Bernoulli).
> 
> - **Nombre d'états possibles :** $2^n$.
> - **Paramètres pour spécifier la jointe $p(x_1, \ldots, x_n)$ :** $2^n - 1$ (le $-1$ vient de la contrainte que les probabilités somment à 1).
> 
> Pour $n = 1024$ (une image $32 \times 32$ binaire), on a $2^{1024} - 1 \approx 10^{308}$ paramètres — plus que le nombre d'atomes dans l'univers observable. **Impossible.**

Ce constat brutal est le point de départ de toute la modélisation générative : **on ne peut pas représenter une distribution jointe arbitraire**. Il faut faire des **hypothèses simplificatrices** sur la structure de $p(x)$.

### B. Première tentative : indépendance totale

L'hypothèse la plus brutale qu'on puisse faire est de supposer toutes les variables **indépendantes** :

$$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i).$$

- **Nombre de paramètres :** $n$ (un Bernoulli par pixel). Pour $n = 1024$, c'est 1024 — tractable.
- **Problème :** échantillonner depuis ce modèle revient à tirer chaque pixel **indépendamment** des autres. On obtient du **bruit**, aucune structure spatiale, aucune corrélation. Inutile.

![[images/3-Apprentissage automatique/Generative Models/vae/im4.png]]
**Figure 2.** Échantillon d'un modèle à indépendance totale. Chaque pixel est tiré indépendamment d'une Bernoulli — résultat : du bruit pur, aucune cohérence.

L'hypothèse est trop forte. Il faut un compromis entre la jointe complète (intractable) et l'indépendance totale (inutile).

### C. La bonne notion : indépendance conditionnelle

> [!warning] Indépendance conditionnelle
> Deux variables aléatoires $X$ et $Y$ sont **conditionnellement indépendantes** sachant $Z$, noté $X \perp Y \mid Z$, si
> 
> $$p(X, Y \mid Z) = p(X \mid Z) \, p(Y \mid Z).$$
> 
> De façon équivalente : $p(X \mid Y, Z) = p(X \mid Z)$. Autrement dit, **connaître $Z$ rend $Y$ inutile pour prédire $X$**. Toute l'information que $Y$ contenait sur $X$ passait déjà par $Z$.

L'indépendance conditionnelle est beaucoup plus faible que l'indépendance marginale, et donc beaucoup plus réaliste. Deux pixels distants d'une image ne sont pas indépendants, mais ils peuvent le devenir **une fois qu'on conditionne** sur les pixels intermédiaires.

### D. La règle de la chaîne (chain rule)

L'identité algébrique fondamentale qui rend l'indépendance conditionnelle utile est la **règle de la chaîne** :

$$p(x_1, \ldots, x_n) = p(x_1) \, p(x_2 \mid x_1) \, p(x_3 \mid x_1, x_2) \cdots p(x_n \mid x_1, \ldots, x_{n-1}) = \prod_{i=1}^{n} p(x_i \mid x_{<i}).$$

![[auto1.png|283]]
Figure. Visuellement on a ça


> [!note]- La chain rule ne réduit rien à elle seule
> En toute généralité, paramétrer chaque $p(x_i \mid x_{<i})$ par une table demande encore $2^n - 1$ paramètres au total (la décomposition est exacte, on n'a rien gagné). **Ce qui fait gagner, c'est la combinaison de la chain rule avec des hypothèses d'indépendance conditionnelle** qui suppriment des conditionnements.

> [!example] Modèle de Markov
> Si on suppose $X_{i+1} \perp X_{<i} \mid X_i$ (chaque variable ne dépend que de la précédente) :
> 
> $$p(x_1, \ldots, x_n) = p(x_1) \prod_{i=2}^{n} p(x_i \mid x_{i-1}).$$
> 
> Le nombre de paramètres tombe à $2n - 1$. Pour $n = 1024$, c'est 2047 — gigantesque saut depuis $2^{1024}$.

Le point clé : **la chain rule fournit le squelette de factorisation, l'indépendance conditionnelle élague les conditionnements**, et c'est cette combinaison qui permet de représenter des distributions de haute dimension avec peu de paramètres.

### E. Bayesian Networks : la généralisation

Le modèle de Markov est un cas particulier d'un cadre plus général : les **réseaux bayésiens** (Bayesian Networks, BN). On garde l'idée — chain rule + hypothèses d'indépendance conditionnelle — mais on autorise des structures de dépendance plus riches.

> [!warning] Bayesian Network
> Un réseau bayésien est défini par un **graphe orienté acyclique** (DAG) $G = (V, E)$ où chaque nœud $i \in V$ correspond à une variable $X_i$, et où la distribution jointe se factorise comme
> 
> $$p(x_1, \ldots, x_n) = \prod_{i=1}^{n} p(x_i \mid x_{\text{Pa}(i)}),$$
> 
> avec $\text{Pa}(i)$ l'ensemble des parents de $i$ dans $G$. Le DAG encode les hypothèses d'indépendance conditionnelle, et chaque facteur $p(x_i \mid x_{\text{Pa}(i)})$ est une **CPD** (conditional probability distribution).

![[images/3-Apprentissage automatique/Generative Models/introduction/im6 (1).png]]
**Figure 3.** Exemple de réseau bayésien. La factorisation jointe se lit directement sur le graphe : un facteur par nœud, conditionné sur ses parents.

**Comptage des paramètres pour un BN.** Si chaque variable est binaire et que le nœud $i$ a $k_i = |\text{Pa}(i)|$ parents, alors la CPD $p(x_i \mid x_{\text{Pa}(i)})$ demande $2^{k_i}$ paramètres (un Bernoulli par configuration des parents). Le total est

$$\sum_{i=1}^{n} 2^{k_i},$$

à comparer avec $2^n - 1$ pour la jointe brute. Le gain est exponentiel **si les degrés entrants restent petits** — c'est exactement la condition pour qu'un BN soit utile.

> [!example] Naive Bayes (exemple minimal)
> Le réseau bayésien le plus simple qui soit utile en pratique : on a une classe $Y$ et des features $X_1, \ldots, X_n$, et on suppose que **les features sont conditionnellement indépendantes sachant $Y$** :
> 
> $$p(y, x_1, \ldots, x_n) = p(y) \prod_{i=1}^{n} p(x_i \mid y).$$
> 
> Le DAG associé : $Y$ pointe vers chaque $X_i$, aucune autre arête. Modèle absurdement simple, mais étonnamment efficace en pratique (classification de spam historique). Sert ici d'exemple canonique de modèle génératif minimal.

### F. Le problème résiduel : les CPDs restent trop grosses

Le BN attaque la **structure** des dépendances, mais chaque CPD $p(x_i \mid x_{\text{Pa}(i)})$ reste paramétrée par une **table** — une entrée pour chaque configuration des parents. Si un nœud a beaucoup de parents, sa table explose à nouveau.

Dans un modèle génératif autorégressif sur des images, par exemple, le pixel $x_i$ peut dépendre de **tous** les pixels précédents. La CPD $p(x_i \mid x_{<i})$ a alors $2^{i-1}$ entrées. On retombe sur le mur combinatoire — au niveau du nœud cette fois.

> [!warning] La vraie solution : paramétrer les CPDs par des fonctions
> Plutôt que de stocker une table pour chaque CPD, on **paramètre chaque CPD par une fonction** $f_\theta(x_{\text{Pa}(i)})$ qui prend en entrée les valeurs des parents et sort la distribution de $x_i$. La taille de $\theta$ est fixée à l'avance et **ne dépend plus du nombre de parents**.

Cette idée — passer du tabulaire au paramétré — est le pivot qui sépare la modélisation graphique classique de la modélisation générative moderne.

### G. Du tabulaire au paramétré : LR puis NN

On illustre l'idée sur la conditionnelle discriminative $p(Y \mid X_1, \ldots, X_n)$ (le raisonnement est identique pour une CPD générative).

**Régression logistique.** La forme paramétrée la plus simple :

$$p(Y = 1 \mid x; \alpha) = \sigma\!\left(\alpha_0 + \sum_{i=1}^{n} \alpha_i x_i\right),$$

avec $\sigma$ la sigmoïde. Au lieu de $2^n$ entrées tabulaires, on a $n + 1$ paramètres. Le prix : la frontière de décision est **linéaire** dans $x$.

**Réseaux de neurones.** On compose des transformations non linéaires :

$$p(Y = 1 \mid x; \theta) = \sigma\!\left(\alpha_0 + \sum_{j=1}^{h} \alpha_j \, f(A_j x + b_j)\right),$$

avec $f$ une non-linéarité (ReLU, tanh, etc.). On retrouve la flexibilité d'une CPD tabulaire — et bien plus, par universalité d'approximation — tout en gardant un nombre fixe de paramètres.

![[]]
**Figure 4.** Du tabulaire au paramétré. À gauche : une CPD comme table (paramètres = entrées de la table). À droite : la même CPD paramétrée par un réseau de neurones (paramètres = poids du réseau, indépendants du nombre de parents).

## V. Synthèse : la recette autorégressive

> [!warning] La recette de la voie autorégressive : structure (BN) + forme (NN)
> Toute la §IV peut se résumer en une équation conceptuelle qui caractérise les **modèles autorégressifs** :
> 
> $$\boxed{\;p_\theta(x) \;=\; \underbrace{\prod_{i=1}^{n} p_\theta(x_i \mid x_{\text{Pa}(i)})}_{\text{structure : chain rule + DAG}} \quad \text{avec chaque CPD} \quad \underbrace{p_\theta(x_i \mid x_{\text{Pa}(i)}) = f_\theta(x_{\text{Pa}(i)})}_{\text{forme : NN}}\;}$$
> 
> - La **structure** (le DAG et la chain rule sur les composantes de $x$) est l'héritage des réseaux bayésiens classiques.
> - La **forme** (NN à la place de tables) est l'apport moderne — c'est ce qui rend les modèles génératifs *deep*.

Cette factorisation est le squelette des **modèles autorégressifs** : NADE, MADE, PixelCNN, WaveNet, et tous les LLM modernes (GPT, Claude, Llama). Les variantes diffèrent par :

- la **structure** du DAG : ordre total gauche-droite (texte), raster scan (images), temporel (audio) ;
- la **forme** des CPDs : MLP partagé (NADE), MLP masqué (MADE), CNN causal (PixelCNN, WaveNet), Transformer causal (GPT) ;
- la façon d'**apprendre** $\theta$ : MLE directe sur la log-vraisemblance, calculable exactement.

> [!note] Cette recette n'est pas universelle
> La factorisation par chain rule sur $x$ caractérise **uniquement** les modèles autorégressifs. Les autres familles génératives — VAE, flows, GAN, diffusion — utilisent des stratégies de représentation *fondamentalement différentes* : variables latentes, bijections, jeux adversariels, débruitage progressif. Voir §VI pour le panorama.

## V bis. Apprentissage par maximum de vraisemblance

La §V répond à la question *représentation* pour la voie autorégressive. Avant de passer aux familles concrètes, on traite la deuxième question des trois piliers — **l'apprentissage** — au niveau de généralité qui convient à toutes les familles dont la densité $p_\theta(x)$ est calculable (autorégressifs, flows) ou bornable (VAE via ELBO). Les GANs et les modèles de diffusion useront de principes différents (adversarial, score matching) qu'on verra le moment venu.

### A. Le problème de la divergence à $p_{\text{data}}$

On a un dataset $\mathcal{D} = \{x^{(1)}, \ldots, x^{(m)}\}$ tiré i.i.d. d'une distribution inconnue $p_{\text{data}}$. On veut choisir $\theta$ pour que $p_\theta$ soit aussi proche que possible de $p_{\text{data}}$. Mais *proche* selon quelle mesure ? La réponse standard est la **divergence de Kullback-Leibler** :

$$D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}(x)}{p_\theta(x)}\right] = \sum_{x} p_{\text{data}}(x) \log \frac{p_{\text{data}}(x)}{p_\theta(x)}.$$

Deux propriétés :

- $D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) \geq 0$ avec égalité ssi $p_\theta = p_{\text{data}}$ presque partout.
- Interprétation en théorie de l'information : c'est la **perte de compression** (en bits ou nats) si l'on code des échantillons de $p_{\text{data}}$ avec un code optimisé pour $p_\theta$.

> [!note]- Asymétrie de la KL
> $D_{\text{KL}}(p \,\|\, q) \neq D_{\text{KL}}(q \,\|\, p)$. Le choix d'ordre n'est pas anodin : ici on prend $p_{\text{data}}$ en premier, ce qui revient à pénaliser fortement les $x$ où $p_{\text{data}}(x)$ est grand mais $p_\theta(x) \approx 0$. Le modèle est forcé d'avoir un *support qui couvre les données* (mode-covering). L'ordre inverse $D_{\text{KL}}(p_\theta \,\|\, p_{\text{data}})$ pousse au mode-seeking et apparaîtra dans la dérivation de l'ELBO pour les VAE.

### B. De la KL à la log-vraisemblance

On développe la KL :

$$D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) = \underbrace{\mathbb{E}_{x \sim p_{\text{data}}}[\log p_{\text{data}}(x)]}_{-H(p_{\text{data}}),\ \text{indépendant de }\theta} - \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)].$$

Le premier terme est l'entropie négative de $p_{\text{data}}$ — il ne dépend pas de $\theta$, donc disparaît à l'optimisation. Il reste :

$$\arg\min_\theta D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) = \arg\max_\theta \; \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)].$$

> [!warning] Minimiser la KL = maximiser la log-vraisemblance espérée
> Cette équivalence est le pont fondamental entre la théorie (rapprochement de distributions) et la pratique (estimation par maximum de vraisemblance). **Toutes les méthodes likelihood-based en découlent.**
>
> Conséquence importante : on ne peut pas connaître la valeur absolue de la KL atteinte par notre modèle (le terme $H(p_{\text{data}})$ est inconnu), mais on peut **comparer** deux modèles par leur log-vraisemblance.

### C. L'estimation empirique : MLE

On ne connaît pas $p_{\text{data}}$, mais on a $m$ échantillons. On remplace l'espérance par sa moyenne empirique (estimateur Monte Carlo non biaisé) :

$$\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)] \;\approx\; \frac{1}{m} \sum_{j=1}^{m} \log p_\theta(x^{(j)}).$$

Le **principe du maximum de vraisemblance** (MLE) consiste à résoudre :

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \; \frac{1}{m} \sum_{j=1}^{m} \log p_\theta(x^{(j)}) \;=\; \arg\max_\theta \; \log \prod_{j=1}^{m} p_\theta(x^{(j)}).$$

C'est exactement maximiser la probabilité que le modèle attribue collectivement aux données observées, en supposant l'indépendance des observations.

### D. Optimisation : SGD via Monte Carlo

Pour les modèles tabulaires des BN classiques, il existe parfois des solutions analytiques (comptage de fréquences). Dès que les CPDs sont paramétrées par des réseaux de neurones, ce n'est plus possible — il faut optimiser numériquement par **descente de gradient** :

$$\theta^{t+1} = \theta^t + \alpha_t \, \nabla_\theta \ell(\theta), \qquad \ell(\theta) = \sum_{j=1}^{m} \log p_\theta(x^{(j)}).$$

Le gradient se calcule par rétropropagation à travers le réseau qui paramètre les CPDs.

**Le problème de l'échelle.** Quand $m$ est grand (millions à milliards d'observations), évaluer le gradient sur tout le dataset à chaque pas est prohibitif. On utilise une **deuxième fois** Monte Carlo, cette fois pour estimer le gradient sur un mini-lot :

$$\nabla_\theta \ell(\theta) = m \cdot \mathbb{E}_{x \sim \mathcal{D}}[\nabla_\theta \log p_\theta(x)] \;\approx\; \frac{m}{B} \sum_{j \in \text{batch}} \nabla_\theta \log p_\theta(x^{(j)}).$$

C'est la **descente de gradient stochastique** (SGD) — l'algorithme standard de l'apprentissage profond. Le facteur $m$ est en pratique absorbé dans le pas d'apprentissage $\alpha$ et on travaille sur la log-vraisemblance moyennée.

> [!note]- Deux usages distincts de Monte Carlo
> On utilise MC deux fois et il est important de les distinguer :
> 1. Pour **estimer l'espérance sous $p_{\text{data}}$** : c'est ce qui transforme la KL théorique en une somme sur les données — c'est ce qu'on fait en construisant le dataset.
> 2. Pour **estimer le gradient sur un sous-ensemble du dataset** : c'est le passage GD → SGD.
>
> Le premier est conceptuel (justifier la MLE) ; le second est computationnel (rendre l'entraînement faisable).

### E. Le cas spécifique des modèles à factorisation

Un modèle factorisable (chain rule, BN, autorégressif...) écrit la jointe comme un produit de conditionnelles. La log-vraisemblance hérite de cette structure :

$$\log p_\theta(x) = \sum_{i=1}^{n} \log p_\theta(x_i \mid x_{\text{Pa}(i)}).$$

Ce **découplage par variable** a deux conséquences pratiques majeures :

- L'objectif d'apprentissage est une **somme de pertes locales** — une par CPD. Chaque CPD se voit attribuer ses propres erreurs, ce qui rend la rétropropagation directe.
- Avec une architecture qui calcule toutes les CPDs **en parallèle** (MADE, PixelCNN, Transformer causal), la log-vraisemblance complète d'un échantillon se calcule en **un seul forward pass**. C'est ce qui rend les modèles autorégressifs modernes entraînables à l'échelle des LLMs.

On reverra cet aspect dans la note `[[01_Modèles autoregressifs]]`.

### F. Conditionnel vs joint

Quand on veut apprendre $p_\theta(y \mid x)$ plutôt que $p_\theta(x)$ (modèles génératifs conditionnels : text-to-speech, image captioning, image-to-image), tout le formalisme se transpose sans modification — on remplace simplement $\log p_\theta(x)$ par $\log p_\theta(y \mid x)$ et l'espérance est prise sur les paires $(x, y) \sim p_{\text{data}}$. C'est aussi le cadre de l'apprentissage supervisé standard, vu sous l'angle probabiliste : la cross-entropy minimisée par un classifieur est exactement la log-vraisemblance négative du modèle $p_\theta(y \mid x)$.

## VI. Aperçu de la suite : les autres voies de représentation

La voie autorégressive (§IV-V) est *une* stratégie parmi plusieurs. Voici un panorama des **quatre grandes stratégies de représentation** qu'on rencontrera dans le dossier, organisées par l'idée centrale de chacune.

### A. Chain rule sur les composantes de $x$ → autorégressif

C'est la voie qu'on vient de développer.

$$p_\theta(x) = \prod_{i=1}^{n} p_\theta(x_i \mid x_{<i})$$

- **Stratégie** : décomposer la jointe sur les composantes, paramétrer chaque conditionnelle par un réseau.
- **Familles** : NADE, MADE, PixelCNN, WaveNet, LLM (GPT, Claude, ...).
- **Densité $p(x)$** : exacte.
- **Apprentissage** : MLE directe.
- **Échantillonnage** : séquentiel — il faut tirer $x_i$ avant $x_{i+1}$, donc lent.

### B. Introduire une variable latente $z$ → VAE

$$p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz$$

- **Stratégie** : faire passer la complexité de $x$ par un **latent** $z$ de petite dimension. Un prior simple $p(z) = \mathcal{N}(0, I)$ est mappé à $x$ par un décodeur.
- **Pas de chain rule sur $x$** : la jointe $p(x, z)$ se factorise en $p(z) p(x \mid z)$, mais $p(x)$ lui-même demande de marginaliser sur $z$.
- **Densité $p(x)$** : intractable (intégrale). On optimise une **borne inférieure** (ELBO).
- **Apprentissage** : ELBO + reparametrization trick.
- **Échantillonnage** : tirer $z$, passer au décodeur — rapide, un seul forward pass.

### C. Bijection $z \leftrightarrow x$ → normalizing flows

$$p_\theta(x) = p_Z\!\big(f_\theta^{-1}(x)\big) \, \left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|$$

- **Stratégie** : imposer que la fonction décodeur $f_\theta$ soit **bijective** (donc inversible), avec un déterminant jacobien calculable. La formule de changement de variable donne alors $p_\theta(x)$ **exactement**.
- **Densité $p(x)$** : exacte (comme l'autorégressif), mais via un mécanisme entièrement différent.
- **Apprentissage** : MLE directe sur la formule de changement de variable.
- **Échantillonnage** : rapide, un seul forward pass.

### D. Apprendre par jeu adversariel → GAN

$$x = G_\theta(z), \quad z \sim p(z) \quad \text{(pas de densité explicite)}$$

- **Stratégie** : abandonner complètement la modélisation de $p(x)$. On apprend juste à **produire des échantillons** indistinguables des données réelles, via un jeu min-max entre un générateur et un discriminateur.
- **Densité $p(x)$** : pas accessible.
- **Apprentissage** : adversariel (JSD pour le vanilla GAN, Wasserstein pour le WGAN).
- **Échantillonnage** : rapide.

### E. Chaîne de Markov de débruitage → diffusion

$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}, \quad p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$$

- **Stratégie** : voir $x$ comme le résultat d'un long processus de débruitage. On apprend à inverser une chaîne de Markov qui ajoute progressivement du bruit gaussien aux données.
- **Pas de chain rule sur les composantes de $x$** : la chain rule est sur les *étapes de débruitage* $x_T \to x_{T-1} \to \cdots \to x_0$.
- **Densité $p(x_0)$** : approchée (ELBO hiérarchique, ou score matching).
- **Apprentissage** : score matching ou ELBO.
- **Échantillonnage** : itératif sur $T$ étapes — plus lent que VAE/GAN/flow mais qualité supérieure.

### Tableau de synthèse

| Famille | Stratégie principale | Densité $p(x)$ ? | Apprentissage | Échantillonnage |
|---|---|---|---|---|
| Autorégressif | Chain rule sur $x$ | Exacte | MLE direct | Séquentiel (lent) |
| VAE | Latent $z$ + décodeur | Borne (ELBO) | ELBO | Rapide |
| Normalizing flow | Bijection $z \leftrightarrow x$ | Exacte | MLE direct | Rapide |
| GAN | Jeu adversariel | Inaccessible | Adversariel | Rapide |
| Diffusion | Markov de débruitage | Approchée | Score matching | Itératif (lent) |

Chacune de ces familles aura sa propre note dans le dossier `Generative Models`. La présente fondation est le langage commun.

---

## Pour aller plus loin

- **Cours.** Stefano Ermon, *CS236 Deep Generative Models*, Stanford. La leçon 1 correspond à la présente note.
- **Référence classique sur les BN.** Koller & Friedman, *Probabilistic Graphical Models* (2009). Couvre représentation, inférence et apprentissage des modèles graphiques avant l'ère deep.
- **Trade-off discriminatif/génératif.** Ng & Jordan, *On Discriminative vs. Generative Classifiers* (NeurIPS 2002). Le résultat asymptotique évoqué en §II.B.
