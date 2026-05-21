---
title: Apprentissage - PGM
---
# Apprentissage

> Cette note couvre la troisième et dernière des trois parties des PGM : **comment ajuster un modèle aux données**. C'est le pilier qui boucle la boucle Représentation → Inférence → Apprentissage : on a vu comment représenter une distribution avec un graphe, comment lui poser des questions ; reste à savoir d'où viennent les paramètres (et même la structure) du graphe lui-même.

> Pré-requis : [[02_Représentation]], [[03_Inférence]].

> ⚠️ **Note importante.** Cette section est en très grande partie à l'arrache et à compléter. Elle suit le plan des **modules 3 à 6 du cours de Daphne Koller (Stanford / Coursera)** :
> - Module 3 : Parameter Estimation in Bayesian Networks
> - Module 4 : Learning Undirected Models
> - Module 5 : Learning BN Structure
> - Module 6 : Learning BNs with Incomplete Data

---

## I. Vue d'ensemble

> 💡 **Le cadre.** En Représentation, on a posé un graphe avec des CPD ou des facteurs **donnés**. En pratique, on n'a presque jamais ces tables — on a un dataset $\mathcal{D}$ d'observations et on doit **apprendre** à la fois la structure du graphe et les paramètres des facteurs.

Deux problèmes distincts à séparer dès le départ :

> [!warning] Apprentissage des paramètres vs apprentissage de la structure
> - **Parameter learning** : la structure du graphe est connue (un expert l'a dessinée), il reste à estimer les paramètres (CPD ou facteurs) à partir de $\mathcal{D}$.
> - **Structure learning** : on ne connaît pas la structure ; il faut l'inférer **et** estimer les paramètres ensuite.

Et deux régimes selon les données disponibles :

> [!warning] Données complètes vs incomplètes
> - **Données complètes** : toutes les variables sont observées dans toutes les observations du dataset.
> - **Données incomplètes** : certaines variables sont latentes (non observées) ou manquantes. Ce cadre est beaucoup plus difficile et nécessite EM, gradient EM, ou des approches variationnelles.

> 💡 **Remarque épistémologique.** La structure apprise représente la distribution **des données qu'on a vues**, ce qui peut très bien ne **pas** correspondre à la vraie causalité du monde. Pour récupérer la causalité il faut un cadre supplémentaire ([[SCM Pearl]], do-calculus, expérimentation). Apprendre un PGM purement à partir d'observations donne au mieux un I-map de la distribution observationnelle.

---

## II. Apprentissage des paramètres dans les Bayesian Networks (données complètes)

*À développer — Module 3 du cours Koller.*

> 💡 **L'idée.** Avec données complètes et structure fixée, l'apprentissage MLE des paramètres d'un BN se **décompose** par variable : chaque CPD $P(X_i \mid \text{pa}(X_i))$ s'estime indépendamment des autres. C'est ce qui rend le problème tractable.

### A. Maximum Likelihood Estimation (MLE)

*À développer.*

> 💡 **Mini-intuition.** Pour chaque CPD $P(X_i \mid \text{pa}(X_i))$, on compte les occurrences dans $\mathcal{D}$ et on normalise — exactement comme une fréquence empirique. Pour des variables discrètes, ça donne directement les paramètres optimaux.

À aborder :
- Décomposition de la log-vraisemblance par CPD (clé du tractable).
- Cas table CPD (multinomiale).
- Cas gaussien (régression linéaire conditionnelle).
- Surapprentissage avec MLE et petit dataset (CPD avec 0 observations).

### B. Bayesian Learning

*À développer.*

> 💡 **Mini-intuition.** Au lieu d'un point estimate, on met un prior sur les paramètres et on intègre. Pour les CPD multinomiales, le prior conjugué est la **distribution de Dirichlet**, ce qui donne des updates en fermée. C'est l'analogue PGM de ce que tu as fait en [[Inférence Bayésienne|stats bayésiennes]].

À aborder :
- Prior Dirichlet pour les CPD multinomiales.
- Posterior Dirichlet et MAP estimate.
- Hyperparamètres et prior knowledge.
- Rejoint le cadre [[Inférence Bayésienne]] et [[Modèles Bayésiens]].

---

## III. Apprentissage des modèles non orientés (MRF, CRF)

*À développer — Module 4 du cours Koller.*

> 💡 **Le hic.** Contrairement aux BN, l'apprentissage MLE d'un MRF **ne se décompose pas** par facteur — la fonction de partition $Z$ couple tous les paramètres. Le calcul du gradient nécessite de faire de l'inférence à chaque pas, ce qui rend l'apprentissage MRF **beaucoup plus difficile** que celui des BN.

### A. Maximum Likelihood pour MRF

*À développer.*

À aborder :
- Forme du gradient de la log-vraisemblance : moments empiriques − moments du modèle.
- Pourquoi calculer les moments du modèle nécessite l'inférence.
- Contrastive divergence (approximation utilisée pour les Boltzmann Machines).
- Pseudo-likelihood comme alternative tractable.

### B. CRF Learning

*À développer.*

> 💡 **L'avantage des CRF.** Comme on modélise $P(Y \mid X)$ et pas $P(X, Y)$, la fonction de partition $Z(X)$ dépend de l'instance. Pour chaque exemple d'entraînement, on calcule $Z(x_i)$ — l'apprentissage reste lourd mais conceptuellement plus clean que MRF général.

---

## IV. Apprentissage de la structure (Bayesian Networks)

*À développer — Module 5 du cours Koller.*

> 💡 **Trois familles d'approches.**
> - **Score-based** : on définit un score qui mesure la qualité d'un graphe (BIC, BDe), et on cherche le graphe qui maximise ce score.
> - **Constraint-based** : on teste des indépendances conditionnelles dans les données (PC algorithm, FCI), et on construit le graphe compatible. Lien direct avec [[SCM Pearl]] côté causal discovery.
> - **Hybrid** : combine les deux (MMHC).

### A. Score-based learning

*À développer.*

À aborder :
- Likelihood score et son problème (sur-fit le graphe complet).
- BIC (Bayesian Information Criterion) avec pénalité sur la complexité.
- BDe (Bayesian Dirichlet equivalent) — score bayésien marginal.
- Recherche dans l'espace des structures : hill climbing, tabu search, sur l'espace des CPDAG.

### B. Constraint-based learning

*À développer.*

À aborder :
- PC algorithm — rejoint exactement le contenu de [[SCM Pearl]].
- Tests d'indépendance conditionnelle (chi², partial correlation, kernel-based).
- Identification du squelette puis orientation des arêtes.
- Limites : sensibilité aux erreurs de tests (cumul d'erreurs sur $p$ tests).

### C. Le problème de l'I-équivalence

*À développer.*

> 💡 **Lien avec [[02_Représentation#V.C I-équivalence|I-équivalence]].** Plusieurs graphes peuvent encoder les mêmes indépendances. Donc en pure observation, on ne peut **pas** identifier un BN unique — au mieux on récupère sa **classe d'équivalence** (un CPDAG). Pour orienter les arêtes restantes, il faut des connaissances additionnelles (causales, expérimentales, temporelles).

---

## V. Apprentissage avec données incomplètes — EM

*À développer — Module 6 du cours Koller.*

> 💡 **Le problème.** Si certaines variables sont latentes ou manquantes, la log-vraisemblance perd sa décomposition propre par CPD. EM (Expectation-Maximization) est l'algorithme standard pour ces cas.

### A. Expectation-Maximization (EM)

*À développer.*

> 💡 **L'idée en deux phrases.**
> - **E-step** : étant donnés les paramètres courants $\theta^{(t)}$, calculer la distribution sur les latentes $q^{(t+1)}(z) = p(z \mid x, \theta^{(t)})$.
> - **M-step** : étant donnée cette distribution, maximiser la log-vraisemblance espérée pour obtenir $\theta^{(t+1)}$.
>
> EM converge vers un maximum local de la vraisemblance des observations.

À aborder :
- Dérivation via l'ELBO (lien direct avec [[Variational Inference]] — EM est un cas particulier de VI où l'E-step est exact).
- Convergence : monotone mais pas vers le global.
- Cas mixture de gaussiennes comme fil rouge.
- Cas HMM → c'est exactement [[03_Inférence#A.7 Apprentissage des paramètres — Baum-Welch|Baum-Welch]].

### B. Limites de EM

*À développer.*

À aborder :
- Sensibilité à l'initialisation (multiples locaux).
- Lent quand beaucoup de latentes.
- Pas adapté aux gros datasets continus → variationnel.

### C. Variational EM

*À développer.*

> 💡 **Quand EM ne suffit pas.** Si l'E-step exact est intractable (latentes continues complexes, structure trop riche), on remplace par une approximation variationnelle — voir [[Variational Inference#II. Mean field|mean field VI]] et VB-EM.

---

## VI. Bibliographie

- [Notes du cours CS228 Stanford (Ermon Group)](https://ermongroup.github.io/cs228-notes/)
- [Vidéo — Probabilistic ML / PGM Learning](https://www.youtube.com/watch?v=TuGDMj43ehw)
- [Slides Tübingen — Probabilistic ML, partie learning PGM](https://uni-tuebingen.de/de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/methoden-des-maschinellen-lernens/lehre/probabilistic-ml/) *(le lien direct vers le PDF expire — passer par la page principale)*
- Koller & Friedman, *Probabilistic Graphical Models: Principles and Techniques*, MIT Press 2009 — chapitres 17 (BN learning), 19 (incomplete data), 20 (MRF learning), 18 (structure learning).
- Cours Coursera Stanford par Daphne Koller — modules 3, 4, 5, 6 du PGM Specialization (cf. plan dans la conversation initiale).
