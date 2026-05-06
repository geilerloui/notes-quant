---
title: Réseaux de Neurones Récurrents
description: Introduction aux RNN pour le traitement séquentiel
weight: 4
outline_depth: "1"
---

# Réseaux de Neurones Récurrents (RNN) - 1980s

## Pourquoi pas les réseaux Feed-Forward ?

Les réseaux de neurones classiques (FFNN) ne conviennent pas au traitement de séquences pour plusieurs raisons :

- **Longueurs variables** : Les inputs et outputs peuvent avoir des longueurs différentes selon les exemples
- **Pas de partage de features** : Un FFNN ne peut pas réutiliser les caractéristiques apprises à différentes positions dans le texte. Par exemple, si le réseau détecte le nom "Harry" en première position puis plus tard dans la séquence, il ne peut pas exploiter le fait qu'il s'agit du même type d'entité

## Architecture RNN : Exemple d'Analyse de Sentiment

Considérons un exemple simple d'analyse de sentiment sur 3 mots : **"good bad movie"** → sentiment **positif/négatif**.

### Données d'exemple

Notre séquence d'entrée :
$$X = \begin{bmatrix} 
\text{"good"} \\ \text{"bad"} \\ \text{"movie"}
\end{bmatrix}$$

Après embedding (transformation en vecteurs), supposons des représentations 2D :
$$X = \begin{bmatrix} 
\textcolor{teal}{\mathbf{0.8}} & \textcolor{teal}{\mathbf{0.2}} \\
\textcolor{orange}{\mathbf{-0.6}} & \textcolor{orange}{\mathbf{0.9}} \\
\textcolor{violet}{\mathbf{0.1}} & \textcolor{violet}{\mathbf{0.3}}
\end{bmatrix}$$

Architecture RNN avec **état caché de dimension 3** :

![[Pasted image 20260419192535.png|388]]

2e représentation
JAI PERDU LIMAGE

![[Pasted image 20260419194323.png|307]]

3e représentation
![[images/1-Mathématiques/Probabilité/Échantillonage/im2.png|95]]

# I - RNN : Fondements des Réseaux Récurrents

## a - Forward Pass Séquentiel

### Initialisation des paramètres

$[W_{xa}]_{2 \times 3}$ : matrice de transformation input → état caché  
$[W_{aa}]_{3 \times 3}$ : matrice de transformation état précédent → état actuel  
$[W_{ay}]_{3 \times 1}$ : matrice de transformation état final → sortie  
$[b_a]_{1 \times 3}$ : biais pour l'état caché  
$[b_y]_{1 \times 1}$ : biais pour la sortie  
$a_0 = \mathbf{0}$ : état caché initial

### Traitement séquentiel

**Temps t=1 : Traitement de "good"**

$$a_1 = \tanh(x_1 W_{xa} + a_0 W_{aa} + b_a)$$

$$a_1 = \tanh\left(\begin{bmatrix} \textcolor{teal}{\mathbf{0.8}} & \textcolor{teal}{\mathbf{0.2}} \end{bmatrix} W_{xa} + \begin{bmatrix} 0 & 0 & 0 \end{bmatrix} W_{aa} + b_a\right)$$

**Temps t=2 : Traitement de "bad" (avec mémoire de "good")**

$$a_2 = \tanh(x_2 W_{xa} + a_1 W_{aa} + b_a)$$

$$a_2 = \tanh\left(\begin{bmatrix} \textcolor{orange}{\mathbf{-0.6}} & \textcolor{orange}{\mathbf{0.9}} \end{bmatrix} W_{xa} + a_1 W_{aa} + b_a\right)$$

**Temps t=3 : Traitement de "movie" (avec mémoire de "good bad")**

$$a_3 = \tanh(x_3 W_{xa} + a_2 W_{aa} + b_a)$$

$$a_3 = \tanh\left(\begin{bmatrix} \textcolor{violet}{\mathbf{0.1}} & \textcolor{violet}{\mathbf{0.3}} \end{bmatrix} W_{xa} + a_2 W_{aa} + b_a\right)$$

> [!note]- 💡 Partage de Paramètres
> **Clé du RNN** : Les matrices $W_{xa}$, $W_{aa}$, et $W_{ay}$ sont **partagées** à chaque pas de temps.
> 
> **Avantage** : Le réseau apprend des patterns qui peuvent s'appliquer à n'importe quelle position dans la séquence.
> 
> **Mémoire** : L'état caché $a_t$ encode l'information de tous les mots précédents.

### Representation Learning séquentiel

Cette étape révèle **la vraie magie des RNN** : la transformation progressive des mots en représentations de plus en plus riches.

**Évolution des représentations dans l'espace latent 3D :**

Supposons avec des paramètres concrets que nous obtenions :
$$a_1 = \begin{bmatrix} \textcolor{teal}{\mathbf{0.49}} & \textcolor{teal}{\mathbf{-0.17}} & \textcolor{teal}{\mathbf{0.64}} \end{bmatrix}$$

$$a_2 = \begin{bmatrix} \textcolor{orange}{\mathbf{-0.005}} & \textcolor{orange}{\mathbf{0.64}} & \textcolor{orange}{\mathbf{-0.38}} \end{bmatrix}$$

$$a_3 = \begin{bmatrix} \textcolor{violet}{\mathbf{0.463}} & \textcolor{violet}{\mathbf{0.329}} & \textcolor{violet}{\mathbf{0.124}} \end{bmatrix}$$

> [!info]- 🧠 La Révélation du Representation Learning
> **Transformation fondamentale** : Chaque $a_t$ n'est pas juste une transformation de $x_t$, mais une **représentation cumulative** :
> 
> - $a_1$ : représentation de "good" seul (2D → 3D)
> - $a_2$ : représentation de "good bad" combinés (historique + nouveau)
> - $a_3$ : représentation de "good bad movie" complète (contexte total)
> 
> **Pourquoi c'est révolutionnaire** : Contrairement aux FFNN où chaque exemple est traité indépendamment, ici chaque $a_t$ **construit** sur le précédent. C'est du representation learning **séquentiel et cumulatif**.
> 
> **L'insight clé** : L'espace latent 3D n'encode plus des mots individuels, mais des **histoires partielles**. $a_3$ contient une représentation dense de toute la séquence "good bad movie".

**Analogie avec les perceptrons multicouches :**
- **MLP** : $x_{2D} \rightarrow a_{3D}$ (transformation statique)
- **RNN** : $x_{2D}^{(t)} + a_{3D}^{(t-1)} \rightarrow a_{3D}^{(t)}$ (accumulation dynamique)

Chaque état caché devient une **empreinte** de plus en plus riche du contexte séquentiel !

### Prédiction finale (Many-to-One)

Seul l'état final $a_3$ est utilisé pour la prédiction :

$$\hat{y} = \sigma(a_3 W_{ay} + b_y)$$

où $\sigma$ est la fonction sigmoïde pour la classification binaire (positif/négatif).

$$\hat{y} = \sigma\left(\begin{bmatrix} a_{3,1} & a_{3,2} & a_{3,3} \end{bmatrix} W_{ay} + b_y\right)$$

**Interprétation** : $\hat{y} \in [0,1]$ représente la probabilité que le sentiment soit positif. Pour notre exemple "good bad movie", le réseau doit apprendre que malgré "bad", le contexte global tend vers un sentiment positif.

## b - Backpropagation Through Time (BPTT)

### Dépendances temporelles

Contrairement aux réseaux classiques, la backpropagation dans les RNN doit tenir compte de la **dépendance temporelle**. L'erreur à l'instant final se propage vers tous les pas de temps précédents.

$$\frac{\partial J}{\partial W_{xa}} = \sum_{t=1}^{3} \frac{\partial J}{\partial a_t} \frac{\partial a_t}{\partial W_{xa}}$$

$$\frac{\partial J}{\partial W_{aa}} = \sum_{t=1}^{3} \frac{\partial J}{\partial a_t} \frac{\partial a_t}{\partial W_{aa}}$$

### Problème du Vanishing Gradient

Le gradient peut **disparaître exponentiellement** lors de la rétropropagation à travers le temps :

$$\frac{\partial a_t}{\partial a_k} = \prod_{i=k+1}^{t} \frac{\partial a_i}{\partial a_{i-1}} = \prod_{i=k+1}^{t} W_{aa} \text{diag}(\tanh'(z_i))$$

Si $\|W_{aa}\| < 1$ et $\tanh'(z) \leq 1$, ce produit tends vers 0 quand $t-k$ augmente.

**Conséquence** : Le réseau a du mal à apprendre des dépendances à long terme. Dans notre exemple, si la séquence était plus longue, l'influence du mot "good" pourrait disparaître avant d'atteindre la prédiction finale.

> [!math]- 🔬 Démonstration rigoureuse du Vanishing/Exploding Gradient
> 
> ### Qu'est-ce que le Vanishing Gradient ?
> 
> Durant la rétropropagation, plus on se rapproche des premières couches ($n=1$), plus on multiplie de gradients $p_i$ entre eux. Si ces gradients $p_i < 1$, le gradient final devient proche de zéro, ce qui signifie que les premières couches ne seront pas mises à jour :
> 
> $$W^0 \leftarrow W^0 - \eta \frac{\partial L(\theta)}{\partial W^0} \approx W^0$$
> 
> ### Qu'est-ce que l'Exploding Gradient ?
> 
> Dans le scénario opposé, si $p_i > 1$ (par exemple $1 \times 4 \times 2 \times 8...$), le gradient devient exponentiellement grand, causant des mises à jour destructrices.
> 
> ### Preuve mathématique pour les RNN
> 
> Considérons le produit des Jacobiens qui propage l'erreur du début à la fin :
> $$\prod_{j=k}^t \frac{\partial a_{j+1}}{\partial a_j}$$
> 
> Pour un réseau avec :
> $$z_j = W a_{j-1} + b \quad \text{et} \quad a_j = f(z_j)$$
> 
> Le Jacobien local est :
> $$\frac{\partial a_j}{\partial a_{j-1}} = \frac{\partial a_j}{\partial z_j} \frac{\partial z_j}{\partial a_{j-1}} = \text{diag}(f'(z_j)) W$$
> 
> En prenant la norme :
> $$\left\|\frac{\partial a_j}{\partial a_{j-1}}\right\| = \left\|\text{diag}(f'(z_j)) W\right\| \leq \left\|\text{diag}(f'(z_j))\right\| \|W\|$$
> 
> Pour les fonctions d'activation bornées :
> - **Sigmoid** : $f'(a_j) \leq \frac{1}{4} = \gamma$
> - **Tanh** : $f'(a_j) \leq 1 = \gamma$
> 
> Donc : $\left\|\frac{\partial a_j}{\partial a_{j-1}}\right\| \leq \gamma \lambda$ où $\lambda = \|W\|$
> 
> Finalement, pour le produit total :
> $$\left\|\frac{\partial a_t}{\partial a_k}\right\| = \left\|\prod_{j=k+1}^t \frac{\partial a_j}{\partial a_{j-1}}\right\| \leq \prod_{j=k+1}^t \gamma \lambda = (\gamma \lambda)^{t-k}$$
> 
> ### Conclusion :
> - **Si $\gamma \lambda < 1$** : le gradient disparaît (vanishing)
> - **Si $\gamma \lambda > 1$** : le gradient explose (exploding)

### Solutions au Vanishing/Exploding Gradient

**Solutions pour l'Exploding Gradient :**

1. **Truncated Backpropagation** : Limiter la propagation à un nombre fixe de pas temporels
2. **Gradient Clipping** : Normaliser le gradient si sa norme dépasse un seuil

**Solutions pour le Vanishing Gradient :**

1. **Truncated Backpropagation** : Même principe 
2. **Fonctions d'activation** : Utiliser ReLU au lieu de sigmoid ($f'(x) = 1$ si $x > 0$)
3. **Initialisation des poids** : Commencer avec une matrice identité pour $W_{aa}$
4. **Gated RNNs** : LSTM et GRU avec mécanismes de portes

## c - Types d'Architectures RNN

### Classifications selon input/output

![[im10 (1).png|517]]

- **One-to-One** : RNN classique (comme FFNN)
- **One-to-Many** : Génération de texte (ex: légendage d'image)
- **Many-to-One** : Notre exemple (classification de séquence)
- **Many-to-Many** : Traduction automatique, reconnaissance vocale

## d - Comparaison RNN vs FFNN

| Aspect | FFNN | RNN |
|--------|------|-----|
| **Longueur d'entrée** | Fixe | Variable |
| **Partage de paramètres** | Non | Oui (temporel) |
| **Mémoire** | Aucune | État caché |
| **Complexité temporelle** | $O(1)$ | $O(T)$ |
| **Applications** | Images, tableaux | Texte, séries temporelles |

Notre exemple "good bad movie" illustre parfaitement pourquoi les RNN sont nécessaires : le réseau doit **intégrer** l'information des trois mots pour comprendre le sentiment global, une tâche impossible pour un FFNN standard.

# II - Gated RNNs : Solutions au Vanishing Gradient

## Motivation : Les limites des RNN classiques

Les RNN classiques (vanilla RNN) souffrent de limitations majeures qui empêchent leur utilisation efficace sur des séquences longues :

1. **Vanishing Gradient** : Les gradients disparaissent exponentiellement lors de la rétropropagation à travers le temps
2. **Mémoire à court terme** : Incapacité à retenir l'information sur de longues séquences  
3. **Saturation des activations** : Les fonctions sigmoid/tanh saturent et bloquent l'apprentissage

**Solution révolutionnaire** : Les **Gated RNNs** introduisent des **mécanismes de portes** qui contrôlent intelligemment le flux d'information, permettant au réseau de **décider** quoi retenir, oublier ou mettre à jour.

## a - Long Short-Term Memory (LSTM) - 1995

**Concept clé** : LSTM introduit un **état de cellule** séparé de l'état caché, agissant comme une "autoroute de l'information" qui préserve les gradients sur de longues séquences.

### Architecture LSTM

LSTM utilise **trois portes** pour contrôler le flux d'information :

**1. Forget Gate (Porte d'oubli)**
$$f_t = \sigma(W_f [a_{t-1}, x_t] + b_f)$$

Décide quelles informations de l'état de cellule précédent doivent être oubliées. 

**2. Input Gate (Porte d'entrée)**  
$$i_t = \sigma(W_i [a_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [a_{t-1}, x_t] + b_C)$$

Détermine quelles nouvelles informations stocker dans l'état de cellule.

**3. Output Gate (Porte de sortie)**
$$o_t = \sigma(W_o [a_{t-1}, x_t] + b_o)$$

Contrôle quelles parties de l'état de cellule sont exposées comme état caché.

**État de cellule et état caché**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$a_t = o_t \odot \tanh(C_t)$$

![[im13.png]]

![[im12.png|306]]

> [!tip]- 🔑 Intuition des portes LSTM
> - **Forget gate** : "Dois-je oublier que le sujet était au singulier ?"
> - **Input gate** : "Dois-je retenir que le nouveau sujet est au pluriel ?"  
> - **Output gate** : "Dois-je révéler cette information grammaticale maintenant ?"
> 
> **L'état de cellule $C_t$** agit comme une "mémoire à long terme" qui peut traverser de nombreux pas de temps avec des modifications minimales, préservant les gradients.

## b - Gated Recurrent Unit (GRU) - 2014

**Philosophie** : Simplifier LSTM en combinant certaines portes tout en conservant les performances.

### Architecture GRU

![[im11.png|275]]

cc
![[im14.png]]

GRU utilise seulement **deux portes** :

**1. Reset Gate (Porte de remise à zéro)**
$$r_t = \sigma(W_r [a_{t-1}, x_t] + b_r)$$

Détermine combien du passé ignorer pour le nouveau candidat.

**2. Update Gate (Porte de mise à jour)**  
$$z_t = \sigma(W_z [a_{t-1}, x_t] + b_z)$$

Contrôle l'équilibre entre l'information passée et nouvelle (combine forget et input gates du LSTM).

**État caché candidat et final**
$$\tilde{a}_t = \tanh(W_a [r_t \odot a_{t-1}, x_t] + b_a)$$
$$a_t = (1 - z_t) \odot a_{t-1} + z_t \odot \tilde{a}_t$$

> [!tip]- ⚡ Avantages du GRU
> **Simplicité** : Moins de paramètres que LSTM (2 portes vs 3)
> **Efficacité** : Calcul plus rapide, moins de mémoire
> **Performance** : Comparable à LSTM sur de nombreuses tâches
> 
> **Trade-off** : Moins de contrôle fin que LSTM, mais plus simple à optimiser

## c - Comparaisons et Applications

### Comparaison LSTM vs GRU vs Vanilla RNN

| Aspect | Vanilla RNN | LSTM | GRU |
|--------|-------------|------|-----|
| **Portes** | Aucune | 3 (forget, input, output) | 2 (reset, update) |
| **Mémoire** | État caché uniquement | État caché + état cellule | État caché uniquement |
| **Paramètres** | Faible | Élevé | Modéré |
| **Complexité** | Simple | Complexe | Modérée |
| **Performance séquences longues** | Faible | Excellente | Très bonne |
| **Vitesse d'entraînement** | Rapide | Lente | Modérée |

### Applications et choix pratiques

**Utiliser LSTM quand :**
- Séquences très longues (>500 pas de temps)
- Précision maximale requise  
- Dépendances complexes à long terme

**Utiliser GRU quand :**
- Ressources computationnelles limitées
- Séquences moyennes (<200 pas de temps)
- Prototypage rapide

### Impact révolutionnaire

Les Gated RNNs ont rendu possibles des applications comme la traduction automatique, la génération de texte, et l'analyse de sentiment sur de longs documents - des tâches impossibles avec les vanilla RNN.

# III - Architectures Encoder-Decoder (2014)

## a - Seq2seq : Le Modèle de Base

### Motivation : Un Modèle de Langage Conditionnel

Seq2seq représente une **révolution conceptuelle** : c'est un **modèle de langage conditionnel** qui génère une séquence de sortie conditionnée par une séquence d'entrée.

**Neural Machine Translation (NMT)** : La traduction automatique neuronale utilise un seul réseau neuronal pour effectuer la traduction. L'architecture Seq2Seq ([Sutskever et al. NIPS 2014](https://papers.nips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html)) se divise en **deux blocs** :

- **Encoder** : prend la séquence d'entrée et l'encode dans un **vecteur de contexte** de taille fixe. Cette représentation résume le sens de toute la séquence
- **Decoder** : utilise ce vecteur de contexte comme "graine" pour générer la séquence de sortie

Pour cette raison, les modèles Seq2seq sont souvent appelés **"modèles encoder-decoder"**.

### Architecture mathématique

Nous avons deux séquences de mots avec leurs variables aléatoires correspondantes :

JAI PERDU LAIMAGE

### Seq2seq comme Modèle de Langage Conditionnel

**Rappel** : Un modèle de langage classique estime la probabilité d'une phrase :

![[im7 (1).png]]

**Seq2seq** : La traduction automatique fonctionne différemment :

![[im8.png]]

> [!info]- 🎯 L'insight révolutionnaire
> **Réseau vert** : **Encoder** qui transforme la phrase d'entrée en représentation latente
> 
> **Réseau violet** : **Decoder** très similaire au modèle de langage classique
> 
> **Différence clé** : Au lieu de commencer avec un vecteur zéro, le decoder utilise l'encodage de la phrase d'entrée. C'est pourquoi on parle de **Modèle de Langage Conditionnel**.
> 
> **Exemple** : Modéliser une phrase anglaise conditionnée par une phrase française : $P(\text{anglais}|\text{français})$

### Algorithmes de Décodage

**Motivation du problème** : D'une phrase source en français $x$, nous voulons trouver la meilleure traduction possible $y$ :

$$P(y^1, \ldots, y^{T_y} | x)$$

Plutôt que d'échantillonner aléatoirement, nous résolvons un problème d'optimisation :

$$\underset{y^1, \ldots, y^{T_y}}{\arg \max} \, P(y^1, \ldots, y^{T_y} | x)$$

**Types d'algorithmes de décodage** :

**(1) Greedy Decoding** : À chaque pas de temps, choisir le mot le plus probable $x_t = \underset{\tilde{x}_t}{\arg \max} \, P(\tilde{x}_t | x_1, \ldots, x_{t-1})$. **Problème** : Pas de retour en arrière possible, erreurs en cascade.

**(2) Exhaustive Search** : Trouver une traduction qui maximise $P(y|x) = \prod_{t=1}^{T} P(y_t | y_1, \ldots, y_{t-1}, x)$. **Problème** : Complexité $O(|V|^T)$ intenable.

**(3) Beam Search** : Maintenir $K$ candidats à chaque pas de temps. **Compromis optimal** entre précision et efficacité.

### Exemple de Beam Search

**Beam width $B = 3$** (garder 3 mots à chaque étape) :

jai perdu l'image

![[im5 (1).png|490]]

### Alternatives Softmax

**Softmax avec température** $\tau$ :

$$P_t(w) = \frac{\exp(s_w / \tau)}{\sum_{w' \in V} \exp(s_{w'} / \tau)}$$

**Augmenter $\tau$** : Distribution plus uniforme → sortie diverse  
**Diminuer $\tau$** : Distribution concentrée → sortie focalisée

## b - Attention : La Solution au Goulot d'Étranglement

### Le Problème du Goulot d'Étranglement

Dans l'architecture Seq2seq standard, **toute l'information** de la phrase source doit être compressée dans le **dernier état caché de l'encoder**. 

**Problème** : **Goulot d'étranglement informationnel** - toute la décision repose sur un seul vecteur !

![[attention1.png]]

> [!warning]- 🚫 Limitations critiques du Seq2seq classique
> **Perte d'information** : Les premiers mots de la séquence source "s'effacent" progressivement
> 
> **Séquences longues** : Performance dégradée car un seul vecteur ne peut pas retenir tous les détails
> 
> **Pas d'alignement** : Le decoder ne sait pas sur quelle partie de la source se concentrer

### La Solution : Mécanisme d'Attention

**Insight clé** : Au lieu de s'appuyer sur un seul vecteur de contexte, permettre au decoder d'avoir une **connexion directe** avec tous les états cachés de l'encoder, et **apprendre** sur quelle partie se concentrer à chaque étape.

![[attention2.png]]

**Algorithme d'Attention (4 étapes)** :

**(1) Calcul des scores d'attention** : $\mathbf{e}^t = [s_t^T a_1, s_t^T a_2, \ldots, s_t^T a_N] \in \mathbb{R}^N$

**(2) Normalisation par softmax** : $\boldsymbol{\alpha}^t = \text{softmax}(\mathbf{e}^t) \in \mathbb{R}^N$

**(3) Calcul du vecteur de contexte** : $\mathbf{c}_t = \sum_{i=1}^{N} \alpha_i^t \mathbf{a}_i \in \mathbb{R}^h$

**(4) Intégration dans le decoder** : $[\mathbf{c}_t; \mathbf{s}_t] \in \mathbb{R}^{2h}$

> [!info]- 🎯 Intuition du mécanisme d'attention
> **Query** : État caché du decoder $s_t$ ("Que cherche-t-on ?")
> 
> **Keys & Values** : États cachés de l'encoder $a_1, \ldots, a_N$ ("Où chercher ?")
> 
> **Attention weights** : $\alpha^t$ ("Combien d'attention porter à chaque position ?")
> 
> **Context vector** : $c_t$ ("Résumé pondéré de l'information pertinente")

### Avantages Révolutionnaires

**Amélioration des performances NMT** : Le decoder se concentre sur les parties **pertinentes** de la phrase source. **Résolution du goulot d'étranglement** : Accès direct à **toute** l'information de l'encoder. **Atténuation du vanishing gradient** : Création de **raccourcis** vers les états éloignés. **Interprétabilité** : Les poids d'attention $\alpha^t$ révèlent **quels mots source** influencent chaque prédiction !

![[attention3 1.png|298]]

> [!tip]- 🔍 Révélation de l'alignement automatique
> **Jamais explicitement entraîné** : Le modèle apprend l'alignement comme effet de bord !
> 
> **Visualisation** : Les matrices d'attention révèlent les correspondances mot-à-mot
> 
> **Applications** : Analyse linguistique, détection d'erreurs, compréhension du modèle

### Variantes d'Attention

**Fonctions de score d'attention** avec $\mathbf{a}_1, \ldots, \mathbf{a}_N \in \mathbb{R}^{d_1}$ (encoder) et $\mathbf{s} \in \mathbb{R}^{d_2}$ (decoder) :

**Dot-product attention** : $e_i = \mathbf{s}^T \mathbf{a}_i \in \mathbb{R}$ (nécessite $d_1 = d_2$)

**Multiplicative attention** : $e_i = \mathbf{s}^T \mathbf{W} \mathbf{a}_i \in \mathbb{R}$ où $\mathbf{W} \in \mathbb{R}^{d_2 \times d_1}$

**Additive attention** : $e_i = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{a}_i + \mathbf{W}_2 \mathbf{s}) \in \mathbb{R}$ où $\mathbf{W}_1 \in \mathbb{R}^{d_3 \times d_1}$, $\mathbf{W}_2 \in \mathbb{R}^{d_3 \times d_2}$, $\mathbf{v} \in \mathbb{R}^{d_3}$

### Impact Transformationnel

**Les architectures encoder-decoder révolutionnent le NLP** :
- **Traduction automatique** : Google Translate, DeepL
- **Résumé automatique** : Génération de résumés
- **Dialogue** : Chatbots et assistants
- **Génération de code** : Autocomplétion intelligente

> [!success]- 🚀 Révolution conceptuelle
> **2014** : Seq2seq introduit le paradigme encoder-decoder
> 
> **2014** : Attention résout le goulot d'étranglement  
> 
> **2017** : "Attention is All You Need" → Transformers
> 
> **Aujourd'hui** : Fondation de tous les LLMs modernes (GPT, BERT, etc.)

L'attention de **Bahdanau et al. 2014** pose les bases conceptuelles de toute l'IA moderne ! 🧠✨