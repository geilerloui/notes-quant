---
title: Word Embeddings - NLP
---
# Word Embeddings

> Cette note couvre les **représentations vectorielles de mots** (word embeddings) — comment passer d'un texte brut à des vecteurs que des modèles peuvent traiter. On suit la progression historique : one-hot (naïf) → représentations distributionnelles (co-occurrence + reweighting + LSA) → représentations *prediction-based* (Word2Vec et ses variantes), avec un mot sur GloVe et l'évaluation.

> 💡 **Position dans la zoologie NLP.** Les word embeddings statiques (Word2Vec, GloVe) ont été remplacés en pratique par les **représentations contextuelles** (BERT, GPT) traitées dans [[06_Représentations contextuelles]]. Mais comprendre Word2Vec reste indispensable : c'est la grammaire commune, et beaucoup d'idées (skip-gram, negative sampling, NCE) sont reprises sous d'autres noms en deep learning moderne.

Bibliographie principale :
- [Demystifying neural networks in skip-gram (blog très clair)](https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling)
- [Vidéo pédagogique (FR)](https://www.youtube.com/watch?v=WgOGzR0p1DM&list=PLzjg2z2kYUrinAmDbCTSH6AwwuIS4Ir39&index=5&t=258s)

---

## I. Représentation one-hot

**Approche.** La méthode la plus naïve. Étant donné un corpus, on considère l'ensemble $V$ de tous les mots uniques (le **vocabulaire**). On représente chaque mot par un vecteur one-hot de taille $|V|$.

![[freq2.png|542]]
**Figure 1.** Représentation one-hot : un 1 à la position du mot, des 0 partout ailleurs.

> [!warning] Problèmes du one-hot
> - $V$ est très grand (ex : 50k pour PTB, 13M pour Google 1T corpus).
> - Aucune notion de **similarité** entre mots.
> - Idéalement, on voudrait que les représentations de *cat* et *dog* (animaux domestiques) soient plus proches que *cat* et *truck*. Avec des one-hot, la distance euclidienne entre **n'importe quels** deux mots du vocabulaire vaut $\sqrt{2}$.

![[freq1.png|496]]
**Figure 2.** La distance euclidienne entre deux mots distincts vaut $\sqrt{2}$, et la distance cosinus vaut 0.

---

## II. Représentations distributionnelles

> [!warning] Définition
> Les **représentations distributionnelles** sont le cas particulier où les données viennent entièrement de comptages de **co-occurrences** dans des corpus. Les vecteurs sont des nombres réels, et les modèles sont appelés **vector space models** (VSM).
>
> Si un réseau de neurones est utilisé pour entraîner les représentations, on parle alors de **neural representations**. Le terme **word embedding** est aussi utilisé pour les représentations distribuées, y compris distributionnelles.

> 💡 **Pourquoi "embedding" ?** Le terme rappelle que les représentations vectorielles n'ont de sens que **plongées dans un espace unifié** (généralement une matrice) avec d'autres représentations du même type, qu'on peut comparer entre elles.

### A. Matrice de co-occurrence

On la construit en deux étapes.

**(1) Choisir le type de matrice.** Plusieurs options :
- $\text{word} \times \text{word}$
- $\text{word} \times \text{document}$
- $\text{word} \times \text{discourse context}$
- ...

**(2) Choisir les paramètres.** Une matrice de co-occurrence se définit par deux choix : la **taille de fenêtre** et le **scaling** (poids uniforme ou décroissant avec la distance au mot central).

![[distrib1.png|410]]
**Figure 3.** Illustration sur la première phrase de *Finnegans Wake* (qui est aussi sa dernière). On choisit une fenêtre, et on peut soit pondérer uniformément les mots de la fenêtre, soit donner moins de poids aux mots éloignés du centre.

**Notre choix.** Fenêtre de taille $k = 2$ et matrice $\text{word} \times \text{context}$ basée sur le corpus précédent.

![[distrib2.png|405]]
**Figure 4.** Matrice de co-occurrence résultante.

### B. Comparaison vectorielle

**Le contexte.** Que veut dire "être similaire" ? On part d'une matrice $\text{word} \times \text{document}$ avec deux documents $x$ et $y$.
- $A$ et $B$ sont similaires si on les rescale : il y a plus de $y$ que de $x$ dans les deux.
- $B$ et $C$ sont similaires en termes de **magnitude globale**.

**Distance euclidienne.** La métrique standard, mesure la distance la plus courte entre deux points :

$$\text{euclidean}(u, v) = \sqrt{\sum_{i=1}^n |u_i - v_i|^2}.$$

> 💡 **Limite.** Euclidienne favorise la **magnitude** plutôt que la similarité de proportion. Linguistiquement, $A$ et $B$ ressemblent à *superb* et *good* (sens proche), tandis que $B$ et $C$ ressemblent à *good* et *bad* (mêmes nombres mais sens opposés). Euclidienne ne capte pas ça.

![[distrib11.png|583]]
**Figure 5.** Distance euclidienne — $A$ et $B$ apparaissent éloignés à cause de la magnitude.

**Length normalization.** On normalise par la **norme L2** :

$$\|u\|_2 = \sqrt{\sum_{i=1}^n u_i^2}, \qquad \tilde u = \left[\frac{u_1}{\|u\|_2}, \frac{u_2}{\|u\|_2}, \ldots, \frac{u_n}{\|u\|_2}\right].$$

Après length normalization, tous les points sont sur la sphère unité — $A$ et $B$ deviennent proches.

![[distrib12.png|539]]
**Figure 6.** Length normalization — magnitude éliminée, seule la direction compte.

**Distance cosinus.** La distance utilisée par défaut quand rien n'est précisé. Elle fait simultanément une distance euclidienne dans l'espace length-normalisé :

$$\cos(u, v) = 1 - \frac{\sum_{i=1}^n u_i v_i}{\|u\|_2 \cdot \|v\|_2}.$$

![[distrib9.png|576]]

> 💡 **Équivalence cosinus / L2-norm puis cosinus.** Si on commence par normaliser L2 puis qu'on calcule la distance cosinus, on obtient le même résultat — le dénominateur de la formule cosinus inclut déjà la normalisation.

![[distrib10.png|574]]

### C. Schémas de repondération (reweighting)

#### C.1 Normalisations

**(i) L2 norming.** Vu juste avant.

**(ii) Distribution de probabilité.** Pour un vecteur $u$ à valeurs positives :

$$\tilde u_i = \frac{u_i}{\sum_{j=1}^n u_j}.$$

> 💡 Ces normalisations **amplifient** les écarts par rapport à l'attente moyenne sur la ligne et la colonne.

#### C.2 PMI (Pointwise Mutual Information)

On passe en log space pour révéler des écarts invisibles dans l'espace original :

$$\boxed{\text{pmi}(x; y) = \log \frac{p(x, y)}{p(x) \, p(y)}}$$

> [!warning] Limite — sur-amplification des petites valeurs
> ![[distrib3.png]]
>
> Sur cette table, le 1 mis en évidence pourrait être une erreur du corpus. Mais comparé à ses lignes et colonnes, il prend une valeur énorme en PMI. **PMI sur-amplifie les petits comptages**.

> [!example] Calcul de PMI
> $$\text{pmi}(A; d_1) = \log \frac{p(A, d_1)}{p(A) \, p(d_1)} = \log \frac{p(A, d_1)}{\sum_D p(A, D) \cdot \sum_W p(d_1, W)} = \log \frac{0.11}{0.44 + 0.33} = -0.28.$$

**Variantes de PMI.**

- **PMI Zero** :

$$\text{PMI}_0(w, c) = \begin{cases} \text{PMI}(w, c) & \text{si } p(x, y) > 0 \\ 0 & \text{sinon} \end{cases}$$

- **Positive PMI** :

$$\text{PPMI}(w, c) = \begin{cases} \text{PMI}(w, c) & \text{si } \text{PMI}(w, c) > 0 \\ 0 & \text{sinon} \end{cases}$$

![[distrib6.png|481]]
**Figure 7.** Matrice de PPMI calculée sur le dataset initial.

#### C.3 TF-IDF

> [!warning] L'idée
> TF-IDF amplifie les valeurs des termes qui apparaissent dans **très peu de documents**. Un terme avec un TF-IDF élevé est très associé spécifiquement à un document — pas à beaucoup d'autres.

![[distrib4.png|519]]

$$\text{TF-IDF} = \text{TF} \times \text{IDF}.$$

**IDF en pratique.** L'IDF est à son **pic** quand un mot apparaît dans **un seul** document — ce qui le rend spécifique à ce document.

![[distrib5.png|438]]

### D. Latent Semantic Analysis (LSA)

**Référence.** Deerwester et al. 1990. Une des techniques de réduction de dimension les plus anciennes et utilisées. Aussi connue sous le nom de **Truncated Singular Value Decomposition (Truncated SVD)**. C'est souvent considéré comme un **baseline standard, très difficile à battre**.

**Motivation.** Même avec tous les schémas de repondération vus, le dataset reste de très grande dimension ($|V|$) et très **creux** (sparse). LSA propose une réduction de dimension par SVD tronquée.

#### D.1 Approximation de rang faible (low-rank approximation)

SVD donne une approximation de rang $k$ de la matrice originale :

$$X = X_{\text{PPMI}, m \times n} = U_{m \times k} \, \Sigma_{k \times k} \, V^T_{k \times n}.$$

Ici $X_{\text{PPMI}}$ (qu'on note $X$) est la matrice de co-occurrence avec valeurs PPMI. SVD donne la **meilleure** approximation de rang $k$ de $X$. Cela permet de **découvrir des sémantiques latentes** dans le corpus.

$$X_{m \times n} = \begin{bmatrix} \uparrow & \cdots & \uparrow \\ u_1 & \cdots & u_k \\ \downarrow & \cdots & \downarrow \end{bmatrix}_{m \times k} \begin{bmatrix} \sigma_1 & & \\ & \ddots & \\ & & \sigma_k \end{bmatrix}_{k \times k} \begin{bmatrix} \leftarrow & v_1^T & \rightarrow \\ & \vdots & \\ \leftarrow & v_k^T & \rightarrow \end{bmatrix}_{k \times n}$$

> 💡 **Lecture.** Le produit peut s'écrire comme une somme de $k$ matrices de rang 1 :
> - Si on tronque à $\sigma_1 u_1 v_1^T$, on a la meilleure approximation de rang 1 de $X$ (théorème SVD).
> - Si on tronque à $\sigma_1 u_1 v_1^T + \sigma_2 u_2 v_2^T$, meilleure approximation de rang 2.
> - Etc.

![[distrib7.png]]
**Figure 8.** Après reconstruction de rang faible avec SVD, la **co-occurrence latente** entre $\{system, machine\}$ et $\{human, user\}$ devient visible — alors qu'elle était cachée dans la matrice originale creuse.

#### D.2 Comparaison vectorielle dans LSA

![[distrib8.png]]

**Représentation compressée.** Le produit scalaire entre les lignes de la matrice $W_{\text{word}} = U \Sigma$ est le même qu'entre les lignes de $\hat X$ :

$$\begin{aligned}
\hat X \hat X^T &= (U \Sigma V^T)(U \Sigma V^T)^T \\
&= (U \Sigma V^T)(V \Sigma U^T) \\
&= U \Sigma \Sigma^T U^T \quad (\text{car } V^T V = I) \\
&= U \Sigma (U \Sigma)^T = W_{\text{word}} W_{\text{word}}^T.
\end{aligned}$$

Conventionnellement :
- $W_{\text{word}} = U \Sigma \in \mathbb{R}^{m \times k}$ : représentation des $m$ mots du vocabulaire.
- $W_{\text{context}} = V$ : représentation des mots du contexte.

---

## III. Embeddings prediction-based — Word2Vec

### A. Introduction

> [!example] Fil rouge — corpus jouet
> On considère le corpus suivant :
>
> - *"Marrakech est connue par sa place jamaa-el-fna et son climat chaud et sec"*
> - *"Le numérique est en train de transformer le monde"*
>
> Pour simplifier, on extrait le vocabulaire :
>
> $V = \{$Marrakech, est, connue, par, sa, place, jamaa-el-fna, et, son, climat, chaud, sec, le, numérique, en, train, de, transformer, monde$\}$
>
> Avec $|V| = 19$.

**Word2Vec.** Un **mot central** $w_t$ est fortement corrélé aux mots qui constituent son **contexte**, où $m$ est un hyperparamètre — la longueur de la fenêtre autour de $w_t$.

![[skip8.png]]
**Figure 9.** Mot central et contexte dans une fenêtre de taille $m$.

> [!warning] Word2Vec comme self-supervised learning
> Word2Vec peut être vu comme une tâche de **pre-training** dans le paradigme self-supervised, qui permet ensuite de poursuivre avec des **downstream tasks** : classification de documents, NER, analyse de sentiments, etc.

> 💡 **Sémantique.** Discipline qui s'intéresse au sens des mots. Un même mot peut représenter plusieurs idées différentes.

> 💡 **L'hypothèse distributionnelle.** Word2Vec adopte la linguistique distributionnelle : *les mots qui apparaissent dans des contextes similaires tendent à avoir des sens similaires.* Exemple :
> - L'étudiant essaie **de comprendre** des notions mathématiques.
> - L'étudiant essaie **d'assimiler** des notions mathématiques.
>
> *Comprendre* et *assimiler* partagent des contextes communs → leurs vecteurs seront similaires.

**A.1 Les espaces**

> [!warning] Espace des sens
> Espace **dense** de dimension $d \ll |V|$ pour représenter les sens des mots. Exemple :
>
> $$\text{assimiler} = \begin{pmatrix} 0.125 & -0.380 & 0.760 & 0.123 & -0.590 & \ldots \end{pmatrix}_{1 \times d}.$$
>
> On définit une matrice $[W_s]_{|V| \times d}$ qui projette un mot dans l'espace des sens.

> [!warning] Espace des contextes
> De façon similaire, un mot du contexte est représenté dans un espace sémantique. Une matrice $[W_c]_{|V| \times d}$ donne sa projection dans l'espace des contextes.

### B. Skip-gram

> [!warning] Définition
> Le modèle **Skip-gram** apprend de façon non supervisée à **prédire le contexte d'un mot à partir du mot central**. Exemple avec une fenêtre de taille 2 :

![[skip9.png]]
**Figure 10.** Skip-gram : à partir du mot central, on prédit chacun des mots du contexte.

Pour un corpus $\mathbf{w}$, on note $\mathbf{C(w)}$ leurs contextes.

#### B.1 Forward

**Matrices.** On a :
- $W_s^T = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_V]$ : espace des sens (joue le rôle d'une **lookup table**).
- $W_c^T = [\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_V]$ : espace des contextes.

En anglais : *word embedding matrix* (lookup table) et *word embedding matrix for context words*.

![[skip1.png]]
**Figure 11.** Forward pass : la probabilité de trouver "palais" dans le contexte de "climat" sélectionné aléatoirement vaut 0.125.

**Paramètres.** $\theta = \{\mathbf{v}_1, \ldots, \mathbf{v}_V, \mathbf{u}_1, \ldots, \mathbf{u}_V\}$ constitue l'ensemble des paramètres du modèle.

![[skip10.png]]

> 💡 **Le produit scalaire $u^T_{\text{chaud}} \cdot v_{\text{climat}}$** caractérise la similarité entre les vecteurs $u_{\text{chaud}}$ et $v_{\text{climat}}$. Plus il est grand et positif, plus $\cos(u_{\text{chaud}}, v_{\text{climat}})$ est proche de 1, et donc plus les vecteurs "vont dans le même sens".

#### B.2 Backward

**Construction de la fonction de coût.**

$$\log L(\theta) = \log \prod_{w \in C_{\text{train}}} \prod_{c \in \mathbf{C}(w)} p(c \mid w; \theta).$$

D'où la log-vraisemblance :

$$\ell(\theta) = \sum_{w \in C_{\text{train}}} \sum_{c \in \mathbf{C}(w)} \log p(c \mid w; \theta).$$

On cherche à minimiser :

$$J(\theta; w) = -\frac{1}{|C_{\text{train}}|} \ell(\theta) = -\frac{1}{|C_{\text{train}}|} \sum_{w \in C_{\text{train}}} \sum_{c \in \mathbf{C}(w)} \log P(c \mid w; \theta).$$

**Simplification (softmax).**

$$p(c \mid w; \theta) = \frac{\exp(u_c^T \cdot v_w)}{\sum_{i=1}^V \exp(u_i^T \cdot v_w)}.$$

D'où :

$$\begin{aligned}
J &= -\sum_{i=1}^V y_i \log \hat y_i \\
&= -\sum_{i=1}^V y_i \left[ u_i^T v_c - \log \sum_{w=1}^V \exp(u_w^T v_c) \right] \\
&= -y_k \left[ u_k^T v_c - \log \sum_{w=1}^V \exp(u_w^T v_c) \right].
\end{aligned}$$

**Update des matrices de poids.** Dérivée par rapport à la matrice d'embeddings d'inputs :

$$\begin{aligned}
\frac{\partial J}{\partial v_c} &= -\left[ u_k - \frac{\sum_{w=1}^V \exp(u_w^T v_c) u_w}{\sum_{x=1}^V \exp(u_x^T v_c)} \right] \\
&= \sum_{w=1}^V \hat y_w u_w - u_k.
\end{aligned}$$

> [!note]- Réécriture matricielle
> En posant $W_c = U = [u_1, u_2, \ldots, u_V]$ (matrice composée des vecteurs colonnes $u_k$), on peut réécrire :
> - $u_k = U \cdot y$ (multiplication matrice/vecteur).
> - $\sum_w \hat y_w u_w = U \cdot \hat y$ (transformation linéaire des $u_w$ scalée par $\hat y_w$).
>
> Donc :
>
> $$\frac{\partial J}{\partial v_c} = U^T (\hat y - y) = W_s^T (\hat y - y).$$

Si on a plusieurs termes dans la fenêtre, on pose $\sum_{c=1}^C e_c = \sum_{c=1}^C (\hat y_c - y)$. La dérivée devient :

$$\boxed{\frac{\partial J}{\partial v_c} = W_s^T \sum_{c=1}^C e_c.}$$

**Backprop.**

$$W_s^{(\text{new})} = W_s^{(\text{old})} - \eta \cdot \left( W_s^T \sum_{c=1}^C e_c \right).$$

Pour la matrice de sortie :

$$\boxed{W_{\text{output}}^{(\text{new})} = W_{\text{output}}^{(\text{old})} - \eta \cdot h \cdot \sum_{c=1}^C e_c.}$$

#### B.3 Application

Une fois l'apprentissage terminé, on extrait la matrice $W_s$ (l'espace des sens), la **word embedding matrix** pour chaque terme. Une **PCA** ramène ensuite à dimension 2, ce qui permet de visualiser des **clusters de mots**. On peut aussi faire l'arithmétique célèbre :

$$\mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{woman}} + \mathbf{v}_{\text{man}} \approx \mathbf{v}_{\text{king}}.$$

### C. Continuous Bag of Words (CBOW)

> [!warning] Définition
> **CBOW** apprend l'inverse de Skip-gram : prédire le **mot central** à partir de la donnée du contexte. On concatène les mots du contexte en entrée.

![[skip7.png|378]]
**Figure 12.** CBOW : on concatène les mots du contexte pour prédire le mot central.

### D. Applications

#### D.1 Word embedding visualization

On projette la matrice de l'espace des sens dans un espace de dimension réduite (typiquement via **t-SNE**) pour obtenir une représentation des sémantiques. On récupère le vecteur du mot central dans $W_s$ et les top 5 mots de $W_c$, puis on projette.

![[app-1.png]]
![[app-2.png]]
**Figure 13.** Projection des top 5 mots pour trois mots centraux différents.

#### D.2 Encodage des analogies

Les vecteurs de mots dans l'espace des sens **obéissent aux lois de l'analogie** et encodent des informations sémantiques et syntactiques. Exemple :

$$\mathbf{v}_{\text{queen}} - \mathbf{v}_{\text{woman}} + \mathbf{v}_{\text{man}} \approx \mathbf{v}_{\text{king}}.$$

![[app-3.png]]
**Figure 14.** Représentation graphique de l'analogie queen-woman+man ≈ king.

---

## IV. Word2Vec — alternatives au softmax

### A. Pourquoi remplacer le softmax ?

Il y a un problème avec le Skip-gram vanilla : le **softmax est computationnellement très cher**. Il faut scanner toute la matrice d'embeddings de sortie ($W_{\text{output}}$) pour calculer la distribution de probabilité sur les $V$ mots, où $V$ peut être de plusieurs millions.

![[images/3-Apprentissage automatique/05_Generative Models/score based/im1-2.png]]
**Figure 15.** Le coût du softmax explose avec la taille du vocabulaire.

### B. Negative Sampling

> [!warning] Idée
> Avec **Negative Sampling**, les vecteurs de mots ne sont plus appris en prédisant les mots de contexte d'un mot central. À la place, le modèle utilise une **sigmoïde** pour apprendre à différencier les vrais mots de contexte (positifs) de mots tirés au hasard (négatifs) depuis une distribution de bruit $P_n(w)$.

#### B.1 Construction de la fonction de coût

Soit $D$ l'ensemble des paires correctes $(w, c)$ dans le corpus, et $D'$ l'ensemble des paires incorrectes $(w, r)$. $D'$ est construit en tirant aléatoirement un mot de contexte $r$ qui n'est jamais apparu avec $w$. Comme avant, $v_w$ est la représentation du mot $w$, $u_c$ celle du contexte, et $u_r$ celle du mot hors-contexte.

> [!example] Exemples de $D$ et $D'$
> $$D = [(\text{sat, on}), (\text{sat, a}), (\text{sat, chair}), (\text{on, a}), (\text{on, chair}), \ldots]$$
> $$D' = [(\text{sat, oxygen}), (\text{sat, magic}), (\text{chair, sad}), (\text{chair, walking})]$$

**Modèle (Local NCE).** Local NCE simplifie le problème d'apprendre $p_\theta(w \mid c)$ à apprendre un classifieur binaire $p(d \mid w, c)$ où :
- $d = 0 \Rightarrow x \sim q(w)$ : le data point est du bruit.
- $d = 1 \Rightarrow x \sim p(w \mid c)$ : c'est un vrai data point.

Pour chaque vrai data point, on tire $k$ samples de $q(x)$ et on leur assigne $d = 0$.

$$\begin{aligned}
p(D = 0 \mid w, c) &= \frac{k \cdot q(w)}{p(w \mid c) + k \cdot q(w)} \\
p(D = 1 \mid w, c) &= \frac{p(w \mid c)}{p(w \mid c) + k \cdot q(w)}.
\end{aligned}$$

**Choix pour le langage.** On prend $k = |V|$ et $q(x) = 1/|V|$ uniforme sur le vocabulaire. La probabilité de tirer un vrai point est environ $1/|V|$.

Avec ces choix, l'objectif NCE se simplifie. Pour $(w, r) \in D'$ on maximise :

$$p(D = 0 \mid x, c) = \frac{1}{f_\theta(x, c) + 1} = \sigma(-u_r^T v_w) = \frac{1}{1 + e^{u_r^T v_w}}.$$

Pour $(w, c) \in D$ on maximise :

$$p(D = 1 \mid x, c) = \sigma(u_c^T v_w) = \frac{1}{1 + e^{-u_c^T v_w}}.$$

#### B.2 Loss finale

$$\mathcal{L}_{\text{LocalNCE,MC}} = \sum_{(x, c) \in D} \left( \log p(D = 1 \mid x, c) + \sum_{i=1, x' \sim q}^k \log p(D = 0 \mid x', c) \right).$$

En combinant les deux cas :

$$\begin{aligned}
& \max_\theta \prod_{(w,c) \in D} p(z = 1 \mid w, c) \prod_{(w, r) \in D'} p(z = 0 \mid w, r) \\
&= \max_\theta \sum_{(w, c) \in D} \log \sigma(v_c^T v_w) + \sum_{(w, r) \in D'} \log \sigma(-v_r^T v_w).
\end{aligned}$$

> 💡 **Interprétation.** $\sigma(v_c^T v_w)$ rapproche les vecteurs contexte et mot. $\sigma(-v_r^T v_w)$ éloigne les vecteurs des mots hors contexte. Bilan : les mots apparaissant ensemble sont rapprochés, les autres sont éloignés.

#### B.3 Backward

On ne déroule pas tout le processus, mais on souligne qu'on **ne calcule plus de softmax**. Sur la dernière couche : en vert le mot central, en orange le contexte, en jaune les mots négatifs.

![[skip4 (1).png]]
**Figure 16.** Backprop avec negative sampling — la dernière couche n'a plus besoin de scanner tout le vocabulaire.

#### B.4 Sampling des fake data — la distribution unigramme modifiée

Dans le papier original (Mikolov et al.), on tire $k$ paires négatives $(w, r)$ pour chaque paire positive $(w, c)$. Le mot de contexte aléatoire est tiré depuis une **distribution unigramme modifiée** :

$$r \sim \frac{\text{count}(r)^{3/4}}{N}.$$

![[skip3 (1).png]]

**Le modèle unigramme.** Un modèle qui suppose que les mots d'une phrase sont **complètement indépendants**. Ex : pour "is and and she" :

$$P(\text{is, and, and, she}) = P(\text{is}) P(\text{and}) P(\text{and}) P(\text{she}).$$

On construit une matrice de fréquences, qu'on convertit en probabilités. Pour un mot $w_i$ de fréquence $f(w_i)$ :

$$P(w_i) = \frac{f(w_i)}{\sum_{j=0}^n f(w_j)}.$$

Selon Mikolov, on remplace $f(w_i)$ par $f(w_i)^{3/4}$ :

$$P(w_i) = \frac{f(w_i)^{3/4}}{\sum_{j=0}^n f(w_j)^{3/4}}.$$

> 💡 **Pourquoi $3/4$ ?** La probabilité originale pour "is" est $0.9$. Après élévation à $3/4$, elle devient $0.92$ — peu de changement. Mais "bombastic" passe de $0.01$ à $0.032$ — un facteur 3. **L'exposant $3/4$ aplatit la distribution** : les mots rares ont une probabilité plus élevée d'être tirés comme négatifs, ce qui rend l'apprentissage plus efficace.

### C. Hierarchical Softmax

#### C.1 Idée — arbre de Huffman

Tous les mots du vocabulaire sont les **feuilles d'un arbre**. On construit un **arbre de Huffman** : les mots rares sont aux niveaux profonds, les mots fréquents proches de la racine.

> 💡 **Intuition information theory.** Comme dans le codage de Huffman classique, plus un mot est utilisé, moins de bits il requiert. Les feuilles loin de la racine sont des mots rares, celles proches sont des mots communs. Les probabilités sur les arêtes sont calculées d'après la fréquence des mots à gauche/droite.

![[skip5 (1).png]]
**Figure 17.** Arbre de Huffman pour le vocabulaire.

La matrice de sortie est modifiée : on a un ensemble de mots × index de nœud (de 0 à 6). On multiplie le vecteur d'embedding du mot courant par son index de nœud, puis on suit les arêtes jusqu'au mot cible.

C'est un **arbre de Huffman** parce que chaque chemin est une séquence de bits — on choisit notre direction selon le produit scalaire entre le mot d'entrée et la ligne (basée sur le nœud) de la matrice de sortie.

![[skip6 (1).png]]
![[hierarch-1.png]]

#### C.2 Décomposition en décisions binaires

Chaque outcome (chaque mot du vocabulaire) est le résultat d'une **séquence de décisions binaires**. Exemple :

$$P(\text{"time"} \mid C) = P_{n_0}(\text{right} \mid C) \cdot P_{n_1}(\text{left} \mid C) \cdot P_{n_2}(\text{right} \mid C),$$

où $P_n(\text{right} \mid C)$ est la probabilité de choisir l'enfant droit en partant du nœud $n$. Comme il n'y a que deux issues :

$$P_n(\text{right} \mid C) = 1 - P_n(\text{left} \mid C).$$

Ces distributions sont modélisées avec la sigmoïde :

$$P_n(\text{left} \mid C) = \sigma(\gamma_n^T \alpha_C),$$

où $\gamma_n$ est un vecteur de coefficients pour chaque nœud interne — ce sont les nouveaux paramètres qui remplacent les $\beta_w$ du softmax.

> 💡 **Le gain.** La probabilité d'un seul outcome $P(w \mid C)$ ne dépend plus que des $\gamma_n$ des nœuds internes sur le **chemin de la racine à la feuille** $w$. Pour un arbre **équilibré**, le nombre de paramètres est seulement **logarithmique** en la taille $|V|$ du vocabulaire — au lieu de linéaire.

---

## V. Relation entre SVD et Word2Vec

*À développer.*

> 💡 **Le résultat clé.** Levy & Goldberg (2014) ont montré que Word2Vec avec negative sampling est implicitement en train de factoriser une **matrice de PMI shiftée** ($\text{PMI} - \log k$ où $k$ est le nombre de samples négatifs). Donc Word2Vec et LSA/SVD ne sont **pas si différents** que ce que l'opposition "count-based vs prediction-based" laisse croire — ils factorisent la même matrice sous-jacente avec des objectifs différents.

---

## VI. GloVe

*À développer.*

> 💡 **L'idée.** GloVe (Global Vectors, Pennington et al. 2014) cherche à combiner les avantages des méthodes count-based (LSA — qui utilisent les statistiques globales du corpus) et prediction-based (Word2Vec — qui captent bien les analogies). L'objectif factorise directement la matrice de log-co-occurrences avec une perte pondérée.

---

## VII. Évaluer les représentations de mots

*À développer.*

> 💡 **Deux familles d'évaluation.**
> - **Intrinsèque** : tester sur des datasets de similarité (WordSim353, SimLex999) ou d'analogies (Google analogies, BATS).
> - **Extrinsèque** : tester sur des tâches downstream (NER, classification, traduction) et voir si les embeddings améliorent les perfs.
