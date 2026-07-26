---
title: Machine Translation
---
# Machine Translation

So far in this class, we've dealt with problems of predicting a single output: an NER label for a word, the single most likely next word in a sentence given the past few, and so on. However, there's a whole class of NLP tasks that rely on sequential output, or outputs that are sequences of potentially varying length. For example:

- **Translation:** taking a sentence in one language as input and outputting the same sentence in another language.
- **Conversation:** taking a statement or question as input and responding to it.
- **Summarization:** taking a large body of text as input and outputting a summary of it.

More specifically we will focus on the Machine Translation task. That can be defined as a task of translating a sentence $x$ from one language (the **source language**) to a sentence $y$ in another language (the **target language**).

$$
\begin{aligned}
&x: \text{L'homme est né libre, et partout il est dans les fers} \\
&y: \text{Man is born free, but everywhere he is in chains}
\end{aligned}
$$

## 1950s-1980: Early Machine Translation

There was a lot of work translating Russian to English during the Cold War. Those systems were mostly rule-based: they would look up the Russian words with their English counterparts.

## 1980s-1990: Example-based MT

*(à compléter)*

## 1990s-2010s: Statistical Machine Translation (SMT)

We want to find the best English sentence $y$, given French sentence $x$. Thus by Bayes' rule:

$$\underset{y}{\arg\max}\ P(y \mid x) = \underset{y}{\arg\max}\ \textcolor{blue}{P(x \mid y)}\ \textcolor{green}{P(y)}$$

- **Translation model** (in blue): Models how words and phrases should be translated (fidelity). Learnt from parallel data.
- **Language model** (in green): Models how to write good English (fluency). Learnt from monolingual data.

**Alignment.**

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im1 (4).png]]
*Upper left to lower right: some words have no counterpart; alignment can be many-to-one, one-to-many, many-to-many, and so on.*

## 2010-: Neural Machine Translation (NMT)

### a. Seq2seq : le modèle de base

#### Motivation : un modèle de langage conditionnel

Seq2seq représente une **révolution conceptuelle** : c'est un **modèle de langage conditionnel** qui génère une séquence de sortie conditionnée par une séquence d'entrée.

**Neural Machine Translation (NMT)** : la traduction automatique neuronale utilise un seul réseau neuronal pour effectuer la traduction. L'architecture Seq2Seq ([Sutskever et al. NIPS 2014](https://papers.nips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html)) se divise en **deux blocs** :

- **Encoder** : prend la séquence d'entrée et l'encode dans un **vecteur de contexte** de taille fixe. Cette représentation résume le sens de toute la séquence.
- **Decoder** : utilise ce vecteur de contexte comme "graine" pour générer la séquence de sortie.

Pour cette raison, les modèles Seq2seq sont souvent appelés **"modèles encoder-decoder"**.

#### Architecture mathématique

Nous avons deux séquences de mots avec leurs variables aléatoires correspondantes :

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im2 (4).png]]

#### Seq2seq comme modèle de langage conditionnel

**Rappel** : un modèle de langage classique estime la probabilité d'une phrase.

![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im7 (2).png]]

**Seq2seq** : la traduction automatique fonctionne différemment.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im8 (2).png]]

> [!info]- 🎯 L'insight révolutionnaire
> **Réseau vert** : **Encoder** qui transforme la phrase d'entrée en représentation latente.
>
> **Réseau violet** : **Decoder** très similaire au modèle de langage classique.
>
> **Différence clé** : au lieu de commencer avec un vecteur zéro, le decoder utilise l'encodage de la phrase d'entrée. C'est pourquoi on parle de **modèle de langage conditionnel**.
>
> **Exemple** : modéliser une phrase anglaise conditionnée par une phrase française : $P(\text{anglais}|\text{français})$.

### b. Le décodage : générer la traduction

D'une phrase source en français $x$, nous voulons trouver la meilleure traduction possible $y$ :

$$P(y^1, \ldots, y^{T_y} \mid x)$$

Plutôt que d'échantillonner aléatoirement, nous résolvons un problème d'optimisation :

$$\underset{y^1, \ldots, y^{T_y}}{\arg\max}\ P(y^1, \ldots, y^{T_y} \mid x)$$

**(1) Greedy decoding.** À chaque pas de temps, on choisit le mot le plus probable :

$$x_t = \operatorname{argmax}_{\tilde{x}_t} P(\tilde{x}_t \mid x_1, \ldots, x_{t-1})$$

**Problème** : pas de retour en arrière possible, les erreurs se propagent en cascade. Exemple : *il m'a entartré* → *he* → *he hit* → *he hit* ~~*a*~~ *...* — une fois l'erreur commise, impossible de revenir en arrière.

Cette technique est efficace et naturelle, mais elle n'explore qu'une petite partie de l'espace de recherche : si une erreur est commise à un pas de temps, le reste de la phrase peut être fortement impacté.

**(2) Exhaustive search.** Trouver la traduction qui maximise :

$$P(y \mid x) = P(y_1 \mid x)\, P(y_2 \mid y_1, x)\, P(y_3 \mid y_1, y_2, x) \cdots P(y_T \mid y_1, \ldots, y_{T-1}, x) = \prod_{t=1}^{T} P(y_t \mid y_1, \ldots, y_{t-1}, x)$$

On pourrait essayer de calculer toutes les séquences $y$ possibles, mais cette complexité $O(|V|^T)$ est bien trop coûteuse — le décodage devient un problème NP-complet.

**(3) Beam search.** Compromis optimal entre précision et efficacité : on maintient $K$ candidats à chaque pas de temps (le *beam width*), au lieu d'un seul (greedy) ou de tous (exhaustive).

$$\mathcal{H}_{t} = \left\{ (x_1^1, \ldots, x_t^1), \ldots, (x_1^K, \ldots, x_t^K) \right\}$$

et on calcule $\mathcal{H}_{t+1}$ en développant $\mathcal{H}_t$ et en gardant les $K$ meilleurs candidats parmi :

$$\tilde{\mathcal{H}}_{t+1} = \bigcup_{k=1}^{K} \mathcal{H}_{t+1}^{k}, \qquad \mathcal{H}_{t+1}^{k} = \left\{ (x_1^k, \ldots, x_t^k, v_1), \ldots, (x_1^k, \ldots, x_t^k, v_{|V|}) \right\}$$

Quand $K$ augmente, on gagne en précision et on devient asymptotiquement exact, mais l'amélioration n'est pas monotone — on choisit un $K$ qui combine performance raisonnable et efficacité computationnelle. C'est pour cela que le beam search est la technique la plus utilisée en NMT.

**Exemple ($B=3$).** On garde 3 mots à chaque étape. (i) On calcule d'abord $p(y^{(1)}|x)$, on obtient trois valeurs : "in", "jane" et "september".

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im4 (1).png]]

(ii) On donne ensuite le mot "in" en entrée, on obtient trois nouvelles prédictions et on évalue :

$$p(y^1=\text{"in"}, y^2 \mid x) = p(y^1=\text{"in"} \mid x)\, p(y^2 \mid x, y^1=\text{"in"})$$

On fait de même pour les deux autres mots — cela donne 9 prédictions au total, dont on garde les 3 meilleures.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im5.png]]

On répète jusqu'à compléter la phrase entière.

### c. Le softmax et la température

Sur chaque pas de temps $t$, le modèle de langage calcule une distribution de probabilité $P_t$ en appliquant la fonction softmax à un vecteur de scores $S \in \mathbb{R}^{|V|}$ :

$$P_t(w) = \frac{\exp(s_w)}{\sum_{w' \in V} \exp(s_{w'})}$$

**Softmax avec température.** On peut appliquer un hyperparamètre de **température** $\tau$ au softmax :

$$P_t(w) = \frac{\exp(s_w / \tau)}{\sum_{w' \in V} \exp(s_{w'} / \tau)}$$

- **Augmenter $\tau$** : distribution plus uniforme → sortie plus diverse (la probabilité se répartit sur tout le vocabulaire).
- **Diminuer $\tau$** : distribution plus concentrée ("spiky") → sortie moins diverse (la probabilité se concentre sur les mots les plus probables).

> [!tip] Astuce mnémotechnique
> Imaginez un bloc de glace : à haute température il fond et se répand (distribution uniforme) ; à très basse température il devient dense et compact (distribution concentrée).

### d. Alternatives pour l'apprentissage

**Teacher forcing.** On peut donner directement les vrais résultats en entrée à chaque étape (plutôt que la propre prédiction du modèle), ce qui stabilise et accélère l'apprentissage.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im6 (2).png]]

## Attention : la solution au goulot d'étranglement

### Le problème du goulot d'étranglement

Dans l'architecture Seq2seq standard, **toute l'information** de la phrase source doit être compressée dans le **dernier état caché de l'encoder**.

**Problème** : goulot d'étranglement informationnel — toute la décision repose sur un seul vecteur !

![[images/3-Apprentissage automatique/06_Natural language processing/reseaux_sequentielles/attention1.png]]

> [!warning]- 🚫 Limitations critiques du Seq2seq classique
> **Perte d'information** : les premiers mots de la séquence source "s'effacent" progressivement.
>
> **Séquences longues** : performance dégradée car un seul vecteur ne peut pas retenir tous les détails.
>
> **Pas d'alignement** : le decoder ne sait pas sur quelle partie de la source se concentrer.

### La solution : le mécanisme d'attention

**Insight clé** : au lieu de s'appuyer sur un seul vecteur de contexte, on permet au decoder d'avoir une **connexion directe** avec tous les états cachés de l'encoder, et d'**apprendre** sur quelle partie se concentrer à chaque étape.

![[images/3-Apprentissage automatique/06_Natural language processing/reseaux_sequentielles/attention2.png]]

**Algorithme d'attention (4 étapes) :**

1. **Calcul des scores d'attention** : $\mathbf{e}^t = [s_t^T a_1, s_t^T a_2, \ldots, s_t^T a_N] \in \mathbb{R}^N$
2. **Normalisation par softmax** : $\boldsymbol{\alpha}^t = \text{softmax}(\mathbf{e}^t) \in \mathbb{R}^N$
3. **Calcul du vecteur de contexte** : $\mathbf{c}_t = \sum_{i=1}^{N} \alpha_i^t \mathbf{a}_i \in \mathbb{R}^h$
4. **Intégration dans le decoder** : $[\mathbf{c}_t; \mathbf{s}_t] \in \mathbb{R}^{2h}$

> [!info]- 🎯 Intuition du mécanisme d'attention
> **Query** : état caché du decoder $s_t$ ("Que cherche-t-on ?")
>
> **Keys & Values** : états cachés de l'encoder $a_1, \ldots, a_N$ ("Où chercher ?")
>
> **Attention weights** : $\alpha^t$ ("Combien d'attention porter à chaque position ?")
>
> **Context vector** : $c_t$ ("Résumé pondéré de l'information pertinente")

### Avantages

Le decoder se concentre sur les parties **pertinentes** de la phrase source (résolution du goulot d'étranglement, accès direct à toute l'information de l'encoder). L'attention **atténue le vanishing gradient** en créant des raccourcis vers les états éloignés, et offre de l'**interprétabilité** : les poids d'attention $\alpha^t$ révèlent quels mots source influencent chaque prédiction.

![[images/3-Apprentissage automatique/06_Natural language processing/reseaux_sequentielles/attention3 1.png|298]]

> [!tip]- 🔍 Révélation de l'alignement automatique
> **Jamais explicitement entraîné** : le modèle apprend l'alignement comme effet de bord !
>
> **Visualisation** : les matrices d'attention révèlent les correspondances mot-à-mot.
>
> **Applications** : analyse linguistique, détection d'erreurs, compréhension du modèle.

### Variantes d'attention

Avec $\mathbf{a}_1, \ldots, \mathbf{a}_N \in \mathbb{R}^{d_1}$ (encoder) et $\mathbf{s} \in \mathbb{R}^{d_2}$ (decoder) :

- **Dot-product attention** : $e_i = \mathbf{s}^T \mathbf{a}_i \in \mathbb{R}$ (nécessite $d_1 = d_2$)
- **Multiplicative attention** : $e_i = \mathbf{s}^T \mathbf{W} \mathbf{a}_i \in \mathbb{R}$ où $\mathbf{W} \in \mathbb{R}^{d_2 \times d_1}$
- **Additive attention** : $e_i = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{a}_i + \mathbf{W}_2 \mathbf{s}) \in \mathbb{R}$ où $\mathbf{W}_1 \in \mathbb{R}^{d_3 \times d_1}$, $\mathbf{W}_2 \in \mathbb{R}^{d_3 \times d_2}$, $\mathbf{v} \in \mathbb{R}^{d_3}$

> [!success]- 🚀 Chronologie
> **2014** : Seq2seq introduit le paradigme encoder-decoder.
>
> **2014** : l'attention (Bahdanau et al.) résout le goulot d'étranglement.
>
> **2017** : "Attention is All You Need" → Transformers.
>
> **Aujourd'hui** : fondation de tous les LLMs modernes (GPT, BERT, etc.)

## c. Un algorithme plus avancé

Voir le Transformer, traité dans le fichier `06_Représentations contextuelles`.

## Hybrid Models (2010-) : SMT mixed with NMT

*(à compléter)*

## Machine Translation Evaluation

**Motivation.** Evaluating the quality of translations is a notoriously tricky and subjective task. In real life, if you give a paragraph of text to ten different translators, you will get back ten different translations. Translations are imperfect and noisy in practice — they attend to different information and emphasize different meanings. One translation can preserve metaphors and the integrity of long-ranging ideas, while another can achieve a more faithful reconstruction of syntax and style, attempting a word-to-word translation. This flexibility is not a burden; it is a testament to the complexity of language and to our ability to decode and interpret meaning.

Note that there is a difference between the objective loss function of your model and the evaluation methods discussed below. Since loss functions are in essence an evaluation of your model's prediction, it is easy to confuse the two ideas — the evaluation metrics ahead offer a final, summative assessment of your model against some measurement criterion, and no single measurement is superior to all others, though some have clear advantages and majority preference.

Evaluating the quality of machine translations has become its own research area, with many proposals like TER, METEOR, MaxSim, SEPIA, and RTE-MT. We will focus here on two baseline evaluation methods and on BLEU.

### Human Evaluation

The first and least surprising method is to have people manually evaluate the correctness, adequacy, and fluency of your system. Like the Turing Test, if you can fool a human into not being able to distinguish a human-made translation from your system's translation, your model passes the test. The obvious problem with this method is that it is costly and inefficient, though it remains the gold standard for machine translation.

### Evaluation Against Another Task

A common way of evaluating models that output a useful representation of data (a representation being a translation or a summary) is: if your predictions are useful for solving some challenging downstream task, then the model must be encoding relevant information in its predictions. For example, you might train your translation predictions on a question-answering task in the translated language — you use the outputs of your system as inputs to a model for that other task. If the second task performs as well on your predictions as it does on well-formed data in the translated language, it means your inputs carry the relevant information for the task.

The issue with this method is that the second task may not be affected by many of the finer points of translation. For example, if you measured translation quality on a query-retrieval task (like pulling up the right webpage for a search query), a translation that preserves the main topic words but ignores syntax and grammar might still fit the task well — without this meaning the translation itself is accurate or faithful. Determining the quality of the translation model is therefore shifted to determining the quality of the downstream task, which may or may not be a good standard.

### Bilingual Evaluation Understudy (BLEU)

BLEU was invented at IBM (Papineni et al., 2002) — "bilingual evaluation understudy" means a substitute for a human evaluating the output of an MT system. One challenge is that for one French sentence there could be several equivalent English sentences. For example, given the French sentence:

$$\text{French: Le chat est sur le tapis.}$$

and reference human-generated translations:

$$
\begin{aligned}
&\text{Reference 1: The cat is on the mat.} \\
&\text{Reference 2: There is a cat on the mat.}
\end{aligned}
$$

we get a high BLEU score if the sentence is very similar to both references.

**Unigram precision.** Consider an extreme example where the machine translation is:

$$\text{MT output: the the the the the the}$$

One way to measure how good the MT is: check if the word "the" appears in the references — this is "precision". Here there are 7 words in the output, "the" appears in both references, so we'd get $7/7$, which is clearly not a good score. Instead we use a *modified* precision: we give each word credit up to the maximum number of times it appears in a single reference. In references 1 and 2, "the" appears 2 and 1 times respectively, so "the" gets credit up to twice: $2/7$.

$$p_1 = \frac{Count_{clip}(\text{"the"})}{Count(\text{"the"})}$$

**Bigrams.** BLEU doesn't only look at unique words — it also looks at bigrams. Consider:

$$\text{MT output: The cat the cat on the mat}$$

The possible bigrams are "the cat", "cat the", "cat on", "on the", "the mat". We count how many of these bigrams appear, clipping each to the maximum number of times it appears in reference 1 or 2.

![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im9 (3).png|428]]

**Formalization.** We define the modified precision for unigrams ($n=1$) as the sum over all unigrams appearing in the MT output $\hat{y}$:

$$p_1 = \frac{\sum_{unigram \in \hat{y}} Count_{clip}(unigram)}{\sum_{unigram \in \hat{y}} Count(unigram)}$$

and more generally, the N-gram version:

$$p_n = \frac{\sum_{n\text{-gram} \in \hat{y}} Count_{clip}(n\text{-gram})}{\sum_{n\text{-gram} \in \hat{y}} Count(n\text{-gram})}$$

These modified precision scores measure the degree to which the MT overlaps with the references. If the MT output is exactly the same as a reference, we get $p_1 = 1.0$, $p_2 = 1.0$, etc.

Finally, combining the unigram-to-4-gram precisions $p_1, p_2, p_3, p_4$, we get the combined BLEU score:

$$\text{BLEU} = BP \cdot \exp\left(\frac{1}{4}\sum_{n=1}^{4} p_n\right)$$

where $BP$ is the **brevity penalty**:

$$BP = \begin{cases}
1 & \text{if MT\_output\_length} > \text{reference\_output\_length} \\
\exp\left(1 - \dfrac{\text{reference\_output\_length}}{\text{MT\_output\_length}}\right) & \text{otherwise}
\end{cases}$$

The BP parameter penalizes short translations, since a short translation can easily achieve a high precision score. The brevity penalty rewards translations whose length is closer to (or larger than) the reference length.
