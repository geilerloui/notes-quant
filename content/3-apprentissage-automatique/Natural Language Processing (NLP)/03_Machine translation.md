
# 1. Machine Translation

So far in this class, we've dealt with problems of predicting a single output: an NER label for a word, the single most likely next word in a sentence given the past few, and so on. However, there's a whole class of NLP tasks that rely on sequential output, or outputs that are sequences of potentially varying length. For example,
\begin{itemize}
    \item \textbf{Translation:} taking a sentence in one language as input and outputting the same sentence in another language.
    \item \textbf{Conversation:} taking a statement or question as input and responding to it.
    \item \textbf{Summarization:} taking a large body of text as input and out-
putting a summary of it.
\end{itemize}


More specifically we will focus on the Machine Translation task. That can be defined as a task of translating a sentence $x$ from one language (the \textbf{source language}) to a sentence $y$ in another language (the \textbf{target language}).

$$
\begin{aligned}
&x: \text{L'homme est né libre, et partout il est dans les fers} \\
&y: \text{Man is born free, but everywhere he is in chains}
\end{aligned}
$$



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{1950s-1980: Early Machine Translation}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

There was a lot of work translation russian to english during the cold war. Those systems were mostly rule-based, they are looking up the russian words with its english counterparts. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{1980s-1990: Example-based MT}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{1990s-2010s: Statistical Machine Translation (SMT)}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

We want to find the best English sentence $y$, given French sentence $x$. Thus by Bayes Rules

\begin{equation}
    \underset{y}{argmax~} P(y \mid x) = \underset{y}{argmax~} \textcolor{blue}{P(x \mid y)} \textcolor{officegreen}{P(y)}
\end{equation}
\begin{itemize}
    \item \textcolor{blue}{Translation model}: Models how words and phrases should be translated (fidelity). Learnt from parallel data
    \item \textcolor{officegreen}{Language model}: Models how to write good english (fluency). Learnt from monolingual data. 
\end{itemize}

\textbf{Alignment:}



![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im1 (4).png]]


\begin{figure}[H]
    \centering
    \includegraphics[scale=0.8]{images/4.machine_translation/im1.png}
    \caption{Upper left to lower right: some words have no counterpart; alignment can be many-to-one; one-to-many; many-to-many and so on}
    \label{fig:my_label}
\end{figure}



# III - 2010 - Neural Machine Translation (NMT)

## a - Seq2seq : Le Modèle de Base

### Motivation : Un Modèle de Langage Conditionnel

Seq2seq représente une **révolution conceptuelle** : c'est un **modèle de langage conditionnel** qui génère une séquence de sortie conditionnée par une séquence d'entrée.

**Neural Machine Translation (NMT)** : La traduction automatique neuronale utilise un seul réseau neuronal pour effectuer la traduction. L'architecture Seq2Seq ([Sutskever et al. NIPS 2014](https://papers.nips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html)) se divise en **deux blocs** :

- **Encoder** : prend la séquence d'entrée et l'encode dans un **vecteur de contexte** de taille fixe. Cette représentation résume le sens de toute la séquence
- **Decoder** : utilise ce vecteur de contexte comme "graine" pour générer la séquence de sortie

Pour cette raison, les modèles Seq2seq sont souvent appelés **"modèles encoder-decoder"**.

### Architecture mathématique

Nous avons deux séquences de mots avec leurs variables aléatoires correspondantes :


![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im2 (4).png]]



### Seq2seq comme Modèle de Langage Conditionnel

**Rappel** : Un modèle de langage classique estime la probabilité d'une phrase :


![[images/3-Apprentissage automatique/04_Computer vision/00_CNN architecture/im7 (2).png]]





**Seq2seq** : La traduction automatique fonctionne différemment :

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im8 (2).png]]




> [!info]- 🎯 L'insight révolutionnaire
> **Réseau vert** : **Encoder** qui transforme la phrase d'entrée en représentation latente
> 
> **Réseau violet** : **Decoder** très similaire au modèle de langage classique
> 
> **Différence clé** : Au lieu de commencer avec un vecteur zéro, le decoder utilise l'encodage de la phrase d'entrée. C'est pourquoi on parle de **Modèle de Langage Conditionnel**.
> 
> **Exemple** : Modéliser une phrase anglaise conditionnée par une phrase française : $P(\text{anglais}|\text{français})$

### alternatives of the de Décodage

**Motivation du problème** : D'une phrase source en français $x$, nous voulons trouver la meilleure traduction possible $y$ :

$$P(y^1, \ldots, y^{T_y} | x)$$

Plutôt que d'échantillonner aléatoirement, nous résolvons un problème d'optimisation :

$$\underset{y^1, \ldots, y^{T_y}}{\arg \max} \, P(y^1, \ldots, y^{T_y} | x)$$

**Types d'algorithmes de décodage** :

**(1) Greedy Decoding** : À chaque pas de temps, choisir le mot le plus probable $x_t = \underset{\tilde{x}_t}{\arg \max} \, P(\tilde{x}_t | x_1, \ldots, x_{t-1})$. **Problème** : Pas de retour en arrière possible, erreurs en cascade.

\textbf{(i). Greedy Decoding (search):} we saw how to generate or ("decode") the target sentence by taking argmax on each step of the decoder. This is the greedy decoding (take the most probable word on each step).The problem with this method is that we cannot backtrack. For example
\begin{itemize}
    \item Input: il m'a entartré (he hit me with a pie)
    \item he ...
    \item he hit ...
    \item he hit \textcolor{red}{a} ... ; There is an error ! we cannot backtrack
\end{itemize}
At each time step, we pick the most probable token. In other words
$$
x_{t}=\operatorname{argmax}_{\tilde{x}_{t}} \mathbb{P}\left(\tilde{x}_{t} | x_{1}, \ldots, x_{n}\right)
$$
This technique is efficient and natural, however it explores a small part of the search space and if we make a mistake at one time step, the rest of the sentence could be heavily impacted.\\



**(2) Exhaustive Search** : Trouver une traduction qui maximise $P(y|x) = \prod_{t=1}^{T} P(y_t | y_1, \ldots, y_{t-1}, x)$. **Problème** : Complexité $O(|V|^T)$ intenable.

Ideally we want to find a (length $T$ translation $y$ that maximizes:

$$
\begin{aligned}
P(y | x) &=P\left(y_{1} | x\right) P\left(y_{2} | y_{1}, x\right) P\left(y_{3} | y_{1}, y_{2}, x\right) \ldots, P\left(y_{T} | y_{1}, \ldots, y_{T-1}, x\right) \\
&=\prod_{t=1}^{T} P\left(y_{t} | y_{1}, \ldots, y_{t-1}, x\right)
\end{aligned}
$$

We could try computing all possible sequences $y$. This $O(V^T)$ complexity is far too expensive.\\

This is the simplest idea. We compute the probability of every possible sequence, and we chose the sequence with the highest probability. However, this technique does not scale at all to large outputs as the search space is exponential in the size of the input. Decoding in this case in NP-complete problem.\\


**(3) Beam Search** : Maintenir $K$ candidats à chaque pas de temps. **Compromis optimal** entre précision et efficacité.

### Exemple de Beam Search

**Beam width $B = 3$** (garder 3 mots à chaque étape) :

the idea is to maintain $K$ candidates at each time step.
$$
\mathcal{H}_{t}=\left\{\left(x_{1}^{1}, \ldots, x_{t}^{1}\right), \ldots,\left(x_{1}^{K}, \ldots, x_{t}^{K}\right)\right\}
$$
and compute $\mathcal{H}_{t+1}$ by expanding $\mathcal{H}_{t}$ and keeping the best $K$ candidates. In other words, we pick the best $K$ sequence in the following set
$$
\tilde{\mathcal{H}}_{t+1}=\bigcup_{k=1}^{K} \mathcal{H}_{t+1}^{\tilde{k}}
$$
where
$$
\mathcal{H}_{t+1}^{\tilde{K}}=\left\{\left(x_{1}^{k}, \ldots, x_{t}^{k}, v_{1}\right), \ldots,\left(x_{1}^{k}, \ldots, x_{t}^{k}, v_{|V|}\right)\right\}
$$
As we increase $K,$ we gain precision and we are asymptotically exact. However, the improvement is not monotonic and we can set a $K$ that combines reasonable performance and computational efficiency. For this reason, beam search is the most commonly used technique in NMT.\\

\textit{Example:} We will set the Beam Width parameter $B=3$ that is how many words we will retain for each prediction. (i) We start by computing $p(y^{(1)}|x)$ we get three values "in", "jane" and "september".

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im4 (1).png]]

(ii) Next, we put the output "in" to the input we get three additional top three predictions and we evaluate 
$$
p(y^{1}="in" , y^{2} | x) = p(y^1="in" |x) p(y^2 | x, y^1="in")
$$
We do the same for the two other words. We evaluate all the 9 predictions and we keep the top three.

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im5.png]]

We repeat until we have completed the full sentence.


### Alternatives Softmax

**Softmax avec température** $\tau$ :

$$P_t(w) = \frac{\exp(s_w / \tau)}{\sum_{w' \in V} \exp(s_{w'} / \tau)}$$

**Augmenter $\tau$** : Distribution plus uniforme → sortie diverse  
**Diminuer $\tau$** : Distribution concentrée → sortie focalisée



\textbf{(i) Softmax:} On timestep $t$, the LM computes a probability distribution $P_t$ by applying the softmax function to a vector of scores $S \in \mathbb{R}^{|V|}$
$$
P_{t}(w)=\frac{\exp \left(s_{w}\right)}{\sum_{w^{\prime} \in V} \exp \left(s_{w^{\prime}}\right)}
$$


\textbf{(ii) Softmax Temperature:} You can apply a \textcolor{blue}{temperature hyperparameter} $\tau$ to the softmax
$$
P_{t}(w)=\frac{\exp \left(s_{w} / \tau\right)}{\sum_{w^{\prime} \in V} \exp \left(s_{w^{\prime}} / \tau\right)}
$$

We can toy with the $\tau$ parameter leading to two different behavior:
\begin{itemize}
    \item \textcolor{blue}{Raise the temperature $\tau$:} $P_t$ becomes more \textcolor{purple}{uniform}. Thus more diverse output (probability is spread around vocab)
    \item \textcolor{blue}{Lower the temperature $\tau$:} $P_t$ becomes more \textcolor{purple}{spiky}. Thus less diverse output (probability is concentrated on top words)
\end{itemize}

Side note: to remember those relations imagine a block of ice that when put into high temperature melts so spread; and when put at very low temperature increases in density.\\

### Alternatives for learning

\textbf{(i) Teacher Forcing:} We can plug directly the true results at every input 

![[images/3-Apprentissage automatique/06_Natural language processing/machine translation/im6 (2).png]]


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


## c - algo plus avancé

on peut voir le transformer qui est dans le fichier 06_Représentations contextuelles 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# subsection{2010-:Hybrid Models: SMT mixed with NMT}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{Machine Translation Evaluation}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\textbf{Motivation:} Evaluating the
quality of translations is a notoriously tricky and subjective task. In
real-life, if you give a paragraph of text to ten different translators,
you will get back ten different translations. Translations are imperfect
and noisy in practice. They attend to different information and emphasize different meanings. One translation can preserve metaphors
and the integrity of long-ranging ideas, while the other can achieve a
more faithful reconstruction of syntax and style, attempting a wordto-word translation. Note that this flexibility is not a burden; it is a
testament to the complexity of language and our abilities to decode
and interpret meaning, and is a wonderful aspect of our communicative faculty.\\

At this point, you should note that there is a difference between
the objective loss function of your model and the evaluation methods
we are going to discuss. Since loss functions are in essence an evaluation of your model prediction, it can be easy to confuse the two ideas.
The evaluation metrics ahead offer a final, summative assessment of
your model against some measurement criterion, and no one measurement is superior to all others, though some have clear advantages
and majority preference.\\


Evaluating the quality of machine learning translations has become it own entire research area, with many proposals like TER,
METEOR, MaxSim, SEPIA, and RTE-MT. We will focus in these notes
on two baseline evaluation methods and BLEU.

\subsubsection{Human Evaluation}

The first and maybe least surprising method is to have people manually evaluate the correctness, adequacy, and fluency of your system.
Like the Turing Test, if you can fool a human into not being able to
distinguish a human-made translation with your system translation,
your model passes the test for looking like a real-life sentence! The
obvious problem with this method is that it is costly and inefficient,
though it remains the gold standard for machine translation

\subsubsection{Evaluation against another task}

A common way of evaluating machine learning models that output
a useful representation of some data (a representation being a translation or summary) is that if your predictions are useful for solving
some challenging task, then the model must be encoding relevant
information in your predictions. For example, you might think of training your translation predictions on a question-answering task in
the translated language. That is, you use the outputs of your system
as inputs to a model for some other task (the question-answering).
If your second task can perform as well on your predictions as it
can on well-formed data in the translated language, it means that
your inputs have the relevant information or patterns for meeting the
demands of the task.\\

The issue with this method is that the second task may not be affected by many of the finer points of translation. For example, if you
measured the quality of translation on a query-retrieval task (like
pulling up the right webpage for a search query), you would find
that a translation that preserves the main topic words of the documents but ignores syntax and grammar might still fit the task well.
But this itself doesn’t mean that the quality of your translations is
accurate or faithful. Therefore, determining the quality of the translation model is just shifted to determining the quality of the task itself,
which may or may not be a good standard.\\

\newpage
\subsubsection{Bilingual Evaluation Understudy (BLEU)}

Bleu was invented by IBM bilingual evaluation understudy, it means a substitute for human to evaluate output of a MT system. It was due to Papinenei et al 2002. One of the challenges is that for one french sentence there could be several equivalent english sentences. For example let's say we have a french sentence
$$
\text{French: Le chat est sur le tapis.}
$$
And we are given a references human generated translation:
$$
\begin{aligned}
&\text{Reference 1: The cat is on the mat.} \\
&\text{Reference 2: There is a cat on the mat.} 
\end{aligned}
$$
We get a high blue score if the sentence is very similar to both References.\\

\textbf{Unigram:} Let's look at an extreme example where the machine translation is:
$$
\text{MT output: the the the the the the}
$$
one way to measure how good the MT is, we look at if the word "the" appears in the references this is called "Precision" here there are 7 words in the output and "the" appears in both references we get $7/7$ which is not good. Instead we use a modified precision measure, we give each word credit up until the maximum time it appears in the reference. In reference 1 and 2, "the" appears 2 and 1. We say that "the" gets credit up to twice. $2/7$ is our score now. $(count_{clip}("the") /count("the")$.\\

\textbf{Bigrams:} In the blue score you don't want to look only at unique words, we will look also at bigrams:
$$
\text{MT output: The cat the cat on the mat}
$$
Here the possible bigrams are "the cat", "cat the", "cat on", "on the", "the mat" for the MT output. Let's count up how many of this bi-grams appears. the clip is the maximum type it appears in reference 1 or 2. 


![[images/3-Apprentissage automatique/04_Computer vision/01_CNN/im9 (3).png|428]]



\textbf{Formalize:} We define the modified precision as where $1$ means unigram, we sum over all the unigrams that appears in the MT output $\hat{y}$
$$
p_1 = \frac{\sum_{unigram \in \hat{y}} Count_{clip}(unigram)}{ \sum_{unigram \in \hat{y}} Count(unigram)  }
$$
We can also define the N-Gram version:
$$
p_n = \frac{\sum_{n-grams \in \hat{y}} Count_{clip}(n-gram)}{ \sum_{n-grams \in \hat{y}} Count(n-grams)  }
$$
Those modified precision scores it allows us to measure the degree how the MT is overlapping with the references. If the MT output is exactly the same as reference 1 or ferecen 1 , we would get a $p_1=1.0$ or $p_2=1.0$\\

Finally we can get the combine blue score, if we have unigram up until four-grams $p_1, p_2, p_3, p_4$ we get:
$$
\text{Combined bleu score = } BP~exp( \frac{1}{4} \sum_{n=1}^{4} p_n)
$$
Where $BP$ is the brevity penalty
$$
BP = \begin{cases}
1\quad \text{if MT\_output\_length } > \text{reference\_output\_length} \\
exp(1- \text{MT\_output\_length/reference\_output\_length) \quad otherwise}
\end{cases}
$$
Basically the BP parameter is a penalty on short translation because short translation can easily get a good result. The BP will yield a better for translation that are larger than the references.
