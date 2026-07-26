---
title: Question Answering
---
# Question Answering (QA)

## Introduction

### Motivation

Question Answering is another success of Deep Learning. A typical example is a request on the Google search engine — for example "Who was Australia's third prime minister?" Basically we ask a question and we receive an answer. Note that Google uses a "Knowledge Graph" (formerly known as Freebase) that has some predefined answers, but this is not what is used here:

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im5.png]]

**There are usually two successive steps in question answering:**

1. Finding a document that might contain an answer → traditional IR / web search.
2. Finding an answer in a paragraph or a document → **Machine Reading Comprehension**.

**Main difference between a QA system and an IR system:** QA takes a specific query → answer; IR takes a general query → list of documents.

### Machine Reading Comprehension

**Reading Comprehension (RC)** — the ability to read text and then answer questions about it — is a challenging task for machines, requiring both an understanding of natural language and knowledge about the world.

**A brief history:**

1. **(1970s)** Reading comprehension is not a new problem — it goes back to the 70s, where early works attempted it. Wendy Lehnert, 1977, *"The Process of Question Answering"*:
   > "Only when we can ask a program to answer questions about what it reads will we be able to begin to assess that program's comprehension."
2. **(1999)** It was revived by Lynette Hirschman, who attempted to build NLP systems that could answer reading comprehension questions at the level of 3rd to 6th graders.
3. **(2013)** Revived again by Chris Burges with MCTest — answering questions over simple story texts. Burges was not an NLP person but a machine learning person; he proposed a challenge with the MCTest corpus (roughly 600 kids' stories), which unfortunately did not go far.
4. **(2015-2016)** With the rise of Deep Learning, interest returned and new datasets were created: first the CNN/DailyMail dataset by Hermann et al. (NIPS 2015, DeepMind), followed by SQuAD (Rajpurkar et al., EMNLP 2016), and subsequently MS MARCO, TriviaQA, RACE, NewsQA, NarrativeQA, etc.

The goal of **open-domain QA** is to answer a question from a large collection of documents. Some milestones since 1964:

1. **(1964)** Simmons et al. did the first exploration of answering questions from an expository text, based on matching dependency parses of a question and answer.
2. **(1993)** Murax (Kupiec) aimed to answer questions over an online encyclopedia using IR and shallow linguistic processing.
3. **(1999)** The NIST TREC QA track began, rigorously investigating answering fact-based questions over a large collection of documents for the first time.
4. **(2011)** IBM's Jeopardy system (DeepQA) brought attention to the problem, using an ensemble of many methods.
5. **(2016)** DrQA used IR followed by neural reading comprehension to bring deep learning to open-domain QA — one of the first neural systems, built by Chen, a PhD student at Stanford.

### Types of Questions

Based on the paper by Chandra et al., ["A Survey on Types of Question Answering System"](http://www.iosrjournals.org/iosr-jce/papers/Vol19-issue6/Version-4/D1906041923.pdf):

- **Factoid type questions** [What, Which, When, Who, How]. *Ex: What is the capital of Korea? → Answer: Seoul (Named Entity).*
- **List type questions** [list of facts or answers]. *Ex: Who are the members of DSBA? → Answer: Pilsung Kang, Junhong Kim, ... (list of Named Entities).*
- **Confirmation questions** [yes or no]. *Ex: Is it Monday today? → Answer: yes.*
- **Causal questions** [why or how]. *Ex: Why was he late? → Answer: because of the traffic jam (description about an entity).*
- **Hypothetical questions** [no specific answer]. Start with "what would happen if...". *Ex: What would happen if South Korea and North Korea unified? → Answer: ???*
- **Complex questions.** *Ex: What are the reasons for air pollution? → Answer: complex, requires looking across many documents.*

### SQuAD Dataset

[SQuAD](https://arxiv.org/abs/1606.05250), by Rajpurkar et al., June 2016.

We have a passage from Wikipedia and a question; the goal of the system is to come up with an answer to the question. By construction, in SQuAD the answer to a question is always a subsequence of words from the passage — you cannot have yes/no questions, or answers that require generating a brand new sentence. The first version created around 100k examples, with about five questions per passage and 20k passages of Wikipedia used. This is often referred to as **extractive question answering**.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im1 (4).png]]

There is variation in the answers: this was done via Mechanical Turk, collecting answers from three different people. For one example, all three annotators said "independent"; for another, one said "independent" and another said "independent school" — an algorithm's response can be correct if it matches any of the three.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im2 (3).png]]

**SQuAD v1.1 evaluation.** Since there are three gold answers per question, two evaluation metrics are used:
- **Exact match**: score 1 if your span matches any of the three gold answers, 0 otherwise; precision is the percentage of correct answers.
- **F1 score**: a softer overlap-based metric between predicted and gold spans.

On the leaderboard, humans were also evaluated (since humans are never perfect either), achieving an F1 score of 91.2. The original baseline built alongside the dataset was a logistic regression model.

**SQuAD 2.0.** SQuAD 2.0 adds examples where the correct answer is that there is no answer in the passage. Interestingly, even systems with high scores on v1.1 often don't understand language particularly well — for example, a system looking for a date because the question contains "when", and matching "destroy" to "kill", might combine these signals and confidently output a date like "1234" that has nothing to do with the actual answer.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im3 (4).png]]

**Limitations.** These systems still make elementary errors — fundamentally, they are still solving a matching problem rather than truly understanding language. The SQuAD dataset itself also has limitations: all answers must be a contiguous span from the passage.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im4 (1).png]]

## Question Answering Models

### Stanford Attentive Reader

This algorithm is based on the paper by [Chen et al., Aug. 2016 — *A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task*](https://arxiv.org/pdf/1606.02858.pdf). It is essentially the simplest QA system that works reasonably well — not the current state of the art, but a good answer to "what's the simplest thing that works decently?"

**1. Question Encoding** (1-layer bidirectional LSTM + GloVe 300d). The question-answering module starts with a question, e.g.:

$$\text{"Which team won Super Bowl 50?"}$$

We build a vector representation of this question. For each word we look up its GloVe embedding, then run an LSTM forward through the question and a second one backward through the question. We take the end states of both LSTMs (each of size $d$) and concatenate them into a "question vector" of size $2d$. We do not need to look at the middle states because the bi-LSTM implicitly propagates information to both extremes during training.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/stan1.png]]

**2. Passage Encoding** (1-layer bidirectional LSTM + GloVe 300d). We run a bi-LSTM forward and backward over the passage to learn its context, then concatenate the hidden states at each position to build $\tilde{p}_i$, the representation of each passage word.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/stan2.png]]

**3. Attention**, with query vector $q$ and paragraph vectors $\tilde{p}_i$. To locate the answer in the passage, we use the question representation to attend over the passage. This is a different application of attention than usual: a single question vector is matched against the passage to locate the answer.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/stan3.png]]

We need to define attention weights for both the **start token** (where the answer begins) and the **end token** (where it ends), via two learned bilinear combinations:

$$\alpha_i = \text{softmax}(q^T W_S \tilde{p}_i) \qquad \alpha_i' = \text{softmax}(q^T W_E \tilde{p}_i)$$

The softmax returns the attention weights as probabilities. We do not impose anything on how the model finds the start and end positions — the neural network learns this on its own. Both matrices $W_S$ and $W_E$ are learned by the network.

Finally, the cost function compares the predicted start and end tokens against the true answer during training.
