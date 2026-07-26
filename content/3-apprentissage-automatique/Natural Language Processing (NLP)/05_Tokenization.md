---
title: Tokenization
---
# Tokenization

## Words Tokenization

**Definition (Word tokenization).** A simple way of tokenizing a text is to split it by spaces, which gives:

$$\text{["Don't", "you", "love", "Transformers?", "We", "sure", "do."]}$$

We can also take punctuation into account:

$$\text{["Don", "'", "t", "you", "love", "Transformers", "?", "We", "sure", "do", "."]}$$

The disadvantage here is how the tokenization deals with the word "Don't" — it stands for "do not", so it would be better tokenized as ["Do", "n't"].

**Definition (Rule-based Tokenization).** spaCy and Moses are two popular rule-based tokenizers:

$$\text{["Do", "n't", "you", "love", "Transformers", "?", "We", "sure", "do", "."]}$$

Here space, punctuation, and rule-based tokenization are mixed.

**Downsides.**
- **Very large vocabulary size** — typically the whole set of words, punctuation, etc.
- **Large number of OOV tokens (Out-Of-Vocabulary).** One solution to reduce the vocabulary is to limit ourselves to the top few hundred words; the rest are termed OOV.
- **Different meaning of very similar words** — for example "dog" and "dogs" would potentially be treated as unrelated tokens.

## Character Tokenization

**Definition (Character Tokenization).** We simply use a character as a token, which reduces the vocabulary size to the size of the alphabet.

**Downsides.**
- Very long sequences.
- Less meaningful individual tokens.

## Subwords Tokenization

Some of the popular subword tokenization algorithms are WordPiece, Byte-Pair Encoding (BPE), Unigram, and SentencePiece. A few of these models use space tokenization as the pre-tokenization method, while others use more advanced pre-tokenization methods provided by Moses, spaCy, or ftfy.

### BPE Encoding

**Definition (Byte-Pair Encoding).** BPE is a simple form of data compression algorithm in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur in the data.

**Example (BPE).** Suppose we have the data $\text{aaabdaaabac}$:

1. The byte pair $aa$ occurs most often, so we replace it with $Z$ (which does not occur in our data). We now have $ZabdZabac$, where $Z=aa$.
2. The next most common byte pair is $ab$, so we replace it with $Y$. We now have $ZYdZYac$, where $Z=aa$ and $Y=ab$.
3. We repeat the process until our data has transformed into $XdXac$, where $X=ZY$, $Y=ab$, and $Z=aa$.

**Definition (BPE for NLP).** BPE ensures that the most common words are represented in the vocabulary as a single token, while rare words are broken down into two or more subword tokens.

**Example (BPE for NLP).** Suppose we have a corpus with the words:

$$Corpus = \{\text{old, older, finest, lowest}\}$$

We count the occurrence of those words in the corpus, adding an end-of-word token $</w>$:

$$\{\text{"old}</w>\text{": 7, "older}</w>\text{": 3, "finest}</w>\text{": 9, "lowest}</w>\text{": 4}\}$$

**(a) First and second iterations.** We start by writing every word as individual characters in a table with their relative frequency. We then merge the two most common characters — here "es" — which sets the frequency of the isolated "s" to zero.

![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-1.png]]

**(b) Before-last and last iterations.** We repeat the process until we end up with merged units like "est" and "old" — able to compose words like "slighest", "oldest", etc.

![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-2.png]]

**(c) Encoding.** To encode a new sequence, we split the words according to our learned frequency/merge table:

$$["the</w>", "high", "est</w>", "range</w>", "in</w>", "Seattle</w>"]$$

Note that the role of the end-of-word delimiter $</w>$ is key — otherwise we could have gotten an ambiguous merge like:

$$["the", "high", "estrange", "in", "Seattle"]$$

**(d) Decoding.** We simply concatenate all the subwords:

$$["the", "highest", "range", "in", "Seattle"]$$

**Remark.** Some models use a special symbol to indicate which part of a word is the start of the token and which part is the continuation. For example, "tokenization" can be split into "token" and "##ization", where "##" indicates that "ization" is a continuation of the previous subword rather than the start of a new word.

### WordPiece

**(i) When BPE fails.** There can be instances where there is more than one way to encode a particular word. It then becomes difficult for the algorithm to choose which subword tokens to prioritize. As a result, the same input can be represented by different encodings, which impacts the accuracy of the learned representations.

**(ii) Example.** Suppose this is the vocabulary for a small corpus and we want to tokenize the input phrase "linear algebra". We could tokenize it as:

$$\text{linear = li + near} \quad \text{or} \quad \text{li + n + ea + r}$$

$$\text{algebra = al + ge + bra} \quad \text{or} \quad \text{al + g + e + bra}$$

There are two different ways to tokenize each word, giving a total of four ways to tokenize the phrase — the same input text can be encoded in four different ways, which is a problem.

![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-3 (1).png|443]]

**(iii) Definition (WordPiece).** The only difference between BPE and WordPiece is *how* symbol pairs are chosen to be added to the vocabulary. At each iterative step, instead of merging the most *frequent* pair (BPE), WordPiece chooses the symbol pair that results in the largest increase in **likelihood** of a learned language model over the vocabulary once merged.

**(iv) Example.** The algorithm checks whether the probability of occurrence of "es" is higher than the probability of "e" followed by "s" occurring independently. The merge happens only if the probability of "es" divided by the probabilities of "e" and "s" is greater than for any other candidate symbol pair.
