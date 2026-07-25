
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X
\subsection{Words Tokenization}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X

\textbf{Definition (Word tokenization).} A simple way of tokenizing a text is to split it by spaces, which would give 

$$
\text{["Don't", "you", "love", "Transformers?", "We", "sure", "do."]}
$$

We can now also take into account the punctuations

$$
\text{["Don", "'", "t", "you", "love",  "Transformers", "?", "We", "sure", "do", "."]}
$$

it is disadvantageous, how the tokenization dealt with the word "Don't". "Don't" stands for "do not", so it would be better tokenized as ["Do", "n't"]. \\

\textbf{Definition (Rule-based Tokenization).} spaCy and Moses are two popular rule-based tokenizers

$$
\text{["Do", "n't", "you", "love", "Transformers", "?", "We", "sure", "do", "."]}
$$

Here space, punctuation and rule-based tokenization are mixed.\\

\textbf{Downsides.}
\begin{itemize}
    \item \textbf{Very large vocabulary size,} typically the whole set of words, punctuations ... 
    \item \textbf{Large number of OOV tokens (Out-of vocabulary).} One solution to reduce the vocabulary is to limit ourselves to the top 500 hundreds words and the test will be termed OOV.
    \item \textbf{Different meaning of very similar words,} for example dog and dogs would have potentially a different meaning
\end{itemize}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X
\newpage
\subsection{Character Tokenization}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X

\textbf{Definition (Character Tokenization).} We simply use a character as a token. Which reduces the vocabulary size to the size of the alphabet.\\


\textbf{Downsides.}
\begin{itemize}
    \item Very long sequences 
    \item Less meaningful individual tokens
\end{itemize}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X
\newpage
\subsection{Subwords Tokenization}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X

Some of the popular subword tokenization algorithms are WordPiece, Byte-Pair Encoding (BPE), Unigram, and SentencePiece. A few of these models use space tokenization as the pre-tokenization method while a few use more advanced pre-tokenization methods provided by Moses, spaCY, ftfy.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X
\subsubsection{BPE Encoding}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X

\textbf{Definition (Byte-Pair Encoding).} BPE is a simple form of data compression algorithm in which the most common pair of consecutives bytes of data is replaced with a byte that does not occur in the data.\\

\textbf{Example (BPE).} Suppose we have the data $\text{aaabdaaabac}$:\\
(a) The byte pair $aa$ occurs most often, so we will replace it with $Z$ as $Z$ does not occur in our data. So now we have $ZabdZabac$ where $Z=aa$.\\
(b) The next common byte pair is $ab$ so let’s replace it with $Y$. We now have $ZYdZYac$ where $Z = aa$ and $Y = ab$. \\
(c) We repeat the process until our data has now transformed into $XdXac$ where $X = ZY$, $Y = ab$, and $Z = aa$.\\

\textbf{Definition (BPE for NLP).} BPE ensures that the most common words are represented in the vocabulary as a single token while the rare words are broken down into two or more subword tokens \\

\textbf{Example (BPE for NLP).} Suppose we have a corpus with the words 

$$
Corpus = \{ old, older, highest, and lowest \}
$$

And we count the occurrence of those words in the corpus. Suppose the frequency of these words is, where we add an additional $</ \mathrm{w}>$ token, the end of word.

$$
\{\text { "old }</ \mathrm{w}>\text { ": } 7 \text {, "older }</ \mathrm{w}>\text { ": } 3 \text {, "finest }</ \mathrm{w}>\text { ": 9, "lowest }</ \mathrm{w}>": 4\}
$$

\textbf{(a) First and second iterations.} We write everything single characters in a table with their relative frequency. Next we merge the two most common characters here "es" which sets to zero "s".

![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-1.png]]


\textbf{(b) Before last and last iterations.} We repeat the previous process until we end up with "est" and "old", like "slighest", "oldest"...


![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-2.png]]

\textbf{(c) Encoding.} To encode our sequence, we split the words according to our frequency table

$$
[\text { "the }</ \mathrm{w}>\text { ", "high", "est }</ \mathrm{w}>\text { ", "range }</ \mathrm{w}>\text { ", "in }</ \mathrm{w}>\text { ", } \text { "Seattle }</ \mathrm{w}>\text { "] }
$$

Note that the role of the word delimiter "w" is key. Otherwise we could have gotten

$$
\text { ["the", "high", "estrange", "in", "Seattle"] }
$$

\textbf{(d) Decoding.} We just basically concatenate all the subwords

$$
\text { ["the", "highest", "range", "in", "Seattle"] }
$$

\textbf{Remark.} Some models use a special symbol to indicate which word is the start of the token and which word is the completion of the start of the token. For example, “tokenization” can be split into “token” and “\#\#ization” which indicates that “token” is the start of the word and “\#\#ization” is the completion of the word.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X
\newpage
\subsubsection{Wordpiece}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%X

\textbf{(i) When BPE fails.} It can have instances where there is more than one way to encode a particular word. It then gets difficult for the algorithm to choose subword tokens as there is no way to prioritize which one to use first. Hence, the same input can be represented by different encodings impacting the accuracy of the learned representations. \\

\textbf{(ii) Example.} Suppose this is the vocabulary for a small corpus and we want to tokenize our input phrase “linear algebra”. We can tokenize it as follows:

$$
\text{linear = li + near or li + n + ea + r}
$$

$$
\text{algebra = al + ge + bra or al + g + e + bra}
$$

We can see that there are two different ways to tokenize each word in the given phrase, giving a total of four ways to tokenize this phrase. So, the same input text can be encoded in four ways and this is indeed a problem.

![[images/3-Apprentissage automatique/06_Natural language processing/tokenization/im1-3 (1).png|443]]

\textbf{(i) Definition (Word-piece).} The only difference between the two models is the way in which symbols pairs are added to the vocabulary. At each iterative step, WordPiece chooses a symbol pair which will result in the largest increase in likelihood upon merging from a learnt language model from some vocabulary.\\

\textbf{(ii) Example.} For example, the algorithm will check if the probability of occurrence of “es” is more than the probability of occurrence of “e” followed by “s”. The merge will happen only if the probability of “es” divided by “e”, “s” is greater than any other symbol pair.
