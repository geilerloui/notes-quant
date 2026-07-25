
\section{Question Answering (QA)}
\subsection{Introduction}
\subsubsection{Motivation: Question Answering}

Question Answering is another success of Deep Learning. A typical example is a request on the Google search engine, for example "Who was Australia's third prime minister ?" Basically we ask a question and we receive an answer. Note that Google uses a "Knowledge Graph" (Formerly known as Freebase) that have some predefined answers but this is not what it uses here:


![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im5.png]]



\textbf{There are usually two successive steps in question answering:}
\begin{enumerate}
    \item Finding a document that might contain an answer $\rightarrow$ Traditional IR / Web search
    \item Finding an answer in a paragraph or a document $\rightarrow$ \underline{Machine Reading Comprehension}
\end{enumerate}

\textbf{Main difference between QA system and IR system:} QA : Query (specific) $\rightarrow$ Answer ; IR : Query (general) $\rightarrow$ Document list.\\

\subsubsection{Machine Reading Comprehension}

introdu followed by types of questions follows by dataset vs systems\\

\textbf{Reading Comprehension (RC)}, or the ability to read text and then answer question about it, is a challenging task for machines, requiring both understanding of natural language and knowledge about the world. 

\begin{enumerate}[label=(\roman*)]
\item (1970) Reading comprehension is a not a new problem it goes back to the 70's where the early works attempted Reading Comprehension. Wendy Lehnert 1977. "The Process of Question Answering"
\begin{center}
    \textit{"Only when we can ask a program to answer quetsions about what it reads will be able to begin to access that program's comprehension."}
\end{center}
\item (1990) It was revived in 1999 by Lynette Hirschman who attempted to build NLP systems that could answer human reading comprehension for 3rd to 6th graders.
\item (2013) Then revived again in 2013 by Chris Burges with MCTest by answering quetsions over simple story texts. Burges was not a NLP person but a Machine learning person. He proposed a challenge with the MCTest corpus as a simple reading challenge, they collected roughly 600 kid stories unfortunately it did not go far.
\item (2015) Finally in 2015-2016 with the rise of Deep Learning people got interested and new datasets were created. The first one by Herman et al (NIPS 2015) of DeepMind the CNN/DM dataset followed by Rajpurkar et al (EMNLP 2016) SQuAD and subsequently MS MARCO, TriviaQA, RACE, NewsQA, NarrativeQA ...
\end{enumerate}



 The goal of \textbf{open-domain QA} is to answer a question from a large collection of documents. We present some milestones since 1964 of the field:
\begin{enumerate}[label=(\roman*)]
\item (1964) Simmons et al did the first exploration of answering questions from an expository text based on matching dependency parses of a question and answer
\item (1993) Murax (Kupiec) aimed to answer questions over an online encyclopedia using IR and shallow linguistic processing.
\item (1999) The NIST TREC QA track begun in 1999 first rigorously investigated answering fact questions over a large collection of documents.
\item (2011) IBM's Jeopardy System (DeepQA) brought attention to a version of the problem; it used an ensemble of many methods.
\item (2016) DrQA uses IR followed by neural reading comprehension to bring deep learning to Open-domain QA. It was one the first neural system brough by Chen a PhD student at Stanford.
\end{enumerate}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsubsection{Types of questions}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\textbf{Types of questions:} based on the paper of Chandra et al \href{http://www.iosrjournals.org/iosr-jce/papers/Vol19-issue6/Version-4/D1906041923.pdf}{A Survey on Types of Question Answering System}
\begin{enumerate}
    \item \textbf{Factoid type questions} [What, Which, When, Who, How]
    \begin{itemize}
        \item Ex. What is the capital of Korea ? $\rightarrow$ Answer : Seoul [Named Entity]
    \end{itemize}
    \item \textbf{List type questions} [list of facts or answers]
    \begin{itemize}
        \item Who are the members of DSBA ? $\rightarrow$ Answer : Pilsung Kang, Junhong Kim ... [list of Named Entities]
    \end{itemize}
    \item \textbf{Confirmation questions} [yes or no]
    \begin{itemize}
        \item is it Monday today ? $\rightarrow$ Answer : yes
    \end{itemize}
    \item \textbf{Causal Questions} [why or how]
    \begin{itemize}
        \item Ex. Why was he late ? $\rightarrow$ Answer : Because of the traffic jam [description about an entity]
    \end{itemize}
    \item \textbf{Hypothetical Questions} [No specific answers] 
    \begin{itemize}
        \item It starts with "what would happen if". What would happen if South Korea and North Korea unified $\rightarrow$ Answer : ???
    \end{itemize}
    \item \textbf{Complex Questions}
    \begin{itemize}
        \item Ex. What are the reasons of Air Pollution ? $\rightarrow$ Answer : [complex we have to look in many documents]
    \end{itemize}
\end{enumerate}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\subsubsection{SQuAD Dataset}

\href{https://arxiv.org/abs/1606.05250}{SQuAD} by Rajpurkar et al. in June 2016.\\

We have a passage from wikipedia and a question, the goal of the system is to come up with an answer of this question. by construction for squad the answer to a question is always a sub sequence of words from the passage. You cannot have questions like yes/no, or a new sentence etc. In the first version they created  around 100k examples, there is like five questions per passage and 20k bits of wikipedia used. This is often referred as extractive question answering.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im1 (4).png]]


One more example: There is variation in the answer, what they did, it was did on mechanical Turk, they got answer from three different people. Here three human being one said independent one say independent school. For the second one they all said the same. Your algorithm can be correct if you response between the three

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im2 (3).png]]




Squad evaluation v1.1 as we said in this model there are three gold answers, then they suggest three evaluation metrics: (i) exact match : if your span is in those three you get one otherwise zero, and the precision is the percent of correct\\

second metric: f1 metric, \\

Leader-board: to the bottom they tested how human did at answering questions, because human are never perfect, they got a F1 score of 91.2; when they built the dataset they built a logistic regression baseline \\

\textbf{Squad 2.0:} 



Example: the answer is that there is no answer. but precisely what happens with systems is that even though those system have high score they do not understand language human quite well. Here it was looking for a date because of when, then looked at "destroy" that is similar to "kill" then .. it puts that together and finds 1234 which is not the response at all.

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im3 (4).png]]


Leaderboard/

\textit{Limitations:} they still make elementary errors it still doing a matching problem.

limitation: The squat dataset has some majors limitations: all answers are a span from the passage

![[images/3-Apprentissage automatique/06_Natural language processing/question answering/im4 (1).png]]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\subsection{Question Answering Models}
\subsubsection{Stanford Attentive Reader}

This algorithm is based on the paper from \href{https://arxiv.org/pdf/1606.02858.pdf}{Chen et al aug. 2016 - A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task}.  This is essentially the simplest QA system that works pretty well, it is not the current state of the art, but if you wonder what is the simplest thing that work decently, that's pretty it.\\
 
\textbf{1. Question Encoding}(1-layer-bidirectional-LSTMs + Glove300d) The question answering module starts off with a question 
$$
\text{"Which team won Super Bowl 50 ?"}
$$
We aim to build a representation of this question as a vector. For each word in the question we look up for their embeddings in particular the Glove embeddings. We then run an LSTM forward the question and a second backward through the question. Then we grab the end state of both LSTM of size $d$ that we concatenate into a vector, the "question vector" of size $2d$. We do not look in the middle of the answer because the bi-LSTM model will implicitly flow this information both extreme when trained.

![[stan1.png]]


\textbf{2. Passage Encoding}(1-layer-bidirectional-LSTM + Glove300d) We look at the passage we will run the bi-LSTM forward and backward to learn the context. Then we will concatenate each hidden layers to build the $\Tilde{p}_i$ the representation of each passage words.

![[stan2.png]]

\textbf{3. Attention} with query vector $q$ and a paragraph vectors $\Tilde{p}_i$.  Then we have to do a little more work to define the answer in the passage, we use the question representation to look up where the question is with attention. This is different type of application of attention, a one question vector that we want to match against to return the answer.

![[stan3.png]]


What we have to define is attention weights for both the start token (when the answer starts) and the end token (when the answer stops). We do that by two learned combinations:
$$
\alpha_i = softmax(q^T W_S \Tilde{p}_i)  \quad 
\alpha_i' = softmax(q^T W_E \Tilde{p}_i) 
$$
The softmax will return probabilities the attention weights. We could also wonder how they find the beginning and the end ? We don't impose anything we say the neural network has to learn it.

Both matrices $W_S$ and $W_E$ will be learnt by the network and they should.\\

Finally the cost function will compare the start token and end token with the real answer during learning.