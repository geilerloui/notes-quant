

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\subsection{Additional Architecture}
\subsubsection{Highway Network}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A highway network is very similar to a feed forward neural network. Recall that for a FFNN if we insert an input $y$ we get:
$$
z = g(Wy + b)
$$
In contrast, in a highway network, only a fraction of the input will be subjected to the step of the FFNN. The remaining fraction is permitted to pass through the network untransformed. The ratio of these fractions is managed by $t$, the transform gate and by $(1-t)$ the carry gate. The value of $t$ is calculated using a sigmoid function. Now we get:
$$
\mathrm{z}=\mathbf{t} \odot g\left(\mathbf{W}_{H} \mathbf{y}+\mathbf{b}_{H}\right)+(\mathbf{1}-\mathbf{t}) \odot \mathbf{y}
$$
\begin{itemize}
    \item Where $W_H, b_H$ are affine transformation.
    \item $\mathbf{t}=\sigma\left(\mathbf{W}_{T}\mathbf{y}+\mathbf{b}_{T}\right)$  is the transform gate
    \item $1-t$ is the carry gate.
\end{itemize}

Upon exiting the network, the transformed fraction of the input is summed with its untransformed fraction.

![[highway1.png|309]]

The highway network’s role is to adjust the relative contribution from the word embedding and the character embedding steps. The logic is that if we are dealing with an OOV word such as “misunderestimate”, we would want to increase the relative importance of the word’s 1D-CNN representation because we know that its GloVe representation is likely to be some random gibberish. On the other hand, when we are dealing with a common and unambiguous English word such as “table”, we might want to have more equal contribution from GloVe and 1D-CNN.