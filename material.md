\section{Transformers Implementation for GPTs (Coding) (Muchang) - 20 pts}

Generative Pretrained Transformers (GPTs) have recently gained prominence as the state-of-the-art architecture for many NLP tasks. If you've ever used ChatGPT, as the name suggests, you've used a GPT model. In the following question, you will develop a \texttt{Pytorch} implementation of the multi-head attention layer of a lightweight GPT model. Using a preprocessed dataset of Shakespeare's plays with a character-level tokenization, you'll then train the model to generate novel Shakespearean text. 

Before beginning to code, you will need to set up your environment. First, git clone this \href{https://github.com/mbahng/gpt2-pytorch}{repository} and create a \texttt{conda} environment with \texttt{numpy} (\texttt{conda install numpy}) and \texttt{torch} (\texttt{pip install torch}) installed. Make sure to activate it before you start. 

Based on our experiments, the model \emph{should} be lightweight enough to be trained on a personal machine. As another option, you may also use your Google Cloud credits to reserve a VM on Google Cloud Platform (GCP) for this question. See the announcement on Ed for more details on obtaining credits. Alternatively, you can also reserve a VM through \href{https://vcm.duke.edu/apps/index}{Duke OIT}, though it may be less powerful than a GCP VM. 

To briefly go over this repository, 
\begin{enumerate}
    \item \texttt{model.py} defines the actual transformer model that we will train. Note that a transformer model consists of multiple \textit{self-attention modules} which each consist of a \textit{self-attention layer} plus a simple \textit{multilayer perceptron} (MLP), including nonlinearities and normalization layers, which help with training. The \texttt{Block} class composes both the \texttt{SelfAttention} class and \texttt{MLP} class for another layer of abstraction. 
    \item These decoder layer modules are stacked on top of each other within the transformer model, which you can see in the \texttt{transformer} attribute of the \texttt{Transformer} class. 
    \item \texttt{train.py} is a script that will train the model. 
    \item \texttt{sample.py} is a script that will generate text from the model. 
\end{enumerate}

To set up some notation, if the embedding dimension is $C$, and the sequence length is $L$, then a sequence of text inputs can be written as a matrix of shape $L \times C$, also called a \textit{rank 2 tensor}.\footnote{A scalar is a rank 0 tensor. A vector, which is an array of scalars, is a rank 1 tensor. A matrix, which is an array of vectors, is a rank 2 tensors. So on and so forth.} 

To parallelize these operations, PyTorch supports \textit{batching} of inputs, allowing you to input a collection of $B$ matrices of shape $L \times C$, stored in a \textit{rank 3 tensor} of shape $B \times L \times C$, and performing the same operations on every single matrix at once. This also helps with parallelizing matrix multiplication (and other operatons) over higher order tensors. For example, if $A$ has shape $B_1 \times \ldots \times B_k \times L \times M$ and $B$ has shape $B_1 \times \ldots \times B_k \times M \times N$ , then $AB$ has shape $B_1 \times \ldots \times B_k \times L \times N$. We are essentially matrix multiplying over the last two ranks. You should get familiar with this concept of batching, as it will save you a lot of time and effort compared to using for loops. \textit{We do not recommend using for loops, as both the code tends to get messy and the runtime will be significantly slower.}

\paragraph{(a)} To begin, we will first perform a quick sanity check to ensure your environment has been set up correctly: run the command \texttt{python sample.py} from the base of the repository to generate a sample of text. You should notice that the results just look like a random sequence of characters. There are two reasons for this: we haven't implemented the attention layer, nor have we trained the model.

Running sample.py, I can see a clear string of gibberish generated.

\paragraph{(b)}  Let's focus on the attention layer first. We will do this step by step by implementing the \texttt{forward} method in \texttt{SelfAttention} class in \texttt{model.py}. Recall that the formula for scaled-dot product single-headed attention is 
\begin{equation} 
  \mathrm{Attention}(Q, K, V) = \mathrm{softmax} \bigg( \frac{Q K^T}{\sqrt{C}} \bigg) V
\end{equation} 
where $Q, K, V$ are the query, key, value matrices, and the softmax operation is done over the \textit{rows} of the input matrix. Each of these steps should not take more than 10 lines of code each. We have provided an outline in the codebase as a guide, but you do not necessarily have to follow it. 

\begin{enumerate}[i.]
    \item The query, key, and value matrices are computed by \texttt{self.c\_attn}, which is a linear layer $\ell$ that maps each token $\mathbf{x}$ to its associated key, query, and value vectors through the map
    \begin{equation}
        \ell(\mathbf{x}; \mathbf{A}, \mathbf{b}) = \mathbf{A} \mathbf{x} + \mathbf{b} = \begin{pmatrix} \mathbf{q} \\ \mathbf{k} \\ \mathbf{v} \end{pmatrix}
    \end{equation}
    where $\mathbf{x} \in \mathbb{R}^{C}$, $\mathbf{A} \in \mathbb{R}^{3C \times C}$, and $\mathbf{b} \in \mathbb{R}^{3C}$. This is batched over the entire sequence length $L$ and over the $B$ batches for each sequence, so our output wouldn't be simply in $\mathbb{R}^{3C}$, but rather of size $B \times L \times 3C$. You want to take this output matrix of parameters and partition it into the query, key, value matrices, each of size $B \times L \times C$. 
    
    \item We want to implement \textit{multihead} attention by further partitioning each matrix into $H$ submatrices, where $H$ is the number of heads. $H$ is available as \texttt{self.n\_head} within the class. 
    \begin{align*}
        Q & \mapsto (Q_1, \ldots, Q_H) \\ 
        K & \mapsto (K_1, \ldots, K_H) \\ 
        V & \mapsto (V_1, \ldots, V_H)
    \end{align*}
    To implement multihead attention, we want to add an extra rank to change the shape of each matrix from $(B \times L \times C) \mapsto (B \times H \times L \times  C/H)$. Note that $H$ must evenly divide $C$ since we want to evenly partition the embedding dimension, so $C/H$ will always be an integer. The same logic should be looped through the $K$ and $V$ matrices as well. A visual is provided for some intuition. 

    \begin{figure}[H]
      \centering 
      \includegraphics[scale=0.25]{2024 Examples/multihead_visual.png}
      \caption{A visualization of how a single element of a batch, of shape $L \times C$, should be reshaped to shape $H \times L \times C/H$. You want to add an extra dimension by stacking these submatrices on top of each other. In the end, we should satisfy \texttt{Q[i, j, :, :].shape = (T, C // H)}.} 
      \label{fig:multihead_visual}
    \end{figure}
    
    \item Now we must actually implement the attention operation described in the equation above. First, implement only the operation 
    \begin{equation}
        \frac{Q_i K_i^T}{\sqrt{C/H}}
    \end{equation} 
    for all $i = 1, \ldots H$ over all $B$ batches (note that there is no index over batches notated, which is perhaps a fault of the notation, but this is common in transformer papers to leave out an index). Once this is done, your result should be of shape $B \times H \times L \times L$. 

    \item We also need a masking step since this is a decoder architecture. We have provided a \texttt{self.mask} attribute consisting of a lower triangular matrix of $1$s. Use this to mask your output so that future tokens are ignored in the attention score. You might find the \texttt{torch.Tensor.masked\_fill()} operation helpful. 

    \item Then, we can apply the softmax operation on the rows of each $L\times L$ matrix within each head and batch. After this should be a dropout layer, which we have implemented for you. Finally, we can multiply by the value matrix $V$ to get the head outputs $y$. After this step, the assertion statement that we have implemented on \texttt{y.shape} should hold true. 
\end{enumerate}


\paragraph{(c)} Train this model using the command \texttt{python train.py} and plot the loss curve over time. Note that training for 2000 epochs on Google Cloud is estimated to take between 5-20 minutes. You should allocate your time accordingly for training. 

\paragraph{(d)} Generate a new brief Shakespearean text using the command \texttt{python sample.py} and post your output. Now you have learned the details of a minimal transformer architecture! 
