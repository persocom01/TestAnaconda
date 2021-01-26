# NLP

# Architectures

RNNs, LSTMs, GRUs (Gated Recurrent Units) preserve word order.

RNNs suffer from the vanishing gradient problem. This means earlier layers in a deep RNN do not learn, and they have a short term memory.

To fix this problem, LSTMs and GRUs were created. Both are used very frequently and whole GRUs are faster than LSTMs, both are often tried to see which work better.

# Embeddings

Word2Vec, GloVe give words a location modeling relationships between words.

# Techniques

Bidirectional RNNs - reads sequences both forward and backward, improving accuracy.
Seq2Seq - do not require input and output to have the same length.
Attention - an improved Seq2Seq.
Memory Networks - answers questions. Story + question = answer. The NN must read and remember the necessary fact in the question.
