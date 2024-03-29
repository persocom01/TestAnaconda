# NLP

A native English speaker knows approx 20k words. This is a reference point when determining max vocabulary in nlp.

# Architectures

RNNs, LSTMs, GRUs (Gated Recurrent Units) preserve word order.

RNNs suffer from the vanishing gradient problem. This means earlier layers in a deep RNN do not learn, and they have a short term memory.

To fix this problem, LSTMs and GRUs were created. Both are used very frequently and whole GRUs are faster than LSTMs, both are often tried to see which work better.

In tensorflow, LSTM layers have two optional parameters: LSTM(dim_number, return_state=False, return_sequence=False)
Return state returns 3 values, an array of arrays hidden states at each time step (if return_sequence=True), or just the final hidden state otherwise. The second is an array of the final hidden state, and the last is an array of the final cell state as usual.

Bidirectional RNNs are used to improve classification when words after the word in question may have a significant impact on its classifiction. Implementing a bidirectional RNN is easy in keras, simple wrap an existing LSTM layer with `Bidirectional()`. However, bidirectional RNNs should not be used when predicting future events if the concern.

# Embeddings

Word2Vec, GloVe give words a location modeling relationships between words.

# Techniques

Bidirectional RNNs - reads sequences both forward and backward, improving accuracy.
Seq2Seq - do not require input and output to have the same length.
Attention - an improved Seq2Seq.
Memory Networks - answers questions. Story + question = answer. The NN must read and remember the necessary fact in the question.
