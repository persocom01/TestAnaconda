# gensim

A gensim testing playground based on the tutorial here: https://www.tutorialspoint.com/gensim/index.htm

quickstart: https://scienceofdata.org/2020/05/24/word2vec-vs-fasttext-a-first-look/

## Usage

### Word2Vec

Continuous Bag-of-Words model (CBOW)
Predicts center word given context.

Skip-gram model
Predicts context given center word.
- better for larger corpus but slower to train.

pre trained models: http://vectors.nlpl.eu/repository/

### GloVe

pre trained models: https://nlp.stanford.edu/projects/glove/

### FastText

FastText can be considered an extension of Word2Vec that adds character n-grams into the mix. The benefit of FastText is its ability to recognize a limited number of Out-Of-Vocabulary (OOV) words. However, as a result it is generally 1.5x slower. It is best used on noisy corpus, while Word2Vec can give better results on cleaner, more formal text.

pre trained models:
1. http://vectors.nlpl.eu/repository/
2. https://fasttext.cc/docs/en/english-vectors.html
