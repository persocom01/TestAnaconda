import spacy
import pytextrank

text = "Sugar is bad to consume. My sister likes to have sugar, but not my father."

nlp = spacy.load('en_core_web_sm')

nlp.add_pipe("textrank")
doc = nlp(text)

for phrase in doc._.phrases:
    topic = phrase.text
    topic = topic.lower().replace('the ', '')
    print('topic: ' + topic)
    print('rank, count:', phrase.rank, phrase.count)
    print('chunks: ' + str(phrase.chunks))
