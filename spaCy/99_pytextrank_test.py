import spacy
import pytextrank

text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."

nlp = spacy.load('en_core_web_sm')

nlp.add_pipe("textrank")
doc = nlp(text)

for phrase in doc._.phrases:
    topic = phrase.text
    topic = topic.lower().replace('the ', '')
    print('topic: ' + topic)
    print('rank, count:', phrase.rank, phrase.count)
    print('chunks: ' + str(phrase.chunks))