# rasa config

## retrieval intents

To use retrieval intents, add the following to the pipeline after a featurizer and intent classifier:

```
pipeline:
- name: Featurizer
- name: Classifier
- name: ResponseSelector
  epochs: 100
  retrieval_intent: retrieval_intent1
- name: ResponseSelector
  epochs: 100
  retrieval_intent: retrieval_intent2
```
