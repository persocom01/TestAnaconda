# rasa config

## pipeline

### retrieval intents

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

## policies

policies is where you fine tune certain aspects of the bot.

1. TEDPolicy

The Transformer Embedding Dialogue Policy (TEDPolicy) is what you adjust when you want to change how many steps back in stories data the bot takes into account when training the model. It is written like this:

```
  - name: TEDPolicy
    max_history: 5
    epochs: 100
```

`max_history` is the main parameter you might want to change. By default, TEDPolicy has no max_history, which means the model looks at the entire story, no matter how long it is.

TEDPolicy has many more optional parameters that are likely too complicated to be worth adjusting, but they can be found here: https://rasa.com/docs/rasa/policies/

2. MemoizationPolicy

MemoizationPolicy determines if the bot should look at and try to match the current conversation to your stories data.

```
  - name: MemoizationPolicy
    max_history: 3
```



2. RulePolicy
