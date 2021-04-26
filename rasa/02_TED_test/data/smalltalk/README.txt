# rasa smalltalk module

## Installation

1. Add the following ResponseSelector to `config.yml` under `pipeline`:

```
- name: ResponseSelector
  epochs: 100
  retrieval_intent: smalltalk
```

2. Add the following ResponseSelector to `config.yml` under `policies`:

```
  - name: RulePolicy
```
