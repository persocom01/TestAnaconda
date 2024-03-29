# rasa config

A readme on how to configure the rasa `config.yml` file.

## pipeline

How to configure the data pipeline: https://rasa.com/docs/rasa/2.4.x/components

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

The Transformer Embedding Dialogue Policy (TEDPolicy) is what you adjust when you want to change how many steps back in stories data the bot takes into account when training the model.

```
  - name: TEDPolicy
    max_history: 5
    epochs: 100
```

`max_history` is the main parameter you might want to change. By default, TEDPolicy has no max_history, which means the model looks at the entire story, no matter how long it is. Setting a finite `max_history` can be desirable to decrease training time.

TEDPolicy has many more optional parameters that are likely too complicated to be worth adjusting, but they can be found here: https://rasa.com/docs/rasa/policies/

2. MemoizationPolicy

MemoizationPolicy determines if the bot should look at and try to match the current conversation to your stories data. It may be a necessary policy when you find that the bot does not work according to the stories you write, as the prediction model appears to be easily confused.

```
  - name: MemoizationPolicy
    max_history: 3

  <!-- To use AugmentedMemoizationPolicy instead. -->
  - name: AugmentedMemoizationPolicy
    max_history: 3
```

`max_history` determines how far back in current conversation history the bot should take into account when looking for a match in stories data. 1 turn includes the message by the user as well as any actions that result from it.

`AugmentedMemoizationPolicy` is used when you do not wish for slots set earlier in the conversation to affect the current prediction. The slots are forgotten when trying to find a match in the story if they are set prior to `max_history`, or only the slots set within `max_history` will affect the prediction. To use this policy specifically so that the conversation will continue without slots, stories should be written with and without slots.

2. RulePolicy

RulePolicy is necessary to use rules.

```
- name: "RulePolicy"
  core_fallback_threshold: 0.3
  core_fallback_action_name: action_default_fallback
  enable_fallback_prediction: true
  restrict_rules: true
  check_for_contradictions: true
```

`core_fallback_threshold` determines the prediction threshold above which the core_fallback_action is triggered.

`core_fallback_action_name` is where you determine what the fallback action is. By default, this is set to `action_default_fallback`, which is a default action inbuilt with rasa that calls the `utter_default` response. This can be customized, if desired, by defining `class ActionDefaultFallback(Action)` in `actions.py`.

`enable_fallback_prediction` (default true) enables usage of the fallback.

`restrict_rules` (default true) restricts rules to 1 user turn. Making this false may result in unexpected behavior.

`check_for_contradictions` (default true) before training, performs a check to make sure that slots and active loops set by actions are defined consistently for all rules.
