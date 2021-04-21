# rasa data

Data is where you enter training phrases for intents and how the bot should respond to them into rasa. Data supports 4 keys:

1. version
2. nlu
3. rules
4. stories

## version

version is determines how rasa parses the domain file. If not defined, it is assumed to be latest version supported by the version of Rasa Open Source you have installed. version is defined in the following way:

```
version: "2.0"
```

rasa 2.0 generally uses 2.0 as the version number.

## nlu

nlu is where you define intent training phrases as well as how rasa recognizes entities.

### intent training phrases

Intent training phrases are defined in the following way:

```
nlu:
- intent: intent1
  examples: |
    - ex1
    - ex2

- intent: intent2
  examples: |
    - ex1
    - ex2
```

If the same intent is defined in another file, the training phrases are added together.

### retrieval intents

Retrieval intents are a way to group multiple intents of the same type into one main type into one big one with many sub-intents. They are written similar to normal intents but with format retrieval_intent/sub_intent:

```
- intent: retrieval_intent/sub_intent1
  examples: |
    - ex1
    - ex2

- intent: retrieval_intent/sub_intent2
  examples: |
    - ex1
    - ex2
```

### entity recognition

Rasa can recognize entities in 3 different ways:
1. Synonyms
2. regex
3. Lookup table

They are added in the following way:

```
nlu:
- synonym: entity1
  examples: |
    - ex1
    - ex2

- regex: email
  examples: |
    - \w+@\w+\.com

- lookup: countries
  examples: |
    - Australia
    - Singapore
```

## rules

rules are a kind of rigid story you create when you always want a specific response to a type of conversation pattern. To use rules, `RulePolicy` must first be added to `config.yml`:

```
policies:
  - name: OtherPolicies
  - name: RulePolicy
```

rules are set in a similar way to stories, in that they have steps:

```
rules:

- rule: Only say `hello` if the user provided a name and only at the start of the conversation.
  conversation_start: true
  condition:
  - slot_was_set:
    - user_provided_name: true
  steps:
  - intent: greet
  - action: utter_greet
  wait_for_user_input: false
```

### retrieval intents

Using retrieval intents require a rule to be written for them:

```
- rule: retrieval intent
  steps:
  - intent: retrieval_intent
  - action: utter_retrieval_intent
```

### Optional rule properties

1. conversation_start
Setting `conversation_start: true` makes the rule only apply at the beginning of the conversation.
2. condition
You may set a condition to be fulfilled for the rule to apply. These can be `slot_was_set` or `active_loop` events.
3. wait_for_user_input
By default, rules implicitly end with `- action: action_listen`. Setting `wait_for_user_input: false` indicates that the bot should execute another action instead of waiting for user input.

## stories

stories is where you define how the bot works by putting together intents and responses
