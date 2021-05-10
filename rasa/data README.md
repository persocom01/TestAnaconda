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

<!-- Retrieval intents -->
- intent: retrieval_intent/sub_intent1
  examples: |
    - ex1
    - ex2

- intent: retrieval_intent/sub_intent2
  examples: |
    - ex1
    - ex2
```

If the same intent is defined in another file, the training phrases are added together.

Retrieval intents are a way to group multiple intents of the same type into one main type into one big one with many sub-intents. They are written similar to normal intents but with format retrieval_intent/sub_intent.

### entity recognition

Recognizing entities in `nlu` is different from the entities defined in `domain`. The reason being that the entities defined here are only for the purpose of recognized as a prediction feature.

If you wish to store and return them to the user, you will need to define them as `entities` and `slots` inside `domain`. Rasa can recognize entities in 3 different ways:
1. synonym
2. regex
3. lookup

They are written the following way:

In `domain`:

```
entities:
  - weapon
  - email
  - goods

slots:
  - weapon
  type: text
  - email
  type: text
  - goods
  type: text
```

In `nlu`:

```
nlu:
<!-- In the case of a synonym, all variations of bows are saved as "bow" (the name of the synonym) when this entity is saved as a slot. You will still need to add synonyms to nlu training examples as synonyms do not affect the prediction model. -->
<!-- Instead of specifying synonyms, you can use dictionary notation to identify words as synonyms instead. They are written in the form: [bow_variant]{"entity": "domain_entity", "value": "bow"} -->
- synonym: bow
  examples: |
    - longbow
    - shortbow
- intent: inform_weapon
  examples: |
    - [sword](weapon)
    - [longbow](weapon)
    - [short bow]{"entity": "weapon", "value": "bow"}

- regex: regex_email
  examples: |
    - ^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$
- intent: inform_email
  examples: |
    - my email is [user@user.com](email)
    - This is my email [user@user.com](email)

<!-- You use lookup when you have a list of entities specific to your application that you wish to help rasa identify. This list should be < 10 million long. Like synonyms, you still need to add lookup to nlu training examples. lookup is case insensitive. -->
- lookup: lookup_goods
  examples: |
    - pheonix down
    - hi-potion
    - elixir
- intent: inform_purchase
  examples: |
    - may I buy a [pheonix down](goods)
    - I want to buy a [hi-potion](goods)
    - do you have an [elixir](goods)
```

Finally in stories:

```
- story: entity slot-filling
  steps:
  - intent: inform_weapon
    entities:
    - weapon: longbow
```

`RegexFeaturizer` needs to be added to pipeline in `config.yml` for regex to be recognized as a feature during intent classification.

`CRFEntityExtractor` or `DIETClassifier` need to be added to pipeline in `config.yml` to use regex entities. However, their matches are not limited to exact matches. More on the issue can be found here: https://github.com/RasaHQ/rasa/issues/3880

research on lookup is to be continued...

## rules

rules are a kind of rigid story you create when you always want a specific response to a type of conversation pattern. To use rules, `RulePolicy` must first be added to `config.yml`:

```
policies:
  - name: ...OtherPolicies
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
    entities:
      - name: "Timmy"
  - action: utter_greet
  wait_for_user_input: false

<!-- Rules for retrieval intents -->
- rule: retrieval intent
  steps:
  - intent: retrieval_intent
  - action: utter_retrieval_intent
```

However, rules are different from stories in that:
1. You cannot have more than 1 intent.
The error you get for violating this is `Found rules 'rule_name' that contain more than 1 user message`. It appears that rules cannot be used require the user to type more than 1 message. This forces rules to either be short or comprising mostly of actions.
2. checkpoint cannot be used.

Using retrieval intents require a rule to be written for them.

### Optional rule properties

1. conversation_start
Setting `conversation_start: true` makes the rule only apply at the beginning of the conversation.
2. condition
You may set a condition to be fulfilled for the rule to apply. These can be `slot_was_set` or `active_loop` events.
3. wait_for_user_input
By default, rules implicitly end with `- action: action_listen`. In practice, this ends the current conversational flow. By setting `wait_for_user_input: false`, the conversation flow does not end when the rule is executed but picks off where it last left off. This means any action that would have been executed will continue to be executed, including fallback actions. To prevent a fallback action from being executed, either do not define a fallback action or make stories where the conversation flow is interrupted by execution of the rule but still continue the next action.

## stories

stories is where you define how the bot works by putting together intents and responses.

stories must always begin with an intent.

When putting retrieval intents into stories, use the main intent name and not sub-intents.

### checkpoint

Using checkpoints is how you modularize your story. For instance, in story_file1 we can have:

```
stories:

- story: start adventure
  steps:
  - intent: start_adventure
  - checkpoint: forest_stage_start

- story: end adventure
  steps:
  - checkpoint: forest_stage_end
  - action: utter_loot
```

and story_file2 we have:

```
stories:

- story: forest_stage_start
  steps:
  - checkpoint: forest_stage_start
  - action: utter_slime_encounter
  - intent: attack
  - action: utter_slime_death
  - intent: loot
  - checkpoint: forest_stage_end
```

Note that checkpoints were used to place the story in the second file in the middle of the first one.

Known limitations of checkpoints are:
* The stories beginning with a checkpoint cannot start with different actions. This is because it appears that rasa does not to consider anything prior to the checkpoint during prediction and will become confused without an intent as to why the story splits into two different actions. If there is a need for a checkpoint module to split into different actions based on prior parts of the story, make copies of it for each prior part of the story that will result in a different ending instead.
* checkpoints cannot be used to make looping stories by starting and ending with the same checkpoint, even if the starting checkpoint is put into a different story.

### or statements

Using `or:`, one can cause multiple intents to converge into a single action. For instance:

```
- action: utter_ask_dinner
- or:
  - intent: steak
  - intent: fish
- action: utter_vegetarian_restaurant
```

Overusing or statements is not recommended, as it will slow down training.
