# rasa domain

This file contains detailed instructions on how to use domain files. Domain files support 7 keys. These are:

1. version
2. intents
3. entities
4. slots
5. responses
6. actions
7. session config

Describing how they are used is the purpose of this document.

## version

version is determines how rasa parses the domain file. If not defined, it is assumed to be latest version supported by the version of Rasa Open Source you have installed. version is defined in the following way:

```
version: "2.0"
```

rasa 2.0 generally uses 2.0 as the version number.

## intents

intents are defined in the following way:

```
intents:
  - intent1
  - intent2
```

## entities

entities are defined in the following way:

```
entities:
  - entity1
  - entity2
```

## slots

slots are the how you ask the bot to store information. They are defined in the following way:

```
slots:
  slot1:
    type: dtype
    initial_value: value
    auto_fill: true_or_false
    influence_conversation: true_or_false

  slot2:
    type: categorical
    values:
      - rare
      - medium
      - well done

  slot3:
    type: float
    min_value: 0.0
    max_value: 1.0
```

### slot datatypes

slots can be of the following dtypes:
1. text
2. bool
bool can be set to either `true` or `false`.
3. categorical
Unlike other types of slots, when defining categorical slots, you need to also define all their possible values.
4. float
The float category is used to store numbers. Unlike other types of slots, it comes with two innate properties: `min_value` and `max_value` which are by default 0.0 and 1.0 respectively. If float is set to any value below or above those limits, those values are treated as equal to those limits for the purposes of influencing conversation.
5. list
You might need to define a custom action to use this slot type.
6. any

### Optional slot properties

1. influence_conversation
Setting `influence_conversation: false` prevents the slot to from affecting the next action prediction. By default, this is true. For the `list` dtype, predictions are only affected by whether the list is empty or filled. dtype `any` cannot influence conversations.

2. auto_fill
By default, if an entity and a slot share the same name, the slot will be set when the entity is identified. Setting `auto_fill: false` prevents this behavior.

```
<!-- Assume the slot "name" is defined in domain -->
stories:

- story: entity slot-filling
  steps:
  - intent: greet
    entities:
    - name: John
```

3. initial_value
Sets the initial value of a slot.

### Custom slots

Slots can also be of a custom dtype. To do this:

1. Create a python file defining the behavior of the custom slot.

Example:

```
from rasa.shared.core.slots import Slot

class CustomSlot(Slot):

    def feature_dimensionality(self):
        return 2

    def as_feature(self):
        r = [0.0] * self.feature_dimensionality()
        if self.value:
            if self.value <= 6:
                r[0] = 1.0
            else:
                r[1] = 1.0
        return r
```

2. Have the file be identified as a python module inside the rasa project folder.

To do this, create a new folder (or use an existing one) inside the rasa project folder with an empty `__init__.py` file. Place the custom slot python file in it.

```
└── rasa_bot
    ├── addons
    │   ├── __init__.py
    │   └── my_custom_slots.py
    ├── config.yml
    ├── credentials.yml
    ├── data
    ├── domain.yml
    ├── endpoints.yml
```

3. Define a slot with the custom slot type.

```
slots:
  slot4:
    type: addons.my_custom_slots.CustomSlot
    influence_conversation: true
```

### setting slots

Slots can be set in three ways:

1. Through identification of entities in intents.
2. Through forms.
3. Via custom actions.

To use a slot, enter the following line in `stories:`

```
- slot_was_set:
  - slot1: slot_value
```

## responses

responses are defined in the following way:

```
responses:
  utter_response1:
  - dtype: "string_or_link"
    dtype: "string_or_link"
  - dtype: "string_or_link"

  utter_response2:
  - dtype: "string_or_link"
    dtype: "string_or_link"
  - dtype: "string_or_link"

<!-- Responses for retrieval intents -->
  utter_retrieval_intent/sub_intent1:
  - dtype: "string_or_link"
    dtype: "string_or_link"
  - dtype: "string_or_link"
  utter_retrieval_intent/sub_intent2:
  - dtype: "string_or_link"
    dtype: "string_or_link"
  - dtype: "string_or_link"
```

Where each `-` indicates a response variation. Each response variation can have multiple datatypes, but each datatype may only appear once.

All responses must start with utter_, otherwise they will be considered custom actions.

If two files share the same response keys, the latest (by alphabetical order) file will take precedence. As such, it is possible to keep the base responses and build on them by overriding them.

Responses for retrieval intents are written in the same way, but named in the format utter_retrieval_intent/sub_intent.

### Response variables

You are able to insert slot data into variables in the following way:

```
responses:
  utter_greeting:
  - text: "Hi {name}."
```



## actions

actions are defined in the following way:

```
actions:
  - action_action1
  - action_action2
```

## session_config

session_config is defined in the following way:

```
session_expiration_time: 60  # value in minutes
carry_over_slots_to_new_session: true
```

session_expiration_time determines the elapsed time before the bot assumes a new session has started.
carry_over_slots_to_new_session determines if slots set in prior sessions affect new sessions.
