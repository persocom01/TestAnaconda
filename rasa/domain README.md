# rasa domain

This file contains detailed instructions on how to use domain files. Domain files support 8 keys. These are:

1. version
2. intents
3. entities
4. slots
5. responses
6. forms
7. actions
8. session config

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

entities is a list of entities that can extracted by an entity extractor in your NLU pipeline defined in `config.yml`. They are defined in the following way:

```
entities:
  - PERSON        # entity extracted by SpacyEntityExtractor
  - time          # entity extracted by DucklingEntityExtractor
  - entity1
      entity1_property1:
      - entity1_prop1_value1
      - entity1_prop1_value2
      entity1_property2:
      - entity1_prop2_value1
      - entity1_prop2_value2
  - entity2
```

`PERSON` and `time` are special entities reserved by the `SpacyEntityExtractor` and `DucklingEntityExtractor` respectively.

If you wish to give entities additional properties, they should be listed and defined here.

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

If the same slot is defined in another file, the latest (by alphabetical order) file will take precedence.

### slot datatypes

slots can be of the following dtypes:
1. text
2. bool
bool can be set to either `true` or `false`. You need to define a custom action to use a bool slot. However, you can get around this by using a text slot and passing it "true" or "false" values as strings using `[user_text]{"entity": "entity_name", "value": "true"}` inside `nlu` intent examples instead.
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
  - text: "string {slot_name}"
    image: "image_url"
  - text: "string"
    buttons:
    - title: "option1"
      payload: '/intent1{{"entity": "value"}}'
    - title: "option2"
      payload: "/intent2"

  utter_response2:
  - string: "string"
    custom:
      json_key1: value
      json_key2:
        - list_value1
        - list_value2

<!-- Responses for retrieval intents are identical but come in the name format utter_retrieval_intent/sub_intent -->
  utter_retrieval_intent/sub_intent1:
  - text: "string {slot_name}"
    image: "image_url"
  - text: "string"
  utter_retrieval_intent/sub_intent2:
  - text: "string"
```

Reponses are governed by a number of rules (or bugs), both written and unwritten in the official documentation.
* All responses must start with utter_
They will be considered custom actions otherwise.
* `-` under a response indicates a response variation.
* Each response variation can have multiple datatypes, but each datatype may only appear once.
* The order of datatypes does not matter. However, if `image` and `button` appear in the same response variation, rasa shell will not give you the option of using the buttons. It works on rasa x, however.
* All response variations must contain the `text` datatype. Even if an empty string is given, it will register as `""` on rasa x.

If the same response is defined in another file, the latest (by alphabetical order) file will take precedence. As such, it is possible to keep the base responses and build on them by overriding them.

### local images

rasa does not support usage of local images. To display local images for development:

1. Create an `img` folder in the rasa project and put all the images you need in it.

2. Create a python file in the rasa project folder and copy the following code in it:

```
from flask import Flask, send_file

app = Flask(__name__)


@app.route('/img/<path:path>')
def get_image(path):
    image_path = 'img/{}'.format(path)

    return send_file(image_path, mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)

```

3. Run the flask server.

```
<!-- If you don't already have Flask. -->
pip install Flask

python img_server.py
```

4. Prefix all image file links with the following code:

```
http://localhost:7000/img/
```

### buttons

Using the button option in responses causes rasa shell to display a series of options for the user to pick using arrow keys. In rasa x, the options are listed from left to right.

```
responses:
  utter_response1:
  - buttons:
    - title: "option1"
      payload: '/intent1{{"entity": "value"}}'
    - title: "option2"
      payload: "/intent2"
```

`title` is the name of each option.
`payload` is the intent an option will trigger when selected. Note the `/` before the intent name. Selecting it will bypass nlu and trigger the intent. However, you will still need to define the intent in nlu with training examples. Also note the specific formatting when using the button to fill a slot. To prevent parsing errors:
* `entity` and `value` bust be surrounded by `""` and not `''`.
* Because of this, the payload string itself must use `''` and not `""`
* Double `{}` must be used.

## forms

forms are a conversation pattern used to collect pieces of information from a user. To use forms:

1. Add `RulePolicy` to `config.yml`:

```
policies:
  - name: ...OtherPolicies
  - name: RulePolicy
```

2. Define a form in the forms section of `domain`:

```
forms:
  restaurant_form:
    cuisine:
      - type: from_entity
        entity: cuisine
        not_intent:
        - goodbye
        - not_listed
    num_people:
      - type: from_entity
        entity: number
```

3. Activate the form in `rules`:

```
rules:
- rule: Activate form
  steps:
  - intent: request_restaurant
  - action: restaurant_form
  - active_loop: restaurant_form
```

To be continued: https://rasa.com/docs/rasa/forms

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
