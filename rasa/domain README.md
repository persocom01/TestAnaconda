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

version is determines how rasa parses the domain file. If not defined, it is assumed to be latest version supported by the Rasa Open Source you have installed. version is defined in the following way:

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

slots are defined in the following way:

```
slots:
  slot1:
    type: dtype
    influence_conversation: true_or_false

  slot2:
    type: dtype
    influence_conversation: true_or_false
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
```

All responses must start with utter_, otherwise they will be considered custom actions.

Each response entry can have multiple datatypes, but each datatype may only appear once.

If two files share the same response keys, the latest (by alphabetical order) file will take precedence. As such, it is possible to keep the base responses and build on them by overriding them.
