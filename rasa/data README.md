# rasa data

Data is where you enter training phrases for intents and how the bot should respond to them into rasa. Data supports 4 keys:

1. version
2. nlu
3. rules
4. stories

## version

version is determines how rasa parses the domain file. If not defined, it is assumed to be latest version supported by the Rasa Open Source you have installed. version is defined in the following way:

```
version: "2.0"
```

rasa 2.0 generally uses 2.0 as the version number.

## nlu

nlu is where you define the intent training phrases. They are defined in the following way:

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
