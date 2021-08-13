# rasa test

This document details the steps in which one should go about testing rasa.

## Testing rasa

Testing rasa is primarily done through the command line command which reads test stories written by the user and outputs the result to the `results` folder.

The command used to run tests on rasa is as follows:

```
rasa test -s test_folder
```

-s = An optional flag for defining a test stories folder. The default folder is the project's `test` folder. To mark a story as a test, the file name must begin with `test_`, for example, `test_stories.yml`

## Test stories

Test stories are written much like rasa training stories, with a few differences as will be detailed here. An example test story file is as follows:

```
stories:
- story: happy path 1
  steps:
  - user: |
      hello there!
    intent: greet
  - action: utter_greet
```

The most notable difference between a test story and a training story is the `user` key that is attached to every intent. Test stories require you to define the user input that is sent to the bot, and the resulting intent it is classified as.

### Test story entities

Testing if rasa correctly identifies entities in user input is written in the following way:

```
stories:
- story: happy path 1
  steps:
  - user: |
      hello, I am [george]{"entity": "name"}!
    intent: greet
  - action: utter_greet
```

In the above case, the user input `george` is identified as the `name` entity.

Occasionally you may need the value of the entity saved as a value other than that which the user input. For example:

```
stories:
- story: happy path 1
  steps:
  - user: |
      hello, I am [george]{"entity": "name", "value": "male_name"}!
    intent: greet
  - action: utter_greet
```

In the above example, instead of saving `george` as value `george`, we are only concerned if the name is a male or female name and thus decide to save it as `male_name` instead.

### Test story forms

Forms are a type of conversation pattern in rasa primarily implemented through rules. They run until all required information from the user is acquired or the user triggers an intent that causes the form the exit prematurely.

As writing a story is not required for them to work, a test story may not have a training story available for reference.

Furthermore, because the intent identified from user input may be irrelevant for the saving of user input for forms, one might be confused as to how a test story for a form is to be written. In general, there are two approaches:

1. Write an intent for every form input

If the form is made to function such that a particular intent has to be identified for user input to be saved, that intent can be written into the test story. For example:

```
stories:
- story: A test story with a form
  steps:
  - user: |
      start job form
    intent: start_job_form
  - action: job_form
  - active_loop: job_form
  - user: |
      I want to be a [fireman](job)
    intent: inform_job
  - action: job_form
  - slot_was_set:
    - job: fireman
  - active_loop: job_form
  - user: |
      3000
    intent: inform_salary
  - action: job_form
  - active_loop: null
  - slot_was_set:
    - salary: 3000
```

Here, every user input is followed by an intent. This intent is defined by the user in `domain`, trained in `nlu` and set as a requirement under `intent` when making the form:

```
forms:
  adventurer_form:
    required_slots:
      job:
        - type: from_entity
          entity: job
          intent:
          - inform_job
      salary:
        - type: from_text
          intent:
          - inform_salary
```

2. Write as the intent what the user input would be identified as outside the from.

```
stories:
- story: A test story with a form
  steps:
  - user: |
      start job form
    intent: start_job_form
  - action: job_form
  - active_loop: job_form
  - user: |
      3000
    intent: number
  - action: job_form
  - active_loop: null
  - slot_was_set:
    - salary: 3000
```

In this case, instead of defining a new intent, we use a previously defined intent `number` is the intent the user input would be identified as. To know what intent to use, you can use the command `rasa shell nlu` or rasa x, type in the input, and see the identified intent.

If the intent is incorrectly identified, the test story will fail. However, there will be no comment as to what failed.

## rasa test success

If rasa test succeeds, you should see the following inside `failed_test_stories.yml`:

```
# None of the test stories failed - all good!
```

If a test story fails the file will contains the test story as well as a comment as to which step failed. For example:

```
version: "2.0"
stories:
- story: happy test
  steps:
  - intent: affirm  # predicted: greet
  - action: utter_greet
```

Here, the user input was identified as `greet` instead of `affirm`.
