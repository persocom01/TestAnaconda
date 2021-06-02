# testing rasa actions

rasa does not natively support testing of custom actions. As such, the 3rd party library `pytest` has been used for this purpose.

This readme details how to use `pytest` to test various type of rasa actions, and assumes the reader already knows how to use `pytest`.

## Installation

Use pip to install pytest.

```
pip install pytest
```

## Usage

1. Create the test file.

Assuming custom actions are located in the `actions` folder of the rasa project, the test file has to be located in the project root folder as follows:

```
└── rasa_bot
    ├── actions
    │   ├── __init__.py
    │   └── actions.py
    ├── pytest.ini
    ├── test_actions.py
```

This is because we will be importing actions as a python module into the test script, and python user modules assume that they are located in the same folder as the script itself.

As a reminder, `pytest` read python files starting with `test_`. `pytest.ini` is an optional but helpful configuration file for `pytest`.

To run the test, open command line in the rasa project folder and enter:

```
pytest
```

## Writing the test file

The most important part of using `pytest` is the writing of the test file itself. This section details how to write tests for various types of rasa actions. It will cover:
* Response actions
These are actions that send messages to rasa via `dispatcher.utter_message()`.
* Slot setting actions
These are actions that set slots via `SlotSet()` events.
* Form validation actions
These are actions designed to validated slot data in forms. The methods are named `validate_<slot_name>`.
* Setting the test Tracker
The Tracker is used to pass information on prior events to the function. Depending on whether the test target uses this information, it may or may not be necessary to write a suitable tracker for test purposes.

To test the actions, we must first import them into the script itself in the following way:

```
import actions.test_target as att
```

### Response actions

Customs actions that send responses to rasa may perform other functions at the same time, such as setting slots. This details how to test if messages sent to rasa via `dispatcher.utter_message()` are correct.

Response actions come in the form:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionName(Action):

    def name(self) -> Text:
        return 'action_name'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        subject = 'world'
        dispatcher.utter_message(text=f'hello {subject}')

        return []
```

Notable feature(s) are:
* The use of `dispatcher.utter_message()`
* The function may or may not return anything.

A test script for `pytest` for the above action can be written as follows:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import actions.test_target as att
import re


def test_action_name_run():
    dispatcher = CollectingDispatcher()
    output = att.ActionName().run(
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)

    subject = 'world'
    message = dispatcher.messages[0]['text']
    pattern = r'^hello (\w+)$'
    search = re.search(pattern, message)
    message_subject = search.groups()[0]
    is_message_correct = message_subject == subject

    assert (output == []) and is_message_correct
```

Thing(s) to note:
* The `dispatcher` object is called to retrieve the message sent to rasa.
For every use of `dispatcher.utter_message()` in the test target, a new item is added to the `dispatcher.messages` list, and must be called using their indexes [0], [1] and so on.
* The message is checked for validity using regex, the variable being extracted and checked through the use of regex groups.

### Slot setting actions

Another common use of custom actions is to set slots. Unlike responses, slots are set via events, which are returned by the custom action to rasa.

Slot setting actions come in the form:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet


class ActionName(Action):

    def name(self) -> Text:
        return 'action_name'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        slot_value = 'value'
        return [SlotSet('slot_name', slot_value)]
```

Notable feature(s) are:
* The method returns a list with the `SlotSet()` event.

A test script for `pytest` for the above action can be written as follows:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet

import actions.test_target as att


def test_action_name_run():
    dispatcher = CollectingDispatcher()
    output = att.ActionName().run(
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)
    slot_value = output[0]['value']
    is_slot_value_correct = slot_value == 'value'
    assert (output == [SlotSet('slot_name', slot_value)]) and is_slot_value_correct
```

Thing(s) to note:
* The value of the slot has been extracted using `output[0]['value']`
If the value of the slot is a fixed value, it is unnecessary to extract the value of the slot as shown here, and one only need `assert output == [SlotSet('slot_name', slot_value)]`

### Form validation actions

The last form of custom action this document will cover is the form validation action. These actions are used to validate that user input to forms is within acceptable value.

Form validation actions come in the form:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher

import re


class ValidateFormName(FormValidationAction):
    def name(self) -> Text:
        return "validate_form"

    def validate_slot(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        pattern = '^value$'
        is_valid = re.search(pattern, slot_value)

        if is_valid:
            return {'slot_name': slot_value}
        else:
            return {'slot_name': None}
```

Notable feature(s) are:
* The method takes in `slot_value` as an argument.
* The method has an `if` statement, which on success, returns a dictionary with the slot(s) to be set and their values. On failure, it returns none.

A test script for `pytest` for the above action can be written as follows:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet

import actions.test_target as att
import pytest


@pytest.mark.parametrize('input, expected_output', [
    ('value', {'name': 'value'}),
    ('wrong_value', {'name': None})
])
def test_validate_slot_name(input, expected_output):
    slot_value = input
    dispatcher = CollectingDispatcher()
    output = att.ValidateFormName().validate_name(
        slot_value=slot_value,
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)

    assert output == expected_output
```

Thing(s) to note:
* pytest.mark.parametrize() has been used to run multiple tests on the same function
This is to test that slot validation works both for correct and incorrect values.
* form validation methods accept `slot_value` as one of the arguments
This is not present in other actions.

### Setting the test Tracker

All the examples given up until this point have used the default `Tracker` thus far. However there are times when a customized Tracker must be used. Official documentation of the Tacker object can be found here: https://rasa.com/docs/action-server/sdk-tracker

Take for example this test target:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionName(Action):

    def name(self) -> Text:
        return 'action_name'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_slot = float(tracker.get_slot('slot_name'))
        last_intent = tracker.get_intent_of_latest_message()
        dispatcher.utter_message(text=f'{slot_name} has a value of {previous_slot} and the last intent was {last_intent}')

        return []
```

In this case:
* `tracker.get_slot()` was called to retrieve a previously set slot value.
* `tracker.get_intent_of_latest_message()` was called to retrieve the intent of the last message.

Both these methods are part of the `Tracker` object, and there is a need to pass a `Tracker` that contains the necessary values to the test function. A simple framework for a custom tracker is as follows:

```
def get_tracker_for_slots(
    slots: Dict,
    latest_msg={},
    followup_action=[]
):

    return Tracker(
            'default',
            slots,
            latest_msg,
            followup_action,
            False,
            None,
            {},
            'action_listen',
        )
```

Of the parameters passed to the tracker:

1. `slots` is used to define previously set slots

To set a slot, pass it a dictionary in the form:

```
{'slot1': value1, 'slot2': value2}
```

2. `latest_msg` is used to define attributes of the last user message

The message has many attributes, but for the purpose of this example, only one is needed:

```
{'intent_ranking': [{'name': 'intent', 'confidence': 1}]}
```

This ensures that the `tracker.get_intent_of_latest_message()` will return the intent that you want.
