# rasa actions

A readme on how to create and use custom action files such as `actions.py` in the `actions` folder.

## Writing custom actions

Custom actions are written as python files in the `actions` folder. A basic action looks like this:

```
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionName(Action):

    def name(self) -> Text:
        return "action_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text=f'string {var}')

        return []
```

Things of note here:
* The `name` function
This determines the name the action is referred to in `domain`. The class name does not matter.
* The `run` function
This determines what the action actually does. All custom actions have this function, except slot validation actions.
* The return value
The return value for custom actions is an optional list of rasa events triggered by the action. Multiple events are separated by commas. One common event is the `SlotSet()` event, but others exist as detailed here: https://rasa.com/docs/action-server/sdk-events

### Custom action type

Custom actions are capable of more than the below list. However, the most common functions they serve include one or more of the following:

1. responses

Custom actions can be used to pass responses to rasa in a similar manner as an utter action. To do so put the following code inside the `run` method of a custom action class:

```
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        <!-- Text responses -->
        dispatcher.utter_message(text=f'string {var}')

        <!-- Responses based on utter actions -->
        dispatcher.utter_message(response='utter_response', reponses_entity=value))

        <!-- Image responses -->
        dispatcher.utter_message(image='link')

        <!-- Json responses -->
        dispatcher.utter_message(json_message={
            'key1': value1,
            'key2': value2
          })

        <!-- Button responses -->
        dispatcher.utter_message(buttons=[
            {'payload': '/intent1', 'title: label1'},
            {'payload': '/intent2', 'title: label2'},
          ])

        return []
```

responses are often used in conjunction with other rasa actions. Unlike utter action responses, custom button responses can be mixed with picture responses and still be usable in rasa shell. Json responses are most often used to deliver custom payloads to other applications.

2. slot setting

Another common application of custom actions is slot setting. Unlike responses, slot setting is done via events through `return` in the `run` method of a custom action:

```
from rasa_sdk.events import SlotSet

def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    return [
        SlotSet('key', value, optional_timestamp),
        SlotSet('key', value, optional_timestamp)
      ]
```

3. Other events

Besides SlotSet(), other events can be triggered via custom actions, including other custom actions. Details can be found here: https://rasa.com/docs/action-server/sdk-events

### Form validation

Form validation custom actions are written differently from the others, notably:
* The action class must be named `validate_<form_name>`
* There is a `name` method but no `run` method
* Instead of `run` methods of the class must be named `validate_<slot_name>`
* There are normally multiple validate methods within the same class
* The methods return a dictionary of the slots to be set in the form `{'slot1': value, 'slot2': value}`

1. Define custom action in domain

The custom action must named `validate_<form_name>`.

```
actions:
- validate_form_name
```

2. Create a form validation action file inside the `action` folder

The inbuilt class `FormValidationAction` simplifies the process of writing a form validation action. To validate a slot, define a method `validate_<slot_name>`.

```
from typing import Text, List, Any, Dict

from rasa_sdk import Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

class ValidateFormName(FormValidationAction):
    def name(self) -> Text:
        return "validate_form_name"

    @staticmethod
    def cuisine_db() -> List[Text]:
        """Database of supported cuisines"""

        return ["caribbean", "chinese", "french"]

    def validate_cuisine(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate cuisine value."""

        if slot_value.lower() in self.cuisine_db():
            # validation succeeded, set the value of the "cuisine" slot to value
            return {"cuisine": slot_value}
        else:
            # validation failed, set this slot to None so that the
            # user will be asked for the slot again
            return {"cuisine": None}
```

Multiple `validate_<slot_name>` methods can be included in the same class. The form contains a slot that does not have a validate method, validation of that slot will be skipped.

## Running an action server

Custom actions are not enabled by default. Enabling them requires modifications to the rasa project as detailed here.

### Local server

1. Modify endpoints.yml

Uncomment the following line:

```
action_endpoint:
 url: "http://localhost:5055/webhook"
```

2. Run server

Enter the following into command line:

```
rasa run actions
```
