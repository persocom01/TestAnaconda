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

Custom actions can be used to pass responses to rasa in a similar manner as an utter action. To do so put the following code inside the run method of a custom action class:

```
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
```

### responses

```
dispatcher.utter_message(
    template="utter_greet",
    name="Sara"
)
```

When using a custom action server:

```
{
  "events":[
    ...
  ],
  "responses":[
    {
      "template":"utter_greet",
      "name":"Sara"
    }
  ]
}
```

### slots

Set slots in the following way:

```
rasa_sdk.events.SlotSet(
    key: Text,
    value: Any = None,
    timestamp: Optional[float] = None
)
```

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
