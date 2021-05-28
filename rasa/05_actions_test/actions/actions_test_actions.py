# Demonstrates how to write various custom actions.
import datetime as dt
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet


# Demonstrates how to create utter type custom actions. Utter can be combined
# with other types of actions.
class ActionShowTime(Action):

    def name(self) -> Text:
        return "action_show_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Must be converted to a timestamp before being passed into the f
        # string or it will be text.
        current_numerical_time = dt.datetime.now().timestamp()
        # Standard text utterance.
        dispatcher.utter_message(text=f'numeral timestamp: {current_numerical_time}')

        # Must be converted to a string before being passed as an argument or
        # it will become a timestamp.
        current_time = str(dt.datetime.now())
        # Response + variable utterance.
        dispatcher.utter_message(response='utter_time_response', saved_time=current_time)

        # Image utterance
        dispatcher.utter_message(image='http://localhost:7000/img/time.jpg')

        # json type utterance, often used to pass custom payloads to other
        # applications.
        t = {  }
        # print('testing')
        # print(tracker.get_intent_of_latest_message())
        # print(tracker.latest_message)

        return []


class ActionSetTime(Action):

    def name(self) -> Text:
        return "action_set_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet('saved_time', f'{dt.datetime.now()}')]
