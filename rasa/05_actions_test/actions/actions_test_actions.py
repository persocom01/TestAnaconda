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
        return 'action_show_time'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        current_time = dt.datetime.now()

        # Must be converted to a timestamp or it will be in the date format:
        # 2021-05-28 17:31:20.243162
        # Standard text utterance.
        dispatcher.utter_message(text=f'numerical timestamp: {current_time.timestamp()}')

        # Must be converted to a string before being passed as an argument or
        # it will become a timestamp: 1622194280.243162
        # Response + variable utterance.
        dispatcher.utter_message(response='utter_time_response', saved_time=str(current_time))

        # Image utterance
        dispatcher.utter_message(image='http://localhost:7000/img/time.jpg')

        # json type utterance, often used to pass custom payloads to other
        # applications.
        t = {
                'year':  current_time.year,
                'month': current_time.month,
                'day': current_time.day
            }
        dispatcher.utter_message(json_message=t)

        # Button type utterance
        dispatcher.utter_message(buttons=[
                {'payload': '/goodbye', 'title': 'end'},
                {'payload': '/set_time', 'title': 'set time'},
            ])

        return []


class ActionSetTime(Action):

    def name(self) -> Text:
        return 'action_set_time'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet('saved_time', f'{dt.datetime.now()}')]
