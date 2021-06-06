# Demonstrates how to write various custom actions.
import datetime as dt
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet, FollowupAction


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
        dispatcher.utter_message(text=f'exit or save time to slot?')
        t = {
                'year':  current_time.year,
                'month': current_time.month,
                'day': current_time.day
            }
        dispatcher.utter_message(json_message=t)

        # Button type utterance
        dispatcher.utter_message(buttons=[
                {'payload': '/exit', 'title': 'exit'},
                {'payload': '/set_time', 'title': 'save time to slot'},
            ])

        return []


class ActionSetTime(Action):

    def name(self) -> Text:
        return 'action_set_time'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        return [SlotSet('saved_time', f'{dt.datetime.now()}')]


class ActionSavedTime(Action):

    def name(self) -> Text:
        return 'action_saved_time'

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        saved_time = tracker.get_slot('saved_time')
        dispatcher.utter_message(response='utter_saved_time', saved_time=str(saved_time))

        return []


class ActionLastIntent(Action):

    def name(self) -> Text:
        return 'action_last_intent'

    # As far as I can tell there is no need for this function to be async.
    # It was made so for the purpose of using pytest on async functions.
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        last_intent = tracker.get_intent_of_latest_message()

        if last_intent == 'exit':
            return [FollowupAction('utter_goodbye')]
        elif last_intent == 'set_time':
            return [FollowupAction('action_set_time')]

        dispatcher.utter_message(text=f'last intent: {last_intent}')

        return []
