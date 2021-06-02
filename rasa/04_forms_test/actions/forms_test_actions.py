from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher

import re


class ValidateAdventurerForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_adventurer_form"

    @staticmethod
    def class_list() -> List[Text]:
        return ['fighter', 'knight', 'crusader', 'priest', 'archpriest']

    def validate_name(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        print('validating name')
        pattern = r'^[a-zA-Z][a-zA-z\s]*'
        is_valid = re.search(pattern, slot_value)
        print(is_valid)

        if is_valid:
            print('pass')
            dispatcher.utter_message(text='noted')
            return {'name': slot_value}
        else:
            print('fail')
            dispatcher.utter_message(text='names may only contain letters and spaces')
            return {'name': None}

    def validate_age(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        print('validating age')
        pattern = r'^[0-9]+$'
        is_valid = re.search(pattern, slot_value)

        if is_valid:
            is_minor = float(is_valid.group()) < 16
            if is_minor:
                dispatcher.utter_message(text='noted')
                return {'age': slot_value, 'minor': True}
            else:
                dispatcher.utter_message(text='noted')
                return {'age': slot_value, 'minor': False}
        else:
            dispatcher.utter_message(text='age can only be a number')
            return {'age': None}

    def validate_class(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        print('validating class')
        if slot_value.lower() in self.class_list():
            dispatcher.utter_message(text='noted')
            return {'class': slot_value}
        else:
            dispatcher.utter_message(text='choose a valid class')
            return {'class': None}

    def validate_experience(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:

        print('validating experience')
        age = float(tracker.get_slot('age'))
        pattern = r'^[0-9]+$'
        is_valid_experience = re.search(pattern, slot_value)
        if is_valid_experience:
            experience = float(is_valid_experience.group())
            is_smaller_than_age = experience < age

        if is_valid_experience and is_smaller_than_age:
            dispatcher.utter_message(text='noted')
            return {'experience': slot_value}
        else:
            dispatcher.utter_message(text='experience must be lower than age')
            return {'experience': None}
