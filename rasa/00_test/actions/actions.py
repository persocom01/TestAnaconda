# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import re

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType


class ActionClearDetails(Action):

    def name(self) -> Text:
        return "action_clear_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Please fill up your personal information in the next few questions.")

        return [
            SlotSet('name', None),
            SlotSet('nric', None),
            SlotSet('mobileNumber', None),
            SlotSet('email', None),
            SlotSet('gender', None),
            SlotSet('race', None),
            SlotSet('nationality', None),
            SlotSet('dob', None)
        ]


class ValidatePersonalDetailForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_personal_detail_form"

    def validate_nric(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate nric."""
        print("Validating NRIC")
        nric_regex_template = '^[SFTG]\d{7}[A-Z]$'
        # except value from slot
        try:
            if type(slot_value) == list:
                slot_value = slot_value[0]
            nric = slot_value.upper() # Convert to uppercase

            # validate NRIC saved in slot to check if it fits the format of a valid NRIC value
            if (re.search(nric_regex_template,nric)):
                print("NRIC validate successfully!")
                return {"nric": nric}
            else:
                dispatcher.utter_message(text="Your NRIC is invalid, please enter the correct NRIC")
                return {"nric": None} # Sets the value of the slot 'nric' to None
        except Exception as e:
            print(e)
            dispatcher.utter_message(text="NRIC validation intent is triggered but action not carried out")

            return {"nric": None}

    def validate_email(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        email_regex_template = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
        # except value from slot
        try:
            email_address = tracker.get_slot("email")

            # validate email saved in slot to check if it fits the format of a valid email address
            if (re.search(email_regex_template,email_address)):
                return {'email': slot_value}
            else:
                dispatcher.utter_message(text="Email Address is invalid, please enter again")
                return {'email': None}

        except:
            dispatcher.utter_message(text="Email validation intent triggered but action not carried out")

        return []

    def validate_mobileNumber(
        self,
        slot_value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        number_regex_template = '^[+]65(6|8|9)\d{7}$'
        try:

            if (re.search(number_regex_template,slot_value)):
                return {'mobileNumber': slot_value}
            else:
                dispatcher.utter_message(text="Phone number is invalid, please enter again")
                return {'mobileNumber': slot_value}

            # validate NRIC

        except:
            dispatcher.utter_message(text="Phone number intent triggered but action not carried out")

        return []
