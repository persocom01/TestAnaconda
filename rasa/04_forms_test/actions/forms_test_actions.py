from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher


class ValidateAdventurerForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_adventurer_form"

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
