import datetime as dt
from typing import Optional, Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet

import actions.actions_test_actions as ata

import pytest
import re


def get_tracker_for_slots(
    slots: Dict,
    latest_msg={},
    followup_action=[]
):

    return Tracker(
            "default",
            slots,
            latest_msg,
            followup_action,
            False,
            None,
            {},
            "action_listen",
        )


@pytest.mark.ata
def test_action_show_time_name():
    assert ata.ActionShowTime().name() == "action_show_time"


@pytest.mark.ata
def test_action_show_time_run():
    dispatcher = CollectingDispatcher()
    output = ata.ActionShowTime().run(
        dispatcher=dispatcher,
        tracker=get_tracker_for_slots({'saved_time': ''}),
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = r'^numerical timestamp: ([0-9- :\.]*)$'
    search = re.search(pattern, message)
    message_var = search.groups()[0]
    is_message_correct = len(message_var) == 17

    assert (output == []) and is_message_correct


@pytest.mark.ata
def test_action_set_time_name():
    assert ata.ActionSetTime().name() == "action_set_time"


@pytest.mark.ata
def test_action_set_time_run():
    dispatcher = CollectingDispatcher()
    output = ata.ActionSetTime().run(
        dispatcher=dispatcher,
        tracker=get_tracker_for_slots({'saved_time': ''}),
        domain=None)
    slot_value = output[0]['value']
    is_slot_value_correct = len(str(slot_value)) == 26
    assert (output == [SlotSet('saved_time', slot_value)]) and is_slot_value_correct
