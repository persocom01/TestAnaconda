import datetime as dt
from typing import Optional, Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet

import actions.forms_test_actions as fta

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


@pytest.mark.fta
def test_action_show_time_name():
    assert fta.ActionShowTime().name() == "action_show_time"


@pytest.mark.fta
def test_action_show_time_run():
    dispatcher = CollectingDispatcher()
    output = fta.ActionShowTime().run(
        dispatcher=dispatcher,
        tracker=get_tracker_for_slots('saved_time', ''),
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = r'the time is ([0-9- :\.]*)'
    search = re.search(pattern, message)
    is_message_correct = len(search.group()) == 38

    assert (output == []) and (is_message_correct)


@pytest.mark.fta
def test_action_set_time_name():
    assert fta.ActionSetTime().name() == "action_set_time"


@pytest.mark.fta
def test_action_set_time_run():
    dispatcher = CollectingDispatcher()
    output = fta.ActionSetTime().run(
        dispatcher=dispatcher,
        tracker=get_tracker_for_slots('saved_time', ''),
        domain=None)
    current_time = output[0]['value']
    assert output == [SlotSet('saved_time', current_time)]
