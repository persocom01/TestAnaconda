import datetime as dt
from typing import Optional, Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet, FollowupAction

import pytest
import re

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import actions.actions_test_actions as ata


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
    assert ata.ActionShowTime().name() == 'action_show_time'


@pytest.mark.ata
def test_action_show_time_run():
    dispatcher = CollectingDispatcher()
    tracker = Tracker
    output = ata.ActionShowTime().run(
        dispatcher=dispatcher,
        tracker=tracker,
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = r'^numerical timestamp: ([0-9- :\.]*)$'
    search = re.search(pattern, message)
    message_var = search.groups()[0]
    is_message_correct1 = len(message_var) == 17

    message = dispatcher.messages[1]['saved_time']
    pattern = r'^[0-9- :\.]*$'
    search = re.search(pattern, message)
    message_var = search.group()
    is_message_correct2 = len(message_var) == 26

    is_message_all_correct = all([is_message_correct1, is_message_correct2])

    assert (output == []) and is_message_all_correct


@pytest.mark.ata
def test_action_set_time_name():
    assert ata.ActionSetTime().name() == 'action_set_time'


@pytest.mark.ata
def test_action_set_time_run():
    dispatcher = CollectingDispatcher()
    tracker = Tracker
    output = ata.ActionSetTime().run(
        dispatcher=dispatcher,
        tracker=tracker,
        domain=None)
    slot_value = output[0]['value']
    is_slot_value_correct = len(str(slot_value)) == 26
    assert (output == [SlotSet('saved_time', slot_value)]) and is_slot_value_correct


@pytest.mark.ata
def test_action_saved_time_name():
    assert ata.ActionSavedTime().name() == 'action_saved_time'


@pytest.mark.ata
def test_action_saved_time_run():
    saved_time = '2021-06-04 08:33:24.718508'
    dispatcher = CollectingDispatcher()
    tracker = get_tracker_for_slots({'saved_time': saved_time})
    output = ata.ActionSavedTime().run(
        dispatcher=dispatcher,
        tracker=tracker,
        domain=None)
    message = dispatcher.messages[0]['saved_time']
    is_message_correct = message == saved_time
    assert (output == []) and is_message_correct


@pytest.mark.ata
def test_action_last_intent_name():
    assert ata.ActionLastIntent().name() == 'action_last_intent'


@pytest.mark.ata
@pytest.mark.asyncio
@pytest.mark.parametrize('intent, expected_output', [
    ('exit', [FollowupAction('utter_goodbye')]),
    ('set_time', [FollowupAction('action_set_time')])
])
async def test_action_last_intent_run(intent, expected_output):
    dispatcher = CollectingDispatcher()
    tracker = get_tracker_for_slots({}, {'intent_ranking': [{'name': intent, 'confidence': 1}]})
    output = await ata.ActionLastIntent().run(
        dispatcher=dispatcher,
        tracker=tracker,
        domain=None)

    assert output == expected_output
