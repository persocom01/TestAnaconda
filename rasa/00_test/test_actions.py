import datetime as dt
from typing import Optional, Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from rasa_sdk.events import SlotSet

import actions.forms_test_actions as fta
import actions.actions_test_actions as ata

import pytest
import re


@pytest.mark.fta
def test_validate_adventurer_form_name():
    assert fta.ValidateAdventurerForm().name() == "validate_adventurer_form"


@pytest.mark.fta
@pytest.mark.parametrize('input, regex, expected_output', [
    ('Sato Kazuma', r'noted', {'name': 'Sato Kazuma'}),
    ('53', r'names may only contain letters and spaces', {'name': None})
])
def test_validate_adventurer_form_validate_name(input, regex, expected_output):
    slot_value = input
    dispatcher = CollectingDispatcher()
    output = fta.ValidateAdventurerForm().validate_name(
        slot_value=slot_value,
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = regex
    search = re.search(pattern, message)

    assert output == expected_output and search


@pytest.mark.fta
@pytest.mark.parametrize('input, regex, expected_output', [
    ('16', r'noted', {'age': '16', 'minor': False}),
    ('15', r'noted', {'age': '15', 'minor': True}),
    ('sixteen', r'age can only be a number', {'age': None})
])
def test_validate_adventurer_form_validate_age(input, regex, expected_output):
    slot_value = input
    dispatcher = CollectingDispatcher()
    output = fta.ValidateAdventurerForm().validate_age(
        slot_value=slot_value,
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = regex
    search = re.search(pattern, message)

    assert output == expected_output and search


@pytest.mark.fta
@pytest.mark.parametrize('input, regex, expected_output', [
    ('fighter', r'noted', {'class': 'fighter'}),
    ('dark magician', r'choose a valid class', {'class': None})
])
def test_validate_adventurer_form_validate_class(input, regex, expected_output):
    slot_value = input
    dispatcher = CollectingDispatcher()
    output = fta.ValidateAdventurerForm().validate_class(
        slot_value=slot_value,
        dispatcher=dispatcher,
        tracker=Tracker,
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = regex
    search = re.search(pattern, message)

    assert output == expected_output and search


@pytest.mark.fta
@pytest.mark.parametrize('input, slots, regex, expected_output', [
    ('0', {'age': '16'}, r'noted', {'experience': '0'}),
    ('20', {'age': '16'}, r'experience must be lower than age', {'experience': None})
])
def test_validate_adventurer_form_validate_experience(input, slots, regex, expected_output):
    slot_value = input
    dispatcher = CollectingDispatcher()
    output = fta.ValidateAdventurerForm().validate_experience(
        slot_value=slot_value,
        dispatcher=dispatcher,
        tracker=get_tracker_for_slots(slots),
        domain=None)

    message = dispatcher.messages[0]['text']
    pattern = regex
    search = re.search(pattern, message)

    assert output == expected_output and search


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
