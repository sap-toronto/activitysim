import pandas as pd
import pytest

from activitysim.abm.models import tour_mode_choice_reallocation as tmcr

from .setup_utils import setup_dirs


@pytest.fixture(scope="module")
def households():
    households = pd.DataFrame(
        data={
            "auto_ownership": [2, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "num_drivers": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "test_case_desc": ['auto sufficient', 'no driver', 'auto sufficient', 'no SOV tour', 'single tour',
                               'single SOV tour', 'no overlap', '2 tour overlap', '3 tour single overlap',
                               '3 tour double overlap']
        },
        index=range(1, 11),
    )

    households.index.name = "household_id"

    return households


@pytest.fixture(scope="module")
def persons():
    persons = pd.DataFrame(
        data={
            "is_driving_age": [True] * 2 + [False] * 2 + [True, False] + [True] * 14,
        },
        index=range(1, 21),
    )

    persons.index.name = "person_id"

    return persons


@pytest.fixture(scope="module")
def tours():
    sov = 'DRIVEALONEFREE'
    walk = 'WALK'

    tours = pd.DataFrame(
        data={
            "household_id": [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2 + [5] * 1 + [6] * 2 + [7] * 2 + [8] * 2 + [9] * 3 + [
                10] * 3,
            "person_id": [*range(9)] + [*range(10, 16)] + [17, 18, 17] + [19, 20, 19],
            "tour_mode": [sov] * 6 + [walk] * 2 + [sov] + [sov, walk] + [sov] * 2 + [sov] * 2 + [sov] * 3 + [sov] * 3,
            "tour_category": ["mandatory"] * 21,
            "start": [14, 15] * 4 + [14] + [14, 15] + [14, 16] + [14, 15] + [14, 15, 16] + [14, 15, 16],
            "end": [15, 17] * 4 + [16] + [15, 17] + [15, 17] + [15, 16] + [15, 16, 17] + [17, 16, 17],
            "mode_choice_logsum": [0] * 21
        },
        index=range(1, 22),
    )

    tours['old_tour_mode'] = tours['tour_mode']
    tours['old_mode_choice_logsum'] = tours['mode_choice_logsum']

    tours.index.name = "tour_id"

    return tours


@pytest.fixture(scope="module")
def no_auto_tours(tours):
    no_auto_tours = tours.copy()
    no_auto_tours['tour_mode'] = 'WALK_ALLTRN'
    no_auto_tours['mode_choice_logsum'] = [0, -1] * 4 + [0] + [0, -1] * 3 + [0, -1, -0.5] * 2

    return no_auto_tours


def test_tour_identification(persons, households, tours):
    setup_dirs()

    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    assert overlapped_tours.groupby('household_id').household_id.max().to_list() == [8, 9, 10]


def test_auto_reallocation_flag(persons, households, tours, no_auto_tours):
    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    reallocated_tours = tmcr.household_auto_reallocation(households, overlapped_tours, no_auto_tours)

    realloc_mask = reallocated_tours.tour_mode != reallocated_tours.old_tour_mode

    answer = {14, 16, 18, 19, 21}

    assert set(reallocated_tours.loc[realloc_mask].index) - answer == set()


def test_auto_reallocation(persons, households, tours, no_auto_tours):
    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    reallocated_tours = tmcr.household_auto_reallocation(households, overlapped_tours, no_auto_tours)

    realloc_mask = reallocated_tours.tour_mode != reallocated_tours.old_tour_mode

    assert reallocated_tours.loc[realloc_mask, 'tour_mode'].to_list() == ['WALK_ALLTRN'] * 5
