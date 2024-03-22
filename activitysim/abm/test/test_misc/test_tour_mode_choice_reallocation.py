import pandas as pd
import pytest

from activitysim.abm.models import tour_mode_choice_reallocation as tmcr

from .setup_utils import setup_dirs


@pytest.fixture(scope="module")
def households():
    households = pd.DataFrame(
        data={
            "auto_ownership": [1] * 2 + [2],
            "num_drivers": [2] * 2 + [3],
        },
        index=range(3),
    )

    households.index.name = "household_id"

    return households


@pytest.fixture(scope="module")
def persons():
    persons = pd.DataFrame(
        data={
            "is_driving_age": [True] * 6,
        },
        index=range(6),
    )

    persons.index.name = "person_id"

    return persons


@pytest.fixture(scope="module")
def tours():
    tours = pd.DataFrame(
        data={
            "household_id": [3] * 3 + [1] * 2 + [2] * 4,
            "person_id": [5, 6, 5, 1, 2, 3, 4, 3, 4],
            "tour_mode": ['DRIVEALONEFREE'] * 9,
            "tour_category": ["joint", "mandatory", "joint"] + ["mandatory"] * 2 + ["joint", "mandatory"] * 2,
            "start": [14, 20, 26, 12, 14, 14, 28, 28, 29],
            "end": [16, 22, 27, 15, 16, 16, 29, 30, 30],
            "mode_choice_logsum": [0] * 9,
        },
        index=range(9),
    )

    tours.index.name = "tour_id"

    return tours


@pytest.fixture(scope="module")
def no_auto_tours():
    no_auto_tours = pd.DataFrame(
        data={
            "household_id": [3] * 3 + [1] * 2 + [2] * 4,
            "person_id": [5, 6, 5, 1, 2, 3, 4, 3, 4],
            "tour_mode": ['WALK_ALLTRN'] * 9,
            "tour_category": ["joint", "mandatory", "joint"] + ["mandatory"] * 2 + ["joint", "mandatory"] * 2,
            "start": [14, 20, 26, 12, 14, 14, 28, 28, 29],
            "end": [16, 22, 27, 15, 16, 16, 29, 30, 30],
            "mode_choice_logsum": [0, 0, -1, -1, 0, 0, -0.5, 0, -1],
        },
        index=range(9),
    )

    no_auto_tours.index.name = "tour_id"

    return no_auto_tours


def test_tour_identification(persons, households, tours):
    setup_dirs()

    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    assert overlapped_tours.groupby('household_id').household_id.max().to_list() == [1, 2]


def test_auto_reallocation_flag(persons, households, tours, no_auto_tours):
    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    reallocated_tours = tmcr.household_auto_reallocation(households, overlapped_tours, no_auto_tours)

    assert reallocated_tours.loc[reallocated_tours.reallocation_flag == 1].index.to_list() == [4, 7]


def test_auto_reallocation(persons, households, tours, no_auto_tours):
    overlapped_tours = tmcr.identify_auto_overallocation(persons, households, tours)

    reallocated_tours = tmcr.household_auto_reallocation(households, overlapped_tours, no_auto_tours)

    assert reallocated_tours.loc[reallocated_tours.reallocation_flag == 1, 'tour_mode'].to_list() == ['WALK_ALLTRN'] * 2
