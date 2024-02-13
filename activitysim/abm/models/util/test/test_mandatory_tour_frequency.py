# ActivitySim
# See full license in LICENSE.txt.


import os

import pandas as pd
import pandas.testing as pdt
import pytest

from activitysim.abm.models.util.tour_frequency import process_mandatory_tours
from activitysim.core import workflow


@pytest.fixture(scope="module")
def configs_dir():
    return os.path.join(os.path.dirname(__file__), "configs")


def mandatory_tour_frequency_alternatives():
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    f = os.path.join(configs_dir, "mandatory_tour_frequency_alternatives.csv")
    df = pd.read_csv(f, comment="#")
    df.set_index("alt", inplace=True)
    return df


def test_mtf():
    state = workflow.State.make_default(__file__)

    persons = pd.DataFrame(
        {
            "is_worker": [True, True, False, False],
            "mandatory_tour_frequency": [
                "work1",
                "work_and_school",
                "work_and_school",
                "school2",
            ],
            "school_zone_id": [1, 2, 3, 4],
            "workplace_zone_id": [10, 20, 30, 40],
            "home_zone_id": [100, 200, 300, 400],
            "household_id": [1, 2, 2, 4],
        },
        index=[10, 20, 30, 40],
    )

    tour_frequency_alternatives = mandatory_tour_frequency_alternatives()

    mandatory_tours = process_mandatory_tours(
        state, persons, tour_frequency_alternatives
    )

    idx = mandatory_tours.index

    pdt.assert_series_equal(
        mandatory_tours.person_id,
        pd.Series([10, 20, 20, 30, 30, 40, 40], index=idx, name="person_id"),
        check_dtype=False,
    )

    # check if the tour_type variable is pandas categorical
    if isinstance(mandatory_tours.tour_type.dtype, pd.api.types.CategoricalDtype):
        pdt.assert_series_equal(
            mandatory_tours.tour_type.astype(str),
            pd.Series(
                ["work", "work", "school", "work", "school", "school", "school"],
                index=idx,
                name="tour_type",
            ),
        )
    else:
        pdt.assert_series_equal(
            mandatory_tours.tour_type,
            pd.Series(
                ["work", "work", "school", "work", "school", "school", "school"],
                index=idx,
                name="tour_type",
            ),
        )

    # tour_nums for work_and_school non-worker should be flipped
    pdt.assert_series_equal(
        mandatory_tours.tour_num,
        pd.Series([1, 1, 2, 2, 1, 1, 2], index=idx, name="tour_num"),
        check_dtype=False,
    )

    pdt.assert_series_equal(
        mandatory_tours.destination,
        pd.Series([10, 20, 2, 30, 3, 4, 4], index=idx, name="destination"),
        check_dtype=False,
    )

    pdt.assert_series_equal(
        mandatory_tours.origin,
        pd.Series([100, 200, 200, 300, 300, 400, 400], index=idx, name="origin"),
        check_dtype=False,
    )
