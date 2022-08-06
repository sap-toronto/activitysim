# ActivitySim
# See full license in LICENSE.txt.
import os
import subprocess

import pandas as pd
import pandas.testing as pdt
import pkg_resources

from activitysim.core import inject


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def _test_arc(recode=False, sharrow=False):
    def example_path(dirname):
        resource = os.path.join("examples", "prototype_arc", dirname)
        return pkg_resources.resource_filename("activitysim", resource)

    def test_path(dirname):
        return os.path.join(os.path.dirname(__file__), dirname)

    def regress():
        regress_trips_df = pd.read_csv(test_path("regress/final_trips.csv"))
        final_trips_df = pd.read_csv(test_path("output/final_trips.csv"))

        # person_id,household_id,tour_id,primary_purpose,trip_num,outbound,trip_count,purpose,
        # destination,origin,destination_logsum,depart,trip_mode,mode_choice_logsum
        # compare_cols = []
        pdt.assert_frame_equal(final_trips_df, regress_trips_df)

    file_path = os.path.join(os.path.dirname(__file__), "simulation.py")

    if recode:
        run_args = [
            "-c",
            test_path("configs_recode"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    elif sharrow:
        run_args = [
            "-c",
            test_path("configs_sharrow"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]
    else:
        run_args = [
            "-c",
            test_path("configs"),
            "-c",
            example_path("configs"),
            "-d",
            example_path("data"),
            "-o",
            test_path("output"),
        ]

    subprocess.run(["coverage", "run", "-a", file_path] + run_args, check=True)

    regress()


def test_arc():
    _test_arc()


def test_arc_recode():
    _test_arc(recode=True)


def test_arc_sharrow():
    _test_arc(sharrow=True)


if __name__ == "__main__":

    _test_arc()
    _test_arc(recode=True)
    _test_arc(sharrow=True)
