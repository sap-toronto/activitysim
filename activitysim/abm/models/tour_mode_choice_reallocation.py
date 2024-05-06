# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
from orca import orca

from activitysim.core import (
    config,
    expressions,
    inject,
    logit,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.pathbuilder import TransitVirtualPathBuilder
from activitysim.core.util import assign_in_place, reindex

from .util import estimation, trip
from .util.mode import run_tour_mode_choice_simulate

logger = logging.getLogger(__name__)

"""
Tour mode choice is run for all tours to determine the transportation mode that
will be used for the tour
"""


def get_alts_from_segmented_nested_logit(model_settings, segment_name, trace_label):
    """
    Infer alts from logit spec

    Parameters
    ----------
    model_settings : dict
    segment_column_name : str
    trace_label : str

    Returns
    -------
    list
    """

    nest_spec = config.get_logit_model_settings(model_settings)
    coefficients = simulate.get_segment_coefficients(model_settings, segment_name)
    nest_spec = simulate.eval_nest_coefficients(nest_spec, coefficients, trace_label)
    tour_mode_alts = []
    for nest in logit.each_nest(nest_spec):
        if nest.is_leaf:
            tour_mode_alts.append(nest.name)

    return tour_mode_alts


def create_logsum_trips(tours, segment_column_name, model_settings, trace_label):
    """
    Construct table of trips from half-tours (1 inbound, 1 outbound) for each tour-mode.

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : dict
    trace_label : str

    Returns
    -------
    pandas.DataFrame
        Table of trips: 2 per tour, with O/D and purpose inherited from tour
    """
    stop_frequency_alts = inject.get_injectable("stop_frequency_alts")
    stop_freq = "0out_0in"  # no intermediate stops
    tours["stop_frequency"] = stop_freq
    tours["primary_purpose"] = tours["tour_purpose"]
    trips = trip.initialize_from_tours(tours, stop_frequency_alts)
    trips["stop_frequency"] = stop_freq
    outbound = trips["outbound"]
    trips["depart"] = reindex(tours.start, trips.tour_id)
    trips.loc[~outbound, "depart"] = reindex(tours.end, trips.loc[~outbound, "tour_id"])

    # actual segment doesn't matter. just need to grab one
    # to get a set of coefficients from the spec
    segment_name = tours.iloc[0][segment_column_name]
    tour_mode_alts = get_alts_from_segmented_nested_logit(
        model_settings, segment_name, trace_label
    )

    # repeat rows from the trips table iterating over tour mode
    logsum_trips = pd.DataFrame()
    for tour_mode in tour_mode_alts:
        trips["tour_mode"] = tour_mode
        logsum_trips = pd.concat((logsum_trips, trips), ignore_index=True)
    assert len(logsum_trips) == len(trips) * len(tour_mode_alts)
    logsum_trips.index.name = "trip_id"

    return logsum_trips


def append_tour_leg_trip_mode_choice_logsums(tours):
    """
    Creates trip mode choice logsum column in tours table for each tour mode and leg

    Parameters
    ----------
    tours : pd.DataFrame

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """
    trips = inject.get_table("trips").to_frame()
    trip_dir_mode_logsums = trips.pivot(
        index="tour_id",
        columns=["tour_mode", "outbound"],
        values="trip_mode_choice_logsum",
    )
    new_cols = [
        "_".join(["logsum", mode, "outbound" if outbound else "inbound"])
        for mode, outbound in trip_dir_mode_logsums.columns
    ]
    trip_dir_mode_logsums.columns = new_cols
    trip_dir_mode_logsums.reindex(tours.index)
    tours = pd.merge(tours, trip_dir_mode_logsums, left_index=True, right_index=True)

    return tours


def get_trip_mc_logsums_for_all_modes(tours, segment_column_name, model_settings, trace_label):
    """
    Creates pseudo-trips from tours and runs trip mode choice to get logsums

    Parameters
    ----------
    tours : pandas.DataFrame
    segment_column_name : str
        column in tours table used for segmenting model spec
    model_settings : dict
    trace_label : str

    Returns
    -------
    tours : pd.DataFrame
        Adds two * n_modes logsum columns to each tour row, e.g. "logsum_DRIVE_outbound"
    """

    # create pseudo-trips from tours for all tour modes
    logsum_trips = create_logsum_trips(
        tours, segment_column_name, model_settings, trace_label
    )

    # temporarily register trips in the pipeline
    pipeline.replace_table("trips", logsum_trips)
    tracing.register_traceable_table("trips", logsum_trips)
    pipeline.get_rn_generator().add_channel("trips", logsum_trips)

    # run trip mode choice on pseudo-trips. use orca instead of pipeline to
    # execute the step because pipeline can only handle one open step at a time
    orca.run(["trip_mode_choice"])

    # add trip mode choice logsums as new cols in tours
    tours = append_tour_leg_trip_mode_choice_logsums(tours)

    # de-register logsum trips table
    pipeline.get_rn_generator().drop_channel("trips")
    tracing.deregister_traceable_table("trips")

    return tours


def code_household_auto_sufficiency(households):
    """
    Adds and codes household auto sufficiency based on household auto ownership and number of drivers in the household

    Parameters
    ----------
    households : pd.Dataframe

    Returns
    -------
    households : pd.Dataframe
        Household table with auto sufficiency column added

    """
    households['auto_sufficiency'] = 'suff'
    households.loc[
        households.auto_ownership.between(0, households.num_drivers, 'neither'), 'auto_sufficiency'] = 'insuff'
    households.loc[households.auto_ownership == 0, 'auto_sufficiency'] = 'nocar'

    return households


def filter_for_multi_drive_tour_households(households, persons, tours):
    """
    Filters tours down to only those from auto insufficient households with multiple drivers who made SOV-based tours

    Parameters
    ----------
    households : pd.Dataframe
    persons : pd.Dataframe
    tours : pd.Dataframe

    Returns
    -------
    tours_insuff_mult_drv : pd.Dataframe
        Tours from auto insufficient households with multiple drivers who made SOV-based tours

    """
    households = code_household_auto_sufficiency(households)

    # Isolate for SOV-based tours by driving aged persons in auto insufficient households.
    households_insuff = households[households.auto_sufficiency == 'insuff']
    person_drivers = persons[persons.is_driving_age]
    # TODO: Expand criteria to cover joint tours once a reallocation method can be developed for it
    tours_insuff_drv = tours[
        tours.household_id.isin(households_insuff.index) &
        tours.person_id.isin(person_drivers.index) &
        (tours.tour_mode.isin(['DRIVEALONEFREE', 'DRIVEALONEPAY']))]

    # Get per household count of driver who made SOV-based tours in auto insufficient households
    unique_pers_insuff_drv = tours_insuff_drv[['household_id', 'person_id']].drop_duplicates()

    count_pers_insuff_drv = unique_pers_insuff_drv.groupby('household_id').person_id.count()
    count_pers_insuff_drv = count_pers_insuff_drv[count_pers_insuff_drv > 1]

    # Isolate for tours from auto insufficient households with multiple drivers who made SOV-based tours
    tours_insuff_mult_drv = tours_insuff_drv[tours_insuff_drv.household_id.isin(count_pers_insuff_drv.index)]
    tours_insuff_mult_drv = tours_insuff_mult_drv.sort_values(['household_id', 'start', 'end'])

    return tours_insuff_mult_drv


def create_time_grid(filtered_tours, num_intervals):
    """
    Creates per tour time grid flagging the 30-min intervals each tour occupies

    Parameters
    ----------
    filtered_tours : pd.Dataframe
        Tours from auto insufficient households with multiple drivers who made SOV-based tours
    num_intervals: int
        Number of time intervals in a day

    Returns
    -------
    tour_time_grid : pd.Dataframe
        Per tour table with a grid of 48 30-min intervals flagged based on tour start and end time

    """
    tour_time_grid = pd.DataFrame({'household_id': filtered_tours['household_id']})
    for i in range(1, num_intervals+1):
        tour_time_grid[i] = False
        interval_mask = (filtered_tours['start'] <= i) & (filtered_tours['end'] >= i)
        tour_time_grid.loc[interval_mask, i] = True

    return tour_time_grid


def find_auto_overallocations(tours, auto_ownership, num_intervals):
    """
    Identifies overlapped auto_based tours that exceed the number of cars owned by the single household

    Parameters
    ----------
    tours : pd.Dataframe
        Expecting auto-based tours from a single household
    auto_ownership : int
    num_intervals: int
        Number of time intervals in a day

    Returns
    -------
    overallocated_tours : pd.Dataframe
        Identified overlapped auto_based tours that exceed the number of cars owned by the single household

    """
    # Find which intervals have overlaps greater than the number of autos the household owns
    overallocation_condition = tours[[*range(1, num_intervals+1)]].sum() > auto_ownership
    overallocated_intervals = tours[[*range(1, num_intervals+1)]].loc[:, overallocation_condition]

    overallocated_tours = overallocated_intervals[overallocated_intervals.any(axis=1)]

    return overallocated_tours


def identify_auto_overallocations(persons, households, tours, num_intervals):
    """
    Identifies all overallocated tours

    Parameters
    ----------
    persons : pd.Dataframe
    households : pd.Dataframe
    tours : pd.Dataframe
    num_intervals: int
        Number of time intervals in a day

    Returns
    -------
    tours_overallocated : pd.Dataframe
        Identified overallocated tours

    """
    # Isolate for tours from auto insufficient households with multiple drivers who made SOV-based tours
    filtered_tours = filter_for_multi_drive_tour_households(households, persons, tours)

    tour_time_grid = create_time_grid(filtered_tours, num_intervals)

    overlap_tour_ids = []
    for name, group in tour_time_grid.groupby('household_id'):
        # get the auto ownership of the household being evaluated
        auto_ownership = households.loc[name, 'auto_ownership']
        overlap_tours = find_auto_overallocations(group, auto_ownership, num_intervals)
        overlap_tour_ids.extend(overlap_tours.index)

    # Isolate for only households with the identified tour overlaps
    tours_overallocated = filtered_tours.loc[overlap_tour_ids].copy()

    # Assert that only auto-based tours are tagged
    assert (tours_overallocated.tour_mode.isin(['DRIVEALONEFREE', 'DRIVEALONEPAY'])).all()
    # Assert that all tagged households has been tagged with multiple tours
    assert tours_overallocated.groupby('household_id').size().min() > 1

    return tours_overallocated


def household_auto_reallocation(households, tours_to_realloc, no_auto_tours):
    """
    Reallocates some overallocated tours to the best non-auto-based alternative based on their loss of utility

    Parameters
    ----------
    households : pd.Dataframe
    tours_to_realloc : pd.Dataframe

    no_auto_tours : pd.Dataframe
        overallocated tours with best non-auto-based tour mode alternative

    Returns
    -------
    reallocated_tours : pd.Dataframe
        Reallocated tours with new non-auto-based tour modes

    """
    output_columns = tours_to_realloc.columns.union(['reallocation_flag'])
    # Join in the non-auto utility/logsum and alternative mode choice from no-auto run
    tours_to_realloc['no_auto_mc_logsum'] = no_auto_tours.mode_choice_logsum
    tours_to_realloc['no_auto_tour_mode'] = no_auto_tours.tour_mode
    # Calculate loss of utility from losing auto mode
    tours_to_realloc['no_auto_logsum_loss'] = tours_to_realloc.mode_choice_logsum - tours_to_realloc.no_auto_mc_logsum
    # Initialize reallocation flag
    tours_to_realloc['reallocation_flag'] = False

    # Find the optimal allocation by sorting based on loss of tour mode choice utility and prioritizing tours where the
    # difference is the greatest.
    reallocated_groups = []
    for name, group in tours_to_realloc.groupby('household_id'):
        # Sort by utility loss in descending order
        group.sort_values(by='no_auto_logsum_loss', ascending=False, inplace=True)
        # flag bottom tours that exceeds number of vehicles in household for reallocation
        auto_ownership = households.loc[name, 'auto_ownership']
        group.iloc[auto_ownership:, group.columns.get_loc("reallocation_flag")] = True
        reallocated_groups.append(group)
    reallocated_tours = pd.concat(reallocated_groups)

    # Allocate new alternative mode
    reallocated_tours.loc[reallocated_tours.reallocation_flag, 'tour_mode'] = reallocated_tours.no_auto_tour_mode
    reallocated_tours.loc[
        reallocated_tours.reallocation_flag, 'mode_choice_logsum'] = reallocated_tours.no_auto_mc_logsum

    # Assert reallocated tours no longer uses auto-based modes
    assert (~reallocated_tours.loc[reallocated_tours.reallocation_flag, 'tour_mode'].isin(
        ['DRIVEALONEFREE', 'DRIVEALONEPAY'])).all()

    return reallocated_tours[output_columns]


@inject.step()
def tour_mode_choice_reallocation_simulate(
        households, persons, tours, persons_merged, network_los, tdd_alts, chunk_size, trace_hh_id
):
    """
    Tour mode choice auto reallocation simulate
    """
    tour_mode_choice_settings = config.read_model_settings('tour_mode_choice.yaml')

    if tour_mode_choice_settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME") is None:
        raise Exception('tour mode choice logsum column required from tour mode choice model. '
                        'Define column in tour_mode_choice.yaml')

    trace_label = "tour_mode_choice_reallocation"
    model_settings_file_name = "tour_mode_choice_reallocation.yaml"
    model_settings = config.read_model_settings(model_settings_file_name)

    logsum_column_name = model_settings.get("MODE_CHOICE_LOGSUM_COLUMN_NAME")
    old_tour_mode_column_name = model_settings.get("OLD_TOUR_MODE_COLUMN_NAME")
    old_logsum_column_name = model_settings.get("OLD_MODE_CHOICE_LOGSUM_COLUMN_NAME")
    mode_column_name = "tour_mode"
    segment_column_name = "tour_purpose"

    if logsum_column_name is None:
        raise Exception('tour mode choice logsum column needed for tour auto mode reallocation. '
                        'Define column in tour_mode_choice_reallocation.yaml')

    households = households.to_frame()
    persons = persons.to_frame()
    primary_tours = tours.to_frame()
    assert not (primary_tours.tour_category == "atwork").any()

    logger.info("Running %s with %d tours" % (trace_label, primary_tours.shape[0]))

    tracing.print_summary("tour_types", primary_tours.tour_type, value_counts=True)

    persons_merged = persons_merged.to_frame()
    primary_tours_merged = pd.merge(
        primary_tours,
        persons_merged,
        left_on="person_id",
        right_index=True,
        how="left",
        suffixes=("", "_r"),
    )

    num_intervals = tdd_alts.start.max()
    overallocated_tours = identify_auto_overallocations(persons, households, primary_tours_merged, num_intervals)

    constants = {}
    # model_constants can appear in expressions
    constants.update(config.get_model_constants(model_settings))

    skim_dict = network_los.get_default_skim_dict()

    # setup skim keys
    orig_col_name = "home_zone_id"
    dest_col_name = "destination"

    out_time_col_name = "start"
    in_time_col_name = "end"
    odt_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="out_period"
    )
    dot_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="in_period"
    )
    odr_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=orig_col_name, dest_key=dest_col_name, dim3_key="in_period"
    )
    dor_skim_stack_wrapper = skim_dict.wrap_3d(
        orig_key=dest_col_name, dest_key=orig_col_name, dim3_key="out_period"
    )
    od_skim_stack_wrapper = skim_dict.wrap(orig_col_name, dest_col_name)

    skims = {
        "odt_skims": odt_skim_stack_wrapper,
        "dot_skims": dot_skim_stack_wrapper,
        "odr_skims": odr_skim_stack_wrapper,  # dot return skims for e.g. TNC bridge return fare
        "dor_skims": dor_skim_stack_wrapper,  # odt return skims for e.g. TNC bridge return fare
        "od_skims": od_skim_stack_wrapper,
        "orig_col_name": orig_col_name,
        "dest_col_name": dest_col_name,
        "out_time_col_name": out_time_col_name,
        "in_time_col_name": in_time_col_name,
    }

    if network_los.zone_system == los.THREE_ZONE:
        # fixme - is this a lightweight object?

        tvpb = network_los.tvpb

        tvpb_logsum_odt = tvpb.wrap_logsum(
            orig_key=orig_col_name,
            dest_key=dest_col_name,
            tod_key="out_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_odt",
        )
        tvpb_logsum_dot = tvpb.wrap_logsum(
            orig_key=dest_col_name,
            dest_key=orig_col_name,
            tod_key="in_period",
            segment_key="demographic_segment",
            cache_choices=True,
            trace_label=trace_label,
            tag="tvpb_logsum_dot",
        )

        skims.update(
            {"tvpb_logsum_odt": tvpb_logsum_odt, "tvpb_logsum_dot": tvpb_logsum_dot}
        )

        # TVPB constants can appear in expressions
        if model_settings.get("use_TVPB_constants", True):
            constants.update(
                network_los.setting("TVPB_SETTINGS.tour_mode_choice.CONSTANTS")
            )

    # don't create estimation data bundle if trip mode choice is being called
    # from another model step (i.e. tour mode choice logsum creation)
    if pipeline.get_rn_generator().step_name != "tour_mode_choice_reallocation_simulate":
        estimator = None
    else:
        estimator = estimation.manager.begin_estimation("tour_mode_choice")
    if estimator:
        estimator.write_coefficients(model_settings=model_settings)
        estimator.write_coefficients_template(model_settings=model_settings)
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        # (run_tour_mode_choice_reallocation_simulate writes choosers post-annotation)

    # FIXME should normalize handling of tour_type and tour_purpose
    # mtctm1 school tour_type includes univ, which has different coefficients from elementary and HS
    # we should either add this column when tours created or add univ to tour_types
    not_university = (overallocated_tours.tour_type != "school") | ~(
        overallocated_tours.is_university.astype(bool)
        if "is_university" in overallocated_tours.columns
        else False
    )
    overallocated_tours["tour_purpose"] = overallocated_tours.tour_type.where(
        not_university, "univ"
    )

    # if trip logsums are used, run trip mode choice and append the logsums
    if model_settings.get("COMPUTE_TRIP_MODE_CHOICE_LOGSUMS", False):
        overallocated_tours = get_trip_mc_logsums_for_all_modes(
            overallocated_tours, segment_column_name, model_settings, trace_label
        )

    #
    choices_list = []
    for tour_purpose, tours_segment in overallocated_tours.groupby(
            segment_column_name
    ):

        logger.info(
            "tour_mode_choice_simulate tour_type '%s' (%s tours)"
            % (
                tour_purpose,
                len(tours_segment.index),
            )
        )

        if network_los.zone_system == los.THREE_ZONE:
            tvpb_logsum_odt.extend_trace_label(tour_purpose)
            tvpb_logsum_dot.extend_trace_label(tour_purpose)

        # name index so tracing knows how to slice
        assert tours_segment.index.name == "tour_id"

        choices_df = run_tour_mode_choice_simulate(
            tours_segment,
            tour_purpose,
            model_settings,
            mode_column_name=mode_column_name,
            logsum_column_name=logsum_column_name,
            network_los=network_los,
            skims=skims,
            constants=constants,
            estimator=estimator,
            chunk_size=chunk_size,
            trace_label=tracing.extend_trace_label(trace_label, tour_purpose),
            trace_choice_name="tour_mode_choice",
        )

        tracing.print_summary(
            "tour_mode_choice_simulate %s choices_df" % tour_purpose,
            choices_df.tour_mode,
            value_counts=True,
        )

        choices_list.append(choices_df)

    choices_df = pd.concat(choices_list)

    # add cached tvpb_logsum tap choices for modes specified in tvpb_mode_path_types
    if network_los.zone_system == los.THREE_ZONE:

        tvpb_mode_path_types = model_settings.get("tvpb_mode_path_types")
        if tvpb_mode_path_types is not None:
            for mode, path_types in tvpb_mode_path_types.items():

                for direction, skim in zip(
                        ["od", "do"], [tvpb_logsum_odt, tvpb_logsum_dot]
                ):

                    path_type = path_types[direction]
                    skim_cache = skim.cache[path_type]

                    print(f"mode {mode} direction {direction} path_type {path_type}")

                    for c in skim_cache:

                        dest_col = f"{direction}_{c}"

                        if dest_col not in choices_df:
                            choices_df[dest_col] = (
                                np.nan
                                if pd.api.types.is_numeric_dtype(skim_cache[c])
                                else ""
                            )
                        choices_df[dest_col].where(
                            choices_df.tour_mode != mode, skim_cache[c], inplace=True
                        )

    if estimator:
        estimator.write_choices(choices_df.tour_mode)
        choices_df.tour_mode = estimator.get_survey_values(
            choices_df.tour_mode, "tours", "tour_mode"
        )
        estimator.write_override_choices(choices_df.tour_mode)
        estimator.end_estimation()

    tracing.print_summary(
        "tour_mode_choice_simulate all tour type choices",
        choices_df.tour_mode,
        value_counts=True,
    )

    # so we can trace with annotations
    assign_in_place(primary_tours, choices_df)

    # update tours table with mode choice (and optionally logsums)
    all_tours = tours.to_frame()
    tours_to_realloc = overallocated_tours[all_tours.columns].copy()

    all_tours['reallocation_flag'] = False
    if old_tour_mode_column_name is not None:
        all_tours[old_tour_mode_column_name] = all_tours.tour_mode

    if old_logsum_column_name is not None:
        all_tours[old_logsum_column_name] = all_tours.mode_choice_logsum

    reallocated_tours = household_auto_reallocation(households, tours_to_realloc, choices_df)

    assign_in_place(all_tours, reallocated_tours)

    pipeline.replace_table("tours", all_tours)

    # - annotate tours table
    if model_settings.get("annotate_tours"):
        tours = inject.get_table("tours").to_frame()
        expressions.assign_columns(
            df=tours,
            model_settings=model_settings.get("annotate_tours"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_tours"),
        )
        pipeline.replace_table("tours", tours)

    if trace_hh_id:
        tracing.trace_df(
            primary_tours,
            label=tracing.extend_trace_label(trace_label, mode_column_name),
            slicer="tour_id",
            index_label="tour_id",
            warn_if_empty=True,
        )
