# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd

from activitysim.abm.tables import shadow_pricing
from activitysim.core import (
    config,
    expressions,
    inject,
    los,
    pipeline,
    simulate,
    tracing,
)
from activitysim.core.interaction_sample import interaction_sample
from activitysim.core.interaction_sample_simulate import interaction_sample_simulate

from .util import estimation
from .util import logsums as logsum
from .util import tour_destination

# import multiprocessing


"""
The school/workplace location model predicts the zones in which various people will
work or attend school.

For locations choices like workplace and school location, we have existing data about the actual
number of workers or students in the various destination zones, and we naturally want the results
of location choice to yield distributions the match these observed distributions as closely as
possible. To achieve this, we use start with size tables with the observed populations by zone
and segment (e.g. number of university, highschool, and gradeschool students in each zone) and
use those populations as attractors (positive utilities) so that high population zones will,
all things being equal, receive more choices. (For instance, we want university-goers to choose
school locations with in zones with university enrollments.)

But since the choice algorithm can result in aggregate distributions of choices (modeled_size)
that don't match observed (predicted_size) counts. The shadow pricing algorithm attempts to
correct these misalignments, by iteratively running the choice model, comparing the modeled_size
of the zones segments to the  predicted size, and computing a shadow_price coefficient that is
applied to the size term to boost or attenuate its influence. This iterative process can be
configures to continue until a specified closeness of fit is achieved, or a maximum number of
iterations has occurred. Since the iterative process can be expensive, a facility is provided
to save the computed shadow prices after every iteration, and to load pre-computed shadow prices
on subsequent runs (warm start) to cut down on runtimes.

Since every individual (always person for now) belongs to at most one segment, each segment
(e.g. 'university', 'highschool' , 'gradeschool' for the 'school' location model) is handled
separately and sequentially withing each shadow-price iteration.

The core algorithm has 3 parts:

Because logsum calculations are expensive, rather than computing logsums for all destination
alternatives, we first build a sample of alternatives using simplified (no-logsum) utilities,
and compute logsums only for that sample, and finally chose from among the sampled alternatives.

* run_location_sample - Build a sample destination alternatives using simplified choice criteria
* run_location_logsums - Compute logsums for travel to those alternatives
* run_location_simulate - Rerun the choice model using the logsums to make a final location choice

With shadow pricing, and iterative treatment of each segment, the structure of the code is:

::

    repeat
        for each segment
            run_location_sample
            run_location_logsums
            run_location_simulate
    until convergence
"""

logger = logging.getLogger(__name__)

# column name of logsum in df returned by run_location_logsums (here because used in more than one place)
ALT_LOGSUM = "mode_choice_logsum"


def write_estimation_specs(estimator, model_settings, settings_file):
    """
    write sample_spec, spec, and coefficients to estimation data bundle

    Parameters
    ----------
    model_settings
    settings_file
    """

    estimator.write_model_settings(model_settings, settings_file)
    # estimator.write_spec(model_settings, tag='SAMPLE_SPEC')
    estimator.write_spec(model_settings, tag="SPEC")
    estimator.write_coefficients(model_settings=model_settings)

    estimator.write_table(
        inject.get_injectable("size_terms"), "size_terms", append=False
    )
    estimator.write_table(
        inject.get_table("land_use").to_frame(), "landuse", append=False
    )


def _location_sample(
    segment_name,
    choosers,
    alternatives,
    skims,
    estimator,
    model_settings,
    alt_dest_col_name,
    chunk_size,
    chunk_tag,
    trace_label,
    zone_layer=None,
):
    """
    select a sample of alternative locations.

    Logsum calculations are expensive, so we build a table of persons * all zones
    and then select a sample subset of potential locations

    The sample subset is generated by making multiple choices (<sample_size> number of choices)
    which results in sample containing up to <sample_size> choices for each choose (e.g. person)
    and a pick_count indicating how many times that choice was selected for that chooser.)

    person_id,  dest_zone_id, rand,            pick_count
    23750,      14,           0.565502716034,  4
    23750,      16,           0.711135838871,  6
    ...
    23751,      12,           0.408038878552,  1
    23751,      14,           0.972732479292,  2
    """
    assert not choosers.empty

    logger.info("Running %s with %d persons" % (trace_label, len(choosers.index)))

    sample_size = model_settings["SAMPLE_SIZE"]
    if config.setting("disable_destination_sampling", False) or (
        estimator and estimator.want_unsampled_alternatives
    ):
        # FIXME interaction_sample will return unsampled complete alternatives with probs and pick_count
        logger.info(
            "Estimation mode for %s using unsampled alternatives short_circuit_choices"
            % (trace_label,)
        )
        sample_size = 0

    locals_d = {
        "skims": skims,
        "segment_size": segment_name,
        "orig_col_name": skims.orig_key,  # added for sharrow flows
        "dest_col_name": skims.dest_key,  # added for sharrow flows
        "timeframe": "timeless",
    }
    constants = config.get_model_constants(model_settings)
    locals_d.update(constants)

    spec = simulate.spec_for_segment(
        model_settings,
        spec_id="SAMPLE_SPEC",
        segment_name=segment_name,
        estimator=estimator,
    )

    # here since presumably we want this when called for either sample or presample
    log_alt_losers = config.setting("log_alt_losers", False)

    choices = interaction_sample(
        choosers,
        alternatives,
        spec=spec,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        zone_layer=zone_layer,
    )

    return choices


def location_sample(
    segment_name,
    persons_merged,
    network_los,
    dest_size_terms,
    estimator,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_label,
):

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings["SIMULATE_CHOOSER_COLUMNS"]
    choosers = persons_merged[chooser_columns]

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap("home_zone_id", "zone_id")

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    choices = _location_sample(
        segment_name,
        choosers,
        dest_size_terms,
        skims,
        estimator,
        model_settings,
        alt_dest_col_name,
        chunk_size,
        chunk_tag,
        trace_label,
    )

    return choices


DEST_TAZ = "dest_TAZ"
HOME_TAZ = "TAZ"
HOME_MAZ = "home_zone_id"
DEST_MAZ = "dest_MAZ"


def aggregate_size_terms(dest_size_terms, network_los, model_settings):
    #
    # aggregate MAZ_size_terms to TAZ_size_terms
    #

    MAZ_size_terms = dest_size_terms.copy()

    # add crosswalk DEST_TAZ column to MAZ_size_terms
    MAZ_size_terms[DEST_TAZ] = network_los.map_maz_to_taz(MAZ_size_terms.index)

    MAZ_size_terms["avail_MAZ"] = np.where(
        (MAZ_size_terms.size_term > 0)
        & (MAZ_size_terms.shadow_price_utility_adjustment > -999),
        1,
        0,
    )

    weighted_average_cols = [
        "shadow_price_size_term_adjustment",
        "shadow_price_utility_adjustment",
    ]
    for c in weighted_average_cols:
        MAZ_size_terms[c] *= MAZ_size_terms["size_term"]  # weighted average

    TAZ_size_terms = MAZ_size_terms.groupby(DEST_TAZ).agg(
        {
            "size_term": "sum",
            "shadow_price_size_term_adjustment": "sum",
            "shadow_price_utility_adjustment": "sum",
        }
    )

    for c in weighted_average_cols:
        TAZ_size_terms[c] /= TAZ_size_terms["size_term"]  # weighted average

    spc = shadow_pricing.load_shadow_price_calculator(model_settings)
    if spc.use_shadow_pricing and (
        spc.shadow_settings["SHADOW_PRICE_METHOD"] == "simulation"
    ):
        # allow TAZs with at least one underassigned MAZ in them, therefore with a shadowprice larger than -999, to be selected again
        TAZ_size_terms["shadow_price_utility_adjustment"] = np.where(
            (TAZ_size_terms["shadow_price_utility_adjustment"] > -999)
            & (TAZ_size_terms["avail_MAZ"] > 0),
            0,
            -999,
        )
        # now, negative size term means shadow price is -999. Setting size_term to 0 so the prob of that MAZ being selected becomes 0
        MAZ_size_terms["size_term"] = np.where(
            MAZ_size_terms["shadow_price_utility_adjustment"] < 0,
            0,
            MAZ_size_terms["size_term"],
        )

    if TAZ_size_terms.isna().any(axis=None):
        logger.warning(
            f"TAZ_size_terms with NAN values\n{TAZ_size_terms[TAZ_size_terms.isna().any(axis=1)]}"
        )
        assert not TAZ_size_terms.isna(axis=None).any()

    # print(f"TAZ_size_terms\n{TAZ_size_terms}")
    #           size_term  shadow_price_size_term_adjustment  shadow_price_utility_adjustment
    # dest_TAZ
    # 2             1.419                                1.0                                0
    # 3            20.511                                1.0                                0
    # 4            19.737                                1.0                                0

    MAZ_size_terms = MAZ_size_terms[[DEST_TAZ, "size_term"]].reset_index(drop=False)
    MAZ_size_terms = MAZ_size_terms.sort_values([DEST_TAZ, "zone_id"]).reset_index(
        drop=True
    )

    # print(f"MAZ_size_terms\n{MAZ_size_terms}")
    #       zone_id  dest_TAZ  size_term
    # 0      106097         2      0.774
    # 1      124251         2      0.258
    # 2      124252         2      0.387
    # 3      106165         3      5.031

    return MAZ_size_terms, TAZ_size_terms


def location_presample(
    segment_name,
    persons_merged,
    network_los,
    dest_size_terms,
    estimator,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_label,
):

    trace_label = tracing.extend_trace_label(trace_label, "presample")

    logger.info(f"{trace_label} location_presample")

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]
    assert DEST_TAZ != alt_dest_col_name

    MAZ_size_terms, TAZ_size_terms = aggregate_size_terms(
        dest_size_terms, network_los, model_settings
    )

    # convert MAZ zone_id to 'TAZ' in choosers (persons_merged)
    # persons_merged[HOME_TAZ] = persons_merged[HOME_MAZ].map(maz_to_taz)
    assert HOME_MAZ in persons_merged
    assert (
        HOME_TAZ in persons_merged
    )  # 'TAZ' should already be in persons_merged from land_use

    # FIXME - MEMORY HACK - only include columns actually used in spec
    # FIXME we don't actually require that land_use provide a TAZ crosswalk
    # FIXME maybe we should add it for multi-zone (from maz_taz) if missing?
    chooser_columns = model_settings["SIMULATE_CHOOSER_COLUMNS"]
    chooser_columns = [HOME_TAZ if c == HOME_MAZ else c for c in chooser_columns]
    choosers = persons_merged[chooser_columns]

    # create wrapper with keys for this lookup - in this case there is a HOME_TAZ in the choosers
    # and a DEST_TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_skim_dict("taz")
    skims = skim_dict.wrap(HOME_TAZ, DEST_TAZ)

    taz_sample = _location_sample(
        segment_name,
        choosers,
        TAZ_size_terms,
        skims,
        estimator,
        model_settings,
        DEST_TAZ,
        chunk_size,
        chunk_tag,
        trace_label,
        zone_layer="taz",
    )

    # print(f"taz_sample\n{taz_sample}")
    #            dest_TAZ      prob  pick_count
    # person_id
    # 55227             7  0.009827           1
    # 55227            10  0.000656           1
    # 55227            18  0.014871           1
    # 55227            20  0.035548           3

    # choose a MAZ for each DEST_TAZ choice, choice probability based on MAZ size_term fraction of TAZ total
    maz_choices = tour_destination.choose_MAZ_for_TAZ(
        taz_sample, MAZ_size_terms, trace_label
    )

    assert DEST_MAZ in maz_choices
    maz_choices = maz_choices.rename(columns={DEST_MAZ: alt_dest_col_name})

    return maz_choices


def run_location_sample(
    segment_name,
    persons_merged,
    network_los,
    dest_size_terms,
    estimator,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_label,
):
    """
    select a sample of alternative locations.

    Logsum calculations are expensive, so we build a table of persons * all zones
    and then select a sample subset of potential locations

    The sample subset is generated by making multiple choices (<sample_size> number of choices)
    which results in sample containing up to <sample_size> choices for each choose (e.g. person)
    and a pick_count indicating how many times that choice was selected for that chooser.)

    person_id,  dest_zone_id, rand,            pick_count
    23750,      14,           0.565502716034,  4
    23750,      16,           0.711135838871,  6
    ...
    23751,      12,           0.408038878552,  1
    23751,      14,           0.972732479292,  2
    """

    logger.debug(
        f"dropping {(~(dest_size_terms.size_term > 0)).sum()} "
        f"of {len(dest_size_terms)} rows where size_term is zero"
    )
    dest_size_terms = dest_size_terms[dest_size_terms.size_term > 0]

    # by default, enable presampling for multizone systems, unless they disable it in settings file
    pre_sample_taz = not (network_los.zone_system == los.ONE_ZONE)
    if pre_sample_taz and not config.setting("want_dest_choice_presampling", True):
        pre_sample_taz = False
        logger.info(
            f"Disabled destination zone presampling for {trace_label} "
            f"because 'want_dest_choice_presampling' setting is False"
        )

    if pre_sample_taz:

        logger.info(
            "Running %s location_presample with %d persons"
            % (trace_label, len(persons_merged))
        )

        choices = location_presample(
            segment_name,
            persons_merged,
            network_los,
            dest_size_terms,
            estimator,
            model_settings,
            chunk_size,
            chunk_tag=f"{chunk_tag}.presample",
            trace_label=trace_label,
        )

    else:

        choices = location_sample(
            segment_name,
            persons_merged,
            network_los,
            dest_size_terms,
            estimator,
            model_settings,
            chunk_size,
            chunk_tag=f"{chunk_tag}.sample",
            trace_label=trace_label,
        )

    return choices


def run_location_logsums(
    segment_name,
    persons_merged_df,
    network_los,
    location_sample_df,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_label,
):
    """
    add logsum column to existing location_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_zone_id) pair
    in location_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | PERID     | dest_zone_id | rand           | pick_count | logsum (added) |
    +===========+==============+================+============+================+
    | 23750     |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-----------+--------------+----------------+------------+----------------+
    + 23750     | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-----------+--------------+----------------+------------+----------------+
    + ...       |              |                |            |                |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-----------+--------------+----------------+------------+----------------+
    """

    assert not location_sample_df.empty

    logsum_settings = config.read_model_settings(model_settings["LOGSUM_SETTINGS"])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged_df = logsum.filter_chooser_columns(
        persons_merged_df, logsum_settings, model_settings
    )

    logger.info(
        "Running %s with %s rows" % (trace_label, len(location_sample_df.index))
    )

    choosers = location_sample_df.join(persons_merged_df, how="left")

    tour_purpose = model_settings["LOGSUM_TOUR_PURPOSE"]
    if isinstance(tour_purpose, dict):
        tour_purpose = tour_purpose[segment_name]

    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings,
        model_settings,
        network_los,
        chunk_size,
        chunk_tag,
        trace_label,
    )

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved
    # logsums now does, since workplace_location_sample was on left side of merge de-dup merge
    location_sample_df[ALT_LOGSUM] = logsums

    return location_sample_df


def run_location_simulate(
    segment_name,
    persons_merged,
    location_sample_df,
    network_los,
    dest_size_terms,
    want_logsums,
    estimator,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_label,
):
    """
    run location model on location_sample annotated with mode_choice logsum
    to select a dest zone from sample alternatives

    Returns
    -------
    choices : pandas.DataFrame indexed by persons_merged_df.index
        choice : location choices (zone ids)
        logsum : float logsum of choice utilities across alternatives

    logsums optional & only returned if DEST_CHOICE_LOGSUM_COLUMN_NAME specified in model_settings
    """
    assert not persons_merged.empty

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings["SIMULATE_CHOOSER_COLUMNS"]
    choosers = persons_merged[chooser_columns]

    alt_dest_col_name = model_settings["ALT_DEST_COL_NAME"]

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    alternatives = pd.merge(
        location_sample_df,
        dest_size_terms,
        left_on=alt_dest_col_name,
        right_index=True,
        how="left",
    )

    logger.info("Running %s with %d persons" % (trace_label, len(choosers)))

    # create wrapper with keys for this lookup - in this case there is a home_zone_id in the choosers
    # and a zone_id in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skim_dict = network_los.get_default_skim_dict()
    skims = skim_dict.wrap("home_zone_id", alt_dest_col_name)

    locals_d = {
        "skims": skims,
        "segment_size": segment_name,
        "orig_col_name": skims.orig_key,  # added for sharrow flows
        "dest_col_name": skims.dest_key,  # added for sharrow flows
        "timeframe": "timeless",
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    if estimator:
        # write choosers after annotation
        estimator.write_choosers(choosers)
        estimator.set_alt_id(alt_dest_col_name)
        estimator.write_interaction_sample_alternatives(alternatives)

    spec = simulate.spec_for_segment(
        model_settings, spec_id="SPEC", segment_name=segment_name, estimator=estimator
    )

    log_alt_losers = config.setting("log_alt_losers", False)

    choices = interaction_sample_simulate(
        choosers,
        alternatives,
        spec=spec,
        choice_column=alt_dest_col_name,
        log_alt_losers=log_alt_losers,
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        chunk_tag=chunk_tag,
        trace_label=trace_label,
        trace_choice_name=model_settings["DEST_CHOICE_COLUMN_NAME"],
        estimator=estimator,
    )

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(choices, pd.Series)
        choices = choices.to_frame("choice")

    assert isinstance(choices, pd.DataFrame)

    return choices


def run_location_choice(
    persons_merged_df,
    network_los,
    shadow_price_calculator,
    want_logsums,
    want_sample_table,
    estimator,
    model_settings,
    chunk_size,
    chunk_tag,
    trace_hh_id,
    trace_label,
):
    """
    Run the three-part location choice algorithm to generate a location choice for each chooser

    Handle the various segments separately and in turn for simplicity of expression files

    Parameters
    ----------
    persons_merged_df : pandas.DataFrame
        persons table merged with households and land_use
    network_los : los.Network_LOS
    shadow_price_calculator : ShadowPriceCalculator
        to get size terms
    want_logsums : boolean
    want_sample_table : boolean
    estimator: Estimator object
    model_settings : dict
    chunk_size : int
    trace_hh_id : int
    trace_label : str

    Returns
    -------
    choices : pandas.DataFrame indexed by persons_merged_df.index
        'choice' : location choices (zone ids)
        'logsum' : float logsum of choice utilities across alternatives

    logsums optional & only returned if DEST_CHOICE_LOGSUM_COLUMN_NAME specified in model_settings
    """

    chooser_segment_column = model_settings["CHOOSER_SEGMENT_COLUMN_NAME"]

    # maps segment names to compact (integer) ids
    segment_ids = model_settings["SEGMENT_IDS"]

    choices_list = []
    sample_list = []
    for segment_name, segment_id in segment_ids.items():

        choosers = persons_merged_df[
            persons_merged_df[chooser_segment_column] == segment_id
        ]

        # size_term and shadow price adjustment - one row per zone
        dest_size_terms = shadow_price_calculator.dest_size_terms(segment_name)

        assert (
            dest_size_terms.index.is_monotonic_increasing
        ), f"shadow_price_calculator.dest_size_terms({segment_name}) not monotonic_increasing"

        if choosers.shape[0] == 0:
            logger.info(f"{trace_label} skipping segment {segment_name}: no choosers")
            continue

        # - location_sample
        location_sample_df = run_location_sample(
            segment_name,
            choosers,
            network_los,
            dest_size_terms,
            estimator,
            model_settings,
            chunk_size,
            chunk_tag,  # run_location_sample will add appropriate suffix for sample or presample
            trace_label=tracing.extend_trace_label(
                trace_label, "sample.%s" % segment_name
            ),
        )

        # - location_logsums
        location_sample_df = run_location_logsums(
            segment_name,
            choosers,
            network_los,
            location_sample_df,
            model_settings,
            chunk_size,
            chunk_tag=f"{chunk_tag}.logsums",
            trace_label=tracing.extend_trace_label(
                trace_label, "logsums.%s" % segment_name
            ),
        )

        # - location_simulate
        choices_df = run_location_simulate(
            segment_name,
            choosers,
            location_sample_df,
            network_los,
            dest_size_terms,
            want_logsums,
            estimator,
            model_settings,
            chunk_size,
            chunk_tag=f"{chunk_tag}.simulate",
            trace_label=tracing.extend_trace_label(
                trace_label, "simulate.%s" % segment_name
            ),
        )

        if estimator:
            if trace_hh_id:
                estimation_trace_label = tracing.extend_trace_label(
                    trace_label, f"estimation.{segment_name}.modeled_choices"
                )
                tracing.trace_df(choices_df, label=estimation_trace_label)

            estimator.write_choices(choices_df.choice)
            choices_df.choice = estimator.get_survey_values(
                choices_df.choice,
                "persons",
                column_names=model_settings["DEST_CHOICE_COLUMN_NAME"],
            )
            estimator.write_override_choices(choices_df.choice)

            if want_logsums:
                # if we override choices, we need to to replace choice logsum with ologsim for override location
                # fortunately, as long as we aren't sampling dest alts, the logsum will be in location_sample_df

                # if we start sampling dest alts, we will need code below to compute override location logsum
                assert estimator.want_unsampled_alternatives

                # merge mode_choice_logsum for the overridden location
                # alt_logsums columns: ['person_id', 'choice', 'logsum']
                alt_dest_col = model_settings["ALT_DEST_COL_NAME"]
                alt_logsums = (
                    location_sample_df[[alt_dest_col, ALT_LOGSUM]]
                    .rename(columns={alt_dest_col: "choice", ALT_LOGSUM: "logsum"})
                    .reset_index()
                )

                # choices_df columns: ['person_id', 'choice']
                choices_df = choices_df[["choice"]].reset_index()

                # choices_df columns: ['person_id', 'choice', 'logsum']
                choices_df = pd.merge(choices_df, alt_logsums, how="left").set_index(
                    "person_id"
                )

                logger.debug(
                    f"{trace_label} segment {segment_name} estimation: override logsums"
                )

            if trace_hh_id:
                estimation_trace_label = tracing.extend_trace_label(
                    trace_label, f"estimation.{segment_name}.survey_choices"
                )
                tracing.trace_df(choices_df, estimation_trace_label)

        choices_list.append(choices_df)

        if want_sample_table:
            # FIXME - sample_table
            location_sample_df.set_index(
                model_settings["ALT_DEST_COL_NAME"], append=True, inplace=True
            )
            sample_list.append(location_sample_df)
        else:
            # del this so we dont hold active reference to it while run_location_sample is creating its replacement
            del location_sample_df

    if len(choices_list) > 0:
        choices_df = pd.concat(choices_list)
    else:
        # this will only happen with small samples (e.g. singleton) with no (e.g.) school segs
        logger.warning("%s no choices", trace_label)
        choices_df = pd.DataFrame(columns=["choice", "logsum"])

    if len(sample_list) > 0:
        save_sample_df = pd.concat(sample_list)
    else:
        # this could happen either with small samples as above, or if no saved sample desired
        save_sample_df = None

    return choices_df, save_sample_df


def iterate_location_choice(
    model_settings,
    persons_merged,
    persons,
    households,
    network_los,
    estimator,
    chunk_size,
    trace_hh_id,
    locutor,
    trace_label,
):
    """
    iterate run_location_choice updating shadow pricing until convergence criteria satisfied
    or max_iterations reached.

    (If use_shadow_pricing not enabled, then just iterate once)

    Parameters
    ----------
    model_settings : dict
    persons_merged : injected table
    persons : injected table
    network_los : los.Network_LOS
    chunk_size : int
    trace_hh_id : int
    locutor : bool
        whether this process is the privileged logger of shadow_pricing when multiprocessing
    trace_label : str

    Returns
    -------
    adds choice column model_settings['DEST_CHOICE_COLUMN_NAME']
    adds logsum column model_settings['DEST_CHOICE_LOGSUM_COLUMN_NAME']- if provided
    adds annotations to persons table
    """

    chunk_tag = trace_label

    # boolean to filter out persons not needing location modeling (e.g. is_worker, is_student)
    chooser_filter_column = model_settings["CHOOSER_FILTER_COLUMN_NAME"]

    dest_choice_column_name = model_settings["DEST_CHOICE_COLUMN_NAME"]
    logsum_column_name = model_settings.get("DEST_CHOICE_LOGSUM_COLUMN_NAME")

    sample_table_name = model_settings.get("DEST_CHOICE_SAMPLE_TABLE_NAME")
    want_sample_table = (
        config.setting("want_dest_choice_sample_tables")
        and sample_table_name is not None
    )

    persons_merged_df = persons_merged.to_frame()

    persons_merged_df = persons_merged_df[persons_merged[chooser_filter_column]]

    persons_merged_df.sort_index(
        inplace=True
    )  # interaction_sample expects chooser index to be monotonic increasing

    # chooser segmentation allows different sets coefficients for e.g. different income_segments or tour_types
    chooser_segment_column = model_settings["CHOOSER_SEGMENT_COLUMN_NAME"]
    segment_ids = model_settings["SEGMENT_IDS"]

    assert (
        chooser_segment_column in persons_merged_df
    ), f"CHOOSER_SEGMENT_COLUMN '{chooser_segment_column}' not in persons_merged table."

    spc = shadow_pricing.load_shadow_price_calculator(model_settings)
    max_iterations = spc.max_iterations
    assert not (spc.use_shadow_pricing and estimator)

    logger.debug("%s max_iterations: %s" % (trace_label, max_iterations))

    choices_df = None  # initialize to None, will be populated in first iteration

    for iteration in range(1, max_iterations + 1):

        persons_merged_df_ = persons_merged_df.copy()

        if spc.use_shadow_pricing and iteration > 1:
            spc.update_shadow_prices()

            if spc.shadow_settings["SHADOW_PRICE_METHOD"] == "simulation":
                # filter from the sampled persons
                persons_merged_df_ = persons_merged_df_[
                    persons_merged_df_.index.isin(spc.sampled_persons.index)
                ]
                persons_merged_df_ = persons_merged_df_.sort_index()

        choices_df_, save_sample_df = run_location_choice(
            persons_merged_df_,
            network_los,
            shadow_price_calculator=spc,
            want_logsums=logsum_column_name is not None,
            want_sample_table=want_sample_table,
            estimator=estimator,
            model_settings=model_settings,
            chunk_size=chunk_size,
            chunk_tag=chunk_tag,
            trace_hh_id=trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, "i%s" % iteration),
        )

        # choices_df is a pandas DataFrame with columns "choice" and (optionally) "logsum"
        if choices_df_ is None:
            break

        if spc.use_shadow_pricing:
            # handle simulation method
            if (
                spc.shadow_settings["SHADOW_PRICE_METHOD"] == "simulation"
                and iteration > 1
            ):
                # if a process ends up with no sampled workers in it, hence an empty choice_df_, then choice_df wil be what it was previously
                if len(choices_df_) != 0:
                    choices_df = pd.concat([choices_df, choices_df_], axis=0)
                    choices_df_index = choices_df_.index.name
                    choices_df = choices_df.reset_index()
                    # update choices of workers/students
                    choices_df = choices_df.drop_duplicates(
                        subset=[choices_df_index], keep="last"
                    )
                    choices_df = choices_df.set_index(choices_df_index)
                    choices_df = choices_df.sort_index()
            else:
                choices_df = choices_df_.copy()

        else:
            choices_df = choices_df_

        spc.set_choices(
            choices=choices_df["choice"],
            segment_ids=persons_merged_df[chooser_segment_column].reindex(
                choices_df.index
            ),
        )

        if locutor:
            spc.write_trace_files(iteration)

        if spc.use_shadow_pricing and spc.check_fit(iteration):
            logging.info(
                "%s converged after iteration %s"
                % (
                    trace_label,
                    iteration,
                )
            )
            break

    # - shadow price table
    if locutor:
        if spc.use_shadow_pricing and "SHADOW_PRICE_TABLE" in model_settings:
            inject.add_table(model_settings["SHADOW_PRICE_TABLE"], spc.shadow_prices)
        if "MODELED_SIZE_TABLE" in model_settings:
            inject.add_table(model_settings["MODELED_SIZE_TABLE"], spc.modeled_size)

    persons_df = persons.to_frame()

    # add the choice values to the dest_choice_column in persons dataframe
    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    # names for location choice and (optional) logsums columns
    NO_DEST_ZONE = -1
    persons_df[dest_choice_column_name] = (
        choices_df["choice"].reindex(persons_df.index).fillna(NO_DEST_ZONE).astype(int)
    )

    # add the dest_choice_logsum column to persons dataframe
    if logsum_column_name:
        persons_df[logsum_column_name] = (
            choices_df["logsum"].reindex(persons_df.index).astype("float")
        )

    if save_sample_df is not None:
        # might be None for tiny samples even if sample_table_name was specified
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # lest they try to put school and workplace samples into the same table
        if pipeline.is_table(sample_table_name):
            raise RuntimeError(
                "dest choice sample table %s already exists" % sample_table_name
            )
        pipeline.extend_table(sample_table_name, save_sample_df)

    # - annotate persons table
    if "annotate_persons" in model_settings:
        expressions.assign_columns(
            df=persons_df,
            model_settings=model_settings.get("annotate_persons"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_persons"),
        )

        pipeline.replace_table("persons", persons_df)

        if trace_hh_id:
            tracing.trace_df(persons_df, label=trace_label, warn_if_empty=True)

    # - annotate households table
    if "annotate_households" in model_settings:

        households_df = households.to_frame()
        expressions.assign_columns(
            df=households_df,
            model_settings=model_settings.get("annotate_households"),
            trace_label=tracing.extend_trace_label(trace_label, "annotate_households"),
        )
        pipeline.replace_table("households", households_df)

        if trace_hh_id:
            tracing.trace_df(households_df, label=trace_label, warn_if_empty=True)

    if logsum_column_name:
        tracing.print_summary(
            logsum_column_name, choices_df["logsum"], value_counts=True
        )

    return persons_df


@inject.step()
def workplace_location(
    persons_merged, persons, households, network_los, chunk_size, trace_hh_id, locutor
):
    """
    workplace location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = "workplace_location"
    model_settings = config.read_model_settings("workplace_location.yaml")

    estimator = estimation.manager.begin_estimation("workplace_location")
    if estimator:
        write_estimation_specs(estimator, model_settings, "workplace_location.yaml")

    # FIXME - debugging code to test multiprocessing failure handling
    # process_name = multiprocessing.current_process().name
    # if multiprocessing.current_process().name =='mp_households_0':
    #     raise RuntimeError(f"fake fail {process_name}")

    # disable locutor for benchmarking
    if config.setting("benchmarking", False):
        locutor = False

    iterate_location_choice(
        model_settings,
        persons_merged,
        persons,
        households,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        locutor,
        trace_label,
    )

    if estimator:
        estimator.end_estimation()


@inject.step()
def school_location(
    persons_merged, persons, households, network_los, chunk_size, trace_hh_id, locutor
):
    """
    School location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = "school_location"
    model_settings = config.read_model_settings("school_location.yaml")

    estimator = estimation.manager.begin_estimation("school_location")
    if estimator:
        write_estimation_specs(estimator, model_settings, "school_location.yaml")

    # disable locutor for benchmarking
    if config.setting("benchmarking", False):
        locutor = False

    iterate_location_choice(
        model_settings,
        persons_merged,
        persons,
        households,
        network_los,
        estimator,
        chunk_size,
        trace_hh_id,
        locutor,
        trace_label,
    )

    if estimator:
        estimator.end_estimation()
