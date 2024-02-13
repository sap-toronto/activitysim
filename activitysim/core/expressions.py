# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import assign, simulate, tracing, workflow
from activitysim.core.configuration.base import PreprocessorSettings, PydanticBase
from activitysim.core.util import (
    assign_in_place,
    parse_suffix_args,
    suffix_expressions_df_str,
)

logger = logging.getLogger(__name__)


def compute_columns(
    state: workflow.State,
    df: pd.DataFrame,
    model_settings: str | dict | PydanticBase,
    locals_dict: dict | None = None,
    trace_label: str = None,
) -> pd.DataFrame:
    """
    Evaluate expressions_spec in context of df, with optional additional pipeline tables in locals

    Parameters
    ----------
    df : pandas DataFrame
        or if None, expect name of pipeline table to be specified by DF in model_settings
    model_settings : dict or str
        dict with keys:
            DF - df_alias and (additionally, if df is None) name of pipeline table to load as df
            SPEC - name of expressions file (csv suffix optional) if different from model_settings
            TABLES - list of pipeline tables to load and make available as (read only) locals
        str:
            name of yaml file in configs_dir to load dict from
    locals_dict : dict, optional
        dict of locals (e.g. utility functions) to add to the execution environment
    trace_label

    Returns
    -------
    results: pandas.DataFrame
        one column for each expression (except temps with ALL_CAP target names)
        same index as df
    """
    if locals_dict is None:
        locals_dict = {}

    if isinstance(model_settings, PydanticBase):
        model_settings = model_settings.dict()

    if isinstance(model_settings, str):
        model_settings_name = model_settings
        model_settings = state.filesystem.read_model_settings(
            "%s.yaml" % model_settings
        )
        assert model_settings, "Found no model settings for %s" % model_settings_name
    else:
        model_settings_name = "dict"
        assert isinstance(model_settings, dict)

    assert "DF" in model_settings, "Expected to find 'DF' in %s" % model_settings_name

    df_name = model_settings.get("DF")
    helper_table_names = model_settings.get("TABLES") or []
    expressions_spec_name = model_settings.get("SPEC", None)

    # Extract suffix for disaggregate accessibilities.
    # The suffix args can either be passed in the model settings or as part of the config file string.
    # Awkward, but avoids having to put positional arguments in every single asim function.
    args = parse_suffix_args(expressions_spec_name)

    expressions_spec_name = args.filename
    suffix = model_settings.get("SUFFIX", args.SUFFIX)
    roots = model_settings.get("ROOTS", args.ROOTS)

    assert isinstance(roots, list)
    assert (suffix is not None and roots) or (suffix is None and not roots), (
        "Expected to find both 'ROOTS' and 'SUFFIX' in %s, missing one"
        % model_settings_name
    )

    assert expressions_spec_name is not None, (
        "Expected to find 'SPEC' in %s" % model_settings_name
    )

    trace_label = tracing.extend_trace_label(trace_label or "", expressions_spec_name)

    if not expressions_spec_name.endswith(".csv"):
        expressions_spec_name = "%s.csv" % expressions_spec_name
    logger.debug(
        f"{trace_label} compute_columns using expression spec file {expressions_spec_name}"
    )

    expressions_spec = assign.read_assignment_spec(
        state.filesystem.get_config_file_path(expressions_spec_name),
    )

    if suffix is not None and roots:
        expressions_spec = suffix_expressions_df_str(expressions_spec, suffix, roots)

    assert expressions_spec.shape[0] > 0, (
        "Expected to find some assignment expressions in %s" % expressions_spec_name
    )

    tables = {t: state.get_dataframe(t) for t in helper_table_names}

    # if df was passed in, df might be a slice, or any other table, but DF is it's local alias
    assert df_name not in tables, "Did not expect to find df '%s' in TABLES" % df_name
    tables[df_name] = df

    # be nice and also give it to them as df?
    tables["df"] = df

    _locals_dict = assign.local_utilities(state)
    _locals_dict.update(locals_dict)
    _locals_dict.update(tables)

    # FIXME a number of asim model preprocessors want skim_dict - should they request it in model_settings.TABLES?
    try:
        if state.settings.sharrow:
            from activitysim.core.flow import skim_dataset_dict  # noqa F401
            from activitysim.core.skim_dataset import skim_dataset  # noqa F401

            _locals_dict["skim_dict"] = state.get_injectable("skim_dataset_dict")
        else:
            _locals_dict["skim_dict"] = state.get_injectable("skim_dict")
    except FileNotFoundError:
        pass  # maybe we don't even need the skims

    results, trace_results, trace_assigned_locals = assign.assign_variables(
        state,
        expressions_spec,
        df,
        _locals_dict,
        trace_rows=state.tracing.trace_targets(df),
    )

    if trace_results is not None:
        state.tracing.trace_df(trace_results, label=trace_label, slicer="NONE")

    if trace_assigned_locals:
        state.tracing.write_csv(
            trace_assigned_locals, file_name="%s_locals" % trace_label
        )

    return results


def assign_columns(
    state: workflow.State, df, model_settings, locals_dict=None, trace_label=None
):
    """
    Evaluate expressions in context of df and assign resulting target columns to df

    Can add new or modify existing columns (if target same as existing df column name)

    Parameters - same as for compute_columns except df must not be None
    Returns - nothing since we modify df in place
    """
    if locals_dict is None:
        locals_dict = {}

    assert df is not None
    assert model_settings is not None

    results = compute_columns(state, df, model_settings, locals_dict, trace_label)

    assign_in_place(
        df, results, state.settings.downcast_int, state.settings.downcast_float
    )


# ##################################################################################################
# helpers
# ##################################################################################################


def annotate_preprocessors(
    state: workflow.State,
    df: pd.DataFrame,
    locals_dict,
    skims,
    model_settings: PydanticBase | dict,
    trace_label: str,
):
    locals_d = {}
    locals_d.update(locals_dict)
    locals_d.update(skims)

    try:
        preprocessor_settings = model_settings.preprocessor
    except AttributeError:
        preprocessor_settings = model_settings.get("preprocessor", [])
    if preprocessor_settings is None:
        preprocessor_settings = []
    if not isinstance(preprocessor_settings, list):
        assert isinstance(preprocessor_settings, dict | PreprocessorSettings)
        preprocessor_settings = [preprocessor_settings]

    simulate.set_skim_wrapper_targets(df, skims)

    for preproc_settings in preprocessor_settings:
        results = compute_columns(
            state,
            df=df,
            model_settings=preproc_settings,
            locals_dict=locals_d,
            trace_label=trace_label,
        )

        assign_in_place(
            df, results, state.settings.downcast_int, state.settings.downcast_float
        )


def filter_chooser_columns(choosers, chooser_columns):
    missing_columns = [c for c in chooser_columns if c not in choosers]
    if missing_columns:
        logger.debug("filter_chooser_columns missing_columns %s" % missing_columns)

    # ignore any columns not appearing in choosers df
    chooser_columns = [c for c in chooser_columns if c in choosers]

    choosers = choosers[chooser_columns]
    return choosers
