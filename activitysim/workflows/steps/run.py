import shlex

from pypyr.errors import KeyNotInContextError

from ...standalone.utils import chdir
from .cmd import run_step as _run_cmd
from .progression import reset_progress_step
from .wrapping import workstep


def _get_formatted(context, key, default):
    try:
        out = context.get_formatted(key)
    except KeyNotInContextError:
        out = None
    if out is None:
        out = default
    return out


@workstep
def run_activitysim(
    label=None,
    cwd=None,
    pre_config_dirs=(),
    config_dirs=("configs",),
    data_dir="data",
    output_dir="output",
    resume_after=None,
    fast=True,
) -> None:
    if isinstance(pre_config_dirs, str):
        pre_config_dirs = [pre_config_dirs]
    else:
        pre_config_dirs = list(pre_config_dirs)
    if isinstance(config_dirs, str):
        config_dirs = [config_dirs]
    else:
        config_dirs = list(config_dirs)
    flags = []
    if resume_after:
        flags.append(f" -r {resume_after}")
    if fast:
        flags.append("--fast")
    flags = " ".join(flags)
    cfgs = " ".join(f"-c {c}" for c in pre_config_dirs + config_dirs)
    args = f"run {cfgs} -d {data_dir} -o {output_dir} {flags}"
    if label is None:
        label = f"activitysim {args}"

    reset_progress_step(description=f"{label}", prefix="[bold green]")

    # Clear all saved state from ORCA
    import orca

    orca.clear_cache()
    orca.clear_all()

    # Re-inject everything from ActivitySim
    from ...core.inject import reinject_decorated_tables

    reinject_decorated_tables(steps=True)

    # Call the run program inside this process
    from activitysim.cli.main import prog

    with chdir(cwd):
        namespace = prog().parser.parse_args(shlex.split(args))
        namespace.afunc(namespace)
