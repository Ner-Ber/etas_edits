"""Build the cross-horizon comparison cache for one rolling experiment.

Fills each horizon's analysis cache (the tables and grids used by
``notebooks/compare_rolling_etas_fine_horizon_bayona.ipynb``), then writes
``<output_root>/horizon_comparison/``. A second run leaves finished horizons
in place. A new realization is added inside that horizon's analysis cache, and
only that horizon's comparison rows are replaced.

``--force`` rebuilds the comparison tables and recomputes window distribution
tests. Intensity grids and Bayona scores stay on disk when the forecast files
are unchanged.

``--from-cache`` does not compute. It uses score files already saved for each
horizon, including a run that was stopped early.

Example::

  python runnable_code/cache_horizon_comparison.py \\
    --output-root outputs/rolling_continuation_hauksson_pre_ridgecrest_short \\
    --config config/rolling_continuation_hauksson_pre_ridgecrest_short.json \\
    --horizon-days 0.5 1 --dh 0.1 --num-simulations 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import etas.horizon_comparison as horizon_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--horizon-days",
        type=float,
        nargs="*",
        default=None,
        help="Forecast lengths to include. Default: every horizon_* directory.",
    )
    parser.add_argument("--dh", type=float, default=0.1)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=None,
        help="Most catalogs to cache per method in each window. "
        "Uses the lowest seed ids, or every catalog when fewer exist.",
    )
    parser.add_argument("--methods", nargs="+", default=["etas", "FINE"])
    parser.add_argument(
        "--skip-window-tests",
        action="store_true",
        help="Skip negative-binomial and binary conditional-likelihood tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild comparison tables and recompute window tests.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Use scores already on disk. Do not compute missing windows.",
    )
    args = parser.parse_args()
    summary = horizon_comparison.update_comparison(
        args.output_root,
        config_path=args.config,
        repo_root=args.repo_root,
        horizon_days=args.horizon_days,
        methods=args.methods,
        dh=args.dh,
        num_simulations=args.num_simulations,
        n_realizations=args.n_realizations,
        include_window_tests=not args.skip_window_tests,
        force=args.force,
        from_cache=args.from_cache,
        progress=lambda message: print(message, flush=True),
    )
    print(
        f"comparison cache: {summary.cache_dir} "
        f"({summary.horizons_checked} horizons, {summary.horizons_refreshed} refreshed)",
        flush=True,
    )


if __name__ == "__main__":
    main()
