"""Fill the rolling-analysis cache for one horizon.

Writes intensity grids, Bayona scores, per-window negative-binomial / binary
conditional-likelihood tests, and soft background probabilities
(``P0 = mu / lambda`` at event times) under ``analysis_cache/``. A second run
leaves existing results in place and only recomputes inputs that changed.
Other notebooks can call ``etas.rolling_analysis.update_cache`` or this
script, then ``load_analysis``, ``window_distribution_tests``, and
``background_probabilities``.

Example::

  python runnable_code/cache_rolling_analysis.py \\
    --output-root outputs/rolling_continuation_hauksson_pre_ridgecrest \\
    --config config/rolling_continuation_hauksson_pre_ridgecrest.json \\
    --horizon-days 7 --dh 0.1 --num-simulations 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import etas.rolling_analysis as rolling_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--horizon-days", type=float, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dh", type=float, default=0.1)
    parser.add_argument("--num-simulations", type=int, default=200)
    parser.add_argument("--methods", nargs="+", default=["etas", "FINE"])
    args = parser.parse_args()
    if args.horizon_dir is not None:
        horizon_dir = args.horizon_dir
    elif args.output_root is not None and args.horizon_days is not None:
        horizon_dir = rolling_analysis.horizon_directory(args.output_root, args.horizon_days)
    else:
        parser.error("Pass --horizon-dir, or both --output-root and --horizon-days")
    summary = rolling_analysis.update_cache(
        horizon_dir,
        config_path=args.config,
        repo_root=args.repo_root,
        methods=args.methods,
        dh=args.dh,
        num_simulations=args.num_simulations,
        progress=lambda message: print(message, flush=True),
    )
    if summary.up_to_date:
        return
    print(f"cache directory: {summary.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
