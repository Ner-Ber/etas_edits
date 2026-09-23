# HGX + rolling run status

**Updated:** 2026-09-22 16:00 IDT (hanamel inventory + HGX session notes)

Operational snapshot for deciding what to run next on HGX vs leave on hanamel.
Not a code TODO — see `docs/CODE_TODO.md` for engineering backlog.

---

## HGX (cluster) — in progress

| Item | Status |
|------|--------|
| Host | `HGX` / `hgx.tau.ac.il`, partition `debug`, 64 CPU, 7×A100-80GB, ~500 GB RAM |
| Code + data | Under `/data/neriberman/etas`, IFS MAGNET, models, catalogs, depth CSVs |
| Env | `/data/neriberman/envs/etas_remote` (rebuilt; needs `PROJ_DATA` + pyproj→conda `share/proj` symlink) |
| **Smoke** | **Running** (as of ~15:53 IDT): JMA T=7, seeds 10–11, `--max-workers 2`, output `outputs/rolling_smoke_jma_hgx` |
| Smoke progress | Past PROJ; both workers ~100% CPU in inversion **calculating distances** (117k sources) |
| Success criteria | Both seeds `exit=0`; `forecast_catalog.csv` under smoke tree |

Check on HGX:

```bash
tail -5 /data/neriberman/etas/outputs/rolling_smoke_jma_hgx/seed_pool_logs/seed_10.log
tail -5 /data/neriberman/etas/outputs/rolling_smoke_jma_hgx/seed_pool_logs/seed_11.log
ps -u neriberman -o pid,pcpu,etime,cmd | grep run_rolling | grep -v grep
```

Launch pattern that works:

```bash
export PROJ_DATA=/data/neriberman/envs/etas_remote/share/proj PROJ_LIB=$PROJ_DATA
export ETAS_FINE_AH_GPU=0 CUDA_VISIBLE_DEVICES=""
export MAGNET_INCREMENTAL_ENCODERS=1 MAGNET_INCREMENTAL_FEATURE_STATE=1 MAGNET_INCREMENTAL_SLIDING=1

python runnable_code/launch_rolling_seed_pool.py \
  --config config/rolling_continuation_<name>.json \
  --magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs \
  --horizon-days 7,60,300 \
  --n-runs N --max-workers W \
  --output-root outputs/<run_name>
```

---

## Hanamel — rolling outputs (no active jobs)

No tmux / no rolling Python processes when checked (2026-09-22 16:00).
Old `*.log` files (2026-09-14) end with **FeatureState ImportError** (PYTHONPATH pointed at `eq_mag_prediction_clean` without IFS). Treat logs as stale; use catalog counts below.

Disk: ~1.8 TB under `outputs/rolling_continuation_*` (do **not** copy wholesale to HGX).

### Completeness snapshot

| Run dir | Size | Horizons | `forecast_catalog.csv` counts | `rolling_summary.csv` | Notes |
|---------|------|----------|-------------------------------|----------------------|--------|
| `rolling_continuation_hauksson_mc_depth` | 332 G | 7d, 60d, 300d, 600d | 7d=15120, 60d=1760, 300d=400, 600d=30 | yes (all) | Richest; older mc_depth experiment |
| `rolling_continuation_hauksson_pre_ridgecrest` | 255 G | 7d, 60d, 300d | 7d=9720, 60d=1140, 300d=240 | yes | Looks largely complete for configured seeds/methods |
| `rolling_continuation_jma_90d_2009` | 868 G | 7d, 60d, 300d | 7d=606, 60d=31, 300d=12 | yes | **Sparse vs hauksson** — incomplete / few seeds finished long horizons |
| `rolling_continuation_nz_180d_2014` | 149 G | **7d only** | 37 | **no** | Incomplete; stopped mid T=7 (seeds ~10–13, partial steps) |
| `rolling_continuation_ucerf3_post_elmayor` | 164 G | **7d only** | 185 | **no** | Incomplete; T=7 in progress historically, no 60/300 |

Config defaults (`n_runs=5`, methods `etas`+`FINE`) understate some dirs that clearly used more seeds (hauksson catalog counts imply ~30 seeds × 2 methods on many steps).

### Implied gaps (good HGX candidates)

1. **NZ** — finish T=7, then add T=60,300 (and more seeds if desired).
2. **UCERF3** — finish T=7, then T=60,300.
3. **JMA** — fill seeds / longer horizons (T=60,300 especially thin); smoke on HGX is the pathfinder.
4. **New prospect windows / test splits** — not started; need new rolling JSON configs.
5. **Hauksson pre-ridgecrest / mc_depth** — already heavy on hanamel; lower priority to re-run on HGX unless regenerating with new FINE path.

---

## Suggested next on HGX (decide after smoke)

**After smoke is green**, preferred order:

1. **JMA full** — same config, `--horizon-days 7,60,300`, `--n-runs 30`, `--max-workers 10–12`, new `output_root` (or continue smoke root only if intentional).
2. **NZ** — resume/complete from config `rolling_continuation_nz_180d_2014.json` (HGX already has catalog + model).
3. **UCERF3** — same for `rolling_continuation_ucerf3_post_elmayor.json`.
4. Parallelize catalogs as separate jobs (~12 CPUs each) once smoke proves stability; keep MAGNET train (GPU) separate if needed.

**Keep on hanamel:** analysis notebooks (`compare_rolling_etas_fine.ipynb`), existing large output trees, light post-processing.

---

## Env gotchas (HGX) — don’t relearn

- Do **not** raw-tar `etas_remote` without `conda-unpack` (segfaulted).
- Cartopy / geo: prefer **conda-forge**; avoid pip `pyproj` fighting conda `proj`.
- Install `proj-data`; symlink away stale `site-packages/pyproj/proj_dir/share/proj` → `$CONDA_PREFIX/share/proj`.
- Always: `--magnet-root /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs`.
- Depth paths rewritten in rolling JSONs to `/data/neriberman/magnet_ingested/*.csv`.
- Cancel orphan `srun` bash jobs; unset stale `SLURM_*` before new `srun`.
