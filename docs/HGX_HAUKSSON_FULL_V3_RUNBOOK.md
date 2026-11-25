# HGX Runbook — Hauksson Full Test Set (Berman et al.) `v3`

**Written:** 2026-11-24  
**Target host:** `neriberman@hgx.tau.ac.il` (`/data/neriberman/etas`, branch `feature/hgx-inversion-interval`)  
**Goal:** Extend the clean `v3` rolling ETAS vs FINE benchmark across the **entire Hauksson test set from Berman et al.** (including the July 2019 M6.4 / M7.1 Ridgecrest sequence and post-Ridgecrest period through `2019-12-31 22:48:20`), into a new output tree (`outputs/rolling_continuation_hauksson_full_v3*`), while leaving `outputs/rolling_continuation_hauksson_pre_ridgecrest_v3*` completely untouched.

---

## 1. Scientific & Configuration Summary

Everything matches `pre_ridgecrest_v3` except `testwindow_end` and `output_root`:

| Field | Value | Notes |
|---|---|---|
| `auxiliary_start` | `1981-01-01 00:00:00` | `train_start_time = 347155200` |
| `timewindow_start` | `2009-02-03 00:00:00` | ETAS target start (validation start `1233664779`) |
| `timewindow_end` | `2016-05-23 23:05:46` | `test_start_time = 1464044746` |
| `testwindow_end` | **`2019-12-31 22:48:20`** | Exact Berman et al. `test_end_time = 1577832500` |
| `mc` / `delta_m` | `2.4` / `0` | Same as `v3`; `3,472` obs (`M ≥ 2.4`), `3,407` (`M > 2.4`) |
| `methods` | `["etas", "FINE"]` | Same frozen `mc_depth/_repetition_0` MAGNET checkpoint |
| `horizons` | `0.5, 1, 4, 7, 60, 300` d | Same 6 horizons |
| `seeds` | `0..29` (`n_runs = 30`) | Per-step derived `rng_seed` (`SeedSequence([seed, step_index])`) |
| `inversion_interval` | `365` days | Uniform across all 6 horizons; `store_pij = false`, `store_distances = false` |
| `max_forecast_events_per_day` | code default (`3000`) | Peak Ridgecrest day has `566` obs (`M ≥ 2.4`), peak 12 h has `299` |

Configs:
- **HGX:** `config/rolling_continuation_hauksson_full_v3_hgx.json` (`depth_source_catalog = /data/neriberman/magnet_ingested/hauksson.csv`)
- **hanamel (post-sync analysis):** `config/rolling_continuation_hauksson_full_v3.json`

---

## 2. Exact Prefix Reuse & Step Schedule

Because `timewindow_start`, `timewindow_end`, `step_index`, per-step `rng_seed`, `inversion_interval = 365`, and `magnet.model_dir` are unchanged, steps `0 .. N_pre - 2` are **bit-for-bit identical** to `pre_ridgecrest_v3`. Only the formerly clipped final pre-Ridgecrest step (`step_{N_pre - 1}`) and the post-`2019-07-01` steps are simulated (~85% compute saved):

| Horizon | Pre-RC steps | Full steps | Seeded prefix (`0 .. start-1`) | `start_step` | Steps to run | Reused anchors | New post-RC anchors |
|---:|---:|---:|---:|---:|---:|---|---|
| `0.5 d` | `2,267` | `2,634` | `2,266` (`0..2265`) | `2266` | **368** (`2266..2633`) | `0, 731, 1462, 2193` | none |
| `1 d` | `1,134` | `1,317` | `1,133` (`0..1132`) | `1133` | **184** (`1133..1316`) | `0, 366, 732, 1098` | none |
| `4 d` | `284` | `330` | `283` (`0..282`) | `283` | **47** (`283..329`) | `0, 92, 184, 276` | none |
| `7 d` | `162` | `189` | `161` (`0..160`) | `161` | **28** (`161..188`) | `0, 53, 106, 159` | none |
| `60 d` | `19` | `22` | `18` (`0..17`) | `18` | **4** (`18..21`) | `0, 7, 14` | **`step_021`** (`2019-11-04`) |
| `300 d` | `4` | `5` | `3` (`0..2`) | `3` | **2** (`3..4`) | `0, 2` | **`step_004`** (`2019-09-05`) |

- Every prefix step (`step_000 .. step_{N_pre - 2}`) is seeded via independent file copies (`--copy-mode copy`, **0 shared inodes**, ~5.1 GB total on HGX), with fresh path-rewritten `parameters_<id>.json`, `active_inversion.json`, and `step_config.json` strictly inside `outputs/rolling_continuation_hauksson_full_v3_h<H>d/` so `pre_ridgecrest_v3` is completely isolated and untouched.
- `run_rolling_continuation.py --start-step <N>` skips simulating steps `< N` while keeping exact full-horizon `step_index`, `rng_seed`, and `reuse_inversion_from_step` pointers.
- Estimated total compute across all 6 horizons × 30 seeds: **~90–145 GPU-hours (~13–21 wall-hours on 7 A100s)**, vs ~450–550 GPU-hours from scratch.

---

## 3. Sync Code to HGX (`feature/hgx-inversion-interval`)

First commit/push on `hanamel` (`feature/rolling-fine-etas-forecasts`), then on **HGX** (`/data/neriberman/etas`, stay on `feature/hgx-inversion-interval`!):

```bash
cd /data/neriberman/etas
git branch --show-current   # must be: feature/hgx-inversion-interval
git fetch personal-origin
# Cherry-pick the commit that adds --start-step + full_v3 configs & scripts:
git cherry-pick <COMMIT_SHA>
```

Preflight check on HGX before submitting anything:

```bash
cd /data/neriberman/etas
test -f input_data/etas_converted_hauksson.csv
test -f input_data/hauksson_polygon_from_notebook.npy
test -f /data/neriberman/magnet_ingested/hauksson.csv
test -d outputs/benchmark_matrix/benchmark_full_20260801/hauksson/FINE_variants/mc_depth/magnet/model_9e2618ba26305201f75f50f798fd38c26f05bd5b/_repetition_0/model
test -f config/rolling_continuation_hauksson_pre_ridgecrest_v3.json
test -f config/rolling_continuation_hauksson_full_v3_hgx.json
squeue -u neriberman
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
```

Dry-run the prefix seeder + inversion planner (read-only, takes ~10 s):

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/neriberman/envs/etas_remote
python scripts/seed_hauksson_full_v3_from_prerc.py --dry-run
bash scripts/submit_hauksson_full_v3_hgx.sh --dry-run
```

---

## 4. Launch on HGX (via Slurm `sbatch`)

### Option A (recommended first): Stage 0 preseed + 1-seed Ridgecrest smoke check (~45 min)

Verifies prefix seeding, both new post-Ridgecrest ETAS anchor inversions (`60d/step_021` and `300d/step_004`), and `seed 0` through Ridgecrest on `7 d` and `0.5 d` (counts, wall-time, no cap hits, no OOM):

```bash
cd /data/neriberman/etas
bash scripts/submit_hauksson_full_v3_hgx.sh --smoke
```

Once `hfv3_preseed` prints `PRESEED_COMPLETE` and the two smoke jobs print `WALK_COMPLETE`, launch the full 30-seed × 6-horizon campaign (skipping preseed; `seed 0` on `7 d`/`0.5 d` is automatically cache-hit):

```bash
bash scripts/submit_hauksson_full_v3_hgx.sh --submit --skip-preseed
```

### Option B: Direct full submission (Stage 0 → Stage 1 walks → Stage 3 scoring)

Submits everything in one Slurm dependency DAG:
- `hfv3_preseed` (1 CPU job, ~15–25 min): copies prefix (0 shared inodes) + fits `60d/step_021` and `300d/step_004`
- `36` GPU walk jobs (`6` horizons × `6` chunks of `5` seeds `0..29`, ordered `300, 60, 7, 4, 1, 0.5 d`), each chained after `hfv3_preseed`
- `6` GPU Stage 3 scoring jobs (`hfv3_score_<H>d`), each automatically chained via `afterok` on its horizon's 6 seed chunks

```bash
cd /data/neriberman/etas
bash scripts/submit_hauksson_full_v3_hgx.sh --submit
```

---

## 5. Monitor on HGX

```bash
squeue -u neriberman
# Check preseed:
tail -n 30 /data/neriberman/etas/logs/hfv3_preseed_*.out
# Check completion markers across all scoring jobs:
grep -h "SCORE_COMPLETE\|AUDIT PASSED\|Forecast " /data/neriberman/etas/logs/hfv3_score_*.out
```

---

## 6. Sync Back to `hanamel` & Build Presentation Caches

Once all 6 scoring jobs print `SCORE_COMPLETE horizon=<H>d`, unify them on `hanamel` under `outputs/rolling_continuation_hauksson_full_v3/` (mirroring `pre_ridgecrest_v3`):

```bash
# On hanamel (after opening SSH ControlMaster socket to HGX):
mkdir -p /a/home/cc/students/csguests/neriberman/Repos/etas/outputs/rolling_continuation_hauksson_full_v3/hgx_score_logs

for H in 0.5 1 4 7 60 300; do
  rsync -av --info=progress2 -e "ssh -S /tmp/ssh_hgx_full_v3.sock" \
    "neriberman@hgx.tau.ac.il:/data/neriberman/etas/outputs/rolling_continuation_hauksson_full_v3_h${H}d/horizon_${H}d" \
    /a/home/cc/students/csguests/neriberman/Repos/etas/outputs/rolling_continuation_hauksson_full_v3/
done

rsync -av -e "ssh -S /tmp/ssh_hgx_full_v3.sock" \
  "neriberman@hgx.tau.ac.il:/data/neriberman/etas/logs/hfv3_preseed_*.out" \
  "neriberman@hgx.tau.ac.il:/data/neriberman/etas/logs/hfv3_score_*.out" \
  /a/home/cc/students/csguests/neriberman/Repos/etas/outputs/rolling_continuation_hauksson_full_v3/hgx_score_logs/
```

Verify locally on `hanamel` and build `horizon_comparison/` + `presentation_tables/`:

```bash
cd /a/home/cc/students/csguests/neriberman/Repos/etas
PY=/a/home/cc/students/csguests/neriberman/anaconda3/envs/etas_remote/bin/python

for H in 0.5 1 4 7 60 300; do
  "$PY" scripts/audit_hauksson_full_v3_horizon.py \
    --horizon-dir "outputs/rolling_continuation_hauksson_full_v3/horizon_${H}d" \
    --config config/rolling_continuation_hauksson_full_v3.json \
    --horizon-days "$H" \
    --summarize
done

"$PY" runnable_code/cache_presentation_tables.py \
  --output-root outputs/rolling_continuation_hauksson_full_v3 \
  --config config/rolling_continuation_hauksson_full_v3.json
```
