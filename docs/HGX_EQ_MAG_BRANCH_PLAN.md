# HGX: put the no-git MAGNET tree on a branch

**Written:** 2026-09-26. For the agent on `neriberman@HGX`.

## Goal

`/data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs` has the HGX edits and **no** `.git`. Publish those files as branch `feature/hgx-clean-ifs` on `https://github.com/Ner-Ber/eq_mag_prediction.git`, then make that path a single git checkout of that branch.

Do not put this code on the etas repo. Do not commit on hanamel.

## Base

The HGX tree was copied from hanamel `eq_mag_prediction_clean-ifs`, branch `feature/magnet-featurestate-growarray` at `f5d6e09` (`Add GrowArray ingest for FeatureState and window buffers.`), which is already on `origin`. The new branch starts there. The HGX file changes are one commit on top. If `git status` is clean after the rsync, stop and do not create an empty commit.

## Steps

```bash
cd /data/neriberman/eq_mag_prediction
git clone --branch feature/magnet-featurestate-growarray \
  https://github.com/Ner-Ber/eq_mag_prediction.git eq_mag_prediction_git
cd eq_mag_prediction_git
git checkout -b feature/hgx-clean-ifs

rsync -a --delete --exclude .git \
  /data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs/ \
  /data/neriberman/eq_mag_prediction/eq_mag_prediction_git/

git status -sb
git diff --stat
```

`rsync` overwrites the clone with the HGX files. `--delete` removes clone files that HGX does not have. The source directory is only read. Jobs keep using it until the rename below.

If status shows changes:

```bash
git add -A
git commit -m "$(cat <<'EOF'
Snapshot the HGX FeatureState tree that was copied without git history.

EOF
)"
git push -u origin feature/hgx-clean-ifs
```

Check for large data files before the push (`git status` / `git diff --stat`). Do not push catalogs or model weights.

## One folder, after the push succeeds

```bash
cd /data/neriberman/eq_mag_prediction
mv eq_mag_prediction_clean-ifs eq_mag_prediction_clean-ifs.nogit
mv eq_mag_prediction_git eq_mag_prediction_clean-ifs
cd eq_mag_prediction_clean-ifs
git status -sb
git log -1 --oneline
```

`--magnet-root` stays `/data/neriberman/eq_mag_prediction/eq_mag_prediction_clean-ifs`. That path is now branch `feature/hgx-clean-ifs`. Delete `eq_mag_prediction_clean-ifs.nogit` only after an import of `eq_mag_prediction.forecasting.incremental_feature_state` works from the new tree.

## Do not

- Do not `rsync` in the other direction.
- Do not rename the live tree before `git push` succeeds.
- Do not point rolling jobs at `eq_mag_prediction_clean` (no FeatureState).
