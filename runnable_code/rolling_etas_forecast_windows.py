"""Pure helpers for rolling ETAS forecast window scheduling (no ETAS imports)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class StepWindows:
    step_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    forecast_start: pd.Timestamp
    forecast_end: pd.Timestamp

    def to_json_dict(self) -> dict[str, str]:
        return {
            "step_index": self.step_index,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "forecast_start": str(self.forecast_start),
            "forecast_end": str(self.forecast_end),
        }


def parse_timedelta(value: str | float | int | pd.Timedelta) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, (int, float)):
        return pd.Timedelta(days=float(value))
    return pd.Timedelta(value)


def duration_from_config(cfg: dict, *, days_key: str, alt_key: str) -> pd.Timedelta:
    if days_key in cfg:
        return parse_timedelta(cfg[days_key])
    if alt_key in cfg:
        return parse_timedelta(cfg[alt_key])
    raise KeyError(f"Config must include {days_key!r} or {alt_key!r}")


def count_steps(total_prediction_time: pd.Timedelta, prediction_window: pd.Timedelta) -> int:
    if prediction_window <= pd.Timedelta(0):
        raise ValueError("prediction_window must be positive")
    if total_prediction_time <= pd.Timedelta(0):
        return 0
    ratio = total_prediction_time / prediction_window
    return max(1, math.ceil(ratio))


def compute_step_windows(
    *,
    auxiliary_start: str | pd.Timestamp,
    timewindow_start: str | pd.Timestamp,
    timewindow_end: str | pd.Timestamp,
    finetuning_time: str | float | pd.Timedelta,
    prediction_window: str | float | pd.Timedelta,
    total_prediction_time: str | float | pd.Timedelta,
    step_index: int,
) -> StepWindows:
    if step_index < 0:
        raise ValueError("step_index must be >= 0")

    aux = pd.to_datetime(auxiliary_start, utc=True).tz_convert(None)
    tw_start_orig = pd.to_datetime(timewindow_start, utc=True).tz_convert(None)
    tw_end_orig = pd.to_datetime(timewindow_end, utc=True).tz_convert(None)
    finetune = parse_timedelta(finetuning_time)
    pred_win = parse_timedelta(prediction_window)
    total_pred = parse_timedelta(total_prediction_time)

    forecast_start = tw_end_orig + step_index * pred_win
    forecast_end_total = tw_end_orig + total_pred
    forecast_end = min(forecast_start + pred_win, forecast_end_total)

    if step_index == 0:
        train_start = tw_start_orig
        train_end = tw_end_orig
    else:
        train_end = forecast_start
        train_start = max(aux, train_end - finetune)

    return StepWindows(
        step_index=step_index,
        train_start=train_start,
        train_end=train_end,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )


def normalize_continuation_methods(methods: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for method in methods:
        key = method.strip().lower()
        if key not in ("etas", "thinning"):
            raise ValueError(f"Unknown continuation method: {method!r}")
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("At least one continuation method is required")
    return tuple(out)


def parse_continuation_methods_arg(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return normalize_continuation_methods(raw.split(","))


def continuation_methods_from_config(cfg: dict) -> tuple[str, ...]:
    raw = cfg.get("continuation_methods", ("etas", "thinning"))
    if isinstance(raw, str):
        return normalize_continuation_methods(raw.split(","))
    return normalize_continuation_methods(raw)


def catalog_names_for_methods(methods: Sequence[str]) -> tuple[str, ...]:
    method_set = {m.lower() for m in methods}
    names: list[str] = []
    if "etas" in method_set:
        names.append("etas_catalog.csv")
    if "thinning" in method_set:
        names.append("thinning_catalog.csv")
    return tuple(names)


def step_dir_name(step_index: int) -> str:
    return f"step_{step_index:03d}"
