# Imports
from datetime import datetime
from datetime import timezone
from pathlib import Path
import subprocess
import sys
import re
import ast
import json
import os
import inspect
import tempfile
import shutil
import argparse
import gin
import pandas as pd
import numpy as np

from eq_mag_prediction.ingestion import catalog_format_converter as cfc
from eq_mag_prediction.utilities import data_utils
from eq_mag_prediction.forecasting import one_region_model
from eq_mag_prediction.forecasting import training_examples
from eq_mag_prediction.utilities import catalog_processing

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation


def run_subprocess(process_path, gin_path, **flags):
    gin_flag = 'gin_config' if process_path.endswith("magnitude_predictor_trainer.py") else 'gin_path'

    # set flag for subprocess
    cmd_flags = []
    for key, value in flags.items():
        cmd_flags.append(f"--{key}={value}")

    command = [
        sys.executable,  # Uses the same python interpreter running this script
        "-u",  # force unbuffered output (crucial for real-time visibility)
        process_path,
        f"--{gin_flag}={gin_path}",
    ] + cmd_flags
    print("--- Starting Subprocess ---")
    print("Command:", " ".join(command))
    try:
        # Run from etas_edits directory to avoid numpy source directory conflicts
        # The scripts should work regardless of CWD since they use absolute paths
        script_dir = Path(__file__).resolve().parent

        env = os.environ.copy()

        # --- CRITICAL FIX START ---
        # Explicitly remove the legacy flag if it exists. 
        # This forces the subprocess to use the standard Keras 2.15 bundled with TF 2.15.
        if "TF_USE_LEGACY_KERAS" in env:
            del env["TF_USE_LEGACY_KERAS"]
        # --- CRITICAL FIX END ---

        # Ensure LD_LIBRARY_PATH includes conda's lib directory for proper libstdc++ resolution
        # This helps with GLIBCXX version issues
        conda_env = os.environ.get('CONDA_PREFIX', '')
        if conda_env:
            lib_path = os.path.join(conda_env, 'lib')
            existing_ld_path = env.get('LD_LIBRARY_PATH', '')
            if existing_ld_path:
                if lib_path not in existing_ld_path:
                    env['LD_LIBRARY_PATH'] = lib_path + os.pathsep + existing_ld_path
            else:
                env['LD_LIBRARY_PATH'] = lib_path

        # No capture_output=True, so it prints directly to console
        subprocess.run(command, check=True, cwd=str(script_dir), env=env)
        print("--- Subprocess Finished Successfully ---")

    except subprocess.CalledProcessError as e:
        print(f"--- Subprocess Failed (Exit Code: {e.returncode}) ---")
        # Optional: Stop the pipeline if this step fails
        raise e


def run_python_script(script_path: str, **flags):
    """
    Run a Python script as: python <script_path> --some_flag=1

    (Used for ETAS runnable scripts that do not accept gin_path/gin_config flags.)
    """
    cmd_flags = []
    for key, value in flags.items():
        cmd_flags.append(f"--{key}={value}")

    command = [sys.executable, '-u', script_path] + cmd_flags
    print('--- Starting Subprocess ---')
    print('Command:', ' '.join(command))
    try:
        # Run from etas_edits directory to avoid numpy source directory conflicts
        script_dir = Path(__file__).resolve().parent
        env = os.environ.copy()
        # Ensure LD_LIBRARY_PATH includes conda's lib directory for proper libstdc++ resolution
        conda_env = os.environ.get('CONDA_PREFIX', '')
        if conda_env:
            lib_path = os.path.join(conda_env, 'lib')
            existing_ld_path = env.get('LD_LIBRARY_PATH', '')
            if existing_ld_path:
                if lib_path not in existing_ld_path:
                    env['LD_LIBRARY_PATH'] = lib_path + os.pathsep + existing_ld_path
            else:
                env['LD_LIBRARY_PATH'] = lib_path
        subprocess.run(command, check=True, cwd=str(script_dir), env=env)
        print('--- Subprocess Finished Successfully ---')

    except subprocess.CalledProcessError as e:
        print(f"--- Subprocess Failed (Exit Code: {e.returncode}) ---")
        raise e


def _dt_string_to_epoch_seconds_utc(dt_str: str) -> int:
    """Parse '%Y-%m-%d %H:%M:%S' as UTC and return epoch seconds."""
    return int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def _build_temp_configs_from_single_source(
    *,
    pipeline_config_path: str,
    val_to_train_time_ratio: float,
) -> dict[str, str]:
    """
    Single-source config mode:
    - User provides ONE JSON file with template paths + overrides.
    - We generate temp gin/json configs (and catalogs) and run MAGNET/ETAS off those.
    - We do NOT modify the template configs.

    Expected pipeline_config JSON shape:
    {
      "templates": {
        "general_gin_config_path": "...",
        "local_gin_config_path": "...",
        "invert_etas_config_json_path": "...",
        "etas_catalog_continuation_config_json_path": "..."
      },
      "overrides": {
        "set_times": {
          "auxiliary_start": "YYYY-mm-dd HH:MM:SS",
          "timewindow_start": "YYYY-mm-dd HH:MM:SS",
          "timewindow_end": "YYYY-mm-dd HH:MM:SS",
          "testwindow_end": "YYYY-mm-dd HH:MM:SS"
        },
        "catalog": { "format": "etas" | "magnet", "path": "/abs/or/rel/path/to/catalog.csv" },
        "train_and_evaluate_magnitude_prediction_model": { "learning_rate": 1e-4, ... },
        ... (other gin parameters)
      }
    }
    """
    with open(pipeline_config_path, "r") as f:
        pcfg = json.load(f)

    templates = pcfg["templates"]
    overrides = pcfg["overrides"]

    tmp_root = Path(tempfile.mkdtemp(prefix="magnet_etas_pipeline_"))
    tmp_cfg_dir = tmp_root / "configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    tmp_data_dir = tmp_root / "data"
    tmp_data_dir.mkdir(parents=True, exist_ok=True)

    # Copy template config files into temp working set
    # Keep original filename for general.gin so include statements work
    tmp_general_gin = tmp_cfg_dir / "magnitude_prediction_general.gin"
    tmp_local_gin = tmp_cfg_dir / "local.gin"
    tmp_invert_json = tmp_cfg_dir / "invert_etas.json"
    tmp_cont_json = tmp_cfg_dir / "simulate_catalog_continuation.json"

    shutil.copyfile(templates["general_gin_config_path"], tmp_general_gin)
    shutil.copyfile(templates["local_gin_config_path"], tmp_local_gin)
    shutil.copyfile(templates["invert_etas_config_json_path"], tmp_invert_json)
    shutil.copyfile(templates["etas_catalog_continuation_config_json_path"], tmp_cont_json)

    # Update include path in temp local.gin to use absolute path so gin can find it
    # Gin requires double quotes for absolute paths, and paths should use forward slashes
    local_gin_content = _read_text_file(str(tmp_local_gin))
    # Replace relative include with absolute path (use double quotes and forward slashes)
    abs_path_str = str(tmp_general_gin).replace('\\', '/')
    local_gin_content = local_gin_content.replace(
        "include 'magnitude_prediction_general.gin'",
        f'include "{abs_path_str}"'
    )
    with open(tmp_local_gin, 'w') as f:
        f.write(local_gin_content)

    # --- Catalog: default is MAGNET format
    catalog_spec = overrides["catalog"]
    catalog_format = catalog_spec.get("format", "magnet")  # Default to MAGNET
    catalog_path = Path(catalog_spec["path"])
    if not catalog_path.is_absolute():
        catalog_path = (Path(pipeline_config_path).resolve().parent / catalog_path).resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    # MAGNET catalog: create in persistent location (results/catalogs/ingested/) with collision detection
    ingested_dir = Path(data_utils.INGESTED_DIRECTORY).resolve()
    persistent_magnet_catalog = _find_or_create_magnet_catalog(
        source_catalog_path=catalog_path,
        source_format=catalog_format,
        ingested_dir=ingested_dir,
    )

    # ETAS catalog: create in persistent location with collision detection
    # Determine ETAS catalog directory from template ETAS config
    template_etas_json = Path(templates["invert_etas_config_json_path"])
    etas_config_dir = template_etas_json.resolve().parent
    etas_catalog_dir = (etas_config_dir / ".." / "input_data" / "catalogs").resolve()

    if catalog_format == "magnet":
        # Create ETAS catalog from the persistent MAGNET catalog
        persistent_etas_catalog = _find_or_create_etas_catalog_for_magnet(
            magnet_catalog_path=persistent_magnet_catalog,
            etas_output_dir=etas_catalog_dir,
            original_catalog_name=persistent_magnet_catalog.name,
        )
        tmp_etas_catalog_rel = str(persistent_etas_catalog)
    elif catalog_format == "etas":
        # User provided ETAS catalog; create MAGNET catalog from it (already done above)
        # Use the provided ETAS catalog (relative to temp config)
        tmp_etas_catalog_rel = os.path.relpath(catalog_path, tmp_cfg_dir)
    else:
        raise ValueError(f"Unsupported catalog.format: {catalog_format} (expected 'magnet' or 'etas')")

    # --- Update temp LOCAL gin: times + catalog binding (absolute path so look_for_file works)
    # Get timing fields from "set_times" subfield
    set_times = overrides["set_times"]
    update_gin = {
        "feature_prep_start": _dt_string_to_epoch_seconds_utc(set_times["auxiliary_start"]),
        "train_start_time": _dt_string_to_epoch_seconds_utc(set_times["timewindow_start"]),
        "test_start_time": _dt_string_to_epoch_seconds_utc(set_times["timewindow_end"]),
        "test_end_time": _dt_string_to_epoch_seconds_utc(set_times["testwindow_end"]),
    }
    update_gin["validation_start_time"] = int((1 - val_to_train_time_ratio) * update_gin["train_start_time"] + val_to_train_time_ratio * update_gin["test_start_time"])

    local_gin_dict = parse_gin_config(_read_text_file(str(tmp_local_gin)))
    current_binding = local_gin_dict.get("bindings", {}).get("catalog", "@jma_dataframe()")
    func_name, file_param_name, _ = _parse_gin_catalog_binding(current_binding)
    if file_param_name is None:
        file_param_name, _ = _default_filename_for_data_utils_function(func_name)
    # Point gin config to persistent MAGNET catalog (use filename only, look_for_file will find it).
    #
    # IMPORTANT: gin-config in this environment does NOT support keyword-args inside @call() bindings
    # (e.g. @hauksson_dataframe(csv_path='...') triggers "Expected ')'").
    # Instead, set the configurable parameter via the standard Gin assignment syntax:
    #   hauksson_dataframe.csv_path = '...'
    update_gin["catalog"] = f"@{func_name}()"
    update_gin[f"{func_name}.{file_param_name}"] = persistent_magnet_catalog.name

    # Add other overrides (not "set_times" or "catalog") to gin file
    # Flatten nested dictionaries using dot notation for gin parameters
    def flatten_dict(d, parent_key='', sep='.'):
        """Flatten nested dictionary using dot notation."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    for key, value in overrides.items():
        if key not in ["set_times", "catalog"]:
            if isinstance(value, dict):
                # Flatten nested dictionaries
                flattened = flatten_dict(value, parent_key=key)
                update_gin.update(flattened)
            else:
                # Simple key-value pair
                update_gin[key] = value

    update_gin_parameters(str(tmp_local_gin), update_gin)

    # --- Update temp ETAS inversion json: times + fn_catalog + data_path (temp output dir)
    etas_out_dir = tmp_root / "etas_output"
    etas_out_dir.mkdir(parents=True, exist_ok=True)
    update_json = {
        "auxiliary_start": set_times["auxiliary_start"],
        "timewindow_start": set_times["timewindow_start"],
        "timewindow_end": set_times["timewindow_end"],
        "testwindow_end": set_times["testwindow_end"],
        "fn_catalog": tmp_etas_catalog_rel,
        "data_path": str(etas_out_dir) + os.sep,
    }
    update_json_parameters(str(tmp_invert_json), update_json)

    # --- Update temp continuation json: keep forecast_duration, but write outputs into temp dir
    sim_out_csv = etas_out_dir / "simulated_catalog_continuation.csv"
    update_json_parameters(
        str(tmp_cont_json),
        {"fn_store_simulation": str(sim_out_csv)},
    )

    return {
        "tmp_root": str(tmp_root),
        "general_gin_config_path": str(tmp_general_gin),
        "local_gin_config_path": str(tmp_local_gin),
        "invert_etas_config_json_path": str(tmp_invert_json),
        "etas_catalog_continuation_config_json_path": str(tmp_cont_json),
    }

# 1. Set the configuration files and load them.
# TODO: decide on format. Perhaps a JSON then a build from that to gin to be readable by MAGNET code?
# common: timings and filtering events by shape
# Hardcoded settings for now (no terminal/env overrides)
# - True: copy ETAS JSON fields -> MAGNET Gin
# - False: copy MAGNET Gin fields -> ETAS JSON


def _read_text_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def _epoch_seconds_to_dt_string(epoch_seconds: int) -> str:
    """
    Convert epoch seconds to ETAS JSON datetime string format: '%Y-%m-%d %H:%M:%S'
    Uses UTC to avoid local timezone drift.
    """
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _get_catalog_path_from_gin_binding(catalog_binding: str) -> Path:
    """
    Extract the catalog file path from a Gin config catalog binding.

    Parses strings like '@jma_dataframe()' or '@jma_dataframe(csv_path='jma.csv')'
    and determines the actual file path using the same logic as data_utils functions.

    Args:
        catalog_binding: The catalog binding string from Gin config, e.g. '@jma_dataframe()'

    Returns:
        Path to the catalog file.

    Raises:
        ValueError: If the binding string cannot be parsed.
        RuntimeError: If the catalog file cannot be found.
    """
    if not catalog_binding.startswith('@'):
        raise ValueError(f"Expected Gin binding starting with '@', got: {catalog_binding}")

    # Parse the binding string: '@function_name(arguments)'
    # Match pattern: @function_name() or @function_name(param='value', ...)
    match = re.match(r'@(\w+)\((.*)\)', catalog_binding)
    if not match:
        raise ValueError(f"Could not parse Gin binding: {catalog_binding}")

    function_name = match.group(1)
    args_str = match.group(2).strip()

    # Extract csv_path parameter (or similar file path parameter)
    # Handle cases like: csv_path='jma.csv', cdf_path='jma.cdf', etc.
    file_param_pattern = re.compile(r'(csv_path|cdf_path|json_path)\s*=\s*[\'"]([^\'"]+)[\'"]')
    param_match = file_param_pattern.search(args_str)

    if param_match:
        filename = param_match.group(2)
    else:
        # No explicit file parameter - get default from the function signature in data_utils
        if not hasattr(data_utils, function_name):
            raise ValueError(f"Function '{function_name}' not found in data_utils module")

        func = getattr(data_utils, function_name)
        sig = inspect.signature(func)

        # Find the file path parameter (csv_path, cdf_path, etc.)
        filename = None
        for param_name, param in sig.parameters.items():
            if param_name.endswith('_path') or param_name.endswith('path'):
                if param.default != inspect.Parameter.empty:
                    filename = param.default
                    break

        # Fallback if no default found
        if filename is None:
            # Try common fallback patterns
            if 'dataframe' in function_name:
                base_name = function_name.replace('_dataframe', '').replace('_from_proto', '')
                filename = f"{base_name}.csv" if 'proto' not in function_name else f"{base_name}.cdf"
            else:
                raise ValueError(f"Could not determine default filename for function '{function_name}'")

    # Use data_utils.look_for_file() directly to find the catalog file
    # This ensures we use the exact same lookup logic as the data_utils functions
    try:
        catalog_path = data_utils.look_for_file(filename)
        return Path(catalog_path).resolve()
    except RuntimeError as e:
        raise RuntimeError(
            f"Catalog file not found: {filename}\n"
            f"Original error: {e}\n"
            f"Have you ingested the relevant catalog? See README.md."
        )

def _parse_gin_catalog_binding(catalog_binding: str) -> tuple[str, str | None, str | None]:
    """
    Parse a gin catalog binding like '@jma_dataframe()' or '@jma_dataframe(csv_path=\"jma.csv\")'.

    Returns:
      (function_name, file_param_name, filename)
        - file_param_name / filename may be None if no explicit file argument is present.
    """
    if not catalog_binding.startswith("@"):
        raise ValueError(f"Expected Gin binding starting with '@', got: {catalog_binding}")

    match = re.match(r"@(\w+)\((.*)\)", catalog_binding)
    if not match:
        raise ValueError(f"Could not parse Gin binding: {catalog_binding}")

    function_name = match.group(1)
    args_str = match.group(2).strip()

    file_param_pattern = re.compile(r"(csv_path|cdf_path|json_path)\s*=\s*['\"]([^'\"]+)['\"]")
    param_match = file_param_pattern.search(args_str)
    if param_match:
        return function_name, param_match.group(1), param_match.group(2)

    return function_name, None, None

def _default_filename_for_data_utils_function(function_name: str) -> tuple[str, str]:
    """
    For a data_utils catalog loader function, determine (file_param_name, default_filename)
    by inspecting the function signature.
    """
    if not hasattr(data_utils, function_name):
        raise ValueError(f"Function '{function_name}' not found in data_utils module")

    func = getattr(data_utils, function_name)
    sig = inspect.signature(func)

    for param_name, param in sig.parameters.items():
        if param_name.endswith("_path") or param_name.endswith("path"):
            if param.default != inspect.Parameter.empty:
                return param_name, str(param.default)

    # Fallback heuristic if signature doesn't expose a default path param.
    if "dataframe" in function_name:
        base_name = function_name.replace("_dataframe", "").replace("_from_proto", "")
        default = f"{base_name}.cdf" if "proto" in function_name else f"{base_name}.csv"
        return "csv_path", default

    raise ValueError(f"Could not determine default filename for function '{function_name}'")

def _safe_unique_path(desired_path: Path) -> Path:
    """
    Return desired_path if it doesn't exist; otherwise append a numeric suffix before the extension.
    Never overwrites existing files.
    """
    if not desired_path.exists():
        return desired_path

    stem = desired_path.stem
    suffix = desired_path.suffix
    parent = desired_path.parent

    for i in range(1, 10_000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find a free filename near {desired_path}")

def _magnet_catalogs_equivalent(
    path_a: Path,
    path_b: Path,
    *,
    chunksize: int = 200_000,
    float_rtol: float = 0.0,
    float_atol: float = 1e-10,
) -> bool:
    """
    Compare two MAGNET-format catalogs for data equivalence (not byte-identical).

    MAGNET ingested catalog schema we care about: time, latitude, longitude, magnitude.
    Additional columns are allowed, but must match if present in both.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    # Fast path: if sizes differ wildly, still could be equal due to formatting;
    # so don't early-return based on size alone.

    reader_a = pd.read_csv(path_a, chunksize=chunksize)
    reader_b = pd.read_csv(path_b, chunksize=chunksize)

    for chunk_a, chunk_b in zip(reader_a, reader_b):
        if list(chunk_a.columns) != list(chunk_b.columns):
            return False
        if len(chunk_a) != len(chunk_b):
            return False

        for col in chunk_a.columns:
            a = chunk_a[col].to_numpy()
            b = chunk_b[col].to_numpy()

            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                if not np.allclose(a, b, rtol=float_rtol, atol=float_atol, equal_nan=True):
                    return False
            else:
                # String/object columns
                if not np.array_equal(a, b):
                    return False

    # Ensure both readers are exhausted (same number of chunks)
    try:
        next(reader_a)
        return False
    except StopIteration:
        pass
    try:
        next(reader_b)
        return False
    except StopIteration:
        pass

    return True

def _find_matching_catalog_in_dir(
    *,
    candidate_path: Path,
    search_dir: Path,
    temp_converted_path: Path,
) -> Path | None:
    """
    Look for an existing file in search_dir that is equivalent to temp_converted_path.
    Checks candidate_path first (if it exists), then scans suffix variants.
    """
    search_dir = Path(search_dir)
    candidate_path = Path(candidate_path)
    temp_converted_path = Path(temp_converted_path)

    # 1) Check the canonical candidate first
    if candidate_path.exists() and _magnet_catalogs_equivalent(candidate_path, temp_converted_path):
        return candidate_path

    # 2) Check suffix variants: stem_*.suffix
    stem = candidate_path.stem
    suffix = candidate_path.suffix
    for existing in sorted(search_dir.glob(f"{stem}_*{suffix}")):
        if existing.is_file() and _magnet_catalogs_equivalent(existing, temp_converted_path):
            return existing

    return None

def _ensure_magnet_catalog_for_etas_catalog(
    *,
    etas_catalog_path: Path,
    current_gin_catalog_binding: str,
) -> tuple[Path, str]:
    """
    Ensure a MAGNET ingested catalog CSV exists for the given ETAS catalog.

    IMPORTANT: Never overwrites an existing file in results/catalogs/ingested.

    Returns:
      (magnet_catalog_path, gin_catalog_binding_to_use)
    """
    function_name, file_param_name, filename = _parse_gin_catalog_binding(current_gin_catalog_binding)

    # Determine the output filename we'd like to use.
    if filename is None or file_param_name is None:
        file_param_name, filename = _default_filename_for_data_utils_function(function_name)

    ingested_dir = Path(data_utils.INGESTED_DIRECTORY).resolve()
    ingested_dir.mkdir(parents=True, exist_ok=True)

    desired_output = ingested_dir / filename

    # Convert ETAS -> MAGNET once into a temp file, then compare against existing ingested catalogs.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"Converting ETAS -> MAGNET to temp:\n  {etas_catalog_path} -> {tmp_path}")
        cfc.convert_etas_to_magnet(etas_catalog_path, tmp_path)

        # If an existing catalog matches (default or suffixed), reuse it and delete temp.
        existing_match = _find_matching_catalog_in_dir(
            candidate_path=desired_output,
            search_dir=ingested_dir,
            temp_converted_path=tmp_path,
        )
        if existing_match is not None:
            tmp_path.unlink(missing_ok=True)

            # Update binding if it explicitly points elsewhere or if match isn't the default name.
            if existing_match.name == filename and desired_output == existing_match:
                return existing_match, current_gin_catalog_binding

            new_binding = f"@{function_name}({file_param_name}='{existing_match.name}')"
            return existing_match, new_binding

        # No match exists; promote temp to a new unique filename WITHOUT overwriting.
        output_path = _safe_unique_path(desired_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, output_path)

        # If we used the default name, keep binding unchanged; otherwise point gin to new filename.
        if output_path.name == filename and desired_output == output_path:
            return output_path, current_gin_catalog_binding

        new_binding = f"@{function_name}({file_param_name}='{output_path.name}')"
        return output_path, new_binding
    finally:
        # Cleanup temp if it still exists (e.g., exceptions)
        if tmp_path.exists():
            tmp_path.unlink()

def _etas_catalogs_equivalent(
    path_a: Path,
    path_b: Path,
    *,
    chunksize: int = 200_000,
    float_rtol: float = 0.0,
    float_atol: float = 1e-10,
) -> bool:
    """
    Compare two ETAS-format catalogs for data equivalence (not byte-identical).

    ETAS catalog schema: id, latitude, longitude, time, magnitude.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)

    reader_a = pd.read_csv(path_a, chunksize=chunksize)
    reader_b = pd.read_csv(path_b, chunksize=chunksize)

    for chunk_a, chunk_b in zip(reader_a, reader_b):
        # ETAS catalogs: id, latitude, longitude, time, magnitude
        required_cols = ["latitude", "longitude", "time", "magnitude"]
        if not all(col in chunk_a.columns for col in required_cols):
            return False
        if not all(col in chunk_b.columns for col in required_cols):
            return False
        if len(chunk_a) != len(chunk_b):
            return False

        # Compare data columns (ignore id column for comparison)
        for col in required_cols:
            a = chunk_a[col].to_numpy()
            b = chunk_b[col].to_numpy()
            if col in ["latitude", "longitude", "magnitude"]:
                if not np.allclose(a, b, rtol=float_rtol, atol=float_atol, equal_nan=True):
                    return False
            elif col == "time":
                # Time is a string, compare as strings
                if not np.array_equal(a, b):
                    return False

    return True


def _find_or_create_magnet_catalog(
    *,
    source_catalog_path: Path,
    source_format: str,
    ingested_dir: Path | None = None,
) -> Path:
    """
    Ensure a MAGNET catalog exists in the ingested directory with collision detection.

    If source is MAGNET format: copy to ingested_dir with collision detection.
    If source is ETAS format: convert to MAGNET and save to ingested_dir with collision detection.

    Args:
        source_catalog_path: Path to source catalog (MAGNET or ETAS format)
        source_format: "magnet" or "etas"
        ingested_dir: Directory where MAGNET catalogs should be stored (defaults to INGESTED_DIRECTORY)

    Returns:
        Path to the MAGNET catalog in ingested_dir (may be existing or newly created)
    """
    if ingested_dir is None:
        ingested_dir = Path(data_utils.INGESTED_DIRECTORY).resolve()
    ingested_dir.mkdir(parents=True, exist_ok=True)

    source_catalog_path = Path(source_catalog_path)
    desired_output = ingested_dir / source_catalog_path.name

    # Create temp file with MAGNET format
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if source_format == "magnet":
            print(f"Copying MAGNET catalog to temp:\n  {source_catalog_path} -> {tmp_path}")
            shutil.copyfile(source_catalog_path, tmp_path)
        elif source_format == "etas":
            print(f"Converting ETAS -> MAGNET to temp:\n  {source_catalog_path} -> {tmp_path}")
            cfc.convert_etas_to_magnet(source_catalog_path, tmp_path)
        else:
            raise ValueError(f"Unsupported source_format: {source_format} (expected 'magnet' or 'etas')")

        # Check if desired_output exists and is identical
        if desired_output.exists() and _magnet_catalogs_equivalent(desired_output, tmp_path):
            print(f"MAGNET catalog already exists and is identical: {desired_output}")
            tmp_path.unlink(missing_ok=True)
            return desired_output

        # Check suffix variants: <name>_1.csv, _2.csv, etc.
        existing_match = _find_matching_catalog_in_dir(
            candidate_path=desired_output,
            search_dir=ingested_dir,
            temp_converted_path=tmp_path,
        )
        if existing_match is not None:
            print(f"Found identical MAGNET catalog: {existing_match}")
            tmp_path.unlink(missing_ok=True)
            return existing_match

        # No match found; create new file with unique name
        output_path = _safe_unique_path(desired_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, output_path)
        print(f"Created new MAGNET catalog: {output_path}")
        return output_path
    finally:
        # Cleanup temp if it still exists
        if tmp_path.exists():
            tmp_path.unlink()


def _find_or_create_etas_catalog_for_magnet(
    *,
    magnet_catalog_path: Path,
    etas_output_dir: Path,
    original_catalog_name: str,
) -> Path:
    """
    Create ETAS catalog from MAGNET catalog with collision detection.

    Naming pattern: etas_converted_<original_name>.csv
    If exists and identical, reuse it. If different, try _1, _2, etc.

    Args:
        magnet_catalog_path: Path to source MAGNET catalog
        etas_output_dir: Directory where ETAS catalogs should be stored
        original_catalog_name: Original catalog filename (without path)

    Returns:
        Path to the ETAS catalog (may be existing or newly created)
    """
    etas_output_dir = Path(etas_output_dir)
    etas_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate base name: etas_converted_<original_name>.csv
    original_stem = Path(original_catalog_name).stem
    base_name = f"etas_converted_{original_stem}.csv"
    desired_path = etas_output_dir / base_name

    # Convert MAGNET -> ETAS to temp file first
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"Converting MAGNET -> ETAS to temp:\n  {magnet_catalog_path} -> {tmp_path}")
        cfc.convert_magnet_to_etas(magnet_catalog_path, tmp_path)

        # Check if desired_path exists and is identical
        if desired_path.exists() and _etas_catalogs_equivalent(desired_path, tmp_path):
            print(f"ETAS catalog already exists and is identical: {desired_path}")
            tmp_path.unlink(missing_ok=True)
            return desired_path

        # Check suffix variants: etas_converted_<name>_1.csv, _2.csv, etc.
        stem = desired_path.stem
        suffix = desired_path.suffix
        for existing in sorted(etas_output_dir.glob(f"{stem}_*{suffix}")):
            if existing.is_file() and _etas_catalogs_equivalent(existing, tmp_path):
                print(f"Found identical ETAS catalog: {existing}")
                tmp_path.unlink(missing_ok=True)
                return existing

        # No match found; create new file with unique name
        output_path = _safe_unique_path(desired_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, output_path)
        print(f"Created new ETAS catalog: {output_path}")
        return output_path
    finally:
        # Cleanup temp if it still exists
        if tmp_path.exists():
            tmp_path.unlink()


def _ensure_etas_catalog_for_magnet_catalog(
    *,
    magnet_catalog_path: Path,
    invert_etas_config_json_path: str,
    current_fn_catalog_value: str | None,
) -> str:
    """
    Ensure an ETAS-format catalog exists for a given MAGNET-format catalog path.

    Returns the ETAS catalog path as a string **relative** to the ETAS JSON config file.
    """
    json_dir = Path(invert_etas_config_json_path).parent
    etas_catalog_dir = json_dir / ".." / "input_data" / "catalogs"
    etas_catalog_dir = etas_catalog_dir.resolve()

    etas_catalog_path = _find_or_create_etas_catalog_for_magnet(
        magnet_catalog_path=magnet_catalog_path,
        etas_output_dir=etas_catalog_dir,
        original_catalog_name=magnet_catalog_path.name,
    )

    return os.path.relpath(etas_catalog_path, json_dir)

def parse_gin_config(content: str) -> dict:
    """
    Parses a gin config string into a dictionary.

    Handles:
    - Global variables
    - Scoped configurations (indented blocks)
    - Function.parameter syntax
    - Includes
    - Python literals (tuples, lists, numbers, booleans)
    - Gin references (@ and %)

    Args:
        content (str): The gin configuration file content.

    Returns:
        dict: A dictionary representation of the config.
    """
    config = {
        'includes': [],
        'bindings': {}
    }

    current_scope = None

    # Regex to capture assignments (key = value)
    # This handles cases where value might contain '=' (though rare in gin keys)
    assignment_pattern = re.compile(r'^([^=]+)\s*=\s*(.*)$')

    lines = content.split('\n')

    for line in lines:
        raw_line = line

        # 1. Remove comments and strip whitespace
        # We split by '#' but need to be careful not to split inside strings
        # Simple split is usually sufficient for config files
        if '#' in line:
            line = line.split('#', 1)[0]

        # Check for indentation to determine scope handling
        indent_level = len(line) - len(line.lstrip())
        stripped_line = line.strip()

        if not stripped_line:
            continue

        # 2. Handle Includes
        if stripped_line.startswith('include '):
            # Extract filename, removing quotes
            filename = stripped_line.split(' ', 1)[1].strip("'\"")
            config['includes'].append(filename)
            continue

        # 3. Handle Scopes (e.g., "estimate_completeness:")
        if stripped_line.endswith(':'):
            current_scope = stripped_line[:-1]
            if current_scope not in config['bindings']:
                config['bindings'][current_scope] = {}
            continue

        # 4. Handle Resetting Scope (if indentation drops)
        # In Gin, top-level definitions usually have 0 indentation. 
        # Scoped definitions are indented.
        if indent_level == 0 and not stripped_line.endswith(':'):
            current_scope = None

        # 5. Handle Assignments (key = value)
        match = assignment_pattern.match(stripped_line)
        if match:
            key, value_str = match.groups()
            key = key.strip()
            value_str = value_str.strip()

            # Attempt to parse the value into a Python type
            parsed_value = _parse_gin_value(value_str)

            if current_scope:
                config['bindings'][current_scope][key] = parsed_value
            else:
                config['bindings'][key] = parsed_value

    return config

def _parse_gin_value(value_str: str):
    """
    Helper to parse string values into Python types (int, float, tuple, etc.)
    Returns the string literal if it's a macro (%) or reference (@).
    """
    # Return immediately for Gin References/Macros
    if value_str.startswith(('@', '%')):
        return value_str

    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # Fallback: return as string if it can't be parsed (e.g., unquoted strings)
        return value_str

def update_gin_parameters(gin_path: str, params_dict: dict):
    """
    Updates multiple parameters in a Gin config file based on a dictionary.

    Args:
        gin_path (str): Path to the .gin file.
        params_dict (dict): A dictionary of {parameter_name: new_value}.
                            e.g. {'learning_rate': 1e-3, 'target_catalog.earthquake_criterion': '@new_config'}
    """

    def format_value(val):
        """Helper to format values for Gin syntax."""
        if isinstance(val, str):
            # Do not quote macros (@) or references (%)
            if val.strip().startswith(('@', '%')):
                return val
            return f"'{val}'"
        # Convert numbers/bools/lists to string representation
        return str(val)

    # Track which keys we have successfully updated in the file
    updated_keys = set()
    new_lines = []

    # Regex to capture: indentation, parameter name, and the rest (value + comment)
    # Group 1: Indentation
    # Group 2: The parameter key (matched non-greedily until the =)
    # Group 3: The rest of the line (value and comments)
    assignment_pattern = re.compile(r'^(\s*)([^#=\s]+)\s*=\s*(.*)$')

    try:
        with open(gin_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            match = assignment_pattern.match(line)
            if match:
                indent, key_in_file, rest = match.groups()

                # Check if this line's key is in our update dictionary
                if key_in_file in params_dict:
                    new_val_str = format_value(params_dict[key_in_file])

                    # Construct new line, preserving indentation
                    # We assume we overwrite the previous value entirely
                    new_lines.append(f"{indent}{key_in_file} = {new_val_str}\n")
                    updated_keys.add(key_in_file)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # Append parameters that weren't found in the file
        missing_keys = set(params_dict.keys()) - updated_keys
        if missing_keys:
            new_lines.append("\n# --- Parameters added by update script ---\n")
            for key in missing_keys:
                val_str = format_value(params_dict[key])
                new_lines.append(f"{key} = {val_str}\n")
                print(f"Appended new parameter: {key}")

        with open(gin_path, 'w') as f:
            f.writelines(new_lines)

        print(f"Successfully updated {len(updated_keys)} parameters in {gin_path}")

    except FileNotFoundError:
        print(f"Error: File {gin_path} not found.")

def update_json_parameters(json_path: str, params_dict: dict):
    """
    Updates multiple parameters in a JSON file based on a dictionary.
    Supports nested keys via dot notation (e.g., 'theta_0.log10_mu').

    Args:
        json_path (str): Path to the .json file.
        params_dict (dict): A dictionary of {key: new_value}.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        for param_key, new_value in params_dict.items():
            keys = param_key.split('.')
            current = data

            # Navigate to the correct nesting level
            for i, key in enumerate(keys[:-1]):
                if key not in current:
                    print(f"Creating missing nested key: '{key}'")
                    current[key] = {}
                current = current[key]

                # Safety check: ensure we aren't trying to access a field of a non-dict
                if not isinstance(current, dict):
                    print(f"Error: Key '{key}' in path '{param_key}' is not a dictionary. Cannot traverse further.")
                    break
            else:
                # This executes only if the loop wasn't broken (valid path)
                last_key = keys[-1]
                current[last_key] = new_value
                # print(f"Updated '{param_key}'")

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Successfully updated JSON parameters in {json_path}")

    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_path}.")



# 2. Train MAGNET on the auxiliary catalog. Save trained model.

# 2a. Load auxiliary catalog.

# 2b. Prepare features and labels.
def run_feature_computation(gin_path, **flags):
    # Resolve script path relative to this file's location
    script_dir = Path(__file__).resolve().parent
    script_path = (script_dir / ".." / ".." / "eq_mag_prediction" / "eq_mag_prediction" / "scripts" / "magnitude_prediction_compute_features.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"MAGNET feature computation script not found: {script_path}")
    run_subprocess(
        str(script_path),
        gin_path,
        **flags
    )


# 2c. Train model and save it.

def _get_model_id_from_gin_config(gin_path: str) -> str | None:
    """
    Generate model ID from gin config without actually training.

    Args:
        gin_path: Path to gin config file

    Returns:
        Model ID string if successful, None if generation fails
    """
    try:
        gin.parse_config_file(gin_path, skip_unknown=True)
        # Create domain from gin config
        domain = training_examples.CatalogDomain()
        # Hash catalog if available
        if hasattr(domain, 'earthquakes_catalog') and domain.earthquakes_catalog is not None:
            catalog_hash = catalog_processing.hash_pandas_object(domain.earthquakes_catalog)
        domain_id = domain.domain_examples_uuid()
        all_encoders = one_region_model.build_encoders(domain)
        encoder_ids = []
        for name in sorted(all_encoders.keys()):
            encoder = all_encoders[name]
            encoder_identifier = encoder.uuid()
            build_features_identifier = encoder.build_features_uuid()
            encoder_ids.append(f'{name}_{encoder_identifier}_build_features_{build_features_identifier}')
        encoders_id = '_'.join(encoder_ids)

        # Extract hyperparameters from gin config
        try:
            learning_rate = gin.query_parameter('train_and_evaluate_magnitude_prediction_model.learning_rate')
        except ValueError:
            learning_rate = None

        try:
            batch_size = gin.query_parameter('train_and_evaluate_magnitude_prediction_model.batch_size')
        except ValueError:
            batch_size = None

        try:
            epochs = gin.query_parameter('train_and_evaluate_magnitude_prediction_model.epochs')
        except ValueError:
            epochs = None

        try:
            pdf_support_stretch = gin.query_parameter('train_and_evaluate_magnitude_prediction_model.pdf_support_stretch')
        except ValueError:
            pdf_support_stretch = 7  # default

        # Check if we have all required parameters
        if learning_rate is None or batch_size is None or epochs is None:
            print("ERROR: Missing required hyperparameters!")
            return None
        # Generate model ID
        model_id = one_region_model.model_training_id(
            domain=domain,
            all_encoders=all_encoders,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            pdf_support_stretch=pdf_support_stretch,
            loss_function_type=None,  # Will be determined during training
        )
        gin.clear_config()
        return model_id
    except Exception as e:
        print(f"Warning: Could not generate model ID from gin config: {e}")
        return None


def run_magnet_trainer(gin_path, output_dir=None, **flags):
    """
    Run MAGNET trainer.

    Args:
        gin_path: Path to gin config file
        output_dir: Directory for saving trained model (required)
        **flags: Additional flags to pass to the subprocess
    """
    if output_dir is None:
        raise ValueError("output_dir is required for run_magnet_trainer")

    # Resolve script path relative to this file's location
    script_dir = Path(__file__).resolve().parent
    script_path = (script_dir / ".." / ".." / "eq_mag_prediction" / "eq_mag_prediction" / "scripts" / "magnitude_predictor_trainer.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"MAGNET trainer script not found: {script_path}")

    # Ensure output_dir exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_subprocess(
        str(script_path),
        gin_path,
        output_dir=str(output_dir),
        **flags
    )


def run_magnet_trainer_or_load(gin_path, output_dir=None, trained_models_base_dir=None, **flags):
    """
    Wrapper for run_magnet_trainer that checks if model already exists and loads it instead of retraining.

    Args:
        gin_path: Path to gin config file
        output_dir: Directory for saving trained model (only used as fallback if model ID generation fails)
        trained_models_base_dir: Base directory for trained models (defaults to standard location)
        **flags: Additional flags to pass to the subprocess

    Returns:
        Path to the model directory (either existing or newly trained)
    """
    if trained_models_base_dir is None:
        trained_models_base_dir = Path("/home/neriberman/REPOS/eq_mag_prediction/results/trained_models")
    else:
        trained_models_base_dir = Path(trained_models_base_dir)

    # Try to generate model ID from gin config
    model_id = _get_model_id_from_gin_config(gin_path)

    if model_id is not None:
        model_dir = trained_models_base_dir / model_id
        model_path = model_dir / "model"

        # Check if model exists
        if model_path.exists() and model_path.is_dir():
            print(f"Found existing trained model: {model_dir}")
            print(f"Skipping training and using existing model.")
            return str(model_dir)
        else:
            print(f"Model ID: {model_id}")
            print(f"Model directory does not exist: {model_dir}")
            print(f"Creating persistent model directory and proceeding with training...")
            # Create the persistent directory and use it as output_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(model_dir)
    else:
        print("Could not generate model ID from gin config.")
        if output_dir is None:
            raise ValueError("output_dir is required for run_magnet_trainer when model ID generation fails")
        print(f"Using provided output_dir as fallback: {output_dir}")

    # Model doesn't exist - proceed with training using persistent location (or fallback)
    run_magnet_trainer(gin_path, output_dir=output_dir, **flags)

    # After training, the model should be saved to trained_models_base_dir/model_id
    # Return the path where it was saved (or output_dir if ID generation failed)
    if model_id is not None:
        model_dir = trained_models_base_dir / model_id
        if model_dir.exists():
            return str(model_dir)

    # Fallback: return output_dir if we can't determine the model directory
    return str(output_dir)

# 3. Use trained model to predict aftershocks in catalog continuation.

# 3a. fit ETAS on train and validation periods.
# subprocess invert_etas.py:


# 3b. Predict magnitudes for aftershocks in test period.
# subprocess simulate_catalog_continuation.py:

# 4. Assess the performance of the predictions.

def run_config_sync(
    *,
    general_gin_config_path: str,
    local_gin_config_path: str,
    invert_etas_config_json_path: str,
    val_to_train_time_ratio: float,
    force_json_on_gin: bool,
) -> None:
    """
    Sync time window params and catalog paths between MAGNET (gin) and ETAS (json),
    according to force_json_on_gin direction.
    """
    general_gin_config = _read_text_file(general_gin_config_path)
    local_gin_config = _read_text_file(local_gin_config_path)

    general_gin_config_dict = parse_gin_config(general_gin_config)
    local_gin_config_dict = parse_gin_config(local_gin_config)

    with open(invert_etas_config_json_path, "r") as f:
        etas_inversion_json_dict = json.load(f)

    if force_json_on_gin:   # copy ETAS's json fields to MAGNET's gin
        update_dict = {
            # Interpret ETAS JSON times as UTC to avoid local timezone drift
            'feature_prep_start': int(datetime.strptime(etas_inversion_json_dict["auxiliary_start"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()),
            'train_start_time': int(datetime.strptime(etas_inversion_json_dict["timewindow_start"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()),
            'test_start_time': int(datetime.strptime(etas_inversion_json_dict["timewindow_end"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()),
            'test_end_time': int(datetime.strptime(etas_inversion_json_dict["testwindow_end"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()),
        }
        update_dict['validation_start_time'] = (1-val_to_train_time_ratio)*update_dict['train_start_time'] + val_to_train_time_ratio*update_dict['test_start_time']

        # Ensure a corresponding MAGNET ingested catalog exists (without overwriting).
        json_dir = Path(invert_etas_config_json_path).parent
        etas_catalog_rel = etas_inversion_json_dict.get(
            "fn_catalog",
            os.path.join("..", "input_data", "catalogs", "converted_jma.csv"),
        )
        etas_catalog_path = (json_dir / etas_catalog_rel).resolve()
        if not etas_catalog_path.exists():
            raise FileNotFoundError(f"ETAS catalog not found: {etas_catalog_path}")

        gin_bindings = local_gin_config_dict.get("bindings", {})
        current_catalog_binding = gin_bindings.get("catalog")
        if not current_catalog_binding:
            raise KeyError("Gin config missing required top-level binding: 'catalog'")

        _, catalog_binding_to_use = _ensure_magnet_catalog_for_etas_catalog(
            etas_catalog_path=etas_catalog_path,
            current_gin_catalog_binding=current_catalog_binding,
        )
        if catalog_binding_to_use != current_catalog_binding:
            update_dict["catalog"] = catalog_binding_to_use

        update_gin_parameters(local_gin_config_path, update_dict)
        return

    # else: copy MAGNET's gin fields to ETAS's json
    gin_bindings = local_gin_config_dict.get("bindings", {})
    catalog_path = _get_catalog_path_from_gin_binding(gin_bindings["catalog"])
    required_gin_keys = ["feature_prep_start", "train_start_time", "test_start_time", "test_end_time"]
    missing = [k for k in required_gin_keys if k not in gin_bindings]
    if missing:
        raise KeyError(
            f"Missing required gin keys in {local_gin_config_path}: {missing}. "
            f"Available top-level keys: {sorted([k for k in gin_bindings.keys() if isinstance(k, str)])}"
        )

    update_dict = {
        "auxiliary_start": _epoch_seconds_to_dt_string(gin_bindings["feature_prep_start"]),
        "timewindow_start": _epoch_seconds_to_dt_string(gin_bindings["train_start_time"]),
        "timewindow_end": _epoch_seconds_to_dt_string(gin_bindings["test_start_time"]),
        "testwindow_end": _epoch_seconds_to_dt_string(gin_bindings["test_end_time"]),
    }
    update_dict["fn_catalog"] = _ensure_etas_catalog_for_magnet_catalog(
        magnet_catalog_path=catalog_path,
        invert_etas_config_json_path=invert_etas_config_json_path,
        current_fn_catalog_value=etas_inversion_json_dict.get("fn_catalog"),
    )
    update_json_parameters(invert_etas_config_json_path, update_dict)


def run_etas_inversion(config_path: str, store_pij: bool) -> str:
    with open(config_path, 'r') as f:
        inversion_config = json.load(f)

    # Prevent overwriting any existing inversion outputs by ensuring a unique ID.
    # etas.inversion.ETASParameterCalculation.store_results writes files like:
    #   parameters_<id>.json, trig_and_bg_probs_<id>.csv, sources_<id>.csv, distances_<id>.csv, (pij_<id>.csv)
    output_dir = inversion_config.get("data_path")
    if output_dir is None:
        raise KeyError("ETAS inversion config missing required key: 'data_path'")

    # store_results does string concatenation, so enforce trailing slash.
    if not output_dir.endswith(("/", os.sep)):
        output_dir = output_dir + os.sep
        inversion_config["data_path"] = output_dir

    out_path = Path(output_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    def _would_overwrite(inv_id: str) -> bool:
        expected = [
            out_path / f"parameters_{inv_id}.json",
            out_path / f"trig_and_bg_probs_{inv_id}.csv",
            out_path / f"sources_{inv_id}.csv",
            out_path / f"distances_{inv_id}.csv",
        ]
        if store_pij:
            expected.append(out_path / f"pij_{inv_id}.csv")
        return any(p.exists() for p in expected)

    # If user supplied an id, keep it only if it won't overwrite.
    inv_id = str(inversion_config.get("id") or "")
    if (not inv_id) or _would_overwrite(inv_id):
        import uuid

        for _ in range(10_000):
            candidate = uuid.uuid4().hex
            if not _would_overwrite(candidate):
                inv_id = candidate
                break
        else:
            raise RuntimeError(f"Could not find a free inversion id in {out_path}")
        inversion_config["id"] = inv_id

    calculation = ETASParameterCalculation(inversion_config)
    calculation.prepare()
    _ = calculation.invert()
    calculation.store_results(inversion_config['data_path'], store_pij=store_pij)
    # Return the parameters JSON path produced by store_results.
    return str(out_path / f"parameters_{inv_id}.json")

def run_etas_catalog_continuation(config_path: str) -> None:
    with open(config_path, 'r') as f:
        simulation_config = json.load(f)
    cfg_dir = Path(config_path).resolve().parent
    fn_inversion_output = (cfg_dir / simulation_config["fn_inversion_output"]).resolve()
    fn_store_simulation = (cfg_dir / simulation_config["fn_store_simulation"]).resolve()
    forecast_duration = simulation_config["forecast_duration"]

    with open(fn_inversion_output, "r") as f:
        inversion_output = json.load(f)
    etas_inversion_reload = ETASParameterCalculation.load_calculation(inversion_output)

    simulation = ETASSimulation(etas_inversion_reload)
    simulation.prepare()
    simulation.simulate_to_csv(
        str(fn_store_simulation),
        forecast_duration,
        1,
        magnitude_generator=simulation_config.get("magnitude_generator", "simulate_magnitudes"),
    )

# region Main Execution
if __name__ == "__main__":
    val_to_train_time_ratio = 3/4
    force_json_on_gin = False  # Gin -> JSON
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline_config_json",
        default="/home/neriberman/REPOS/etas_edits/config/pipeline_single_source.json",
        help="Single-source pipeline config JSON (templates + overrides).",
    )
    args = parser.parse_args()

    temp_paths = _build_temp_configs_from_single_source(
        pipeline_config_path=args.pipeline_config_json,
        val_to_train_time_ratio=val_to_train_time_ratio,
    )

    general_gin_config_path = temp_paths["general_gin_config_path"]
    local_gin_config_path = temp_paths["local_gin_config_path"]
    invert_etas_config_json_path = temp_paths["invert_etas_config_json_path"]
    etas_catalog_continuation_config_json_path = temp_paths["etas_catalog_continuation_config_json_path"]

    print(f"Using temp pipeline workspace: {temp_paths['tmp_root']}")

    # ---- MAGNET stages (feature computation + training)
    # Hardcoded to current local gin config; add flags as needed.
    run_feature_computation(local_gin_config_path)

    # Create output directory for trained model in temp workspace
    tmp_root = Path(temp_paths['tmp_root'])
    magnet_output_dir = tmp_root / "magnet_output"
    # Use wrapper that checks for existing model and loads it if available
    model_dir = run_magnet_trainer_or_load(local_gin_config_path, output_dir=str(magnet_output_dir))
    print(f"Using model from: {model_dir}")

    # ---- ETAS stages (inversion + catalog continuation simulation)
    # These scripts read their own JSON configs (hardcoded inside those scripts).
    # Run inversion and update the continuation config to point at the newly-created parameters_<id>.json.
    fn_parameters_json = run_etas_inversion(invert_etas_config_json_path, store_pij=True)
    cfg_dir = Path(etas_catalog_continuation_config_json_path).resolve().parent
    fn_parameters_rel = os.path.relpath(fn_parameters_json, cfg_dir)
    update_json_parameters(etas_catalog_continuation_config_json_path, {"fn_inversion_output": fn_parameters_rel})
    run_etas_catalog_continuation(etas_catalog_continuation_config_json_path)
# endregion Main Execution