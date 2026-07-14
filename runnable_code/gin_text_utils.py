"""Pure-text gin helpers (no TensorFlow / gin runtime required)."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def read_text_file(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_gin_config(content: str) -> dict:
    """Parse gin text into ``{includes, bindings}`` with nested scopes as dicts."""
    config = {"includes": [], "bindings": {}}
    current_scope = None
    assignment_pattern = re.compile(r"^([^=]+)\s*=\s*(.*)$")

    for line in content.split("\n"):
        if "#" in line:
            line = line.split("#", 1)[0]
        indent_level = len(line) - len(line.lstrip())
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("include "):
            config["includes"].append(stripped_line.split(" ", 1)[1].strip("'\""))
            continue
        if stripped_line.endswith(":"):
            current_scope = stripped_line[:-1]
            config["bindings"].setdefault(current_scope, {})
            continue
        if indent_level == 0 and not stripped_line.endswith(":"):
            current_scope = None
        match = assignment_pattern.match(stripped_line)
        if match:
            key, value_str = match.groups()
            key = key.strip()
            parsed_value = _parse_gin_value(value_str.strip())
            if current_scope:
                config["bindings"][current_scope][key] = parsed_value
            else:
                config["bindings"][key] = parsed_value
    return config


def _parse_gin_value(value_str: str):
    if value_str.startswith(("@", "%")):
        return value_str
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


def format_gin_value_for_output(val) -> str:
    if isinstance(val, str):
        if val.strip().startswith(("%", "@")):
            return val
        return f"'{val}'"
    if isinstance(val, bool):
        return str(val)
    return str(val)


def flatten_gin_bindings(bindings: dict) -> dict:
    """Flatten ``{scope: {key: val}}`` to ``scope.key -> val``."""
    result = {}
    for key, value in bindings.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                result[f"{key}.{sub_key}"] = sub_value
        else:
            result[key] = value
    return result


def trainer_hyperparams_from_gin_text(gin_path: str | Path) -> dict:
    """
    Read trainer hypers from gin text (nested or dotted).

    Prefer this over gin.query_parameter under skip_unknown=True when the
    trainer configurable is not registered.
    """
    flat = flatten_gin_bindings(parse_gin_config(read_text_file(gin_path)).get("bindings", {}))
    scope = "train_and_evaluate_magnitude_prediction_model"
    return {
        "learning_rate": flat.get(f"{scope}.learning_rate"),
        "batch_size": flat.get(f"{scope}.batch_size"),
        "epochs": flat.get(f"{scope}.epochs"),
        "pdf_support_stretch": flat.get(f"{scope}.pdf_support_stretch", 7),
    }


def update_gin_parameters(gin_path: str | Path, params_dict: dict) -> None:
    """
    Update gin parameters using fully-qualified keys.

    Nested block assignments are ``Scope.key``. Top-level macros use bare names.
    Bare keys never overwrite indented bindings in another scope.
    """
    gin_path = Path(gin_path)

    def format_value(val):
        if isinstance(val, str):
            if val.strip().startswith(("@", "%")):
                return val
            return f"'{val}'"
        if isinstance(val, bool):
            return str(val)
        return str(val)

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    assignment_pattern = re.compile(r"^(\s*)([^#=\s]+)\s*=\s*(.*)$")
    current_scope: str | None = None

    lines = gin_path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line in lines:
        stripped = line.split("#", 1)[0].rstrip()
        indent_level = len(line) - len(line.lstrip(" \t")) if line.strip() else 0
        bare = stripped.strip()

        if bare.endswith(":") and "=" not in bare:
            current_scope = bare[:-1].strip()
            new_lines.append(line)
            continue

        match = assignment_pattern.match(line)
        if match:
            indent, key_in_file, _rest = match.groups()
            key_in_file = key_in_file.strip()
            if indent_level == 0:
                current_scope = None
                qualified = key_in_file
            elif current_scope:
                qualified = f"{current_scope}.{key_in_file}"
            else:
                qualified = key_in_file

            if qualified in params_dict:
                new_lines.append(f"{indent}{key_in_file} = {format_value(params_dict[qualified])}\n")
                updated_keys.add(qualified)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    missing_keys = set(params_dict.keys()) - updated_keys
    if missing_keys:
        new_lines.append("\n# --- Parameters added by update script ---\n")
        for key in sorted(missing_keys):
            new_lines.append(f"{key} = {format_value(params_dict[key])}\n")
            print(f"Appended new parameter: {key}")

    gin_path.write_text("".join(new_lines), encoding="utf-8")
    print(f"Successfully updated {len(updated_keys)} parameters in {gin_path}")
