"""
Environment / secrets handling for model configurations.

API keys are kept out of the model config JSON files. Instead, a config value
can reference an environment variable with ``${VAR_NAME}``, e.g.::

    "api_key": "${OPENROUTER_API_KEY}"

Variables are read from the process environment, which is first populated from
the ``.env`` file at the repository root (see ``.env.example``). Variables that
are already set in the environment take precedence over values in ``.env``.
"""

import os
import re
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATH = REPO_ROOT / ".env"

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_env(dotenv_path: Union[str, Path] = DEFAULT_DOTENV_PATH) -> bool:
    """
    Loads variables from a ``.env`` file into the process environment without
    overriding variables that are already set.

    Parameters:
        dotenv_path (`str` or `Path`, optional): Path to the ``.env`` file. Defaults to ``<repo root>/.env``.

    Returns:
        `bool`: True if at least one variable was set from the file.
    """
    return load_dotenv(dotenv_path, override=False)


def expand_env_vars(value: Any, source: str = "config") -> Any:
    """
    Recursively replaces ``${VAR_NAME}`` references in strings (including those
    nested in dicts and lists) with the value of the environment variable.

    Parameters:
        value (`Any`): Value to expand (str, dict, list, or anything else, which is returned unchanged)
        source (`str`, optional): Where the value comes from, used in error messages

    Returns:
        `Any`: The value with all references expanded

    Raises:
        `KeyError`: If a referenced environment variable is not set or is empty
    """
    if isinstance(value, dict):
        return {k: expand_env_vars(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v, source) for v in value]
    if not isinstance(value, str):
        return value

    def substitute(match: re.Match) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise KeyError(
                f"Environment variable '{name}' referenced in {source} is not set. "
                f"Add it to {DEFAULT_DOTENV_PATH} (see .env.example) or export it in your shell."
            )
        return resolved

    return _ENV_REFERENCE.sub(substitute, value)
