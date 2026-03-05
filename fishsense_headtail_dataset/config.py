import logging
import logging.handlers
import os
import time
from importlib.metadata import version
from pathlib import Path

import platformdirs
import validators
from dynaconf import Dynaconf, Validator

IS_DOCKER = os.environ.get("E4EFS_DOCKER", False)
platform_dirs = platformdirs.PlatformDirs("e4efs_api_workflow_worker")


def get_log_path() -> Path:
    """Get log path

    Returns:
        Path: Path to log directory
    """
    if IS_DOCKER:
        return Path("/e4efs/logs")
    log_path = platform_dirs.user_log_path
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_config_path() -> Path:
    """Get config path

    Returns:
        Path: Path to config directory
    """
    if IS_DOCKER:
        return Path("/e4efs/config")
    config_path = Path(".")
    return config_path


def path_validator(path: str) -> bool:
    """Validator to check if a given path exists.

    Args:
        path (str): Path to validate

    Returns:
        bool: True if path exists, False otherwise
    """
    return Path(path).exists()


validators = [
    Validator("e4e_nas.url", required=True, cast=str, condition=validators.url),
    Validator("e4e_nas.username", required=True, cast=str),
    Validator("e4e_nas.password", required=True, cast=str),
    Validator("fishsense_api.url", required=True, cast=str, condition=validators.url),
    Validator("fishsense_api.username", cast=str),
    Validator("fishsense_api.password", cast=str),
]

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml", ".secrets.toml"],
    validators=validators,
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
