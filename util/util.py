"""Utility code"""
from pathlib import Path
from typing import NoReturn, Union

import click
from click.core import Context, Option


def validate_click_file_suffix(
    _: Context, param: Option, value: Path, suffix: str = ".json"
) -> Union[NoReturn, Path]:
    """
    Validate that file path passed as click argument is of file type `suffix`

    NOTE: when using as a click callback, `suffix` must be binded via
    `functools.partial`

    Args:
        param: Option key for binded param
        value: Value of binded Option

    Returns:
        The original file path if it is the right format else exception is raised
    """
    if value.suffix != suffix:
        raise click.BadParameter(f"{param} should be of type {suffix}")
    return value
