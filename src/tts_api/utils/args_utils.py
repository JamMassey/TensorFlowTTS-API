from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import OrderedDict

logger = logging.getLogger(__name__)


def add_boolean_arg(parser: ArgumentParser, name: str, desc: str, default: bool = False) -> None:
    """Adds a boolean arg to the arg parser allowing --arg and --no-arg for True and False respectively
    Parameters
    ----------
    parser : ArgumentParser
        Arg parser to add the argument to
    name : str
        Name of the argument
    desc : str
        Description of the arg to add
    default : bool, optional
        Default value of the boolean flag, by default False
    """
    dest = name.replace("-", "_")
    group = parser.add_argument_group(f"{name} options:", desc)
    me_group = group.add_mutually_exclusive_group(required=False)
    me_group.add_argument(f"--{name}", dest=dest, action="store_true", help="(default)" if default else "")
    me_group.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help="(default)" if not default else "",
    )
    parser.set_defaults(**{dest: default})


@dataclass
class FlaskServerArgs:
    """Data Class for storing CL args"""

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = os.getenv("PORT", 5000)
    log_level: int = os.getenv("LOG_LEVEL", logging.INFO)
    console_log: bool = os.getenv("CONSOLE_LOG", True)
    filesystem_root = os.getenv("FILESYSTEM_ROOT", "filesystem")


def parse_flask_server_args() -> FlaskServerArgs:
    """Parses CL args into a Args object
    Returns
    -------
    Args
        Args object containing all the
    """
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-ho", "--host", type=str, dest="host", default="0.0.0.0")
    arg_parser.add_argument("-p", "--port", type=int, dest="port", default=5000)
    arg_parser.add_argument(
        "-ll",
        "--log-level",
        default=logging.INFO,
        type=int,
        dest="log_level",
        help="The log level of logging",
    )
    arg_parser.add_argument(
        "-fsr",
        "--filesystem-root",
        default="/app/filesystem",
        type=str,
        dest="filesystem_root",
        help="The root of the filesystem to use",
    )
    add_boolean_arg(arg_parser, "console-log", "Log to console", default=True)
    return FlaskServerArgs(**OrderedDict(vars(arg_parser.parse_args())))
