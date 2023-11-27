from __future__ import annotations

import os
import tempfile
import zipfile

from tts_api.utils.args_utils import parse_flask_server_args


def zip(src: str, dst: str) -> None:
    """Zip the contents of a directory into a zip file.

    Args:
        src: The path to the directory to zip.
        dst: The path to the zip file to create.
    """
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, _d, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1 :]
            zf.write(absname, arcname)
    zf.close()


def temp_zip(src: str) -> str:
    """Zip the contents of a directory into a temporary zip file.

    Args:
        src: The path to the directory to zip.

    Returns:
        The path to the created temporary zip file.
    """
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        abs_src = os.path.abspath(src)
        for dirname, _, files in os.walk(src):
            for filename in files:
                absname = os.path.abspath(os.path.join(dirname, filename))
                arcname = absname[len(abs_src) + 1 :]
                zf.write(absname, arcname)
    return temp_zip.name


def initialise_filesystem(root: str | os.PathLike[str] = "filesystem") -> None:
    """Initialise the filesystem for the application. Only creates directories if they do not already exist.

    Args:
        root: The path to the root directory of the filesystem.
    """
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "models")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "base"), exist_ok=True)
    os.makedirs(os.path.join(path, "vocoder"), exist_ok=True)
    os.makedirs(os.path.join(root, "jobs"), exist_ok=True)


if __name__ == "__main__":
    args = parse_flask_server_args()
    initialise_filesystem(args.filesystem_root)
