import os
import zipfile


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


def initialise_filesystem(root: str | os.PathLike[str]) -> None:
    """Initialise the filesystem for the application. Only creates directories if they do not already exist.

    Args:
        root: The path to the root directory of the filesystem.
    """
    path = os.path.join(root, "filesystem")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(root, "filesystem", "models")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "base"), exist_ok=True)
    os.makedirs(os.path.join(path, "vocoder"), exist_ok=True)
    os.makedirs(os.path.join(root, "filesystem", "jobs"), exist_ok=True)
