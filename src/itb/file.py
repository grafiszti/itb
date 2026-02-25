import glob
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from itb.collection import to_upper


def make_dirs(paths: List[str | Path]) -> None:
    """
    Create directories if they do not exist.
    :param paths: list of paths to create.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def remove_files(paths: List[str | Path]) -> None:
    """
    Remove files if they exist.
    :param paths: list of paths to remove.
    """
    for path in paths:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                os.remove(p)


def find_images(
    directory: str,
    images_extensions: Tuple[str] = ("jpg", "jpeg", "png", "bmp", "gif", "webp"),
) -> List[str]:
    """
    Recursively find images in the given directory.
    :param directory: source directory where the images should be searched.
    :param images_extensions: images extension that should be searched. Processing
    lowercase and uppercase set of extensions.
    :return: the list of found images paths.
    """
    return find_files(directory, images_extensions + tuple(to_upper(images_extensions)))


def find_files(directory: str, extensions: Tuple[str]) -> List[str]:
    """
    Recursively find files with the given extensions in the given directory.
    :param directory: source directory where the files should be searched.
    :param extensions: files extensions that should be searched.
    :return: the list of found files paths.
    """
    found_files = []
    for ext in extensions:
        found_files.extend(list(glob.glob(f"{directory}/**/*.{ext}", recursive=True)))
    return found_files
