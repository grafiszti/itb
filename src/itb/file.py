import glob
from typing import List, Tuple

from itb.collection import to_upper


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
    found_files = []
    for ext in extensions:
        found_files.extend(list(glob.glob(f"{directory}/**/*.{ext}", recursive=True)))
    return found_files
