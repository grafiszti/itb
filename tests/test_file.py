from pathlib import Path

from itb.consts import TEST_IMGS_DIR
from itb.file import find_files, find_images, make_dirs, remove_files


def test_find_images():
    assert len(find_images(TEST_IMGS_DIR)) == 2


def test_find_files():
    assert len(find_files(TEST_IMGS_DIR, ("jpg",))) == 2


def test_make_dirs_and_remove_files():
    test_dir = Path(TEST_IMGS_DIR) / "test_make_dirs"
    test_subdir = test_dir / "subdir"
    make_dirs([TEST_IMGS_DIR, test_subdir])

    assert test_subdir.exists() and test_subdir.is_dir()

    remove_files([test_subdir])
    remove_files([test_dir])

    assert not test_subdir.exists()
    assert not test_dir.exists()
