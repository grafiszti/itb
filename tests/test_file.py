from itb.consts import TEST_IMGS_DIR
from itb.file import find_images


def test_find_images():
    assert len(find_images(TEST_IMGS_DIR)) == 3
