from itb.file import find_images
from itb.consts import TEST_IMGS_DIR


def test_find_images():
    assert len(find_images(TEST_IMGS_DIR)) == 2
