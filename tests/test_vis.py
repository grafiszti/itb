import matplotlib
import pytest

from itb.consts import TEST_IMG_1, TEST_IMG_3
from itb.img import read
from itb.vis import draw, thumbnails


def test_thumbnails_creation():
    created_thumbnails = thumbnails([read(TEST_IMG_1), read(TEST_IMG_3)])

    assert len(created_thumbnails) == 2
    assert created_thumbnails[0].shape == (67, 100, 3)
    assert created_thumbnails[1].shape == (100, 67, 3)


def test_thumbnails_errors():
    with pytest.raises(ValueError):
        # cannot create thumbnails with int as input
        thumbnails([1, 2])


def test_draw():
    matplotlib.use("Agg")  # don't show plot at the end

    images = [read(TEST_IMG_1), read(TEST_IMG_3)]
    draw(images, titles=["test1", "test2"])

    draw(TEST_IMG_1, titles=["test1"], columns_number=1)

    draw([TEST_IMG_1, TEST_IMG_3], titles=["test1", "test2"], columns_number=1)

    draw(
        [TEST_IMG_1, TEST_IMG_3],
        titles=["test1", "test2"],
        columns_number=1,
        image_resize_max_dim=100,
    )

    assert True
