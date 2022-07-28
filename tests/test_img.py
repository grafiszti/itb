from itb.img import read, resize, rgb2gray, gray2rgb, rgb2bgr
from itb.consts import TEST_IMG_1


def test_image_reading():
    assert read(TEST_IMG_1).shape == (346, 519, 3)


def test_image_resizing():
    img = read(TEST_IMG_1)

    # reduce the size
    assert resize(img, 100).shape == (67, 100, 3)

    # increase the size
    assert resize(img, 1000).shape == (667, 1000, 3)


def test_color_schemes_changing():
    assert rgb2gray(read(TEST_IMG_1)).shape == (346, 519)
