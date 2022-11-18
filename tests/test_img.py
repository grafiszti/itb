import numpy as np

from itb.consts import TEST_IMG_1
from itb.img import (
    add_rectangles,
    bgr2rgb,
    download,
    gray2rgb,
    read,
    resize,
    rgb2bgr,
    rgb2gray,
    rotate90,
    rotate180,
    rotate270,
)


def test_image_reading():
    assert read(TEST_IMG_1).shape == (346, 519, 3)


def test_image_resizing():
    img = read(TEST_IMG_1)

    # reduce the size
    assert resize(img, 100).shape == (67, 100, 3)

    # increase the size
    assert resize(img, 1000).shape == (667, 1000, 3)


def test_color_schemes_changing():
    SINGLE_RED_PIXEL = np.array([[[1, 0, 0]]], dtype=np.uint8)
    SINGLE_BLUE_PIXEL = np.array([[[0, 0, 1]]], dtype=np.uint8)
    EXAMPLE_ONES_IMAGE = np.ones((10, 10), dtype=np.uint8)

    # gray2rgb
    assert gray2rgb(EXAMPLE_ONES_IMAGE).shape == (10, 10, 3)

    # rgb2gray
    assert rgb2gray(read(TEST_IMG_1)).shape == (346, 519)

    # rgb2bgr
    assert np.array_equal(SINGLE_BLUE_PIXEL, rgb2bgr(SINGLE_RED_PIXEL))

    # bgr2rgb
    assert np.array_equal(SINGLE_RED_PIXEL, bgr2rgb(SINGLE_BLUE_PIXEL))


def test_image_downloading():
    test_image_path = (
        "https://raw.githubusercontent.com/grafiszti/itb/master/test_data/img1.jpg"
    )
    assert download(test_image_path).shape == (346, 519, 3)


def test_images_rotations():
    EXAMPLE_ONES_IMAGE = np.ones((10, 20, 3), dtype=np.uint8)

    # rotation 90 degree
    assert rotate90(EXAMPLE_ONES_IMAGE).shape == (20, 10, 3)

    # rotation 180 degree
    assert rotate180(EXAMPLE_ONES_IMAGE).shape == (10, 20, 3)

    # rotation 270 degree
    assert rotate270(EXAMPLE_ONES_IMAGE).shape == (20, 10, 3)


def test_adding_rectangles():
    EMPTY_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)
    result = add_rectangles(EMPTY_IMAGE, [(0.0, 0.0, 1.0, 1.0)], line_thickness=-1)
    assert np.array_equal(result, np.array([[[255, 0, 0]]], dtype=np.uint8))
