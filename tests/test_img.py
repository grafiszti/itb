import numpy as np
import pytest

from itb.consts import TEST_IMG_1
from itb.img import (
    add_points,
    add_rectangles,
    bgr2rgb,
    download,
    gray2rgb,
    merge,
    minmax_norm,
    read,
    resize,
    rgb2bgr,
    rgb2gray,
    rotate90,
    rotate180,
    rotate270,
    to_img,
)


def test_image_reading():
    assert read(TEST_IMG_1).shape == (346, 519, 3)


def test_image_resizing():
    img = read(TEST_IMG_1)

    # reduce the size relative to max dimension
    assert resize(img, 100).shape == (67, 100, 3)

    # reduce the size relative to min dimension
    assert resize(img, 100, "min").shape == (100, 150, 3)

    # increase the size
    assert resize(img, 1000).shape == (667, 1000, 3)

    # change size to exact size
    assert resize(img, (100, 200)).shape == (100, 200, 3)

    # change size by scaling the dimensions
    assert resize(img, (0.5, 0.5)).shape == (173, 260, 3)

    # exception raises when send dimensions in 3 values tuple
    with pytest.raises(Exception):
        assert resize(img, (1, 2, 3))


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


def test_merge():
    EMPTY_IMAGE = np.zeros((1, 1, 3), dtype=np.uint8)
    EMPTY_BLACK_IMAGE = (np.ones((1, 1, 3)) * 255).astype(np.uint8)

    # merge two empty white images
    result = merge(EMPTY_IMAGE, EMPTY_IMAGE)
    assert result.shape == (1, 1, 3)
    assert result.tolist() == [[[0, 0, 0]]]

    # merge empty white image with black image
    result = merge(EMPTY_IMAGE, EMPTY_BLACK_IMAGE)
    assert result.shape == (1, 1, 3)
    assert result.tolist() == [[[128, 128, 128]]]

    # merge empty black image with white image wit different proportions
    # merge empty white image with black image
    result = merge(EMPTY_IMAGE, EMPTY_BLACK_IMAGE, alpha=0.125)
    assert result.shape == (1, 1, 3)
    assert result.tolist() == [[[223, 223, 223]]]


def test_minmax_norm():
    input_img = np.array(
        [
            [[0, 0, 0], [100, 100, 100]],
            [[100, 100, 100], [10, 10, 10]],
        ]
    )

    output_img = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.1, 0.1, 0.1]],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(minmax_norm(input_img), output_img)


def test_add_points():
    img = np.zeros((100, 100), dtype=np.uint8)
    img = add_points(img, [(10, 10), (20, 20)], radius=1, color=1)

    assert img[10, 10] == 1
    assert img[20, 20] == 1


def test_to_img():
    img = np.random.rand(100, 100)
    img[50, 50] = 3151253
    img = to_img(img)

    assert img.shape == (100, 100)
    assert img.dtype == np.uint8
    assert img[50, 50] == 255
    assert img[0, 0] == 0
