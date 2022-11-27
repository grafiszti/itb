import io
import os.path
from typing import List, Tuple, Union

import cv2
import numpy as np
import requests
from PIL import Image

from itb.color import RED


def gray2rgb(image: np.ndarray) -> np.ndarray:
    """
    Converse the GRAY color scheme to an RGB color scheme.
    :param image: input image
    :return: output converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def rgb2gray(image: np.ndarray) -> np.ndarray:
    """
    Converse the RGB color scheme to an GRAY color scheme.
    :param image: input image
    :return: output converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb2bgr(image: np.ndarray) -> np.ndarray:
    """
    Converse the RGB color scheme to an BGR color scheme.
    :param image: input image
    :return: output converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr2rgb(image: np.ndarray) -> np.ndarray:
    """
    Converse the BGR color scheme to an RGB color scheme.
    :param image: input image
    :return: output converted image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read(img_path: str) -> np.ndarray:
    """
    Reads image from a disk with conversion to RGB palette.
    :param img_path: image path
    :return: image as a numpy array
    """

    assert os.path.exists(img_path), f"File: {img_path} does not exist."

    return bgr2rgb(cv2.imread(img_path))


def download(url: str) -> np.ndarray:
    """
    Downloads image from given URL and return as np.ndarray.
    :param url: URL of image to download
    :return: np.ndarray represents downloaded image
    """
    response = requests.get(url)
    bytes_im = io.BytesIO(response.content)
    return np.array(Image.open(bytes_im))


def write(img_path: str, image: np.ndarray) -> bool:
    """
    Writes the image to a disk to given path.
    :param img_path: image path where the image will be saved
    :param image: an image to save in RGB color schema
    :return: true if image was successfully saved, false otherwise
    """
    return cv2.imwrite(img_path, rgb2bgr(image))


def _resize_max_dim(img: np.ndarray, max_dim: int):
    assert max_dim > 0, "Maximum output dimension should be > 0."

    resize_factor = max_dim / max(img.shape[:2])

    # If the size is increasing the CUBIC interpolation is used,
    # if downsized, the AREA interpolation is used
    interpolation = cv2.INTER_CUBIC if resize_factor > 1.0 else cv2.INTER_AREA

    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(round(w * resize_factor)), int(round(h * resize_factor))),
        interpolation=interpolation,
    )


def _resize_exact_dim(img: np.ndarray, exact_dim: Tuple):
    assert len(exact_dim) == 2, "The dimension length should be 2."

    h, w = img.shape[:2]
    new_h, new_w = exact_dim

    # when float passed, original values are scaled by the given dimension values
    if isinstance(new_h, float) and isinstance(new_w, float):
        new_h = int(round(h * new_h))
        new_w = int(round(w * new_w))

    interpolation = cv2.INTER_CUBIC if new_w > w or new_h > h else cv2.INTER_AREA

    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def resize(img: np.ndarray, dimension: Union[int, Tuple]) -> np.ndarray:
    """
    Resize an image to set bigger dimension equal to dimension
    keeping the original image ratio.
    :param img: image to resize, numpy ndarray
    :param dimension: desired dimension of resized output image, may be an int
    or a Tuple. When single int passed, bigger dimension would be resized
    to it and the smaller using the correct ratio. When the tuple is passed,
    each dimension of an image will be resized using corresponding tuple new dimension
    value. If the integer value passed, the image will be resized to those values,
    when the float passed, the values will be scaled by the tuple values.
    The tuple should have 2 dimensions, in the form of [new height, new width]
    :return: resized image, numpy ndarray
    """

    if isinstance(dimension, int):
        return _resize_max_dim(img, dimension)
    elif isinstance(dimension, Tuple):
        return _resize_exact_dim(img, dimension)


def rotate90(img: np.ndarray) -> np.ndarray:
    """
    Rotates the image 90 degree clockwise.
    :param img: image to rotate, numpy ndarray
    :return: rotated image, numpy ndarray
    """
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def rotate180(img: np.ndarray) -> np.ndarray:
    """
    Rotates the image 180 degree.
    :param img: image to rotate, numpy ndarray
    :return: rotated image, numpy ndarray
    """
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate270(img: np.ndarray) -> np.ndarray:
    """
    Rotates the image 270 degree clockwise.
    :param img: image to rotate, numpy ndarray
    :return: rotated image, numpy ndarray
    """
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _add_rectangles(
    img: np.ndarray,
    rectangles: List,
    color: Tuple[int, int, int],
    line_thickness: int,
    labels: List[str],
    label_text_size: float,
) -> np.ndarray:
    if labels is not None:
        assert isinstance(labels, List), "Labels has to be list."
        assert len(rectangles) == len(
            labels
        ), "Number of labels and rectangles should be equal."

    img_copy = img.copy()

    for i, rectangle in enumerate(rectangles):
        x1, y1, x2, y2 = rectangle

        # if all coordinates are between (0, 1) the corresponding values
        # are multiplied by width and height of an image
        if 0 <= x1 <= 1.0 and 0 <= y1 <= 1.0 and 0 <= x2 <= 1.0 and 0 <= y2 <= 1.0:
            img_h, img_w = img.shape[:2]

            x1 *= img_w
            y1 *= img_h
            x2 *= img_w
            y2 *= img_h

        img_copy = cv2.rectangle(
            img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness
        )

        # adding labels to the rectangles
        if labels is not None:
            cv2.putText(
                img_copy,
                labels[i],
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_text_size,
                color,
            )

    return img_copy


def add_rectangles(
    img: np.ndarray,
    rectangles: Union[
        List[Tuple[float, float, float, float]],
        Tuple[float, float, float, float],
        List[Tuple[int, int, int, int]],
        Tuple[int, int, int, int],
    ],
    color: Union[str, Tuple[int, int, int]] = RED,
    line_thickness: int = 1,
    labels=None,
    label_text_size: float = 0.5,
) -> np.ndarray:
    """
    Draws the rectangles on an images. Support or single rectangle
    or a collection of rectangles. Each rectangle should be represented
    as Tuple of numbers represents the top left and bottom right
    corners of the rectangle. Numbers could be integers (pixel values)
    or floats in range [0.0 - 1.0] (represents the percentage values
    of top left corner and bottom right corner of the rectangle.
    Each rectangle could have the label added. Labels should be
    passed for each rectangle.

    :param img: image to drawn rectangles on, numpy ndarray
    :param rectangles: List of Tuples or Tuple represented the
    rectangles to draw.
    :param color: color of rectangle in (R, G, B) format
    :param line_thickness: the thickness of rectangle in pixels,
    negative value means filled rectangle
    :param labels: List of labels
    :param label_text_size: size of the labels texts
    :return: a copy of source images with drawn rectangles, numpy ndarray
    """
    if isinstance(rectangles, (list, tuple)) and len(rectangles) > 0:
        if isinstance(rectangles[0], (int, float)):
            return _add_rectangles(
                img, [rectangles], color, line_thickness, labels, label_text_size
            )
        elif isinstance(rectangles[0], (list, tuple)):
            return _add_rectangles(
                img, rectangles, color, line_thickness, labels, label_text_size
            )
        else:
            raise ValueError(f"List of bboxes has unsupported type: {type(rectangles)}")
    else:
        raise ValueError(
            f"Bboxes has unsupported type: {type(rectangles)}, "
            f"should be list of bboxes or Tuple represents single bbox."
        )


def _add_circles(
    img: np.ndarray,
    points: List,
    color: Tuple[int, int, int],
    radius: int,
    line_thickness: int,
) -> np.ndarray:
    assert radius >= 0, "Radius should be >= 0."

    img_copy = img.copy()

    for point in points:
        x, y = point

        # if all coordinates are between (0, 1) the corresponding values
        # are multiplied by width and height of an image
        if 0 <= x <= 1.0 and 0 <= y <= 1.0:
            img_h, img_w = img.shape[:2]

            x *= img_w
            y *= img_h

        img_copy = cv2.circle(img_copy, (int(x), int(y)), radius, color, line_thickness)

    return img_copy


def add_circles(
    img: np.ndarray,
    centers: Union[
        List[Tuple[float, float]],
        Tuple[float, float],
        List[Tuple[int, int]],
        Tuple[int, int],
    ],
    color: Union[str, Tuple[int, int, int]] = RED,
    radius: int = 10,
    line_thickness: int = 1,
) -> np.ndarray:
    """
    Draws the circles on an images. Support or single circle or
    a collection of circles. Each circle should be represented as
    Tuple of numbers represents its center point. Numbers could be
    integers (pixel values) or floats in range [0.0 - 1.0]
    (represents the percentage values of center of a circle).

    :param img: image to drawn circles on, numpy ndarray
    :param centers: List of Tuples or Tuple represented the center
    of a circle to draw.
    :param color: color of circle in (R, G, B) format
    :param radius: radius of a circle in pixels
    :param line_thickness: the thickness of outline of a circle in
    pixels, negative value means filled circle
    :return: a copy of source images with drawn circles, numpy ndarray
    """
    if isinstance(centers, (list, tuple)) and len(centers) > 0:
        if isinstance(centers[0], (int, float)):
            return _add_circles(img, [centers], color, radius, line_thickness)
        elif isinstance(centers[0], (list, tuple)):
            return _add_circles(img, centers, color, radius, line_thickness)
        else:
            raise ValueError(f"List of centers has unsupported type: {type(centers)}")
    else:
        raise ValueError(
            f"Unsupported type of centers: {type(centers)},"
            + " should be list of points or Tuple represents"
            + " single center points x and y values."
        )


def add_points(
    img: np.ndarray,
    centers: Union[
        List[Tuple[float, float]],
        Tuple[float, float],
        List[Tuple[int, int]],
        Tuple[int, int],
    ],
    color: Union[str, Tuple[int, int, int]] = RED,
    radius: int = 1,
    line_thickness: int = -1,
) -> np.ndarray:
    """
    Draws the points on an images. Support or single point or a collection of points.
    Each point should be represented as Tuple of numbers represents its center point.
    Numbers could be integers (pixel values) or floats in range [0.0 - 1.0]
    (represents the percentage values of center of a point]).

    :param img: image to drawn points on, numpy ndarray
    :param centers: List of Tuples or Tuple represented the center of a point to draw.
    :param color: color of point in (R, G, B) format
    :param radius: radius of a point in pixels
    :param line_thickness: the thickness of outline of a point in pixels, negative value
    means filled point
    :return: a copy of source images with drawn points, numpy ndarray
    """
    return add_circles(img, centers, color, radius, line_thickness)
