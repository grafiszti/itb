import os.path

import numpy as np
import cv2


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


def write(img_path: str, image: np.ndarray) -> bool:
    """
    Writes the image to a disk to given path.
    :param img_path: image path where the image will be saved
    :param image: an image to save in RGB color schema
    :return: true if image was successfully saved, false otherwise
    """
    return cv2.imwrite(img_path, rgb2bgr(image))


def resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Resize an image to set bigger dimension equal to max_dim keeping the original image ratio.
    :param img: image to resize, numpy ndarray
    :param max_dim: maximum dimension of resized output image
    :return: resized image, numpy ndarray
    """

    assert max_dim > 0, "Maximum output dimension should be > 0."

    resize_factor = max_dim / max(img.shape[:2])

    # If the size is increasing the CUBIC interpolation is used,
    # if downsized, the AREA interpolation is used
    interpolation = cv2.INTER_CUBIC if resize_factor > 1.0 else cv2.INTER_AREA

    h, w = img.shape[:2]
    return cv2.resize(img, (int(round(w * resize_factor)), int(round(h * resize_factor))), interpolation=interpolation)


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
