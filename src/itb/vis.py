from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from itb.img import read, resize


def _validate_titles(titles: List[str], images_number: int) -> None:
    if titles is not None and len(titles) != 0 and len(titles) != images_number:
        raise ValueError(
            "Incorrect number of titles, should be the same number as images."
        )


def _validate_img(img: Union[str, np.array]) -> None:
    if not isinstance(img, (str, np.ndarray)):
        raise ValueError(
            "Incorrect image type, should be 'str' "
            + f" or 'np.ndarray', but {type(img)} found."
        )


def _draw_tiled_images_set(
    images: List[Union[np.ndarray, str]],
    titles: List[str],
    fig_size: Tuple[int, int],
    title_font_size: int,
    columns_number: int,
    image_resize_max_dim: int,
):
    _validate_titles(titles, len(images))

    images_number = len(images)
    rows_number = int(images_number / columns_number) + int(
        images_number % columns_number > 0
    )

    if columns_number > len(images):
        columns_number = len(images)

    fig, axs = plt.subplots(rows_number, columns_number, figsize=fig_size)

    for i, image in enumerate(images):
        _validate_img(image)

        img_title = titles[i] if titles is not None and i < len(titles) else ""

        if type(image) == str:
            image = read(image)

        if image_resize_max_dim:
            image = resize(image, max_dim=image_resize_max_dim)

        col_index = i % columns_number
        row_index = int(i / columns_number)

        if rows_number == 1 and columns_number == 1:
            axs.imshow(image)
            axs.set_title(img_title, fontsize=title_font_size)
        elif rows_number == 1:
            axs[col_index].imshow(image)
            axs[col_index].set_title(img_title, fontsize=title_font_size)
        elif columns_number == 1:
            axs[row_index].imshow(image)
            axs[row_index].set_title(img_title, fontsize=title_font_size)
        else:
            axs[row_index, col_index].imshow(image)
            axs[row_index, col_index].set_title(img_title, fontsize=title_font_size)

    plt.tight_layout()
    plt.show()
    plt.close("all")


def draw(
    images: Union[str, np.ndarray, List[Union[np.ndarray, str]]],
    titles: List[str] = (),
    fig_size: Tuple[int, int] = (16, 16),
    title_font_size: int = 20,
    columns_number: int = 4,
    image_resize_max_dim: int = None,
) -> None:
    """
    Draws a list of images or a single image to a notebook in form of
    a grid.
    :param images: images to print may be single 'str' or a 'np.ndarray'
    or a list of string or np.ndarray
    :param titles: list of titles which will be added above the printed
    images, should be a list of strings
    :param fig_size: size of a final figure with all images
    :param title_font_size: font size of a title of a sub image
    :param columns_number: number of columns which should be used for
    drawing the images
    :param image_resize_max_dim: an output maximum size of a sub image
    drawn on grid
    :return:
    """
    if isinstance(images, (List, Tuple)):
        _draw_tiled_images_set(
            images,
            titles,
            fig_size,
            title_font_size,
            columns_number,
            image_resize_max_dim,
        )
    elif isinstance(images, (str, np.ndarray)):
        _draw_tiled_images_set(
            [images],
            titles,
            fig_size,
            title_font_size,
            columns_number,
            image_resize_max_dim,
        )
    else:
        raise ValueError(f"Unsupported images type: {type(images)}.")


def _thumbnail(image: Union[str, np.ndarray], max_dim: int) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return resize(image, max_dim)
    elif isinstance(image, str):
        return resize(read(image), max_dim)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def thumbnails(
    images: Union[str, np.ndarray, List[Union[np.ndarray, str]]], max_dim: int = 100
) -> List[np.ndarray]:
    """
    Resizes given list of images into a list of thumbnails.
    :param images: list of images to resize, numpy ndarrays
    :param max_dim: maximum dimension of each resized output image
    :return: thumbnails of an images as list of numpy ndarrays
    """
    if isinstance(images, (List, Tuple)):
        return [_thumbnail(image, max_dim) for image in images]
    elif isinstance(images, (str, np.ndarray)):
        return [_thumbnail(images, max_dim)]
    else:
        raise ValueError(f"Unsupported images type: {type(images)}.")
