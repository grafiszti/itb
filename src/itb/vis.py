from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from itb.img import resize, read


def _validate_titles(titles: List[str], images_number: int) -> None:
    if titles is not None and len(titles) != 0 and len(titles) != images_number:
        raise ValueError(f"Incorrect number of titles, should be the same number as images.")


def _validate_img(img: Union[str, np.array]) -> None:
    if not isinstance(img, (str, np.ndarray)):
        raise ValueError(f"Incorrect image type, should by 'str' or 'np.ndarray', but {type(img)} found.")


def _draw_tiled_images_set(
        images: List[Union[np.ndarray, str]],
        titles: List[str],
        fig_size: Tuple[int, int],
        title_font_size: int,
        columns_number: int,
        image_resize_max_dim: int
):
    _validate_titles(titles, len(images))

    images_number = len(images)
    rows_number = int(images_number / columns_number) + int(images_number % columns_number > 0)

    if columns_number > len(images):
        columns_number = len(images)

    fig, axs = plt.subplots(rows_number, columns_number, figsize=fig_size)

    for i, img in enumerate(images):
        _validate_img(img)

        img_title = titles[i] if titles is not None and i < len(titles) else ""

        if type(img) == str:
            img = read(img)

        if image_resize_max_dim:
            img = resize(img, max_dim=image_resize_max_dim)

        col_index = i % columns_number
        row_index = int(i / columns_number)

        if rows_number == 1 and columns_number == 1:
            axs.imshow(img)
            axs.set_title(img_title, fontsize=title_font_size)
        elif rows_number == 1:
            axs[col_index].imshow(img)
            axs[col_index].set_title(img_title, fontsize=title_font_size)
        elif columns_number == 1:
            axs[row_index].imshow(img)
            axs[row_index].set_title(img_title, fontsize=title_font_size)
        else:
            axs[row_index, col_index].imshow(img)
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
        image_resize_max_dim: int = None
) -> None:
    """
    Draws a list of images or a single image to a notebook in form of a grid.
    :param images: images to print may be single 'str' or a 'np.ndarray' or a list of
    string or np.ndarray
    :param titles: list of titles which will be added above the printed images,
    should be a list of strings
    :param fig_size: size of a final figure with all images
    :param title_font_size: font size of a title of a sub image
    :param columns_number: number of columns which should be used for drawing the images
    :param image_resize_max_dim: an output maximum size of a sub image drawn on grid
    :return:
    """
    if isinstance(images, (List, Tuple)):
        _draw_tiled_images_set(images, titles, fig_size, title_font_size, columns_number, image_resize_max_dim)
    elif isinstance(images, (str, np.ndarray)):
        _draw_tiled_images_set([images], titles, fig_size, title_font_size, columns_number, image_resize_max_dim)
    else:
        print(f"Unsupported images type: {type(images)}.")
