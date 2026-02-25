from itb.color import get_random_color


def test_random_color():
    color = get_random_color()
    assert isinstance(color, tuple)
    assert len(color) == 3
    assert all(isinstance(c, int) and 0 <= c <= 255 for c in color)
