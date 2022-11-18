from typing import Iterable, List


def to_upper(collection: Iterable[str]) -> List[str]:
    """
    Uppercase each string in a collection.
    :param collection: iterable collection of strings.
    :return: Uppercase List of strings.
    """
    return [v.upper() for v in collection]
