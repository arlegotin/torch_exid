from typing import Iterator
from torch_exid import ExtendedIterableDataset


class IntegersDataset(ExtendedIterableDataset):
    def generator(self) -> Iterator[int]:
        n = 0
        while True:
            yield n
            n += 1


"""
We're asserting dataset content twice
to make sure that it was properly reset after first call.
"""


def test_limit():
    ds = IntegersDataset(limit=3)
    assert list(ds) == [0, 1, 2]
    assert list(ds) == [0, 1, 2]


def test_offset():
    ds = IntegersDataset(offset=5, limit=2)
    assert list(ds) == [5, 6]
    assert list(ds) == [5, 6]


def test_shuffle():
    ds = IntegersDataset(limit=5, shuffle_buffer=3, shuffle_seed=42)
    assert list(ds) == [1, 0, 2, 4, 3]
    assert list(ds) == [1, 0, 2, 4, 3]

    ds = IntegersDataset(limit=5, shuffle_buffer=3, shuffle_seed=43)
    assert list(ds) == [2, 1, 0, 4, 3]
    assert list(ds) == [2, 1, 0, 4, 3]

    ds = IntegersDataset(limit=5, shuffle_buffer=10, shuffle_seed=42)
    assert list(ds) == [3, 1, 2, 4, 0]
    assert list(ds) == [3, 1, 2, 4, 0]

    ds = IntegersDataset(limit=5, shuffle_buffer=10, shuffle_seed=43)
    assert list(ds) == [1, 4, 3, 2, 0]
    assert list(ds) == [1, 4, 3, 2, 0]
